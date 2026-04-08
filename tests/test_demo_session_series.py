from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from iris_bot.artifacts import wrap_artifact
from iris_bot.config import Settings
from iris_bot.demo_session_series import (
    build_series_runtime_paths,
    ensure_active_demo_session_series,
    load_series_registry,
    record_demo_session_result,
    start_demo_session_series,
)
from iris_bot.durable_io import durable_write_json


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "data", replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"))
    object.__setattr__(settings, "demo_execution", replace(settings.demo_execution, target_symbol="EURUSD"))
    return settings


def test_start_demo_session_series_sets_active_registry(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = start_demo_session_series(settings, symbol="EURUSD")
    registry = load_series_registry(settings)
    assert registry["active_series_id"] == state.session_series_id
    assert state.target_sessions >= 1


def test_ensure_active_series_reuses_existing_active_series(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = start_demo_session_series(settings, symbol="EURUSD")
    reused = ensure_active_demo_session_series(settings, symbol="EURUSD")
    assert reused.session_series_id == state.session_series_id


def test_record_demo_session_result_updates_aggregate_counts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = start_demo_session_series(settings, symbol="EURUSD")
    base = build_series_runtime_paths(settings)["base"]
    evidence_path = base / "session_evidence.json"
    review_path = base / "session_review.json"
    durable_write_json(
        evidence_path,
        wrap_artifact(
            "demo_session_evidence",
            {
                "session_id": "sess-1",
                "session_series_id": state.session_series_id,
                "signal_summary": {"signals_generated": 1, "blocked_signals": 0, "no_trade_signals": 0},
                "execution_summary": {"orders_sent": 1, "orders_rejected": 0, "decisions_executed": 1},
                "trade_summary": {"trades_opened": 1, "trades_closed": 1},
                "performance_summary": {"realized_pnl_usd": 1.25, "position_lifetime_seconds": 120.0},
                "divergence_summary": {"divergence_events": 0, "reconcile_events": 1},
                "restore_recovery_summary": {"restore_events": 1, "recovery_events": 0},
                "final_state_summary": {"kill_switch_active": False, "circuit_breaker_triggered": False},
            },
        ),
    )
    durable_write_json(
        review_path,
        wrap_artifact(
            "demo_session_review",
            {
                "session_id": "sess-1",
                "session_series_id": state.session_series_id,
                "classification": "healthy",
                "evidence_complete": True,
                "valid_for_forward_validation": True,
                "recommendation": "continue",
            },
        ),
    )
    update = record_demo_session_result(
        settings,
        session_series_id=state.session_series_id,
        session_id="sess-1",
        session_evidence_path=evidence_path,
        session_review_path=review_path,
        session_evidence_payload={
            "execution_summary": {"orders_sent": 1, "orders_rejected": 0},
            "divergence_summary": {"divergence_events": 0, "reconcile_events": 1},
            "restore_recovery_summary": {"restore_events": 1, "recovery_events": 0},
            "final_state_summary": {"kill_switch_active": False, "circuit_breaker_triggered": False},
        },
        session_review_payload={"classification": "healthy"},
    )
    assert update["series_state"]["aggregate_counts"]["successful_sessions"] == 1
    assert update["series_state"]["aggregate_counts"]["orders_sent"] == 1
    assert Path(update["series_evidence_path"]).exists()
