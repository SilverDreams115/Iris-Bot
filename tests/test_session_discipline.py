"""BLOQUE 6 — Session Discipline and Runbook Tests."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from iris_bot.config import (
    BacktestConfig,
    OperationalConfig,
    RecoveryConfig,
    Settings,
)
from iris_bot.kill_switch import activate_kill_switch, activate_no_trade_mode
from iris_bot.operational import (
    AccountState,
    AlertRecord,
    BrokerSyncStatus,
    DailyLossTracker,
    PaperEngineState,
)
from iris_bot.resilient import build_runtime_state_path, fresh_state, persist_runtime_state
from iris_bot.session_discipline import (
    review_demo_series,
    review_demo_session,
    evaluate_session_decision,
    generate_session_runbook,
    session_startup_check,
    write_session_discipline_report,
)


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "data", replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"))
    object.__setattr__(settings, "recovery", RecoveryConfig(reconnect_retries=1, reconnect_backoff_seconds=0.0, require_state_restore_clean=True))
    object.__setattr__(settings, "backtest", BacktestConfig(use_atr_stops=False, fixed_stop_loss_pct=0.002, fixed_take_profit_pct=0.004, max_holding_bars=5))
    object.__setattr__(settings, "operational", OperationalConfig(persistence_state_filename="runtime_state.json"))
    return settings


def _clean_state() -> PaperEngineState:
    return PaperEngineState(
        account_state=AccountState(1000.0, 1000.0, 1000.0),
        broker_sync_status=BrokerSyncStatus(),
        daily_loss_tracker=DailyLossTracker("2026-01-01", 0.0, 50.0, False),
    )


# ── Runbook structure ────────────────────────────────────────────────────────

def test_runbook_generated_with_required_phases() -> None:
    runbook = generate_session_runbook()
    phases = {s.phase for s in runbook.steps}
    required = {"startup", "pre_run", "restart", "mismatch", "disconnect", "close", "post_run"}
    assert required.issubset(phases), f"Missing phases: {required - phases}"


def test_runbook_has_abort_hold_continue_criteria() -> None:
    runbook = generate_session_runbook()
    assert len(runbook.abort_criteria) >= 5
    assert len(runbook.hold_criteria) >= 3
    assert len(runbook.continue_criteria) >= 4


def test_runbook_steps_have_required_fields() -> None:
    runbook = generate_session_runbook()
    for step in runbook.steps:
        assert step.step_id != ""
        assert step.title != ""
        assert step.action != ""
        assert step.on_failure in {"abort", "hold", "continue"}


def test_runbook_post_run_checklist_present() -> None:
    runbook = generate_session_runbook()
    assert len(runbook.post_run_checklist) >= 5


def test_runbook_is_serialisable_as_json() -> None:
    runbook = generate_session_runbook()
    d = runbook.to_dict()
    assert json.loads(json.dumps(d))


def test_runbook_version_field_present() -> None:
    runbook = generate_session_runbook()
    assert runbook.version != ""


# ── Session startup check: clean state → proceed ────────────────────────────

def test_startup_check_fresh_state_proceeds(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    # No state file → fresh start
    report = session_startup_check(settings)
    assert report.decision == "proceed"
    assert report.ok is True
    assert report.failed_checks == []


def test_startup_check_with_valid_state_proceeds(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = fresh_state(settings.backtest.starting_balance_usd, "test")
    persist_runtime_state(build_runtime_state_path(settings), state, {})
    report = session_startup_check(settings)
    assert report.decision == "proceed"
    assert report.ok is True


def test_startup_check_kill_switch_active_aborts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = fresh_state(settings.backtest.starting_balance_usd, "test")
    alerts: list[AlertRecord] = []
    activate_kill_switch(state, "test_kill", "manual", alerts)
    persist_runtime_state(build_runtime_state_path(settings), state, {})
    report = session_startup_check(settings)
    assert report.decision == "abort"
    assert "kill_switch_not_active" in report.failed_checks


def test_startup_check_no_trade_mode_holds(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = fresh_state(settings.backtest.starting_balance_usd, "test")
    alerts: list[AlertRecord] = []
    activate_no_trade_mode(state, "volatility", "circuit_breaker", alerts)
    persist_runtime_state(build_runtime_state_path(settings), state, {})
    report = session_startup_check(settings)
    # no_trade_mode is a warning → hold, not abort
    assert report.decision == "hold"
    assert "no_trade_mode_active" in report.warnings


def test_startup_check_daily_loss_blocked_holds(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = fresh_state(settings.backtest.starting_balance_usd, "test")
    state.daily_loss_tracker.blocked = True
    persist_runtime_state(build_runtime_state_path(settings), state, {})
    report = session_startup_check(settings)
    assert report.decision == "hold"
    assert "daily_loss_blocked" in report.warnings


def test_startup_check_corrupt_state_aborts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = build_runtime_state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{bad json", encoding="utf-8")
    report = session_startup_check(settings)
    assert report.decision == "abort"
    assert "state_restore" in report.failed_checks


# ── Session decision evaluation ──────────────────────────────────────────────

def test_evaluate_session_decision_all_clear() -> None:
    state = _clean_state()
    decision = evaluate_session_decision(state, reconcile_ok=True, reconnect_ok=True, critical_alerts=0)
    assert decision.action == "continue"
    assert decision.reason == "all_clear"


def test_evaluate_session_decision_kill_switch_aborts() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    activate_kill_switch(state, "reason", "manual", alerts)
    decision = evaluate_session_decision(state, reconcile_ok=True, reconnect_ok=True, critical_alerts=0)
    assert decision.action == "abort"
    assert "kill_switch" in decision.reason


def test_evaluate_session_decision_reconcile_fail_aborts() -> None:
    state = _clean_state()
    decision = evaluate_session_decision(state, reconcile_ok=False, reconnect_ok=True, critical_alerts=0)
    assert decision.action == "abort"


def test_evaluate_session_decision_reconnect_pending_holds() -> None:
    state = _clean_state()
    decision = evaluate_session_decision(state, reconcile_ok=True, reconnect_ok=False, critical_alerts=0)
    assert decision.action == "hold"


def test_evaluate_session_decision_critical_alerts_aborts() -> None:
    state = _clean_state()
    decision = evaluate_session_decision(state, reconcile_ok=True, reconnect_ok=True, critical_alerts=1)
    assert decision.action == "abort"


def test_evaluate_session_decision_serialisable() -> None:
    state = _clean_state()
    decision = evaluate_session_decision(state, reconcile_ok=True, reconnect_ok=True, critical_alerts=0)
    assert json.loads(json.dumps(decision.to_dict()))


def test_review_demo_session_healthy_when_complete(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    startup = session_startup_check(settings)
    review = review_demo_session(
        session_id="sess-01",
        session_series_id="series-01",
        startup_report=startup,
        session_evidence={
            "preflight_ok": True,
            "artifact_paths": {
                "preflight_report": "a.json",
                "execution_report": "b.json",
                "reconciliation_report": "c.json",
                "session_evidence": "d.json",
            },
            "execution_summary": {"orders_sent": 1, "orders_rejected": 0},
            "divergence_summary": {"divergence_events": 0},
            "restore_recovery_summary": {"restore_events": 1, "recovery_events": 0},
            "final_state_summary": {"kill_switch_active": False, "circuit_breaker_triggered": False},
        },
    )
    assert review.classification == "healthy"
    assert review.valid_for_forward_validation is True


def test_review_demo_session_fails_on_kill_switch() -> None:
    review = review_demo_session(
        session_id="sess-02",
        session_series_id="series-01",
        startup_report=None,
        session_evidence={
            "preflight_ok": True,
            "artifact_paths": {
                "preflight_report": "a.json",
                "execution_report": "b.json",
                "reconciliation_report": "c.json",
                "session_evidence": "d.json",
            },
            "execution_summary": {"orders_sent": 1, "orders_rejected": 0},
            "divergence_summary": {"divergence_events": 0},
            "restore_recovery_summary": {"restore_events": 1, "recovery_events": 0},
            "final_state_summary": {"kill_switch_active": True, "circuit_breaker_triggered": False},
        },
    )
    assert review.classification == "failed"
    assert review.valid_for_forward_validation is False


def test_review_demo_series_requires_complete_target_sessions() -> None:
    review = review_demo_series(
        session_series_id="series-01",
        target_sessions=3,
        aggregate_counts={
            "successful_sessions": 2,
            "aborted_sessions": 0,
            "kill_switch_events": 0,
            "circuit_breaker_triggers": 0,
            "sessions_with_divergence": 0,
            "sessions_with_recovery": 0,
        },
        session_reviews=[
            {"valid_for_forward_validation": True, "evidence_complete": True},
            {"valid_for_forward_validation": True, "evidence_complete": True},
        ],
    )
    assert review.classification == "failed"
    assert review.valid_forward_series is False


# ── Write startup report to disk ─────────────────────────────────────────────

def test_write_session_discipline_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = session_startup_check(settings)
    report_path = tmp_path / "session_startup_report.json"
    write_session_discipline_report(report_path, report)
    assert report_path.exists()
    loaded = json.loads(report_path.read_text())
    assert "decision" in loaded
    assert "evaluated_at" in loaded
