from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from iris_bot.config import Settings
from iris_bot.durable_io import durable_write_json
from iris_bot.prolonged_serious_demo import generate_prolonged_serious_demo_report
from iris_bot.artifacts import wrap_artifact


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "data", replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"))
    return settings


def test_prolonged_gate_blocks_without_forward_series(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_prolonged_serious_demo_report(settings)
    assert report["decision"] == "not_ready_for_prolonged_serious_demo"
    assert "no_forward_series_evidence" in report["blockers"]


def test_prolonged_gate_ready_with_valid_series(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    runtime = settings.data.runtime_dir / "demo_forward_validation"
    artifacts = runtime / "series_artifacts"
    series_id = "eurusd_20260406T000000Z_forward_series"
    registry_path = runtime / "demo_session_series_registry.json"
    durable_write_json(
        registry_path,
        wrap_artifact(
            "demo_session_series",
            {
                "active_series_id": "",
                "series": {
                    series_id: {
                        "session_series_id": series_id,
                        "symbol": "EURUSD",
                        "status": "completed",
                        "started_at": "2026-04-06T00:00:00+00:00",
                        "ended_at": "2026-04-06T01:00:00+00:00",
                        "target_sessions": 3,
                        "latest_series_review": {"valid_forward_series": True},
                    }
                },
            },
        ),
    )
    durable_write_json(
        artifacts / f"{series_id}.json",
        wrap_artifact(
            "demo_session_series",
            {
                "session_series_id": series_id,
                "aggregate_counts": {
                    "successful_sessions": 3,
                    "aborted_sessions": 0,
                    "kill_switch_events": 0,
                    "circuit_breaker_triggers": 0,
                    "sessions_with_divergence": 0,
                    "sessions_with_recovery": 0,
                },
                "session_reviews": [
                    {"evidence_complete": True, "valid_for_forward_validation": True, "recommendation": "continue"},
                    {"evidence_complete": True, "valid_for_forward_validation": True, "recommendation": "continue"},
                    {"evidence_complete": True, "valid_for_forward_validation": True, "recommendation": "continue"},
                ],
            },
        ),
    )
    monkeypatch.setattr(
        "iris_bot.prolonged_serious_demo.generate_serious_demo_control_report",
        lambda settings: {"decision": "ready_for_controlled_serious_demo"},
    )
    report = generate_prolonged_serious_demo_report(settings)
    assert report["decision"] == "ready_for_prolonged_serious_demo"


def test_prolonged_gate_with_degrading_reconcile_returns_reservations(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    runtime = settings.data.runtime_dir / "demo_forward_validation"
    artifacts = runtime / "series_artifacts"
    series_id = "eurusd_20260406T000000Z_forward_series"
    durable_write_json(
        runtime / "demo_session_series_registry.json",
        wrap_artifact(
            "demo_session_series",
            {
                "active_series_id": "",
                "series": {
                    series_id: {
                        "session_series_id": series_id,
                        "symbol": "EURUSD",
                        "status": "completed",
                        "started_at": "2026-04-06T00:00:00+00:00",
                        "ended_at": "2026-04-06T01:00:00+00:00",
                        "target_sessions": 3,
                        "latest_series_review": {"valid_forward_series": True},
                    }
                },
            },
        ),
    )
    durable_write_json(
        artifacts / f"{series_id}.json",
        wrap_artifact(
            "demo_session_series",
            {
                "session_series_id": series_id,
                "aggregate_counts": {
                    "successful_sessions": 3,
                    "aborted_sessions": 0,
                    "kill_switch_events": 0,
                    "circuit_breaker_triggers": 0,
                    "sessions_with_divergence": 2,
                    "sessions_with_recovery": 0,
                },
                "session_reviews": [
                    {"evidence_complete": True, "valid_for_forward_validation": True, "recommendation": "continue"},
                    {"evidence_complete": True, "valid_for_forward_validation": True, "recommendation": "hold"},
                    {"evidence_complete": True, "valid_for_forward_validation": True, "recommendation": "continue"},
                ],
            },
        ),
    )
    monkeypatch.setattr(
        "iris_bot.prolonged_serious_demo.generate_serious_demo_control_report",
        lambda settings: {"decision": "ready_for_controlled_serious_demo"},
    )
    report = generate_prolonged_serious_demo_report(settings)
    assert report["decision"] == "prolonged_serious_demo_with_reservations"


# ---------------------------------------------------------------------------
# demo_serious_validated gate tests
# ---------------------------------------------------------------------------

from iris_bot.prolonged_serious_demo import generate_demo_serious_validated_report


def _write_registry_with_series(tmp_path: Path, series_list: list[dict]) -> None:
    from iris_bot.durable_io import durable_write_json
    from iris_bot.artifacts import wrap_artifact

    runtime = tmp_path / "runtime" / "demo_forward_validation"
    runtime.mkdir(parents=True, exist_ok=True)
    artifacts = runtime / "series_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    series_map = {}
    for s in series_list:
        sid = s["session_series_id"]
        series_map[sid] = {
            "session_series_id": sid,
            "symbol": "EURUSD",
            "status": s.get("status", "completed"),
            "started_at": s.get("started_at", "2026-04-07T22:00:00+00:00"),
            "ended_at": s.get("ended_at"),
            "target_sessions": 3,
            "session_ids": s.get("session_ids", []),
            "session_evidence_paths": [],
            "session_review_paths": [],
            "aggregate_counts": s.get("aggregate_counts", {
                "successful_sessions": 3, "aborted_sessions": 0, "hold_sessions": 0,
                "orders_sent": 3, "orders_rejected": 0, "reconciliations": 3,
                "recoveries": 0, "restore_events": 0, "circuit_breaker_triggers": 0,
                "kill_switch_events": 0, "sessions_with_divergence": 0, "sessions_with_recovery": 0,
            }),
            "latest_series_review": {"valid_forward_series": True},
        }
        if s.get("write_artifact", True):
            durable_write_json(
                artifacts / f"{sid}.json",
                wrap_artifact("demo_session_series", {
                    "session_series_id": sid,
                    "aggregate_counts": series_map[sid]["aggregate_counts"],
                    "session_reviews": [
                        {"evidence_complete": True, "valid_for_forward_validation": True, "recommendation": "continue"},
                        {"evidence_complete": True, "valid_for_forward_validation": True, "recommendation": "continue"},
                        {"evidence_complete": True, "valid_for_forward_validation": True, "recommendation": "continue"},
                    ],
                }),
            )

    durable_write_json(
        runtime / "demo_session_series_registry.json",
        wrap_artifact("demo_session_series", {
            "active_series_id": "",
            "series": series_map,
        }),
    )


def test_demo_serious_validated_blocks_without_series(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "iris_bot.prolonged_serious_demo.generate_serious_demo_control_report",
        lambda s: {"decision": "ready_for_controlled_serious_demo"},
    )
    report = generate_demo_serious_validated_report(settings)
    assert report["decision"] == "not_yet_demo_validated"
    blockers_str = " ".join(report["blockers"])
    assert "insufficient_completed_valid_series" in blockers_str


def test_demo_serious_validated_with_reservations_same_day(tmp_path: Path, monkeypatch) -> None:
    """Two valid series within same day → demo_validated_with_reservations."""
    settings = _settings(tmp_path)
    _write_registry_with_series(tmp_path, [
        {"session_series_id": "s1", "started_at": "2026-04-07T22:00:00+00:00"},
        {"session_series_id": "s2", "started_at": "2026-04-07T22:30:00+00:00"},
    ])
    monkeypatch.setattr(
        "iris_bot.prolonged_serious_demo.generate_serious_demo_control_report",
        lambda s: {"decision": "ready_for_controlled_serious_demo"},
    )
    report = generate_demo_serious_validated_report(settings)
    assert report["decision"] == "demo_validated_with_reservations"
    assert "limited_temporal_diversity:same_day_operation" in report["warnings"]
    assert report["cumulative_summary"]["completed_valid_series"] == 2
    assert report["cumulative_summary"]["total_successful_sessions"] == 6


def test_demo_serious_validated_full_with_multi_day(tmp_path: Path, monkeypatch) -> None:
    """Two valid series > 1 hour apart → demo_serious_validated."""
    settings = _settings(tmp_path)
    _write_registry_with_series(tmp_path, [
        {"session_series_id": "s1", "started_at": "2026-04-07T08:00:00+00:00"},
        {"session_series_id": "s2", "started_at": "2026-04-07T16:00:00+00:00"},
    ])
    monkeypatch.setattr(
        "iris_bot.prolonged_serious_demo.generate_serious_demo_control_report",
        lambda s: {"decision": "ready_for_controlled_serious_demo"},
    )
    report = generate_demo_serious_validated_report(settings)
    assert report["decision"] == "demo_serious_validated"
    assert report["blockers"] == []
    assert report["warnings"] == []


def test_demo_serious_validated_blocks_on_critical_failure(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    _write_registry_with_series(tmp_path, [
        {"session_series_id": "s1", "started_at": "2026-04-07T08:00:00+00:00",
         "aggregate_counts": {
             "successful_sessions": 3, "aborted_sessions": 1, "hold_sessions": 0,
             "orders_sent": 3, "orders_rejected": 0, "reconciliations": 0,
             "recoveries": 0, "restore_events": 0, "circuit_breaker_triggers": 0,
             "kill_switch_events": 1, "sessions_with_divergence": 0, "sessions_with_recovery": 0,
         }},
        {"session_series_id": "s2", "started_at": "2026-04-07T16:00:00+00:00"},
    ])
    monkeypatch.setattr(
        "iris_bot.prolonged_serious_demo.generate_serious_demo_control_report",
        lambda s: {"decision": "ready_for_controlled_serious_demo"},
    )
    report = generate_demo_serious_validated_report(settings)
    assert report["decision"] == "not_yet_demo_validated"
    assert "critical_failures_detected" in report["blockers"]
