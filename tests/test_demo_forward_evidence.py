"""BLOQUE 3 — Demo forward evidence consolidation tests.

Validates:
- Evidence artifact builds with correct structure
- Evidence is JSON-serializable
- Evidence captures preflight, signals, trades, and decisions
- write_demo_session_evidence creates a valid file
- Evidence includes session_id, symbol, time range, and provenance fields
"""
from __future__ import annotations

import json
from pathlib import Path

from iris_bot.demo_forward_evidence import (
    DemoSessionEvidence,
    build_demo_session_evidence,
    write_demo_session_evidence,
)


def _minimal_preflight() -> dict:
    return {
        "ok": True,
        "checks": {
            "config_enabled": {"ok": True},
            "target_symbol": {"ok": True},
            "operational_state": {"ok": True, "kill_switch_active": False},
        },
    }


# ── Build: required fields ───────────────────────────────────────────────────

def test_build_demo_session_evidence_returns_correct_type() -> None:
    evidence = build_demo_session_evidence(
        session_id="sess-001",
        symbol="EURUSD",
        start_time="2026-01-01T09:00:00Z",
        end_time="2026-01-01T10:00:00Z",
        preflight_report=_minimal_preflight(),
    )
    assert isinstance(evidence, DemoSessionEvidence)


def test_build_evidence_session_id_and_symbol() -> None:
    evidence = build_demo_session_evidence(
        session_id="sess-002",
        symbol="USDJPY",
        start_time="2026-01-01T09:00:00Z",
        end_time="2026-01-01T10:00:00Z",
        preflight_report=_minimal_preflight(),
    )
    assert evidence.session_id == "sess-002"
    assert evidence.symbol == "USDJPY"


def test_build_evidence_preflight_ok_extracted() -> None:
    ok_preflight = {"ok": True, "checks": {}}
    evidence = build_demo_session_evidence(
        session_id="sess-003", symbol="EURUSD",
        start_time="2026-01-01T09:00:00Z", end_time="2026-01-01T10:00:00Z",
        preflight_report=ok_preflight,
    )
    assert evidence.preflight_ok is True


def test_build_evidence_preflight_not_ok_extracted() -> None:
    bad_preflight = {"ok": False, "checks": {"operational_state": {"ok": False}}}
    evidence = build_demo_session_evidence(
        session_id="sess-004", symbol="EURUSD",
        start_time="2026-01-01T09:00:00Z", end_time="2026-01-01T10:00:00Z",
        preflight_report=bad_preflight,
    )
    assert evidence.preflight_ok is False


def test_build_evidence_trade_counters() -> None:
    evidence = build_demo_session_evidence(
        session_id="sess-005", symbol="EURUSD",
        start_time="2026-01-01T09:00:00Z", end_time="2026-01-01T10:00:00Z",
        preflight_report=_minimal_preflight(),
        trades_opened=2, trades_closed=2,
        signals_evaluated=5, no_trade_signals=3, blocked_signals=0,
    )
    assert evidence.trade_summary["trades_opened"] == 2
    assert evidence.trade_summary["trades_closed"] == 2
    assert evidence.signal_summary["signals_generated"] == 5
    assert evidence.signal_summary["no_trade_signals"] == 3
    assert evidence.signal_summary["blocked_signals"] == 0


def test_build_evidence_decision_log_populated() -> None:
    decisions = [
        {"cycle": 1, "action": "continue", "reason": "all_clear"},
        {"cycle": 2, "action": "hold", "reason": "no_trade_mode_active"},
    ]
    evidence = build_demo_session_evidence(
        session_id="sess-006", symbol="EURUSD",
        start_time="2026-01-01T09:00:00Z", end_time="2026-01-01T10:00:00Z",
        preflight_report=_minimal_preflight(),
        session_decision_log=decisions,
    )
    assert len(evidence.session_decision_log) == 2
    assert evidence.session_decision_log[0]["action"] == "continue"


def test_build_evidence_final_state_summary() -> None:
    summary = {"balance": 1050.0, "open_positions": 0, "blocked": False}
    evidence = build_demo_session_evidence(
        session_id="sess-007", symbol="EURUSD",
        start_time="2026-01-01T09:00:00Z", end_time="2026-01-01T10:00:00Z",
        preflight_report=_minimal_preflight(),
        final_state_summary=summary,
    )
    assert evidence.final_state_summary["balance"] == 1050.0


def test_build_evidence_defaults_to_empty_collections() -> None:
    evidence = build_demo_session_evidence(
        session_id="sess-008", symbol="EURUSD",
        start_time="2026-01-01T09:00:00Z", end_time="2026-01-01T10:00:00Z",
        preflight_report=_minimal_preflight(),
    )
    assert evidence.session_decision_log == []
    assert evidence.final_state_summary == {}
    assert evidence.trade_summary["trades_opened"] == 0


# ── Serialization ─────────────────────────────────────────────────────────────

def test_evidence_is_json_serializable() -> None:
    evidence = build_demo_session_evidence(
        session_id="sess-009", symbol="EURUSD",
        start_time="2026-01-01T09:00:00Z", end_time="2026-01-01T10:00:00Z",
        preflight_report=_minimal_preflight(),
        session_decision_log=[{"action": "continue"}],
        final_state_summary={"balance": 1000.0},
    )
    d = evidence.to_dict()
    assert json.loads(json.dumps(d))


def test_evidence_to_dict_has_all_fields() -> None:
    evidence = build_demo_session_evidence(
        session_id="sess-010", symbol="EURUSD",
        start_time="2026-01-01T09:00:00Z", end_time="2026-01-01T10:00:00Z",
        preflight_report=_minimal_preflight(),
    )
    d = evidence.to_dict()
    required_fields = {
        "session_id", "session_series_id", "series_position", "symbol", "start_time", "end_time",
        "preflight_ok", "preflight_checks", "signal_summary", "execution_summary",
        "trade_summary", "performance_summary", "divergence_summary",
        "restore_recovery_summary", "session_decision_log", "final_state_summary",
        "artifact_paths", "generated_at",
    }
    assert required_fields.issubset(set(d.keys()))


# ── Write to disk ─────────────────────────────────────────────────────────────

def test_write_demo_session_evidence_creates_file(tmp_path: Path) -> None:
    evidence = build_demo_session_evidence(
        session_id="sess-011", symbol="EURUSD",
        start_time="2026-01-01T09:00:00Z", end_time="2026-01-01T10:00:00Z",
        preflight_report=_minimal_preflight(),
    )
    path = tmp_path / "evidence" / "session_evidence.json"
    write_demo_session_evidence(path, evidence)
    assert path.exists()
    loaded = json.loads(path.read_text())
    payload = loaded.get("payload", loaded)
    assert payload["session_id"] == "sess-011"
    assert payload["symbol"] == "EURUSD"


def test_write_demo_session_evidence_creates_parent_dirs(tmp_path: Path) -> None:
    evidence = build_demo_session_evidence(
        session_id="sess-012", symbol="GBPUSD",
        start_time="2026-01-01T09:00:00Z", end_time="2026-01-01T10:00:00Z",
        preflight_report=_minimal_preflight(),
    )
    deep_path = tmp_path / "a" / "b" / "c" / "evidence.json"
    write_demo_session_evidence(deep_path, evidence)
    assert deep_path.exists()
