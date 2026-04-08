"""BLOQUE 2 — Demo session guard tests.

Validates:
- Session precheck proceeds on clean state
- Session precheck aborts on kill_switch active
- Session precheck holds on no_trade_mode active (without kill_switch)
- Session precheck holds on daily_loss_blocked
- Session precheck aborts on max_active_positions exceeded
- Report is JSON-serializable and writable to disk
- Idempotency: same state yields same decision
"""
from __future__ import annotations

import json
from pathlib import Path

from iris_bot.demo_session_guard import (
    DemoSessionGatingReport,
    DemoSessionLimits,
    run_demo_session_precheck,
    write_demo_session_gating_report,
)
from iris_bot.kill_switch import activate_kill_switch, activate_no_trade_mode
from iris_bot.operational import (
    AccountState,
    AlertRecord,
    BrokerSyncStatus,
    DailyLossTracker,
    PaperEngineState,
    PaperPosition,
)


def _clean_state() -> PaperEngineState:
    return PaperEngineState(
        account_state=AccountState(1000.0, 1000.0, 1000.0),
        broker_sync_status=BrokerSyncStatus(),
        daily_loss_tracker=DailyLossTracker("2026-01-01", 0.0, 50.0, False),
    )


# ── Clean state: proceed ─────────────────────────────────────────────────────

def test_session_precheck_proceeds_on_clean_state() -> None:
    state = _clean_state()
    report = run_demo_session_precheck(state)
    assert report.ok is True
    assert report.decision == "proceed"
    assert report.failed_checks == []
    assert report.warnings == []


def test_session_precheck_passed_checks_not_empty_on_clean_state() -> None:
    state = _clean_state()
    report = run_demo_session_precheck(state)
    assert len(report.passed_checks) >= 2


# ── Kill switch: abort ───────────────────────────────────────────────────────

def test_session_precheck_aborts_on_kill_switch() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    activate_kill_switch(state, "test_reason", "manual", alerts)
    report = run_demo_session_precheck(state)
    assert report.ok is False
    assert report.decision == "abort"
    assert "kill_switch_not_active" in report.failed_checks


def test_session_precheck_abort_details_include_blocked_reasons() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    activate_kill_switch(state, "audit_test", "manual", alerts)
    report = run_demo_session_precheck(state)
    assert report.details["kill_switch_active"] is True
    assert any("kill_switch:audit_test" in r for r in report.details.get("blocked_reasons", []))


# ── No-trade mode: hold ──────────────────────────────────────────────────────

def test_session_precheck_holds_on_no_trade_mode() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    activate_no_trade_mode(state, "volatility_spike", "circuit_breaker", alerts)
    report = run_demo_session_precheck(state)
    assert report.ok is False
    assert report.decision == "hold"
    assert "no_trade_mode_active" in report.warnings


def test_session_precheck_kill_switch_takes_precedence_over_no_trade() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    activate_kill_switch(state, "hard_stop", "manual", alerts)
    activate_no_trade_mode(state, "extra", "circuit_breaker", alerts)
    report = run_demo_session_precheck(state)
    # Kill switch → abort (higher severity than hold)
    assert report.decision == "abort"


# ── Daily loss: hold ─────────────────────────────────────────────────────────

def test_session_precheck_holds_on_daily_loss_blocked() -> None:
    state = _clean_state()
    state.daily_loss_tracker.blocked = True
    report = run_demo_session_precheck(state)
    assert report.ok is False
    assert report.decision == "hold"
    assert "daily_loss_blocked" in report.warnings


# ── Max active positions: abort ──────────────────────────────────────────────

def test_session_precheck_aborts_on_max_positions_exceeded() -> None:
    state = _clean_state()
    # Add 1 open position → at limit (max=1 by default)
    state.open_positions["EURUSD"] = PaperPosition(
        symbol="EURUSD", timeframe="M15", direction=1,
        entry_timestamp="2026-01-01T00:00:00Z", signal_timestamp="2026-01-01T00:00:00Z",
        entry_index=0, volume_lots=0.01, entry_price=1.1000,
        stop_loss_price=1.0980, take_profit_price=1.1030,
        commission_entry_usd=0.0, bars_held=0,
        probability_long=0.7, probability_short=0.3,
        stop_policy="fixed", target_policy="fixed",
    )
    limits = DemoSessionLimits(max_active_positions=1)
    report = run_demo_session_precheck(state, limits)
    assert report.ok is False
    assert report.decision == "abort"
    assert "max_active_positions_exceeded" in report.failed_checks


def test_session_precheck_proceeds_when_under_position_limit() -> None:
    state = _clean_state()
    # No positions → under limit
    limits = DemoSessionLimits(max_active_positions=2)
    report = run_demo_session_precheck(state, limits)
    assert report.decision == "proceed"
    assert "max_active_positions_within_limit" in report.passed_checks


# ── Serialization and disk artifact ─────────────────────────────────────────

def test_session_gating_report_is_json_serializable() -> None:
    state = _clean_state()
    report = run_demo_session_precheck(state)
    assert isinstance(report, DemoSessionGatingReport)
    assert json.loads(json.dumps(report.to_dict()))


def test_write_demo_session_gating_report(tmp_path: Path) -> None:
    state = _clean_state()
    report = run_demo_session_precheck(state)
    path = tmp_path / "session_gating_report.json"
    write_demo_session_gating_report(path, report)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["decision"] == "proceed"
    assert "evaluated_at" in loaded


def test_session_precheck_report_has_evaluated_at() -> None:
    state = _clean_state()
    report = run_demo_session_precheck(state)
    assert report.evaluated_at != ""
    assert "T" in report.evaluated_at  # ISO format


# ── Idempotency ──────────────────────────────────────────────────────────────

def test_session_precheck_idempotent_on_same_state() -> None:
    state = _clean_state()
    r1 = run_demo_session_precheck(state)
    r2 = run_demo_session_precheck(state)
    assert r1.decision == r2.decision
    assert r1.failed_checks == r2.failed_checks
    assert r1.warnings == r2.warnings
