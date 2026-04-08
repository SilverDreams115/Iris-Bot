"""BLOQUE 5 — Kill Switch / Circuit Breaker / No-Trade Mode Tests.

Validates:
- Manual kill switch activation
- No-trade mode activation
- Circuit breaker condition evaluation
- Auditable artifact production
- State persistence (blocked_reasons survive restart)
- Idempotency
"""
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
from iris_bot.kill_switch import (
    KillSwitchReport,
    activate_kill_switch,
    activate_no_trade_mode,
    build_default_circuit_breaker_conditions,
    circuit_breaker_check,
    is_kill_switch_active,
    is_no_trade_mode_active,
    write_kill_switch_report,
)
from iris_bot.operational import (
    AccountState,
    AlertRecord,
    BrokerSyncStatus,
    DailyLossTracker,
    PaperEngineState,
)
from iris_bot.resilient import (
    build_runtime_state_path,
    fresh_state,
    persist_runtime_state,
    restore_runtime_state,
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


# ── Kill switch activation ───────────────────────────────────────────────────

def test_kill_switch_adds_blocked_reason() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    activate_kill_switch(state, "manual_abort", "manual", alerts)
    assert any("kill_switch:manual_abort" in r for r in state.blocked_reasons)


def test_kill_switch_emits_critical_alert() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    activate_kill_switch(state, "test_reason", "manual", alerts)
    assert len(alerts) == 1
    assert alerts[0].severity == "critical"
    assert alerts[0].category == "kill_switch_activated"


def test_kill_switch_returns_auditable_report() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    report = activate_kill_switch(state, "test_reason", "manual", alerts)
    assert isinstance(report, KillSwitchReport)
    assert report.event_type == "kill_switch"
    assert report.triggered_by == "manual"
    assert report.reason == "test_reason"
    assert report.triggered_at != ""
    # Report must be JSON-serialisable
    assert json.loads(json.dumps(report.to_dict()))


def test_kill_switch_is_active_after_activation() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    assert not is_kill_switch_active(state)
    activate_kill_switch(state, "reason", "manual", alerts)
    assert is_kill_switch_active(state)


def test_kill_switch_idempotent() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    activate_kill_switch(state, "reason", "manual", alerts)
    r2 = activate_kill_switch(state, "reason", "manual", alerts)
    # Second activation: no new blocked_reason added
    assert r2.blocked_reasons_added == []
    assert state.blocked_reasons.count("kill_switch:reason") == 1


# ── No-trade mode ────────────────────────────────────────────────────────────

def test_no_trade_mode_adds_blocked_reason() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    activate_no_trade_mode(state, "volatility_spike", "circuit_breaker", alerts)
    assert any("no_trade_mode:volatility_spike" in r for r in state.blocked_reasons)


def test_no_trade_mode_emits_warning_alert() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    activate_no_trade_mode(state, "reason", "circuit_breaker", alerts)
    assert alerts[0].severity == "warning"
    assert alerts[0].category == "no_trade_mode_activated"


def test_no_trade_mode_active_after_activation() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    assert not is_no_trade_mode_active(state)
    activate_no_trade_mode(state, "reason", "circuit_breaker", alerts)
    assert is_no_trade_mode_active(state)


def test_kill_switch_implies_no_trade_mode() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    activate_kill_switch(state, "reason", "manual", alerts)
    # kill_switch is more severe than no_trade; no_trade check must also return True
    assert is_no_trade_mode_active(state)


# ── Circuit breaker ──────────────────────────────────────────────────────────

def test_circuit_breaker_triggers_kill_switch_on_critical_discrepancy() -> None:
    state = _clean_state()
    state.broker_sync_status.critical_discrepancy_count = 1
    alerts: list[AlertRecord] = []
    conditions = build_default_circuit_breaker_conditions(max_critical_discrepancies=0)
    result = circuit_breaker_check(state, conditions, alerts)
    assert result is not None
    assert result.event_type == "kill_switch"
    assert is_kill_switch_active(state)


def test_circuit_breaker_triggers_no_trade_on_daily_loss() -> None:
    state = _clean_state()
    state.daily_loss_tracker.blocked = True
    alerts: list[AlertRecord] = []
    # Use a condition list without the critical-discrepancy trigger
    conditions = [c for c in build_default_circuit_breaker_conditions() if c.name == "max_daily_loss_blocked"]
    result = circuit_breaker_check(state, conditions, alerts)
    assert result is not None
    assert result.event_type == "no_trade_mode"
    assert is_no_trade_mode_active(state)


def test_circuit_breaker_no_trigger_on_clean_state() -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    conditions = build_default_circuit_breaker_conditions()
    result = circuit_breaker_check(state, conditions, alerts)
    assert result is None
    assert not is_kill_switch_active(state)
    assert alerts == []


def test_circuit_breaker_custom_condition() -> None:
    state = _clean_state()
    state.blocked_reasons.append("some_block")
    state.blocked_reasons.append("another_block")
    state.blocked_reasons.append("yet_another")
    state.blocked_reasons.append("one_more")  # 4 > max=3
    alerts: list[AlertRecord] = []
    conditions = build_default_circuit_breaker_conditions(max_blocked_reasons=3)
    # Skip first condition (no discrepancy)
    non_critical = [c for c in conditions if c.name != "critical_broker_discrepancy"]
    result = circuit_breaker_check(state, non_critical, alerts)
    assert result is not None
    assert result.condition_name == "accumulating_blocks"


# ── Auditable artifact (write to disk) ──────────────────────────────────────

def test_kill_switch_report_written_to_disk(tmp_path: Path) -> None:
    state = _clean_state()
    alerts: list[AlertRecord] = []
    report = activate_kill_switch(state, "audit_test", "manual", alerts)
    report_path = tmp_path / "kill_switch_report.json"
    write_kill_switch_report(report_path, report)
    assert report_path.exists()
    loaded = json.loads(report_path.read_text())
    assert loaded["event_type"] == "kill_switch"
    assert loaded["reason"] == "audit_test"


# ── Kill switch state persists across restart ────────────────────────────────

def test_kill_switch_state_persists_across_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = fresh_state(settings.backtest.starting_balance_usd, "kill-switch-test")
    alerts: list[AlertRecord] = []
    activate_kill_switch(state, "persist_test", "manual", alerts)

    persist_runtime_state(build_runtime_state_path(settings), state, {})
    restored, report = restore_runtime_state(build_runtime_state_path(settings), require_clean=True)

    assert report.ok
    assert restored is not None
    assert is_kill_switch_active(restored)
    assert any("kill_switch:persist_test" in r for r in restored.blocked_reasons)


# ── Default conditions structure ─────────────────────────────────────────────

def test_default_circuit_breaker_conditions_coverage() -> None:
    conditions = build_default_circuit_breaker_conditions()
    names = {c.name for c in conditions}
    assert "critical_broker_discrepancy" in names
    assert "accumulating_blocks" in names
    assert "max_daily_loss_blocked" in names
    # Each condition must have a severity and reason
    for c in conditions:
        assert c.severity in {"kill_switch", "no_trade"}
        assert c.reason != ""
