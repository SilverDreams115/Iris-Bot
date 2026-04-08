"""BLOQUE 1 + BLOQUE 4 — Demo execution fail-safe behavior tests.

Validates:
- Preflight reads operational_state from correct nesting (runtime_state["state"])
- Preflight blocks when kill_switch is active in persisted state
- Preflight blocks when no_trade_mode is active in persisted state
- Preflight blocks when circuit_breaker conditions would trigger
- Clean state allows operational_state check to pass
- run_demo_execution_command returns non-zero when preflight fails
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from iris_bot.config import BacktestConfig, MT5Config, OperationalConfig, Settings
from iris_bot.demo_execution import demo_execution_preflight_payload, run_demo_execution_command
from iris_bot.mt5 import MT5Client, MT5ValidationReport


class _FakeDemoClient(MT5Client):
    """Minimal client for preflight testing — never connects to real MT5."""

    def __init__(self) -> None:
        super().__init__(MT5Config(enabled=True), mt5_module=object())

    def connect(self) -> bool:
        return False

    def shutdown(self) -> None:
        return None

    def check(self, symbols: tuple[str, ...]) -> MT5ValidationReport:
        return MT5ValidationReport(False, False, False, [], {})


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "data", replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"))
    object.__setattr__(settings, "backtest", BacktestConfig(use_atr_stops=False, fixed_stop_loss_pct=0.002, fixed_take_profit_pct=0.004, max_holding_bars=5))
    object.__setattr__(settings, "operational", OperationalConfig(persistence_state_filename="runtime_state.json"))
    return settings


def _write_runtime_state(settings: Settings, state_dict: dict) -> None:
    """Write a runtime state file with the given inner state payload."""
    runtime_dir = settings.data.runtime_dir
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / settings.operational.persistence_state_filename
    path.write_text(
        json.dumps({"saved_at": "2026-01-01T00:00:00Z", "schema_version": 1, "state": state_dict}),
        encoding="utf-8",
    )


# ── Regression: operational_state reads from nested "state" key ──────────────

def test_preflight_reads_blocked_reasons_from_nested_state_key(tmp_path: Path) -> None:
    """Regression: blocked_reasons must come from runtime_state['state'], not runtime_state."""
    settings = _settings(tmp_path)
    _write_runtime_state(settings, {"blocked_reasons": ["kill_switch:nesting_test"]})
    payload = demo_execution_preflight_payload(settings, client=_FakeDemoClient())
    assert payload["checks"]["operational_state"]["kill_switch_active"] is True, (
        "preflight did not read blocked_reasons from runtime_state['state']"
    )


def test_preflight_outer_envelope_blocked_reasons_not_used(tmp_path: Path) -> None:
    """Regression: blocked_reasons in outer envelope must not be mistaken for state."""
    settings = _settings(tmp_path)
    runtime_dir = settings.data.runtime_dir
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / settings.operational.persistence_state_filename
    # Kill switch only in OUTER envelope (wrong place); inner state is clean
    path.write_text(
        json.dumps({
            "saved_at": "2026-01-01T00:00:00Z",
            "blocked_reasons": ["kill_switch:should_not_be_read"],
            "state": {"blocked_reasons": []},
        }),
        encoding="utf-8",
    )
    payload = demo_execution_preflight_payload(settings, client=_FakeDemoClient())
    # Inner state is clean → operational_state should pass
    assert payload["checks"]["operational_state"]["ok"] is True


# ── Kill switch blocks preflight ─────────────────────────────────────────────

def test_preflight_blocks_on_kill_switch_active(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_runtime_state(settings, {"blocked_reasons": ["kill_switch:manual_abort"]})
    payload = demo_execution_preflight_payload(settings, client=_FakeDemoClient())
    os_check = payload["checks"]["operational_state"]
    assert os_check["ok"] is False
    assert os_check["kill_switch_active"] is True
    assert os_check["reason"] == "kill_switch_active"


def test_preflight_operational_state_contains_blocked_reasons(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_runtime_state(settings, {"blocked_reasons": ["kill_switch:a", "kill_switch:b"]})
    payload = demo_execution_preflight_payload(settings, client=_FakeDemoClient())
    os_check = payload["checks"]["operational_state"]
    assert "kill_switch:a" in os_check["blocked_reasons"]
    assert "kill_switch:b" in os_check["blocked_reasons"]


# ── No-trade mode blocks preflight ───────────────────────────────────────────

def test_preflight_blocks_on_no_trade_mode_active(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_runtime_state(settings, {"blocked_reasons": ["no_trade_mode:volatility_spike"]})
    payload = demo_execution_preflight_payload(settings, client=_FakeDemoClient())
    os_check = payload["checks"]["operational_state"]
    assert os_check["ok"] is False
    assert os_check["no_trade_mode_active"] is True
    assert os_check["reason"] == "no_trade_mode_active"


# ── Circuit breaker: critical discrepancy blocks preflight ───────────────────

def test_preflight_blocks_on_critical_discrepancy(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_runtime_state(settings, {
        "blocked_reasons": [],
        "broker_sync_status": {
            "critical_discrepancy_count": 1,
            "last_sync_timestamp": "",
            "sync_ok": False,
        },
    })
    payload = demo_execution_preflight_payload(settings, client=_FakeDemoClient())
    os_check = payload["checks"]["operational_state"]
    assert os_check["ok"] is False
    # Either the raw count check or the circuit breaker fires
    assert os_check["critical_discrepancy_count"] >= 1 or os_check["circuit_breaker_triggered"] is True


# ── Clean state passes operational_state check ───────────────────────────────

def test_preflight_operational_state_passes_on_clean_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_runtime_state(settings, {
        "blocked_reasons": [],
        "broker_sync_status": {
            "critical_discrepancy_count": 0,
            "last_sync_timestamp": "",
            "sync_ok": True,
        },
    })
    payload = demo_execution_preflight_payload(settings, client=_FakeDemoClient())
    os_check = payload["checks"]["operational_state"]
    assert os_check["ok"] is True
    assert os_check["kill_switch_active"] is False
    assert os_check["no_trade_mode_active"] is False
    assert os_check["circuit_breaker_triggered"] is False


def test_preflight_operational_state_passes_when_no_state_file(tmp_path: Path) -> None:
    """No persisted state (fresh start) → operational_state should be clean."""
    settings = _settings(tmp_path)
    settings.data.runtime_dir.mkdir(parents=True, exist_ok=True)
    payload = demo_execution_preflight_payload(settings, client=_FakeDemoClient())
    os_check = payload["checks"]["operational_state"]
    assert os_check["ok"] is True


# ── run_demo_execution_command aborts on preflight failure ───────────────────

def test_run_demo_execution_returns_nonzero_on_preflight_failure(tmp_path: Path) -> None:
    """run_demo_execution_command must abort (exit != 0) when preflight fails."""
    settings = _settings(tmp_path)
    # Kill switch active → preflight.operational_state fails
    _write_runtime_state(settings, {"blocked_reasons": ["kill_switch:abort_test"]})
    exit_code = run_demo_execution_command(settings)
    assert exit_code != 0, "run_demo_execution must not proceed when preflight fails"


def test_run_demo_execution_returns_nonzero_without_registry(tmp_path: Path) -> None:
    """run_demo_execution_command must abort if no demo execution registry is set up."""
    settings = _settings(tmp_path)
    settings.data.runtime_dir.mkdir(parents=True, exist_ok=True)
    # No registry, no model, no symbol → multiple checks fail
    exit_code = run_demo_execution_command(settings)
    assert exit_code != 0
