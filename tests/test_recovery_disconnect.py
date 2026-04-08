"""BLOQUE 3 — Recovery After Disconnect Drills.

Four scenarios demonstrating that disconnection / loss of broker access
does not degrade state integrity, ownership, or execution discipline.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from iris_bot.config import (
    BacktestConfig,
    MT5Config,
    OperationalConfig,
    RecoveryConfig,
    ReconciliationConfig,
    RiskConfig,
    SessionConfig,
    Settings,
)
from iris_bot.mt5 import BrokerSnapshot, MT5Client
from iris_bot.resilient import (
    broker_snapshot_from_mt5,
    build_operational_status,
    build_runtime_state_path,
    fresh_state,
    persist_runtime_state,
    reconcile_state,
    reconnect_mt5,
    restore_runtime_state,
    run_recovery_drills,
)


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "data", replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"))
    object.__setattr__(settings, "recovery", RecoveryConfig(reconnect_retries=3, reconnect_backoff_seconds=0.0, require_state_restore_clean=True))
    object.__setattr__(settings, "reconciliation", ReconciliationConfig(policy="hard_fail", price_tolerance=0.001, volume_tolerance=0.001))
    object.__setattr__(settings, "backtest", BacktestConfig(use_atr_stops=False, fixed_stop_loss_pct=0.002, fixed_take_profit_pct=0.004, max_holding_bars=5))
    object.__setattr__(settings, "risk", RiskConfig(max_daily_loss_usd=50.0))
    object.__setattr__(settings, "session", SessionConfig(enabled=True, allowed_weekdays=(0, 1, 2, 3, 4), allowed_start_hour_utc=0, allowed_end_hour_utc=23))
    object.__setattr__(settings, "operational", OperationalConfig(persistence_state_filename="runtime_state.json"))
    return settings


class _FakeClientFactory:
    """Minimal MT5Client stand-in for drill scenarios."""

    class _Impl(MT5Client):
        def __init__(self, connect_sequence: list[bool], snapshot: BrokerSnapshot | None = None) -> None:
            super().__init__(MT5Config(enabled=True))
            self._seq = connect_sequence
            self._call = 0
            self._snapshot = snapshot or BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [], [], [])

        def connect(self) -> bool:
            result = self._seq[min(self._call, len(self._seq) - 1)]
            self._call += 1
            self._connected = result
            return result

        def last_error(self) -> object:
            return (1, "Success") if self._connected else (500, "Disconnected")

        def broker_state_snapshot(self, symbols: tuple[str, ...]) -> BrokerSnapshot:
            return self._snapshot

    @classmethod
    def make(cls, connect_sequence: list[bool], snapshot: BrokerSnapshot | None = None) -> "_Impl":
        return cls._Impl(connect_sequence, snapshot)


# ── Scenario 1: Brief disconnect, clean reconnect ────────────────────────────

def test_brief_disconnect_clean_reconnect(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    recovery = settings.recovery
    # First call fails, second succeeds
    client = _FakeClientFactory.make([False, True])
    report = reconnect_mt5(client, recovery)
    assert report.ok is True
    assert report.final_state == "connected"
    assert len(report.attempts) == 2
    assert not report.attempts[0]["ok"]
    assert report.attempts[1]["ok"]


def test_brief_disconnect_reports_all_attempts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    recovery = settings.recovery
    client = _FakeClientFactory.make([False, False, True])
    report = reconnect_mt5(client, recovery)
    assert report.ok is True
    assert len(report.attempts) == 3
    assert all("timestamp" in a for a in report.attempts)
    assert all("health" in a for a in report.attempts)


# ── Scenario 2: Reconnect with broker state changed ──────────────────────────

def test_reconnect_broker_state_changed_detects_mismatch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    # Broker now reports a position that doesn't exist in local state
    new_pos = {
        "ticket": 100, "symbol": "USDJPY", "type": 0, "volume": 0.1,
        "price_open": 145.0, "sl": 144.0, "tp": 146.5, "time": 1735689600,
    }
    snapshot = BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [new_pos], [], [])
    client = _FakeClientFactory.make([True], snapshot)

    # Local state has no positions
    local = fresh_state(settings.backtest.starting_balance_usd, "recovery-drill")
    reconnect = reconnect_mt5(client, settings.recovery)
    broker = broker_snapshot_from_mt5(client.broker_state_snapshot(("USDJPY",)))
    reconcile = reconcile_state(local, broker, settings.reconciliation)

    assert reconnect.ok
    assert reconcile.ok is False
    assert any(d.category == "missing_in_local_state" for d in reconcile.discrepancies)
    mismatch = next(d for d in reconcile.discrepancies if d.category == "missing_in_local_state")
    assert mismatch.severity == "critical"
    assert mismatch.details["symbol"] == "USDJPY"


def test_reconnect_changed_state_report_serialisable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    snapshot = BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [], [], [])
    client = _FakeClientFactory.make([True], snapshot)
    reconnect = reconnect_mt5(client, settings.recovery)
    assert json.loads(json.dumps(reconnect.to_dict()))


# ── Scenario 3: Reconnect with prior artifacts present ───────────────────────

def test_reconnect_with_prior_artifacts_state_preserved(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = build_runtime_state_path(settings)

    # Persist a state with known events
    prior = fresh_state(settings.backtest.starting_balance_usd, "recovery-drill")
    prior.processing_state.processed_event_ids.extend(["evt:100", "evt:101"])
    persist_runtime_state(path, prior, {})

    # Simulate restart: restore state, then reconnect
    restored, restore_report = restore_runtime_state(path, require_clean=True)
    client = _FakeClientFactory.make([True])
    reconnect = reconnect_mt5(client, settings.recovery)

    assert restore_report.ok
    assert reconnect.ok
    assert restored is not None
    # Prior event IDs must survive restart
    assert "evt:100" in restored.processing_state.processed_event_ids
    assert "evt:101" in restored.processing_state.processed_event_ids


def test_reconnect_with_prior_artifacts_no_new_duplicates(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = build_runtime_state_path(settings)
    prior = fresh_state(settings.backtest.starting_balance_usd, "recovery-drill")
    # prevent_duplicate_processing stores "event:{source_event_id}" — store the prefixed form
    prior.processing_state.processed_event_ids.append("event:evt:prior-001")
    persist_runtime_state(path, prior, {})
    restored, _ = restore_runtime_state(path, require_clean=True)
    assert restored is not None
    # Old event blocked (passing source_event_id generates "event:evt:prior-001")
    from iris_bot.resilient import prevent_duplicate_processing
    assert not prevent_duplicate_processing(restored, "EURUSD", "2026-01-01T00:00:00", "evt:prior-001")
    # New event allowed
    assert prevent_duplicate_processing(restored, "EURUSD", "2026-01-01T00:15:00", "evt:new-002")


# ── Scenario 4: Reconnect + reconcile chain (full sequence) ──────────────────

def test_reconnect_reconcile_chain_all_ok(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = build_runtime_state_path(settings)

    # State with no open positions
    prior = fresh_state(settings.backtest.starting_balance_usd, "recovery-drill")
    persist_runtime_state(path, prior, {})

    # Restart sequence: restore → reconnect → reconcile → operational_status
    restored, restore_report = restore_runtime_state(path, require_clean=True)
    snapshot = BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [], [], [])
    client = _FakeClientFactory.make([True], snapshot)
    reconnect = reconnect_mt5(client, settings.recovery)
    broker = broker_snapshot_from_mt5(client.broker_state_snapshot(()))
    reconcile = reconcile_state(
        restored or fresh_state(settings.backtest.starting_balance_usd, "recovery-drill"),
        broker,
        ReconciliationConfig(policy="log_only", price_tolerance=0.001, volume_tolerance=0.001),
    )
    status = build_operational_status(
        restored or fresh_state(settings.backtest.starting_balance_usd, "recovery-drill"),
        reconcile,
        restore_report,
        [],
    )

    assert restore_report.ok
    assert reconnect.ok
    assert reconcile.ok
    assert status["restore_ok"] is True
    assert status["reconciliation_ok"] is True
    assert status["open_positions"] == 0


def test_reconnect_exhausted_retries_reports_blocked(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = _FakeClientFactory.make([False, False, False])
    report = reconnect_mt5(client, settings.recovery)
    assert report.ok is False
    assert report.final_state == "blocked"
    assert len(report.attempts) == 3


# ── Integrated drill: run_recovery_drills generates full report ───────────────

def test_recovery_drills_all_pass(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    exit_code, report = run_recovery_drills(settings)
    assert exit_code == 0, f"Recovery drills failed: {report.get('failed_scenarios')}"
    assert report["ok"] is True
    assert report["scenarios_total"] == 4
    expected = {
        "clean_disconnect_reconnect",
        "reconnect_broker_state_changed",
        "reconnect_with_prior_artifacts",
        "reconnect_reconcile_chain",
    }
    assert expected.issubset(report["scenarios"].keys())
    # Report is serialisable as audit artifact
    assert json.loads(json.dumps(report))
