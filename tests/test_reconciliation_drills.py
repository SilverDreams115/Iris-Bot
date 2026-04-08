"""BLOQUE 2 — Reconciliation Drills.

Six reconciliation scenarios covering all required desalignment cases.
Each scenario validates detection, classification, and policy action.
Artifacts are auditable (all results serialisable to JSON).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from iris_bot.config import (
    BacktestConfig,
    OperationalConfig,
    RecoveryConfig,
    ReconciliationConfig,
    RiskConfig,
    SessionConfig,
    Settings,
)
from iris_bot.operational import (
    AccountState,
    PaperEngineState,
    PaperPosition,
    SessionStatus,
)
from iris_bot.resilient import (
    BrokerPositionSnapshot,
    BrokerStateSnapshot,
    reconcile_state,
    run_reconciliation_drills,
)


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "data", replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"))
    object.__setattr__(settings, "recovery", RecoveryConfig(reconnect_retries=2, reconnect_backoff_seconds=0.0, require_state_restore_clean=True))
    object.__setattr__(settings, "reconciliation", ReconciliationConfig(policy="hard_fail", price_tolerance=0.0001, volume_tolerance=0.000001))
    object.__setattr__(settings, "backtest", BacktestConfig(use_atr_stops=False, fixed_stop_loss_pct=0.002, fixed_take_profit_pct=0.004, max_holding_bars=5))
    object.__setattr__(settings, "risk", RiskConfig(max_daily_loss_usd=50.0))
    object.__setattr__(settings, "session", SessionConfig(enabled=True, allowed_weekdays=(0, 1, 2, 3, 4), allowed_start_hour_utc=0, allowed_end_hour_utc=23))
    object.__setattr__(settings, "operational", OperationalConfig(persistence_state_filename="runtime_state.json"))
    return settings


def _local_with_open_position(symbol: str = "EURUSD", volume: float = 0.1, price: float = 1.1000) -> PaperEngineState:
    state = PaperEngineState(account_state=AccountState(1000.0, 1000.0, 1000.0))
    state.open_positions[symbol] = PaperPosition(
        symbol=symbol, timeframe="M15", direction=1,
        entry_timestamp="2026-01-01T00:15:00", signal_timestamp="2026-01-01T00:00:00",
        entry_index=1, volume_lots=volume, entry_price=price,
        stop_loss_price=price - 0.002, take_profit_price=price + 0.004,
        commission_entry_usd=1.0, bars_held=1,
        probability_long=0.8, probability_short=0.05,
        stop_policy="static", target_policy="static",
    )
    return state


def _broker_pos(symbol: str, side: str = "buy", volume: float = 0.1, price: float = 1.1000) -> BrokerPositionSnapshot:
    return BrokerPositionSnapshot(
        ticket="t1", symbol=symbol, side=side, volume_lots=volume,
        price_open=price, stop_loss=price - 0.002, take_profit=price + 0.004,
        time="2026-01-01T00:15:00",
    )


cfg = ReconciliationConfig(policy="hard_fail", price_tolerance=0.0001, volume_tolerance=0.000001)


# ── Scenario 1: Local open, broker missing → missing_in_broker (critical) ───

def test_drill_local_open_broker_missing_detected() -> None:
    local = _local_with_open_position()
    broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [], [], [], {})
    result = reconcile_state(local, broker, cfg)
    assert result.ok is False
    assert result.action == "hard_fail"
    cats = [d.category for d in result.discrepancies]
    assert "missing_in_broker" in cats
    # Verify it's classified as critical
    critical = [d for d in result.discrepancies if d.category == "missing_in_broker"]
    assert all(d.severity == "critical" for d in critical)
    # Artifact is serialisable
    assert json.loads(json.dumps(result.to_dict()))


# ── Scenario 2: Broker open, local missing → missing_in_local_state ─────────

def test_drill_broker_open_local_missing_detected() -> None:
    local = PaperEngineState(account_state=AccountState(1000.0, 1000.0, 1000.0))
    broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [_broker_pos("EURUSD")], [], [], {})
    result = reconcile_state(local, broker, cfg)
    assert result.ok is False
    cats = [d.category for d in result.discrepancies]
    assert "missing_in_local_state" in cats
    assert all(d.severity == "critical" for d in result.discrepancies if d.category == "missing_in_local_state")


# ── Scenario 3: Volume divergence → volume_mismatch (critical) ───────────────

def test_drill_volume_divergence_detected() -> None:
    local = _local_with_open_position(volume=0.1)
    broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [_broker_pos("EURUSD", volume=0.5)], [], [], {})
    result = reconcile_state(local, broker, cfg)
    assert result.ok is False
    cats = [d.category for d in result.discrepancies]
    assert "volume_mismatch" in cats
    vol_disc = next(d for d in result.discrepancies if d.category == "volume_mismatch")
    assert vol_disc.severity == "critical"
    assert vol_disc.details["local"] == 0.1
    assert vol_disc.details["broker"] == 0.5


# ── Scenario 4: Price within tolerance → no critical discrepancy ─────────────

def test_drill_price_within_tolerance_no_critical() -> None:
    local = _local_with_open_position(price=1.1000)
    # Price offset of 0.00005 < tolerance 0.0001
    broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [_broker_pos("EURUSD", price=1.10005)], [], [], {})
    result = reconcile_state(local, broker, cfg)
    assert result.ok is True
    critical = [d for d in result.discrepancies if d.severity == "critical"]
    assert not critical


# ── Scenario 5: Price beyond tolerance → price_mismatch (warning) ────────────

def test_drill_price_beyond_tolerance_classified_as_warning() -> None:
    local = _local_with_open_position(price=1.1000)
    # Price offset of 0.005 >> tolerance 0.0001
    broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [_broker_pos("EURUSD", price=1.1050)], [], [], {})
    result = reconcile_state(local, broker, cfg)
    # price_mismatch is severity=warning, not critical → outcome.ok can be True
    price_disc = [d for d in result.discrepancies if d.category == "price_mismatch"]
    assert price_disc, "Expected price_mismatch discrepancy"
    assert all(d.severity == "warning" for d in price_disc)
    # Artifact serialisable
    assert json.loads(json.dumps(result.to_dict()))


# ── Scenario 6: Stale local state detected ───────────────────────────────────

def test_drill_stale_state_detection() -> None:
    local = _local_with_open_position()
    local.current_session_status = SessionStatus("s1", "test", "running", "2026-06-01T12:00:00")
    # Broker closed trades are from long before the local session timestamp
    broker = BrokerStateSnapshot(
        True, 1000.0, 1000.0,
        [_broker_pos("EURUSD")],
        [{"time": "2026-01-01T00:01:00"}],  # older than local timestamp
        [],
        {},
    )
    result = reconcile_state(local, broker, cfg)
    stale_disc = [d for d in result.discrepancies if d.category == "stale_state"]
    assert stale_disc, "Expected stale_state discrepancy"
    assert stale_disc[0].severity == "warning"


# ── Scenario 7: Duplicate local event IDs → critical ────────────────────────

def test_drill_duplicate_event_ids_detected() -> None:
    local = _local_with_open_position()
    local.processing_state.processed_event_ids.extend(["evt:001", "evt:001"])
    broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [_broker_pos("EURUSD")], [], [], {})
    result = reconcile_state(local, broker, cfg)
    dup_disc = [d for d in result.discrepancies if d.category == "duplicate_state"]
    assert dup_disc, "Expected duplicate_state discrepancy"
    assert dup_disc[0].severity == "critical"


# ── Integrated drill: run_reconciliation_drills generates full report ─────────

def test_reconciliation_drills_all_pass(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    exit_code, report = run_reconciliation_drills(settings)
    assert exit_code == 0, f"Drills failed: {report.get('failed_scenarios')}"
    assert report["ok"] is True
    assert report["scenarios_total"] == 6
    assert report["scenarios_passed"] == 6
    # All 6 required scenarios must be present
    expected = {
        "local_open_broker_empty",
        "broker_open_local_empty",
        "volume_divergence",
        "price_within_tolerance",
        "price_beyond_tolerance",
        "stale_state_detection",
    }
    assert expected.issubset(report["scenarios"].keys())
    # Report is serialisable as audit artifact
    assert json.loads(json.dumps(report))


# ── Corrective policy consistency: hard_fail → action must be hard_fail ──────

def test_reconciliation_policy_consistent_with_design() -> None:
    local = _local_with_open_position()
    broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [], [], [], {})
    hard_result = reconcile_state(local, broker, ReconciliationConfig(policy="hard_fail", price_tolerance=0.0001, volume_tolerance=0.000001))
    soft_result = reconcile_state(local, broker, ReconciliationConfig(policy="soft_resync", price_tolerance=0.0001, volume_tolerance=0.000001))
    assert hard_result.action == "hard_fail"
    assert soft_result.action == "soft_resync"
    # Both detect the same discrepancy
    assert any(d.category == "missing_in_broker" for d in hard_result.discrepancies)
    assert any(d.category == "missing_in_broker" for d in soft_result.discrepancies)
