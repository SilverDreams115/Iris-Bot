from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
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
from iris_bot.operational import (
    AccountState,
    AlertRecord,
    BrokerSyncStatus,
    DailyLossTracker,
    PaperEngineState,
    PaperPosition,
    ProcessingState,
    SessionStatus,
)
from iris_bot.processed_dataset import ProcessedRow
from iris_bot.resilient import (
    BrokerStateSnapshot,
    build_runtime_state_path,
    classify_broker_event,
    emit_alert,
    is_session_allowed,
    persist_runtime_state,
    prevent_duplicate_processing,
    reconcile_state,
    reconnect_mt5,
    restore_runtime_state,
    run_operational_status,
    run_reconcile_state,
    run_resilient_session,
    run_restore_state_check,
)


def _row(ts: datetime, symbol: str = "EURUSD", price: float = 1.1) -> ProcessedRow:
    return ProcessedRow(
        timestamp=ts,
        symbol=symbol,
        timeframe="M15",
        open=price,
        high=price + 0.0008,
        low=price - 0.0008,
        close=price,
        volume=100.0,
        label=1,
        label_reason="test",
        horizon_end_timestamp=ts.isoformat(),
        features={"atr_10": 0.0005, "atr_5": 0.0005},
    )


class FakeReference:
    threshold = 0.5
    run_dir = Path("/tmp/fake_experiment")


class FakeClient(MT5Client):
    def __init__(self, should_connect: list[bool], snapshot: BrokerSnapshot | None = None) -> None:
        super().__init__(MT5Config(enabled=True))
        self._should_connect = should_connect
        self._snapshot = snapshot or BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [], [], [])
        self._connect_calls = 0

    def connect(self) -> bool:
        self._connect_calls += 1
        result = self._should_connect[min(self._connect_calls - 1, len(self._should_connect) - 1)]
        self._connected = result
        return result

    def last_error(self) -> object:
        return (1, "Success") if self._connected else (500, "Connect failed")

    def broker_state_snapshot(self, symbols: tuple[str, ...]) -> BrokerSnapshot:
        return self._snapshot

    def dry_run_market_order(self, order):  # type: ignore[override]
        from iris_bot.mt5 import DryRunOrderResult

        return DryRunOrderResult(True, "dry_run_only", order.volume, {"symbol": order.symbol}, [])


def _state_with_position() -> PaperEngineState:
    return PaperEngineState(
        account_state=AccountState(1000.0, 1000.0, 1000.0),
        open_positions={
            "EURUSD": PaperPosition(
                symbol="EURUSD",
                timeframe="M15",
                direction=1,
                entry_timestamp="2026-01-01T00:15:00",
                signal_timestamp="2026-01-01T00:00:00",
                entry_index=1,
                volume_lots=0.1,
                entry_price=1.1000,
                stop_loss_price=1.0980,
                take_profit_price=1.1040,
                commission_entry_usd=1.0,
                bars_held=1,
                probability_long=0.9,
                probability_short=0.05,
                stop_policy="static",
                target_policy="static",
            )
        },
        daily_loss_tracker=DailyLossTracker("2026-01-01", 0.0, 50.0, False),
        current_session_status=SessionStatus("s1", "paper", "running", "2026-01-01T00:15:00"),
        broker_sync_status=BrokerSyncStatus(),
        processing_state=ProcessingState({"EURUSD": "2026-01-01T00:15:00"}, ["e1"]),
    )


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "data", replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"))
    object.__setattr__(settings, "recovery", RecoveryConfig(reconnect_retries=2, reconnect_backoff_seconds=0.0, require_state_restore_clean=True))
    object.__setattr__(settings, "reconciliation", ReconciliationConfig(policy="hard_fail", price_tolerance=0.0001, volume_tolerance=0.000001))
    object.__setattr__(settings, "session", SessionConfig(enabled=True, allowed_weekdays=(0, 1, 2, 3, 4), allowed_start_hour_utc=0, allowed_end_hour_utc=23))
    object.__setattr__(settings, "operational", OperationalConfig(repeated_rejection_alert_threshold=1, persistence_state_filename="runtime_state.json"))
    object.__setattr__(settings, "backtest", BacktestConfig(use_atr_stops=False, fixed_stop_loss_pct=0.002, fixed_take_profit_pct=0.004, max_holding_bars=5))
    object.__setattr__(settings, "risk", RiskConfig(max_daily_loss_usd=50.0))
    return settings


def test_broker_local_state_mismatch_detected() -> None:
    outcome = reconcile_state(
        _state_with_position(),
        BrokerStateSnapshot(True, 1000.0, 1000.0, [], [], [], {}),
        ReconciliationConfig(policy="hard_fail", price_tolerance=0.0001, volume_tolerance=0.000001),
    )
    assert outcome.ok is False
    assert any(item.category == "missing_in_broker" for item in outcome.discrepancies)


def test_soft_and_hard_reconciliation_paths() -> None:
    broker = BrokerStateSnapshot(True, 900.0, 900.0, [], [], [], {})
    soft = reconcile_state(_state_with_position(), broker, ReconciliationConfig(policy="soft_resync", price_tolerance=0.0001, volume_tolerance=0.000001))
    hard = reconcile_state(_state_with_position(), broker, ReconciliationConfig(policy="hard_fail", price_tolerance=0.0001, volume_tolerance=0.000001))
    assert soft.action == "soft_resync"
    assert hard.action == "hard_fail"


def test_reconnect_success_and_failure() -> None:
    success = reconnect_mt5(FakeClient([False, True]), RecoveryConfig(reconnect_retries=2, reconnect_backoff_seconds=0.0, require_state_restore_clean=True))
    failure = reconnect_mt5(FakeClient([False, False]), RecoveryConfig(reconnect_retries=2, reconnect_backoff_seconds=0.0, require_state_restore_clean=True))
    assert success.ok is True
    assert failure.ok is False


def test_reconnect_multiple_failures_then_success() -> None:
    """Succeeds on the 3rd attempt (retries=3); report should record all attempt details."""
    config = RecoveryConfig(reconnect_retries=3, reconnect_backoff_seconds=0.0, require_state_restore_clean=True)
    report = reconnect_mt5(FakeClient([False, False, True]), config)
    assert report.ok is True
    assert len(report.attempts) == 3
    assert report.attempts[0]["ok"] is False
    assert report.attempts[1]["ok"] is False
    assert report.attempts[2]["ok"] is True


def test_reconnect_exhausted_retries_with_intermittent_pattern() -> None:
    """All attempts fail even when pattern varies; blocked status is returned."""
    config = RecoveryConfig(reconnect_retries=4, reconnect_backoff_seconds=0.0, require_state_restore_clean=True)
    report = reconnect_mt5(FakeClient([False, False, False, False]), config)
    assert report.ok is False
    assert report.final_state == "blocked"
    assert len(report.attempts) == 4


def test_state_restore_success_and_corruption_block(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime.json"
    persist_runtime_state(state_path, _state_with_position(), {})
    restored, report = restore_runtime_state(state_path, True)
    assert restored is not None
    assert report.ok is True

    state_path.write_text("{broken", encoding="utf-8")
    restored_bad, report_bad = restore_runtime_state(state_path, True)
    assert restored_bad is None
    assert report_bad.ok is False
    assert report_bad.action == "blocked"


def test_partial_fill_and_rejection_handling() -> None:
    partial = classify_broker_event({"reason": "partial fill"})
    rejection = classify_broker_event({"accepted": False, "reason": "not enough money"})
    assert partial.classification == "partial_fill"
    assert rejection.classification == "not_enough_money"
    assert rejection.block_operation is True


def test_duplicate_event_prevention_and_idempotent_restart_behavior() -> None:
    state = _state_with_position()
    assert prevent_duplicate_processing(state, "EURUSD", "2026-01-01T00:30:00") is True
    assert prevent_duplicate_processing(state, "EURUSD", "2026-01-01T00:30:00") is False


def test_market_session_blocked() -> None:
    allowed, reason = is_session_allowed(
        datetime(2026, 1, 3, 12, 0, 0),
        SessionConfig(enabled=True, allowed_weekdays=(0, 1, 2, 3, 4), allowed_start_hour_utc=0, allowed_end_hour_utc=23),
    )
    assert allowed is False
    assert reason == "market_session_blocked_weekday"


def test_max_daily_loss_persists_across_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = _state_with_position()
    state.daily_loss_tracker.blocked = True
    state.daily_loss_tracker.realized_pnl_usd = -55.0
    persist_runtime_state(build_runtime_state_path(settings), state, {})
    restored, _ = restore_runtime_state(build_runtime_state_path(settings), True)
    assert restored is not None
    assert restored.daily_loss_tracker.blocked is True


def test_alert_emission() -> None:
    alerts: list[AlertRecord] = []
    emit_alert(alerts, "critical", "disconnect", "Disconnected", {"attempt": 1})
    assert alerts[0].category == "disconnect"


def test_operational_status_report_and_restore_check(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    persist_runtime_state(build_runtime_state_path(settings), _state_with_position(), {})
    status_code, _ = run_operational_status(settings)
    restore_code, _ = run_restore_state_check(settings)
    assert status_code == 0
    assert restore_code == 0


def test_safe_blocking_on_critical_inconsistency(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    persist_runtime_state(build_runtime_state_path(settings), _state_with_position(), {})
    rows = [_row(datetime(2026, 1, 1, 0, 0, 0)), _row(datetime(2026, 1, 1, 0, 15, 0))]
    probs = [{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.2, 0: 0.6, -1: 0.2}]
    monkeypatch.setattr("iris_bot.resilient.load_paper_context", lambda settings: (FakeReference(), rows, probs))
    snapshot = BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [], [], [])
    code, run_dir = run_resilient_session(settings, "demo_dry", True, client_factory=lambda: FakeClient([True], snapshot))
    assert code == 3
    payload = json.loads((run_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False


def test_reconcile_state_command(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    persist_runtime_state(build_runtime_state_path(settings), _state_with_position(), {})
    snapshot = BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [], [], [])
    code, _ = run_reconcile_state(settings, client_factory=lambda: FakeClient([True], snapshot))
    assert code == 3


def test_resilient_run_generates_artifacts(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    rows = [
        _row(datetime(2026, 1, 1, 0, 0, 0)),
        _row(datetime(2026, 1, 1, 0, 15, 0)),
        _row(datetime(2026, 1, 1, 0, 30, 0)),
    ]
    probs = [{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.2, 0: 0.6, -1: 0.2}, {1: 0.2, 0: 0.6, -1: 0.2}]
    monkeypatch.setattr("iris_bot.resilient.load_paper_context", lambda settings: (FakeReference(), rows, probs))
    snapshot = BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [], [], [])
    code, run_dir = run_resilient_session(settings, "demo_dry", True, client_factory=lambda: FakeClient([True], snapshot))
    assert code == 0
    assert (run_dir / "reconciliation_report.json").exists()
    assert (run_dir / "restore_state_report.json").exists()
    assert (run_dir / "operational_status.json").exists()
    assert (run_dir / "alerts_log.jsonl").exists()
