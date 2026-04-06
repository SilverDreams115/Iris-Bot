import json
from collections import namedtuple
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from iris_bot.artifacts import read_artifact_payload, wrap_artifact
from iris_bot.backtest import run_backtest_engine
from iris_bot.config import MT5Config, load_settings
from iris_bot.corrective import run_corrective_audit
from iris_bot.main import COMMAND_HANDLERS
from iris_bot.mt5 import MT5Client
from iris_bot.processed_dataset import build_processed_dataset, write_processed_dataset
from iris_bot.operational import AccountState, PaperEngineState, PaperPosition
from iris_bot.resilient import (
    BrokerPositionSnapshot,
    BrokerStateSnapshot,
    build_processing_event_id,
    prevent_duplicate_processing,
    reconcile_state,
)
from iris_bot.sessions import canonical_session_name, session_definition_report, session_flags
from iris_bot.symbol_validation import run_strategy_validation
from iris_bot.data import Bar, write_bars


pytest.importorskip("xgboost")


FakeRate = namedtuple("FakeRate", ["time", "open", "high", "low", "close", "tick_volume", "spread"])


class ConnectMT5:
    TIMEFRAME_M5 = 5
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_FOK = 2
    ORDER_FILLING_RETURN = 4
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 10
    ORDER_TIME_GTC = 20

    def __init__(
        self,
        *,
        initialize_ok: bool = True,
        login_ok: bool = True,
        auth_initialize_ok: bool | None = None,
    ) -> None:
        self.initialize_ok = initialize_ok
        self.login_ok = login_ok
        self.auth_initialize_ok = auth_initialize_ok
        self.shutdown_called = False
        self.login_calls = 0
        self.initialize_calls: list[dict[str, object]] = []

    def initialize(self, **kwargs: object) -> bool:
        self.initialize_calls.append(kwargs)
        if "login" in kwargs and self.auth_initialize_ok is not None:
            return self.auth_initialize_ok
        return self.initialize_ok

    def login(self, *_: object, **__: object) -> bool:
        self.login_calls += 1
        return self.login_ok

    def shutdown(self) -> None:
        self.shutdown_called = True

    def terminal_info(self):
        auth_connected = self.auth_initialize_ok is True
        return {"connected": True} if (self.initialize_ok and self.login_ok) or auth_connected else None


def _bars(symbol: str, count: int) -> list[Bar]:
    start = datetime(2026, 1, 1, 0, 0, 0)
    price = 1.1000
    bars: list[Bar] = []
    for index in range(count):
        price += 0.0002 if index % 5 in (0, 1, 2) else -0.00015
        bars.append(
            Bar(
                timestamp=start + timedelta(minutes=15 * index),
                symbol=symbol,
                timeframe="M15",
                open=price - 0.0001,
                high=price + 0.0005,
                low=price - 0.0005,
                close=price,
                volume=100 + index % 10,
                spread=10.0,
            )
        )
    return bars


def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IRIS_PRIMARY_TIMEFRAME", "M15")
    monkeypatch.setenv("IRIS_WF_TRAIN_WINDOW", "40")
    monkeypatch.setenv("IRIS_WF_VALIDATION_WINDOW", "15")
    monkeypatch.setenv("IRIS_WF_TEST_WINDOW", "15")
    monkeypatch.setenv("IRIS_WF_STEP", "15")
    monkeypatch.setenv("IRIS_XGB_NUM_BOOST_ROUND", "8")
    monkeypatch.setenv("IRIS_XGB_EARLY_STOPPING_ROUNDS", "3")
    settings = load_settings()
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    runs_dir = tmp_path / "runs"
    runtime_dir = tmp_path / "data" / "runtime"
    for path in (raw_dir, processed_dir, runs_dir, runtime_dir):
        path.mkdir(parents=True, exist_ok=True)
    object.__setattr__(settings.data, "raw_dir", raw_dir)
    object.__setattr__(settings.data, "processed_dir", processed_dir)
    object.__setattr__(settings.data, "runs_dir", runs_dir)
    object.__setattr__(settings.data, "runtime_dir", runtime_dir)
    object.__setattr__(settings.experiment, "_processed_dir", processed_dir)
    return settings


def _write_dataset(settings, bars: list[Bar]) -> None:
    write_bars(settings.data.raw_dataset_path, bars)
    processed = build_processed_dataset(bars, settings.labeling)
    write_processed_dataset(
        processed,
        settings.experiment.processed_dataset_path,
        settings.experiment.processed_manifest_path,
        settings.experiment.processed_schema_path,
    )


def test_connect_initialize_ok_login_fail_is_not_connected() -> None:
    client = MT5Client(
        replace(MT5Config(), enabled=True, login=123456, password="secret", server="MetaQuotes-Demo"),
        mt5_module=ConnectMT5(initialize_ok=True, login_ok=False, auth_initialize_ok=False),
    )
    assert client.connect() is False
    assert client.is_connected() is False


def test_connect_initialize_fail_is_not_connected() -> None:
    client = MT5Client(
        replace(MT5Config(), enabled=True, login=123456, password="secret", server="MetaQuotes-Demo"),
        mt5_module=ConnectMT5(initialize_ok=False, login_ok=True, auth_initialize_ok=False),
    )
    assert client.connect() is False
    assert client.is_connected() is False


def test_connect_initialize_ok_login_ok_is_connected() -> None:
    mt5 = ConnectMT5(
        initialize_ok=True,
        login_ok=True,
        auth_initialize_ok=True,
    )
    client = MT5Client(
        replace(MT5Config(), enabled=True, login=123456, password="secret", server="MetaQuotes-Demo"),
        mt5_module=mt5,
    )
    assert client.connect() is True
    assert client.is_connected() is True
    assert mt5.login_calls == 0


def test_connect_reconnect_succeeds_after_initial_login_failure() -> None:
    mt5 = ConnectMT5(initialize_ok=True, login_ok=False, auth_initialize_ok=False)
    client = MT5Client(
        replace(MT5Config(), enabled=True, login=123456, password="secret", server="MetaQuotes-Demo"),
        mt5_module=mt5,
    )
    assert client.connect() is False
    assert client.is_connected() is False
    mt5.login_ok = True
    assert client.connect() is True
    assert client.is_connected() is True


def test_connect_authenticated_initialize_success_skips_separate_login() -> None:
    mt5 = ConnectMT5(initialize_ok=True, login_ok=False, auth_initialize_ok=True)
    client = MT5Client(
        replace(MT5Config(), enabled=True, login=123456, password="secret", server="MetaQuotes-Demo", path="C:/mt5/terminal64.exe"),
        mt5_module=mt5,
    )

    assert client.connect() is True
    assert mt5.initialize_calls[0]["login"] == 123456
    assert mt5.initialize_calls[0]["server"] == "MetaQuotes-Demo"
    assert mt5.login_calls == 0


def test_processed_event_ids_used_for_real_event_id_mode() -> None:
    state = type("Obj", (), {"processing_state": type("P", (), {"last_processed_timestamp_by_symbol": {}, "processed_event_ids": [], "idempotency_mode_counts": {}})()})()
    assert prevent_duplicate_processing(state, "EURUSD", "2026-01-01T00:00:00", source_event_id="abc123") is True
    assert prevent_duplicate_processing(state, "EURUSD", "2026-01-01T00:00:01", source_event_id="abc123") is False
    assert state.processing_state.idempotency_mode_counts["event_id"] == 1


def test_processed_event_ids_fallback_is_explicit() -> None:
    event_id, mode = build_processing_event_id("EURUSD", "2026-01-01T00:00:00", None)
    assert mode == "timestamp_fallback"
    assert event_id.startswith("fallback:")


def test_session_taxonomy_is_consistent() -> None:
    assert canonical_session_name(datetime(2026, 1, 1, 2, 0, 0)) == "asia"
    assert canonical_session_name(datetime(2026, 1, 1, 9, 0, 0)) == "london"
    assert canonical_session_name(datetime(2026, 1, 1, 15, 0, 0)) == "new_york"
    assert session_flags(datetime(2026, 1, 1, 15, 0, 0)) == (0.0, 0.0, 1.0)
    assert session_definition_report()["definitions"][0]["name"] == "asia"


def test_backtest_optimized_path_preserves_results() -> None:
    processed = build_processed_dataset(_bars("EURUSD", 80), load_settings().labeling)
    rows = processed.rows[:30]
    probabilities = [{1: 0.85, 0: 0.10, -1: 0.05} for _ in rows]
    metrics_a, trades_a, curve_a = run_backtest_engine(rows, probabilities, 0.6, load_settings().backtest, load_settings().risk)
    metrics_b, trades_b, curve_b = run_backtest_engine(rows, probabilities, 0.6, load_settings().backtest, load_settings().risk)
    assert metrics_a == metrics_b
    assert len(trades_a) == len(trades_b)
    assert len(curve_a) == len(curve_b)


def test_cli_registry_is_modular_and_includes_corrective_command() -> None:
    assert "run-corrective-audit" in COMMAND_HANDLERS
    assert "fetch-historical" in COMMAND_HANDLERS
    assert "run-symbol-research" in COMMAND_HANDLERS


def test_artifact_schema_roundtrip() -> None:
    payload = wrap_artifact("symbol_enablement", {"symbols": {"EURUSD": {"state": "enabled"}}})
    path = Path("/tmp/test_artifact_schema_roundtrip.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        loaded = read_artifact_payload(path, expected_type="symbol_enablement")
        assert loaded["symbols"]["EURUSD"]["state"] == "enabled"
    finally:
        if path.exists():
            path.unlink()


def test_reconciliation_filters_foreign_broker_positions() -> None:
    local_state = PaperEngineState(account_state=AccountState(balance_usd=1000.0, cash_usd=1000.0, equity_usd=1000.0))
    broker_state = BrokerStateSnapshot(
        connected=True,
        balance_usd=1000.0,
        equity_usd=1000.0,
        positions=[],
        closed_trades=[],
        pending_orders=[],
        raw_account={},
        scope_report={"ignored_positions": 2, "ownership_filter_active": True},
    )
    outcome = reconcile_state(local_state, broker_state, load_settings().reconciliation)
    assert outcome.ok is True
    assert not any(item.category == "missing_in_local_state" for item in outcome.discrepancies)


def test_reconciliation_real_bot_position_mismatch_stays_critical() -> None:
    state = PaperEngineState(account_state=AccountState(balance_usd=1000.0, cash_usd=1000.0, equity_usd=1000.0))
    state.open_positions["EURUSD"] = PaperPosition(
        symbol="EURUSD",
        timeframe="M15",
        direction=1,
        entry_timestamp="2026-01-01T00:00:00",
        signal_timestamp="2026-01-01T00:00:00",
        entry_index=0,
        volume_lots=0.10,
        entry_price=1.1000,
        stop_loss_price=1.0950,
        take_profit_price=1.1100,
        commission_entry_usd=0.35,
        bars_held=0,
        probability_long=0.8,
        probability_short=0.1,
        stop_policy="static",
        target_policy="static",
    )
    broker_state = BrokerStateSnapshot(
        connected=True,
        balance_usd=1000.0,
        equity_usd=1000.0,
        positions=[
            BrokerPositionSnapshot(
                ticket="1",
                symbol="EURUSD",
                side="sell",
                volume_lots=0.10,
                price_open=1.1000,
                stop_loss=1.1050,
                take_profit=1.0900,
                time="2026-01-01T00:00:00",
            )
        ],
        closed_trades=[],
        pending_orders=[],
        raw_account={},
        scope_report={"ignored_positions": 0, "ownership_filter_active": True},
    )
    outcome = reconcile_state(state, broker_state, load_settings().reconciliation)
    assert outcome.ok is False
    assert any(item.category == "side_mismatch" and item.severity == "critical" for item in outcome.discrepancies)


def test_leakage_fix_report_marks_test_as_unused_for_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _write_dataset(settings, _bars("EURUSD", 160))
    assert run_strategy_validation(settings) == 0
    report = list(settings.data.runs_dir.glob("*_strategy_validation/leakage_fix_report.json"))[0]
    payload = read_artifact_payload(report, expected_type="strategy_validation")
    assert payload["test_used_for_selection"] is False


def test_corrective_audit_generates_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _write_dataset(settings, _bars("EURUSD", 160))
    assert run_strategy_validation(settings) == 0
    assert run_corrective_audit(settings) == 0
    assert list(settings.data.runs_dir.glob("*_corrective_audit/corrective_audit_report.json"))
