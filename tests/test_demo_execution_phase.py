from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from datetime import datetime, UTC

from iris_bot.artifacts import wrap_artifact
from iris_bot.config import load_settings
from iris_bot.demo_execution import (
    _classify_order_result,
    demo_execution_preflight_payload,
    run_demo_execution_command,
)
from iris_bot.demo_execution_registry import save_demo_execution_registry
from iris_bot.exits import SymbolExitProfile, build_exit_policies
from iris_bot.model_artifacts import (
    build_model_artifact_manifest,
    validate_model_artifact,
    write_model_artifact_manifest,
)
from iris_bot.mt5 import MT5Config, OrderResult
from iris_bot.processed_dataset import ProcessedRow
from iris_bot.structural_rework import _decision_for_symbol
from iris_bot.symbols import write_symbol_strategy_profiles
from iris_bot.profile_registry import save_strategy_profile_registry
from iris_bot.xgb_model import XGBoostMultiClassModel


pytest.importorskip("xgboost")


def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IRIS_PRIMARY_TIMEFRAME", "M15")
    monkeypatch.setenv("IRIS_DEMO_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("IRIS_DEMO_EXECUTION_TARGET_SYMBOL", "EURUSD")
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
    object.__setattr__(settings.mt5, "enabled", True)
    return settings


def _write_approved_demo_profile(settings) -> None:
    payload = {
        "profiles": {
            "EURUSD": [
                {
                    "artifact_type": "strategy_profile",
                    "schema_version": 2,
                    "symbol": "EURUSD",
                    "profile_id": "EURUSD-approved",
                    "created_at": "2026-04-06T00:00:00+00:00",
                    "source_run_id": "20260406T000000Z_strategy_validation",
                    "model_variant": "global_model",
                    "enablement_state": "enabled",
                    "promotion_state": "approved_demo",
                    "promotion_reason": "test_fixture",
                    "rollback_target": None,
                    "checksum": "",
                    "profile_payload": {
                        "symbol": "EURUSD",
                        "enabled_state": "enabled",
                        "allowed_timeframes": ["M15"],
                        "allowed_sessions": ["asia", "london", "new_york"],
                        "threshold": 0.45,
                        "allow_long": True,
                        "allow_short": True,
                        "risk_multiplier": 1.0,
                        "max_open_positions": 4,
                        "stop_policy": "static",
                        "target_policy": "static",
                        "stop_atr_multiplier": 1.5,
                        "target_atr_multiplier": 3.0,
                        "stop_min_pct": 0.001,
                        "stop_max_pct": 0.01,
                        "target_min_pct": 0.0015,
                        "target_max_pct": 0.02,
                        "no_trade_min_expectancy_usd": 0.0,
                        "regime_filter": "off",
                        "notes": "test_fixture",
                        "profile_id": "EURUSD-approved",
                        "model_variant": "global_model",
                        "source_run_id": "20260406T000000Z_strategy_validation",
                        "promotion_state": "approved_demo",
                        "promotion_reason": "test_fixture",
                        "rollback_target": None,
                    },
                }
            ]
        },
        "active_profiles": {"EURUSD": "EURUSD-approved"},
    }
    from iris_bot.profile_registry import _profile_checksum

    payload["profiles"]["EURUSD"][0]["checksum"] = _profile_checksum(payload["profiles"]["EURUSD"][0]["profile_payload"])
    save_strategy_profile_registry(settings, payload)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": payload["profiles"]["EURUSD"][0]["profile_payload"]})
    lifecycle_dir = settings.data.runs_dir / "20260406T000000Z_lifecycle_reconciliation"
    lifecycle_dir.mkdir(parents=True, exist_ok=True)
    (lifecycle_dir / "lifecycle_reconciliation_report.json").write_text(
        json.dumps(
            wrap_artifact(
                "lifecycle_reconciliation",
                {"symbols": {"EURUSD": {"critical_mismatch_count": 0}}},
            )
        ),
        encoding="utf-8",
    )
    endurance_dir = settings.data.runs_dir / "20260406T000000Z_symbol_endurance"
    endurance_dir.mkdir(parents=True, exist_ok=True)
    (endurance_dir / "symbol_stability_report.json").write_text(
        json.dumps(
            wrap_artifact(
                "symbol_stability",
                {"symbols": {"EURUSD": {"decision": "go", "cycles_completed": 3}}},
            )
        ),
        encoding="utf-8",
    )


def _write_model_manifest(settings) -> Path:
    artifact_dir = settings.data.runtime_dir / "demo_execution_models" / "EURUSD"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = XGBoostMultiClassModel(settings.xgboost)
    train_rows = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [-0.8, -0.9], [0.1, 0.1], [0.9, 0.8]]
    train_labels = [-1, 0, 1, -1, 0, 1]
    validation_rows = [[-0.7, -0.7], [0.0, 0.1], [0.8, 0.7]]
    validation_labels = [-1, 0, 1]
    model.fit(train_rows, train_labels, validation_rows, validation_labels, feature_names=["f1", "f2"])
    model_path = artifact_dir / "xgboost_model.json"
    metadata_path = artifact_dir / "xgboost_metadata.json"
    model.save(model_path, metadata_path, ["f1", "f2"])
    manifest = build_model_artifact_manifest(
        settings=settings,
        symbol="EURUSD",
        model_path=model_path,
        metadata_path=metadata_path,
        feature_names=["f1", "f2"],
        threshold=0.45,
        threshold_metric="macro_f1",
        threshold_value=0.5,
        model_variant="symbol_specific",
        source_run_dir="runs/test",
        base_profile_snapshot={
            "profile_id": "EURUSD-approved",
            "promotion_state": "approved_demo",
            "stop_policy": "static",
            "target_policy": "static",
        },
        evaluation_summary={"test": {"net_pnl_usd": 5.0}},
    )
    return write_model_artifact_manifest(artifact_dir / "model_artifact_manifest.json", manifest)


def _write_demo_registry(settings, manifest_path: Path, *, active: bool = True, approved: bool = True) -> None:
    save_demo_execution_registry(
        settings,
        {
            "symbols": {
                "EURUSD": {
                    "symbol": "EURUSD",
                    "decision": "APPROVED_FOR_DEMO_EXECUTION" if approved else "REJECT_FOR_DEMO_EXECUTION",
                    "approved_for_demo_execution": approved,
                    "active_for_demo_execution": active,
                    "base_profile_id": "EURUSD-approved",
                    "base_promotion_state": "approved_demo",
                    "model_variant": "symbol_specific",
                    "threshold": 0.45,
                    "stop_policy": "static",
                    "target_policy": "static",
                    "model_artifact_manifest_path": str(manifest_path),
                    "reasons": [],
                }
            },
            "active_symbol": "EURUSD" if active else "",
            "gate_open": active,
        },
    )


class FakeDemoClient:
    def __init__(self, config: MT5Config) -> None:
        self.config = config
        self.sent_orders: list[dict[str, object]] = []
        self.closed_orders: list[dict[str, object]] = []
        self._position_open = False
        self._mt5 = SimpleNamespace(symbol_info_tick=lambda symbol: SimpleNamespace(ask=1.1002, bid=1.1000))

    def connect(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def account_info(self) -> dict[str, object]:
        return {"server": "MetaQuotes-Demo", "company": "MetaQuotes Demo", "name": "Demo Account"}

    def check(self, symbols: tuple[str, ...]):
        return SimpleNamespace(connected=True, symbols={symbols[0]: {"issues": [], "symbol_info": {"volume_min": 0.01}}})

    def fetch_historical_bars(self, symbol: str, timeframe: str, count: int):
        from datetime import datetime, timedelta
        from iris_bot.data import Bar

        bars = []
        start = datetime(2026, 1, 1, 0, 0, 0)
        price = 1.10 if symbol == "EURUSD" else 1.20
        for index in range(60):
            price += 0.0002 if index % 4 else -0.00005
            bars.append(
                Bar(
                    timestamp=start + timedelta(minutes=15 * index),
                    symbol=symbol,
                    timeframe="M15",
                    open=price - 0.0001,
                    high=price + 0.0003,
                    low=price - 0.0003,
                    close=price,
                    volume=100 + index,
                )
            )
        return bars

    def broker_state_snapshot(self, symbols: tuple[str, ...]):
        positions = []
        if self._position_open:
            positions.append({"symbol": "EURUSD", "ticket": 123456, "volume": 0.01, "type": 0})
        return SimpleNamespace(to_dict=lambda: {"positions": positions}, positions=positions)

    def send_market_order(self, order):
        self.sent_orders.append(asdict(order))
        self._position_open = True
        return OrderResult(True, 10009, "done", 123456, order.volume, 1.1002, {"symbol": order.symbol, "type": 0})

    def close_position(self, ticket: int, symbol: str, volume: float, side: str):
        self.closed_orders.append({"ticket": ticket, "symbol": symbol, "volume": volume, "side": side})
        self._position_open = False
        return OrderResult(True, 10009, "done", 123457, volume, 1.1000, {"symbol": symbol, "type": 1})


def test_structural_decision_is_conservative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    decision, reasons = _decision_for_symbol(
        settings,
        "EURUSD",
        {"trade_count": 12, "expectancy_usd": 1.0, "profit_factor": 1.2, "max_drawdown_usd": 20.0, "no_trade_ratio": 0.2},
        {"valid_folds": 3, "positive_folds": 1, "aggregate": {"total_net_pnl_usd": -5.0, "mean_expectancy_usd": -0.1, "mean_profit_factor": 0.9, "mean_no_trade_ratio": 0.2, "worst_fold_drawdown_usd": 20.0}},
    )
    assert decision in {"REJECT_FOR_DEMO_EXECUTION", "CANDIDATE_FOR_DEMO_EXECUTION"}
    assert reasons


def test_validate_model_artifact_succeeds_and_blocks_checksum_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    ok_report = validate_model_artifact(settings, symbol="EURUSD", manifest_path=manifest_path)
    assert ok_report["ok"] is True

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["payload"]["model_sha256"] = "bad"
    manifest_path.write_text(json.dumps(wrap_artifact("model_artifact_manifest", payload["payload"])), encoding="utf-8")
    bad_report = validate_model_artifact(settings, symbol="EURUSD", manifest_path=manifest_path)
    assert bad_report["ok"] is False
    assert "model_checksum_mismatch" in bad_report["reasons"]


def test_validate_model_artifact_blocks_invalid_active_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    manifest_path = _write_model_manifest(settings)
    report = validate_model_artifact(settings, symbol="EURUSD", manifest_path=manifest_path)
    assert report["ok"] is False
    assert "active_profile_invalid" in report["reasons"]


def test_demo_execution_preflight_blocks_when_final_gate_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    _write_demo_registry(settings, manifest_path, active=False, approved=True)

    payload = demo_execution_preflight_payload(settings, client=FakeDemoClient(settings.mt5))

    assert payload["ok"] is False
    assert payload["checks"]["registry_gate"]["ok"] is False


def test_broker_result_classification() -> None:
    assert _classify_order_result(OrderResult(True, 10009, "done", 1, 0.01, 1.1, {})) == "filled"
    assert _classify_order_result(OrderResult(False, 10006, "reject", None, None, None, {})) == "rejected"
    assert _classify_order_result(OrderResult(False, 10004, "requote", None, None, None, {})) == "requote"


def test_run_demo_execution_sends_order_and_writes_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward series default: auto_close_after_entry=False — position NOT closed by bot."""
    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    _write_demo_registry(settings, manifest_path, active=True, approved=True)

    fake_client = FakeDemoClient(settings.mt5)

    class FakeModel:
        def predict_probabilities(self, rows):
            return [{-1: 0.1, 0: 0.2, 1: 0.7}]

    monkeypatch.setattr("iris_bot.demo_execution.MT5Client", lambda cfg: fake_client)
    monkeypatch.setattr("iris_bot.demo_execution.load_validated_model", lambda settings, symbol, manifest_path: (FakeModel(), {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []}))
    monkeypatch.setattr("iris_bot.demo_execution.validate_model_artifact", lambda settings, symbol, manifest_path: {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []})
    monkeypatch.setattr("iris_bot.demo_execution._latest_row_features", lambda settings, client, symbol, feature_names: ({"f1": 1.0, "f2": 1.0}, {"timestamp": "2026-04-06T00:00:00", "symbol": symbol, "timeframe": "M15"}))

    exit_code = run_demo_execution_command(settings)

    assert exit_code == 0
    assert fake_client.sent_orders
    # With auto_close_after_entry=False (default for forward series), bot must NOT close the position.
    assert not fake_client.closed_orders
    run_dir = sorted(settings.data.runs_dir.glob("*_run_demo_execution"))[-1]
    assert (run_dir / "demo_execution_report.json").exists()
    assert (run_dir / "broker_order_trace.json").exists()
    assert (run_dir / "post_trade_reconciliation_report.json").exists()
    trace = json.loads((run_dir / "broker_order_trace.json").read_text(encoding="utf-8"))
    assert trace["payload"]["request"]["sl"] != trace["payload"]["request"]["tp"]


def test_run_demo_execution_auto_close_when_explicitly_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit probe mode: auto_close_after_entry=True closes position immediately (round-trip verification)."""
    monkeypatch.setenv("IRIS_DEMO_EXECUTION_AUTO_CLOSE_AFTER_ENTRY", "true")
    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    _write_demo_registry(settings, manifest_path, active=True, approved=True)

    fake_client = FakeDemoClient(settings.mt5)

    class FakeModel:
        def predict_probabilities(self, rows):
            return [{-1: 0.1, 0: 0.2, 1: 0.7}]

    monkeypatch.setattr("iris_bot.demo_execution.MT5Client", lambda cfg: fake_client)
    monkeypatch.setattr("iris_bot.demo_execution.load_validated_model", lambda settings, symbol, manifest_path: (FakeModel(), {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []}))
    monkeypatch.setattr("iris_bot.demo_execution.validate_model_artifact", lambda settings, symbol, manifest_path: {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []})
    monkeypatch.setattr("iris_bot.demo_execution._latest_row_features", lambda settings, client, symbol, feature_names: ({"f1": 1.0, "f2": 1.0}, {"timestamp": "2026-04-06T00:00:00", "symbol": symbol, "timeframe": "M15"}))

    exit_code = run_demo_execution_command(settings)

    assert exit_code == 0
    assert fake_client.sent_orders
    # With auto_close_after_entry=True, bot closes immediately (probe/round-trip verification mode).
    assert fake_client.closed_orders


def test_run_demo_execution_does_not_send_when_gate_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    _write_demo_registry(settings, manifest_path, active=False, approved=True)
    fake_client = FakeDemoClient(settings.mt5)
    monkeypatch.setattr("iris_bot.demo_execution.MT5Client", lambda cfg: fake_client)

    exit_code = run_demo_execution_command(settings)

    assert exit_code == 2
    assert fake_client.sent_orders == []


def test_run_demo_execution_blocked_when_position_already_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate: position_already_open blocks order — one_position_per_symbol contract enforced."""
    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    _write_demo_registry(settings, manifest_path, active=True, approved=True)

    fake_client = FakeDemoClient(settings.mt5)
    fake_client._position_open = True  # simulate pre-existing EURUSD position on broker

    class FakeModel:
        def predict_probabilities(self, rows):
            return [{-1: 0.1, 0: 0.2, 1: 0.7}]

    monkeypatch.setattr("iris_bot.demo_execution.MT5Client", lambda cfg: fake_client)
    monkeypatch.setattr("iris_bot.demo_execution.load_validated_model", lambda settings, symbol, manifest_path: (FakeModel(), {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []}))
    monkeypatch.setattr("iris_bot.demo_execution.validate_model_artifact", lambda settings, symbol, manifest_path: {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []})
    monkeypatch.setattr("iris_bot.demo_execution._latest_row_features", lambda settings, client, symbol, feature_names: ({"f1": 1.0, "f2": 1.0}, {"timestamp": "2026-04-06T00:00:00", "symbol": symbol, "timeframe": "M15"}))

    exit_code = run_demo_execution_command(settings)

    assert exit_code == 5  # 5 = position_already_open (distinct from 3 = no_trade_signal)
    assert fake_client.sent_orders == []  # order must NOT be sent
    run_dir = sorted(settings.data.runs_dir.glob("*_run_demo_execution"))[-1]
    report_path = run_dir / "position_already_open_report.json"
    assert report_path.exists(), "auditable block report must be written"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["payload"]["reason"] == "position_already_open"
    assert report["payload"]["existing_ticket"] == 123456
    assert report["payload"]["one_position_per_symbol"] is True


def test_run_demo_execution_one_position_gate_bypassed_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When one_position_per_symbol=False the gate must not block even with an existing position."""
    monkeypatch.setenv("IRIS_ONE_POSITION_PER_SYMBOL", "false")
    settings = _settings(tmp_path, monkeypatch)
    # Patch the trading config directly since there may not be an env var loader for this field.
    object.__setattr__(settings.trading, "one_position_per_symbol", False)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    _write_demo_registry(settings, manifest_path, active=True, approved=True)

    fake_client = FakeDemoClient(settings.mt5)
    fake_client._position_open = True  # pre-existing position — gate should be skipped

    class FakeModel:
        def predict_probabilities(self, rows):
            return [{-1: 0.1, 0: 0.2, 1: 0.7}]

    monkeypatch.setattr("iris_bot.demo_execution.MT5Client", lambda cfg: fake_client)
    monkeypatch.setattr("iris_bot.demo_execution.load_validated_model", lambda settings, symbol, manifest_path: (FakeModel(), {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []}))
    monkeypatch.setattr("iris_bot.demo_execution.validate_model_artifact", lambda settings, symbol, manifest_path: {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []})
    monkeypatch.setattr("iris_bot.demo_execution._latest_row_features", lambda settings, client, symbol, feature_names: ({"f1": 1.0, "f2": 1.0}, {"timestamp": "2026-04-06T00:00:00", "symbol": symbol, "timeframe": "M15"}))

    exit_code = run_demo_execution_command(settings)

    assert exit_code == 0  # gate bypassed, order proceeds
    assert fake_client.sent_orders  # order was sent despite pre-existing position


# ---------------------------------------------------------------------------
# Exit parity constants and helpers
# ---------------------------------------------------------------------------

# Full profile snapshot matching the real approved EURUSD profile.
# These values mirror data/runtime/demo_execution_models/EURUSD/model_artifact_manifest.json
# so that test fixtures exercise the same code path as production.
_FULL_PROFILE_SNAPSHOT: dict[str, object] = {
    "profile_id": "EURUSD-approved",
    "promotion_state": "approved_demo",
    "stop_policy": "static",
    "target_policy": "static",
    "stop_atr_multiplier": 1.5,
    "target_atr_multiplier": 3.0,
    "stop_min_pct": 0.001,
    "stop_max_pct": 0.01,
    "target_min_pct": 0.0015,
    "target_max_pct": 0.02,
}

# FakeDemoClient tick prices — entry_price for buy is ask.
_FAKE_ASK = 1.1002
_FAKE_BID = 1.1000


def _manifest_stub(profile_snapshot: dict[str, object]) -> dict[str, object]:
    """Return a load_validated_model manifest stub with the given profile snapshot."""
    return {
        "feature_names": ["f1", "f2", "atr_10", "atr_5", "rolling_volatility_10", "rolling_volatility_5"],
        "threshold": 0.45,
        "base_profile_snapshot": profile_snapshot,
    }


def _expected_exits(
    settings,
    entry_price: float,
    direction: int,
    features: dict[str, float],
    profile_snapshot: dict[str, object],
) -> tuple[float, float]:
    """Independently compute expected (sl_price, tp_price) using exits.py directly."""
    exit_profile = SymbolExitProfile(
        stop_policy=str(profile_snapshot.get("stop_policy", "static")),
        target_policy=str(profile_snapshot.get("target_policy", "static")),
        stop_atr_multiplier=float(profile_snapshot.get("stop_atr_multiplier", settings.risk.atr_stop_loss_multiplier)),
        target_atr_multiplier=float(profile_snapshot.get("target_atr_multiplier", settings.risk.atr_take_profit_multiplier)),
        stop_min_pct=float(profile_snapshot["stop_min_pct"]) if "stop_min_pct" in profile_snapshot else None,
        stop_max_pct=float(profile_snapshot["stop_max_pct"]) if "stop_max_pct" in profile_snapshot else None,
        target_min_pct=float(profile_snapshot["target_min_pct"]) if "target_min_pct" in profile_snapshot else None,
        target_max_pct=float(profile_snapshot["target_max_pct"]) if "target_max_pct" in profile_snapshot else None,
    )
    stop_obj, target_obj = build_exit_policies(exit_profile.stop_policy, exit_profile.target_policy)
    row = ProcessedRow(
        timestamp=datetime.now(tz=UTC),
        symbol="EURUSD",
        timeframe="M15",
        open=entry_price,
        high=entry_price,
        low=entry_price,
        close=entry_price,
        volume=0.0,
        label=0,
        label_reason="",
        horizon_end_timestamp="",
        features=features,
    )
    sl = stop_obj.stop_loss_price(row, entry_price, direction, settings.backtest, settings.risk, settings.dynamic_exits, exit_profile)
    tp = target_obj.take_profit_price(row, entry_price, direction, settings.backtest, settings.risk, settings.dynamic_exits, exit_profile)
    return round(sl.price, 8), round(tp.price, 8)


def _run_parity_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    atr_10: float,
    atr_5: float,
    signal_direction: int = 1,
    profile_snapshot: dict[str, object] | None = None,
) -> tuple[object, dict[str, object], Path]:
    """Shared scaffold for parity tests: runs demo execution and returns (settings, trace, run_dir)."""
    snapshot = profile_snapshot if profile_snapshot is not None else _FULL_PROFILE_SNAPSHOT
    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    _write_demo_registry(settings, manifest_path, active=True, approved=True)

    fake_client = FakeDemoClient(settings.mt5)

    proba = {-1: 0.1, 0: 0.2, 1: 0.7} if signal_direction == 1 else {-1: 0.7, 0: 0.2, 1: 0.1}

    class FakeModel:
        def predict_probabilities(self, rows):
            return [proba]

    manifest_stub = _manifest_stub(snapshot)
    monkeypatch.setattr("iris_bot.demo_execution.MT5Client", lambda cfg: fake_client)
    monkeypatch.setattr(
        "iris_bot.demo_execution.load_validated_model",
        lambda s, symbol, manifest_path: (FakeModel(), {"ok": True, "manifest": manifest_stub, "reasons": []}),
    )
    monkeypatch.setattr(
        "iris_bot.demo_execution.validate_model_artifact",
        lambda s, symbol, manifest_path: {"ok": True, "manifest": manifest_stub, "reasons": []},
    )
    features = {"f1": 1.0, "f2": 1.0, "atr_10": atr_10, "atr_5": atr_5,
                "rolling_volatility_10": 0.0, "rolling_volatility_5": 0.0}
    monkeypatch.setattr(
        "iris_bot.demo_execution._latest_row_features",
        lambda s, c, sym, fn: (features, {"timestamp": "2026-04-06T00:00:00", "symbol": sym, "timeframe": "M15"}),
    )

    exit_code = run_demo_execution_command(settings)
    assert exit_code == 0, f"Expected exit_code=0, got {exit_code}"

    run_dir = sorted(settings.data.runs_dir.glob("*_run_demo_execution"))[-1]
    trace = json.loads((run_dir / "broker_order_trace.json").read_text(encoding="utf-8"))["payload"]
    return settings, trace, run_dir


def test_sl_tp_parity_floor_dominates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parity: when ATR is very low, fixed_stop_loss_pct floor dominates.
    SL/TP from demo execution must exactly match exits.py with the same inputs.
    """
    atr_10, atr_5 = 0.0001, 0.00009  # tiny ATR → floor wins
    settings, trace, _ = _run_parity_session(tmp_path, monkeypatch, atr_10=atr_10, atr_5=atr_5)

    features = {"atr_10": atr_10, "atr_5": atr_5, "rolling_volatility_10": 0.0, "rolling_volatility_5": 0.0}
    expected_sl, expected_tp = _expected_exits(settings, _FAKE_ASK, 1, features, _FULL_PROFILE_SNAPSHOT)

    actual_sl = trace["request"]["sl"]
    actual_tp = trace["request"]["tp"]

    assert actual_sl == expected_sl, f"SL mismatch: demo={actual_sl} exits.py={expected_sl}"
    assert actual_tp == expected_tp, f"TP mismatch: demo={actual_tp} exits.py={expected_tp}"

    # Structural check: floor distance = entry * fixed_stop_loss_pct
    expected_sl_dist = round(_FAKE_ASK * settings.backtest.fixed_stop_loss_pct, 8)
    actual_sl_dist = round(_FAKE_ASK - actual_sl, 8)
    assert abs(actual_sl_dist - expected_sl_dist) < 1e-7, (
        f"Floor-dominant SL distance wrong: {actual_sl_dist} vs expected {expected_sl_dist}"
    )


def test_sl_tp_parity_atr_dominates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parity: when ATR is large, atr*price*multiplier dominates over floor.
    SL/TP from demo execution must exactly match exits.py, and must NOT match old hardcoded values.
    """
    atr_10, atr_5 = 0.003, 0.0027  # large ATR → ATR-based distance beats 0.2% floor
    settings, trace, _ = _run_parity_session(tmp_path, monkeypatch, atr_10=atr_10, atr_5=atr_5)

    features = {"atr_10": atr_10, "atr_5": atr_5, "rolling_volatility_10": 0.0, "rolling_volatility_5": 0.0}
    expected_sl, expected_tp = _expected_exits(settings, _FAKE_ASK, 1, features, _FULL_PROFILE_SNAPSHOT)

    actual_sl = trace["request"]["sl"]
    actual_tp = trace["request"]["tp"]

    assert actual_sl == expected_sl, f"SL mismatch: demo={actual_sl} exits.py={expected_sl}"
    assert actual_tp == expected_tp, f"TP mismatch: demo={actual_tp} exits.py={expected_tp}"

    # Regression guard: ATR-derived SL distance must be larger than the old hardcoded floor (0.001).
    # If someone reintroduces `max(price * 0.0010, 0.0005)`, this assertion fails.
    old_hardcoded_sl_dist = max(_FAKE_ASK * 0.0010, 0.0005)
    actual_sl_dist = _FAKE_ASK - actual_sl
    assert actual_sl_dist > old_hardcoded_sl_dist, (
        f"Regression: SL distance {actual_sl_dist:.6f} must exceed old hardcoded floor "
        f"{old_hardcoded_sl_dist:.6f} when ATR is high — "
        "check that demo_execution.py uses exits.py instead of hardcoded constants"
    )


def test_sl_tp_parity_sell_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parity: sell signal — entry from bid, SL above entry, TP below entry; both match exits.py."""
    atr_10, atr_5 = 0.0001, 0.00009
    settings, trace, _ = _run_parity_session(
        tmp_path, monkeypatch, atr_10=atr_10, atr_5=atr_5, signal_direction=-1
    )

    features = {"atr_10": atr_10, "atr_5": atr_5, "rolling_volatility_10": 0.0, "rolling_volatility_5": 0.0}
    expected_sl, expected_tp = _expected_exits(settings, _FAKE_BID, -1, features, _FULL_PROFILE_SNAPSHOT)

    actual_sl = trace["request"]["sl"]
    actual_tp = trace["request"]["tp"]

    assert actual_sl == expected_sl, f"SL mismatch (sell): demo={actual_sl} exits.py={expected_sl}"
    assert actual_tp == expected_tp, f"TP mismatch (sell): demo={actual_tp} exits.py={expected_tp}"

    # For a sell: SL must be above entry, TP must be below entry.
    assert actual_sl > _FAKE_BID, "Sell SL must be above entry price"
    assert actual_tp < _FAKE_BID, "Sell TP must be below entry price"


def test_broker_order_trace_has_exit_policy_details(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Artifact structure: broker_order_trace must contain exit_policy with sl/tp details for auditability."""
    settings, trace, run_dir = _run_parity_session(tmp_path, monkeypatch, atr_10=0.0002, atr_5=0.00018)

    # exit_policy block must be present
    assert "exit_policy" in trace, "broker_order_trace missing exit_policy block"
    ep = trace["exit_policy"]
    assert ep["stop_policy"] == "static"
    assert ep["target_policy"] == "static"

    # sl_details and tp_details must be present and contain the engine breakdown
    sl_details = ep["sl_details"]
    tp_details = ep["tp_details"]
    for key in ("policy", "entry_price", "atr_fraction", "atr_distance", "floor_distance", "final_distance", "direction"):
        assert key in sl_details, f"sl_details missing key: {key}"
        assert key in tp_details, f"tp_details missing key: {key}"

    # entry_price, sl, tp must be in the request block
    req = trace["request"]
    assert "price" in req
    assert "sl" in req
    assert "tp" in req
    assert req["sl"] != req["tp"], "SL and TP must differ"
    assert req["sl"] < req["price"], "Buy SL must be below entry price"
    assert req["tp"] > req["price"], "Buy TP must be above entry price"


# ---------------------------------------------------------------------------
# Fix 1 — pending intent cleanup when position closed
# ---------------------------------------------------------------------------

def test_stale_pending_intent_cleared_when_no_broker_position(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix 1: if runtime_state has a pending intent but broker has no position, intent is removed."""
    import json as _json
    from iris_bot.resilient_state import persist_runtime_state, state_from_dict
    from iris_bot.operational import PendingIntent, PaperEngineState, AccountState

    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    _write_demo_registry(settings, manifest_path, active=True, approved=True)

    # Pre-seed runtime_state with a stale EURUSD pending intent.
    rt_path = settings.data.runtime_dir / settings.operational.persistence_state_filename
    state = PaperEngineState(account_state=AccountState(1000.0, 1000.0, 1000.0))
    state.pending_intents.append(PendingIntent(
        symbol="EURUSD", created_at="2026-04-07T00:00:00+00:00",
        signal_timestamp="2026-04-07T00:00:00+00:00", side="buy",
        volume_lots=0.01, active_profile_id="", model_variant="", promotion_state="approved_demo",
    ))
    persist_runtime_state(rt_path, state, {})
    assert len(state_from_dict(_json.loads(rt_path.read_text())["state"]).pending_intents) == 1

    fake_client = FakeDemoClient(settings.mt5)
    # _position_open=False → broker has no EURUSD position → intent should be cleared.

    class FakeModel:
        def predict_probabilities(self, rows):
            return [{-1: 0.1, 0: 0.2, 1: 0.7}]

    monkeypatch.setattr("iris_bot.demo_execution.MT5Client", lambda cfg: fake_client)
    monkeypatch.setattr("iris_bot.demo_execution.load_validated_model", lambda s, symbol, manifest_path: (FakeModel(), {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []}))
    monkeypatch.setattr("iris_bot.demo_execution.validate_model_artifact", lambda s, symbol, manifest_path: {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []})
    monkeypatch.setattr("iris_bot.demo_execution._latest_row_features", lambda s, c, sym, fn: ({"f1": 1.0, "f2": 1.0}, {"timestamp": "2026-04-06T00:00:00", "symbol": sym, "timeframe": "M15"}))

    exit_code = run_demo_execution_command(settings)

    assert exit_code == 0
    remaining = state_from_dict(_json.loads(rt_path.read_text())["state"]).pending_intents
    eurusd_intents = [p for p in remaining if p.symbol == "EURUSD"]
    # After cleanup: stale intent removed; a new one was registered because the order succeeded.
    # The new intent reflects the session that just opened a position.
    assert len(eurusd_intents) == 1
    assert eurusd_intents[0].side == "buy"  # freshly registered from the new open


def test_pending_intent_not_cleared_when_position_still_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix 1 (inverse): when a broker position exists, the cleanup must NOT remove the intent."""
    import json as _json
    from iris_bot.resilient_state import persist_runtime_state, state_from_dict
    from iris_bot.operational import PendingIntent, PaperEngineState, AccountState

    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    _write_demo_registry(settings, manifest_path, active=True, approved=True)

    rt_path = settings.data.runtime_dir / settings.operational.persistence_state_filename
    state = PaperEngineState(account_state=AccountState(1000.0, 1000.0, 1000.0))
    state.pending_intents.append(PendingIntent(
        symbol="EURUSD", created_at="2026-04-07T00:00:00+00:00",
        signal_timestamp="2026-04-07T00:00:00+00:00", side="buy",
        volume_lots=0.01, active_profile_id="", model_variant="", promotion_state="approved_demo",
    ))
    persist_runtime_state(rt_path, state, {})

    fake_client = FakeDemoClient(settings.mt5)
    fake_client._position_open = True  # broker has position → gate fires, intent NOT cleaned

    class FakeModel:
        def predict_probabilities(self, rows):
            return [{-1: 0.1, 0: 0.2, 1: 0.7}]

    monkeypatch.setattr("iris_bot.demo_execution.MT5Client", lambda cfg: fake_client)
    monkeypatch.setattr("iris_bot.demo_execution.load_validated_model", lambda s, symbol, manifest_path: (FakeModel(), {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []}))
    monkeypatch.setattr("iris_bot.demo_execution.validate_model_artifact", lambda s, symbol, manifest_path: {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []})
    monkeypatch.setattr("iris_bot.demo_execution._latest_row_features", lambda s, c, sym, fn: ({"f1": 1.0, "f2": 1.0}, {"timestamp": "2026-04-06T00:00:00", "symbol": sym, "timeframe": "M15"}))

    exit_code = run_demo_execution_command(settings)

    assert exit_code == 5  # position_already_open gate
    remaining = state_from_dict(_json.loads(rt_path.read_text())["state"]).pending_intents
    eurusd_intents = [p for p in remaining if p.symbol == "EURUSD"]
    assert len(eurusd_intents) == 1, "Intent must NOT be cleared when broker position still exists"


def test_stale_pending_intent_cleared_even_when_signal_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: cleanup must fire before the signal==0 early return, not after.

    Before the fix, _clear_stale_pending_intent_if_position_closed was placed after
    ``if signal == 0: return 3``, so no-signal sessions never cleaned stale intents.
    """
    import json as _json
    from iris_bot.resilient_state import persist_runtime_state, state_from_dict
    from iris_bot.operational import PendingIntent, PaperEngineState, AccountState

    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    _write_demo_registry(settings, manifest_path, active=True, approved=True)

    # Pre-seed runtime_state with a stale EURUSD pending intent.
    rt_path = settings.data.runtime_dir / settings.operational.persistence_state_filename
    state = PaperEngineState(account_state=AccountState(1000.0, 1000.0, 1000.0))
    state.pending_intents.append(PendingIntent(
        symbol="EURUSD", created_at="2026-04-07T00:00:00+00:00",
        signal_timestamp="2026-04-07T00:00:00+00:00", side="buy",
        volume_lots=0.01, active_profile_id="", model_variant="", promotion_state="approved_demo",
    ))
    persist_runtime_state(rt_path, state, {})
    assert len(state_from_dict(_json.loads(rt_path.read_text())["state"]).pending_intents) == 1

    fake_client = FakeDemoClient(settings.mt5)
    # No broker position → cleanup should fire; signal=0 → no order sent.

    class NoSignalModel:
        def predict_probabilities(self, rows):
            return [{-1: 0.2, 0: 0.5, 1: 0.3}]  # all below threshold=0.45

    monkeypatch.setattr("iris_bot.demo_execution.MT5Client", lambda cfg: fake_client)
    monkeypatch.setattr("iris_bot.demo_execution.load_validated_model", lambda s, symbol, manifest_path: (NoSignalModel(), {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []}))
    monkeypatch.setattr("iris_bot.demo_execution.validate_model_artifact", lambda s, symbol, manifest_path: {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []})
    monkeypatch.setattr("iris_bot.demo_execution._latest_row_features", lambda s, c, sym, fn: ({"f1": 1.0, "f2": 1.0}, {"timestamp": "2026-04-07T00:00:00", "symbol": sym, "timeframe": "M15"}))

    exit_code = run_demo_execution_command(settings)

    assert exit_code == 3  # no trade signal
    remaining = state_from_dict(_json.loads(rt_path.read_text())["state"]).pending_intents
    eurusd_intents = [p for p in remaining if p.symbol == "EURUSD"]
    assert len(eurusd_intents) == 0, "Stale intent must be cleared even when signal==0"


# ---------------------------------------------------------------------------
# Fix 2 — threshold from base_profile_snapshot
# ---------------------------------------------------------------------------

def test_threshold_uses_base_profile_snapshot_over_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix 2: when base_profile_snapshot has a threshold, it takes precedence over the registry value."""
    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    _write_demo_registry(settings, manifest_path, active=True, approved=True)

    fake_client = FakeDemoClient(settings.mt5)

    # Manifest with profile_snapshot.threshold=0.45 but manifest-level threshold=0.40.
    manifest_stub = {
        "feature_names": ["f1", "f2"],
        "threshold": 0.40,  # registry/manifest value
        "base_profile_snapshot": {
            "profile_id": "EURUSD-approved",
            "promotion_state": "approved_demo",
            "stop_policy": "static",
            "target_policy": "static",
            "threshold": 0.45,  # profile snapshot — must win
        },
    }

    class FakeModelNeverTrades:
        """Always returns probabilities that pass 0.40 but NOT 0.45."""
        def predict_probabilities(self, rows):
            # class 1 probability = 0.42 → passes 0.40, blocked by 0.45
            return [{-1: 0.10, 0: 0.48, 1: 0.42}]

    monkeypatch.setattr("iris_bot.demo_execution.MT5Client", lambda cfg: fake_client)
    monkeypatch.setattr("iris_bot.demo_execution.load_validated_model", lambda s, symbol, manifest_path: (FakeModelNeverTrades(), {"ok": True, "manifest": manifest_stub, "reasons": []}))
    monkeypatch.setattr("iris_bot.demo_execution.validate_model_artifact", lambda s, symbol, manifest_path: {"ok": True, "manifest": manifest_stub, "reasons": []})
    monkeypatch.setattr("iris_bot.demo_execution._latest_row_features", lambda s, c, sym, fn: ({"f1": 1.0, "f2": 1.0}, {"timestamp": "2026-04-06T00:00:00", "symbol": sym, "timeframe": "M15"}))

    exit_code = run_demo_execution_command(settings)

    # With threshold=0.45 (from profile snapshot), probability 0.42 → signal=0 → no trade.
    # If the old registry threshold 0.40 were used, signal would be 1 → order sent → exit_code=0.
    assert exit_code == 3, (
        "With profile_snapshot.threshold=0.45, prob=0.42 must produce signal=0. "
        "If exit_code==0, the code is using the registry threshold (0.40) instead."
    )
    assert fake_client.sent_orders == [], "No order must be sent when signal=0 from correct threshold"

    # Verify the threshold recorded in inference_preflight_report is 0.45.
    run_dir = sorted(settings.data.runs_dir.glob("*_run_demo_execution"))[-1]
    inference = json.loads((run_dir / "inference_preflight_report.json").read_text())["payload"]
    assert abs(inference["threshold"] - 0.45) < 1e-9, (
        f"inference_preflight_report must record threshold=0.45, got {inference['threshold']}"
    )


# ---------------------------------------------------------------------------
# Fix 3 — exit_code=5 for position_already_open (distinct from 3 = no_trade_signal)
# ---------------------------------------------------------------------------

def test_position_already_open_returns_exit_code_5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix 3: position_already_open returns exit_code=5, not 3, to distinguish from signal=0."""
    settings = _settings(tmp_path, monkeypatch)
    _write_approved_demo_profile(settings)
    manifest_path = _write_model_manifest(settings)
    _write_demo_registry(settings, manifest_path, active=True, approved=True)

    fake_client = FakeDemoClient(settings.mt5)
    fake_client._position_open = True  # existing position → gate fires

    class FakeModel:
        def predict_probabilities(self, rows):
            return [{-1: 0.1, 0: 0.2, 1: 0.7}]

    monkeypatch.setattr("iris_bot.demo_execution.MT5Client", lambda cfg: fake_client)
    monkeypatch.setattr("iris_bot.demo_execution.load_validated_model", lambda s, symbol, manifest_path: (FakeModel(), {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []}))
    monkeypatch.setattr("iris_bot.demo_execution.validate_model_artifact", lambda s, symbol, manifest_path: {"ok": True, "manifest": {"feature_names": ["f1", "f2"], "threshold": 0.45}, "reasons": []})
    monkeypatch.setattr("iris_bot.demo_execution._latest_row_features", lambda s, c, sym, fn: ({"f1": 1.0, "f2": 1.0}, {"timestamp": "2026-04-06T00:00:00", "symbol": sym, "timeframe": "M15"}))

    exit_code = run_demo_execution_command(settings)

    assert exit_code == 5, f"position_already_open must return 5, got {exit_code}"
    # Sanity: signal=0 still returns 3 (unchanged)
    # (covered by test_run_demo_execution_sends_order_and_writes_reports and no_trade variants)
