from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from iris_bot.artifacts import wrap_artifact
from iris_bot.config import load_settings
from iris_bot.demo_execution import (
    _classify_order_result,
    demo_execution_preflight_payload,
    run_demo_execution_command,
)
from iris_bot.demo_execution_registry import save_demo_execution_registry
from iris_bot.model_artifacts import (
    build_model_artifact_manifest,
    validate_model_artifact,
    write_model_artifact_manifest,
)
from iris_bot.mt5 import MT5Config, OrderResult
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
    assert fake_client.closed_orders
    run_dir = sorted(settings.data.runs_dir.glob("*_run_demo_execution"))[-1]
    assert (run_dir / "demo_execution_report.json").exists()
    assert (run_dir / "broker_order_trace.json").exists()
    assert (run_dir / "post_trade_reconciliation_report.json").exists()
    trace = json.loads((run_dir / "broker_order_trace.json").read_text(encoding="utf-8"))
    assert trace["payload"]["request"]["sl"] != trace["payload"]["request"]["tp"]


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
