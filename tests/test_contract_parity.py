from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from iris_bot.backtest import run_backtest
from iris_bot.config import load_settings
from iris_bot.data import Bar, write_bars
from iris_bot.experiments import run_experiment
from iris_bot.paper import run_paper_session
from iris_bot.processed_dataset import ProcessedRow, build_processed_dataset, write_processed_dataset
from iris_bot.symbols import write_symbol_strategy_profiles
from iris_bot.wf_backtest import run_walkforward_economic_backtest


def _wf_rows(n: int) -> list[ProcessedRow]:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows: list[ProcessedRow] = []
    price = 1.1000
    for index in range(n):
        price += 0.0003 if index % 4 < 2 else -0.0001
        timestamp = start + timedelta(minutes=15 * index)
        rows.append(
            ProcessedRow(
                timestamp=timestamp,
                symbol="EURUSD",
                timeframe="M15",
                open=price - 0.0001,
                high=price + 0.0010,
                low=price - 0.0010,
                close=price,
                volume=100.0,
                label=1 if index % 3 == 0 else (0 if index % 3 == 1 else -1),
                label_reason="test",
                horizon_end_timestamp=timestamp.isoformat(),
                features={
                    "return_1": 0.0003,
                    "momentum_3": 0.001 * (index % 5 - 2),
                    "rolling_volatility_5": 0.001,
                    "atr_5": 0.0004 + (index % 3) * 0.0001,
                },
            )
        )
    return rows


def test_walkforward_training_contract_passes_sample_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, _config) -> None:
            return None

        def fit(
            self,
            train_rows: list[list[float]],
            train_labels: list[int],
            validation_rows: list[list[float]],
            validation_labels: list[int],
            feature_names: list[str] | None = None,
            sample_weights: list[float] | None = None,
        ) -> None:
            captured["train_rows"] = len(train_rows)
            captured["train_labels"] = len(train_labels)
            captured["validation_rows"] = len(validation_rows)
            captured["feature_names"] = feature_names
            captured["sample_weights"] = sample_weights

        def predict_probabilities(self, rows: list[list[float]]) -> list[dict[int, float]]:
            return [{1: 0.85, 0: 0.10, -1: 0.05} for _ in rows]

    monkeypatch.setattr("iris_bot.wf_backtest.XGBoostMultiClassModel", FakeModel)

    settings = load_settings()
    import dataclasses

    settings = dataclasses.replace(
        settings,
        walk_forward=dataclasses.replace(
            settings.walk_forward,
            enabled=True,
            train_window=60,
            validation_window=20,
            test_window=20,
            step=20,
        ),
        backtest=dataclasses.replace(
            settings.backtest,
            starting_balance_usd=1000.0,
            use_atr_stops=False,
            fixed_stop_loss_pct=0.005,
            fixed_take_profit_pct=0.010,
            spread_pips=0.0,
            slippage_pips=0.0,
            commission_per_lot_per_side_usd=0.0,
            max_holding_bars=10,
            intrabar_policy="conservative",
        ),
        risk=dataclasses.replace(
            settings.risk,
            max_daily_loss_usd=999.0,
            max_open_positions=4,
            cooldown_bars_after_loss=0,
        ),
        data=dataclasses.replace(settings.data, runs_dir=tmp_path / "runs"),
    )

    run_dir = tmp_path / "wf_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = run_walkforward_economic_backtest(
        rows=_wf_rows(130),
        feature_names=["return_1", "momentum_3", "rolling_volatility_5", "atr_5"],
        settings=settings,
        run_dir=run_dir,
        logger=logging.getLogger("wf_contract_test"),
    )

    assert result["total_folds"] >= 1
    sample_weights = captured["sample_weights"]
    assert isinstance(sample_weights, list)
    assert len(sample_weights) == captured["train_rows"]
    assert captured["feature_names"] == ["return_1", "momentum_3", "rolling_volatility_5", "atr_5"]


pytest.importorskip("xgboost")


def test_backtest_and_paper_persist_matching_evaluation_contract_threshold_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    runtime_dir = tmp_path / "data" / "runtime"
    runs_dir = tmp_path / "runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    bars: list[Bar] = []
    start = datetime(2026, 1, 1, 0, 0, 0)
    price = 1.1000
    for index in range(420):
        price += 0.0004 if index % 5 in (0, 1, 2) else -0.0002
        bars.append(
            Bar(
                timestamp=start + timedelta(minutes=15 * index),
                symbol="EURUSD",
                timeframe="M15",
                open=price - 0.0002,
                high=price + 0.0008,
                low=price - 0.0008,
                close=price,
                volume=100 + (index % 20),
            )
        )
    write_bars(raw_dir / "market.csv", bars)

    monkeypatch.setenv("IRIS_PRIMARY_TIMEFRAME", "M15")
    monkeypatch.setenv("IRIS_WF_TRAIN_WINDOW", "150")
    monkeypatch.setenv("IRIS_WF_VALIDATION_WINDOW", "50")
    monkeypatch.setenv("IRIS_WF_TEST_WINDOW", "50")
    monkeypatch.setenv("IRIS_WF_STEP", "50")
    monkeypatch.setenv("IRIS_GOVERNANCE_REQUIRE_ACTIVE_PROFILE", "false")

    settings = load_settings()
    object.__setattr__(settings.data, "raw_dir", raw_dir)
    object.__setattr__(settings.data, "processed_dir", processed_dir)
    object.__setattr__(settings.data, "runtime_dir", runtime_dir)
    object.__setattr__(settings.data, "runs_dir", runs_dir)
    object.__setattr__(settings.experiment, "_processed_dir", processed_dir)

    processed = build_processed_dataset(bars, settings.labeling)
    write_processed_dataset(
        processed,
        settings.experiment.processed_dataset_path,
        settings.experiment.processed_manifest_path,
        settings.experiment.processed_schema_path,
    )
    write_symbol_strategy_profiles(
        settings,
        common_profile={},
        symbol_profiles={
            "EURUSD": {
                "threshold": 0.8,
                "enabled_state": "enabled",
                "allowed_timeframes": ["M15"],
                "allowed_sessions": ["asia", "london", "new_york"],
            }
        },
    )

    assert run_experiment(settings) == 0
    experiment_run_dir = sorted(runs_dir.glob("*_experiment"))[-1]
    object.__setattr__(settings.backtest, "experiment_run_dir", str(experiment_run_dir))

    assert run_backtest(settings) == 0
    backtest_run_dir = sorted(runs_dir.glob("*_backtest"))[-1]
    backtest_report = json.loads((backtest_run_dir / "backtest_report.json").read_text(encoding="utf-8"))

    code, paper_run_dir = run_paper_session(settings, "paper")
    assert code == 0
    paper_config = json.loads((paper_run_dir / "config_used.json").read_text(encoding="utf-8"))

    assert backtest_report["evaluation_contract"]["threshold_application"]["policy"] == "max_global_and_profile_threshold"
    assert paper_config["evaluation_contract"]["threshold_application"]["policy"] == "max_global_and_profile_threshold"
    assert backtest_report["evaluation_contract"]["threshold_application"]["threshold_by_symbol"]["EURUSD"] == 0.8
    assert paper_config["evaluation_contract"]["threshold_application"]["threshold_by_symbol"]["EURUSD"] == 0.8
    assert backtest_report["training_contract_version"] == paper_config["training_contract_version"] == "1.0"
    assert backtest_report["evaluation_contract_version"] == paper_config["evaluation_contract_version"] == "1.0"
    assert backtest_report["artifact_provenance"]["correlation_keys"]["experiment_run_id"] == experiment_run_dir.name
    assert paper_config["artifact_provenance"]["correlation_keys"]["experiment_run_id"] == experiment_run_dir.name
    assert backtest_report["artifact_provenance"]["contract_hashes"]["bundle"] == backtest_report["contract_hashes"]["bundle"]
    assert paper_config["artifact_provenance"]["contract_hashes"]["bundle"] == paper_config["contract_hashes"]["bundle"]
    assert backtest_report["contract_hashes"]["training_contract"] == paper_config["contract_hashes"]["training_contract"]
