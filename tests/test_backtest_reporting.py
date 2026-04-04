import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("xgboost")

from iris_bot.backtest import run_backtest
from iris_bot.config import load_settings
from iris_bot.data import Bar, write_bars
from iris_bot.experiments import run_experiment
from iris_bot.processed_dataset import build_processed_dataset, write_processed_dataset


def test_run_backtest_creates_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    runtime_dir = tmp_path / "data" / "runtime"
    runs_dir = tmp_path / "runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    bars = []
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

    assert run_experiment(settings) == 0
    experiment_run_dir = sorted(runs_dir.glob("*_experiment"))[-1]
    object.__setattr__(settings.backtest, "experiment_run_dir", str(experiment_run_dir))

    assert run_backtest(settings) == 0
    backtest_run_dir = sorted(runs_dir.glob("*_backtest"))[-1]
    assert (backtest_run_dir / "trade_log.csv").exists()
    assert (backtest_run_dir / "equity_curve.csv").exists()
    payload = json.loads((backtest_run_dir / "backtest_report.json").read_text(encoding="utf-8"))
    assert "metrics" in payload
    assert "experiment_reference" in payload
