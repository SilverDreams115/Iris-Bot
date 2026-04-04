import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from iris_bot.config import settings
from iris_bot.data import Bar, write_bars
from iris_bot.main import build_dataset_command
from iris_bot.processed_dataset import load_processed_dataset


pytest.importorskip("xgboost")


def test_experiment_with_xgboost_creates_processed_dataset_and_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    runs_dir = tmp_path / "runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
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
                high=price + 0.0006,
                low=price - 0.0006,
                close=price,
                volume=100 + (index % 20),
            )
        )
    write_bars(raw_dir / "market.csv", bars)

    monkeypatch.setenv("IRIS_LABEL_MODE", "triple_barrier")
    monkeypatch.setenv("IRIS_PRIMARY_TIMEFRAME", "M15")
    monkeypatch.setenv("IRIS_WF_TRAIN_WINDOW", "150")
    monkeypatch.setenv("IRIS_WF_VALIDATION_WINDOW", "50")
    monkeypatch.setenv("IRIS_WF_TEST_WINDOW", "50")
    monkeypatch.setenv("IRIS_WF_STEP", "50")

    from iris_bot.config import load_settings
    from iris_bot.experiments import run_experiment

    local_settings = load_settings()
    object.__setattr__(local_settings.data, "raw_dir", raw_dir)
    object.__setattr__(local_settings.data, "processed_dir", processed_dir)
    object.__setattr__(local_settings.data, "runs_dir", runs_dir)
    object.__setattr__(local_settings.experiment, "_processed_dir", processed_dir)

    dataset = load_processed_dataset if False else None
    del dataset

    from iris_bot.processed_dataset import build_processed_dataset, write_processed_dataset

    processed = build_processed_dataset(bars, local_settings.labeling)
    write_processed_dataset(
        processed,
        local_settings.experiment.processed_dataset_path,
        local_settings.experiment.processed_manifest_path,
        local_settings.experiment.processed_schema_path,
    )

    exit_code = run_experiment(local_settings)

    assert exit_code == 0
    report_files = list(runs_dir.glob("*_experiment/experiment_report.json"))
    assert report_files
    payload = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert "xgboost" in payload
    assert "baseline" in payload
