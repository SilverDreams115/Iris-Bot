import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from iris_bot.data import Bar, write_bars
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
    assert payload["training_contract_version"] == "1.0"
    assert payload["evaluation_contract_version"] == "1.0"
    assert payload["artifact_provenance"]["run_id"].endswith("_experiment")
    assert payload["artifact_provenance"]["correlation_keys"]["command"] == "experiment"
    assert payload["artifact_provenance"]["contract_hashes"]["bundle"] == payload["contract_hashes"]["bundle"]
    assert payload["training_contract"]["economic_sample_weighting"]["enabled"] is True
    assert payload["evaluation_contract"]["threshold_application"]["policy"] == "global_threshold_only"
    assert "bundle" in payload["contract_hashes"]
    assert payload["xgboost"]["class_weighting"]["enabled"] is True
    assert "weights" in payload["xgboost"]["class_weighting"]
    assert payload["xgboost"]["probability_calibration"]["enabled"] is True
    assert payload["xgboost"]["probability_calibration"]["method"] == "global_temperature"
    assert payload["xgboost"]["probability_calibration"]["temperature"] > 0.0
    assert payload["xgboost"]["probability_calibration"]["class_temperatures"] == {}
    assert payload["xgboost"]["probability_calibration"]["validation_log_loss_before"] is not None
    assert payload["xgboost"]["probability_calibration"]["validation_log_loss_after"] is not None
    comparison = payload["xgboost"]["probability_calibration"]["comparison"]
    assert set(comparison["validation_log_loss_by_method"]) == {
        "uncalibrated",
        "global_temperature",
        "classwise_temperature",
    }
    assert comparison["best_method"] in comparison["validation_log_loss_by_method"]
    assert comparison["applied_method"] == "global_temperature"
    assert comparison["configured_method"] == "global_temperature"
    assert isinstance(comparison["applied_matches_best"], bool)
    assert (
        payload["xgboost"]["probability_calibration"]["validation_log_loss_after"]
        <= payload["xgboost"]["probability_calibration"]["validation_log_loss_before"]
    )

    model_metadata_files = list(runs_dir.glob("*_experiment/models/xgboost_metadata.json"))
    assert model_metadata_files
    metadata = json.loads(model_metadata_files[0].read_text(encoding="utf-8"))
    assert "class_weights" in metadata
    assert metadata["probability_calibration"]["method"] == "global_temperature"
    assert metadata["probability_calibration"]["temperature"] > 0.0
    assert metadata["probability_calibration"]["class_temperatures"] == {}
    assert metadata["probability_calibration"]["validation_log_loss_before"] is not None
    assert metadata["probability_calibration"]["validation_log_loss_after"] is not None
    assert metadata["probability_calibration"]["comparison"]["applied_method"] == "global_temperature"
    assert metadata["probability_calibration"]["comparison"]["configured_method"] == "global_temperature"


def test_experiment_with_classwise_calibration_persists_method(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv("IRIS_XGB_PROBABILITY_CALIBRATION_METHOD", "classwise_temperature")

    from iris_bot.config import load_settings
    from iris_bot.experiments import run_experiment
    from iris_bot.processed_dataset import build_processed_dataset, write_processed_dataset

    local_settings = load_settings()
    object.__setattr__(local_settings.data, "raw_dir", raw_dir)
    object.__setattr__(local_settings.data, "processed_dir", processed_dir)
    object.__setattr__(local_settings.data, "runs_dir", runs_dir)
    object.__setattr__(local_settings.experiment, "_processed_dir", processed_dir)

    processed = build_processed_dataset(bars, local_settings.labeling)
    write_processed_dataset(
        processed,
        local_settings.experiment.processed_dataset_path,
        local_settings.experiment.processed_manifest_path,
        local_settings.experiment.processed_schema_path,
    )

    assert run_experiment(local_settings) == 0
    report_files = list(runs_dir.glob("*_experiment/experiment_report.json"))
    assert report_files
    payload = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert payload["xgboost"]["probability_calibration"]["method"] == "classwise_temperature"
    assert payload["xgboost"]["probability_calibration"]["temperature"] == 1.0
    assert payload["xgboost"]["probability_calibration"]["class_temperatures"]
    comparison = payload["xgboost"]["probability_calibration"]["comparison"]
    assert set(comparison["validation_log_loss_by_method"]) == {
        "uncalibrated",
        "global_temperature",
        "classwise_temperature",
    }
    assert comparison["best_method"] in comparison["validation_log_loss_by_method"]
    assert comparison["applied_method"] == "classwise_temperature"
    assert comparison["configured_method"] == "classwise_temperature"
    assert isinstance(comparison["applied_matches_best"], bool)

    model_metadata_files = list(runs_dir.glob("*_experiment/models/xgboost_metadata.json"))
    assert model_metadata_files
    metadata = json.loads(model_metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["probability_calibration"]["method"] == "classwise_temperature"
    assert metadata["probability_calibration"]["temperature"] == 1.0
    assert metadata["probability_calibration"]["class_temperatures"]
    assert metadata["probability_calibration"]["comparison"]["applied_method"] == "classwise_temperature"
    assert metadata["probability_calibration"]["comparison"]["configured_method"] == "classwise_temperature"


def test_experiment_with_auto_calibration_applies_best_method(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv("IRIS_XGB_PROBABILITY_CALIBRATION_METHOD", "auto")

    from iris_bot.config import load_settings
    from iris_bot.experiments import run_experiment
    from iris_bot.processed_dataset import build_processed_dataset, write_processed_dataset

    local_settings = load_settings()
    object.__setattr__(local_settings.data, "raw_dir", raw_dir)
    object.__setattr__(local_settings.data, "processed_dir", processed_dir)
    object.__setattr__(local_settings.data, "runs_dir", runs_dir)
    object.__setattr__(local_settings.experiment, "_processed_dir", processed_dir)

    processed = build_processed_dataset(bars, local_settings.labeling)
    write_processed_dataset(
        processed,
        local_settings.experiment.processed_dataset_path,
        local_settings.experiment.processed_manifest_path,
        local_settings.experiment.processed_schema_path,
    )

    assert run_experiment(local_settings) == 0
    report_files = list(runs_dir.glob("*_experiment/experiment_report.json"))
    assert report_files
    payload = json.loads(report_files[0].read_text(encoding="utf-8"))
    comparison = payload["xgboost"]["probability_calibration"]["comparison"]
    assert comparison["configured_method"] == "auto"
    assert comparison["best_method"] in {"uncalibrated", "global_temperature", "classwise_temperature"}
    assert comparison["applied_method"] in {"global_temperature", "classwise_temperature"}
    assert comparison["applied_matches_best"] == (comparison["applied_method"] == comparison["best_method"])
    if comparison["best_method"] == "uncalibrated":
        assert comparison["applied_method"] == "global_temperature"

    model_metadata_files = list(runs_dir.glob("*_experiment/models/xgboost_metadata.json"))
    assert model_metadata_files
    metadata = json.loads(model_metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["probability_calibration"]["comparison"]["configured_method"] == "auto"


def test_experiment_with_significance_enabled_persists_report_and_trials_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setenv("IRIS_SIGNIFICANCE_ENABLED", "true")
    monkeypatch.setenv("IRIS_SIGNIFICANCE_TRIALS", "3")
    monkeypatch.setenv("IRIS_SIGNIFICANCE_SEED", "11")

    from iris_bot.config import load_settings
    from iris_bot.experiments import run_experiment
    from iris_bot.processed_dataset import build_processed_dataset, write_processed_dataset

    local_settings = load_settings()
    object.__setattr__(local_settings.data, "raw_dir", raw_dir)
    object.__setattr__(local_settings.data, "processed_dir", processed_dir)
    object.__setattr__(local_settings.data, "runs_dir", runs_dir)
    object.__setattr__(local_settings.experiment, "_processed_dir", processed_dir)

    processed = build_processed_dataset(bars, local_settings.labeling)
    write_processed_dataset(
        processed,
        local_settings.experiment.processed_dataset_path,
        local_settings.experiment.processed_manifest_path,
        local_settings.experiment.processed_schema_path,
    )

    assert run_experiment(local_settings) == 0

    report_files = list(runs_dir.glob("*_experiment/experiment_report.json"))
    assert report_files
    payload = json.loads(report_files[0].read_text(encoding="utf-8"))
    significance = payload["significance"]
    assert significance["enabled"] is True
    assert significance["status"] == "completed"
    assert significance["evaluation_mode"] == "walk_forward"
    assert significance["metric_name"] == "total_net_pnl_usd"
    assert significance["trials_completed"] == 3
    assert significance["trials_used_in_null_distribution"] == 3
    assert len(significance["trial_results"]) == 3
    assert significance["real_result"]["valid_folds"] >= 1
    assert significance["null_distribution_summary"]["count"] == 3
    assert "p_value" in significance
    assert "deflated_sharpe_ratio" in significance
    assert "status" in significance["deflated_sharpe_ratio"]

    trials_csv = list(runs_dir.glob("*_experiment/significance_trials.csv"))
    assert trials_csv
    lines = trials_csv[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4  # header + 3 trials
