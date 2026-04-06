from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from iris_bot.baselines import MomentumSignBaseline
from iris_bot.config import Settings
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.metrics import classification_metrics
from iris_bot.preprocessing import validate_feature_rows
from iris_bot.processed_dataset import ProcessedDataset, ProcessedRow, load_processed_dataset
from iris_bot.significance import run_walkforward_permutation_significance
from iris_bot.splits import temporal_train_validation_test_split
from iris_bot.thresholds import (
    apply_probability_threshold,
    apply_score_threshold,
    select_threshold_from_probabilities,
    select_threshold_from_scores,
)
from iris_bot.walk_forward import generate_walk_forward_windows
from iris_bot.xgb_model import XGBoostMultiClassModel


def _filter_rows(dataset: ProcessedDataset, settings: Settings) -> list[ProcessedRow]:
    rows = dataset.rows
    if settings.experiment.use_primary_timeframe_only:
        rows = [row for row in rows if row.timeframe == settings.trading.primary_timeframe]
    return rows


def _rows_to_feature_payload(rows: list[ProcessedRow], feature_names: list[str]) -> tuple[list[dict[str, float]], list[list[float]], list[int]]:
    feature_dicts = [row.features for row in rows]
    matrix = [[row.features[name] for name in feature_names] for row in rows]
    labels = [row.label for row in rows]
    validate_feature_rows(matrix)
    return feature_dicts, matrix, labels


def _write_predictions_csv(path: Path, rows: list[ProcessedRow], predictions: list[int], scores: list[float] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "symbol", "timeframe", "label", "prediction", "score"])
        for index, row in enumerate(rows):
            writer.writerow(
                [
                    row.timestamp.isoformat(),
                    row.symbol,
                    row.timeframe,
                    row.label,
                    predictions[index],
                    "" if scores is None else f"{scores[index]:.10f}",
                ]
            )


def _write_significance_trials_csv(path: Path, payload: dict[str, object]) -> None:
    trial_results = payload.get("trial_results", [])
    if not isinstance(trial_results, list):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial_index", "metric_value", "valid_folds", "skipped_folds"])
        for item in trial_results:
            if not isinstance(item, dict):
                continue
            writer.writerow(
                [
                    item.get("trial_index"),
                    item.get("metric_value"),
                    item.get("valid_folds"),
                    item.get("skipped_folds"),
                ]
            )


def _compute_economic_sample_weights(rows: list[ProcessedRow], cap: float = 3.0) -> list[float]:
    """Weight each training sample by ATR-relative economic significance.

    Bars with larger ATR (wider potential move) are more economically significant:
    correctly predicting direction there yields proportionally more P&L.  Weights are
    normalized so the median bar has weight 1.0, and capped at `cap` to prevent extreme
    outliers from dominating gradient updates.

    Falls back to uniform weights when ATR is unavailable or the median is zero.
    """
    atrs = [row.features.get("atr_5", 0.0) for row in rows]
    sorted_atrs = sorted(atrs)
    median_atr = sorted_atrs[len(sorted_atrs) // 2] if sorted_atrs else 0.0
    if median_atr <= 0.0:
        return [1.0] * len(rows)
    return [min(atr / median_atr, cap) for atr in atrs]


def run_experiment(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "experiment")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    dataset = load_processed_dataset(
        settings.experiment.processed_dataset_path,
        settings.experiment.processed_schema_path,
        settings.experiment.processed_manifest_path,
    )
    rows = _filter_rows(dataset, settings)
    if len(rows) < 30:
        logger.error("Insufficient processed rows for experiment: %s", len(rows))
        return 1

    split = temporal_train_validation_test_split(
        rows,
        settings.split.train_ratio,
        settings.split.validation_ratio,
        settings.split.test_ratio,
    )
    feature_names = dataset.feature_names
    train_feature_dicts, train_matrix, train_labels = _rows_to_feature_payload(split.train, feature_names)
    validation_feature_dicts, validation_matrix, validation_labels = _rows_to_feature_payload(split.validation, feature_names)
    _, test_matrix, test_labels = _rows_to_feature_payload(split.test, feature_names)

    baseline = MomentumSignBaseline()
    baseline_validation_scores = baseline.score(validation_feature_dicts)
    baseline_threshold = select_threshold_from_scores(
        scores=baseline_validation_scores,
        labels=validation_labels,
        grid=settings.threshold.grid,
        metric_name=settings.threshold.objective_metric,
        refinement_steps=settings.threshold.refinement_steps,
    )
    baseline_test_scores = baseline.score([row.features for row in split.test])
    baseline_predictions = apply_score_threshold(baseline_test_scores, baseline_threshold.threshold)
    baseline_metrics = classification_metrics(test_labels, baseline_predictions)

    economic_weights = _compute_economic_sample_weights(split.train)

    xgb_model = XGBoostMultiClassModel(settings.xgboost)
    try:
        xgb_model.fit(
            train_matrix,
            train_labels,
            validation_matrix,
            validation_labels,
            feature_names=feature_names,
            sample_weights=economic_weights,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        return 2

    validation_probabilities = xgb_model.predict_probabilities(validation_matrix)
    threshold_result = select_threshold_from_probabilities(
        probabilities=validation_probabilities,
        labels=validation_labels,
        grid=settings.threshold.grid,
        metric_name=settings.threshold.objective_metric,
        refinement_steps=settings.threshold.refinement_steps,
    )
    test_probabilities = xgb_model.predict_probabilities(test_matrix)
    xgb_predictions = apply_probability_threshold(test_probabilities, threshold_result.threshold)
    xgb_metrics = classification_metrics(test_labels, xgb_predictions)

    model_dir = run_dir / "models"
    xgb_model.save(model_dir / "xgboost_model.json", model_dir / "xgboost_metadata.json", feature_names)

    walk_forward_payload: list[dict[str, object]] = []
    if settings.walk_forward.enabled:
        windows = generate_walk_forward_windows(
            total_rows=len(rows),
            train_window=settings.walk_forward.train_window,
            validation_window=settings.walk_forward.validation_window,
            test_window=settings.walk_forward.test_window,
            step=settings.walk_forward.step,
        )
        for window in windows:
            train_rows = rows[window.train_start : window.train_end]
            validation_rows = rows[window.validation_start : window.validation_end]
            test_rows = rows[window.test_start : window.test_end]
            _, wf_train_matrix, wf_train_labels = _rows_to_feature_payload(train_rows, feature_names)
            wf_validation_dicts, wf_validation_matrix, wf_validation_labels = _rows_to_feature_payload(validation_rows, feature_names)
            _, wf_test_matrix, wf_test_labels = _rows_to_feature_payload(test_rows, feature_names)

            fold_baseline = MomentumSignBaseline()
            fold_validation_scores = fold_baseline.score(wf_validation_dicts)
            fold_baseline_threshold = select_threshold_from_scores(
                fold_validation_scores,
                wf_validation_labels,
                settings.threshold.grid,
                settings.threshold.objective_metric,
                refinement_steps=settings.threshold.refinement_steps,
            )
            fold_test_scores = fold_baseline.score([row.features for row in test_rows])
            fold_baseline_predictions = apply_score_threshold(fold_test_scores, fold_baseline_threshold.threshold)

            fold_model = XGBoostMultiClassModel(settings.xgboost)
            fold_model.fit(wf_train_matrix, wf_train_labels, wf_validation_matrix, wf_validation_labels)
            fold_validation_probabilities = fold_model.predict_probabilities(wf_validation_matrix)
            fold_threshold = select_threshold_from_probabilities(
                fold_validation_probabilities,
                wf_validation_labels,
                settings.threshold.grid,
                settings.threshold.objective_metric,
                refinement_steps=settings.threshold.refinement_steps,
            )
            fold_test_probabilities = fold_model.predict_probabilities(wf_test_matrix)
            fold_xgb_predictions = apply_probability_threshold(fold_test_probabilities, fold_threshold.threshold)
            walk_forward_payload.append(
                {
                    "window": window.to_dict(),
                    "baseline_threshold": asdict(fold_baseline_threshold),
                    "baseline_metrics": classification_metrics(wf_test_labels, fold_baseline_predictions),
                    "xgboost_threshold": asdict(fold_threshold),
                    "xgboost_metrics": classification_metrics(wf_test_labels, fold_xgb_predictions),
                }
            )

    significance_payload: dict[str, object] = {"enabled": False, "status": "disabled"}
    if settings.significance.enabled:
        if settings.walk_forward.enabled:
            significance_payload = run_walkforward_permutation_significance(
                rows=rows,
                feature_names=feature_names,
                settings=settings,
                logger=logger,
            )
        else:
            significance_payload = {
                "enabled": True,
                "status": "skipped",
                "reason": "significance requires walk_forward.enabled=True",
                "evaluation_mode": "walk_forward",
                "metric_name": settings.significance.metric_name,
            }

    if settings.experiment.save_predictions_csv:
        _write_predictions_csv(run_dir / "baseline_test_predictions.csv", split.test, baseline_predictions, baseline_test_scores)
        xgb_scores = [max(row.get(1, 0.0), row.get(-1, 0.0)) for row in test_probabilities]
        _write_predictions_csv(run_dir / "xgboost_test_predictions.csv", split.test, xgb_predictions, xgb_scores)
    if settings.significance.enabled:
        _write_significance_trials_csv(run_dir / "significance_trials.csv", significance_payload)

    write_json_report(
        run_dir,
        "experiment_report.json",
        {
            "dataset_manifest": dataset.manifest,
            "dataset_schema": dataset.schema,
            "feature_names": feature_names,
            "split_summary": [asdict(item) for item in split.summaries],
            "baseline": {
                "name": settings.experiment.benchmark_name,
                "threshold": asdict(baseline_threshold),
                "metrics": baseline_metrics,
            },
            "xgboost": {
                "threshold": asdict(threshold_result),
                "metrics": xgb_metrics,
                "best_iteration": xgb_model.best_iteration,
                "best_score": xgb_model.best_score,
                "class_weighting": {
                    "enabled": settings.xgboost.use_class_weights,
                    "max_multiplier": settings.xgboost.class_weight_max_multiplier,
                    "weights": {str(label): weight for label, weight in xgb_model.class_weights.items()},
                },
                "probability_calibration": xgb_model.probability_calibration_metadata(),
                "feature_importance": xgb_model.feature_importance(),
                "economic_sample_weights": {
                    "enabled": True,
                    "cap": 3.0,
                    "min": min(economic_weights),
                    "max": max(economic_weights),
                    "mean": sum(economic_weights) / len(economic_weights),
                },
            },
            "walk_forward": walk_forward_payload,
            "significance": significance_payload,
        },
    )
    write_json_report(
        run_dir,
        "experiment_config.json",
        {
            "labeling": asdict(settings.labeling),
            "split": asdict(settings.split),
            "walk_forward": asdict(settings.walk_forward),
            "threshold": asdict(settings.threshold),
            "xgboost": asdict(settings.xgboost),
            "significance": asdict(settings.significance),
        },
    )
    logger.info("experiment rows=%s feature_count=%s run_dir=%s", len(rows), len(feature_names), run_dir)
    return 0
