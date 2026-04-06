from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import pstdev
from typing import Any

from iris_bot.backtest import run_backtest_engine
from iris_bot.config import Settings, env_source
from iris_bot.data import load_bars
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.metrics import classification_metrics
from iris_bot.preprocessing import validate_feature_rows
from iris_bot.processed_dataset import ProcessedDataset, ProcessedRow, build_processed_dataset
from iris_bot.splits import TemporalSplit, temporal_train_validation_test_split
from iris_bot.thresholds import ThresholdSelectionResult, apply_probability_threshold, select_threshold_from_probabilities
from iris_bot.walk_forward import generate_walk_forward_windows
from iris_bot.xgb_model import LABEL_TO_CLASS, XGBoostMultiClassModel


_CONFIG_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("IRIS_LABEL_MODE", "labeling", "mode"),
    ("IRIS_LABEL_MIN_ABS_RETURN", "labeling", "min_abs_return"),
    ("IRIS_LABEL_HORIZON_BARS", "labeling", "horizon_bars"),
    ("IRIS_LABEL_TAKE_PROFIT_PCT", "labeling", "take_profit_pct"),
    ("IRIS_LABEL_STOP_LOSS_PCT", "labeling", "stop_loss_pct"),
    ("IRIS_LABEL_ALLOW_NO_TRADE", "labeling", "allow_no_trade"),
    ("IRIS_XGB_ENABLED", "xgboost", "enabled"),
    ("IRIS_XGB_NUM_BOOST_ROUND", "xgboost", "num_boost_round"),
    ("IRIS_XGB_EARLY_STOPPING_ROUNDS", "xgboost", "early_stopping_rounds"),
    ("IRIS_XGB_ETA", "xgboost", "eta"),
    ("IRIS_XGB_MAX_DEPTH", "xgboost", "max_depth"),
    ("IRIS_XGB_MIN_CHILD_WEIGHT", "xgboost", "min_child_weight"),
    ("IRIS_XGB_SUBSAMPLE", "xgboost", "subsample"),
    ("IRIS_XGB_COLSAMPLE_BYTREE", "xgboost", "colsample_bytree"),
    ("IRIS_XGB_REG_LAMBDA", "xgboost", "reg_lambda"),
    ("IRIS_XGB_REG_ALPHA", "xgboost", "reg_alpha"),
    ("IRIS_XGB_SEED", "xgboost", "seed"),
    ("IRIS_XGB_USE_CLASS_WEIGHTS", "xgboost", "use_class_weights"),
    ("IRIS_XGB_USE_PROBABILITY_CALIBRATION", "xgboost", "use_probability_calibration"),
    ("IRIS_XGB_PROBABILITY_CALIBRATION_METHOD", "xgboost", "probability_calibration_method"),
    ("IRIS_SIGNIFICANCE_ENABLED", "significance", "enabled"),
    ("IRIS_USE_PRIMARY_TIMEFRAME_ONLY", "experiment", "use_primary_timeframe_only"),
    ("IRIS_PRIMARY_TIMEFRAME", "trading", "primary_timeframe"),
)

_NEUTRAL_PROBABILITY = {-1: 0.0, 0: 1.0, 1: 0.0}
_MIN_SYMBOL_TRAIN_ROWS = 30
_MIN_SYMBOL_VALIDATION_ROWS = 10
_MIN_WF_TRAIN_ROWS = 30
_MIN_WF_VALIDATION_ROWS = 10
_MIN_WF_TEST_ROWS = 5


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    hypothesis: str
    label_overrides: dict[str, object] = field(default_factory=dict)
    xgb_overrides: dict[str, object] = field(default_factory=dict)
    model_mode: str = "global"
    selection_block: str = "primary"


@dataclass
class TrainedVariant:
    mode: str
    feature_names: list[str]
    threshold: float
    threshold_by_symbol: dict[str, float]
    model: XGBoostMultiClassModel | None = None
    models_by_symbol: dict[str, XGBoostMultiClassModel] = field(default_factory=dict)
    context_symbols: tuple[str, ...] = ()
    context_timeframes: tuple[str, ...] = ()
    skipped_symbols: dict[str, str] = field(default_factory=dict)
    training_summary: dict[str, object] = field(default_factory=dict)


def _read_env_file(project_root: Path) -> dict[str, str]:
    env_path = project_root / ".env"
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed = value.strip()
        if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {"'", '"'}:
            parsed = parsed[1:-1]
        values[key] = parsed
    return values


def _safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _distribution(labels: list[int]) -> dict[str, object]:
    counts = Counter(labels)
    total = len(labels)
    return {
        "count": total,
        "counts": {str(label): counts.get(label, 0) for label in (-1, 0, 1)},
        "ratios": {str(label): (counts.get(label, 0) / total if total else 0.0) for label in (-1, 0, 1)},
    }


def _prediction_distribution(predictions: list[int]) -> dict[str, object]:
    counts = Counter(predictions)
    total = len(predictions)
    no_trade_ratio = counts.get(0, 0) / total if total else 0.0
    return {
        "count": total,
        "counts": {str(label): counts.get(label, 0) for label in (-1, 0, 1)},
        "ratios": {str(label): (counts.get(label, 0) / total if total else 0.0) for label in (-1, 0, 1)},
        "no_trade_ratio": no_trade_ratio,
    }


def _economic_sample_weights(rows: list[ProcessedRow], cap: float = 3.0) -> list[float]:
    atrs = [row.features.get("atr_5", 0.0) for row in rows]
    sorted_atrs = sorted(atrs)
    median_atr = sorted_atrs[len(sorted_atrs) // 2] if sorted_atrs else 0.0
    if median_atr <= 0.0:
        return [1.0] * len(rows)
    return [min(atr / median_atr, cap) for atr in atrs]


def _multiclass_log_loss(probabilities: list[dict[int, float]], labels: list[int]) -> float | None:
    if not probabilities or len(probabilities) != len(labels):
        return None
    total = 0.0
    for row, label in zip(probabilities, labels, strict=False):
        klass = LABEL_TO_CLASS[label]
        prob = min(max(_safe_float(row.get(label, row.get(klass, 0.0)), 0.0), 1e-12), 1.0)
        total -= math.log(prob)
    return total / len(labels)


def _filter_rows(dataset: ProcessedDataset, settings: Settings) -> list[ProcessedRow]:
    rows = dataset.rows
    if settings.experiment.use_primary_timeframe_only:
        rows = [row for row in rows if row.timeframe == settings.trading.primary_timeframe]
    return sorted(rows, key=lambda row: (row.timestamp, row.symbol, row.timeframe))


def _context_feature_space(rows: list[ProcessedRow]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(sorted({row.symbol for row in rows})),
        tuple(sorted({row.timeframe for row in rows})),
    )


def _matrix_rows(
    rows: list[ProcessedRow],
    feature_names: list[str],
    *,
    mode: str,
    context_symbols: tuple[str, ...] = (),
    context_timeframes: tuple[str, ...] = (),
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        values = [row.features[name] for name in feature_names]
        if mode == "global_with_symbol_context":
            values.extend(1.0 if row.symbol == symbol else 0.0 for symbol in context_symbols)
            values.extend(1.0 if row.timeframe == timeframe else 0.0 for timeframe in context_timeframes)
        matrix.append(values)
    validate_feature_rows(matrix)
    return matrix


def _effective_feature_names(
    base_feature_names: list[str],
    *,
    mode: str,
    context_symbols: tuple[str, ...] = (),
    context_timeframes: tuple[str, ...] = (),
) -> list[str]:
    if mode != "global_with_symbol_context":
        return list(base_feature_names)
    return list(base_feature_names) + [f"context_symbol_{symbol}" for symbol in context_symbols] + [
        f"context_timeframe_{timeframe}" for timeframe in context_timeframes
    ]


def _fit_global_variant(
    train_rows: list[ProcessedRow],
    validation_rows: list[ProcessedRow],
    settings: Settings,
    base_feature_names: list[str],
    *,
    mode: str,
) -> TrainedVariant:
    context_symbols, context_timeframes = _context_feature_space(train_rows + validation_rows)
    effective_feature_names = _effective_feature_names(
        base_feature_names,
        mode=mode,
        context_symbols=context_symbols,
        context_timeframes=context_timeframes,
    )
    train_matrix = _matrix_rows(
        train_rows,
        base_feature_names,
        mode=mode,
        context_symbols=context_symbols,
        context_timeframes=context_timeframes,
    )
    validation_matrix = _matrix_rows(
        validation_rows,
        base_feature_names,
        mode=mode,
        context_symbols=context_symbols,
        context_timeframes=context_timeframes,
    )
    train_labels = [row.label for row in train_rows]
    validation_labels = [row.label for row in validation_rows]
    model = XGBoostMultiClassModel(settings.xgboost)
    model.fit(
        train_matrix,
        train_labels,
        validation_matrix,
        validation_labels,
        feature_names=effective_feature_names,
        sample_weights=_economic_sample_weights(train_rows),
    )
    validation_probabilities = model.predict_probabilities(validation_matrix)
    threshold = select_threshold_from_probabilities(
        probabilities=validation_probabilities,
        labels=validation_labels,
        grid=settings.threshold.grid,
        metric_name=settings.threshold.objective_metric,
        refinement_steps=settings.threshold.refinement_steps,
    )
    return TrainedVariant(
        mode=mode,
        model=model,
        feature_names=effective_feature_names,
        threshold=threshold.threshold,
        threshold_by_symbol={},
        context_symbols=context_symbols,
        context_timeframes=context_timeframes,
        training_summary={
            "best_iteration": model.best_iteration,
            "best_score": model.best_score,
            "threshold": asdict(threshold),
            "probability_calibration": model.probability_calibration_metadata(),
            "class_weighting": {
                "enabled": settings.xgboost.use_class_weights,
                "weights": {str(label): weight for label, weight in model.class_weights.items()},
            },
        },
    )


def _fit_symbol_specific_variant(
    train_rows: list[ProcessedRow],
    validation_rows: list[ProcessedRow],
    settings: Settings,
    base_feature_names: list[str],
) -> TrainedVariant:
    models_by_symbol: dict[str, XGBoostMultiClassModel] = {}
    threshold_by_symbol: dict[str, float] = {}
    per_symbol: dict[str, dict[str, object]] = {}
    skipped_symbols: dict[str, str] = {}
    for symbol in sorted({row.symbol for row in train_rows + validation_rows}):
        train_subset = [row for row in train_rows if row.symbol == symbol]
        validation_subset = [row for row in validation_rows if row.symbol == symbol]
        if len(train_subset) < _MIN_SYMBOL_TRAIN_ROWS:
            skipped_symbols[symbol] = f"train_rows={len(train_subset)} < {_MIN_SYMBOL_TRAIN_ROWS}"
            continue
        if len(validation_subset) < _MIN_SYMBOL_VALIDATION_ROWS:
            skipped_symbols[symbol] = f"validation_rows={len(validation_subset)} < {_MIN_SYMBOL_VALIDATION_ROWS}"
            continue
        train_matrix = _matrix_rows(train_subset, base_feature_names, mode="global")
        validation_matrix = _matrix_rows(validation_subset, base_feature_names, mode="global")
        train_labels = [row.label for row in train_subset]
        validation_labels = [row.label for row in validation_subset]
        model = XGBoostMultiClassModel(settings.xgboost)
        model.fit(
            train_matrix,
            train_labels,
            validation_matrix,
            validation_labels,
            feature_names=base_feature_names,
            sample_weights=_economic_sample_weights(train_subset),
        )
        validation_probabilities = model.predict_probabilities(validation_matrix)
        threshold = select_threshold_from_probabilities(
            probabilities=validation_probabilities,
            labels=validation_labels,
            grid=settings.threshold.grid,
            metric_name=settings.threshold.objective_metric,
            refinement_steps=settings.threshold.refinement_steps,
        )
        models_by_symbol[symbol] = model
        threshold_by_symbol[symbol] = threshold.threshold
        per_symbol[symbol] = {
            "train_rows": len(train_subset),
            "validation_rows": len(validation_subset),
            "best_iteration": model.best_iteration,
            "best_score": model.best_score,
            "threshold": asdict(threshold),
            "probability_calibration": model.probability_calibration_metadata(),
        }
    best_iterations = [payload["best_iteration"] for payload in per_symbol.values() if payload.get("best_iteration") is not None]
    best_scores = [payload["best_score"] for payload in per_symbol.values() if payload.get("best_score") is not None]
    return TrainedVariant(
        mode="symbol_specific",
        feature_names=list(base_feature_names),
        threshold=0.0,
        threshold_by_symbol=threshold_by_symbol,
        models_by_symbol=models_by_symbol,
        skipped_symbols=skipped_symbols,
        training_summary={
            "per_symbol": per_symbol,
            "skipped_symbols": skipped_symbols,
            "aggregate": {
                "trained_symbol_count": len(models_by_symbol),
                "skipped_symbol_count": len(skipped_symbols),
                "mean_best_iteration": sum(best_iterations) / len(best_iterations) if best_iterations else None,
                "max_best_iteration": max(best_iterations) if best_iterations else None,
                "mean_best_score": sum(best_scores) / len(best_scores) if best_scores else None,
                "min_best_score": min(best_scores) if best_scores else None,
            },
        },
    )


def _fit_variant(
    train_rows: list[ProcessedRow],
    validation_rows: list[ProcessedRow],
    settings: Settings,
    base_feature_names: list[str],
    *,
    mode: str,
) -> TrainedVariant:
    if mode == "symbol_specific":
        return _fit_symbol_specific_variant(train_rows, validation_rows, settings, base_feature_names)
    return _fit_global_variant(train_rows, validation_rows, settings, base_feature_names, mode=mode)


def _predict_probabilities(trained: TrainedVariant, rows: list[ProcessedRow], base_feature_names: list[str]) -> list[dict[int, float]]:
    if trained.mode == "symbol_specific":
        probabilities: list[dict[int, float]] = []
        for row in rows:
            model = trained.models_by_symbol.get(row.symbol)
            if model is None:
                probabilities.append(dict(_NEUTRAL_PROBABILITY))
                continue
            matrix = _matrix_rows([row], base_feature_names, mode="global")
            probabilities.append(model.predict_probabilities(matrix)[0])
        return probabilities
    if trained.model is None:
        raise RuntimeError("Global variant model is missing")
    matrix = _matrix_rows(
        rows,
        base_feature_names,
        mode=trained.mode,
        context_symbols=trained.context_symbols,
        context_timeframes=trained.context_timeframes,
    )
    return trained.model.predict_probabilities(matrix)


def _apply_thresholds(trained: TrainedVariant, rows: list[ProcessedRow], probabilities: list[dict[int, float]]) -> list[int]:
    if trained.threshold_by_symbol:
        predictions: list[int] = []
        for row, probability in zip(rows, probabilities, strict=False):
            predictions.append(
                apply_probability_threshold([probability], trained.threshold_by_symbol.get(row.symbol, 1.01))[0]
            )
        return predictions
    return apply_probability_threshold(probabilities, trained.threshold)


def _group_classification(rows: list[ProcessedRow], predictions: list[int], group_key: str) -> dict[str, object]:
    grouped_rows: dict[str, list[int]] = defaultdict(list)
    grouped_preds: dict[str, list[int]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=False):
        key = getattr(row, group_key)
        grouped_rows[key].append(row.label)
        grouped_preds[key].append(prediction)
    return {
        key: {
            "row_count": len(grouped_rows[key]),
            "label_distribution": _distribution(grouped_rows[key]),
            "prediction_distribution": _prediction_distribution(grouped_preds[key]),
            "classification_metrics": classification_metrics(grouped_rows[key], grouped_preds[key]),
        }
        for key in sorted(grouped_rows)
    }


def _group_economic(
    rows: list[ProcessedRow],
    probabilities: list[dict[int, float]],
    settings: Settings,
    trained: TrainedVariant,
    group_key: str,
) -> dict[str, object]:
    grouped_rows: dict[str, list[ProcessedRow]] = defaultdict(list)
    grouped_probabilities: dict[str, list[dict[int, float]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=False):
        key = getattr(row, group_key)
        grouped_rows[key].append(row)
        grouped_probabilities[key].append(probability)
    payload: dict[str, object] = {}
    for key in sorted(grouped_rows):
        threshold_by_symbol: dict[str, float] | None = None
        if trained.threshold_by_symbol:
            symbols = {row.symbol for row in grouped_rows[key]}
            threshold_by_symbol = {
                symbol: trained.threshold_by_symbol.get(symbol, 1.01)
                for symbol in symbols
            }
        metrics, _, _ = run_backtest_engine(
            rows=grouped_rows[key],
            probabilities=grouped_probabilities[key],
            threshold=trained.threshold,
            backtest=settings.backtest,
            risk=settings.risk,
            intrabar_policy=settings.backtest.intrabar_policy,
            exit_policy_config=settings.exit_policy,
            dynamic_exit_config=settings.dynamic_exits,
            threshold_by_symbol=threshold_by_symbol,
        )
        payload[key] = metrics
    return payload


def _evaluate_split(
    split_name: str,
    rows: list[ProcessedRow],
    probabilities: list[dict[int, float]],
    predictions: list[int],
    settings: Settings,
    trained: TrainedVariant,
) -> dict[str, object]:
    labels = [row.label for row in rows]
    threshold_by_symbol = trained.threshold_by_symbol or None
    economic_metrics, _, _ = run_backtest_engine(
        rows=rows,
        probabilities=probabilities,
        threshold=trained.threshold,
        backtest=settings.backtest,
        risk=settings.risk,
        intrabar_policy=settings.backtest.intrabar_policy,
        exit_policy_config=settings.exit_policy,
        dynamic_exit_config=settings.dynamic_exits,
        threshold_by_symbol=threshold_by_symbol,
    )
    return {
        "split_name": split_name,
        "row_count": len(rows),
        "label_distribution": _distribution(labels),
        "prediction_distribution": _prediction_distribution(predictions),
        "classification_metrics": classification_metrics(labels, predictions),
        "log_loss": _multiclass_log_loss(probabilities, labels),
        "economic_metrics": economic_metrics,
        "per_symbol": {
            "classification": _group_classification(rows, predictions, "symbol"),
            "economic": _group_economic(rows, probabilities, settings, trained, "symbol"),
        },
        "per_timeframe": {
            "classification": _group_classification(rows, predictions, "timeframe"),
            "economic": _group_economic(rows, probabilities, settings, trained, "timeframe"),
        },
    }


def _variant_collapse_report(result: dict[str, object]) -> dict[str, object]:
    split_validation = result["splits"]["validation"]  # type: ignore[index]
    validation_pred = split_validation["prediction_distribution"]  # type: ignore[index]
    dominant_ratio = max(validation_pred["ratios"].values()) if validation_pred["ratios"] else 0.0  # type: ignore[index]
    training_summary = result["training_summary"]  # type: ignore[index]
    best_iteration = training_summary.get("best_iteration")
    if best_iteration is None:
        best_iteration = ((training_summary.get("aggregate") or {}).get("mean_best_iteration"))
    detected = bool(best_iteration == 0 or dominant_ratio >= 0.90)
    return {
        "detected": detected,
        "best_iteration": best_iteration,
        "dominant_prediction_ratio_validation": dominant_ratio,
        "reason": "best_iteration_zero_or_prediction_collapse" if detected else "no_collapse_detected",
    }


def _evaluate_walk_forward(
    rows: list[ProcessedRow],
    base_feature_names: list[str],
    settings: Settings,
    spec: ExperimentSpec,
) -> dict[str, object]:
    windows = generate_walk_forward_windows(
        total_rows=len(rows),
        train_window=settings.walk_forward.train_window,
        validation_window=settings.walk_forward.validation_window,
        test_window=settings.walk_forward.test_window,
        step=settings.walk_forward.step,
    )
    fold_summaries: list[dict[str, object]] = []
    for window in windows:
        train_rows = rows[window.train_start : window.train_end]
        validation_rows = rows[window.validation_start : window.validation_end]
        test_rows = rows[window.test_start : window.test_end]
        if len(train_rows) < _MIN_WF_TRAIN_ROWS or len(validation_rows) < _MIN_WF_VALIDATION_ROWS or len(test_rows) < _MIN_WF_TEST_ROWS:
            fold_summaries.append(
                {
                    "fold_index": window.fold_index,
                    "skipped": True,
                    "reason": "insufficient_rows",
                    "train_rows": len(train_rows),
                    "validation_rows": len(validation_rows),
                    "test_rows": len(test_rows),
                }
            )
            continue
        trained = _fit_variant(train_rows, validation_rows, settings, base_feature_names, mode=spec.model_mode)
        probabilities = _predict_probabilities(trained, test_rows, base_feature_names)
        predictions = _apply_thresholds(trained, test_rows, probabilities)
        split_report = _evaluate_split("test", test_rows, probabilities, predictions, settings, trained)
        economic = split_report["economic_metrics"]  # type: ignore[index]
        fold_summaries.append(
            {
                "fold_index": window.fold_index,
                "skipped": False,
                "threshold": trained.threshold,
                "threshold_by_symbol": trained.threshold_by_symbol,
                "trade_count": economic["total_trades"],
                "net_pnl_usd": economic["net_pnl_usd"],
                "profit_factor": economic["profit_factor"],
                "max_drawdown_usd": economic["max_drawdown_usd"],
                "expectancy_usd": economic["expectancy_usd"],
                "no_trade_ratio": split_report["prediction_distribution"]["no_trade_ratio"],  # type: ignore[index]
                "macro_f1": split_report["classification_metrics"]["macro_f1"],  # type: ignore[index]
                "best_iteration": trained.training_summary.get("best_iteration")
                if "best_iteration" in trained.training_summary
                else (trained.training_summary.get("aggregate") or {}).get("mean_best_iteration"),
            }
        )
    valid_folds = [fold for fold in fold_summaries if not fold.get("skipped")]
    net_pnls = [_safe_float(fold.get("net_pnl_usd")) for fold in valid_folds]
    profit_factors = [_safe_float(fold.get("profit_factor")) for fold in valid_folds]
    drawdowns = [_safe_float(fold.get("max_drawdown_usd")) for fold in valid_folds]
    expectancies = [_safe_float(fold.get("expectancy_usd")) for fold in valid_folds]
    no_trade_ratios = [_safe_float(fold.get("no_trade_ratio")) for fold in valid_folds]
    return {
        "total_folds": len(fold_summaries),
        "valid_folds": len(valid_folds),
        "skipped_folds": len(fold_summaries) - len(valid_folds),
        "fold_summaries": fold_summaries,
        "aggregate": {
            "total_net_pnl_usd": sum(net_pnls),
            "mean_net_pnl_usd": (sum(net_pnls) / len(net_pnls)) if net_pnls else 0.0,
            "net_pnl_usd_stddev": pstdev(net_pnls) if len(net_pnls) > 1 else 0.0,
            "mean_profit_factor": (sum(profit_factors) / len(profit_factors)) if profit_factors else 0.0,
            "profit_factor_stddev": pstdev(profit_factors) if len(profit_factors) > 1 else 0.0,
            "worst_fold_drawdown_usd": max(drawdowns) if drawdowns else 0.0,
            "mean_expectancy_usd": (sum(expectancies) / len(expectancies)) if expectancies else 0.0,
            "expectancy_stddev": pstdev(expectancies) if len(expectancies) > 1 else 0.0,
            "mean_no_trade_ratio": (sum(no_trade_ratios) / len(no_trade_ratios)) if no_trade_ratios else 0.0,
            "no_trade_ratio_stddev": pstdev(no_trade_ratios) if len(no_trade_ratios) > 1 else 0.0,
            "positive_folds": sum(1 for value in net_pnls if value > 0.0),
            "negative_folds": sum(1 for value in net_pnls if value < 0.0),
        },
    }


def _build_dataset_for_spec(settings: Settings, spec: ExperimentSpec) -> tuple[Settings, ProcessedDataset]:
    effective_settings = settings
    if spec.label_overrides:
        effective_settings = replace(effective_settings, labeling=replace(effective_settings.labeling, **spec.label_overrides))
    if spec.xgb_overrides:
        effective_settings = replace(effective_settings, xgboost=replace(effective_settings.xgboost, **spec.xgb_overrides))
    bars = load_bars(effective_settings.data.raw_dataset_path)
    if not bars:
        raise FileNotFoundError(f"Raw dataset not found: {effective_settings.data.raw_dataset_path}")
    dataset = build_processed_dataset(bars, effective_settings.labeling)
    return effective_settings, dataset


def _label_reason_distribution(rows: list[ProcessedRow]) -> dict[str, int]:
    counts = Counter(row.label_reason for row in rows)
    return dict(sorted(counts.items()))


def _dataset_diagnostics(dataset: ProcessedDataset, rows: list[ProcessedRow]) -> dict[str, object]:
    by_symbol: dict[str, dict[str, object]] = {}
    by_timeframe: dict[str, dict[str, object]] = {}
    for symbol in sorted({row.symbol for row in rows}):
        symbol_rows = [row for row in rows if row.symbol == symbol]
        by_symbol[symbol] = {
            "row_count": len(symbol_rows),
            "label_distribution": _distribution([row.label for row in symbol_rows]),
            "label_reasons": _label_reason_distribution(symbol_rows),
        }
    for timeframe in sorted({row.timeframe for row in rows}):
        timeframe_rows = [row for row in rows if row.timeframe == timeframe]
        by_timeframe[timeframe] = {
            "row_count": len(timeframe_rows),
            "label_distribution": _distribution([row.label for row in timeframe_rows]),
            "label_reasons": _label_reason_distribution(timeframe_rows),
        }
    return {
        "manifest": dataset.manifest,
        "schema": dataset.schema,
        "filtered_row_count": len(rows),
        "feature_count": len(dataset.feature_names),
        "has_explicit_symbol_feature": any("symbol" in name.lower() for name in dataset.feature_names),
        "has_explicit_timeframe_feature": any("timeframe" in name.lower() for name in dataset.feature_names),
        "label_reasons": _label_reason_distribution(rows),
        "by_symbol": by_symbol,
        "by_timeframe": by_timeframe,
    }


def _evaluate_spec(settings: Settings, spec: ExperimentSpec, run_dir: Path) -> dict[str, object]:
    effective_settings, dataset = _build_dataset_for_spec(settings, spec)
    rows = _filter_rows(dataset, effective_settings)
    split = temporal_train_validation_test_split(
        rows,
        effective_settings.split.train_ratio,
        effective_settings.split.validation_ratio,
        effective_settings.split.test_ratio,
    )
    trained = _fit_variant(split.train, split.validation, effective_settings, dataset.feature_names, mode=spec.model_mode)
    split_reports: dict[str, object] = {}
    for split_name, split_rows in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        probabilities = _predict_probabilities(trained, split_rows, dataset.feature_names)
        predictions = _apply_thresholds(trained, split_rows, probabilities)
        split_reports[split_name] = _evaluate_split(split_name, split_rows, probabilities, predictions, effective_settings, trained)
    walk_forward = _evaluate_walk_forward(rows, dataset.feature_names, effective_settings, spec)
    result = {
        "experiment_id": spec.experiment_id,
        "hypothesis": spec.hypothesis,
        "model_mode": spec.model_mode,
        "selection_block": spec.selection_block,
        "effective_config": {
            "labeling": asdict(effective_settings.labeling),
            "xgboost": asdict(effective_settings.xgboost),
            "split": asdict(effective_settings.split),
            "walk_forward": asdict(effective_settings.walk_forward),
            "threshold": asdict(effective_settings.threshold),
            "backtest": asdict(effective_settings.backtest),
            "experiment": {
                "use_primary_timeframe_only": effective_settings.experiment.use_primary_timeframe_only,
                "primary_timeframe": effective_settings.trading.primary_timeframe,
            },
        },
        "dataset": _dataset_diagnostics(dataset, rows),
        "split_summary": [asdict(item) for item in split.summaries],
        "training_summary": trained.training_summary,
        "splits": split_reports,
        "walk_forward": walk_forward,
    }
    result["collapse_diagnostics"] = _variant_collapse_report(result)
    experiment_dir = run_dir / spec.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json_report(experiment_dir, "experiment_result.json", result)
    return result


def audit_effective_config_payload(settings: Settings) -> dict[str, object]:
    env_file_values = _read_env_file(settings.project_root)
    resolved: dict[str, object] = {}
    for env_name, section_name, field_name in _CONFIG_FIELDS:
        section = getattr(settings, section_name)
        resolved[env_name] = {
            "section": section_name,
            "field": field_name,
            "effective_value": getattr(section, field_name),
            "source": env_source(env_name),
            "in_process_env": env_name in os.environ,
            "in_project_env_file": env_name in env_file_values,
            "project_env_file_value": env_file_values.get(env_name),
            "process_env_value": os.environ.get(env_name),
        }
    return {
        "runtime_config": {
            "labeling": asdict(settings.labeling),
            "xgboost": asdict(settings.xgboost),
            "significance": asdict(settings.significance),
            "split": asdict(settings.split),
            "walk_forward": asdict(settings.walk_forward),
            "threshold": asdict(settings.threshold),
            "experiment": {
                "use_primary_timeframe_only": settings.experiment.use_primary_timeframe_only,
                "processed_dataset_path": str(settings.experiment.processed_dataset_path),
            },
            "trading": {
                "symbols": list(settings.trading.symbols),
                "timeframes": list(settings.trading.timeframes),
                "primary_timeframe": settings.trading.primary_timeframe,
            },
        },
        "tracked_variables": resolved,
    }


def audit_symbol_context_payload(settings: Settings) -> dict[str, object]:
    baseline_settings, dataset = _build_dataset_for_spec(settings, _baseline_spec())
    rows = _filter_rows(dataset, baseline_settings)
    split = temporal_train_validation_test_split(
        rows,
        baseline_settings.split.train_ratio,
        baseline_settings.split.validation_ratio,
        baseline_settings.split.test_ratio,
    )
    series_counter = Counter((row.symbol, row.timeframe) for row in rows)
    return {
        "model_sees_explicit_symbol_feature": any("symbol" in name.lower() for name in dataset.feature_names),
        "model_sees_explicit_timeframe_feature": any("timeframe" in name.lower() for name in dataset.feature_names),
        "training_is_global_mixed_across_symbols": len({row.symbol for row in split.train}) > 1,
        "training_is_global_mixed_across_timeframes": len({row.timeframe for row in split.train}) > 1,
        "primary_timeframe_only": baseline_settings.experiment.use_primary_timeframe_only,
        "primary_timeframe": baseline_settings.trading.primary_timeframe,
        "row_count": len(rows),
        "symbols": sorted({row.symbol for row in rows}),
        "timeframes": sorted({row.timeframe for row in rows}),
        "series_counts": {
            f"{symbol}:{timeframe}": count for (symbol, timeframe), count in sorted(series_counter.items())
        },
        "per_symbol_rows": dict(sorted(Counter(row.symbol for row in rows).items())),
        "per_timeframe_rows": dict(sorted(Counter(row.timeframe for row in rows).items())),
        "split_symbol_balance": {
            split_name: dict(sorted(Counter(row.symbol for row in split_rows).items()))
            for split_name, split_rows in (("train", split.train), ("validation", split.validation), ("test", split.test))
        },
        "split_timeframe_balance": {
            split_name: dict(sorted(Counter(row.timeframe for row in split_rows).items()))
            for split_name, split_rows in (("train", split.train), ("validation", split.validation), ("test", split.test))
        },
        "dominant_diagnosis": {
            "symbol_mixing_without_explicit_context": True,
            "timeframe_context_absent_in_model_features": True,
            "timeframe_mixing_active_in_current_training": False,
            "summary": (
                "Rows are globally mixed across symbols inside each temporal split, "
                "but the feature vector has no explicit symbol or timeframe identifiers."
            ),
        },
    }


def _baseline_spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="exp0_baseline",
        hypothesis="Baseline product configuration with no experimental overrides.",
    )


def _primary_specs() -> list[ExperimentSpec]:
    return [
        _baseline_spec(),
        ExperimentSpec(
            experiment_id="exp1_config_only",
            hypothesis="Only config plumbing/calibration changes help once runtime config is applied cleanly.",
            xgb_overrides={"probability_calibration_method": "auto"},
            selection_block="config",
        ),
        ExperimentSpec(
            experiment_id="exp2_labels_asymmetric",
            hypothesis="Asymmetric triple-barrier labels improve signal quality without tuning changes.",
            label_overrides={"take_profit_pct": 0.0030, "stop_loss_pct": 0.0015},
            selection_block="labels",
        ),
        ExperimentSpec(
            experiment_id="exp3_xgb_mcw_5",
            hypothesis="Moderate XGBoost tuning with min_child_weight=5 improves learning without overfitting.",
            xgb_overrides={
                "num_boost_round": 300,
                "early_stopping_rounds": 30,
                "eta": 0.05,
                "min_child_weight": 5.0,
            },
            selection_block="xgb_tuning",
        ),
        ExperimentSpec(
            experiment_id="exp3_xgb_mcw_8",
            hypothesis="Moderate XGBoost tuning with min_child_weight=8 improves learning without overfitting.",
            xgb_overrides={
                "num_boost_round": 300,
                "early_stopping_rounds": 30,
                "eta": 0.05,
                "min_child_weight": 8.0,
            },
            selection_block="xgb_tuning",
        ),
        ExperimentSpec(
            experiment_id="exp3_xgb_mcw_20",
            hypothesis="Heavy child-weight regularization with min_child_weight=20 improves robustness.",
            xgb_overrides={
                "num_boost_round": 300,
                "early_stopping_rounds": 30,
                "eta": 0.05,
                "min_child_weight": 20.0,
            },
            selection_block="xgb_tuning",
        ),
        ExperimentSpec(
            experiment_id="exp4_symbol_context",
            hypothesis="Adding explicit symbol/timeframe context to the global model resolves structural mixing.",
            model_mode="global_with_symbol_context",
            selection_block="symbol_context",
        ),
        ExperimentSpec(
            experiment_id="exp4_symbol_specific",
            hypothesis="Training one model per symbol resolves structural mixing better than a single global model.",
            model_mode="symbol_specific",
            selection_block="symbol_context",
        ),
    ]


def _selection_score(result: dict[str, object], baseline: dict[str, object]) -> tuple[float, float, float, float]:
    test_metrics = result["splits"]["test"]["economic_metrics"]  # type: ignore[index]
    baseline_test_metrics = baseline["splits"]["test"]["economic_metrics"]  # type: ignore[index]
    walk_forward = result["walk_forward"]["aggregate"]  # type: ignore[index]
    baseline_walk_forward = baseline["walk_forward"]["aggregate"]  # type: ignore[index]
    collapse = result["collapse_diagnostics"]  # type: ignore[index]
    penalty = -1.0 if collapse["detected"] else 0.0
    return (
        penalty + (_safe_float(walk_forward.get("total_net_pnl_usd")) - _safe_float(baseline_walk_forward.get("total_net_pnl_usd"))),
        _safe_float(test_metrics.get("profit_factor")) - _safe_float(baseline_test_metrics.get("profit_factor")),
        _safe_float(test_metrics.get("net_pnl_usd")) - _safe_float(baseline_test_metrics.get("net_pnl_usd")),
        -(_safe_float(test_metrics.get("max_drawdown_usd")) - _safe_float(baseline_test_metrics.get("max_drawdown_usd"))),
    )


def _select_best_result(
    results: dict[str, dict[str, object]],
    baseline: dict[str, object],
    block: str,
) -> dict[str, object] | None:
    candidates = [result for result in results.values() if result.get("selection_block") == block]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: _selection_score(item, baseline), reverse=True)[0]


def _combined_spec(base_id: str, best_result: dict[str, object], *, combine: str) -> ExperimentSpec:
    effective = best_result["effective_config"]  # type: ignore[index]
    label_overrides: dict[str, object] = {}
    xgb_overrides: dict[str, object] = {}
    model_mode = "global"
    if combine in {"labels_xgb", "labels_context"}:
        label_overrides = {
            "take_profit_pct": 0.0030,
            "stop_loss_pct": 0.0015,
        }
    if combine == "labels_xgb":
        xgb = effective["xgboost"]  # type: ignore[index]
        xgb_overrides = {
            "num_boost_round": xgb["num_boost_round"],
            "early_stopping_rounds": xgb["early_stopping_rounds"],
            "eta": xgb["eta"],
            "min_child_weight": xgb["min_child_weight"],
        }
        hypothesis = "Labels asimétricos combinados con la mejor variante aislada de tuning XGBoost."
    else:
        model_mode = str(best_result["model_mode"])
        hypothesis = "Labels asimétricos combinados con la mejor variante aislada de contexto por símbolo."
    return ExperimentSpec(
        experiment_id=base_id,
        hypothesis=hypothesis,
        label_overrides=label_overrides,
        xgb_overrides=xgb_overrides,
        model_mode=model_mode,
        selection_block="combined",
    )


def _delta_vs_baseline(result: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    delta: dict[str, object] = {}
    for split_name in ("validation", "test"):
        current_split = result["splits"][split_name]  # type: ignore[index]
        baseline_split = baseline["splits"][split_name]  # type: ignore[index]
        current_classification = current_split["classification_metrics"]  # type: ignore[index]
        baseline_classification = baseline_split["classification_metrics"]  # type: ignore[index]
        current_economic = current_split["economic_metrics"]  # type: ignore[index]
        baseline_economic = baseline_split["economic_metrics"]  # type: ignore[index]
        delta[split_name] = {
            "macro_f1": _safe_float(current_classification.get("macro_f1")) - _safe_float(baseline_classification.get("macro_f1")),
            "balanced_accuracy": _safe_float(current_classification.get("balanced_accuracy")) - _safe_float(baseline_classification.get("balanced_accuracy")),
            "directional_precision": _safe_float(current_classification.get("directional_precision")) - _safe_float(baseline_classification.get("directional_precision")),
            "log_loss": (_safe_float(current_split.get("log_loss")) - _safe_float(baseline_split.get("log_loss"))),
            "net_pnl_usd": _safe_float(current_economic.get("net_pnl_usd")) - _safe_float(baseline_economic.get("net_pnl_usd")),
            "profit_factor": _safe_float(current_economic.get("profit_factor")) - _safe_float(baseline_economic.get("profit_factor")),
            "expectancy_usd": _safe_float(current_economic.get("expectancy_usd")) - _safe_float(baseline_economic.get("expectancy_usd")),
            "max_drawdown_usd": _safe_float(current_economic.get("max_drawdown_usd")) - _safe_float(baseline_economic.get("max_drawdown_usd")),
            "no_trade_ratio": _safe_float(current_split["prediction_distribution"].get("no_trade_ratio"))
            - _safe_float(baseline_split["prediction_distribution"].get("no_trade_ratio")),
        }
    current_wf = result["walk_forward"]["aggregate"]  # type: ignore[index]
    baseline_wf = baseline["walk_forward"]["aggregate"]  # type: ignore[index]
    delta["walk_forward"] = {
        "total_net_pnl_usd": _safe_float(current_wf.get("total_net_pnl_usd")) - _safe_float(baseline_wf.get("total_net_pnl_usd")),
        "mean_profit_factor": _safe_float(current_wf.get("mean_profit_factor")) - _safe_float(baseline_wf.get("mean_profit_factor")),
        "worst_fold_drawdown_usd": _safe_float(current_wf.get("worst_fold_drawdown_usd")) - _safe_float(baseline_wf.get("worst_fold_drawdown_usd")),
        "mean_no_trade_ratio": _safe_float(current_wf.get("mean_no_trade_ratio")) - _safe_float(baseline_wf.get("mean_no_trade_ratio")),
    }
    return delta


def _recommendation(results: dict[str, dict[str, object]], baseline: dict[str, object]) -> dict[str, object]:
    ranked = sorted(
        (result for result in results.values() if result["experiment_id"] != baseline["experiment_id"]),
        key=lambda item: _selection_score(item, baseline),
        reverse=True,
    )
    promote_candidates: list[dict[str, object]] = []
    for result in ranked:
        delta = _delta_vs_baseline(result, baseline)
        if result["collapse_diagnostics"]["detected"]:  # type: ignore[index]
            continue
        if delta["test"]["net_pnl_usd"] <= 0.0:  # type: ignore[index]
            continue
        if delta["walk_forward"]["total_net_pnl_usd"] <= 0.0:  # type: ignore[index]
            continue
        if delta["test"]["profit_factor"] < 0.0:  # type: ignore[index]
            continue
        if delta["test"]["max_drawdown_usd"] > 0.0:  # type: ignore[index]
            continue
        promote_candidates.append(result)

    best = promote_candidates[0] if promote_candidates else None
    if best is None:
        context_best = _select_best_result(results, baseline, "symbol_context")
        if context_best and _selection_score(context_best, baseline)[0] > _selection_score(baseline, baseline)[0]:
            decision = "REQUIRE_STRUCTURAL_REWORK"
            rationale = "Only symbol-context variants show meaningful upside; baseline and tuning-only variants are not robust."
        else:
            decision = "KEEP_BASELINE"
            rationale = "No experiment demonstrated a robust improvement over baseline under conservative promotion rules."
    else:
        experiment_id = best["experiment_id"]
        if experiment_id == "exp1_config_only":
            decision = "ADOPT_CONFIG_FIX_ONLY"
        elif experiment_id == "exp2_labels_asymmetric":
            decision = "ADOPT_LABEL_CHANGE"
        elif experiment_id.startswith("exp3_"):
            decision = "ADOPT_XGB_TUNING"
        elif experiment_id.startswith("exp4_"):
            decision = "ADOPT_SYMBOL_CONTEXT_CHANGE"
        else:
            decision = "ADOPT_COMBINED_CHANGE"
        rationale = f"Experiment {experiment_id} improved test and walk-forward economics without worsening drawdown."
    return {
        "final_decision": decision,
        "best_experiment": None if best is None else best["experiment_id"],
        "rationale": rationale,
        "ranked_experiments": [
            {
                "experiment_id": result["experiment_id"],
                "selection_block": result["selection_block"],
                "collapse_detected": result["collapse_diagnostics"]["detected"],
                "delta_vs_baseline": _delta_vs_baseline(result, baseline),
            }
            for result in ranked
        ],
    }


def run_experiment_matrix(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "experiment_matrix")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    logger.info("experiment_matrix starting raw_dataset=%s", settings.data.raw_dataset_path)

    config_report = audit_effective_config_payload(settings)
    write_json_report(run_dir, "config_effective_report.json", config_report)
    symbol_context_report = audit_symbol_context_payload(settings)
    write_json_report(run_dir, "symbol_context_comparison_report.json", symbol_context_report)

    primary_results: dict[str, dict[str, object]] = {}
    label_diagnostics: dict[str, object] = {}
    for spec in _primary_specs():
        logger.info("experiment_matrix running=%s mode=%s", spec.experiment_id, spec.model_mode)
        result = _evaluate_spec(settings, spec, run_dir)
        primary_results[spec.experiment_id] = result
        label_diagnostics[spec.experiment_id] = {
            "hypothesis": spec.hypothesis,
            "effective_labeling": result["effective_config"]["labeling"],  # type: ignore[index]
            "dataset": result["dataset"],  # type: ignore[index]
        }

    baseline = primary_results["exp0_baseline"]
    best_xgb = _select_best_result(primary_results, baseline, "xgb_tuning")
    best_context = _select_best_result(primary_results, baseline, "symbol_context")

    combined_specs: list[ExperimentSpec] = []
    if best_xgb is not None:
        combined_specs.append(_combined_spec("exp5_labels_plus_best_xgb", best_xgb, combine="labels_xgb"))
    if best_context is not None:
        combined_specs.append(_combined_spec("exp6_labels_plus_best_context", best_context, combine="labels_context"))

    combined_results: dict[str, dict[str, object]] = {}
    for spec in combined_specs:
        logger.info("experiment_matrix running=%s mode=%s", spec.experiment_id, spec.model_mode)
        combined_results[spec.experiment_id] = _evaluate_spec(settings, spec, run_dir)

    all_results = {**primary_results, **combined_results}
    matrix_report = {
        "baseline_experiment_id": baseline["experiment_id"],
        "selected_xgb_variant": None if best_xgb is None else best_xgb["experiment_id"],
        "selected_symbol_context_variant": None if best_context is None else best_context["experiment_id"],
        "experiments": {
            experiment_id: {
                "hypothesis": result["hypothesis"],
                "selection_block": result["selection_block"],
                "model_mode": result["model_mode"],
                "collapse_diagnostics": result["collapse_diagnostics"],
                "delta_vs_baseline": _delta_vs_baseline(result, baseline) if experiment_id != baseline["experiment_id"] else {},
                "validation": result["splits"]["validation"],
                "test": result["splits"]["test"],
                "walk_forward": result["walk_forward"],
            }
            for experiment_id, result in all_results.items()
        },
    }
    xgb_tuning_report = {
        "baseline_experiment_id": baseline["experiment_id"],
        "best_variant": None if best_xgb is None else best_xgb["experiment_id"],
        "variants": {
            experiment_id: {
                "delta_vs_baseline": _delta_vs_baseline(result, baseline),
                "training_summary": result["training_summary"],
            }
            for experiment_id, result in all_results.items()
            if experiment_id.startswith("exp3_")
        },
    }
    per_symbol_report = {
        experiment_id: {
            "validation": result["splits"]["validation"]["per_symbol"],  # type: ignore[index]
            "test": result["splits"]["test"]["per_symbol"],  # type: ignore[index]
            "validation_timeframe": result["splits"]["validation"]["per_timeframe"],  # type: ignore[index]
            "test_timeframe": result["splits"]["test"]["per_timeframe"],  # type: ignore[index]
        }
        for experiment_id, result in all_results.items()
    }
    recommendation = _recommendation(all_results, baseline)

    write_json_report(run_dir, "label_diagnostics_report.json", label_diagnostics)
    write_json_report(run_dir, "xgb_tuning_comparison_report.json", xgb_tuning_report)
    write_json_report(run_dir, "per_symbol_experiment_report.json", per_symbol_report)
    write_json_report(run_dir, "experiment_matrix_report.json", matrix_report)
    write_json_report(run_dir, "experiment_recommendation_report.json", recommendation)

    logger.info(
        "experiment_matrix complete experiments=%s decision=%s run_dir=%s",
        len(all_results),
        recommendation["final_decision"],
        run_dir,
    )
    return 0


def audit_effective_config_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "audit_effective_config")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    write_json_report(run_dir, "config_effective_report.json", audit_effective_config_payload(settings))
    logger.info("audit_effective_config run_dir=%s", run_dir)
    return 0


def audit_symbol_context_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "audit_symbol_context")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    write_json_report(run_dir, "symbol_context_comparison_report.json", audit_symbol_context_payload(settings))
    logger.info("audit_symbol_context run_dir=%s", run_dir)
    return 0


def compare_experiment_results_command(settings: Settings) -> int:
    configured = os.getenv("IRIS_EXPERIMENT_COMPARE_RUN_DIR", "").strip()
    if configured:
        source_dir = Path(configured)
    else:
        candidates = sorted(settings.data.runs_dir.glob("*_experiment_matrix"))
        if not candidates:
            return 1
        source_dir = candidates[-1]
    report_path = source_dir / "experiment_matrix_report.json"
    recommendation_path = source_dir / "experiment_recommendation_report.json"
    if not report_path.exists() or not recommendation_path.exists():
        return 2
    matrix_payload = json.loads(report_path.read_text(encoding="utf-8"))
    recommendation_payload = json.loads(recommendation_path.read_text(encoding="utf-8"))
    run_dir = build_run_directory(settings.data.runs_dir, "experiment_compare")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    write_json_report(
        run_dir,
        "experiment_matrix_report.json",
        {
            "source_run_dir": str(source_dir),
            "baseline_experiment_id": matrix_payload.get("baseline_experiment_id"),
            "selected_xgb_variant": matrix_payload.get("selected_xgb_variant"),
            "selected_symbol_context_variant": matrix_payload.get("selected_symbol_context_variant"),
            "experiments": matrix_payload.get("experiments", {}),
        },
    )
    write_json_report(
        run_dir,
        "experiment_recommendation_report.json",
        {
            "source_run_dir": str(source_dir),
            **recommendation_payload,
        },
    )
    logger.info("compare_experiment_results source=%s run_dir=%s", source_dir, run_dir)
    return 0
