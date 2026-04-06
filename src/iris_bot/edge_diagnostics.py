from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from iris_bot.artifacts import read_artifact_payload, wrap_artifact
from iris_bot.backtest import TradeRecord, run_backtest_engine
from iris_bot.config import LabelingConfig, Settings
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.processed_dataset import ProcessedRow
from iris_bot.regime_rework import (
    FOCUS_SYMBOL,
    REGIME_FEATURE_NAMES,
    SECONDARY_SYMBOL,
    TARGET_SYMBOLS,
    VARIANT_SPECS,
    _build_symbol_rows,
    _feature_names,
    _latest_run_dir,
    _regime_bucket,
    _run_variant,
    _session_name,
    _train_model,
)
from iris_bot.runtime_provenance import load_runtime_provenance_from_env
from iris_bot.splits import temporal_train_validation_test_split
from iris_bot.thresholds import apply_probability_threshold
from iris_bot.walk_forward import generate_walk_forward_windows


BASELINE_VARIANT_ID = "baseline_symbol_specific_actual"
CURRENT_REGIME_VARIANT_ID = "baseline_plus_explicit_regime"
EXTENDED_BASELINE_VARIANT_ID = "baseline_plus_expanded_sample"
EXTENDED_REGIME_VARIANT_ID = "expanded_sample_plus_explicit_regime"
FINAL_EDGE_DECISIONS = (
    "EDGE_NOT_STRONG_ENOUGH",
    "EDGE_EXISTS_BUT_CONTEXT_MISSING",
    "LABEL_DESIGN_IS_PRIMARY_PROBLEM",
    "HORIZON_EXIT_ALIGNMENT_IS_PRIMARY_PROBLEM",
    "REGIME_MODELING_IS_PRIMARY_PROBLEM",
    "REQUIRE_DIFFERENT_SIGNAL_FAMILY",
)


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


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    ordered = sorted(values)

    def _pick(ratio: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "p25": _pick(0.25),
        "median": _pick(0.50),
        "p75": _pick(0.75),
        "p90": _pick(0.90),
        "max": float(ordered[-1]),
        "mean": float(mean(ordered)),
    }


def _normalize_importance(raw: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in raw.values())
    if total <= 0.0:
        return {}
    return {
        str(name): float(value) / total
        for name, value in sorted(raw.items(), key=lambda item: item[1], reverse=True)
        if float(value) > 0.0
    }


def _baseline_variant():
    for variant in VARIANT_SPECS:
        if variant.variant_id == BASELINE_VARIANT_ID:
            return variant
    raise RuntimeError(f"Variante baseline no encontrada: {BASELINE_VARIANT_ID}")


def _variant_by_id(variant_id: str):
    for variant in VARIANT_SPECS:
        if variant.variant_id == variant_id:
            return variant
    raise RuntimeError(f"Variante no encontrada: {variant_id}")


def _label_config_for_variant(settings: Settings, variant_id: str) -> LabelingConfig:
    variant = _variant_by_id(variant_id)
    return replace(
        settings.labeling,
        take_profit_pct=variant.label_tp_pct,
        stop_loss_pct=variant.label_sl_pct,
        horizon_bars=variant.label_horizon_bars,
    )


def _timeframe_minutes(timeframe: str) -> int:
    if timeframe == "M5":
        return 5
    if timeframe == "M15":
        return 15
    if timeframe == "H1":
        return 60
    raise ValueError(f"timeframe desconocido: {timeframe}")


def _bars_between(start: datetime, end_iso: str, timeframe: str) -> int:
    end = datetime.fromisoformat(end_iso)
    delta_minutes = max(1, _timeframe_minutes(timeframe))
    return max(0, round((end - start).total_seconds() / 60.0 / delta_minutes))


def _row_regime(row: ProcessedRow) -> str:
    return _regime_bucket(row)


def _row_key(row: ProcessedRow, key_name: str) -> str:
    if key_name == "session":
        return _session_name(row)
    if key_name == "timeframe":
        return row.timeframe
    if key_name == "regime":
        return _row_regime(row)
    if key_name == "label_reason":
        return row.label_reason
    raise ValueError(f"row key desconocido: {key_name}")


def _row_confidence(probability: dict[int, float]) -> dict[str, float]:
    prob_long = _safe_float(probability.get(1))
    prob_short = _safe_float(probability.get(-1))
    prob_neutral = _safe_float(probability.get(0))
    directional_confidence = max(prob_long, prob_short)
    return {
        "prob_long": prob_long,
        "prob_short": prob_short,
        "prob_neutral": prob_neutral,
        "directional_confidence": directional_confidence,
        "directional_score": prob_long - prob_short,
    }


def _trade_metrics_from_subset(trades: list[TradeRecord]) -> dict[str, Any]:
    total_trades = len(trades)
    gross_profit = sum(trade.net_pnl_usd for trade in trades if trade.net_pnl_usd > 0.0)
    gross_loss = -sum(trade.net_pnl_usd for trade in trades if trade.net_pnl_usd < 0.0)
    net_pnl = sum(trade.net_pnl_usd for trade in trades)
    expectancy = net_pnl / total_trades if total_trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0)
    return {
        "trade_count": total_trades,
        "net_pnl_usd": net_pnl,
        "expectancy_usd": expectancy,
        "profit_factor": profit_factor,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
    }


def _split_bundle(
    settings: Settings,
    *,
    symbol: str,
    dataset_mode: str = "current",
    feature_mode: str = "baseline",
    variant_id: str = BASELINE_VARIANT_ID,
) -> dict[str, Any]:
    label_cfg = _label_config_for_variant(settings, variant_id)
    rows = _build_symbol_rows(settings, dataset_mode=dataset_mode, symbol=symbol, label_cfg=label_cfg)
    split = temporal_train_validation_test_split(
        rows,
        settings.split.train_ratio,
        settings.split.validation_ratio,
        settings.split.test_ratio,
    )
    feature_names = _feature_names(feature_mode)
    model, threshold, threshold_report = _train_model(settings, split.train, split.validation, feature_names)
    bundle: dict[str, Any] = {
        "symbol": symbol,
        "rows": rows,
        "split": split,
        "feature_names": feature_names,
        "feature_mode": feature_mode,
        "dataset_mode": dataset_mode,
        "variant_id": variant_id,
        "threshold": threshold,
        "threshold_report": threshold_report,
        "model": model,
        "label_config": label_cfg,
    }
    for split_name, split_rows in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        matrix = [[row.features[name] for name in feature_names] for row in split_rows]
        probabilities = model.predict_probabilities(matrix) if matrix else []
        predictions = apply_probability_threshold(probabilities, threshold) if probabilities else []
        metrics, trades, _ = run_backtest_engine(
            rows=split_rows,
            probabilities=probabilities,
            threshold=threshold,
            backtest=settings.backtest,
            risk=settings.risk,
            intrabar_policy=settings.backtest.intrabar_policy,
            exit_policy_config=settings.exit_policy,
            dynamic_exit_config=settings.dynamic_exits,
        )
        bundle[split_name] = {
            "rows": split_rows,
            "probabilities": probabilities,
            "predictions": predictions,
            "metrics": metrics,
            "trades": trades,
        }
    return bundle


def _rows_with_scores(rows: list[ProcessedRow], probabilities: list[dict[int, float]], predictions: list[int]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row, probability, prediction in zip(rows, probabilities, predictions, strict=False):
        scores = _row_confidence(probability)
        payload.append(
            {
                "timestamp": row.timestamp.isoformat(),
                "symbol": row.symbol,
                "timeframe": row.timeframe,
                "label": row.label,
                "label_reason": row.label_reason,
                "session": _session_name(row),
                "regime": _row_regime(row),
                "prediction": prediction,
                "predicted_trade": prediction != 0,
                "horizon_bars_to_label_end": _bars_between(row.timestamp, row.horizon_end_timestamp, row.timeframe),
                **scores,
            }
        )
    return payload


def _classification_by_key(rows_with_scores: list[dict[str, Any]], key_name: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows_with_scores:
        grouped[str(item[key_name])].append(item)
    payload: dict[str, Any] = {}
    for key, items in sorted(grouped.items()):
        total = len(items)
        errors = sum(1 for item in items if int(item["prediction"]) != int(item["label"]))
        directional = [item for item in items if int(item["label"]) != 0]
        directional_errors = sum(1 for item in directional if int(item["prediction"]) != int(item["label"]))
        payload[key] = {
            "row_count": total,
            "selected_row_count": sum(1 for item in items if bool(item["predicted_trade"])),
            "selected_ratio": sum(1 for item in items if bool(item["predicted_trade"])) / total if total else 0.0,
            "error_ratio": errors / total if total else 0.0,
            "directional_error_ratio": directional_errors / len(directional) if directional else 0.0,
            "mean_directional_confidence": mean(float(item["directional_confidence"]) for item in items) if items else 0.0,
        }
    return payload


def _trade_breakdown(
    rows: list[ProcessedRow],
    probabilities: list[dict[int, float]],
    predictions: list[int],
    trades: list[TradeRecord],
    key_name: str,
) -> dict[str, Any]:
    row_index = {row.timestamp.isoformat(): row for row in rows}
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, probability, prediction in zip(rows, probabilities, predictions, strict=False):
        grouped_rows[_row_key(row, key_name)].append(
            {"prediction": prediction, **_row_confidence(probability)}
        )
    grouped_trades: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        row = row_index.get(trade.signal_timestamp)
        if row is None:
            continue
        grouped_trades[_row_key(row, key_name)].append(trade)
    payload: dict[str, Any] = {}
    for key, items in sorted(grouped_rows.items()):
        trade_subset = grouped_trades.get(key, [])
        payload[key] = {
            "row_count": len(items),
            "trade_density": len(trade_subset) / len(items) if items else 0.0,
            "no_trade_ratio": sum(1 for item in items if int(item["prediction"]) == 0) / len(items) if items else 0.0,
            "mean_directional_confidence": mean(float(item["directional_confidence"]) for item in items) if items else 0.0,
            **_trade_metrics_from_subset(trade_subset),
        }
    return payload


def _exit_breakdown(trades: list[TradeRecord]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for exit_reason in sorted({trade.exit_reason for trade in trades}):
        subset = [trade for trade in trades if trade.exit_reason == exit_reason]
        payload[exit_reason] = _trade_metrics_from_subset(subset)
    return payload


def _score_distributions(rows_with_scores: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, Any] = {}
    for label in (-1, 0, 1):
        subset = [item for item in rows_with_scores if int(item["label"]) == label]
        key = str(label)
        if label == 1:
            class_scores = [float(item["prob_long"]) for item in subset]
            other_scores = [float(item["prob_long"]) for item in rows_with_scores if int(item["label"]) != 1]
        elif label == -1:
            class_scores = [float(item["prob_short"]) for item in subset]
            other_scores = [float(item["prob_short"]) for item in rows_with_scores if int(item["label"]) != -1]
        else:
            class_scores = [float(item["prob_neutral"]) for item in subset]
            other_scores = [float(item["prob_neutral"]) for item in rows_with_scores if int(item["label"]) != 0]
        other_median = median(other_scores) if other_scores else 0.0
        class_q75 = _quantiles(class_scores)["p75"] if class_scores else 0.0
        overlap = sum(1 for value in other_scores if value >= class_q75) / len(other_scores) if other_scores else 0.0
        by_label[key] = {
            "row_count": len(subset),
            "class_probability_quantiles": _quantiles(class_scores),
            "other_probability_quantiles": _quantiles(other_scores),
            "median_gap_vs_others": (median(class_scores) if class_scores else 0.0) - other_median,
            "above_other_median_ratio": (
                sum(1 for value in class_scores if value > other_median) / len(class_scores) if class_scores else 0.0
            ),
            "tail_overlap_ratio": overlap,
        }
    selected = [item for item in rows_with_scores if bool(item["predicted_trade"])]
    non_selected = [item for item in rows_with_scores if not bool(item["predicted_trade"])]
    return {
        "by_true_label": by_label,
        "selected_trade_confidence": _quantiles([float(item["directional_confidence"]) for item in selected]),
        "non_selected_confidence": _quantiles([float(item["directional_confidence"]) for item in non_selected]),
    }


def _feature_importance_report(bundle: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_importance(bundle["model"].feature_importance())
    top_features = list(normalized.items())[:10]
    regime_share = sum(normalized.get(feature, 0.0) for feature in REGIME_FEATURE_NAMES)
    return {
        "top_features": [{"feature": name, "normalized_gain": value} for name, value in top_features],
        "regime_feature_importance_share": regime_share,
        "feature_count_with_gain": len(normalized),
    }


def _walk_forward_diagnostics(
    settings: Settings,
    *,
    symbol: str,
    dataset_mode: str = "current",
    feature_mode: str = "baseline",
    variant_id: str = BASELINE_VARIANT_ID,
) -> dict[str, Any]:
    label_cfg = _label_config_for_variant(settings, variant_id)
    rows = _build_symbol_rows(settings, dataset_mode=dataset_mode, symbol=symbol, label_cfg=label_cfg)
    feature_names = _feature_names(feature_mode)
    windows = generate_walk_forward_windows(
        total_rows=len(rows),
        train_window=settings.walk_forward.train_window,
        validation_window=settings.walk_forward.validation_window,
        test_window=settings.walk_forward.test_window,
        step=settings.walk_forward.step,
    )
    folds: list[dict[str, Any]] = []
    thresholds: list[float] = []
    score_means: list[float] = []
    for window in windows:
        train_rows = rows[window.train_start : window.train_end]
        validation_rows = rows[window.validation_start : window.validation_end]
        test_rows = rows[window.test_start : window.test_end]
        if not train_rows or not validation_rows or not test_rows:
            folds.append({"fold_index": window.fold_index, "skipped": True, "reason": "insufficient_rows"})
            continue
        model, threshold, _ = _train_model(settings, train_rows, validation_rows, feature_names)
        test_matrix = [[row.features[name] for name in feature_names] for row in test_rows]
        probabilities = model.predict_probabilities(test_matrix)
        predictions = apply_probability_threshold(probabilities, threshold)
        metrics, trades, _ = run_backtest_engine(
            rows=test_rows,
            probabilities=probabilities,
            threshold=threshold,
            backtest=settings.backtest,
            risk=settings.risk,
            intrabar_policy=settings.backtest.intrabar_policy,
            exit_policy_config=settings.exit_policy,
            dynamic_exit_config=settings.dynamic_exits,
        )
        row_scores = _rows_with_scores(test_rows, probabilities, predictions)
        thresholds.append(threshold)
        score_means.append(mean(float(item["directional_confidence"]) for item in row_scores) if row_scores else 0.0)
        folds.append(
            {
                "fold_index": window.fold_index,
                "skipped": False,
                "threshold": threshold,
                "trade_count": _safe_int(metrics.get("total_trades")),
                "net_pnl_usd": _safe_float(metrics.get("net_pnl_usd")),
                "profit_factor": _safe_float(metrics.get("profit_factor")),
                "expectancy_usd": _safe_float(metrics.get("expectancy_usd")),
                "no_trade_ratio": sum(1 for prediction in predictions if prediction == 0) / len(predictions) if predictions else 0.0,
                "mean_directional_confidence": mean(float(item["directional_confidence"]) for item in row_scores) if row_scores else 0.0,
                "regime_trade_breakdown": _trade_breakdown(test_rows, probabilities, predictions, trades, "regime"),
            }
        )
    valid = [fold for fold in folds if not fold.get("skipped")]
    return {
        "total_folds": len(folds),
        "valid_folds": len(valid),
        "folds": folds,
        "threshold_stability": {
            "mean": mean(thresholds) if thresholds else 0.0,
            "stddev": pstdev(thresholds) if len(thresholds) > 1 else 0.0,
            "min": min(thresholds) if thresholds else 0.0,
            "max": max(thresholds) if thresholds else 0.0,
        },
        "score_stability": {
            "mean_directional_confidence": mean(score_means) if score_means else 0.0,
            "stddev_directional_confidence": pstdev(score_means) if len(score_means) > 1 else 0.0,
        },
    }


def _baseline_symbol_report(settings: Settings, symbol: str) -> dict[str, Any]:
    bundle = _split_bundle(settings, symbol=symbol)
    test_rows = bundle["test"]["rows"]
    test_probabilities = bundle["test"]["probabilities"]
    test_predictions = bundle["test"]["predictions"]
    test_trades = bundle["test"]["trades"]
    rows_with_scores = _rows_with_scores(test_rows, test_probabilities, test_predictions)
    label_resolution_bars = [_bars_between(row.timestamp, row.horizon_end_timestamp, row.timeframe) for row in test_rows]
    return {
        "symbol": symbol,
        "variant_id": BASELINE_VARIANT_ID,
        "dataset_mode": bundle["dataset_mode"],
        "feature_mode": bundle["feature_mode"],
        "labeling": {
            "mode": bundle["label_config"].mode,
            "horizon_bars": bundle["label_config"].horizon_bars,
            "take_profit_pct": bundle["label_config"].take_profit_pct,
            "stop_loss_pct": bundle["label_config"].stop_loss_pct,
        },
        "threshold": bundle["threshold"],
        "threshold_report": bundle["threshold_report"],
        "split_metrics": {
            split_name: {
                "row_count": len(bundle[split_name]["rows"]),
                "trade_count": _safe_int(bundle[split_name]["metrics"].get("total_trades")),
                "net_pnl_usd": _safe_float(bundle[split_name]["metrics"].get("net_pnl_usd")),
                "profit_factor": _safe_float(bundle[split_name]["metrics"].get("profit_factor")),
                "expectancy_usd": _safe_float(bundle[split_name]["metrics"].get("expectancy_usd")),
            }
            for split_name in ("train", "validation", "test")
        },
        "what_the_model_learns": _feature_importance_report(bundle),
        "score_distributions": _score_distributions(rows_with_scores),
        "performance_by_session": _trade_breakdown(test_rows, test_probabilities, test_predictions, test_trades, "session"),
        "performance_by_timeframe": _trade_breakdown(test_rows, test_probabilities, test_predictions, test_trades, "timeframe"),
        "performance_by_regime": _trade_breakdown(test_rows, test_probabilities, test_predictions, test_trades, "regime"),
        "performance_by_label_reason": _trade_breakdown(test_rows, test_probabilities, test_predictions, test_trades, "label_reason"),
        "performance_by_exit_reason": _exit_breakdown(test_trades),
        "classification_by_regime": _classification_by_key(rows_with_scores, "regime"),
        "classification_by_session": _classification_by_key(rows_with_scores, "session"),
        "classification_by_label_reason": _classification_by_key(rows_with_scores, "label_reason"),
        "label_resolution_bars": _quantiles(label_resolution_bars),
        "walk_forward": _walk_forward_diagnostics(settings, symbol=symbol),
    }


def _label_noise_symbol_report(settings: Settings, symbol: str) -> dict[str, Any]:
    bundle = _split_bundle(settings, symbol=symbol)
    test_rows = bundle["test"]["rows"]
    rows_with_scores = _rows_with_scores(test_rows, bundle["test"]["probabilities"], bundle["test"]["predictions"])
    label_reason_counts = Counter(row.label_reason for row in test_rows)
    timeout_reasons = {
        "triple_barrier_timeout_direction",
        "triple_barrier_timeout_small_move",
    }
    timeout_rows = [row for row in test_rows if row.label_reason in timeout_reasons]
    clean_rows = [row for row in test_rows if row.label_reason in {"triple_barrier_take_profit", "triple_barrier_stop_loss"}]
    errors = [item for item in rows_with_scores if int(item["prediction"]) != int(item["label"])]
    timeout_errors = [item for item in errors if str(item["label_reason"]) in timeout_reasons]
    clean_errors = [item for item in errors if str(item["label_reason"]) in {"triple_barrier_take_profit", "triple_barrier_stop_loss"}]
    return {
        "symbol": symbol,
        "total_rows": len(test_rows),
        "label_reason_counts": dict(sorted(label_reason_counts.items())),
        "timeout_ratio": len(timeout_rows) / len(test_rows) if test_rows else 0.0,
        "clean_ratio": len(clean_rows) / len(test_rows) if test_rows else 0.0,
        "timeout_direction_ratio": label_reason_counts.get("triple_barrier_timeout_direction", 0) / len(test_rows) if test_rows else 0.0,
        "timeout_small_move_ratio": label_reason_counts.get("triple_barrier_timeout_small_move", 0) / len(test_rows) if test_rows else 0.0,
        "error_share_timeout_labels": len(timeout_errors) / len(errors) if errors else 0.0,
        "error_share_clean_labels": len(clean_errors) / len(errors) if errors else 0.0,
        "label_reason_classification": _classification_by_key(rows_with_scores, "label_reason"),
        "timeout_rows_by_regime": _classification_by_key([item for item in rows_with_scores if str(item["label_reason"]) in timeout_reasons], "regime"),
        "timeout_rows_by_session": _classification_by_key([item for item in rows_with_scores if str(item["label_reason"]) in timeout_reasons], "session"),
        "dominant_label_noise_assessment": (
            "timeout_labels_dominate_errors"
            if len(timeout_errors) > len(clean_errors)
            else "clean_labels_still_dominate_or_tie"
        ),
    }


def _horizon_exit_symbol_report(settings: Settings, symbol: str) -> dict[str, Any]:
    bundle = _split_bundle(settings, symbol=symbol)
    test_rows = bundle["test"]["rows"]
    test_trades = bundle["test"]["trades"]
    label_horizon = bundle["label_config"].horizon_bars
    resolution_bars = [_bars_between(row.timestamp, row.horizon_end_timestamp, row.timeframe) for row in test_rows]
    exit_reasons = Counter(trade.exit_reason for trade in test_trades)
    bars_held = [trade.bars_held for trade in test_trades]
    time_exit_ratio = exit_reasons.get("time_exit", 0) / len(test_trades) if test_trades else 0.0
    timeout_label_ratio = (
        sum(1 for row in test_rows if row.label_reason.startswith("triple_barrier_timeout")) / len(test_rows)
        if test_rows
        else 0.0
    )
    return {
        "symbol": symbol,
        "label_horizon_bars": label_horizon,
        "backtest_max_holding_bars": settings.backtest.max_holding_bars,
        "label_resolution_bars": _quantiles(resolution_bars),
        "trade_duration_bars": _quantiles(bars_held),
        "trade_exit_reason_counts": dict(sorted(exit_reasons.items())),
        "trade_exit_reason_metrics": _exit_breakdown(test_trades),
        "timeout_label_ratio": timeout_label_ratio,
        "time_exit_trade_ratio": time_exit_ratio,
        "tp_trade_ratio": exit_reasons.get("take_profit", 0) / len(test_trades) if test_trades else 0.0,
        "sl_trade_ratio": (
            (exit_reasons.get("stop_loss", 0) + exit_reasons.get("stop_loss_same_bar", 0)) / len(test_trades)
            if test_trades
            else 0.0
        ),
        "alignment_assessment": (
            "horizon_too_short_or_barriers_too_far"
            if timeout_label_ratio >= 0.30 or time_exit_ratio >= 0.30
            else "horizon_exit_alignment_not_dominant"
        ),
    }


def _regime_value_symbol_report(settings: Settings, symbol: str) -> dict[str, Any]:
    baseline_bundle = _split_bundle(settings, symbol=symbol)
    regime_bundle = _split_bundle(settings, symbol=symbol, feature_mode="regime", variant_id=CURRENT_REGIME_VARIANT_ID)
    baseline_scores = _feature_importance_report(baseline_bundle)
    regime_scores = _feature_importance_report(regime_bundle)
    variant_results = {
        variant_id: _run_variant(settings, symbol, _variant_by_id(variant_id))
        for variant_id in (
            BASELINE_VARIANT_ID,
            CURRENT_REGIME_VARIANT_ID,
            EXTENDED_BASELINE_VARIANT_ID,
            EXTENDED_REGIME_VARIANT_ID,
        )
    }
    baseline_result = variant_results[BASELINE_VARIANT_ID]
    comparison: dict[str, Any] = {}
    baseline_wf = _safe_float(baseline_result["walk_forward"]["aggregate"]["total_net_pnl_usd"])
    baseline_pf = _safe_float(baseline_result["test_metrics"]["profit_factor"])
    for variant_id, result in variant_results.items():
        if variant_id == BASELINE_VARIANT_ID:
            continue
        comparison[variant_id] = {
            "dataset_mode": result["dataset_mode"],
            "feature_mode": result["feature_mode"],
            "decision": result["decision"],
            "walk_forward_delta_vs_baseline_usd": _safe_float(result["walk_forward"]["aggregate"]["total_net_pnl_usd"]) - baseline_wf,
            "test_profit_factor_delta_vs_baseline": _safe_float(result["test_metrics"]["profit_factor"]) - baseline_pf,
            "test_expectancy_delta_vs_baseline": _safe_float(result["test_metrics"]["expectancy_usd"]) - _safe_float(baseline_result["test_metrics"]["expectancy_usd"]),
            "regime_effect_delta_vs_baseline": _safe_float(result["regime_effect_summary"]["trend_minus_range_expectancy_usd"]) - _safe_float(
                baseline_result["regime_effect_summary"]["trend_minus_range_expectancy_usd"]
            ),
        }
    return {
        "symbol": symbol,
        "baseline_result": {
            "decision": baseline_result["decision"],
            "walk_forward_total_net_pnl_usd": baseline_wf,
            "test_profit_factor": baseline_pf,
            "test_expectancy_usd": _safe_float(baseline_result["test_metrics"]["expectancy_usd"]),
            "regime_effect_summary": baseline_result["regime_effect_summary"],
        },
        "current_regime_feature_importance": regime_scores,
        "baseline_feature_importance": baseline_scores,
        "regime_incremental_value_comparison": comparison,
        "regime_value_assessment": (
            "regime_features_not_helpful_enough"
            if all(item["walk_forward_delta_vs_baseline_usd"] <= 0.0 for item in comparison.values())
            else "regime_has_partial_incremental_value"
        ),
    }


def _class_separability_symbol_report(settings: Settings, symbol: str) -> dict[str, Any]:
    bundle = _split_bundle(settings, symbol=symbol)
    rows_with_scores = _rows_with_scores(bundle["test"]["rows"], bundle["test"]["probabilities"], bundle["test"]["predictions"])
    profitable_trades = [trade for trade in bundle["test"]["trades"] if trade.net_pnl_usd > 0.0]
    losing_trades = [trade for trade in bundle["test"]["trades"] if trade.net_pnl_usd < 0.0]
    profitable_confidence = [max(trade.probability_long, trade.probability_short) for trade in profitable_trades]
    losing_confidence = [max(trade.probability_long, trade.probability_short) for trade in losing_trades]
    return {
        "symbol": symbol,
        "threshold": bundle["threshold"],
        "score_distributions": _score_distributions(rows_with_scores),
        "trade_confidence_profitability": {
            "profitable_trade_confidence": _quantiles(profitable_confidence),
            "losing_trade_confidence": _quantiles(losing_confidence),
            "mean_confidence_gap": (mean(profitable_confidence) if profitable_confidence else 0.0) - (mean(losing_confidence) if losing_confidence else 0.0),
        },
        "walk_forward": _walk_forward_diagnostics(settings, symbol=symbol),
        "separability_assessment": (
            "weak_class_overlap_and_threshold_fragility"
            if _score_distributions(rows_with_scores)["by_true_label"]["1"]["tail_overlap_ratio"] >= 0.25
            or _score_distributions(rows_with_scores)["by_true_label"]["-1"]["tail_overlap_ratio"] >= 0.25
            else "class_overlap_not_severe"
        ),
    }


def _costly_error_symbol_report(settings: Settings, symbol: str) -> dict[str, Any]:
    bundle = _split_bundle(settings, symbol=symbol)
    rows = bundle["test"]["rows"]
    probabilities = bundle["test"]["probabilities"]
    predictions = bundle["test"]["predictions"]
    trades = bundle["test"]["trades"]
    row_index = {row.timestamp.isoformat(): row for row in rows}
    losing_trades = sorted([trade for trade in trades if trade.net_pnl_usd < 0.0], key=lambda trade: trade.net_pnl_usd)
    expensive_false_positives = [
        {
            "signal_timestamp": trade.signal_timestamp,
            "net_pnl_usd": trade.net_pnl_usd,
            "exit_reason": trade.exit_reason,
            "bars_held": trade.bars_held,
            "session": _session_name(row_index[trade.signal_timestamp]) if trade.signal_timestamp in row_index else "unknown",
            "regime": _row_regime(row_index[trade.signal_timestamp]) if trade.signal_timestamp in row_index else "unknown",
            "label_reason": row_index[trade.signal_timestamp].label_reason if trade.signal_timestamp in row_index else "unknown",
            "probability_long": trade.probability_long,
            "probability_short": trade.probability_short,
        }
        for trade in losing_trades[:10]
    ]
    missed_opportunity_proxy: list[dict[str, Any]] = []
    for row, probability, prediction in zip(rows, probabilities, predictions, strict=False):
        if prediction != 0 or row.label == 0:
            continue
        directional_confidence = max(_safe_float(probability.get(1)), _safe_float(probability.get(-1)))
        if directional_confidence < max(0.0, bundle["threshold"] - 0.05):
            continue
        missed_opportunity_proxy.append(
            {
                "timestamp": row.timestamp.isoformat(),
                "label": row.label,
                "label_reason": row.label_reason,
                "session": _session_name(row),
                "regime": _row_regime(row),
                "directional_confidence": directional_confidence,
                "threshold_gap": bundle["threshold"] - directional_confidence,
            }
        )
    return {
        "symbol": symbol,
        "expensive_false_positives": expensive_false_positives,
        "missed_opportunity_proxy": missed_opportunity_proxy[:20],
        "losing_trade_breakdown_by_regime": _trade_breakdown(
            rows,
            probabilities,
            predictions,
            [trade for trade in trades if trade.net_pnl_usd < 0.0],
            "regime",
        ),
        "losing_trade_breakdown_by_session": _trade_breakdown(
            rows,
            probabilities,
            predictions,
            [trade for trade in trades if trade.net_pnl_usd < 0.0],
            "session",
        ),
    }


def _hypothesis_entry(hypothesis_id: str, statement: str, support_score: float, evidence_for: list[str], evidence_against: list[str]) -> dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "statement": statement,
        "support_score": max(0.0, min(1.0, support_score)),
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
    }


def _conservative_final_decision(hypotheses: list[dict[str, Any]]) -> str:
    ranked = sorted(hypotheses, key=lambda item: float(item["support_score"]), reverse=True)
    top = ranked[0]["hypothesis_id"] if ranked else ""
    mapping = {
        "H1": "LABEL_DESIGN_IS_PRIMARY_PROBLEM",
        "H2": "HORIZON_EXIT_ALIGNMENT_IS_PRIMARY_PROBLEM",
        "H3": "REGIME_MODELING_IS_PRIMARY_PROBLEM",
        "H4": "EDGE_NOT_STRONG_ENOUGH",
        "H5": "EDGE_EXISTS_BUT_CONTEXT_MISSING",
        "H6": "REQUIRE_DIFFERENT_SIGNAL_FAMILY",
    }
    return mapping.get(top, "EDGE_NOT_STRONG_ENOUGH")


def _hypothesis_matrix(settings: Settings) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline = {symbol: _baseline_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS}
    label_noise = {symbol: _label_noise_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS}
    horizon = {symbol: _horizon_exit_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS}
    regime = {symbol: _regime_value_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS}
    separability = {symbol: _class_separability_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS}
    costly = {symbol: _costly_error_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS}

    gbp_label = label_noise[FOCUS_SYMBOL]
    gbp_horizon = horizon[FOCUS_SYMBOL]
    gbp_regime = regime[FOCUS_SYMBOL]
    gbp_sep = separability[FOCUS_SYMBOL]
    gbp_baseline = baseline[FOCUS_SYMBOL]
    eur_baseline = baseline[SECONDARY_SYMBOL]

    hypotheses = [
        _hypothesis_entry(
            "H1",
            "El problema principal son timeout labels y ruido de label design.",
            0.25 + 0.35 * _safe_float(gbp_label["error_share_timeout_labels"]) + 0.20 * _safe_float(gbp_label["timeout_ratio"]),
            [
                f"timeout_ratio={_safe_float(gbp_label['timeout_ratio']):.3f}",
                f"timeout_error_share={_safe_float(gbp_label['error_share_timeout_labels']):.3f}",
            ],
            [
                f"clean_ratio={_safe_float(gbp_label['clean_ratio']):.3f}",
            ],
        ),
        _hypothesis_entry(
            "H2",
            "El problema principal es desalineación entre horizon, TP/SL y duración real.",
            0.20 + 0.35 * _safe_float(gbp_horizon["timeout_label_ratio"]) + 0.30 * _safe_float(gbp_horizon["time_exit_trade_ratio"]),
            [
                f"timeout_label_ratio={_safe_float(gbp_horizon['timeout_label_ratio']):.3f}",
                f"time_exit_trade_ratio={_safe_float(gbp_horizon['time_exit_trade_ratio']):.3f}",
                f"median_trade_bars={_safe_float(gbp_horizon['trade_duration_bars']['median']):.2f}",
            ],
            [
                f"tp_trade_ratio={_safe_float(gbp_horizon['tp_trade_ratio']):.3f}",
            ],
        ),
        _hypothesis_entry(
            "H3",
            "El problema principal es régimen mal capturado o poco útil para esta señal.",
            0.25
            + 0.30 * (1.0 if str(gbp_regime["regime_value_assessment"]) == "regime_features_not_helpful_enough" else 0.0)
            + 0.20 * (1.0 - _safe_float(gbp_regime["current_regime_feature_importance"]["regime_feature_importance_share"])),
            [
                gbp_regime["regime_value_assessment"],
                f"regime_feature_share={_safe_float(gbp_regime['current_regime_feature_importance']['regime_feature_importance_share']):.3f}",
            ],
            [
                f"baseline_trend_minus_range={_safe_float(gbp_regime['baseline_result']['regime_effect_summary']['trend_minus_range_expectancy_usd']):.3f}",
            ],
        ),
        _hypothesis_entry(
            "H4",
            "El edge existe pero es demasiado débil/inestable para automatización conservadora.",
            0.30
            + 0.25 * _safe_float(gbp_baseline["walk_forward"]["threshold_stability"]["stddev"])
            + 0.25 * _safe_float(gbp_baseline["split_metrics"]["test"]["trade_count"] == 0)
            + 0.20 * (_safe_float(gbp_sep["walk_forward"]["threshold_stability"]["stddev"]) > 0.05),
            [
                f"wf_threshold_stddev={_safe_float(gbp_sep['walk_forward']['threshold_stability']['stddev']):.3f}",
                f"test_trade_count={_safe_int(gbp_baseline['split_metrics']['test']['trade_count'])}",
                f"eurusd_test_pf={_safe_float(eur_baseline['split_metrics']['test']['profit_factor']):.3f}",
            ],
            [
                f"gbpusd_test_pf={_safe_float(gbp_baseline['split_metrics']['test']['profit_factor']):.3f}",
            ],
        ),
        _hypothesis_entry(
            "H5",
            "El thresholding actual mata demasiada muestra útil.",
            0.20
            + 0.35 * _safe_float(gbp_baseline["performance_by_regime"]["trend_mid_vol"]["no_trade_ratio"] if "trend_mid_vol" in gbp_baseline["performance_by_regime"] else 0.0)
            + 0.20 * min(1.0, len(costly[FOCUS_SYMBOL]["missed_opportunity_proxy"]) / 20.0),
            [
                f"missed_opportunity_proxy={len(costly[FOCUS_SYMBOL]['missed_opportunity_proxy'])}",
                f"test_threshold={_safe_float(gbp_baseline['threshold']):.3f}",
            ],
            [
                f"test_trade_count={_safe_int(gbp_baseline['split_metrics']['test']['trade_count'])}",
            ],
        ),
        _hypothesis_entry(
            "H6",
            "El baseline ya está cerca del límite de lo que esta familia de señal puede extraer.",
            0.30
            + 0.30 * (1.0 if str(gbp_regime["regime_value_assessment"]) == "regime_features_not_helpful_enough" else 0.0)
            + 0.20 * (1.0 if _safe_float(gbp_baseline["split_metrics"]["test"]["profit_factor"]) > _safe_float(eur_baseline["split_metrics"]["test"]["profit_factor"]) else 0.0),
            [
                "baseline sigue siendo la mejor variante frente a muestra ampliada y régimen explícito",
                f"gbpusd_test_pf={_safe_float(gbp_baseline['split_metrics']['test']['profit_factor']):.3f}",
            ],
            [
                f"eurusd_test_pf={_safe_float(eur_baseline['split_metrics']['test']['profit_factor']):.3f}",
            ],
        ),
    ]
    hypotheses = sorted(hypotheses, key=lambda item: float(item["support_score"]), reverse=True)
    final_decision = _conservative_final_decision(hypotheses)
    recommendation = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "final_conservative_decision": final_decision,
        "top_hypothesis": hypotheses[0]["hypothesis_id"] if hypotheses else None,
        "recommended_next_actions": [
            "Probar una sola variante de label/horizon centrada en reducir timeout labels si H1/H2 domina tambien en GBPUSD extendido.",
            "Mantener GBPUSD en research conservador sin promoción si la separabilidad sigue débil fuera de muestra.",
            "Usar EURUSD solo como control secundario; no promoverlo mientras siga por detrás de GBPUSD.",
            "Evitar tuning adicional y nuevas features hasta invalidar o confirmar la hipótesis dominante.",
        ],
        "not_recommended_next_actions": [
            "No bajar gates cuantitativos.",
            "No meter más features de régimen sin evidencia incremental.",
            "No promover a demo execution con este edge todavía.",
        ],
    }
    return baseline, costly, {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "hypotheses": hypotheses,
        "final_conservative_decision": final_decision,
    }, recommendation


def _write_run_artifacts(run_dir: Path, reports: dict[str, dict[str, Any]]) -> None:
    for filename, payload in reports.items():
        write_json_report(run_dir, filename, payload)


def _load_existing_report(path: Path, expected_type: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return read_artifact_payload(path, expected_type=expected_type)
    except Exception:
        return None


def _existing_regime_matrix(settings: Settings) -> dict[str, Any] | None:
    run_dir = _latest_run_dir(settings, "compare_regime_experiments")
    if run_dir is None:
        return None
    return _load_existing_report(run_dir / "regime_aware_experiment_matrix_report.json", "regime_aware_experiment_matrix_report")


def run_audit_edge_baseline(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "audit_edge_baseline")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        matrix = _existing_regime_matrix(settings)
        payload = {
            "focus_symbol": FOCUS_SYMBOL,
            "secondary_symbol": SECONDARY_SYMBOL,
            "environment_provenance": load_runtime_provenance_from_env(),
            "baseline_symbol_reports": {symbol: _baseline_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS},
            "existing_regime_matrix_snapshot": matrix,
        }
    except Exception as exc:
        logger.error(str(exc))
        return 1
    _write_run_artifacts(run_dir, {"baseline_edge_diagnostic_report.json": wrap_artifact("baseline_edge_diagnostic_report", payload)})
    logger.info("baseline_edge_diagnostic=%s", run_dir / "baseline_edge_diagnostic_report.json")
    return 0


def run_audit_label_noise(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "audit_label_noise")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        payload = {
            "focus_symbol": FOCUS_SYMBOL,
            "secondary_symbol": SECONDARY_SYMBOL,
            "environment_provenance": load_runtime_provenance_from_env(),
            "symbols": {symbol: _label_noise_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS},
        }
    except Exception as exc:
        logger.error(str(exc))
        return 1
    _write_run_artifacts(run_dir, {"label_noise_report.json": wrap_artifact("label_noise_report", payload)})
    logger.info("label_noise=%s", run_dir / "label_noise_report.json")
    return 0


def run_audit_horizon_exits(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "audit_horizon_exits")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        payload = {
            "focus_symbol": FOCUS_SYMBOL,
            "secondary_symbol": SECONDARY_SYMBOL,
            "environment_provenance": load_runtime_provenance_from_env(),
            "symbols": {symbol: _horizon_exit_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS},
        }
    except Exception as exc:
        logger.error(str(exc))
        return 1
    _write_run_artifacts(run_dir, {"horizon_exit_alignment_report.json": wrap_artifact("horizon_exit_alignment_report", payload)})
    logger.info("horizon_exit_alignment=%s", run_dir / "horizon_exit_alignment_report.json")
    return 0


def run_audit_regime_value(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "audit_regime_value")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        payload = {
            "focus_symbol": FOCUS_SYMBOL,
            "secondary_symbol": SECONDARY_SYMBOL,
            "environment_provenance": load_runtime_provenance_from_env(),
            "symbols": {symbol: _regime_value_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS},
        }
    except Exception as exc:
        logger.error(str(exc))
        return 1
    _write_run_artifacts(run_dir, {"regime_value_report.json": wrap_artifact("regime_value_report", payload)})
    logger.info("regime_value=%s", run_dir / "regime_value_report.json")
    return 0


def run_audit_class_separability(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "audit_class_separability")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        payload = {
            "focus_symbol": FOCUS_SYMBOL,
            "secondary_symbol": SECONDARY_SYMBOL,
            "environment_provenance": load_runtime_provenance_from_env(),
            "symbols": {symbol: _class_separability_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS},
        }
    except Exception as exc:
        logger.error(str(exc))
        return 1
    _write_run_artifacts(run_dir, {"class_separability_report.json": wrap_artifact("class_separability_report", payload)})
    logger.info("class_separability=%s", run_dir / "class_separability_report.json")
    return 0


def run_audit_edge_hypotheses(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "audit_edge_hypotheses")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        baseline, costly, matrix, recommendation = _hypothesis_matrix(settings)
        reports = {
            "baseline_edge_diagnostic_report.json": wrap_artifact(
                "baseline_edge_diagnostic_report",
                {
                    "focus_symbol": FOCUS_SYMBOL,
                    "secondary_symbol": SECONDARY_SYMBOL,
                    "environment_provenance": load_runtime_provenance_from_env(),
                    "baseline_symbol_reports": baseline,
                    "existing_regime_matrix_snapshot": _existing_regime_matrix(settings),
                },
            ),
            "label_noise_report.json": wrap_artifact(
                "label_noise_report",
                {
                    "focus_symbol": FOCUS_SYMBOL,
                    "secondary_symbol": SECONDARY_SYMBOL,
                    "environment_provenance": load_runtime_provenance_from_env(),
                    "symbols": {symbol: _label_noise_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS},
                },
            ),
            "horizon_exit_alignment_report.json": wrap_artifact(
                "horizon_exit_alignment_report",
                {
                    "focus_symbol": FOCUS_SYMBOL,
                    "secondary_symbol": SECONDARY_SYMBOL,
                    "environment_provenance": load_runtime_provenance_from_env(),
                    "symbols": {symbol: _horizon_exit_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS},
                },
            ),
            "regime_value_report.json": wrap_artifact(
                "regime_value_report",
                {
                    "focus_symbol": FOCUS_SYMBOL,
                    "secondary_symbol": SECONDARY_SYMBOL,
                    "environment_provenance": load_runtime_provenance_from_env(),
                    "symbols": {symbol: _regime_value_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS},
                },
            ),
            "class_separability_report.json": wrap_artifact(
                "class_separability_report",
                {
                    "focus_symbol": FOCUS_SYMBOL,
                    "secondary_symbol": SECONDARY_SYMBOL,
                    "environment_provenance": load_runtime_provenance_from_env(),
                    "symbols": {symbol: _class_separability_symbol_report(settings, symbol) for symbol in TARGET_SYMBOLS},
                },
            ),
            "costly_error_analysis_report.json": wrap_artifact(
                "costly_error_analysis_report",
                {
                    "focus_symbol": FOCUS_SYMBOL,
                    "secondary_symbol": SECONDARY_SYMBOL,
                    "environment_provenance": load_runtime_provenance_from_env(),
                    "symbols": costly,
                },
            ),
            "hypothesis_matrix_report.json": wrap_artifact("hypothesis_matrix_report", matrix),
            "edge_diagnosis_recommendation_report.json": wrap_artifact(
                "edge_diagnosis_recommendation_report",
                {
                    "focus_symbol": FOCUS_SYMBOL,
                    "secondary_symbol": SECONDARY_SYMBOL,
                    "environment_provenance": load_runtime_provenance_from_env(),
                    **recommendation,
                },
            ),
        }
    except Exception as exc:
        logger.error(str(exc))
        return 1
    _write_run_artifacts(run_dir, reports)
    logger.info("edge_hypotheses=%s", run_dir)
    return 0
