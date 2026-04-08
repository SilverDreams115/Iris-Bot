from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from statistics import mean, pstdev
from typing import Any

from iris_bot.artifacts import read_artifact_payload, wrap_artifact
from iris_bot.backtest import TradeRecord, run_backtest_engine
from iris_bot.config import LabelingConfig, Settings
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.processed_dataset import ProcessedRow
from iris_bot.regime_rework import (
    FOCUS_SYMBOL,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    MIN_VALIDATION_ROWS,
    OUT_OF_FOCUS_SYMBOLS,
    SECONDARY_SYMBOL,
    _build_symbol_rows,
    _feature_names,
    _latest_run_dir,
    _train_model,
)
from iris_bot.runtime_provenance import load_runtime_provenance_from_env
from iris_bot.splits import temporal_train_validation_test_split
from iris_bot.thresholds import apply_probability_threshold
from iris_bot.walk_forward import generate_walk_forward_windows


@dataclass(frozen=True)
class RealignmentVariant:
    variant_id: str
    hypothesis: str
    horizon_bars: int
    take_profit_pct: float
    stop_loss_pct: float
    max_holding_bars: int
    timeout_handling_mode: str = "directional"
    timeout_direction_min_barrier_fraction: float = 0.0
    dataset_mode: str = "current"


VARIANTS: tuple[RealignmentVariant, ...] = (
    RealignmentVariant(
        variant_id="baseline_actual",
        hypothesis="Baseline actual actual: horizon 8, TP/SL 0.20/0.20 y time exit en 8 barras.",
        horizon_bars=8,
        take_profit_pct=0.0020,
        stop_loss_pct=0.0020,
        max_holding_bars=8,
    ),
    RealignmentVariant(
        variant_id="longer_horizon_base",
        hypothesis="Extender horizon y max_holding a 12 barras manteniendo TP/SL base para liberar la truncacion observada en 8 barras.",
        horizon_bars=12,
        take_profit_pct=0.0020,
        stop_loss_pct=0.0020,
        max_holding_bars=12,
    ),
    RealignmentVariant(
        variant_id="longer_horizon_aligned_barriers",
        hypothesis="Mismo horizon 12, pero TP/SL realineados a 0.25/0.20 para dar mas espacio de extension sin empeorar el stop.",
        horizon_bars=12,
        take_profit_pct=0.0025,
        stop_loss_pct=0.0020,
        max_holding_bars=12,
    ),
    RealignmentVariant(
        variant_id="longer_horizon_timeout_filtered",
        hypothesis="Horizon 12 con timeout handling conservador: timeout sin al menos 50% de una barrera se vuelve neutral.",
        horizon_bars=12,
        take_profit_pct=0.0020,
        stop_loss_pct=0.0020,
        max_holding_bars=12,
        timeout_handling_mode="neutral_by_barrier_fraction",
        timeout_direction_min_barrier_fraction=0.50,
    ),
    RealignmentVariant(
        variant_id="aligned_combo_timeout_filtered",
        hypothesis="Mejor combinacion acotada: horizon 12, TP/SL 0.25/0.20 y timeout conservador por fraccion de barrera.",
        horizon_bars=12,
        take_profit_pct=0.0025,
        stop_loss_pct=0.0020,
        max_holding_bars=12,
        timeout_handling_mode="neutral_by_barrier_fraction",
        timeout_direction_min_barrier_fraction=0.50,
    ),
)

CONTROL_VARIANT_IDS = ("baseline_actual", "aligned_combo_timeout_filtered")
TIMEOUT_LABEL_REASONS = {
    "triple_barrier_timeout_small_move",
    "triple_barrier_timeout_direction",
    "triple_barrier_timeout_filtered_small_move",
}


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


def _bars_between(row: ProcessedRow) -> int:
    horizon_end = datetime.fromisoformat(row.horizon_end_timestamp)
    timeframe_minutes = {"M5": 5, "M15": 15, "H1": 60}[row.timeframe]
    return max(0, round((horizon_end - row.timestamp).total_seconds() / 60.0 / timeframe_minutes))


def _trade_duration_summary(trades: list[TradeRecord]) -> dict[str, Any]:
    durations = [float(trade.bars_held) for trade in trades]
    exit_counts = Counter(trade.exit_reason for trade in trades)
    total = len(trades)
    return {
        "trade_count": total,
        "duration_bars": _quantiles(durations),
        "exit_reason_counts": dict(sorted(exit_counts.items())),
        "take_profit_ratio": exit_counts.get("take_profit", 0) / total if total else 0.0,
        "stop_loss_ratio": (exit_counts.get("stop_loss", 0) + exit_counts.get("stop_loss_same_bar", 0)) / total if total else 0.0,
        "time_exit_ratio": exit_counts.get("time_exit", 0) / total if total else 0.0,
    }


def _label_timeout_summary(rows: list[ProcessedRow]) -> dict[str, Any]:
    counts = Counter(row.label_reason for row in rows)
    total = len(rows)
    timeout_count = sum(counts.get(reason, 0) for reason in TIMEOUT_LABEL_REASONS)
    return {
        "row_count": total,
        "label_reason_counts": dict(sorted(counts.items())),
        "timeout_label_count": timeout_count,
        "timeout_ratio": timeout_count / total if total else 0.0,
        "filtered_timeout_ratio": counts.get("triple_barrier_timeout_filtered_small_move", 0) / total if total else 0.0,
    }


def _threshold_utility(predictions: list[int], threshold: float, rows: list[ProcessedRow], probabilities: list[dict[int, float]]) -> dict[str, Any]:
    total = len(predictions)
    selected_indexes = [index for index, prediction in enumerate(predictions) if prediction != 0]
    selected = len(selected_indexes)
    missed_near_threshold = 0
    for index, prediction in enumerate(predictions):
        if prediction != 0:
            continue
        directional_confidence = max(_safe_float(probabilities[index].get(1)), _safe_float(probabilities[index].get(-1)))
        if directional_confidence >= max(0.0, threshold - 0.05):
            missed_near_threshold += 1
    return {
        "threshold": threshold,
        "selected_trade_ratio": selected / total if total else 0.0,
        "no_trade_ratio": sum(1 for prediction in predictions if prediction == 0) / total if total else 0.0,
        "selected_trade_count": selected,
        "missed_near_threshold_count": missed_near_threshold,
        "missed_near_threshold_ratio": missed_near_threshold / total if total else 0.0,
        "selected_timeout_label_ratio": (
            sum(1 for index in selected_indexes if rows[index].label_reason in TIMEOUT_LABEL_REASONS) / selected
            if selected
            else 0.0
        ),
    }


def _effective_settings(settings: Settings, variant: RealignmentVariant) -> tuple[Settings, LabelingConfig]:
    label_cfg = replace(
        settings.labeling,
        horizon_bars=variant.horizon_bars,
        take_profit_pct=variant.take_profit_pct,
        stop_loss_pct=variant.stop_loss_pct,
        timeout_handling_mode=variant.timeout_handling_mode,
        timeout_direction_min_barrier_fraction=variant.timeout_direction_min_barrier_fraction,
    )
    effective = replace(
        settings,
        labeling=label_cfg,
        backtest=replace(settings.backtest, max_holding_bars=variant.max_holding_bars),
    )
    return effective, label_cfg


def _decision(settings: Settings, result: dict[str, Any], baseline: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    test_metrics = result["test_metrics"]
    walk_forward = result["walk_forward"]["aggregate"]
    timeout_ratio = _safe_float(result["timeout_label_summary"]["timeout_ratio"])
    time_exit_ratio = _safe_float(result["trade_duration_summary"]["time_exit_ratio"])
    threshold_stddev = _safe_float(result["walk_forward"]["threshold_stability"]["stddev"])
    if _safe_float(test_metrics["expectancy_usd"]) <= 0.0:
        reasons.append("test_expectancy_non_positive")
    if _safe_float(test_metrics["profit_factor"]) < settings.strategy.min_profit_factor:
        reasons.append("test_profit_factor_below_floor")
    if int(test_metrics["trade_count"]) < settings.strategy.min_validation_trades:
        reasons.append("test_trade_count_below_floor")
    if _safe_float(test_metrics["no_trade_ratio"]) > settings.strategy.caution_no_trade_ratio:
        reasons.append("test_no_trade_ratio_above_floor")
    if _safe_float(walk_forward["total_net_pnl_usd"]) <= 0.0:
        reasons.append("walk_forward_non_positive")
    if _safe_float(walk_forward["mean_profit_factor"]) < settings.strategy.min_profit_factor:
        reasons.append("walk_forward_profit_factor_below_floor")
    if _safe_float(walk_forward["positive_fold_ratio"]) < settings.strategy.min_positive_walkforward_ratio:
        reasons.append("walk_forward_positive_fold_ratio_below_floor")
    if timeout_ratio >= 0.60:
        reasons.append("timeout_labels_still_dominant")
    if time_exit_ratio >= 0.70:
        reasons.append("time_exit_still_dominant")
    if threshold_stddev > 0.08:
        reasons.append("threshold_fragility_still_high")

    baseline_timeout = _safe_float(baseline["timeout_label_summary"]["timeout_ratio"])
    baseline_time_exit = _safe_float(baseline["trade_duration_summary"]["time_exit_ratio"])
    baseline_wf = _safe_float(baseline["walk_forward"]["aggregate"]["total_net_pnl_usd"])
    improvement_points = 0
    if timeout_ratio <= baseline_timeout - 0.10:
        improvement_points += 1
    if time_exit_ratio <= baseline_time_exit - 0.20:
        improvement_points += 1
    if int(test_metrics["trade_count"]) >= int(baseline["test_metrics"]["trade_count"]) + 2:
        improvement_points += 1
    if _safe_float(walk_forward["total_net_pnl_usd"]) > baseline_wf:
        improvement_points += 1
    if _safe_float(test_metrics["profit_factor"]) > _safe_float(baseline["test_metrics"]["profit_factor"]):
        improvement_points += 1

    if not reasons:
        return "CANDIDATE_FOR_DEMO_EXECUTION", reasons
    if result["variant_id"] != baseline["variant_id"] and improvement_points >= 3:
        return "IMPROVED_BUT_NOT_ENOUGH", reasons
    return "REJECT_FOR_DEMO_EXECUTION", reasons


def _walk_forward_report(settings: Settings, rows: list[ProcessedRow], feature_names: list[str]) -> dict[str, Any]:
    windows = generate_walk_forward_windows(
        total_rows=len(rows),
        train_window=settings.walk_forward.train_window,
        validation_window=settings.walk_forward.validation_window,
        test_window=settings.walk_forward.test_window,
        step=settings.walk_forward.step,
    )
    folds: list[dict[str, Any]] = []
    thresholds: list[float] = []
    for window in windows:
        train_rows = rows[window.train_start : window.train_end]
        validation_rows = rows[window.validation_start : window.validation_end]
        test_rows = rows[window.test_start : window.test_end]
        if len(train_rows) < MIN_TRAIN_ROWS or len(validation_rows) < MIN_VALIDATION_ROWS or len(test_rows) < MIN_TEST_ROWS:
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
        thresholds.append(threshold)
        duration = _trade_duration_summary(trades)
        timeout_summary = _label_timeout_summary(test_rows)
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
                "time_exit_ratio": duration["time_exit_ratio"],
                "timeout_ratio": timeout_summary["timeout_ratio"],
            }
        )
    valid = [fold for fold in folds if not fold.get("skipped")]
    net_pnls = [_safe_float(fold["net_pnl_usd"]) for fold in valid]
    pfs = [_safe_float(fold["profit_factor"]) for fold in valid]
    no_trade = [_safe_float(fold["no_trade_ratio"]) for fold in valid]
    thresholds_valid = [_safe_float(fold["threshold"]) for fold in valid]
    return {
        "total_folds": len(folds),
        "valid_folds": len(valid),
        "fold_summaries": folds,
        "aggregate": {
            "total_net_pnl_usd": sum(net_pnls),
            "mean_profit_factor": sum(pfs) / len(pfs) if pfs else 0.0,
            "mean_no_trade_ratio": sum(no_trade) / len(no_trade) if no_trade else 0.0,
            "positive_fold_ratio": sum(1 for value in net_pnls if value > 0.0) / len(net_pnls) if net_pnls else 0.0,
            "net_pnl_stddev": pstdev(net_pnls) if len(net_pnls) > 1 else 0.0,
        },
        "threshold_stability": {
            "mean": sum(thresholds_valid) / len(thresholds_valid) if thresholds_valid else 0.0,
            "stddev": pstdev(thresholds_valid) if len(thresholds_valid) > 1 else 0.0,
            "min": min(thresholds_valid) if thresholds_valid else 0.0,
            "max": max(thresholds_valid) if thresholds_valid else 0.0,
        },
    }


def _variant_report(settings: Settings, symbol: str, variant: RealignmentVariant) -> dict[str, Any]:
    effective_settings, label_cfg = _effective_settings(settings, variant)
    rows = _build_symbol_rows(effective_settings, dataset_mode=variant.dataset_mode, symbol=symbol, label_cfg=label_cfg)
    split = temporal_train_validation_test_split(
        rows,
        effective_settings.split.train_ratio,
        effective_settings.split.validation_ratio,
        effective_settings.split.test_ratio,
    )
    feature_names = _feature_names("baseline")
    model, threshold, threshold_report = _train_model(effective_settings, split.train, split.validation, feature_names)
    test_matrix = [[row.features[name] for name in feature_names] for row in split.test]
    test_probabilities = model.predict_probabilities(test_matrix)
    predictions = apply_probability_threshold(test_probabilities, threshold)
    metrics, trades, _ = run_backtest_engine(
        rows=split.test,
        probabilities=test_probabilities,
        threshold=threshold,
        backtest=effective_settings.backtest,
        risk=effective_settings.risk,
        intrabar_policy=effective_settings.backtest.intrabar_policy,
        exit_policy_config=effective_settings.exit_policy,
        dynamic_exit_config=effective_settings.dynamic_exits,
    )
    timeout_summary = _label_timeout_summary(split.test)
    duration_summary = _trade_duration_summary(trades)
    return {
        "symbol": symbol,
        "variant_id": variant.variant_id,
        "hypothesis": variant.hypothesis,
        "dataset_mode": variant.dataset_mode,
        "labeling": {
            "horizon_bars": variant.horizon_bars,
            "take_profit_pct": variant.take_profit_pct,
            "stop_loss_pct": variant.stop_loss_pct,
            "timeout_handling_mode": variant.timeout_handling_mode,
            "timeout_direction_min_barrier_fraction": variant.timeout_direction_min_barrier_fraction,
        },
        "backtest_exit": {
            "max_holding_bars": variant.max_holding_bars,
            "tp_sl_ratio": variant.take_profit_pct / variant.stop_loss_pct if variant.stop_loss_pct > 0.0 else 0.0,
        },
        "threshold": threshold,
        "threshold_report": threshold_report,
        "test_metrics": {
            "trade_count": _safe_int(metrics.get("total_trades")),
            "net_pnl_usd": _safe_float(metrics.get("net_pnl_usd")),
            "expectancy_usd": _safe_float(metrics.get("expectancy_usd")),
            "profit_factor": _safe_float(metrics.get("profit_factor")),
            "max_drawdown_usd": _safe_float(metrics.get("max_drawdown_usd")),
            "no_trade_ratio": sum(1 for prediction in predictions if prediction == 0) / len(predictions) if predictions else 0.0,
        },
        "trade_duration_summary": duration_summary,
        "timeout_label_summary": timeout_summary,
        "threshold_utility": _threshold_utility(predictions, threshold, split.test, test_probabilities),
        "walk_forward": _walk_forward_report(effective_settings, rows, feature_names),
    }


def _trade_duration_diagnostic(settings: Settings) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    baseline = VARIANTS[0]
    for symbol in (FOCUS_SYMBOL, SECONDARY_SYMBOL):
        report = _variant_report(settings, symbol, baseline)
        payload[symbol] = {
            "baseline_variant_id": report["variant_id"],
            "label_horizon_bars": report["labeling"]["horizon_bars"],
            "max_holding_bars": report["backtest_exit"]["max_holding_bars"],
            "trade_duration_summary": report["trade_duration_summary"],
            "timeout_label_summary": report["timeout_label_summary"],
            "threshold_utility": report["threshold_utility"],
        }
    return payload


def run_audit_trade_duration(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "audit_trade_duration")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        payload = {
            "focus_symbol": FOCUS_SYMBOL,
            "secondary_symbol": SECONDARY_SYMBOL,
            "out_of_focus_symbols": list(OUT_OF_FOCUS_SYMBOLS),
            "environment_provenance": load_runtime_provenance_from_env(),
            "symbols": _trade_duration_diagnostic(settings),
        }
    except Exception as exc:
        logger.error(str(exc))
        return 1
    write_json_report(run_dir, "trade_duration_distribution_report.json", wrap_artifact("trade_duration_distribution_report", payload))
    write_json_report(run_dir, "horizon_alignment_report.json", wrap_artifact("horizon_alignment_report", payload))
    logger.info("audit_trade_duration=%s", run_dir)
    return 0


def run_audit_timeout_impact(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "audit_timeout_impact")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        baseline = VARIANTS[0]
        filtered = VARIANTS[3]
        payload = {
            "focus_symbol": FOCUS_SYMBOL,
            "secondary_symbol": SECONDARY_SYMBOL,
            "out_of_focus_symbols": list(OUT_OF_FOCUS_SYMBOLS),
            "environment_provenance": load_runtime_provenance_from_env(),
            "symbols": {
                symbol: {
                    baseline.variant_id: _variant_report(settings, symbol, baseline)["timeout_label_summary"],
                    filtered.variant_id: _variant_report(settings, symbol, filtered)["timeout_label_summary"],
                }
                for symbol in (FOCUS_SYMBOL, SECONDARY_SYMBOL)
            },
        }
    except Exception as exc:
        logger.error(str(exc))
        return 1
    write_json_report(run_dir, "timeout_label_impact_report.json", wrap_artifact("timeout_label_impact_report", payload))
    logger.info("audit_timeout_impact=%s", run_dir / "timeout_label_impact_report.json")
    return 0


def _select_best_focus_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        results,
        key=lambda item: (
            2 if item["decision"] == "CANDIDATE_FOR_DEMO_EXECUTION" else 1 if item["decision"] == "IMPROVED_BUT_NOT_ENOUGH" else 0,
            _safe_float(item["walk_forward"]["aggregate"]["total_net_pnl_usd"]),
            -_safe_float(item["timeout_label_summary"]["timeout_ratio"]),
            -_safe_float(item["trade_duration_summary"]["time_exit_ratio"]),
            _safe_float(item["test_metrics"]["profit_factor"]),
        ),
    )


def run_label_horizon_realignment(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "run_label_horizon_realignment")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        focus_results = [_variant_report(settings, FOCUS_SYMBOL, variant) for variant in VARIANTS]
        baseline_focus = next(item for item in focus_results if item["variant_id"] == "baseline_actual")
        focus_results = [
            {
                **item,
                **dict(zip(("decision", "decision_reasons"), _decision(settings, item, baseline_focus), strict=False)),
            }
            for item in focus_results
        ]
        best_focus = _select_best_focus_result(focus_results)
        control_variants = [variant for variant in VARIANTS if variant.variant_id in {CONTROL_VARIANT_IDS[0], best_focus["variant_id"]}]
        secondary_results = [_variant_report(settings, SECONDARY_SYMBOL, variant) for variant in control_variants]
        baseline_secondary = next(item for item in secondary_results if item["variant_id"] == "baseline_actual")
        secondary_results = [
            {
                **item,
                **dict(zip(("decision", "decision_reasons"), _decision(settings, item, baseline_secondary), strict=False)),
            }
            for item in secondary_results
        ]
        best_secondary = _select_best_focus_result(secondary_results)
    except Exception as exc:
        logger.error(str(exc))
        return 1

    duration_payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "out_of_focus_symbols": list(OUT_OF_FOCUS_SYMBOLS),
        "environment_provenance": load_runtime_provenance_from_env(),
        "symbols": {
            FOCUS_SYMBOL: {item["variant_id"]: item["trade_duration_summary"] for item in focus_results},
            SECONDARY_SYMBOL: {item["variant_id"]: item["trade_duration_summary"] for item in secondary_results},
        },
    }
    timeout_payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "out_of_focus_symbols": list(OUT_OF_FOCUS_SYMBOLS),
        "environment_provenance": load_runtime_provenance_from_env(),
        "symbols": {
            FOCUS_SYMBOL: {item["variant_id"]: item["timeout_label_summary"] for item in focus_results},
            SECONDARY_SYMBOL: {item["variant_id"]: item["timeout_label_summary"] for item in secondary_results},
        },
    }
    tp_sl_payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "environment_provenance": load_runtime_provenance_from_env(),
        "symbols": {
            FOCUS_SYMBOL: {
                item["variant_id"]: {
                    "labeling": item["labeling"],
                    "backtest_exit": item["backtest_exit"],
                    "trade_duration_summary": item["trade_duration_summary"],
                    "test_metrics": item["test_metrics"],
                }
                for item in focus_results
            },
            SECONDARY_SYMBOL: {
                item["variant_id"]: {
                    "labeling": item["labeling"],
                    "backtest_exit": item["backtest_exit"],
                    "trade_duration_summary": item["trade_duration_summary"],
                    "test_metrics": item["test_metrics"],
                }
                for item in secondary_results
            },
        },
    }
    threshold_payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "environment_provenance": load_runtime_provenance_from_env(),
        "symbols": {
            FOCUS_SYMBOL: {item["variant_id"]: item["threshold_utility"] | {"threshold_stability": item["walk_forward"]["threshold_stability"]} for item in focus_results},
            SECONDARY_SYMBOL: {item["variant_id"]: item["threshold_utility"] | {"threshold_stability": item["walk_forward"]["threshold_stability"]} for item in secondary_results},
        },
    }
    matrix_payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "out_of_focus_symbols": list(OUT_OF_FOCUS_SYMBOLS),
        "environment_provenance": load_runtime_provenance_from_env(),
        "variants": [asdict(item) for item in VARIANTS],
        "results_by_symbol": {
            FOCUS_SYMBOL: focus_results,
            SECONDARY_SYMBOL: secondary_results,
        },
        "selected_focus_variant": best_focus["variant_id"],
        "selected_secondary_variant": best_secondary["variant_id"],
    }
    recommendation_payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "out_of_focus_symbols": list(OUT_OF_FOCUS_SYMBOLS),
        "environment_provenance": load_runtime_provenance_from_env(),
        "selected_focus_variant": best_focus["variant_id"],
        "selected_secondary_variant": best_secondary["variant_id"],
        "focus_decision": best_focus["decision"],
        "secondary_decision": best_secondary["decision"],
        "structural_recommendation": (
            "Aceptar solo una mejora que reduzca timeout/time_exit y mejore walk-forward a la vez; si no, conservar REJECT_FOR_DEMO_EXECUTION."
        ),
    }
    candidate_payload = {
        "environment_provenance": load_runtime_provenance_from_env(),
        "focus_symbol": {
            "symbol": FOCUS_SYMBOL,
            "decision": best_focus["decision"],
            "variant_id": best_focus["variant_id"],
            "reasons": best_focus["decision_reasons"],
            "has_real_candidate": best_focus["decision"] == "CANDIDATE_FOR_DEMO_EXECUTION",
        },
        "secondary_symbol": {
            "symbol": SECONDARY_SYMBOL,
            "decision": best_secondary["decision"],
            "variant_id": best_secondary["variant_id"],
            "reasons": best_secondary["decision_reasons"],
            "has_real_candidate": best_secondary["decision"] == "CANDIDATE_FOR_DEMO_EXECUTION",
        },
        "approved_for_demo_execution_exists": False,
    }

    reports = {
        "trade_duration_distribution_report.json": wrap_artifact("trade_duration_distribution_report", duration_payload),
        "horizon_alignment_report.json": wrap_artifact("horizon_alignment_report", duration_payload),
        "timeout_label_impact_report.json": wrap_artifact("timeout_label_impact_report", timeout_payload),
        "tp_sl_alignment_report.json": wrap_artifact("tp_sl_alignment_report", tp_sl_payload),
        "threshold_utility_report.json": wrap_artifact("threshold_utility_report", threshold_payload),
        "label_horizon_exit_matrix_report.json": wrap_artifact("label_horizon_exit_matrix_report", matrix_payload),
        "edge_realignment_recommendation_report.json": wrap_artifact("edge_realignment_recommendation_report", recommendation_payload),
        "demo_execution_candidate_report.json": wrap_artifact("demo_execution_candidate_report", candidate_payload),
    }
    for filename, report in reports.items():
        write_json_report(run_dir, filename, report)
    logger.info("run_label_horizon_realignment=%s", run_dir)
    return 0


def run_compare_exit_alignment(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "compare_exit_alignment")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    source_dir = _latest_run_dir(settings, "run_label_horizon_realignment")
    if source_dir is None:
        logger.error("No hay corrida previa de run-label-horizon-realignment")
        return 1
    try:
        matrix = read_artifact_payload(source_dir / "label_horizon_exit_matrix_report.json", expected_type="label_horizon_exit_matrix_report")
    except Exception as exc:
        logger.error(str(exc))
        return 1
    compact = {
        "source_run_dir": str(source_dir),
        "focus_symbol": matrix["focus_symbol"],
        "secondary_symbol": matrix["secondary_symbol"],
        "selected_focus_variant": matrix["selected_focus_variant"],
        "selected_secondary_variant": matrix["selected_secondary_variant"],
        "focus_variant_scores": {
            item["variant_id"]: {
                "decision": item["decision"],
                "walk_forward_total_net_pnl_usd": item["walk_forward"]["aggregate"]["total_net_pnl_usd"],
                "test_profit_factor": item["test_metrics"]["profit_factor"],
                "timeout_ratio": item["timeout_label_summary"]["timeout_ratio"],
                "time_exit_ratio": item["trade_duration_summary"]["time_exit_ratio"],
            }
            for item in matrix["results_by_symbol"][FOCUS_SYMBOL]
        },
        "secondary_variant_scores": {
            item["variant_id"]: {
                "decision": item["decision"],
                "walk_forward_total_net_pnl_usd": item["walk_forward"]["aggregate"]["total_net_pnl_usd"],
                "test_profit_factor": item["test_metrics"]["profit_factor"],
                "timeout_ratio": item["timeout_label_summary"]["timeout_ratio"],
                "time_exit_ratio": item["trade_duration_summary"]["time_exit_ratio"],
            }
            for item in matrix["results_by_symbol"][SECONDARY_SYMBOL]
        },
    }
    write_json_report(run_dir, "label_horizon_exit_matrix_report.json", wrap_artifact("label_horizon_exit_matrix_report", compact))
    logger.info("compare_exit_alignment=%s", run_dir / "label_horizon_exit_matrix_report.json")
    return 0


def run_evaluate_label_horizon_candidate(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "evaluate_label_horizon_candidate")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    source_dir = _latest_run_dir(settings, "run_label_horizon_realignment")
    if source_dir is None:
        logger.error("No hay corrida previa de run-label-horizon-realignment")
        return 1
    try:
        candidate = read_artifact_payload(source_dir / "demo_execution_candidate_report.json", expected_type="demo_execution_candidate_report")
    except Exception as exc:
        logger.error(str(exc))
        return 1
    write_json_report(run_dir, "demo_execution_candidate_report.json", wrap_artifact("demo_execution_candidate_report", candidate))
    logger.info("evaluate_label_horizon_candidate=%s", run_dir / "demo_execution_candidate_report.json")
    return 0
