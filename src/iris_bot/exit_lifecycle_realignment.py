from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from statistics import mean, pstdev
from typing import Any

from iris_bot.artifacts import read_artifact_payload, wrap_artifact
from iris_bot.backtest import TradeRecord, run_backtest_engine
from iris_bot.config import LabelingConfig, Settings
from iris_bot.label_horizon_realignment import (
    _label_timeout_summary,
    _quantiles,
    _safe_float,
    _threshold_utility,
)
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
class ExitLifecycleVariant:
    variant_id: str
    hypothesis: str
    horizon_bars: int = 12
    label_take_profit_pct: float = 0.0020
    label_stop_loss_pct: float = 0.0020
    fixed_take_profit_pct: float = 0.0030
    fixed_stop_loss_pct: float = 0.0020
    max_holding_bars: int = 12
    timeout_handling_mode: str = "directional"
    timeout_direction_min_barrier_fraction: float = 0.0
    dataset_mode: str = "current"


VARIANTS: tuple[ExitLifecycleVariant, ...] = (
    ExitLifecycleVariant(
        variant_id="baseline_h12_actual",
        hypothesis="Referencia quirurgica: labels h12 0.20/0.20, target floor economico 0.30 y time exit 12.",
    ),
    ExitLifecycleVariant(
        variant_id="h12_timeexit_14",
        hypothesis="Mismo setup h12, pero con time exit un poco mas paciente en 14 barras.",
        max_holding_bars=14,
    ),
    ExitLifecycleVariant(
        variant_id="h12_target_025",
        hypothesis="Mismo h12 actual, pero target economico floor realineado de 0.30 a 0.25 para reducir time_exit sin tocar el stop.",
        fixed_take_profit_pct=0.0025,
    ),
    ExitLifecycleVariant(
        variant_id="h12_target_025_timeexit_14",
        hypothesis="Combina target floor mas cercano con time exit 14 para ver si baja time_exit sin colapsar muestra util.",
        fixed_take_profit_pct=0.0025,
        max_holding_bars=14,
    ),
    ExitLifecycleVariant(
        variant_id="h12_target_025_timeexit_14_timeout_soft",
        hypothesis="Misma combinacion, agregando timeout guard suave: timeout menor al 25% de barrera se vuelve neutral.",
        fixed_take_profit_pct=0.0025,
        max_holding_bars=14,
        timeout_handling_mode="neutral_by_barrier_fraction",
        timeout_direction_min_barrier_fraction=0.25,
    ),
)

CONTROL_VARIANT_IDS = ("baseline_h12_actual",)


def _trade_duration_summary(trades: list[TradeRecord]) -> dict[str, Any]:
    durations = [float(trade.bars_held) for trade in trades]
    exit_counts = Counter(trade.exit_reason for trade in trades)
    total = len(trades)
    tp_count = exit_counts.get("take_profit", 0) + exit_counts.get("take_profit_same_bar", 0)
    sl_count = exit_counts.get("stop_loss", 0) + exit_counts.get("stop_loss_same_bar", 0)
    return {
        "trade_count": total,
        "duration_bars": _quantiles(durations),
        "exit_reason_counts": dict(sorted(exit_counts.items())),
        "take_profit_ratio": tp_count / total if total else 0.0,
        "stop_loss_ratio": sl_count / total if total else 0.0,
        "time_exit_ratio": exit_counts.get("time_exit", 0) / total if total else 0.0,
        "end_of_data_ratio": exit_counts.get("end_of_data", 0) / total if total else 0.0,
        "time_exit_minus_tp_ratio": (exit_counts.get("time_exit", 0) - tp_count) / total if total else 0.0,
    }


def _exit_reason_metrics(trades: list[TradeRecord]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for exit_reason in sorted({trade.exit_reason for trade in trades}):
        subset = [trade for trade in trades if trade.exit_reason == exit_reason]
        total = len(subset)
        gross_profit = sum(trade.net_pnl_usd for trade in subset if trade.net_pnl_usd > 0.0)
        gross_loss = -sum(trade.net_pnl_usd for trade in subset if trade.net_pnl_usd < 0.0)
        payload[exit_reason] = {
            "trade_count": total,
            "net_pnl_usd": sum(trade.net_pnl_usd for trade in subset),
            "expectancy_usd": sum(trade.net_pnl_usd for trade in subset) / total if total else 0.0,
            "gross_profit_usd": gross_profit,
            "gross_loss_usd": gross_loss,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0),
            "mean_duration_bars": mean(float(trade.bars_held) for trade in subset) if subset else 0.0,
        }
    return payload


def _lifecycle_loss_analysis(result: dict[str, Any]) -> dict[str, Any]:
    exit_summary = result["trade_duration_summary"]
    timeout_summary = result["timeout_label_summary"]
    time_exit_ratio = _safe_float(exit_summary["time_exit_ratio"])
    tp_ratio = _safe_float(exit_summary["take_profit_ratio"])
    sl_ratio = _safe_float(exit_summary["stop_loss_ratio"])
    timeout_ratio = _safe_float(timeout_summary["timeout_ratio"])
    dominance_floor = 0.50
    return {
        "time_exit_ratio": time_exit_ratio,
        "take_profit_ratio": tp_ratio,
        "stop_loss_ratio": sl_ratio,
        "timeout_ratio": timeout_ratio,
        "time_exit_excess_over_non_dominant_floor": max(0.0, time_exit_ratio - dominance_floor),
        "timeout_excess_over_non_dominant_floor": max(0.0, timeout_ratio - dominance_floor),
        "time_exit_minus_tp_ratio": time_exit_ratio - tp_ratio,
        "time_exit_minus_sl_ratio": time_exit_ratio - sl_ratio,
        "selected_timeout_label_ratio": _safe_float(result["threshold_utility"]["selected_timeout_label_ratio"]),
        "lifecycle_failure_mode": (
            "time_exit_and_timeout_dominate"
            if time_exit_ratio >= 0.70 and timeout_ratio >= 0.60
            else "mixed_or_non_dominant"
        ),
    }


def _effective_settings(settings: Settings, variant: ExitLifecycleVariant) -> tuple[Settings, LabelingConfig]:
    label_cfg = replace(
        settings.labeling,
        horizon_bars=variant.horizon_bars,
        take_profit_pct=variant.label_take_profit_pct,
        stop_loss_pct=variant.label_stop_loss_pct,
        timeout_handling_mode=variant.timeout_handling_mode,
        timeout_direction_min_barrier_fraction=variant.timeout_direction_min_barrier_fraction,
    )
    effective = replace(
        settings,
        labeling=label_cfg,
        backtest=replace(
            settings.backtest,
            fixed_take_profit_pct=variant.fixed_take_profit_pct,
            fixed_stop_loss_pct=variant.fixed_stop_loss_pct,
            max_holding_bars=variant.max_holding_bars,
        ),
    )
    return effective, label_cfg


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
        duration_summary = _trade_duration_summary(trades)
        timeout_summary = _label_timeout_summary(test_rows)
        folds.append(
            {
                "fold_index": window.fold_index,
                "skipped": False,
                "threshold": threshold,
                "trade_count": int(_safe_float(metrics.get("total_trades"))),
                "net_pnl_usd": _safe_float(metrics.get("net_pnl_usd")),
                "profit_factor": _safe_float(metrics.get("profit_factor")),
                "expectancy_usd": _safe_float(metrics.get("expectancy_usd")),
                "no_trade_ratio": sum(1 for prediction in predictions if prediction == 0) / len(predictions) if predictions else 0.0,
                "timeout_ratio": timeout_summary["timeout_ratio"],
                "time_exit_ratio": duration_summary["time_exit_ratio"],
                "take_profit_ratio": duration_summary["take_profit_ratio"],
            }
        )
    valid = [fold for fold in folds if not fold.get("skipped")]
    net_pnls = [_safe_float(fold["net_pnl_usd"]) for fold in valid]
    pfs = [_safe_float(fold["profit_factor"]) for fold in valid]
    no_trade_ratios = [_safe_float(fold["no_trade_ratio"]) for fold in valid]
    return {
        "total_folds": len(folds),
        "valid_folds": len(valid),
        "fold_summaries": folds,
        "aggregate": {
            "total_net_pnl_usd": sum(net_pnls),
            "mean_profit_factor": sum(pfs) / len(pfs) if pfs else 0.0,
            "mean_no_trade_ratio": sum(no_trade_ratios) / len(no_trade_ratios) if no_trade_ratios else 0.0,
            "positive_fold_ratio": sum(1 for value in net_pnls if value > 0.0) / len(net_pnls) if net_pnls else 0.0,
            "net_pnl_stddev": pstdev(net_pnls) if len(net_pnls) > 1 else 0.0,
        },
        "threshold_stability": {
            "mean": sum(thresholds) / len(thresholds) if thresholds else 0.0,
            "stddev": pstdev(thresholds) if len(thresholds) > 1 else 0.0,
            "min": min(thresholds) if thresholds else 0.0,
            "max": max(thresholds) if thresholds else 0.0,
        },
    }


def _variant_report(settings: Settings, symbol: str, variant: ExitLifecycleVariant) -> dict[str, Any]:
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
    trade_duration_summary = _trade_duration_summary(trades)
    result = {
        "symbol": symbol,
        "variant_id": variant.variant_id,
        "hypothesis": variant.hypothesis,
        "dataset_mode": variant.dataset_mode,
        "labeling": {
            "horizon_bars": variant.horizon_bars,
            "take_profit_pct": variant.label_take_profit_pct,
            "stop_loss_pct": variant.label_stop_loss_pct,
            "timeout_handling_mode": variant.timeout_handling_mode,
            "timeout_direction_min_barrier_fraction": variant.timeout_direction_min_barrier_fraction,
        },
        "backtest_exit": {
            "fixed_take_profit_pct": variant.fixed_take_profit_pct,
            "fixed_stop_loss_pct": variant.fixed_stop_loss_pct,
            "max_holding_bars": variant.max_holding_bars,
            "target_stop_ratio": (
                variant.fixed_take_profit_pct / variant.fixed_stop_loss_pct if variant.fixed_stop_loss_pct > 0.0 else 0.0
            ),
        },
        "threshold": threshold,
        "threshold_report": threshold_report,
        "test_metrics": {
            "trade_count": int(_safe_float(metrics.get("total_trades"))),
            "net_pnl_usd": _safe_float(metrics.get("net_pnl_usd")),
            "expectancy_usd": _safe_float(metrics.get("expectancy_usd")),
            "profit_factor": _safe_float(metrics.get("profit_factor")),
            "max_drawdown_usd": _safe_float(metrics.get("max_drawdown_usd")),
            "no_trade_ratio": sum(1 for prediction in predictions if prediction == 0) / len(predictions) if predictions else 0.0,
        },
        "trade_duration_summary": trade_duration_summary,
        "timeout_label_summary": timeout_summary,
        "threshold_utility": _threshold_utility(predictions, threshold, split.test, test_probabilities),
        "exit_reason_metrics": _exit_reason_metrics(trades),
        "walk_forward": _walk_forward_report(effective_settings, rows, feature_names),
    }
    result["lifecycle_loss_analysis"] = _lifecycle_loss_analysis(result)
    return result


def _decision(settings: Settings, result: dict[str, Any], baseline: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    test_metrics = result["test_metrics"]
    walk_forward = result["walk_forward"]["aggregate"]
    timeout_ratio = _safe_float(result["timeout_label_summary"]["timeout_ratio"])
    time_exit_ratio = _safe_float(result["trade_duration_summary"]["time_exit_ratio"])
    tp_ratio = _safe_float(result["trade_duration_summary"]["take_profit_ratio"])
    selected_trade_ratio = _safe_float(result["threshold_utility"]["selected_trade_ratio"])
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
    if selected_trade_ratio <= 0.03:
        reasons.append("selected_trade_ratio_degenerate")
    if threshold_stddev > 0.08:
        reasons.append("threshold_fragility_still_high")
    if tp_ratio <= 0.10:
        reasons.append("take_profit_not_material")

    baseline_timeout = _safe_float(baseline["timeout_label_summary"]["timeout_ratio"])
    baseline_time_exit = _safe_float(baseline["trade_duration_summary"]["time_exit_ratio"])
    baseline_selected = _safe_float(baseline["threshold_utility"]["selected_trade_ratio"])
    baseline_wf = _safe_float(baseline["walk_forward"]["aggregate"]["total_net_pnl_usd"])
    improvement_points = 0
    if timeout_ratio <= baseline_timeout - 0.08:
        improvement_points += 1
    if time_exit_ratio <= baseline_time_exit - 0.15:
        improvement_points += 1
    if selected_trade_ratio >= baseline_selected:
        improvement_points += 1
    if _safe_float(walk_forward["total_net_pnl_usd"]) > baseline_wf:
        improvement_points += 1
    if _safe_float(test_metrics["profit_factor"]) > _safe_float(baseline["test_metrics"]["profit_factor"]):
        improvement_points += 1
    if tp_ratio >= _safe_float(baseline["trade_duration_summary"]["take_profit_ratio"]) + 0.10:
        improvement_points += 1

    if not reasons:
        return "CANDIDATE_FOR_DEMO_EXECUTION", reasons
    if result["variant_id"] != baseline["variant_id"] and improvement_points >= 4:
        return "IMPROVED_BUT_NOT_ENOUGH", reasons
    return "REJECT_FOR_DEMO_EXECUTION", reasons


def _overall_focus_conclusion(best_focus: dict[str, Any]) -> str:
    if best_focus["decision"] == "CANDIDATE_FOR_DEMO_EXECUTION":
        return "CANDIDATE_FOR_DEMO_EXECUTION"
    time_exit_ratio = _safe_float(best_focus["trade_duration_summary"]["time_exit_ratio"])
    timeout_ratio = _safe_float(best_focus["timeout_label_summary"]["timeout_ratio"])
    selected_trade_ratio = _safe_float(best_focus["threshold_utility"]["selected_trade_ratio"])
    if time_exit_ratio >= 0.75 and timeout_ratio >= 0.60:
        return "REQUIRE_DIFFERENT_SIGNAL_FAMILY"
    if selected_trade_ratio <= 0.03:
        return "REQUIRE_DIFFERENT_SIGNAL_FAMILY"
    return str(best_focus["decision"])


def _select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        results,
        key=lambda item: (
            2 if item["decision"] == "CANDIDATE_FOR_DEMO_EXECUTION" else 1 if item["decision"] == "IMPROVED_BUT_NOT_ENOUGH" else 0,
            _safe_float(item["walk_forward"]["aggregate"]["total_net_pnl_usd"]),
            _safe_float(item["test_metrics"]["profit_factor"]),
            -_safe_float(item["timeout_label_summary"]["timeout_ratio"]),
            -_safe_float(item["trade_duration_summary"]["time_exit_ratio"]),
            _safe_float(item["threshold_utility"]["selected_trade_ratio"]),
        ),
    )


def run_audit_exit_lifecycle(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "audit_exit_lifecycle")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        baseline = _variant_report(settings, FOCUS_SYMBOL, VARIANTS[0])
        control = _variant_report(settings, SECONDARY_SYMBOL, VARIANTS[0])
        payload = {
            "focus_symbol": FOCUS_SYMBOL,
            "secondary_symbol": SECONDARY_SYMBOL,
            "out_of_focus_symbols": list(OUT_OF_FOCUS_SYMBOLS),
            "environment_provenance": load_runtime_provenance_from_env(),
            "symbols": {
                FOCUS_SYMBOL: baseline,
                SECONDARY_SYMBOL: control,
            },
        }
    except Exception as exc:
        logger.error(str(exc))
        return 1
    write_json_report(run_dir, "exit_lifecycle_diagnostic_report.json", wrap_artifact("exit_lifecycle_diagnostic_report", payload))
    logger.info("audit_exit_lifecycle=%s", run_dir / "exit_lifecycle_diagnostic_report.json")
    return 0


def run_h12_exit_realignment(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "run_h12_exit_realignment")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        focus_results = [_variant_report(settings, FOCUS_SYMBOL, variant) for variant in VARIANTS]
        baseline_focus = next(item for item in focus_results if item["variant_id"] == "baseline_h12_actual")
        focus_results = [
            {
                **item,
                **dict(zip(("decision", "decision_reasons"), _decision(settings, item, baseline_focus), strict=False)),
            }
            for item in focus_results
        ]
        best_focus = _select_best_result(focus_results)
        overall_focus_conclusion = _overall_focus_conclusion(best_focus)
        control_variants = [VARIANTS[0], next(variant for variant in VARIANTS if variant.variant_id == best_focus["variant_id"])]
        seen: set[str] = set()
        deduped_variants: list[ExitLifecycleVariant] = []
        for variant in control_variants:
            if variant.variant_id in seen:
                continue
            seen.add(variant.variant_id)
            deduped_variants.append(variant)
        control_variants = deduped_variants
        secondary_results = [_variant_report(settings, SECONDARY_SYMBOL, variant) for variant in control_variants]
        baseline_secondary = next(item for item in secondary_results if item["variant_id"] == "baseline_h12_actual")
        secondary_results = [
            {
                **item,
                **dict(zip(("decision", "decision_reasons"), _decision(settings, item, baseline_secondary), strict=False)),
            }
            for item in secondary_results
        ]
        best_secondary = _select_best_result(secondary_results)
    except Exception as exc:
        logger.error(str(exc))
        return 1

    exit_diag_payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "out_of_focus_symbols": list(OUT_OF_FOCUS_SYMBOLS),
        "environment_provenance": load_runtime_provenance_from_env(),
        "symbols": {
            FOCUS_SYMBOL: baseline_focus,
            SECONDARY_SYMBOL: baseline_secondary,
        },
    }
    reduction_payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "environment_provenance": load_runtime_provenance_from_env(),
        "focus_reduction": {
            item["variant_id"]: {
                "timeout_ratio": item["timeout_label_summary"]["timeout_ratio"],
                "time_exit_ratio": item["trade_duration_summary"]["time_exit_ratio"],
                "take_profit_ratio": item["trade_duration_summary"]["take_profit_ratio"],
                "timeout_delta_vs_baseline": item["timeout_label_summary"]["timeout_ratio"] - baseline_focus["timeout_label_summary"]["timeout_ratio"],
                "time_exit_delta_vs_baseline": item["trade_duration_summary"]["time_exit_ratio"] - baseline_focus["trade_duration_summary"]["time_exit_ratio"],
                "take_profit_delta_vs_baseline": item["trade_duration_summary"]["take_profit_ratio"] - baseline_focus["trade_duration_summary"]["take_profit_ratio"],
            }
            for item in focus_results
        },
    }
    preservation_payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "environment_provenance": load_runtime_provenance_from_env(),
        "symbols": {
            FOCUS_SYMBOL: {
                item["variant_id"]: {
                    "trade_count": item["test_metrics"]["trade_count"],
                    "selected_trade_ratio": item["threshold_utility"]["selected_trade_ratio"],
                    "no_trade_ratio": item["test_metrics"]["no_trade_ratio"],
                    "threshold": item["threshold"],
                    "threshold_stability": item["walk_forward"]["threshold_stability"],
                }
                for item in focus_results
            },
            SECONDARY_SYMBOL: {
                item["variant_id"]: {
                    "trade_count": item["test_metrics"]["trade_count"],
                    "selected_trade_ratio": item["threshold_utility"]["selected_trade_ratio"],
                    "no_trade_ratio": item["test_metrics"]["no_trade_ratio"],
                    "threshold": item["threshold"],
                    "threshold_stability": item["walk_forward"]["threshold_stability"],
                }
                for item in secondary_results
            },
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
        "overall_focus_conclusion": overall_focus_conclusion,
    }
    recommendation_payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "out_of_focus_symbols": list(OUT_OF_FOCUS_SYMBOLS),
        "environment_provenance": load_runtime_provenance_from_env(),
        "selected_focus_variant": best_focus["variant_id"],
        "focus_variant_decision": best_focus["decision"],
        "overall_focus_conclusion": overall_focus_conclusion,
        "selected_secondary_variant": best_secondary["variant_id"],
        "secondary_variant_decision": best_secondary["decision"],
        "lifecycle_recommendation": (
            "Si el mejor h12 no baja materialmente time_exit/timeout sin destruir selected_trade_ratio, aceptar que la familia necesita otra señal."
        ),
    }
    candidate_payload = {
        "environment_provenance": load_runtime_provenance_from_env(),
        "focus_symbol": {
            "symbol": FOCUS_SYMBOL,
            "decision": best_focus["decision"],
            "overall_conclusion": overall_focus_conclusion,
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
        "exit_lifecycle_diagnostic_report.json": wrap_artifact("exit_lifecycle_diagnostic_report", exit_diag_payload),
        "timeout_timeexit_reduction_report.json": wrap_artifact("timeout_timeexit_reduction_report", reduction_payload),
        "trade_count_preservation_report.json": wrap_artifact("trade_count_preservation_report", preservation_payload),
        "h12_exit_variant_matrix_report.json": wrap_artifact("h12_exit_variant_matrix_report", matrix_payload),
        "edge_lifecycle_recommendation_report.json": wrap_artifact("edge_lifecycle_recommendation_report", recommendation_payload),
        "demo_execution_candidate_report.json": wrap_artifact("demo_execution_candidate_report", candidate_payload),
    }
    for filename, report in reports.items():
        write_json_report(run_dir, filename, report)
    logger.info("run_h12_exit_realignment=%s", run_dir)
    return 0


def run_compare_h12_exit_variants(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "compare_h12_exit_variants")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    source_dir = _latest_run_dir(settings, "run_h12_exit_realignment")
    if source_dir is None:
        logger.error("No hay corrida previa de run-h12-exit-realignment")
        return 1
    try:
        matrix = read_artifact_payload(source_dir / "h12_exit_variant_matrix_report.json", expected_type="h12_exit_variant_matrix_report")
    except Exception as exc:
        logger.error(str(exc))
        return 1
    compact = {
        "source_run_dir": str(source_dir),
        "focus_symbol": matrix["focus_symbol"],
        "secondary_symbol": matrix["secondary_symbol"],
        "selected_focus_variant": matrix["selected_focus_variant"],
        "overall_focus_conclusion": matrix["overall_focus_conclusion"],
        "focus_variant_scores": {
            item["variant_id"]: {
                "decision": item["decision"],
                "walk_forward_total_net_pnl_usd": item["walk_forward"]["aggregate"]["total_net_pnl_usd"],
                "test_profit_factor": item["test_metrics"]["profit_factor"],
                "trade_count": item["test_metrics"]["trade_count"],
                "selected_trade_ratio": item["threshold_utility"]["selected_trade_ratio"],
                "timeout_ratio": item["timeout_label_summary"]["timeout_ratio"],
                "time_exit_ratio": item["trade_duration_summary"]["time_exit_ratio"],
                "take_profit_ratio": item["trade_duration_summary"]["take_profit_ratio"],
            }
            for item in matrix["results_by_symbol"][FOCUS_SYMBOL]
        },
        "secondary_variant_scores": {
            item["variant_id"]: {
                "decision": item["decision"],
                "walk_forward_total_net_pnl_usd": item["walk_forward"]["aggregate"]["total_net_pnl_usd"],
                "test_profit_factor": item["test_metrics"]["profit_factor"],
                "trade_count": item["test_metrics"]["trade_count"],
                "timeout_ratio": item["timeout_label_summary"]["timeout_ratio"],
                "time_exit_ratio": item["trade_duration_summary"]["time_exit_ratio"],
            }
            for item in matrix["results_by_symbol"][SECONDARY_SYMBOL]
        },
    }
    write_json_report(run_dir, "h12_exit_variant_matrix_report.json", wrap_artifact("h12_exit_variant_matrix_report", compact))
    logger.info("compare_h12_exit_variants=%s", run_dir / "h12_exit_variant_matrix_report.json")
    return 0


def run_evaluate_gbpusd_demo_candidate(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "evaluate_gbpusd_demo_candidate")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    source_dir = _latest_run_dir(settings, "run_h12_exit_realignment")
    if source_dir is None:
        logger.error("No hay corrida previa de run-h12-exit-realignment")
        return 1
    try:
        candidate = read_artifact_payload(source_dir / "demo_execution_candidate_report.json", expected_type="demo_execution_candidate_report")
    except Exception as exc:
        logger.error(str(exc))
        return 1
    write_json_report(run_dir, "demo_execution_candidate_report.json", wrap_artifact("demo_execution_candidate_report", candidate))
    logger.info("evaluate_gbpusd_demo_candidate=%s", run_dir / "demo_execution_candidate_report.json")
    return 0
