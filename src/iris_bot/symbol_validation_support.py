from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from iris_bot.backtest import run_backtest_engine
from iris_bot.config import Settings
from iris_bot.exits import SymbolExitProfile
from iris_bot.metrics import classification_metrics
from iris_bot.processed_dataset import ProcessedDataset, ProcessedRow, load_processed_dataset
from iris_bot.splits import temporal_train_validation_test_split
from iris_bot.sessions import canonical_session_name
from iris_bot.walk_forward import generate_walk_forward_windows
from iris_bot.xgb_model import XGBoostMultiClassModel


@dataclass(frozen=True)
class EconomicThresholdResult:
    threshold: float
    metric_name: str
    metric_value: float
    total_trades: int
    no_trade_ratio: float
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ThresholdSweepResult:
    threshold: float
    retained_signal_count: int
    excluded_opportunities: int
    no_trade_ratio: float
    metric_value: float
    total_trades: int
    blocked_entry_count: int
    net_pnl_usd: float
    profit_factor: float
    max_drawdown_usd: float
    expectancy_usd: float


@dataclass(frozen=True)
class LeakageSafeSplit:
    fit_train: list[ProcessedRow]
    fit_validation: list[ProcessedRow]
    selection: list[ProcessedRow]
    final_test: list[ProcessedRow]
    reports: dict[str, Any]


def _rows_to_matrix(rows: list[ProcessedRow], feature_names: list[str]) -> tuple[list[list[float]], list[int]]:
    return [[row.features[name] for name in feature_names] for row in rows], [row.label for row in rows]


def _metric_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _metric_int(metrics: dict[str, Any], key: str, default: int = 0) -> int:
    value = metrics.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _prune_feature_names(rows: list[ProcessedRow], feature_names: list[str], settings: Settings) -> tuple[list[str], dict[str, Any]]:
    kept: list[str] = []
    pruned: list[dict[str, Any]] = []
    for name in feature_names:
        values = [row.features[name] for row in rows]
        if not values:
            continue
        variance = 0.0
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        ordered = sorted(values)
        dominant_count = max(ordered.count(ordered[0]), ordered.count(ordered[-1]))
        dominance = dominant_count / len(values)
        if variance <= settings.strategy.feature_min_variance:
            pruned.append({"feature": name, "reason": "low_variance", "variance": variance})
            continue
        if dominance >= settings.strategy.feature_dominance_threshold:
            pruned.append({"feature": name, "reason": "dominant_value", "dominance": dominance})
            continue
        kept.append(name)
    return kept or feature_names, {"kept": kept or feature_names, "pruned": pruned}


def _predict_probabilities(
    model: XGBoostMultiClassModel,
    rows: list[ProcessedRow],
    feature_names: list[str],
) -> list[dict[int, float]]:
    matrix, _ = _rows_to_matrix(rows, feature_names)
    return model.predict_probabilities(matrix)


def _build_leakage_safe_split(rows: list[ProcessedRow], settings: Settings) -> LeakageSafeSplit:
    outer_split = temporal_train_validation_test_split(
        rows,
        settings.split.train_ratio,
        settings.split.validation_ratio,
        settings.split.test_ratio,
    )
    train_rows = outer_split.train
    validation_rows = outer_split.validation
    test_rows = outer_split.test
    fit_cut = max(1, int(len(train_rows) * 0.8))
    if fit_cut >= len(train_rows):
        fit_cut = max(1, len(train_rows) - 1)
    fit_train = train_rows[:fit_cut]
    fit_validation = train_rows[fit_cut:]
    return LeakageSafeSplit(
        fit_train=fit_train,
        fit_validation=fit_validation,
        selection=validation_rows,
        final_test=test_rows,
        reports={
            "fit_train_rows": len(fit_train),
            "fit_validation_rows": len(fit_validation),
            "selection_rows": len(validation_rows),
            "final_test_rows": len(test_rows),
            "leakage_safe": True,
        },
    )


def _confidence_bins(probabilities: list[dict[int, float]], labels: list[int]) -> list[dict[str, Any]]:
    bins = [(0.0, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.01)]
    payload: list[dict[str, Any]] = []
    for low, high in bins:
        scores = []
        correct = 0
        for probs, label in zip(probabilities, labels, strict=False):
            prediction = max(probs.items(), key=lambda item: item[1])[0]
            confidence = max(probs.values())
            if low <= confidence < high:
                scores.append(confidence)
                if prediction == label:
                    correct += 1
        payload.append(
            {
                "low": low,
                "high": high,
                "count": len(scores),
                "avg_confidence": 0.0 if not scores else sum(scores) / len(scores),
                "accuracy": 0.0 if not scores else correct / len(scores),
            }
        )
    return payload


def _build_trade_session_breakdown(trades: list[Any]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0, "net_pnl_usd": 0.0})
    for trade in trades:
        session = canonical_session_name(datetime.fromisoformat(trade.entry_timestamp))
        buckets[session]["trades"] += 1
        buckets[session]["net_pnl_usd"] += trade.net_pnl_usd
    return dict(buckets)


def _threshold_predictions(probabilities: list[dict[int, float]], threshold: float) -> list[int]:
    predictions: list[int] = []
    for probs in probabilities:
        long_score = probs.get(1, 0.0)
        short_score = probs.get(-1, 0.0)
        neutral_score = probs.get(0, 0.0)
        if long_score >= threshold and long_score > short_score and long_score >= neutral_score:
            predictions.append(1)
        elif short_score >= threshold and short_score > long_score and short_score >= neutral_score:
            predictions.append(-1)
        else:
            predictions.append(0)
    return predictions


def _score_distribution(probabilities: list[dict[int, float]]) -> dict[str, Any]:
    if not probabilities:
        return {
            "count": 0,
            "avg_max_confidence": 0.0,
            "min_max_confidence": 0.0,
            "max_max_confidence": 0.0,
            "argmax_class_counts": {"-1": 0, "0": 0, "1": 0},
        }
    max_scores = [max(item.values()) for item in probabilities]
    argmax_counts = {"-1": 0, "0": 0, "1": 0}
    for item in probabilities:
        argmax_counts[str(max(item.items(), key=lambda pair: pair[1])[0])] += 1
    return {
        "count": len(probabilities),
        "avg_max_confidence": sum(max_scores) / len(max_scores),
        "min_max_confidence": min(max_scores),
        "max_max_confidence": max(max_scores),
        "argmax_class_counts": argmax_counts,
    }


def _evaluate_threshold_grid(
    rows: list[ProcessedRow],
    probabilities: list[dict[int, float]],
    settings: Settings,
    exit_policy: SymbolExitProfile,
) -> list[ThresholdSweepResult]:
    sweep: list[ThresholdSweepResult] = []
    for threshold in settings.threshold.grid:
        local_settings = dataclasses.replace(
            settings,
            exit_policy=dataclasses.replace(
                settings.exit_policy,
                stop_policy=exit_policy.stop_policy,
                target_policy=exit_policy.target_policy,
            ),
        )
        metrics, _, _ = run_backtest_engine(
            rows=rows,
            probabilities=probabilities,
            threshold=threshold,
            backtest=local_settings.backtest,
            risk=local_settings.risk,
            intrabar_policy=local_settings.backtest.intrabar_policy,
            exit_policy_config=local_settings.exit_policy,
            dynamic_exit_config=local_settings.dynamic_exits,
            symbol_exit_profiles={rows[0].symbol: exit_policy} if rows else {},
        )
        predictions = _threshold_predictions(probabilities, threshold)
        no_trade_ratio = 0.0 if not predictions else predictions.count(0) / len(predictions)
        retained_signal_count = len(predictions) - predictions.count(0)
        sweep.append(
            ThresholdSweepResult(
                threshold=threshold,
                retained_signal_count=retained_signal_count,
                excluded_opportunities=len(predictions) - retained_signal_count,
                no_trade_ratio=no_trade_ratio,
                metric_value=_metric_float(metrics, "expectancy_usd"),
                total_trades=_metric_int(metrics, "total_trades"),
                blocked_entry_count=_metric_int(metrics, "blocked_entry_count"),
                net_pnl_usd=_metric_float(metrics, "net_pnl_usd"),
                profit_factor=_metric_float(metrics, "profit_factor"),
                max_drawdown_usd=_metric_float(metrics, "max_drawdown_usd"),
                expectancy_usd=_metric_float(metrics, "expectancy_usd"),
            )
        )
    return sweep


def _select_threshold_by_economics(
    rows: list[ProcessedRow],
    probabilities: list[dict[int, float]],
    settings: Settings,
    exit_policy: SymbolExitProfile,
) -> tuple[EconomicThresholdResult, list[dict[str, Any]]]:
    best: EconomicThresholdResult | None = None
    sweep = _evaluate_threshold_grid(rows, probabilities, settings, exit_policy)
    for item in sweep:
        candidate = EconomicThresholdResult(
            threshold=item.threshold,
            metric_name=settings.strategy.threshold_metric,
            metric_value=item.metric_value,
            total_trades=item.total_trades,
            no_trade_ratio=item.no_trade_ratio,
            metrics={
                "expectancy_usd": item.expectancy_usd,
                "net_pnl_usd": item.net_pnl_usd,
                "profit_factor": item.profit_factor,
                "max_drawdown_usd": item.max_drawdown_usd,
                "blocked_entry_count": item.blocked_entry_count,
                "total_trades": item.total_trades,
            },
        )
        if best is None:
            best = candidate
            continue
        current_score = (
            candidate.metric_value,
            candidate.metrics.get("net_pnl_usd", 0.0),
            -candidate.metrics.get("max_drawdown_usd", 0.0),
            -candidate.no_trade_ratio,
        )
        best_score = (
            best.metric_value,
            best.metrics.get("net_pnl_usd", 0.0),
            -best.metrics.get("max_drawdown_usd", 0.0),
            -best.no_trade_ratio,
        )
        if current_score > best_score:
            best = candidate
    assert best is not None
    return best, [asdict(item) for item in sweep]


def _evaluate_configuration(
    model_name: str,
    model: XGBoostMultiClassModel,
    fit_train_rows: list[ProcessedRow],
    fit_validation_rows: list[ProcessedRow],
    selection_rows: list[ProcessedRow],
    final_test_rows: list[ProcessedRow],
    feature_names: list[str],
    settings: Settings,
    symbol: str,
) -> dict[str, Any]:
    train_matrix, train_labels = _rows_to_matrix(fit_train_rows, feature_names)
    validation_matrix, validation_labels = _rows_to_matrix(fit_validation_rows, feature_names)
    model.fit(train_matrix, train_labels, validation_matrix, validation_labels)
    selection_probs = _predict_probabilities(model, selection_rows, feature_names)
    test_probs = _predict_probabilities(model, final_test_rows, feature_names)
    static_profile = SymbolExitProfile(
        stop_policy="static",
        target_policy="static",
        stop_atr_multiplier=settings.risk.atr_stop_loss_multiplier,
        target_atr_multiplier=settings.risk.atr_take_profit_multiplier,
    )
    dynamic_profile = SymbolExitProfile(
        stop_policy="atr_dynamic",
        target_policy="atr_dynamic",
        stop_atr_multiplier=settings.risk.atr_stop_loss_multiplier,
        target_atr_multiplier=settings.risk.atr_take_profit_multiplier,
        stop_min_pct=settings.dynamic_exits.min_stop_loss_pct,
        stop_max_pct=settings.dynamic_exits.max_stop_loss_pct,
        target_min_pct=settings.dynamic_exits.min_take_profit_pct,
        target_max_pct=settings.dynamic_exits.max_take_profit_pct,
    )
    static_threshold, static_threshold_sweep = _select_threshold_by_economics(selection_rows, selection_probs, settings, static_profile)
    dynamic_threshold, dynamic_threshold_sweep = _select_threshold_by_economics(selection_rows, selection_probs, settings, dynamic_profile)

    def evaluate_rows(
        rows: list[ProcessedRow],
        probabilities: list[dict[int, float]],
        profile: SymbolExitProfile,
        threshold: float,
    ) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
        local_settings = dataclasses.replace(
            settings,
            exit_policy=dataclasses.replace(
                settings.exit_policy,
                stop_policy=profile.stop_policy,
                target_policy=profile.target_policy,
            ),
        )
        metrics, trades, _ = run_backtest_engine(
            rows=rows,
            probabilities=probabilities,
            threshold=threshold,
            backtest=local_settings.backtest,
            risk=local_settings.risk,
            intrabar_policy=local_settings.backtest.intrabar_policy,
            exit_policy_config=local_settings.exit_policy,
            dynamic_exit_config=local_settings.dynamic_exits,
            symbol_exit_profiles={symbol: profile},
        )
        _, labels = _rows_to_matrix(rows, feature_names)
        predictions = _threshold_predictions(probabilities, threshold)
        return metrics, trades, classification_metrics(labels, predictions)

    selection_static_metrics, selection_static_trades, selection_static_classification = evaluate_rows(
        selection_rows, selection_probs, static_profile, static_threshold.threshold
    )
    selection_dynamic_metrics, selection_dynamic_trades, selection_dynamic_classification = evaluate_rows(
        selection_rows, selection_probs, dynamic_profile, dynamic_threshold.threshold
    )
    final_static_metrics, final_static_trades, final_static_classification = evaluate_rows(
        final_test_rows, test_probs, static_profile, static_threshold.threshold
    )
    final_dynamic_metrics, final_dynamic_trades, final_dynamic_classification = evaluate_rows(
        final_test_rows, test_probs, dynamic_profile, dynamic_threshold.threshold
    )

    dynamic_is_better = (
        selection_dynamic_metrics.get("net_pnl_usd", 0.0),
        selection_dynamic_metrics.get("expectancy_usd", 0.0),
        -selection_dynamic_metrics.get("max_drawdown_usd", 0.0),
    ) > (
        selection_static_metrics.get("net_pnl_usd", 0.0),
        selection_static_metrics.get("expectancy_usd", 0.0),
        -selection_static_metrics.get("max_drawdown_usd", 0.0),
    )

    return {
        "model_name": model_name,
        "feature_names": feature_names,
        "selection_protocol": {
            "fit_train_rows": len(fit_train_rows),
            "fit_validation_rows": len(fit_validation_rows),
            "selection_rows": len(selection_rows),
            "final_test_rows": len(final_test_rows),
            "test_used_for_selection": False,
        },
        "thresholds": {
            "static": asdict(static_threshold),
            "atr_dynamic": asdict(dynamic_threshold),
        },
        "threshold_sensitivity": {
            "selection_score_distribution": _score_distribution(selection_probs),
            "final_test_score_distribution": _score_distribution(test_probs),
            "static": static_threshold_sweep,
            "atr_dynamic": dynamic_threshold_sweep,
        },
        "calibration_review": {
            "selection_bins": _confidence_bins(selection_probs, [row.label for row in selection_rows]),
            "test_bins": _confidence_bins(test_probs, [row.label for row in final_test_rows]),
        },
        "class_balance": {
            "train": {"-1": train_labels.count(-1), "0": train_labels.count(0), "1": train_labels.count(1)},
            "validation": {"-1": validation_labels.count(-1), "0": validation_labels.count(0), "1": validation_labels.count(1)},
            "selection": {"-1": sum(1 for row in selection_rows if row.label == -1), "0": sum(1 for row in selection_rows if row.label == 0), "1": sum(1 for row in selection_rows if row.label == 1)},
            "test": {"-1": sum(1 for row in final_test_rows if row.label == -1), "0": sum(1 for row in final_test_rows if row.label == 0), "1": sum(1 for row in final_test_rows if row.label == 1)},
        },
        "static": {
            "selection_evaluation": {
                "economic_metrics": selection_static_metrics,
                "classification_metrics": selection_static_classification,
                "session_breakdown": _build_trade_session_breakdown(selection_static_trades),
            },
            "final_test_evaluation": {
                "economic_metrics": final_static_metrics,
                "classification_metrics": final_static_classification,
                "session_breakdown": _build_trade_session_breakdown(final_static_trades),
            },
        },
        "atr_dynamic": {
            "selection_evaluation": {
                "economic_metrics": selection_dynamic_metrics,
                "classification_metrics": selection_dynamic_classification,
                "session_breakdown": _build_trade_session_breakdown(selection_dynamic_trades),
            },
            "final_test_evaluation": {
                "economic_metrics": final_dynamic_metrics,
                "classification_metrics": final_dynamic_classification,
                "session_breakdown": _build_trade_session_breakdown(final_dynamic_trades),
            },
        },
        "preferred_exit_policy": "atr_dynamic" if dynamic_is_better else "static",
    }


def _choose_best_model(comparison: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    winner_name = "global_model"
    winner_payload = comparison["global_model"]
    for name, payload in comparison.items():
        candidate_policy = payload[payload["preferred_exit_policy"]]["selection_evaluation"]["economic_metrics"]
        winner_policy = winner_payload[winner_payload["preferred_exit_policy"]]["selection_evaluation"]["economic_metrics"]
        candidate_score = (
            candidate_policy.get("net_pnl_usd", 0.0),
            candidate_policy.get("expectancy_usd", 0.0),
            -candidate_policy.get("max_drawdown_usd", 0.0),
        )
        winner_score = (
            winner_policy.get("net_pnl_usd", 0.0),
            winner_policy.get("expectancy_usd", 0.0),
            -winner_policy.get("max_drawdown_usd", 0.0),
        )
        if candidate_score > winner_score:
            winner_name = name
            winner_payload = payload
    return winner_name, winner_payload


def _symbol_walkforward(
    symbol_rows: list[ProcessedRow],
    all_rows: list[ProcessedRow],
    chosen_model_name: str,
    chosen_feature_names: list[str],
    chosen_exit_policy: str,
    settings: Settings,
    symbol: str,
) -> dict[str, Any]:
    windows = generate_walk_forward_windows(
        total_rows=len(symbol_rows),
        train_window=settings.walk_forward.train_window,
        validation_window=settings.walk_forward.validation_window,
        test_window=settings.walk_forward.test_window,
        step=settings.walk_forward.step,
    )
    if not windows:
        return {"valid_folds": 0, "positive_folds": 0, "folds": []}
    folds: list[dict[str, Any]] = []
    positive = 0
    for window in windows:
        symbol_train = symbol_rows[window.train_start : window.train_end]
        symbol_val = symbol_rows[window.validation_start : window.validation_end]
        symbol_test = symbol_rows[window.test_start : window.test_end]
        if len(symbol_train) < 30 or len(symbol_val) < 10 or len(symbol_test) < 5:
            continue
        if chosen_model_name == "global_model":
            train_start = symbol_train[0].timestamp
            train_end = symbol_train[-1].timestamp
            val_start = symbol_val[0].timestamp
            val_end = symbol_val[-1].timestamp
            global_train = [row for row in all_rows if train_start <= row.timestamp <= train_end]
            global_val = [row for row in all_rows if val_start <= row.timestamp <= val_end]
            train_rows = global_train
            val_rows = global_val
        else:
            train_rows = symbol_train
            val_rows = symbol_val
        train_matrix, train_labels = _rows_to_matrix(train_rows, chosen_feature_names)
        val_matrix, val_labels = _rows_to_matrix(val_rows, chosen_feature_names)
        model = XGBoostMultiClassModel(settings.xgboost)
        model.fit(train_matrix, train_labels, val_matrix, val_labels)
        val_probs = _predict_probabilities(model, symbol_val, chosen_feature_names)
        test_probs = _predict_probabilities(model, symbol_test, chosen_feature_names)
        exit_profile = SymbolExitProfile(
            stop_policy=chosen_exit_policy,
            target_policy=chosen_exit_policy,
            stop_atr_multiplier=settings.risk.atr_stop_loss_multiplier,
            target_atr_multiplier=settings.risk.atr_take_profit_multiplier,
            stop_min_pct=settings.dynamic_exits.min_stop_loss_pct,
            stop_max_pct=settings.dynamic_exits.max_stop_loss_pct,
            target_min_pct=settings.dynamic_exits.min_take_profit_pct,
            target_max_pct=settings.dynamic_exits.max_take_profit_pct,
        )
        threshold, _ = _select_threshold_by_economics(symbol_val, val_probs, settings, exit_profile)
        local_settings = dataclasses.replace(
            settings,
            exit_policy=dataclasses.replace(
                settings.exit_policy,
                stop_policy=chosen_exit_policy,
                target_policy=chosen_exit_policy,
            ),
        )
        metrics, _, _ = run_backtest_engine(
            rows=symbol_test,
            probabilities=test_probs,
            threshold=threshold.threshold,
            backtest=local_settings.backtest,
            risk=local_settings.risk,
            intrabar_policy=local_settings.backtest.intrabar_policy,
            exit_policy_config=local_settings.exit_policy,
            dynamic_exit_config=local_settings.dynamic_exits,
            symbol_exit_profiles={symbol: exit_profile},
        )
        if _metric_float(metrics, "net_pnl_usd") > 0.0:
            positive += 1
        folds.append(
            {
                "fold_index": window.fold_index,
                "threshold": threshold.threshold,
                "metrics": metrics,
            }
        )
    return {
        "valid_folds": len(folds),
        "positive_folds": positive,
        "positive_ratio": 0.0 if not folds else positive / len(folds),
        "folds": folds,
    }


def _enablement_decision(
    symbol: str,
    row_count: int,
    chosen_model: str,
    chosen_payload: dict[str, Any],
    walkforward: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    chosen_policy = chosen_payload[chosen_payload["preferred_exit_policy"]]["selection_evaluation"]["economic_metrics"]
    reasons: list[str] = []
    state = "enabled"
    if row_count < settings.strategy.min_symbol_rows:
        state = "disabled"
        reasons.append("insufficient_symbol_rows")
    if chosen_policy.get("total_trades", 0) < settings.strategy.min_validation_trades:
        state = "disabled"
        reasons.append("insufficient_selection_trades")
    if chosen_policy.get("expectancy_usd", 0.0) <= settings.strategy.min_expectancy_usd:
        state = "disabled"
        reasons.append("expectancy_not_positive")
    if chosen_policy.get("profit_factor", 0.0) < settings.strategy.min_profit_factor:
        state = "disabled"
        reasons.append("profit_factor_below_gate")
    if chosen_policy.get("max_drawdown_usd", 0.0) > settings.strategy.max_drawdown_usd:
        state = "disabled"
        reasons.append("drawdown_above_limit")
    if walkforward.get("valid_folds", 0) == 0:
        state = "disabled"
        reasons.append("walkforward_not_available")
    elif walkforward.get("positive_ratio", 0.0) < settings.strategy.min_positive_walkforward_ratio:
        state = "caution" if state != "disabled" else state
        reasons.append("walkforward_positive_ratio_low")
    if chosen_payload["thresholds"][chosen_payload["preferred_exit_policy"]]["no_trade_ratio"] >= settings.strategy.caution_no_trade_ratio and state == "enabled":
        state = "caution"
        reasons.append("no_trade_ratio_high")
    return {
        "symbol": symbol,
        "state": state,
        "enabled": state == "enabled",
        "chosen_model": chosen_model,
        "chosen_exit_policy": chosen_payload["preferred_exit_policy"],
        "reasons": reasons,
        "selection_based": True,
    }


def _gate_status(actual: float, gate: float, *, comparator: str) -> dict[str, Any]:
    if comparator == "min":
        passed = actual >= gate
        gap = actual - gate
    else:
        passed = actual <= gate
        gap = gate - actual
    if passed:
        severity = "pass"
    elif abs(gap) <= max(abs(gate) * 0.1, 1e-9):
        severity = "near_miss"
    else:
        severity = "hard_fail"
    return {"actual": actual, "gate": gate, "passed": passed, "gap": gap, "severity": severity}


def _block_reason_matrix(
    symbol: str,
    row_count: int,
    chosen_payload: dict[str, Any],
    walkforward: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    chosen_exit_policy = chosen_payload["preferred_exit_policy"]
    chosen_metrics = chosen_payload[chosen_exit_policy]["selection_evaluation"]["economic_metrics"]
    threshold_payload = chosen_payload["thresholds"][chosen_exit_policy]
    reasons = {
        "insufficient_data": _gate_status(float(row_count), float(settings.strategy.min_symbol_rows), comparator="min"),
        "insufficient_selection_trades": _gate_status(float(chosen_metrics.get("total_trades", 0)), float(settings.strategy.min_validation_trades), comparator="min"),
        "expectancy_non_positive": _gate_status(float(chosen_metrics.get("expectancy_usd", 0.0)), float(settings.strategy.min_expectancy_usd), comparator="min"),
        "profit_factor_below_floor": _gate_status(float(chosen_metrics.get("profit_factor", 0.0)), float(settings.strategy.min_profit_factor), comparator="min"),
        "drawdown_above_limit": _gate_status(float(chosen_metrics.get("max_drawdown_usd", 0.0)), float(settings.strategy.max_drawdown_usd), comparator="max"),
        "walkforward_instability": _gate_status(float(walkforward.get("positive_ratio", 0.0)), float(settings.strategy.min_positive_walkforward_ratio), comparator="min"),
        "no_trade_ratio_too_high": _gate_status(float(threshold_payload.get("no_trade_ratio", 0.0)), float(settings.strategy.caution_no_trade_ratio), comparator="max"),
    }
    ordered_failures = [{"reason": reason, **payload} for reason, payload in reasons.items() if not payload["passed"]]
    severity_rank = {"hard_fail": 0, "near_miss": 1, "pass": 2}
    ordered_failures.sort(key=lambda item: severity_rank.get(item["severity"], 9))
    dominant = ordered_failures[0]["reason"] if ordered_failures else "none"
    return {
        "symbol": symbol,
        "chosen_exit_policy": chosen_exit_policy,
        "row_count": row_count,
        "reasons": reasons,
        "ordered_failures": ordered_failures,
        "dominant_reason": dominant,
        "selection_total_trades": int(chosen_metrics.get("total_trades", 0)),
        "selection_expectancy_usd": float(chosen_metrics.get("expectancy_usd", 0.0)),
        "selection_profit_factor": float(chosen_metrics.get("profit_factor", 0.0)),
        "selection_max_drawdown_usd": float(chosen_metrics.get("max_drawdown_usd", 0.0)),
        "selection_no_trade_ratio": float(threshold_payload.get("no_trade_ratio", 0.0)),
        "walkforward_valid_folds": int(walkforward.get("valid_folds", 0)),
        "walkforward_positive_ratio": float(walkforward.get("positive_ratio", 0.0)),
    }


def _session_timeframe_diagnostics(
    symbol: str,
    chosen_payload: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    chosen_exit_policy = chosen_payload["preferred_exit_policy"]
    chosen_eval = chosen_payload[chosen_exit_policy]
    return {
        "symbol": symbol,
        "chosen_exit_policy": chosen_exit_policy,
        "allowed_timeframes": [settings.trading.primary_timeframe],
        "selection_session_breakdown": chosen_eval["selection_evaluation"]["session_breakdown"],
        "final_test_session_breakdown": chosen_eval["final_test_evaluation"]["session_breakdown"],
        "timeframe_assessment": {
            settings.trading.primary_timeframe: {
                "selection_total_trades": chosen_eval["selection_evaluation"]["economic_metrics"].get("total_trades", 0),
                "final_test_total_trades": chosen_eval["final_test_evaluation"]["economic_metrics"].get("total_trades", 0),
            }
        },
    }


def _exit_policy_diagnostics(symbol: str, chosen_payload: dict[str, Any]) -> dict[str, Any]:
    static_selection = chosen_payload["static"]["selection_evaluation"]["economic_metrics"]
    dynamic_selection = chosen_payload["atr_dynamic"]["selection_evaluation"]["economic_metrics"]
    static_test = chosen_payload["static"]["final_test_evaluation"]["economic_metrics"]
    dynamic_test = chosen_payload["atr_dynamic"]["final_test_evaluation"]["economic_metrics"]
    return {
        "symbol": symbol,
        "preferred_exit_policy": chosen_payload["preferred_exit_policy"],
        "selection": {
            "static": static_selection,
            "atr_dynamic": dynamic_selection,
            "delta_net_pnl_usd": dynamic_selection.get("net_pnl_usd", 0.0) - static_selection.get("net_pnl_usd", 0.0),
            "delta_expectancy_usd": dynamic_selection.get("expectancy_usd", 0.0) - static_selection.get("expectancy_usd", 0.0),
        },
        "final_test": {
            "static": static_test,
            "atr_dynamic": dynamic_test,
            "delta_net_pnl_usd": dynamic_test.get("net_pnl_usd", 0.0) - static_test.get("net_pnl_usd", 0.0),
            "delta_expectancy_usd": dynamic_test.get("expectancy_usd", 0.0) - static_test.get("expectancy_usd", 0.0),
        },
    }


def _recommendation_for_symbol(decision: dict[str, Any], block_matrix: dict[str, Any]) -> dict[str, Any]:
    if decision["state"] == "enabled":
        recommendation = "KEEP_VALIDATED"
        confidence = "medium"
        next_action = "run governance validation and require endurance evidence before any approval"
    elif decision["state"] == "caution":
        recommendation = "MOVE_TO_CAUTION"
        confidence = "medium"
        next_action = "keep out of approval and gather more endurance evidence"
    else:
        dominant = block_matrix["dominant_reason"]
        recommendation = "KEEP_BLOCKED"
        confidence = "high" if dominant != "none" else "medium"
        if dominant == "profit_factor_below_floor":
            next_action = "verify profit-factor gate with fresh validation outputs and keep blocked if still below floor"
        elif dominant == "walkforward_instability":
            next_action = "retest in sandbox by subcontext before any governance transition"
        else:
            next_action = "keep out of demo promotion until evidence improves"
    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "next_action": next_action,
    }


def _load_dataset(settings: Settings) -> ProcessedDataset:
    return load_processed_dataset(
        settings.experiment.processed_dataset_path,
        settings.experiment.processed_schema_path,
        settings.experiment.processed_manifest_path,
    )


def _primary_rows(dataset: ProcessedDataset, settings: Settings) -> list[ProcessedRow]:
    rows = [row for row in dataset.rows if row.timeframe == settings.trading.primary_timeframe]
    rows.sort(key=lambda row: (row.timestamp, row.symbol))
    return rows
