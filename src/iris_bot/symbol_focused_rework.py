"""
Symbol-focused quantitative rework.

Applies a structured, isolated variant comparison to a single focus symbol
to determine whether a real tradeable edge exists. The process:

  1. Select focus symbol with explicit justification.
  2. Diagnose label quality, feature redundancy, and fold-level behavior.
  3. Run a small, well-chosen variant matrix (not a parameter sweep).
  4. Apply hard economic gates to each variant.
  5. Report REJECT / IMPROVED / CANDIDATE — default is REJECT.

No operational layer is touched. No model is promoted without earning it.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Any

from iris_bot.artifacts import wrap_artifact
from iris_bot.backtest import run_backtest_engine
from iris_bot.config import LabelingConfig, Settings
from iris_bot.data import load_bars
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.processed_dataset import FEATURE_NAMES_BASE, ProcessedRow, build_processed_dataset
from iris_bot.splits import temporal_train_validation_test_split
from iris_bot.thresholds import apply_probability_threshold, select_threshold_from_probabilities
from iris_bot.walk_forward import generate_walk_forward_windows
from iris_bot.xgb_model import XGBoostMultiClassModel


# ---------------------------------------------------------------------------
# Focus symbol selection — justified by diagnostic evidence.
#
# GBPUSD:
#   - Walk-forward total PnL = -24.11 USD (closest to breakeven).
#   - 4/14 folds strongly positive (PF 4.1, 2.8, 1.9, 2.3) → conditional edge exists.
#   - Clean labels (TP/SL hits) are balanced: 52.4% long / 47.6% short.
#   - Timeout labels are short-biased (56.1% short) due to 3-month downtrend period.
#   - Dominant failure cause: noisy timeout-direction labels + wrong threshold metric.
#
# EURUSD (secondary):
#   - Test PnL = -3.05 USD (essentially breakeven in test).
#   - WF total = -96 USD (too variable to be primary focus).
#   - Monitored but not deeply reworked in this phase.
#
# USDJPY: excluded per project constraints (operational blocking, not quant focus).
# AUDUSD: deepest losses, WF total = -234 USD, no conditional edge visible.
# ---------------------------------------------------------------------------
FOCUS_SYMBOL = "GBPUSD"
SECONDARY_SYMBOL = "EURUSD"

# ---------------------------------------------------------------------------
# Feature pruning — confirmed duplicates/noise.
#
# Confirmed by inspection of processed dataset:
#   return_3 == momentum_3 (max_diff = 0.0, exact duplicate)
#   return_5 == momentum_5 (max_diff = 0.0, exact duplicate)
#   return_1 ≈ log_return_1 (max_diff = 1.07e-6, numerical near-duplicate)
#   usd_strength_index ≈ 0 (stdev = 0.001, no informative signal)
#   return_autocorr_3, return_autocorr_5: noisy short lags (keep only _10)
#
# Removing 6 features: 29 → 23.
# ---------------------------------------------------------------------------
_FEATURES_TO_PRUNE: frozenset[str] = frozenset({
    "return_1",
    "return_3",
    "return_5",
    "return_autocorr_3",
    "return_autocorr_5",
    "usd_strength_index",
})

# ---------------------------------------------------------------------------
# Variant specifications.
#
# Each variant changes exactly one or two things vs baseline.
# Changes are isolated to allow attribution of improvement/damage.
# ---------------------------------------------------------------------------
_VARIANTS: list[dict[str, Any]] = [
    {
        "variant_id": "V1_baseline",
        "hypothesis": "Current symbol-specific baseline. Establishes comparison floor.",
        "label_tp_pct": 0.0020,
        "label_sl_pct": 0.0020,
        "label_horizon_bars": 8,
        "feature_set": "full",
        "threshold_metric": "macro_f1",
    },
    {
        "variant_id": "V2_economic_threshold",
        "hypothesis": "Same labels/features as baseline, but threshold selected by economic expectancy "
                      "on validation instead of macro_f1. Fixes mismatch between training objective "
                      "and economic objective.",
        "label_tp_pct": 0.0020,
        "label_sl_pct": 0.0020,
        "label_horizon_bars": 8,
        "feature_set": "full",
        "threshold_metric": "economic",
    },
    {
        "variant_id": "V3_asymmetric_tp",
        "hypothesis": "Widen TP from 0.20% to 0.30% (SL stays 0.20%). Gives 1.5:1 raw RR. "
                      "Reduces required win rate for breakeven. More bars get clean TP labels "
                      "instead of timeout-direction noise. Economic threshold selection.",
        "label_tp_pct": 0.0030,
        "label_sl_pct": 0.0020,
        "label_horizon_bars": 8,
        "feature_set": "full",
        "threshold_metric": "economic",
    },
    {
        "variant_id": "V4_patient_horizon",
        "hypothesis": "Extend horizon from 8 to 12 bars. TP=0.25% SL=0.25%. Gives price "
                      "more time to reach barriers, reducing noisy timeout labels. "
                      "Economic threshold selection.",
        "label_tp_pct": 0.0025,
        "label_sl_pct": 0.0025,
        "label_horizon_bars": 12,
        "feature_set": "full",
        "threshold_metric": "economic",
    },
    {
        "variant_id": "V5_combined_pruned",
        "hypothesis": "Best structural changes combined: asymmetric TP (0.30/0.20%), patient "
                      "horizon (12 bars), pruned feature set (remove 6 confirmed redundant/noisy "
                      "features), economic threshold selection.",
        "label_tp_pct": 0.0030,
        "label_sl_pct": 0.0020,
        "label_horizon_bars": 12,
        "feature_set": "pruned",
        "threshold_metric": "economic",
    },
]

_MIN_TRAIN_ROWS = 30
_MIN_VALIDATION_ROWS = 10
_MIN_WF_TEST_ROWS = 5
_MIN_TRADES_FOR_ECONOMIC_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Feature utilities
# ---------------------------------------------------------------------------

def _feature_names(*, pruned: bool) -> list[str]:
    if pruned:
        return [f for f in FEATURE_NAMES_BASE if f not in _FEATURES_TO_PRUNE]
    return list(FEATURE_NAMES_BASE)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _build_symbol_rows(settings: Settings, label_cfg: LabelingConfig) -> list[ProcessedRow]:
    bars = load_bars(settings.data.raw_dataset_path)
    if not bars:
        raise FileNotFoundError(f"No bars at {settings.data.raw_dataset_path}")
    dataset = build_processed_dataset(bars, label_cfg)
    rows = [r for r in dataset.rows if r.symbol == FOCUS_SYMBOL and r.timeframe == settings.trading.primary_timeframe]
    rows.sort(key=lambda r: r.timestamp)
    return rows


# ---------------------------------------------------------------------------
# Economic threshold selection
# ---------------------------------------------------------------------------

def _select_threshold_economic(
    probabilities: list[dict[int, float]],
    rows: list[ProcessedRow],
    grid: tuple[float, ...],
    settings: Settings,
    min_trades: int = _MIN_TRADES_FOR_ECONOMIC_THRESHOLD,
) -> tuple[float, float]:
    """Select the threshold that maximizes economic expectancy on the given rows.

    Only considers thresholds that achieve at least min_trades trades.
    Falls back to the first grid value if nothing qualifies.
    Returns (threshold, best_expectancy_usd).
    """
    best_threshold = grid[0]
    best_expectancy = float("-inf")
    found = False

    for threshold in grid:
        metrics, _, _ = run_backtest_engine(
            rows=rows,
            probabilities=probabilities,
            threshold=threshold,
            backtest=settings.backtest,
            risk=settings.risk,
            intrabar_policy=settings.backtest.intrabar_policy,
            exit_policy_config=settings.exit_policy,
            dynamic_exit_config=settings.dynamic_exits,
        )
        if int(metrics.get("total_trades", 0)) < min_trades:
            continue
        expectancy = float(metrics.get("expectancy_usd", float("-inf")))
        if expectancy > best_expectancy:
            best_expectancy = expectancy
            best_threshold = threshold
            found = True

    if not found:
        return grid[0], 0.0
    return best_threshold, best_expectancy


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _economic_weights(rows: list[ProcessedRow], cap: float = 3.0) -> list[float]:
    atrs = [row.features.get("atr_5", 0.0) for row in rows]
    sorted_atrs = sorted(atrs)
    median_atr = sorted_atrs[len(sorted_atrs) // 2] if sorted_atrs else 0.0
    if median_atr <= 0.0:
        return [1.0] * len(rows)
    return [min(atr / median_atr, cap) for atr in atrs]


def _train_model(
    settings: Settings,
    train_rows: list[ProcessedRow],
    validation_rows: list[ProcessedRow],
    feature_names: list[str],
    threshold_metric: str,
) -> tuple[XGBoostMultiClassModel, float, dict[str, Any]]:
    """Train one XGBoost model and select threshold.

    threshold_metric: "economic" uses backtest-based selection.
                      Any other value is passed to select_threshold_from_probabilities.
    """
    train_matrix = [[row.features[name] for name in feature_names] for row in train_rows]
    validation_matrix = [[row.features[name] for name in feature_names] for row in validation_rows]
    train_labels = [row.label for row in train_rows]
    validation_labels = [row.label for row in validation_rows]
    weights = _economic_weights(train_rows)

    model = XGBoostMultiClassModel(settings.xgboost)
    model.fit(
        train_matrix,
        train_labels,
        validation_matrix,
        validation_labels,
        feature_names=feature_names,
        sample_weights=weights,
    )
    validation_probabilities = model.predict_probabilities(validation_matrix)

    if threshold_metric == "economic":
        threshold, metric_value = _select_threshold_economic(
            validation_probabilities, validation_rows, settings.threshold.grid, settings
        )
        threshold_report: dict[str, Any] = {
            "threshold": threshold,
            "metric_name": "economic_expectancy",
            "metric_value": metric_value,
        }
    else:
        result = select_threshold_from_probabilities(
            probabilities=validation_probabilities,
            labels=validation_labels,
            grid=settings.threshold.grid,
            metric_name=threshold_metric,
            refinement_steps=settings.threshold.refinement_steps,
        )
        threshold_report = asdict(result)
        threshold = result.threshold

    return model, threshold, threshold_report


# ---------------------------------------------------------------------------
# Economic evaluation
# ---------------------------------------------------------------------------

def _evaluate_rows(
    settings: Settings,
    rows: list[ProcessedRow],
    probabilities: list[dict[int, float]],
    threshold: float,
) -> dict[str, Any]:
    predictions = apply_probability_threshold(probabilities, threshold)
    metrics, _, _ = run_backtest_engine(
        rows=rows,
        probabilities=probabilities,
        threshold=threshold,
        backtest=settings.backtest,
        risk=settings.risk,
        intrabar_policy=settings.backtest.intrabar_policy,
        exit_policy_config=settings.exit_policy,
        dynamic_exit_config=settings.dynamic_exits,
    )
    no_trade_ratio = Counter(predictions).get(0, 0) / len(predictions) if predictions else 1.0
    return {
        "row_count": len(rows),
        "trade_count": int(metrics["total_trades"]),
        "net_pnl_usd": float(metrics["net_pnl_usd"]),
        "expectancy_usd": float(metrics["expectancy_usd"]),
        "profit_factor": float(metrics["profit_factor"]),
        "max_drawdown_usd": float(metrics["max_drawdown_usd"]),
        "no_trade_ratio": no_trade_ratio,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Walk-forward per variant
# ---------------------------------------------------------------------------

def _walk_forward_variant(
    settings: Settings,
    rows: list[ProcessedRow],
    feature_names: list[str],
    threshold_metric: str,
) -> dict[str, Any]:
    windows = generate_walk_forward_windows(
        total_rows=len(rows),
        train_window=settings.walk_forward.train_window,
        validation_window=settings.walk_forward.validation_window,
        test_window=settings.walk_forward.test_window,
        step=settings.walk_forward.step,
    )
    summaries: list[dict[str, Any]] = []
    for window in windows:
        train_rows = rows[window.train_start: window.train_end]
        validation_rows = rows[window.validation_start: window.validation_end]
        test_rows = rows[window.test_start: window.test_end]
        if (
            len(train_rows) < _MIN_TRAIN_ROWS
            or len(validation_rows) < _MIN_VALIDATION_ROWS
            or len(test_rows) < _MIN_WF_TEST_ROWS
        ):
            summaries.append({
                "fold_index": window.fold_index,
                "skipped": True,
                "reason": "insufficient_rows",
            })
            continue
        model, threshold, threshold_report = _train_model(
            settings, train_rows, validation_rows, feature_names, threshold_metric
        )
        test_matrix = [[row.features[name] for name in feature_names] for row in test_rows]
        probabilities = model.predict_probabilities(test_matrix)
        fold_metrics = _evaluate_rows(settings, test_rows, probabilities, threshold)
        summaries.append({
            "fold_index": window.fold_index,
            "skipped": False,
            "train_start": train_rows[0].timestamp.isoformat() if train_rows else "",
            "test_end": test_rows[-1].timestamp.isoformat() if test_rows else "",
            "best_iteration": model.best_iteration,
            "threshold_report": threshold_report,
            **fold_metrics,
        })

    valid = [s for s in summaries if not s.get("skipped")]
    net_pnls = [float(s["net_pnl_usd"]) for s in valid]
    pfs = [float(s["profit_factor"]) for s in valid]
    expectancies = [float(s["expectancy_usd"]) for s in valid]
    no_trades = [float(s["no_trade_ratio"]) for s in valid]
    drawdowns = [float(s["max_drawdown_usd"]) for s in valid]
    trade_counts = [int(s["trade_count"]) for s in valid]

    positive_folds = sum(1 for p in net_pnls if p > 0.0)
    valid_count = len(valid)

    return {
        "total_folds": len(summaries),
        "valid_folds": valid_count,
        "positive_folds": positive_folds,
        "positive_fold_ratio": positive_folds / valid_count if valid_count else 0.0,
        "fold_summaries": summaries,
        "aggregate": {
            "total_net_pnl_usd": sum(net_pnls),
            "mean_net_pnl_usd": mean(net_pnls) if net_pnls else 0.0,
            "mean_profit_factor": mean(pfs) if pfs else 0.0,
            "mean_expectancy_usd": mean(expectancies) if expectancies else 0.0,
            "worst_fold_drawdown_usd": max(drawdowns) if drawdowns else 0.0,
            "mean_no_trade_ratio": mean(no_trades) if no_trades else 0.0,
            "net_pnl_stddev": pstdev(net_pnls) if len(net_pnls) > 1 else 0.0,
            "total_trades": sum(trade_counts),
        },
    }


# ---------------------------------------------------------------------------
# Label diagnostic (no training)
# ---------------------------------------------------------------------------

def _label_diagnostic(rows: list[ProcessedRow]) -> dict[str, Any]:
    label_counts = Counter(r.label for r in rows)
    reason_counts = Counter(r.label_reason for r in rows)
    total = len(rows)

    # Classify labels by quality: TP/SL hits are clean; timeout labels are noisy
    clean_labels = [r.label for r in rows if r.label_reason in ("triple_barrier_take_profit", "triple_barrier_stop_loss")]
    timeout_dir_labels = [r.label for r in rows if r.label_reason == "triple_barrier_timeout_direction"]
    timeout_neutral_labels = [r.label for r in rows if r.label_reason == "triple_barrier_timeout_small_move"]

    clean_counts = Counter(clean_labels)
    timeout_dir_counts = Counter(timeout_dir_labels)
    clean_total = len(clean_labels)
    timeout_dir_total = len(timeout_dir_labels)

    return {
        "total_rows": total,
        "label_distribution": {
            str(label): {
                "count": label_counts.get(label, 0),
                "ratio": label_counts.get(label, 0) / total if total else 0.0,
            }
            for label in (-1, 0, 1)
        },
        "label_reason_counts": dict(reason_counts),
        "clean_signal_ratio": clean_total / total if total else 0.0,
        "clean_signal": {
            "count": clean_total,
            "long_ratio": clean_counts.get(1, 0) / clean_total if clean_total else 0.0,
            "short_ratio": clean_counts.get(-1, 0) / clean_total if clean_total else 0.0,
        },
        "timeout_direction_signal": {
            "count": timeout_dir_total,
            "long_ratio": timeout_dir_counts.get(1, 0) / timeout_dir_total if timeout_dir_total else 0.0,
            "short_ratio": timeout_dir_counts.get(-1, 0) / timeout_dir_total if timeout_dir_total else 0.0,
        },
        "timeout_neutral_count": len(timeout_neutral_labels),
        "label_quality_assessment": (
            "high_noise: timeout_direction_labels exceed 50% of dataset"
            if timeout_dir_total / total > 0.50 and total
            else "acceptable"
        ),
    }


# ---------------------------------------------------------------------------
# Feature redundancy report (no training)
# ---------------------------------------------------------------------------

def _feature_redundancy_report(rows: list[ProcessedRow]) -> dict[str, Any]:
    def _max_abs_diff(feat_a: str, feat_b: str) -> float:
        diffs = [abs(r.features[feat_a] - r.features[feat_b]) for r in rows if feat_a in r.features and feat_b in r.features]
        return max(diffs) if diffs else float("nan")

    def _feature_stdev(feat: str) -> float:
        vals = [r.features[feat] for r in rows if feat in r.features]
        return stdev(vals) if len(vals) > 1 else 0.0

    confirmed_duplicates = {
        "return_3_vs_momentum_3": _max_abs_diff("return_3", "momentum_3"),
        "return_5_vs_momentum_5": _max_abs_diff("return_5", "momentum_5"),
        "return_1_vs_log_return_1": _max_abs_diff("return_1", "log_return_1"),
    }
    near_zero_features = {
        "usd_strength_index": _feature_stdev("usd_strength_index"),
    }
    pruned = list(_FEATURES_TO_PRUNE)
    kept = _feature_names(pruned=True)
    return {
        "total_features": len(FEATURE_NAMES_BASE),
        "pruned_features": pruned,
        "kept_features_count": len(kept),
        "confirmed_duplicates_max_diff": confirmed_duplicates,
        "near_zero_stdev_features": near_zero_features,
        "pruning_rationale": "Remove exact/near duplicates and near-constant features to reduce noise.",
    }


# ---------------------------------------------------------------------------
# Session-stratified analysis
# ---------------------------------------------------------------------------

def _session_breakdown(rows: list[ProcessedRow]) -> dict[str, Any]:
    sessions = {"asia": "session_asia", "london": "session_london", "new_york": "session_new_york"}
    result: dict[str, Any] = {}
    for name, feature in sessions.items():
        session_rows = [r for r in rows if r.features.get(feature, 0.0) > 0.0]
        if not session_rows:
            result[name] = {"count": 0}
            continue
        labels = Counter(r.label for r in session_rows)
        total = len(session_rows)
        result[name] = {
            "count": total,
            "ratio_of_all": total / len(rows) if rows else 0.0,
            "label_distribution": {
                str(label): labels.get(label, 0) / total if total else 0.0
                for label in (-1, 0, 1)
            },
        }
    return result


# ---------------------------------------------------------------------------
# Economic gate logic
# ---------------------------------------------------------------------------

_DEMO_CANDIDATE_SOFT_WF_FLOOR = -15.0


def _apply_variant_gates(
    settings: Settings,
    test_metrics: dict[str, Any],
    walk_forward: dict[str, Any],
) -> tuple[str, list[str]]:
    """Apply hard economic gates. Default: REJECT.

    States:
      APPROVED_FOR_DEMO_EXECUTION  — all gates pass (never forced, must be earned).
      CANDIDATE_FOR_DEMO_EXECUTION — test gates pass, WF slightly soft.
      IMPROVED_BUT_NOT_ENOUGH      — improved vs baseline but still fails gates.
      REJECT_FOR_DEMO_EXECUTION    — default if no state above is reached.
    """
    reasons: list[str] = []
    gate = settings.approved_demo_gate

    # Test gates
    if test_metrics["trade_count"] < gate.min_trade_count:
        reasons.append("test_trade_count_below_floor")
    if test_metrics["expectancy_usd"] <= 0.0:
        reasons.append("test_expectancy_non_positive")
    if test_metrics["profit_factor"] < gate.min_profit_factor:
        reasons.append("test_profit_factor_below_floor")
    if test_metrics["max_drawdown_usd"] > settings.strategy.max_drawdown_usd:
        reasons.append("test_drawdown_above_floor")
    if test_metrics["no_trade_ratio"] > gate.max_no_trade_ratio:
        reasons.append("test_no_trade_ratio_above_floor")

    # Walk-forward gates
    agg = walk_forward["aggregate"]
    valid_folds = int(walk_forward["valid_folds"])
    positive_ratio = float(walk_forward["positive_fold_ratio"])

    if valid_folds < 2:
        reasons.append("walk_forward_folds_insufficient")
    if agg["total_net_pnl_usd"] <= 0.0:
        reasons.append("walk_forward_total_net_pnl_non_positive")
    if agg["mean_expectancy_usd"] <= 0.0:
        reasons.append("walk_forward_expectancy_non_positive")
    if agg["mean_profit_factor"] < gate.min_profit_factor:
        reasons.append("walk_forward_profit_factor_below_floor")
    if positive_ratio < settings.strategy.min_positive_walkforward_ratio:
        reasons.append("walk_forward_positive_ratio_below_floor")
    if agg["worst_fold_drawdown_usd"] > settings.strategy.max_drawdown_usd:
        reasons.append("walk_forward_drawdown_above_floor")

    if not reasons:
        return "APPROVED_FOR_DEMO_EXECUTION", reasons

    # CANDIDATE: test passes, WF slightly soft (total PnL > floor, not non-positive)
    wf_soft_reasons = {
        "walk_forward_total_net_pnl_non_positive",
        "walk_forward_expectancy_non_positive",
        "walk_forward_profit_factor_below_floor",
        "walk_forward_positive_ratio_below_floor",
    }
    test_hard_failures = {r for r in reasons if not r.startswith("walk_forward_")}
    wf_failures = {r for r in reasons if r.startswith("walk_forward_")}

    if (
        not test_hard_failures
        and wf_failures.issubset(wf_soft_reasons)
        and test_metrics["expectancy_usd"] > 0.0
        and test_metrics["profit_factor"] >= 1.0
        and agg["total_net_pnl_usd"] >= _DEMO_CANDIDATE_SOFT_WF_FLOOR
    ):
        return "CANDIDATE_FOR_DEMO_EXECUTION", reasons

    # IMPROVED_BUT_NOT_ENOUGH: test expectancy > 0 but gates not all met
    if test_metrics["expectancy_usd"] > 0.0 and not test_hard_failures:
        return "IMPROVED_BUT_NOT_ENOUGH", reasons

    return "REJECT_FOR_DEMO_EXECUTION", reasons


# ---------------------------------------------------------------------------
# Single variant runner
# ---------------------------------------------------------------------------

def _run_one_variant(
    settings: Settings,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Run a full train/test/walk-forward cycle for one variant spec."""
    label_cfg = replace(
        settings.labeling,
        take_profit_pct=spec["label_tp_pct"],
        stop_loss_pct=spec["label_sl_pct"],
        horizon_bars=spec["label_horizon_bars"],
    )
    effective_settings = replace(settings, labeling=label_cfg)

    rows = _build_symbol_rows(effective_settings, label_cfg)
    feature_names = _feature_names(pruned=(spec["feature_set"] == "pruned"))

    if len(rows) < (_MIN_TRAIN_ROWS + _MIN_VALIDATION_ROWS + _MIN_WF_TEST_ROWS):
        return {
            "variant_id": spec["variant_id"],
            "skipped": True,
            "reason": "insufficient_rows",
            "row_count": len(rows),
        }

    split = temporal_train_validation_test_split(
        rows,
        settings.split.train_ratio,
        settings.split.validation_ratio,
        settings.split.test_ratio,
    )
    model, threshold, threshold_report = _train_model(
        effective_settings, split.train, split.validation, feature_names, spec["threshold_metric"]
    )
    test_matrix = [[row.features[name] for name in feature_names] for row in split.test]
    test_probabilities = model.predict_probabilities(test_matrix)
    test_metrics = _evaluate_rows(effective_settings, split.test, test_probabilities, threshold)

    walk_forward = _walk_forward_variant(
        effective_settings, rows, feature_names, spec["threshold_metric"]
    )
    decision, reasons = _apply_variant_gates(effective_settings, test_metrics, walk_forward)

    label_diag = _label_diagnostic(rows)

    return {
        "variant_id": spec["variant_id"],
        "hypothesis": spec["hypothesis"],
        "skipped": False,
        "feature_set": spec["feature_set"],
        "feature_count": len(feature_names),
        "threshold_metric": spec["threshold_metric"],
        "label_config": {
            "take_profit_pct": spec["label_tp_pct"],
            "stop_loss_pct": spec["label_sl_pct"],
            "horizon_bars": spec["label_horizon_bars"],
        },
        "label_diagnostic": label_diag,
        "threshold_report": threshold_report,
        "test_metrics": test_metrics,
        "walk_forward": walk_forward,
        "decision": decision,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Comparison summary across variants
# ---------------------------------------------------------------------------

def _compare_variants(variant_results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [v for v in variant_results if not v.get("skipped")]
    if not valid:
        return {"best_variant_id": None, "ranking": []}

    def _score(v: dict[str, Any]) -> float:
        """Simple ranking score: normalized economic metrics."""
        t = v["test_metrics"]
        wf = v["walk_forward"]["aggregate"]
        # Primary: test expectancy + WF expectancy + WF total PnL (scaled)
        return (
            t["expectancy_usd"]
            + wf["mean_expectancy_usd"]
            + wf["total_net_pnl_usd"] / 100.0
        )

    decision_rank = {
        "APPROVED_FOR_DEMO_EXECUTION": 0,
        "CANDIDATE_FOR_DEMO_EXECUTION": 1,
        "IMPROVED_BUT_NOT_ENOUGH": 2,
        "REJECT_FOR_DEMO_EXECUTION": 3,
    }

    ranked = sorted(
        valid,
        key=lambda v: (decision_rank.get(v["decision"], 3), -_score(v)),
    )

    ranking = [
        {
            "rank": i + 1,
            "variant_id": v["variant_id"],
            "decision": v["decision"],
            "test_pnl_usd": v["test_metrics"]["net_pnl_usd"],
            "test_expectancy_usd": v["test_metrics"]["expectancy_usd"],
            "test_profit_factor": v["test_metrics"]["profit_factor"],
            "test_trades": v["test_metrics"]["trade_count"],
            "wf_total_pnl_usd": v["walk_forward"]["aggregate"]["total_net_pnl_usd"],
            "wf_mean_expectancy_usd": v["walk_forward"]["aggregate"]["mean_expectancy_usd"],
            "wf_positive_folds": v["walk_forward"]["positive_folds"],
            "wf_valid_folds": v["walk_forward"]["valid_folds"],
            "threshold": v["test_metrics"]["threshold"],
            "threshold_metric": v["threshold_metric"],
        }
        for i, v in enumerate(ranked)
    ]

    best = ranked[0]
    return {
        "best_variant_id": best["variant_id"],
        "best_decision": best["decision"],
        "ranking": ranking,
    }


# ---------------------------------------------------------------------------
# Symbol selection justification report
# ---------------------------------------------------------------------------

def _symbol_selection_justification() -> dict[str, Any]:
    return {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "excluded": ["USDJPY", "AUDUSD"],
        "justification": {
            "GBPUSD": {
                "selected_as": "primary_focus",
                "reasoning": [
                    "WF total PnL = -24.11 USD — closest to breakeven among non-blocked symbols.",
                    "4/14 WF folds are strongly positive (PF 4.1, 2.8, 1.9, 2.3) — conditional edge exists.",
                    "Clean labels (TP/SL hits) are balanced: 52.4% long / 47.6% short.",
                    "Timeout labels have 56.1% short bias (regime artifact from 3-month downtrend).",
                    "Dominant failure cause is diagnosable and structurally addressable.",
                ],
            },
            "EURUSD": {
                "selected_as": "secondary_monitor",
                "reasoning": [
                    "Test PnL = -3.05 USD (essentially breakeven in test window).",
                    "WF total = -96 USD — too variable for primary focus in this phase.",
                    "Will be evaluated only if GBPUSD rework succeeds.",
                ],
            },
            "USDJPY": {
                "selected_as": "excluded",
                "reasoning": [
                    "Excluded per project constraints (not quant focus in this phase).",
                    "Test PnL positive (+23.20) but WF negative (-31.90) and many zero-trade folds.",
                ],
            },
            "AUDUSD": {
                "selected_as": "excluded",
                "reasoning": [
                    "Worst performer: WF total = -234 USD. No conditional edge visible.",
                    "Model over-trades (no_trade_ratio ≈ 0 in most folds) with near-zero signal.",
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_audit_symbol_signal(settings: Settings) -> int:
    """Diagnostic only: label quality, feature redundancy, session breakdown.

    No training. No predictions. Fast.
    """
    run_dir = build_run_directory(settings.data.runs_dir, "audit_symbol_signal")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)

    bars = load_bars(settings.data.raw_dataset_path)
    if not bars:
        logger.error("No bars found at %s", settings.data.raw_dataset_path)
        return 1
    dataset = build_processed_dataset(bars, settings.labeling)
    focus_rows = [
        r for r in dataset.rows
        if r.symbol == FOCUS_SYMBOL and r.timeframe == settings.trading.primary_timeframe
    ]
    focus_rows.sort(key=lambda r: r.timestamp)

    label_diag = _label_diagnostic(focus_rows)
    feature_redundancy = _feature_redundancy_report(focus_rows)
    session_breakdown = _session_breakdown(focus_rows)
    selection_justification = _symbol_selection_justification()

    # WF fold date coverage (no training, just date ranges)
    windows = generate_walk_forward_windows(
        total_rows=len(focus_rows),
        train_window=settings.walk_forward.train_window,
        validation_window=settings.walk_forward.validation_window,
        test_window=settings.walk_forward.test_window,
        step=settings.walk_forward.step,
    )
    fold_coverage = []
    for w in windows:
        train_rows = focus_rows[w.train_start: w.train_end]
        test_rows = focus_rows[w.test_start: w.test_end]
        fold_coverage.append({
            "fold_index": w.fold_index,
            "train_start": train_rows[0].timestamp.isoformat() if train_rows else "",
            "train_end": train_rows[-1].timestamp.isoformat() if train_rows else "",
            "test_start": test_rows[0].timestamp.isoformat() if test_rows else "",
            "test_end": test_rows[-1].timestamp.isoformat() if test_rows else "",
        })

    payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "symbol_selection": selection_justification,
        "label_diagnostic": label_diag,
        "feature_redundancy": feature_redundancy,
        "session_breakdown": session_breakdown,
        "fold_coverage": fold_coverage,
        "row_count": len(focus_rows),
        "primary_timeframe": settings.trading.primary_timeframe,
    }
    write_json_report(run_dir, "symbol_focus_diagnostic_report.json", wrap_artifact("symbol_focus_diagnostic", payload))
    logger.info("audit_symbol_signal focus=%s rows=%d run_dir=%s", FOCUS_SYMBOL, len(focus_rows), run_dir)
    return 0


def run_compare_symbol_variants(settings: Settings) -> int:
    """Run the variant matrix for the focus symbol and compare results."""
    run_dir = build_run_directory(settings.data.runs_dir, "compare_symbol_variants")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)

    variant_results: list[dict[str, Any]] = []
    for spec in _VARIANTS:
        logger.info("Running variant %s ...", spec["variant_id"])
        result = _run_one_variant(settings, spec)
        variant_results.append(result)
        logger.info(
            "variant=%s decision=%s test_pnl=%.2f wf_total=%.2f",
            result["variant_id"],
            result.get("decision", "skipped"),
            result.get("test_metrics", {}).get("net_pnl_usd", 0.0),
            result.get("walk_forward", {}).get("aggregate", {}).get("total_net_pnl_usd", 0.0),
        )

    comparison = _compare_variants(variant_results)

    write_json_report(
        run_dir,
        "structural_rework_matrix_report.json",
        wrap_artifact("structural_variant_comparison", {
            "focus_symbol": FOCUS_SYMBOL,
            "variant_count": len(variant_results),
            "variants": variant_results,
            "comparison": comparison,
        }),
    )
    logger.info(
        "compare_symbol_variants best=%s decision=%s run_dir=%s",
        comparison["best_variant_id"],
        comparison["best_decision"],
        run_dir,
    )
    return 0


def run_evaluate_demo_execution_candidate(settings: Settings) -> int:
    """Load the most recent variant comparison and apply hard demo execution gates."""
    run_dir = build_run_directory(settings.data.runs_dir, "evaluate_demo_candidate")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)

    # Find most recent compare_symbol_variants run
    candidates = sorted(settings.data.runs_dir.glob("*_compare_symbol_variants"))
    if not candidates:
        logger.error("No compare_symbol_variants run found. Run compare-symbol-variants first.")
        return 1
    source_run = candidates[-1]
    matrix_report_path = source_run / "structural_rework_matrix_report.json"
    if not matrix_report_path.exists():
        logger.error("structural_rework_matrix_report.json not found in %s", source_run)
        return 1

    raw = json.loads(matrix_report_path.read_text(encoding="utf-8"))
    payload = raw.get("payload", raw)
    variants = payload.get("variants", [])
    comparison = payload.get("comparison", {})
    best_variant_id = comparison.get("best_variant_id")

    best_variant = next(
        (v for v in variants if v.get("variant_id") == best_variant_id and not v.get("skipped")),
        None,
    )

    if best_variant is None:
        decision = "REJECT_FOR_DEMO_EXECUTION"
        reasons = ["no_valid_variant_found"]
        candidate_report: dict[str, Any] = {
            "focus_symbol": FOCUS_SYMBOL,
            "best_variant_id": None,
            "decision": decision,
            "reasons": reasons,
            "approved_for_demo_execution": False,
        }
    else:
        decision = best_variant.get("decision", "REJECT_FOR_DEMO_EXECUTION")
        reasons = best_variant.get("reasons", [])
        candidate_report = {
            "focus_symbol": FOCUS_SYMBOL,
            "best_variant_id": best_variant_id,
            "decision": decision,
            "approved_for_demo_execution": decision == "APPROVED_FOR_DEMO_EXECUTION",
            "reasons": reasons,
            "test_metrics": best_variant.get("test_metrics", {}),
            "walk_forward_aggregate": best_variant.get("walk_forward", {}).get("aggregate", {}),
            "label_config": best_variant.get("label_config", {}),
            "feature_set": best_variant.get("feature_set"),
            "threshold_metric": best_variant.get("threshold_metric"),
            "threshold": best_variant.get("test_metrics", {}).get("threshold"),
            "ranking": comparison.get("ranking", []),
            "source_run_dir": str(source_run),
        }

    write_json_report(
        run_dir,
        "demo_execution_candidate_report.json",
        wrap_artifact("demo_execution_candidate", candidate_report),
    )
    logger.info(
        "evaluate_demo_candidate symbol=%s decision=%s approved=%s run_dir=%s",
        FOCUS_SYMBOL,
        decision,
        candidate_report.get("approved_for_demo_execution", False),
        run_dir,
    )
    return 0


def run_symbol_structural_rework(settings: Settings) -> int:
    """Full rework pipeline: audit → variants → gates → all reports."""
    run_dir = build_run_directory(settings.data.runs_dir, "symbol_structural_rework")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)

    # --- 1. Symbol selection justification ---
    selection = _symbol_selection_justification()
    logger.info("Focus symbol: %s", FOCUS_SYMBOL)

    # --- 2. Signal audit (no training) ---
    bars = load_bars(settings.data.raw_dataset_path)
    if not bars:
        logger.error("No bars found at %s", settings.data.raw_dataset_path)
        return 1
    dataset = build_processed_dataset(bars, settings.labeling)
    focus_rows = [
        r for r in dataset.rows
        if r.symbol == FOCUS_SYMBOL and r.timeframe == settings.trading.primary_timeframe
    ]
    focus_rows.sort(key=lambda r: r.timestamp)

    label_diag = _label_diagnostic(focus_rows)
    feature_redundancy = _feature_redundancy_report(focus_rows)
    session_breakdown = _session_breakdown(focus_rows)

    write_json_report(
        run_dir,
        "symbol_focus_diagnostic_report.json",
        wrap_artifact("symbol_focus_diagnostic", {
            "focus_symbol": FOCUS_SYMBOL,
            "symbol_selection": selection,
            "label_diagnostic": label_diag,
            "feature_redundancy": feature_redundancy,
            "session_breakdown": session_breakdown,
            "row_count": len(focus_rows),
        }),
    )

    # --- 3. Variant comparison matrix ---
    variant_results: list[dict[str, Any]] = []
    for spec in _VARIANTS:
        logger.info("Running variant %s ...", spec["variant_id"])
        result = _run_one_variant(settings, spec)
        variant_results.append(result)
        logger.info(
            "variant=%s decision=%s test_pnl=%.2f wf_total=%.2f",
            result["variant_id"],
            result.get("decision", "skipped"),
            result.get("test_metrics", {}).get("net_pnl_usd", 0.0),
            result.get("walk_forward", {}).get("aggregate", {}).get("total_net_pnl_usd", 0.0),
        )

    comparison = _compare_variants(variant_results)
    best_variant_id = comparison["best_variant_id"]
    best_decision = comparison["best_decision"]
    best_variant = next((v for v in variant_results if v.get("variant_id") == best_variant_id), None)

    write_json_report(
        run_dir,
        "structural_rework_matrix_report.json",
        wrap_artifact("structural_variant_comparison", {
            "focus_symbol": FOCUS_SYMBOL,
            "variant_count": len(variant_results),
            "variants": variant_results,
            "comparison": comparison,
        }),
    )

    # --- 4. Feature signal analysis ---
    write_json_report(
        run_dir,
        "feature_signal_analysis_report.json",
        wrap_artifact("feature_signal_analysis", {
            "focus_symbol": FOCUS_SYMBOL,
            "feature_redundancy": feature_redundancy,
            "pruned_features": list(_FEATURES_TO_PRUNE),
            "kept_features": _feature_names(pruned=True),
            "full_feature_count": len(FEATURE_NAMES_BASE),
            "pruned_feature_count": len(_feature_names(pruned=True)),
        }),
    )

    # --- 5. Label/exit interaction report ---
    write_json_report(
        run_dir,
        "label_exit_interaction_report.json",
        wrap_artifact("label_exit_interaction", {
            "focus_symbol": FOCUS_SYMBOL,
            "baseline_label_config": {
                "take_profit_pct": settings.labeling.take_profit_pct,
                "stop_loss_pct": settings.labeling.stop_loss_pct,
                "horizon_bars": settings.labeling.horizon_bars,
            },
            "label_diagnostic": label_diag,
            "dominant_cause_assessment": (
                "timeout_direction_labels_exceed_50pct — "
                "over half of training labels are regime-biased noise from horizon expiry"
                if label_diag["clean_signal_ratio"] < 0.40
                else "label_quality_acceptable"
            ),
            "variants_tested": [v["variant_id"] for v in variant_results],
        }),
    )

    # --- 6. Walk-forward stability report (from best variant) ---
    if best_variant and not best_variant.get("skipped"):
        wf_data = best_variant.get("walk_forward", {})
        write_json_report(
            run_dir,
            "walkforward_stability_report.json",
            wrap_artifact("walkforward_stability", {
                "focus_symbol": FOCUS_SYMBOL,
                "best_variant_id": best_variant_id,
                "walk_forward": wf_data,
            }),
        )

    # --- 7. Threshold/trade density report ---
    threshold_reports = [
        {
            "variant_id": v["variant_id"],
            "threshold": v.get("test_metrics", {}).get("threshold"),
            "threshold_metric": v.get("threshold_metric"),
            "test_no_trade_ratio": v.get("test_metrics", {}).get("no_trade_ratio"),
            "test_trade_count": v.get("test_metrics", {}).get("trade_count"),
            "wf_mean_no_trade_ratio": v.get("walk_forward", {}).get("aggregate", {}).get("mean_no_trade_ratio"),
        }
        for v in variant_results if not v.get("skipped")
    ]
    write_json_report(
        run_dir,
        "threshold_trade_density_report.json",
        wrap_artifact("threshold_trade_density", {
            "focus_symbol": FOCUS_SYMBOL,
            "threshold_reports": threshold_reports,
        }),
    )

    # --- 8. Candidate and recommendation reports ---
    approved_variants = [v["variant_id"] for v in variant_results if v.get("decision") == "APPROVED_FOR_DEMO_EXECUTION"]
    candidate_variants = [v["variant_id"] for v in variant_results if v.get("decision") == "CANDIDATE_FOR_DEMO_EXECUTION"]
    improved_variants = [v["variant_id"] for v in variant_results if v.get("decision") == "IMPROVED_BUT_NOT_ENOUGH"]

    candidate_payload: dict[str, Any] = {
        "focus_symbol": FOCUS_SYMBOL,
        "best_variant_id": best_variant_id,
        "best_decision": best_decision,
        "approved_for_demo_execution": best_decision == "APPROVED_FOR_DEMO_EXECUTION",
        "approved_variants": approved_variants,
        "candidate_variants": candidate_variants,
        "improved_variants": improved_variants,
        "reasons": best_variant.get("reasons", []) if best_variant else ["no_valid_variant"],
        "ranking": comparison["ranking"],
    }
    write_json_report(
        run_dir,
        "demo_execution_candidate_report.json",
        wrap_artifact("demo_execution_candidate", candidate_payload),
    )

    recommendation: dict[str, Any] = {
        "focus_symbol": FOCUS_SYMBOL,
        "best_variant": best_variant_id,
        "decision": best_decision,
        "approved_for_demo_execution": best_decision == "APPROVED_FOR_DEMO_EXECUTION",
        "what_was_tried": [v["variant_id"] for v in variant_results],
        "dominant_cause_pre_rework": (
            "noisy_timeout_labels_and_wrong_threshold_metric"
            if label_diag["clean_signal_ratio"] < 0.40
            else "insufficient_directional_edge"
        ),
        "what_should_come_next": _next_steps(best_decision, best_variant),
        "readme_modified": False,
        "live_real_touched": False,
        "operational_layer_modified": False,
    }
    write_json_report(
        run_dir,
        "structural_rework_recommendation_report.json",
        wrap_artifact("structural_rework_recommendation", recommendation),
    )

    logger.info(
        "symbol_structural_rework focus=%s best=%s decision=%s approved=%s run_dir=%s",
        FOCUS_SYMBOL,
        best_variant_id,
        best_decision,
        best_decision == "APPROVED_FOR_DEMO_EXECUTION",
        run_dir,
    )
    return 0


def _next_steps(decision: str, best_variant: dict[str, Any] | None) -> list[str]:
    if decision == "APPROVED_FOR_DEMO_EXECUTION":
        return [
            "Verify approved model artifact is sound.",
            "Promote model via governance pipeline.",
            "Run endurance/lifecycle checks before demo execution.",
        ]
    if decision == "CANDIDATE_FOR_DEMO_EXECUTION":
        return [
            "Collect more live M15 data to extend sample size.",
            "Re-run variant analysis with extended dataset.",
            "Do not promote without walk-forward turning positive.",
        ]
    if decision == "IMPROVED_BUT_NOT_ENOUGH":
        return [
            "Some improvement found — identify which variant drove it.",
            "Consider extending data collection before further rework.",
            "Do not lower gates to accommodate weak signal.",
        ]
    return [
        "No edge found for this symbol in this rework phase.",
        "Options: wait for more data, revisit feature engineering, or park symbol.",
        "Do not lower gates. Do not force approval.",
    ]
