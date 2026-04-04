"""
Walk-forward economic backtest for IRIS-Bot.

Each fold is fully independent:
  1. A fresh XGBoost model is trained on fold-train rows.
  2. The decision threshold is selected on fold-validation rows.
  3. An economic backtest is run on fold-test rows with a fresh starting balance.
  4. Consistency is validated automatically.
  5. Artifacts are saved under run_dir/fold_NN/.

This gives a realistic fold-by-fold picture of strategy performance without
look-ahead contamination between folds.

Note on balance independence
-----------------------------
Each fold starts with settings.backtest.starting_balance_usd. PnL is NOT
compounded across folds. This makes fold metrics directly comparable to each
other without position-size drift. The aggregate section reports total and mean
PnL across folds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from iris_bot.backtest import run_backtest_engine, write_equity_curve, write_trade_log
from iris_bot.config import Settings
from iris_bot.consistency import verify_engine_consistency
from iris_bot.logging_utils import write_json_report
from iris_bot.preprocessing import validate_feature_rows
from iris_bot.processed_dataset import ProcessedRow
from iris_bot.symbols import load_symbol_strategy_profiles
from iris_bot.thresholds import select_threshold_from_probabilities
from iris_bot.walk_forward import generate_walk_forward_windows
from iris_bot.xgb_model import XGBoostMultiClassModel


@dataclass
class WalkForwardFoldSummary:
    fold_index: int
    skipped: bool
    skip_reason: str | None
    train_rows: int
    val_rows: int
    test_rows: int
    threshold: float
    total_trades: int
    net_pnl_usd: float
    win_rate: float
    profit_factor: float
    max_drawdown_usd: float
    return_pct: float
    intrabar_ambiguous_count: int
    blocked_entry_count: int
    consistency_is_clean: bool
    consistency_errors: int
    consistency_warnings: int


def _extract_matrix(
    rows: list[ProcessedRow],
    feature_names: list[str],
) -> tuple[list[list[float]], list[int]]:
    matrix = [[row.features[name] for name in feature_names] for row in rows]
    labels = [row.label for row in rows]
    validate_feature_rows(matrix)
    return matrix, labels


def _skipped_summary(fold_index: int, reason: str) -> WalkForwardFoldSummary:
    return WalkForwardFoldSummary(
        fold_index=fold_index,
        skipped=True,
        skip_reason=reason,
        train_rows=0,
        val_rows=0,
        test_rows=0,
        threshold=0.0,
        total_trades=0,
        net_pnl_usd=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        max_drawdown_usd=0.0,
        return_pct=0.0,
        intrabar_ambiguous_count=0,
        blocked_entry_count=0,
        consistency_is_clean=True,
        consistency_errors=0,
        consistency_warnings=0,
    )


def _summary_to_dict(s: WalkForwardFoldSummary) -> dict[str, object]:
    return {
        "fold_index": s.fold_index,
        "skipped": s.skipped,
        "skip_reason": s.skip_reason,
        "train_rows": s.train_rows,
        "val_rows": s.val_rows,
        "test_rows": s.test_rows,
        "threshold": s.threshold,
        "total_trades": s.total_trades,
        "net_pnl_usd": s.net_pnl_usd,
        "win_rate": s.win_rate,
        "profit_factor": s.profit_factor,
        "max_drawdown_usd": s.max_drawdown_usd,
        "return_pct": s.return_pct,
        "intrabar_ambiguous_count": s.intrabar_ambiguous_count,
        "blocked_entry_count": s.blocked_entry_count,
        "consistency_is_clean": s.consistency_is_clean,
        "consistency_errors": s.consistency_errors,
        "consistency_warnings": s.consistency_warnings,
    }


def _compute_aggregate(
    summaries: list[WalkForwardFoldSummary],
    starting_balance: float,
) -> dict[str, object]:
    if not summaries:
        return {
            "valid_folds": 0,
            "message": "No valid folds to aggregate.",
        }

    total_trades = sum(s.total_trades for s in summaries)
    total_net_pnl = sum(s.net_pnl_usd for s in summaries)
    n = len(summaries)

    win_rates = [s.win_rate for s in summaries if s.total_trades > 0]
    profit_factors = [s.profit_factor for s in summaries if s.total_trades > 0 and s.profit_factor > 0.0]
    drawdowns = [s.max_drawdown_usd for s in summaries]

    return {
        "valid_folds": n,
        "starting_balance_per_fold_usd": starting_balance,
        "total_trades_across_folds": total_trades,
        "total_net_pnl_usd": total_net_pnl,
        "mean_net_pnl_usd_per_fold": total_net_pnl / n,
        "mean_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0.0,
        "mean_profit_factor": sum(profit_factors) / len(profit_factors) if profit_factors else 0.0,
        "worst_fold_drawdown_usd": max(drawdowns) if drawdowns else 0.0,
        "folds_with_positive_pnl": sum(1 for s in summaries if s.net_pnl_usd > 0.0),
        "folds_with_negative_pnl": sum(1 for s in summaries if s.net_pnl_usd < 0.0),
        "total_intrabar_ambiguous": sum(s.intrabar_ambiguous_count for s in summaries),
        "total_blocked_entries": sum(s.blocked_entry_count for s in summaries),
        "total_consistency_errors": sum(s.consistency_errors for s in summaries),
        "note": (
            "Balance is NOT compounded across folds. "
            "Each fold starts fresh at starting_balance_per_fold_usd."
        ),
    }


def run_walkforward_economic_backtest(
    rows: list[ProcessedRow],
    feature_names: list[str],
    settings: Settings,
    run_dir: Path,
    logger: logging.Logger,
    aux_rates: dict[str, float] | None = None,
) -> dict[str, object]:
    """
    Execute a fold-by-fold economic backtest using walk-forward windows.

    Parameters
    ----------
    rows:          Time-sorted ProcessedRow objects (typically primary timeframe only).
    feature_names: Feature columns in the model's expected order.
    settings:      Full settings (walk_forward, backtest, risk, xgboost, threshold).
    run_dir:       Root directory for fold artifacts.
    logger:        Logger instance.
    aux_rates:     Optional currency conversion rates for cross pairs.

    Returns
    -------
    dict with keys: folds, fold_summaries, aggregate, total_folds, valid_folds, skipped_folds.
    """
    policy = settings.backtest.intrabar_policy
    symbol_profiles = load_symbol_strategy_profiles(settings)

    windows = generate_walk_forward_windows(
        total_rows=len(rows),
        train_window=settings.walk_forward.train_window,
        validation_window=settings.walk_forward.validation_window,
        test_window=settings.walk_forward.test_window,
        step=settings.walk_forward.step,
    )

    if not windows:
        logger.warning(
            "No se generaron ventanas walk-forward con %s filas "
            "(train=%s val=%s test=%s step=%s). Se necesitan al menos %s filas.",
            len(rows),
            settings.walk_forward.train_window,
            settings.walk_forward.validation_window,
            settings.walk_forward.test_window,
            settings.walk_forward.step,
            settings.walk_forward.train_window
            + settings.walk_forward.validation_window
            + settings.walk_forward.test_window,
        )
        return {
            "folds": [],
            "fold_summaries": [],
            "aggregate": {"valid_folds": 0, "message": "No walk-forward windows generated."},
            "total_folds": 0,
            "valid_folds": 0,
            "skipped_folds": 0,
        }

    MIN_TRAIN = 30
    MIN_VAL = 10
    MIN_TEST = 5

    fold_summaries: list[WalkForwardFoldSummary] = []
    fold_reports: list[dict[str, object]] = []

    for window in windows:
        fold_index = window.fold_index
        logger.info(
            "wf_economic fold=%02d  train=[%d,%d)  val=[%d,%d)  test=[%d,%d)",
            fold_index,
            window.train_start, window.train_end,
            window.validation_start, window.validation_end,
            window.test_start, window.test_end,
        )

        train_rows = rows[window.train_start : window.train_end]
        val_rows = rows[window.validation_start : window.validation_end]
        test_rows = rows[window.test_start : window.test_end]

        # Minimum size checks
        if len(train_rows) < MIN_TRAIN:
            reason = f"train_rows={len(train_rows)} < {MIN_TRAIN}"
            fold_summaries.append(_skipped_summary(fold_index, reason))
            fold_reports.append({"fold": fold_index, "skipped": True, "reason": reason, "window": window.to_dict()})
            logger.warning("fold=%02d saltado: %s", fold_index, reason)
            continue

        if len(val_rows) < MIN_VAL:
            reason = f"val_rows={len(val_rows)} < {MIN_VAL}"
            fold_summaries.append(_skipped_summary(fold_index, reason))
            fold_reports.append({"fold": fold_index, "skipped": True, "reason": reason, "window": window.to_dict()})
            logger.warning("fold=%02d saltado: %s", fold_index, reason)
            continue

        if len(test_rows) < MIN_TEST:
            reason = f"test_rows={len(test_rows)} < {MIN_TEST}"
            fold_summaries.append(_skipped_summary(fold_index, reason))
            fold_reports.append({"fold": fold_index, "skipped": True, "reason": reason, "window": window.to_dict()})
            logger.warning("fold=%02d saltado: %s", fold_index, reason)
            continue

        # Feature extraction
        try:
            train_matrix, train_labels = _extract_matrix(train_rows, feature_names)
            val_matrix, val_labels = _extract_matrix(val_rows, feature_names)
            test_matrix, _ = _extract_matrix(test_rows, feature_names)
        except (ValueError, KeyError) as exc:
            reason = f"feature extraction failed: {exc}"
            fold_summaries.append(_skipped_summary(fold_index, reason))
            fold_reports.append({"fold": fold_index, "skipped": True, "reason": reason, "window": window.to_dict()})
            logger.warning("fold=%02d saltado: %s", fold_index, reason)
            continue

        # Train fold-specific model
        fold_model = XGBoostMultiClassModel(settings.xgboost)
        try:
            fold_model.fit(train_matrix, train_labels, val_matrix, val_labels)
        except RuntimeError as exc:
            reason = f"model training failed: {exc}"
            fold_summaries.append(_skipped_summary(fold_index, reason))
            fold_reports.append({"fold": fold_index, "skipped": True, "reason": reason, "window": window.to_dict()})
            logger.warning("fold=%02d saltado: %s", fold_index, reason)
            continue

        # Select threshold on fold validation set
        val_probs = fold_model.predict_probabilities(val_matrix)
        threshold_result = select_threshold_from_probabilities(
            probabilities=val_probs,
            labels=val_labels,
            grid=settings.threshold.grid,
            metric_name=settings.threshold.objective_metric,
        )

        # Economic replay on fold test set
        test_probs = fold_model.predict_probabilities(test_matrix)
        metrics, trades, equity_curve = run_backtest_engine(
            rows=test_rows,
            probabilities=test_probs,
            threshold=threshold_result.threshold,
            backtest=settings.backtest,
            risk=settings.risk,
            intrabar_policy=policy,
            aux_rates=aux_rates,
            exit_policy_config=settings.exit_policy,
            dynamic_exit_config=settings.dynamic_exits,
            symbol_exit_profiles={symbol: profile.exit_profile for symbol, profile in symbol_profiles.items()},
        )

        # Consistency validation
        consistency = verify_engine_consistency(
            trades=trades,
            equity_curve=equity_curve,
            starting_balance=settings.backtest.starting_balance_usd,
        )
        if not consistency.is_clean:
            logger.warning(
                "fold=%02d consistency FAILED errors=%s",
                fold_index, consistency.error_count
            )

        # Save fold artifacts
        fold_dir = run_dir / f"fold_{fold_index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        if trades:
            write_trade_log(fold_dir / "trade_log.csv", trades)
        write_equity_curve(fold_dir / "equity_curve.csv", equity_curve)

        fold_report: dict[str, object] = {
            "fold": fold_index,
            "skipped": False,
            "window": window.to_dict(),
            "intrabar_policy": policy,
            "threshold": threshold_result.threshold,
            "threshold_metric": threshold_result.metric_name,
            "threshold_value": threshold_result.metric_value,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "metrics": metrics,
            "trade_count": len(trades),
            "consistency": consistency.to_dict(),
        }
        write_json_report(fold_dir, "fold_report.json", fold_report)
        fold_reports.append(fold_report)

        summary = WalkForwardFoldSummary(
            fold_index=fold_index,
            skipped=False,
            skip_reason=None,
            train_rows=len(train_rows),
            val_rows=len(val_rows),
            test_rows=len(test_rows),
            threshold=threshold_result.threshold,
            total_trades=int(metrics["total_trades"]),
            net_pnl_usd=float(metrics["net_pnl_usd"]),
            win_rate=float(metrics["win_rate"]),
            profit_factor=float(metrics["profit_factor"]),
            max_drawdown_usd=float(metrics["max_drawdown_usd"]),
            return_pct=float(metrics["return_pct"]),
            intrabar_ambiguous_count=int(metrics.get("intrabar_ambiguous_count", 0)),
            blocked_entry_count=int(metrics.get("blocked_entry_count", 0)),
            consistency_is_clean=consistency.is_clean,
            consistency_errors=consistency.error_count,
            consistency_warnings=consistency.warning_count,
        )
        fold_summaries.append(summary)

        logger.info(
            "fold=%02d done  trades=%d  net_pnl=%.2f  win_rate=%.2f  "
            "pf=%.2f  ambiguous=%d  consistency=%s",
            fold_index,
            summary.total_trades,
            summary.net_pnl_usd,
            summary.win_rate,
            summary.profit_factor,
            summary.intrabar_ambiguous_count,
            "ok" if consistency.is_clean else f"ERRORS={consistency.error_count}",
        )

    valid_summaries = [s for s in fold_summaries if not s.skipped]
    aggregate = _compute_aggregate(valid_summaries, settings.backtest.starting_balance_usd)

    return {
        "folds": fold_reports,
        "fold_summaries": [_summary_to_dict(s) for s in fold_summaries],
        "aggregate": aggregate,
        "total_folds": len(windows),
        "valid_folds": len(valid_summaries),
        "skipped_folds": len(fold_summaries) - len(valid_summaries),
    }
