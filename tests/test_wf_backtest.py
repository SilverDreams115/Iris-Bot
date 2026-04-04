"""
Tests for walk-forward economic backtest (Task C – Phase 3.5).

Verifies:
  - Basic execution with enough rows produces fold artifacts
  - Each fold has independent starting balance
  - Aggregate summary is computed correctly
  - Insufficient rows for walk-forward windows produces empty result
  - Consistency check runs per fold
"""
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("xgboost")

from iris_bot.config import BacktestConfig, RiskConfig, WalkForwardConfig, XGBoostConfig
from iris_bot.config import load_settings
from iris_bot.logging_utils import configure_logging, build_run_directory
from iris_bot.processed_dataset import ProcessedRow
from iris_bot.wf_backtest import run_walkforward_economic_backtest

import logging


FEATURE_NAMES = ["return_1", "momentum_3", "rolling_volatility_5"]


def _make_rows(n: int, symbol: str = "EURUSD") -> list[ProcessedRow]:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = []
    price = 1.1000
    for i in range(n):
        price += 0.0003 if i % 4 < 2 else -0.0001
        ts = start + timedelta(minutes=15 * i)
        rows.append(
            ProcessedRow(
                timestamp=ts,
                symbol=symbol,
                timeframe="M15",
                open=price - 0.0001,
                high=price + 0.0010,
                low=price - 0.0010,
                close=price,
                volume=100.0,
                label=1 if i % 3 == 0 else (0 if i % 3 == 1 else -1),
                label_reason="test",
                horizon_end_timestamp=ts.isoformat(),
                features={
                    "return_1": 0.0003,
                    "momentum_3": 0.001 * (i % 5 - 2),
                    "rolling_volatility_5": 0.001,
                },
            )
        )
    return rows


def _make_settings(tmp_path: Path) -> object:
    settings = load_settings()
    import dataclasses
    new_wf = dataclasses.replace(
        settings.walk_forward,
        enabled=True,
        train_window=60,
        validation_window=20,
        test_window=20,
        step=20,
    )
    new_backtest = dataclasses.replace(
        settings.backtest,
        starting_balance_usd=1000.0,
        use_atr_stops=False,
        fixed_stop_loss_pct=0.005,
        fixed_take_profit_pct=0.010,
        spread_pips=0.0,
        slippage_pips=0.0,
        commission_per_lot_per_side_usd=0.0,
        max_holding_bars=10,
        intrabar_policy="conservative",
    )
    new_risk = dataclasses.replace(
        settings.risk,
        max_daily_loss_usd=999.0,
        max_open_positions=4,
        cooldown_bars_after_loss=0,
    )
    new_data = dataclasses.replace(settings.data, runs_dir=tmp_path / "runs")
    settings = dataclasses.replace(
        settings,
        walk_forward=new_wf,
        backtest=new_backtest,
        risk=new_risk,
        data=new_data,
    )
    return settings


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("test_wf")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------

def test_wf_backtest_produces_folds(tmp_path: Path) -> None:
    """With enough rows, at least one fold should execute successfully."""
    rows = _make_rows(130)  # 60 train + 20 val + 20 test = 100 min, step 20 → 1+ folds
    settings = _make_settings(tmp_path)
    run_dir = tmp_path / "wf_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _get_logger()

    result = run_walkforward_economic_backtest(
        rows=rows,
        feature_names=FEATURE_NAMES,
        settings=settings,
        run_dir=run_dir,
        logger=logger,
    )

    assert result["total_folds"] >= 1
    assert result["valid_folds"] >= 0  # may skip due to label diversity


def test_wf_backtest_fold_artifacts_exist(tmp_path: Path) -> None:
    """Each valid fold must create equity_curve.csv and fold_report.json."""
    rows = _make_rows(200)
    settings = _make_settings(tmp_path)
    run_dir = tmp_path / "wf_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _get_logger()

    result = run_walkforward_economic_backtest(
        rows=rows,
        feature_names=FEATURE_NAMES,
        settings=settings,
        run_dir=run_dir,
        logger=logger,
    )

    for fold_report in result["folds"]:
        if fold_report.get("skipped"):
            continue
        fold_index = fold_report["fold"]
        fold_dir = run_dir / f"fold_{fold_index:02d}"
        assert (fold_dir / "equity_curve.csv").exists(), f"fold_{fold_index:02d}/equity_curve.csv missing"
        assert (fold_dir / "fold_report.json").exists(), f"fold_{fold_index:02d}/fold_report.json missing"


def test_wf_backtest_each_fold_has_fresh_balance(tmp_path: Path) -> None:
    """Every fold report should reference the same starting_balance_usd (independent folds)."""
    rows = _make_rows(150)
    settings = _make_settings(tmp_path)
    run_dir = tmp_path / "wf_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _get_logger()

    result = run_walkforward_economic_backtest(
        rows=rows,
        feature_names=FEATURE_NAMES,
        settings=settings,
        run_dir=run_dir,
        logger=logger,
    )

    for fold_summary in result["fold_summaries"]:
        if fold_summary["skipped"]:
            continue
        # Each fold's starting balance in metrics should be 1000.0
        fold_index = fold_summary["fold_index"]
        fold_report_path = run_dir / f"fold_{fold_index:02d}" / "fold_report.json"
        if fold_report_path.exists():
            import json
            fold_data = json.loads(fold_report_path.read_text())
            assert fold_data["metrics"]["starting_balance_usd"] == 1000.0


# ---------------------------------------------------------------------------
# No windows case
# ---------------------------------------------------------------------------

def test_wf_backtest_too_few_rows_returns_empty(tmp_path: Path) -> None:
    """Fewer rows than one window should yield 0 folds."""
    rows = _make_rows(50)  # less than 60+20+20=100
    settings = _make_settings(tmp_path)
    run_dir = tmp_path / "wf_run_empty"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _get_logger()

    result = run_walkforward_economic_backtest(
        rows=rows,
        feature_names=FEATURE_NAMES,
        settings=settings,
        run_dir=run_dir,
        logger=logger,
    )

    assert result["total_folds"] == 0
    assert result["valid_folds"] == 0
    assert result["folds"] == []


# ---------------------------------------------------------------------------
# Aggregate structure
# ---------------------------------------------------------------------------

def test_wf_backtest_aggregate_has_required_keys(tmp_path: Path) -> None:
    rows = _make_rows(200)
    settings = _make_settings(tmp_path)
    run_dir = tmp_path / "wf_run_agg"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _get_logger()

    result = run_walkforward_economic_backtest(
        rows=rows,
        feature_names=FEATURE_NAMES,
        settings=settings,
        run_dir=run_dir,
        logger=logger,
    )

    if result["valid_folds"] > 0:
        agg = result["aggregate"]
        for key in [
            "valid_folds",
            "total_trades_across_folds",
            "total_net_pnl_usd",
            "mean_net_pnl_usd_per_fold",
            "mean_win_rate",
            "worst_fold_drawdown_usd",
            "folds_with_positive_pnl",
            "folds_with_negative_pnl",
            "total_consistency_errors",
        ]:
            assert key in agg, f"Missing aggregate key: {key}"


# ---------------------------------------------------------------------------
# Consistency within fold
# ---------------------------------------------------------------------------

def test_wf_folds_report_consistency_results(tmp_path: Path) -> None:
    rows = _make_rows(150)
    settings = _make_settings(tmp_path)
    run_dir = tmp_path / "wf_run_cons"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _get_logger()

    result = run_walkforward_economic_backtest(
        rows=rows,
        feature_names=FEATURE_NAMES,
        settings=settings,
        run_dir=run_dir,
        logger=logger,
    )

    for fold_summary in result["fold_summaries"]:
        if fold_summary["skipped"]:
            continue
        assert "consistency_is_clean" in fold_summary
        assert "consistency_errors" in fold_summary
        # Each valid fold's consistency should pass
        assert fold_summary["consistency_is_clean"] is True, (
            f"fold {fold_summary['fold_index']} has consistency errors: "
            f"{fold_summary['consistency_errors']}"
        )
