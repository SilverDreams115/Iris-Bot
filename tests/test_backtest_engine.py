from datetime import datetime, timedelta

from iris_bot.backtest import run_backtest_engine
from iris_bot.config import BacktestConfig, RiskConfig
from iris_bot.processed_dataset import ProcessedRow


def _row(ts: datetime, close: float, symbol: str = "EURUSD") -> ProcessedRow:
    return ProcessedRow(
        timestamp=ts,
        symbol=symbol,
        timeframe="M15",
        open=close - 0.0002,
        high=close + 0.0010,
        low=close - 0.0010,
        close=close,
        volume=100.0,
        label=1,
        label_reason="test",
        horizon_end_timestamp=ts.isoformat(),
        features={"atr_10": 0.001, "atr_5": 0.001},
    )


def test_backtest_engine_opens_and_closes_trade_with_costs() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        _row(start, 1.1000),
        _row(start + timedelta(minutes=15), 1.1010),
        _row(start + timedelta(minutes=30), 1.1030),
        _row(start + timedelta(minutes=45), 1.1040),
    ]
    probabilities = [
        {1: 0.8, 0: 0.1, -1: 0.1},
        {1: 0.3, 0: 0.4, -1: 0.3},
        {1: 0.3, 0: 0.4, -1: 0.3},
        {1: 0.3, 0: 0.4, -1: 0.3},
    ]
    metrics, trades, equity_curve = run_backtest_engine(
        rows=rows,
        probabilities=probabilities,
        threshold=0.6,
        backtest=BacktestConfig(
            starting_balance_usd=1000.0,
            spread_pips=1.0,
            slippage_pips=0.0,
            commission_per_lot_per_side_usd=1.0,
            use_atr_stops=False,
            fixed_stop_loss_pct=0.001,
            fixed_take_profit_pct=0.001,
            max_holding_bars=3,
        ),
        risk=RiskConfig(risk_per_trade=0.01, max_daily_loss_usd=100.0),
    )

    assert len(trades) == 1
    assert trades[0].exit_reason in {"take_profit", "time_exit", "stop_loss", "stop_loss_same_bar"}
    assert len(equity_curve) == len(rows)
    assert metrics["total_trades"] == 1


def test_backtest_engine_respects_no_trade_threshold() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        _row(start, 1.1000),
        _row(start + timedelta(minutes=15), 1.1005),
        _row(start + timedelta(minutes=30), 1.1010),
    ]
    probabilities = [
        {1: 0.40, 0: 0.35, -1: 0.25},
        {1: 0.41, 0: 0.34, -1: 0.25},
        {1: 0.42, 0: 0.33, -1: 0.25},
    ]
    metrics, trades, _ = run_backtest_engine(
        rows=rows,
        probabilities=probabilities,
        threshold=0.60,
        backtest=BacktestConfig(starting_balance_usd=1000.0),
        risk=RiskConfig(),
    )

    assert trades == []
    assert metrics["total_trades"] == 0
