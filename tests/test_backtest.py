from datetime import datetime, timedelta

from iris_bot.backtest import run_backtest
from iris_bot.config import RiskConfig, TradingConfig
from iris_bot.data import Bar
from iris_bot.features import build_feature_vectors


def test_backtest_runs_and_returns_metrics() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    bars = []
    price = 1.1000
    for index in range(220):
        drift = 0.0004 if index % 2 == 0 else -0.0001
        price += drift
        bars.append(
            Bar(
                timestamp=start + timedelta(minutes=index * 5),
                symbol="EURUSD",
                timeframe="M5",
                open=price - 0.0002,
                high=price + 0.0005,
                low=price - 0.0005,
                close=price,
                volume=100 + index,
            )
        )

    rows = build_feature_vectors(bars)
    result = run_backtest(
        rows=rows,
        starting_balance=1000.0,
        risk=RiskConfig(min_confidence=0.5),
        trading=TradingConfig(training_window=50),
    )

    assert result.ending_balance > 0.0
    assert result.total_return > -1.0
