from iris_bot.backtest import EquityPoint, TradeRecord, _trade_metrics


def test_trade_metrics_compute_drawdown() -> None:
    trades = [
        TradeRecord("EURUSD", "M15", 1, "2026-01-01T00:00:00", "2026-01-01T00:15:00", "2026-01-01T00:00:00", 1.1, 1.101, 1.099, 1.102, 0.1, 10.0, 10.0, 0.0, 0.0, 0.0, "time_exit", 1, 0.8, 0.1),
        TradeRecord("EURUSD", "M15", 1, "2026-01-01T00:30:00", "2026-01-01T00:45:00", "2026-01-01T00:30:00", 1.1, 1.098, 1.099, 1.102, 0.1, -20.0, -20.0, 0.0, 0.0, 0.0, "stop_loss", 1, 0.8, 0.1),
    ]
    equity_curve = [
        EquityPoint("2026-01-01T00:15:00", 1010.0, 1010.0, 0),
        EquityPoint("2026-01-01T00:45:00", 990.0, 990.0, 0),
    ]

    metrics = _trade_metrics(trades, 1000.0, 990.0, equity_curve, total_steps=2)

    assert metrics["max_drawdown_usd"] == 20.0
    assert metrics["profit_factor"] == 0.5


def test_trade_metrics_profit_factor_is_high_when_no_losses() -> None:
    trades = [
        TradeRecord("EURUSD", "M15", 1, "2026-01-01T00:00:00", "2026-01-01T00:15:00", "2026-01-01T00:00:00", 1.1, 1.101, 1.099, 1.102, 0.1, 10.0, 10.0, 0.0, 0.0, 0.0, "time_exit", 1, 0.8, 0.1),
        TradeRecord("EURUSD", "M15", 1, "2026-01-01T00:30:00", "2026-01-01T00:45:00", "2026-01-01T00:30:00", 1.1, 1.102, 1.099, 1.103, 0.1, 20.0, 20.0, 0.0, 0.0, 0.0, "take_profit", 1, 0.8, 0.1),
    ]
    equity_curve = [
        EquityPoint("2026-01-01T00:15:00", 1010.0, 1010.0, 0),
        EquityPoint("2026-01-01T00:45:00", 1030.0, 1030.0, 0),
    ]

    metrics = _trade_metrics(trades, 1000.0, 1030.0, equity_curve, total_steps=2)

    assert metrics["gross_loss_usd"] == 0.0
    assert metrics["profit_factor"] == 999.0
