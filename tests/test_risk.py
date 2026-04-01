from iris_bot.risk import calculate_position_size


def test_position_size_uses_risk_budget() -> None:
    size = calculate_position_size(
        balance=1000.0,
        risk_per_trade=0.01,
        entry_price=1.1000,
        stop_loss_price=1.0950,
        pip_value=10000.0,
        min_lot=0.01,
        lot_step=0.01,
    )

    assert size > 0.0
    assert size <= 0.2
