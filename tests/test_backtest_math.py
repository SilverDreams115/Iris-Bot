from iris_bot.risk import ForexInstrument, calculate_position_size, realized_pnl_usd


def test_position_size_is_mt5_consistent_for_eurusd() -> None:
    instrument = ForexInstrument("EURUSD", contract_size=100000.0, min_lot=0.01, lot_step=0.01, max_lot=100.0)
    lots = calculate_position_size(
        balance=1000.0,
        risk_per_trade=0.01,
        entry_price=1.1000,
        stop_loss_price=1.0980,
        instrument=instrument,
    )

    assert lots > 0.0
    assert lots <= 0.05


def test_realized_pnl_usd_for_long_trade() -> None:
    instrument = ForexInstrument("EURUSD", contract_size=100000.0, min_lot=0.01, lot_step=0.01, max_lot=100.0)
    pnl = realized_pnl_usd(
        instrument=instrument,
        entry_price=1.1000,
        exit_price=1.1010,
        direction=1,
        volume_lots=0.10,
    )

    assert round(pnl, 2) == 10.0
