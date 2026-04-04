"""
Tests for engine constraint enforcement (Task E – Phase 3.5).

Constraints verified:
  - max_daily_loss_usd blocks new entries after limit is hit
  - cooldown_bars_after_loss blocks entry for N bars after a loss
  - one_position_per_symbol: second signal same symbol rejected
  - max_open_positions: new entries blocked once limit reached
  - end_of_data closes open positions
  - commission is applied exactly once per side (entry + exit = 2 × per_side)
"""
from datetime import datetime, timedelta

import pytest

from iris_bot.backtest import run_backtest_engine
from iris_bot.config import BacktestConfig, RiskConfig
from iris_bot.processed_dataset import ProcessedRow


def _row(
    ts: datetime,
    price: float = 1.1000,
    symbol: str = "EURUSD",
    bar_low: float | None = None,
    bar_high: float | None = None,
) -> ProcessedRow:
    low = bar_low if bar_low is not None else price - 0.0005
    high = bar_high if bar_high is not None else price + 0.0005
    return ProcessedRow(
        timestamp=ts,
        symbol=symbol,
        timeframe="M15",
        open=price,
        high=high,
        low=low,
        close=price,
        volume=100.0,
        label=1,
        label_reason="test",
        horizon_end_timestamp=ts.isoformat(),
        features={"atr_10": 0.0005, "atr_5": 0.0005},
    )


def _prob_signal(direction: int = 1) -> dict[int, float]:
    if direction == 1:
        return {1: 0.9, 0: 0.05, -1: 0.05}
    return {-1: 0.9, 0: 0.05, 1: 0.05}


def _prob_neutral() -> dict[int, float]:
    return {1: 0.3, 0: 0.4, -1: 0.3}


SMALL_BACKTEST = BacktestConfig(
    starting_balance_usd=1000.0,
    use_atr_stops=False,
    fixed_stop_loss_pct=0.002,
    fixed_take_profit_pct=0.004,
    spread_pips=0.0,
    slippage_pips=0.0,
    commission_per_lot_per_side_usd=1.0,
    max_holding_bars=20,
)


# ---------------------------------------------------------------------------
# max_daily_loss_usd
# ---------------------------------------------------------------------------

def test_max_daily_loss_blocks_entry_after_limit_hit() -> None:
    """
    After a large loss on bar 1, a new signal on bar 2 should be blocked
    because daily_realized < -max_daily_loss_usd.
    """
    start = datetime(2026, 1, 1, 0, 0, 0)
    # Build rows: signal bar 0, entry bar 1 (big loss via SL), signal bar 2, entry bar 3
    rows = [
        _row(start, 1.1000),                                           # signal bar 0
        _row(start + timedelta(minutes=15), 1.1000,
             bar_low=1.0900, bar_high=1.1005),                        # entry + big SL hit
        _row(start + timedelta(minutes=30), 1.1000),                  # signal bar 2
        _row(start + timedelta(minutes=45), 1.1000),                  # would be entry bar 3
    ]
    probs = [_prob_signal(), _prob_neutral(), _prob_signal(), _prob_neutral()]

    # max_daily_loss very small so that even 1 losing trade blocks next entry
    _, trades, _ = run_backtest_engine(
        rows=rows,
        probabilities=probs,
        threshold=0.5,
        backtest=SMALL_BACKTEST,
        risk=RiskConfig(
            risk_per_trade=0.01,
            max_daily_loss_usd=0.001,  # any loss triggers block
            max_open_positions=4,
        ),
    )
    # The first trade should be opened and closed (SL hit)
    # The second trade should NOT be opened (daily loss exceeded)
    assert len(trades) <= 1
    if trades:
        assert trades[0].exit_reason in {"stop_loss", "stop_loss_same_bar", "end_of_data", "time_exit"}


# ---------------------------------------------------------------------------
# cooldown_bars_after_loss
# ---------------------------------------------------------------------------

def test_cooldown_blocks_entry_immediately_after_loss() -> None:
    """
    With cooldown=2, after a losing trade at bar 1, signals at bars 2 and 3
    should be blocked (series index < entry_index + cooldown).
    """
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        _row(start, 1.1000),                                          # signal
        _row(start + timedelta(minutes=15), 1.1000,
             bar_low=1.0900, bar_high=1.1005),                        # entry → SL hit
        _row(start + timedelta(minutes=30), 1.1000),                  # signal during cooldown
        _row(start + timedelta(minutes=45), 1.1000),                  # entry would be here (blocked)
        _row(start + timedelta(minutes=60), 1.1000),                  # signal after cooldown
        _row(start + timedelta(minutes=75), 1.1000),                  # entry allowed
        _row(start + timedelta(minutes=90), 1.1000),                  # hold
    ]
    probs = [
        _prob_signal(),
        _prob_neutral(),
        _prob_signal(),
        _prob_neutral(),
        _prob_signal(),
        _prob_neutral(),
        _prob_neutral(),
    ]
    _, trades, _ = run_backtest_engine(
        rows=rows,
        probabilities=probs,
        threshold=0.5,
        backtest=SMALL_BACKTEST,
        risk=RiskConfig(
            risk_per_trade=0.01,
            max_daily_loss_usd=999.0,
            max_open_positions=4,
            cooldown_bars_after_loss=2,
        ),
    )
    # First trade (the loss) + possibly second trade after cooldown
    # Should not have a trade at bar 3 (during cooldown)
    entry_timestamps = [t.entry_timestamp for t in trades]
    # Bar 3 entry timestamp is start + 45 min
    blocked_ts = (start + timedelta(minutes=45)).isoformat()
    assert blocked_ts not in entry_timestamps


# ---------------------------------------------------------------------------
# one_position_per_symbol (enforced by "symbol not in positions" check)
# ---------------------------------------------------------------------------

def test_one_position_per_symbol_respected() -> None:
    """Two consecutive signals for the same symbol should open at most 1 position at a time."""
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        _row(start, 1.1000),
        _row(start + timedelta(minutes=15), 1.1005),
        _row(start + timedelta(minutes=30), 1.1010),
        _row(start + timedelta(minutes=45), 1.1015),
        _row(start + timedelta(minutes=60), 1.1020),
    ]
    # Signal on bars 0, 1, 2 — all for same EURUSD
    probs = [
        _prob_signal(),
        _prob_signal(),
        _prob_signal(),
        _prob_neutral(),
        _prob_neutral(),
    ]
    _, trades, equity_curve = run_backtest_engine(
        rows=rows,
        probabilities=probs,
        threshold=0.5,
        backtest=BacktestConfig(
            use_atr_stops=False,
            fixed_stop_loss_pct=0.002,
            fixed_take_profit_pct=0.010,
            max_holding_bars=5,
        ),
        risk=RiskConfig(max_open_positions=4),
    )
    # At most 1 open position at any time for EURUSD
    for point in equity_curve:
        assert point.open_positions <= 1


# ---------------------------------------------------------------------------
# max_open_positions
# ---------------------------------------------------------------------------

def test_max_open_positions_blocks_5th_entry() -> None:
    """With max_open_positions=4, a 5th concurrent signal should be blocked."""
    start = datetime(2026, 1, 1, 0, 0, 0)
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    rows = []
    probs = []

    # Bar 0: signal for all 4 symbols
    for sym in symbols:
        rows.append(_row(start, 1.1000, symbol=sym))
        probs.append(_prob_signal())

    # Bar 1: entry for all 4, plus signal for a 5th (EURCAD)
    for sym in symbols:
        rows.append(_row(start + timedelta(minutes=15), 1.1000, symbol=sym))
        probs.append(_prob_neutral())
    rows.append(_row(start + timedelta(minutes=15), 1.1000, symbol="EURCAD"))
    probs.append(_prob_signal())

    # Bar 2: would-be entry for EURCAD
    rows.append(_row(start + timedelta(minutes=30), 1.1000, symbol="EURCAD"))
    probs.append(_prob_neutral())
    for sym in symbols:
        rows.append(_row(start + timedelta(minutes=30), 1.1000, symbol=sym))
        probs.append(_prob_neutral())

    # Sort by (timestamp, symbol) as engine expects
    rows_sorted = sorted(rows, key=lambda r: (r.timestamp, r.symbol))
    probs_sorted = [
        probs[rows.index(r)] for r in rows_sorted
    ]

    _, trades, equity_curve = run_backtest_engine(
        rows=rows_sorted,
        probabilities=probs_sorted,
        threshold=0.5,
        backtest=BacktestConfig(
            use_atr_stops=False,
            fixed_stop_loss_pct=0.010,
            fixed_take_profit_pct=0.020,
            max_holding_bars=10,
        ),
        risk=RiskConfig(max_open_positions=4, max_daily_loss_usd=9999.0),
    )
    # open_positions never exceeds 4
    for point in equity_curve:
        assert point.open_positions <= 4


# ---------------------------------------------------------------------------
# end_of_data closes open positions
# ---------------------------------------------------------------------------

def test_end_of_data_closes_open_position() -> None:
    """An open position at end of data must appear in trades with exit_reason='end_of_data'."""
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        _row(start, 1.1000),
        _row(start + timedelta(minutes=15), 1.1000),
        # No more bars — position should be force-closed
    ]
    probs = [_prob_signal(), _prob_neutral()]
    _, trades, _ = run_backtest_engine(
        rows=rows,
        probabilities=probs,
        threshold=0.5,
        backtest=BacktestConfig(
            use_atr_stops=False,
            fixed_stop_loss_pct=0.001,
            fixed_take_profit_pct=0.010,
            max_holding_bars=100,  # won't hit time_exit with only 2 bars
        ),
        risk=RiskConfig(),
    )
    end_of_data_trades = [t for t in trades if t.exit_reason == "end_of_data"]
    assert len(end_of_data_trades) == 1


# ---------------------------------------------------------------------------
# Costs applied exactly once per side
# ---------------------------------------------------------------------------

def test_commission_applied_exactly_once_per_side() -> None:
    """
    total_commission_usd must equal 2 × commission_per_lot_per_side × volume_lots.
    (entry side + exit side, each charged once.)
    """
    start = datetime(2026, 1, 1, 0, 0, 0)
    commission_per_side = 5.0
    rows = [
        _row(start, 1.1000),
        _row(start + timedelta(minutes=15), 1.1000),
        _row(start + timedelta(minutes=30), 1.1000),
    ]
    probs = [_prob_signal(), _prob_neutral(), _prob_neutral()]
    _, trades, _ = run_backtest_engine(
        rows=rows,
        probabilities=probs,
        threshold=0.5,
        backtest=BacktestConfig(
            starting_balance_usd=1000.0,
            use_atr_stops=False,
            fixed_stop_loss_pct=0.002,
            fixed_take_profit_pct=0.004,
            spread_pips=0.0,
            slippage_pips=0.0,
            commission_per_lot_per_side_usd=commission_per_side,
            min_lot=0.01,
            lot_step=0.01,
            max_lot=100.0,
            max_holding_bars=2,
        ),
        risk=RiskConfig(risk_per_trade=0.01),  # 1% risk → gets > 0 lots
    )
    assert len(trades) == 1
    trade = trades[0]
    expected_commission = 2 * commission_per_side * trade.volume_lots
    assert abs(trade.total_commission_usd - expected_commission) < 1e-6


def test_net_pnl_equals_gross_minus_commission() -> None:
    """net_pnl = gross_pnl - total_commission for every trade."""
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        _row(start, 1.1000),
        _row(start + timedelta(minutes=15), 1.1010),
        _row(start + timedelta(minutes=30), 1.1020),
        _row(start + timedelta(minutes=45), 1.1030),
    ]
    probs = [_prob_signal(), _prob_neutral(), _prob_neutral(), _prob_neutral()]
    _, trades, _ = run_backtest_engine(
        rows=rows,
        probabilities=probs,
        threshold=0.5,
        backtest=BacktestConfig(
            use_atr_stops=False,
            fixed_stop_loss_pct=0.002,
            fixed_take_profit_pct=0.008,
            spread_pips=1.0,
            slippage_pips=0.2,
            commission_per_lot_per_side_usd=2.0,
        ),
        risk=RiskConfig(),
    )
    for trade in trades:
        expected = trade.gross_pnl_usd - trade.total_commission_usd
        assert abs(trade.net_pnl_usd - expected) < 0.001, (
            f"net={trade.net_pnl_usd} gross={trade.gross_pnl_usd} "
            f"comm={trade.total_commission_usd} expected_net={expected}"
        )
