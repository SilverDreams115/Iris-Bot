"""
Tests for same-bar SL+TP ambiguity resolution (Task A – Phase 3.5).

Scenarios covered:
  - Long / short with both barriers hit → conservative → SL wins
  - Long / short with both barriers hit → optimistic   → TP wins
  - Only SL hit (no ambiguity)
  - Only TP hit (no ambiguity)
  - is_intrabar_ambiguous flag set correctly
  - Invalid policy raises ValueError
"""
from datetime import datetime, timedelta

import pytest

from iris_bot.backtest import _resolve_intrabar_exit, run_backtest_engine
from iris_bot.config import BacktestConfig, RiskConfig
from iris_bot.processed_dataset import ProcessedRow


# ---------------------------------------------------------------------------
# Unit tests for _resolve_intrabar_exit
# ---------------------------------------------------------------------------

def test_long_conservative_sl_wins_when_both_hit() -> None:
    price, exit_r, ambiguous = _resolve_intrabar_exit(
        direction=1,
        bar_low=1.0980,   # touches SL (below entry 1.1000, SL at 1.0990)
        bar_high=1.1020,  # touches TP (above entry 1.1000, TP at 1.1010)
        stop_loss_price=1.0990,
        take_profit_price=1.1010,
        policy="conservative",
    )
    assert price == 1.0990
    assert exit_r == "stop_loss_same_bar"
    assert ambiguous is True


def test_long_optimistic_tp_wins_when_both_hit() -> None:
    price, exit_r, ambiguous = _resolve_intrabar_exit(
        direction=1,
        bar_low=1.0980,
        bar_high=1.1020,
        stop_loss_price=1.0990,
        take_profit_price=1.1010,
        policy="optimistic",
    )
    assert price == 1.1010
    assert exit_r == "take_profit_same_bar"
    assert ambiguous is True


def test_short_conservative_sl_wins_when_both_hit() -> None:
    # Short: SL is above entry, TP is below entry
    price, exit_r, ambiguous = _resolve_intrabar_exit(
        direction=-1,
        bar_low=1.0980,   # touches TP
        bar_high=1.1020,  # touches SL
        stop_loss_price=1.1010,
        take_profit_price=1.0990,
        policy="conservative",
    )
    assert price == 1.1010
    assert exit_r == "stop_loss_same_bar"
    assert ambiguous is True


def test_short_optimistic_tp_wins_when_both_hit() -> None:
    price, exit_r, ambiguous = _resolve_intrabar_exit(
        direction=-1,
        bar_low=1.0980,
        bar_high=1.1020,
        stop_loss_price=1.1010,
        take_profit_price=1.0990,
        policy="optimistic",
    )
    assert price == 1.0990
    assert exit_r == "take_profit_same_bar"
    assert ambiguous is True


def test_only_sl_hit_long_no_ambiguity() -> None:
    price, exit_r, ambiguous = _resolve_intrabar_exit(
        direction=1,
        bar_low=1.0980,   # below SL
        bar_high=1.1005,  # does NOT reach TP at 1.1010
        stop_loss_price=1.0990,
        take_profit_price=1.1010,
        policy="conservative",
    )
    assert price == 1.0990
    assert exit_r == "stop_loss"
    assert ambiguous is False


def test_only_tp_hit_long_no_ambiguity() -> None:
    price, exit_r, ambiguous = _resolve_intrabar_exit(
        direction=1,
        bar_low=1.0995,   # does NOT reach SL at 1.0990
        bar_high=1.1020,  # above TP
        stop_loss_price=1.0990,
        take_profit_price=1.1010,
        policy="conservative",
    )
    assert price == 1.1010
    assert exit_r == "take_profit"
    assert ambiguous is False


def test_no_exit_within_bar() -> None:
    price, exit_r, ambiguous = _resolve_intrabar_exit(
        direction=1,
        bar_low=1.0995,
        bar_high=1.1005,
        stop_loss_price=1.0990,
        take_profit_price=1.1010,
        policy="conservative",
    )
    assert price is None
    assert exit_r is None
    assert ambiguous is False


def test_invalid_policy_raises_value_error() -> None:
    with pytest.raises(ValueError, match="intrabar_policy"):
        _row = _make_row(datetime(2026, 1, 1), 1.1000, sl=1.0980, tp=1.1020)
        run_backtest_engine(
            rows=[_row, _make_row(datetime(2026, 1, 1, 0, 15), 1.1000)],
            probabilities=[{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.3, 0: 0.4, -1: 0.3}],
            threshold=0.5,
            backtest=BacktestConfig(use_atr_stops=False),
            risk=RiskConfig(),
            intrabar_policy="invalid_policy",
        )


# ---------------------------------------------------------------------------
# Integration test: is_intrabar_ambiguous shows up in TradeRecord
# ---------------------------------------------------------------------------

def _make_row(
    ts: datetime,
    price: float,
    sl: float | None = None,
    tp: float | None = None,
    symbol: str = "EURUSD",
) -> ProcessedRow:
    bar_low = sl if sl is not None else price - 0.0010
    bar_high = tp if tp is not None else price + 0.0010
    return ProcessedRow(
        timestamp=ts,
        symbol=symbol,
        timeframe="M15",
        open=price,
        high=bar_high,
        low=bar_low,
        close=price,
        volume=100.0,
        label=1,
        label_reason="test",
        horizon_end_timestamp=ts.isoformat(),
        features={"atr_10": 0.0005, "atr_5": 0.0005},
    )


def test_is_intrabar_ambiguous_set_in_trade_record() -> None:
    """
    A bar where both SL and TP are hit should produce a trade with
    is_intrabar_ambiguous=True.
    """
    start = datetime(2026, 1, 1, 0, 0, 0)
    # Bar 0: signal (prob=0.9)
    # Bar 1: entry open, and this bar has both SL and TP in range
    rows = [
        ProcessedRow(
            timestamp=start,
            symbol="EURUSD",
            timeframe="M15",
            open=1.1000,
            high=1.1005,
            low=1.0998,
            close=1.1000,
            volume=100.0,
            label=1,
            label_reason="test",
            horizon_end_timestamp=start.isoformat(),
            features={"atr_10": 0.0005, "atr_5": 0.0005},
        ),
        ProcessedRow(
            timestamp=start + timedelta(minutes=15),
            symbol="EURUSD",
            timeframe="M15",
            open=1.1000,
            high=1.1050,  # hits TP (entry ≈ 1.1002, TP ≈ 1.1002 * 1.002 ≈ 1.1024)
            low=1.0960,   # hits SL (entry ≈ 1.1002, SL ≈ 1.1002 * 0.998 ≈ 1.0980)
            close=1.1000,
            volume=100.0,
            label=1,
            label_reason="test",
            horizon_end_timestamp=(start + timedelta(minutes=15)).isoformat(),
            features={"atr_10": 0.0005, "atr_5": 0.0005},
        ),
    ]
    probabilities = [
        {1: 0.9, 0: 0.05, -1: 0.05},
        {1: 0.3, 0: 0.35, -1: 0.35},
    ]
    _, trades, _ = run_backtest_engine(
        rows=rows,
        probabilities=probabilities,
        threshold=0.5,
        backtest=BacktestConfig(
            use_atr_stops=False,
            fixed_stop_loss_pct=0.002,
            fixed_take_profit_pct=0.002,
            spread_pips=0.0,
            slippage_pips=0.0,
            commission_per_lot_per_side_usd=0.0,
        ),
        risk=RiskConfig(risk_per_trade=0.01, max_daily_loss_usd=200.0),
        intrabar_policy="conservative",
    )
    assert len(trades) == 1
    trade = trades[0]
    assert trade.is_intrabar_ambiguous is True
    assert trade.exit_reason == "stop_loss_same_bar"


def test_is_intrabar_ambiguous_false_when_only_tp_hit() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        ProcessedRow(
            timestamp=start,
            symbol="EURUSD",
            timeframe="M15",
            open=1.1000,
            high=1.1005,
            low=1.0998,
            close=1.1000,
            volume=100.0,
            label=1,
            label_reason="test",
            horizon_end_timestamp=start.isoformat(),
            features={"atr_10": 0.0005, "atr_5": 0.0005},
        ),
        ProcessedRow(
            timestamp=start + timedelta(minutes=15),
            symbol="EURUSD",
            timeframe="M15",
            open=1.1000,
            high=1.1050,   # hits TP
            low=1.0990,    # does NOT hit SL (SL ≈ 1.0980)
            close=1.1020,
            volume=100.0,
            label=1,
            label_reason="test",
            horizon_end_timestamp=(start + timedelta(minutes=15)).isoformat(),
            features={"atr_10": 0.0005, "atr_5": 0.0005},
        ),
    ]
    probabilities = [
        {1: 0.9, 0: 0.05, -1: 0.05},
        {1: 0.3, 0: 0.35, -1: 0.35},
    ]
    _, trades, _ = run_backtest_engine(
        rows=rows,
        probabilities=probabilities,
        threshold=0.5,
        backtest=BacktestConfig(
            use_atr_stops=False,
            fixed_stop_loss_pct=0.002,
            fixed_take_profit_pct=0.002,
            spread_pips=0.0,
            slippage_pips=0.0,
            commission_per_lot_per_side_usd=0.0,
        ),
        risk=RiskConfig(risk_per_trade=0.01, max_daily_loss_usd=200.0),
        intrabar_policy="conservative",
    )
    assert len(trades) == 1
    assert trades[0].is_intrabar_ambiguous is False
    assert trades[0].exit_reason == "take_profit"
