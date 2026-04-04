"""
Tests for engine consistency validation (Task D – Phase 3.5).

Verifies that verify_engine_consistency() correctly:
  - Passes clean backtest output with no violations
  - Detects PnL math errors
  - Detects timestamp ordering violations
  - Detects duplicate positions
  - Detects final balance mismatch
  - Detects invalid direction
  - Detects invalid volume
  - Detects invalid bars_held
"""
from datetime import datetime, timedelta

from iris_bot.backtest import EquityPoint, TradeRecord, run_backtest_engine
from iris_bot.config import BacktestConfig, RiskConfig
from iris_bot.consistency import verify_engine_consistency
from iris_bot.processed_dataset import ProcessedRow


def _trade(
    symbol: str = "EURUSD",
    entry_ts: str = "2026-01-01T00:15:00",
    exit_ts: str = "2026-01-01T00:30:00",
    signal_ts: str = "2026-01-01T00:00:00",
    direction: int = 1,
    gross_pnl: float = 10.0,
    net_pnl: float = 8.0,
    commission: float = 2.0,
    volume_lots: float = 0.10,
    bars_held: int = 1,
    exit_reason: str = "take_profit",
    is_ambiguous: bool = False,
) -> TradeRecord:
    return TradeRecord(
        symbol=symbol,
        timeframe="M15",
        direction=direction,
        entry_timestamp=entry_ts,
        exit_timestamp=exit_ts,
        signal_timestamp=signal_ts,
        entry_price=1.1000,
        exit_price=1.1010,
        stop_loss_price=1.0990,
        take_profit_price=1.1020,
        volume_lots=volume_lots,
        gross_pnl_usd=gross_pnl,
        net_pnl_usd=net_pnl,
        total_commission_usd=commission,
        spread_cost_usd=0.5,
        slippage_cost_usd=0.5,
        exit_reason=exit_reason,
        bars_held=bars_held,
        probability_long=0.8,
        probability_short=0.1,
        is_intrabar_ambiguous=is_ambiguous,
    )


def _equity(balance: float = 1000.0, ts: str = "2026-01-01T00:30:00") -> EquityPoint:
    return EquityPoint(timestamp=ts, balance=balance, equity=balance, open_positions=0)


# ---------------------------------------------------------------------------
# Clean output
# ---------------------------------------------------------------------------

def test_clean_output_passes() -> None:
    trades = [_trade(gross_pnl=10.0, net_pnl=8.0, commission=2.0)]
    equity = [_equity(balance=1008.0)]
    report = verify_engine_consistency(trades, equity, starting_balance=1000.0)
    assert report.is_clean
    assert report.error_count == 0
    assert report.checks_passed > 0


def test_empty_trades_passes() -> None:
    report = verify_engine_consistency([], [_equity(1000.0)], starting_balance=1000.0)
    assert report.is_clean


# ---------------------------------------------------------------------------
# PnL math check
# ---------------------------------------------------------------------------

def test_detects_pnl_math_error() -> None:
    # net should be 8.0 (10 - 2) but we set 9.0
    trades = [_trade(gross_pnl=10.0, net_pnl=9.0, commission=2.0)]
    equity = [_equity(1009.0)]
    report = verify_engine_consistency(trades, equity, starting_balance=1000.0)
    assert not report.is_clean
    errors = [v for v in report.violations if v.severity == "error" and "PnL math" in v.message]
    assert len(errors) == 1


def test_pnl_math_within_tolerance_passes() -> None:
    # diff of 0.005 < default tolerance 0.01 → passes
    trades = [_trade(gross_pnl=10.0, net_pnl=8.005, commission=2.0)]
    equity = [_equity(1008.005)]
    report = verify_engine_consistency(trades, equity, starting_balance=1000.0, tolerance=0.01)
    assert report.is_clean


# ---------------------------------------------------------------------------
# Timestamp ordering
# ---------------------------------------------------------------------------

def test_detects_entry_after_exit() -> None:
    trades = [_trade(
        entry_ts="2026-01-01T01:00:00",
        exit_ts="2026-01-01T00:30:00",  # exit before entry
    )]
    equity = [_equity()]
    report = verify_engine_consistency(trades, equity, starting_balance=1000.0)
    assert not report.is_clean
    errors = [v for v in report.violations if "entry" in v.message and "exit" in v.message]
    assert len(errors) >= 1


def test_detects_signal_after_entry() -> None:
    trades = [_trade(
        signal_ts="2026-01-01T00:30:00",  # signal after entry
        entry_ts="2026-01-01T00:15:00",
        exit_ts="2026-01-01T00:45:00",
    )]
    equity = [_equity()]
    report = verify_engine_consistency(trades, equity, starting_balance=1000.0)
    assert not report.is_clean
    errors = [v for v in report.violations if "signal" in v.message]
    assert len(errors) >= 1


# ---------------------------------------------------------------------------
# Duplicate positions
# ---------------------------------------------------------------------------

def test_detects_duplicate_position() -> None:
    t1 = _trade(symbol="EURUSD", entry_ts="2026-01-01T00:15:00")
    t2 = _trade(symbol="EURUSD", entry_ts="2026-01-01T00:15:00")  # same symbol + ts
    equity = [_equity(1016.0)]
    report = verify_engine_consistency([t1, t2], equity, starting_balance=1000.0)
    assert not report.is_clean
    errors = [v for v in report.violations if "Duplicate" in v.message]
    assert len(errors) == 1


def test_different_symbols_same_entry_ts_ok() -> None:
    t1 = _trade(symbol="EURUSD", entry_ts="2026-01-01T00:15:00")
    t2 = _trade(symbol="GBPUSD", entry_ts="2026-01-01T00:15:00")
    equity = [_equity(1016.0)]
    report = verify_engine_consistency([t1, t2], equity, starting_balance=1000.0)
    # No duplicate error (different symbols)
    dup_errors = [v for v in report.violations if "Duplicate" in v.message]
    assert len(dup_errors) == 0


# ---------------------------------------------------------------------------
# Final balance
# ---------------------------------------------------------------------------

def test_detects_final_balance_mismatch() -> None:
    trades = [_trade(gross_pnl=10.0, net_pnl=8.0, commission=2.0)]
    # Correct final = 1000 + 8 = 1008, but we set 1050
    equity = [_equity(balance=1050.0)]
    report = verify_engine_consistency(trades, equity, starting_balance=1000.0)
    assert not report.is_clean
    errors = [v for v in report.violations if "Final balance" in v.message]
    assert len(errors) >= 1


# ---------------------------------------------------------------------------
# Direction / volume / bars_held
# ---------------------------------------------------------------------------

def test_detects_invalid_direction() -> None:
    trades = [_trade(direction=0)]  # 0 is invalid
    equity = [_equity()]
    report = verify_engine_consistency(trades, equity, starting_balance=1000.0)
    errors = [v for v in report.violations if "direction" in v.message]
    assert len(errors) >= 1


def test_detects_zero_volume() -> None:
    trades = [_trade(volume_lots=0.0)]
    equity = [_equity()]
    report = verify_engine_consistency(trades, equity, starting_balance=1000.0)
    errors = [v for v in report.violations if "volume_lots" in v.message]
    assert len(errors) >= 1


def test_detects_zero_bars_held() -> None:
    trades = [_trade(bars_held=0)]
    equity = [_equity()]
    report = verify_engine_consistency(trades, equity, starting_balance=1000.0)
    errors = [v for v in report.violations if "bars_held" in v.message]
    assert len(errors) >= 1


# ---------------------------------------------------------------------------
# Integration: engine output passes consistency check
# ---------------------------------------------------------------------------

def _row(ts: datetime, price: float = 1.1000, symbol: str = "EURUSD") -> ProcessedRow:
    return ProcessedRow(
        timestamp=ts,
        symbol=symbol,
        timeframe="M15",
        open=price,
        high=price + 0.0015,
        low=price - 0.0015,
        close=price,
        volume=100.0,
        label=1,
        label_reason="test",
        horizon_end_timestamp=ts.isoformat(),
        features={"atr_10": 0.0005, "atr_5": 0.0005},
    )


def test_engine_output_passes_consistency() -> None:
    start = datetime(2026, 1, 1)
    rows = [_row(start + timedelta(minutes=15 * i), 1.1000 + i * 0.0001) for i in range(10)]
    probabilities = [
        {1: 0.8, 0: 0.1, -1: 0.1},
        *[{1: 0.3, 0: 0.4, -1: 0.3} for _ in range(9)],
    ]
    metrics, trades, equity_curve = run_backtest_engine(
        rows=rows,
        probabilities=probabilities,
        threshold=0.6,
        backtest=BacktestConfig(
            use_atr_stops=False,
            fixed_stop_loss_pct=0.005,
            fixed_take_profit_pct=0.010,
            spread_pips=0.0,
            slippage_pips=0.0,
            commission_per_lot_per_side_usd=0.0,
            starting_balance_usd=1000.0,
        ),
        risk=RiskConfig(risk_per_trade=0.01, max_daily_loss_usd=500.0),
    )
    report = verify_engine_consistency(trades, equity_curve, starting_balance=1000.0)
    assert report.is_clean, f"Violations: {[v.message for v in report.violations]}"


def test_consistency_report_to_dict() -> None:
    report = verify_engine_consistency([], [], starting_balance=1000.0)
    d = report.to_dict()
    assert "is_clean" in d
    assert "checks_passed" in d
    assert "error_count" in d
    assert "warning_count" in d
    assert "violations" in d
    assert isinstance(d["violations"], list)
