"""
Tests for generalized forex PnL and position sizing (Task B – Phase 3.5).

Covers:
  - EURUSD (XXXUSD — quote is USD)
  - GBPUSD (XXXUSD)
  - AUDUSD (XXXUSD)
  - USDJPY (USDXXX — base is USD)
  - EURGBP (cross — neither currency is USD)
  - EURJPY (cross)
  - GBPJPY (cross)
  - Blocked entry when aux_rates missing for cross pair
  - quote_to_account_rate math verification
  - calculate_position_size math verification
"""
import pytest

from iris_bot.risk import (
    ConversionRateError,
    ForexInstrument,
    calculate_position_size,
    pip_value_usd_per_lot,
    quote_to_account_rate,
    realized_pnl_usd,
)


# ---------------------------------------------------------------------------
# quote_to_account_rate
# ---------------------------------------------------------------------------

def test_eurusd_rate_is_one() -> None:
    assert quote_to_account_rate("EURUSD", 1.1000) == 1.0


def test_gbpusd_rate_is_one() -> None:
    assert quote_to_account_rate("GBPUSD", 1.2700) == 1.0


def test_audusd_rate_is_one() -> None:
    assert quote_to_account_rate("AUDUSD", 0.6500) == 1.0


def test_usdjpy_rate_is_inverse_price() -> None:
    price = 150.00
    rate = quote_to_account_rate("USDJPY", price)
    assert abs(rate - 1.0 / price) < 1e-10


def test_usdchf_rate_is_inverse_price() -> None:
    price = 0.9000
    rate = quote_to_account_rate("USDCHF", price)
    assert abs(rate - 1.0 / price) < 1e-10


def test_eurgbp_with_aux_rates() -> None:
    # EURGBP: quote=GBP → needs GBPUSD rate
    gbpusd = 1.2700
    rate = quote_to_account_rate("EURGBP", 0.8700, aux_rates={"GBP": gbpusd})
    assert rate == gbpusd


def test_eurjpy_with_aux_rates() -> None:
    # EURJPY: quote=JPY → needs JPYUSD rate
    jpyusd = 1.0 / 150.0
    rate = quote_to_account_rate("EURJPY", 163.00, aux_rates={"JPY": jpyusd})
    assert abs(rate - jpyusd) < 1e-12


def test_eurgbp_without_aux_rates_raises() -> None:
    with pytest.raises(ConversionRateError, match="EURGBP"):
        quote_to_account_rate("EURGBP", 0.8700)


def test_eurjpy_without_aux_rates_raises() -> None:
    with pytest.raises(ConversionRateError):
        quote_to_account_rate("EURJPY", 163.00)


def test_gbpjpy_without_aux_rates_raises() -> None:
    with pytest.raises(ConversionRateError):
        quote_to_account_rate("GBPJPY", 192.00)


def test_invalid_aux_rate_raises() -> None:
    with pytest.raises(ConversionRateError, match="must be > 0"):
        quote_to_account_rate("EURGBP", 0.87, aux_rates={"GBP": -1.0})


# ---------------------------------------------------------------------------
# realized_pnl_usd
# ---------------------------------------------------------------------------

def _inst(symbol: str) -> ForexInstrument:
    return ForexInstrument(symbol, contract_size=100_000.0, min_lot=0.01, lot_step=0.01, max_lot=100.0)


def test_pnl_eurusd_long_10_pips() -> None:
    # 10 pips on EURUSD = 0.0010, 0.10 lots → $10
    pnl = realized_pnl_usd(_inst("EURUSD"), 1.1000, 1.1010, 1, 0.10)
    assert round(pnl, 2) == 10.0


def test_pnl_eurusd_short_10_pips() -> None:
    # Short: price falls 10 pips → profit
    pnl = realized_pnl_usd(_inst("EURUSD"), 1.1010, 1.1000, -1, 0.10)
    assert round(pnl, 2) == 10.0


def test_pnl_gbpusd_long() -> None:
    # 20 pips on GBPUSD, 0.05 lots → 0.0020 * 100000 * 0.05 * 1.0 = $10
    pnl = realized_pnl_usd(_inst("GBPUSD"), 1.2700, 1.2720, 1, 0.05)
    assert round(pnl, 2) == 10.0


def test_pnl_audusd_long() -> None:
    # 10 pips on AUDUSD, 0.10 lots → $10
    pnl = realized_pnl_usd(_inst("AUDUSD"), 0.6500, 0.6510, 1, 0.10)
    assert round(pnl, 2) == 10.0


def test_pnl_usdjpy_long() -> None:
    """
    USDJPY long: buy 0.10 lots at 150.00, exit 150.10 (10 pips).
    PnL in JPY = (150.10 - 150.00) * 100000 * 0.10 = 1000 JPY
    JPY→USD = 1000 / 150.10 ≈ 6.66 USD
    """
    pnl = realized_pnl_usd(_inst("USDJPY"), 150.00, 150.10, 1, 0.10)
    expected = 0.10 * 100_000.0 * 0.10 / 150.10  # 0.10 JPY pip delta
    assert abs(pnl - expected) < 0.001


def test_pnl_usdjpy_short_loss() -> None:
    """Short USDJPY: price rises → loss."""
    pnl = realized_pnl_usd(_inst("USDJPY"), 150.00, 150.10, -1, 0.10)
    assert pnl < 0.0


def test_pnl_eurgbp_with_aux_rates() -> None:
    """
    EURGBP long: 0.10 lots, 10 pips move (0.0010), GBPUSD=1.2700
    PnL in GBP = 0.0010 * 100000 * 0.10 = 10 GBP
    GBP→USD = 10 * 1.2700 = 12.70 USD
    """
    pnl = realized_pnl_usd(
        _inst("EURGBP"), 0.8700, 0.8710, 1, 0.10,
        aux_rates={"GBP": 1.2700},
    )
    assert abs(pnl - 12.70) < 0.001


def test_pnl_eurjpy_with_aux_rates() -> None:
    """
    EURJPY long: 0.10 lots, 10 pips (1.00 JPY), JPYUSD=1/150
    PnL in JPY = 1.00 * 100000 * 0.10 = 10000 JPY
    JPY→USD = 10000 / 150 ≈ 66.67 USD
    """
    jpyusd = 1.0 / 150.0
    pnl = realized_pnl_usd(
        _inst("EURJPY"), 163.00, 164.00, 1, 0.10,
        aux_rates={"JPY": jpyusd},
    )
    expected = 1.00 * 100_000 * 0.10 * jpyusd
    assert abs(pnl - expected) < 0.01


def test_pnl_eurgbp_without_aux_rates_raises() -> None:
    with pytest.raises(ConversionRateError):
        realized_pnl_usd(_inst("EURGBP"), 0.8700, 0.8710, 1, 0.10)


# ---------------------------------------------------------------------------
# pip_value_usd_per_lot
# ---------------------------------------------------------------------------

def test_pip_value_eurusd() -> None:
    # 1 pip on EURUSD = 0.0001 * 100000 = $10 per lot
    pv = pip_value_usd_per_lot(_inst("EURUSD"), 1.1000)
    assert abs(pv - 10.0) < 0.001


def test_pip_value_usdjpy() -> None:
    # 1 pip on USDJPY = 0.01 * 100000 / 150.0 ≈ $6.67 per lot
    pv = pip_value_usd_per_lot(_inst("USDJPY"), 150.00)
    expected = 0.01 * 100_000 / 150.0
    assert abs(pv - expected) < 0.001


# ---------------------------------------------------------------------------
# calculate_position_size
# ---------------------------------------------------------------------------

def test_sizing_eurusd_basic() -> None:
    inst = _inst("EURUSD")
    lots = calculate_position_size(
        balance=1000.0,
        risk_per_trade=0.01,
        entry_price=1.1000,
        stop_loss_price=1.0980,
        instrument=inst,
    )
    # risk = $10, stop = 20 pips = 0.0020
    # stop_risk_per_lot = 0.0020 * 100000 * 1.0 = $200
    # raw = 10 / 200 = 0.05 lots
    assert lots == 0.05


def test_sizing_usdjpy() -> None:
    inst = _inst("USDJPY")
    lots = calculate_position_size(
        balance=1000.0,
        risk_per_trade=0.01,
        entry_price=150.00,
        stop_loss_price=149.50,  # 50 pips
        instrument=inst,
    )
    # risk = $10
    # stop = 0.50 * 100000 * (1/150) ≈ $333.33
    # raw = 10 / 333.33 ≈ 0.03 lots
    assert lots > 0.0
    assert lots <= 0.03


def test_sizing_eurgbp_with_aux_rates() -> None:
    inst = _inst("EURGBP")
    lots = calculate_position_size(
        balance=1000.0,
        risk_per_trade=0.01,
        entry_price=0.8700,
        stop_loss_price=0.8680,  # 20 pips
        instrument=inst,
        aux_rates={"GBP": 1.2700},
    )
    # risk = $10
    # stop = 0.0020 * 100000 * 1.2700 = $254
    # raw = 10 / 254 ≈ 0.039 → 0.03 lots after stepping
    assert lots > 0.0


def test_sizing_eurgbp_without_aux_rates_returns_zero() -> None:
    """Trade is blocked (0.0 lots) when conversion rate is unavailable."""
    inst = _inst("EURGBP")
    lots = calculate_position_size(
        balance=1000.0,
        risk_per_trade=0.01,
        entry_price=0.8700,
        stop_loss_price=0.8680,
        instrument=inst,
        aux_rates=None,
    )
    assert lots == 0.0


def test_sizing_zero_stop_distance_returns_zero() -> None:
    inst = _inst("EURUSD")
    lots = calculate_position_size(
        balance=1000.0,
        risk_per_trade=0.01,
        entry_price=1.1000,
        stop_loss_price=1.1000,  # same → distance 0
        instrument=inst,
    )
    assert lots == 0.0


def test_sizing_zero_balance_returns_zero() -> None:
    inst = _inst("EURUSD")
    lots = calculate_position_size(
        balance=0.0,
        risk_per_trade=0.01,
        entry_price=1.1000,
        stop_loss_price=1.0980,
        instrument=inst,
    )
    assert lots == 0.0


# ---------------------------------------------------------------------------
# ForexInstrument.requires_aux_rate()
# ---------------------------------------------------------------------------

def test_eurusd_does_not_require_aux_rate() -> None:
    assert _inst("EURUSD").requires_aux_rate() is False


def test_usdjpy_does_not_require_aux_rate() -> None:
    assert _inst("USDJPY").requires_aux_rate() is False


def test_eurgbp_requires_aux_rate() -> None:
    assert _inst("EURGBP").requires_aux_rate() is True


def test_eurjpy_requires_aux_rate() -> None:
    assert _inst("EURJPY").requires_aux_rate() is True
