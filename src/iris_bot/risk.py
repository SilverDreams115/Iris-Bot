from __future__ import annotations

from dataclasses import dataclass


class ConversionRateError(Exception):
    """
    Raised when a quote→account currency conversion rate is unavailable.

    Typically occurs for non-USD cross pairs (e.g. EURGBP, EURJPY, GBPJPY)
    where neither the base nor the quote currency is the account currency (USD).

    Resolution: provide aux_rates[quote_currency] = rate_vs_account_currency.

    Example:
        EURGBP with account=USD → needs aux_rates["GBP"] = GBPUSD_price
        EURJPY with account=USD → needs aux_rates["JPY"] = 1 / USDJPY_price
    """


@dataclass(frozen=True)
class ForexInstrument:
    """
    Descriptor for a forex trading instrument.

    Attributes:
        symbol:           6-char forex symbol, e.g. "EURUSD"
        contract_size:    Standard lot size in base currency units (default 100,000)
        min_lot:          Minimum tradeable volume in lots
        lot_step:         Lot increment
        max_lot:          Maximum tradeable volume in lots
        account_currency: Currency of the trading account (default "USD")
    """

    symbol: str
    contract_size: float
    min_lot: float
    lot_step: float
    max_lot: float
    account_currency: str = "USD"

    @property
    def base_currency(self) -> str:
        return self.symbol[:3]

    @property
    def quote_currency(self) -> str:
        return self.symbol[3:6]

    @property
    def pip_size(self) -> float:
        """0.01 for JPY-quoted pairs, 0.0001 for all others."""
        return 0.01 if self.quote_currency == "JPY" else 0.0001

    def requires_aux_rate(self) -> bool:
        """
        True when PnL cannot be resolved without an external conversion rate.

        This is the case for cross pairs where neither currency is the
        account currency, e.g. EURGBP (EUR/GBP) with account=USD.
        """
        q = self.quote_currency
        b = self.base_currency
        acc = self.account_currency
        return q != acc and b != acc


def quote_to_account_rate(
    symbol: str,
    price: float,
    account_currency: str = "USD",
    aux_rates: dict[str, float] | None = None,
) -> float:
    """
    Conversion factor from quote-currency PnL to account currency.

    This is the multiplier used in:
        realized_pnl_account = price_delta * contract_size * lots * rate

    Three cases:

    1. XXXUSD (e.g. EURUSD, GBPUSD, AUDUSD):
          quote_currency == account_currency → rate = 1.0

    2. USDXXX (e.g. USDJPY, USDCAD):
          base_currency == account_currency → rate = 1 / price
          Reason: 1 JPY = (1/USDJPY) USD when account is USD.

    3. Cross pairs (e.g. EURGBP, EURJPY, GBPJPY):
          Neither currency is the account currency.
          rate = aux_rates[quote_currency]
          Raises ConversionRateError if aux_rates not provided or missing key.

    Args:
        symbol:           6-char forex symbol (e.g. "EURUSD")
        price:            Current market price; used for USDXXX inversion
        account_currency: Account denomination (default "USD")
        aux_rates:        dict mapping currency_code → rate_vs_account_currency
                          e.g. {"GBP": 1.2700, "JPY": 0.006667, "EUR": 1.0800}

    Returns:
        Conversion rate > 0.

    Raises:
        ConversionRateError: When the rate cannot be determined.
    """
    base = symbol[:3]
    quote = symbol[3:6]

    # Case 1: XXXUSD — quote is the account currency
    if quote == account_currency:
        return 1.0

    # Case 2: USDXXX — base is the account currency
    if base == account_currency:
        if price <= 0.0:
            raise ConversionRateError(
                f"Cannot compute {account_currency}/{quote} rate for {symbol}: "
                f"price must be > 0, got {price}"
            )
        return 1.0 / price

    # Case 3: Cross pair — need explicit rate for the quote currency
    if aux_rates is not None:
        rate = aux_rates.get(quote)
        if rate is not None:
            if rate <= 0.0:
                raise ConversionRateError(
                    f"aux_rates['{quote}'] = {rate} is invalid for {symbol}: must be > 0"
                )
            return rate

    raise ConversionRateError(
        f"Cannot convert PnL for {symbol} to {account_currency}: "
        f"quote_currency={quote!r} is not {account_currency!r} and "
        f"base_currency={base!r} is not {account_currency!r}. "
        f"Provide aux_rates={{'{quote}': <{quote}/{account_currency}_spot_rate>}}. "
        f"Example for EURGBP with account=USD: aux_rates={{'GBP': 1.2700}}."
    )


def pip_value_usd_per_lot(
    instrument: ForexInstrument,
    price: float,
    aux_rates: dict[str, float] | None = None,
) -> float:
    """USD value of a 1-pip move for 1 standard lot at the given price."""
    rate = quote_to_account_rate(instrument.symbol, price, instrument.account_currency, aux_rates)
    return instrument.contract_size * instrument.pip_size * rate


def round_lot_size(raw_lots: float, min_lot: float, lot_step: float, max_lot: float) -> float:
    """Apply lot constraints: floor to step, clamp to [min_lot, max_lot].

    Uses round() before int() to avoid floating-point truncation errors
    (e.g. 0.05 / 0.01 = 4.9999... → int() = 4 without rounding).
    """
    if raw_lots <= 0.0:
        return 0.0
    stepped = int(round(raw_lots / lot_step, 8)) * lot_step
    bounded = min(max_lot, stepped)
    return max(min_lot, round(bounded, 8)) if bounded >= min_lot else 0.0


def calculate_position_size(
    balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    instrument: ForexInstrument,
    aux_rates: dict[str, float] | None = None,
) -> float:
    """
    Risk-based position sizing.

    Formula:
        risk_amount = balance * risk_per_trade
        stop_risk_usd_per_lot = |entry - stop_loss| * contract_size * quote_to_account_rate
        raw_lots = risk_amount / stop_risk_usd_per_lot

    Returns 0.0 (trade blocked) when:
        - balance or risk_per_trade is zero
        - stop_distance is zero
        - ConversionRateError: aux_rates missing for a cross pair
        - sized lots fall below min_lot after rounding
    """
    risk_amount = max(balance, 0.0) * max(risk_per_trade, 0.0)
    stop_distance = abs(entry_price - stop_loss_price)
    if risk_amount == 0.0 or stop_distance == 0.0:
        return 0.0

    try:
        quote_to_account = quote_to_account_rate(
            instrument.symbol, entry_price, instrument.account_currency, aux_rates
        )
    except ConversionRateError:
        return 0.0

    stop_risk_usd_per_lot = stop_distance * instrument.contract_size * quote_to_account
    if stop_risk_usd_per_lot <= 0.0:
        return 0.0

    raw_size = risk_amount / stop_risk_usd_per_lot
    return round_lot_size(raw_size, instrument.min_lot, instrument.lot_step, instrument.max_lot)


def realized_pnl_usd(
    instrument: ForexInstrument,
    entry_price: float,
    exit_price: float,
    direction: int,
    volume_lots: float,
    aux_rates: dict[str, float] | None = None,
) -> float:
    """
    Realized P&L converted to account currency (USD by default).

    Formula:
        price_delta = (exit_price - entry_price) * direction
        quote_pnl   = price_delta * contract_size * volume_lots
        pnl_usd     = quote_pnl * quote_to_account_rate(exit_price)

    Note: conversion rate is evaluated at exit_price, which is the standard
    convention for marking closed positions.

    Args:
        instrument:   ForexInstrument descriptor
        entry_price:  Execution price at entry (after spread/slippage)
        exit_price:   Execution price at exit  (after spread/slippage)
        direction:    1 for long, -1 for short
        volume_lots:  Position size in standard lots
        aux_rates:    Auxiliary conversion rates for cross pairs

    Raises:
        ConversionRateError: If the quote→account conversion is unavailable.
    """
    price_delta = (exit_price - entry_price) * direction
    quote_pnl = price_delta * instrument.contract_size * volume_lots
    rate = quote_to_account_rate(instrument.symbol, exit_price, instrument.account_currency, aux_rates)
    return quote_pnl * rate
