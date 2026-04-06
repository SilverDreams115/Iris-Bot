from __future__ import annotations

from iris_bot.config import BacktestConfig
from iris_bot.risk import ForexInstrument, realized_pnl_usd


def half_spread_price(instrument: ForexInstrument, config: BacktestConfig) -> float:
    return instrument.pip_size * config.spread_pips / 2.0


def slippage_price(instrument: ForexInstrument, config: BacktestConfig) -> float:
    return instrument.pip_size * config.slippage_pips


def entry_execution_price(
    raw_open: float,
    direction: int,
    instrument: ForexInstrument,
    config: BacktestConfig,
) -> float:
    adverse = half_spread_price(instrument, config) + slippage_price(instrument, config)
    return raw_open + adverse if direction == 1 else raw_open - adverse


def exit_execution_price(
    raw_price: float,
    direction: int,
    instrument: ForexInstrument,
    config: BacktestConfig,
) -> float:
    adverse = half_spread_price(instrument, config) + slippage_price(instrument, config)
    return raw_price - adverse if direction == 1 else raw_price + adverse


def commission_usd(volume_lots: float, config: BacktestConfig) -> float:
    return volume_lots * config.commission_per_lot_per_side_usd


def estimate_cost_breakdown(
    instrument: ForexInstrument,
    entry_raw_price: float,
    exit_raw_price: float,
    direction: int,
    volume_lots: float,
    config: BacktestConfig,
    aux_rates: dict[str, float] | None = None,
) -> tuple[float, float]:
    spread_only_entry = (
        entry_raw_price + half_spread_price(instrument, config)
        if direction == 1
        else entry_raw_price - half_spread_price(instrument, config)
    )
    spread_only_exit = (
        exit_raw_price - half_spread_price(instrument, config)
        if direction == 1
        else exit_raw_price + half_spread_price(instrument, config)
    )
    full_entry = entry_execution_price(entry_raw_price, direction, instrument, config)
    full_exit = exit_execution_price(exit_raw_price, direction, instrument, config)

    gross_without_costs = realized_pnl_usd(instrument, entry_raw_price, exit_raw_price, direction, volume_lots, aux_rates)
    pnl_with_spread = realized_pnl_usd(instrument, spread_only_entry, spread_only_exit, direction, volume_lots, aux_rates)
    pnl_with_full = realized_pnl_usd(instrument, full_entry, full_exit, direction, volume_lots, aux_rates)

    spread_cost = gross_without_costs - pnl_with_spread
    slippage_cost = pnl_with_spread - pnl_with_full
    return spread_cost, slippage_cost


def build_instrument(symbol: str, config: BacktestConfig) -> ForexInstrument:
    return ForexInstrument(
        symbol=symbol,
        contract_size=config.contract_size,
        min_lot=config.min_lot,
        lot_step=config.lot_step,
        max_lot=config.max_lot,
    )


def resolve_intrabar_exit(
    direction: int,
    bar_low: float,
    bar_high: float,
    stop_loss_price: float,
    take_profit_price: float,
    policy: str,
) -> tuple[float | None, str | None, bool]:
    if direction == 1:
        sl_hit = bar_low <= stop_loss_price
        tp_hit = bar_high >= take_profit_price
    else:
        sl_hit = bar_high >= stop_loss_price
        tp_hit = bar_low <= take_profit_price

    if sl_hit and tp_hit:
        if policy == "optimistic":
            return take_profit_price, "take_profit_same_bar", True
        return stop_loss_price, "stop_loss_same_bar", True

    if sl_hit:
        return stop_loss_price, "stop_loss", False

    if tp_hit:
        return take_profit_price, "take_profit", False

    return None, None, False
