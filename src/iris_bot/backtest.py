"""
Backtest engine for IRIS-Bot.

Engine assumptions (explicit, auditable)
-----------------------------------------
- Entry:    executed at the OPEN of the bar AFTER the signal bar (no look-ahead)
- Exit:     triggered by stop_loss / take_profit / time_exit / end_of_data
- Intrabar: when both SL and TP are hit on the same bar, the chosen outcome
            depends on `intrabar_policy` (see BacktestConfig):
              "conservative" → stop loss wins  (default, worst outcome)
              "optimistic"   → take profit wins (best outcome)
            The TradeRecord field `is_intrabar_ambiguous` is set to True in
            either case so callers can identify and analyse these events.
- Costs:    applied exactly once per side — entry side and exit side are
            independent. Total commission = 2 × commission_per_lot_per_side.
- Cooldown: after a loss, new entries for that symbol are blocked for
            `cooldown_bars_after_loss` bars (0 = no cooldown).
- Risk:     max_open_positions and max_daily_loss_usd are checked before entry.
- Sizing:   risk-based, using stop distance and quote→account conversion.
            Returns 0.0 (blocked) if conversion rate is unavailable.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from iris_bot.config import BacktestConfig, DynamicExitConfig, ExitPolicyRuntimeConfig, RiskConfig, Settings
from iris_bot.consistency import verify_engine_consistency
from iris_bot.exits import SymbolExitProfile, build_exit_policies
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.processed_dataset import ProcessedDataset, ProcessedRow, load_processed_dataset
from iris_bot.risk import ForexInstrument, calculate_position_size, realized_pnl_usd
from iris_bot.symbols import load_symbol_strategy_profiles, row_allowed_by_profile
from iris_bot.thresholds import apply_probability_threshold
from iris_bot.xgb_model import XGBoostMultiClassModel


@dataclass(frozen=True)
class ExperimentReference:
    run_dir: Path
    model_path: Path
    report_path: Path
    threshold: float
    threshold_metric: str
    threshold_value: float
    feature_names: list[str]
    test_start_timestamp: str
    test_end_timestamp: str


@dataclass
class PendingSignal:
    direction: int
    probability_long: float
    probability_short: float
    generated_at: datetime


@dataclass
class SimulatedPosition:
    symbol: str
    timeframe: str
    direction: int
    volume_lots: float
    entry_timestamp: datetime
    entry_index: int
    signal_timestamp: datetime
    signal_probability_long: float
    signal_probability_short: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    stop_policy: str
    target_policy: str
    stop_policy_details: dict[str, object]
    target_policy_details: dict[str, object]
    commission_entry_usd: float
    bars_held: int = 0


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    timeframe: str
    direction: int
    entry_timestamp: str
    exit_timestamp: str
    signal_timestamp: str
    entry_price: float
    exit_price: float
    stop_loss_price: float
    take_profit_price: float
    volume_lots: float
    gross_pnl_usd: float
    net_pnl_usd: float
    total_commission_usd: float
    spread_cost_usd: float
    slippage_cost_usd: float
    exit_reason: str
    bars_held: int
    probability_long: float
    probability_short: float
    stop_policy: str = "static"
    target_policy: str = "static"
    stop_policy_details: dict[str, object] | None = None
    target_policy_details: dict[str, object] | None = None
    is_intrabar_ambiguous: bool = False
    # True when both SL and TP were touched on the same bar.
    # The actual outcome is determined by intrabar_policy.


@dataclass(frozen=True)
class EquityPoint:
    timestamp: str
    balance: float
    equity: float
    open_positions: int


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

def _half_spread_price(instrument: ForexInstrument, config: BacktestConfig) -> float:
    return instrument.pip_size * config.spread_pips / 2.0


def _slippage_price(instrument: ForexInstrument, config: BacktestConfig) -> float:
    return instrument.pip_size * config.slippage_pips


def _entry_execution_price(
    raw_open: float,
    direction: int,
    instrument: ForexInstrument,
    config: BacktestConfig,
) -> float:
    """Raw open ± (half-spread + slippage), adverse to the trader."""
    adverse = _half_spread_price(instrument, config) + _slippage_price(instrument, config)
    return raw_open + adverse if direction == 1 else raw_open - adverse


def _exit_execution_price(
    raw_price: float,
    direction: int,
    instrument: ForexInstrument,
    config: BacktestConfig,
) -> float:
    """Raw exit price ± (half-spread + slippage), adverse to the trader."""
    adverse = _half_spread_price(instrument, config) + _slippage_price(instrument, config)
    return raw_price - adverse if direction == 1 else raw_price + adverse


def _commission_usd(volume_lots: float, config: BacktestConfig) -> float:
    return volume_lots * config.commission_per_lot_per_side_usd


def _estimate_cost_breakdown(
    instrument: ForexInstrument,
    entry_raw_price: float,
    exit_raw_price: float,
    direction: int,
    volume_lots: float,
    config: BacktestConfig,
    aux_rates: dict[str, float] | None = None,
) -> tuple[float, float]:
    """
    Decompose total execution friction into spread cost and slippage cost.

    Returns (spread_cost_usd, slippage_cost_usd).

    spread_cost  = PnL_at_raw_prices  − PnL_at_spread_only_prices
    slippage_cost = PnL_at_spread_only − PnL_at_full_execution_prices
    """
    spread_only_entry = (
        entry_raw_price + _half_spread_price(instrument, config)
        if direction == 1
        else entry_raw_price - _half_spread_price(instrument, config)
    )
    spread_only_exit = (
        exit_raw_price - _half_spread_price(instrument, config)
        if direction == 1
        else exit_raw_price + _half_spread_price(instrument, config)
    )
    full_entry = _entry_execution_price(entry_raw_price, direction, instrument, config)
    full_exit = _exit_execution_price(exit_raw_price, direction, instrument, config)

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


# ---------------------------------------------------------------------------
# Intrabar ambiguity resolution
# ---------------------------------------------------------------------------

def _resolve_intrabar_exit(
    direction: int,
    bar_low: float,
    bar_high: float,
    stop_loss_price: float,
    take_profit_price: float,
    policy: str,
) -> tuple[float | None, str | None, bool]:
    """
    Determine if and how a position exits within a single bar.

    Parameters
    ----------
    direction:         1 for long, -1 for short
    bar_low:           Bar's low price
    bar_high:          Bar's high price
    stop_loss_price:   Stop loss level for the position
    take_profit_price: Take profit level for the position
    policy:            "conservative" (SL wins) or "optimistic" (TP wins)

    Returns
    -------
    (raw_exit_price, exit_reason, is_intrabar_ambiguous)

    - raw_exit_price:       None if no exit triggered this bar
    - exit_reason:          None | "stop_loss" | "take_profit" |
                            "stop_loss_same_bar" | "take_profit_same_bar"
    - is_intrabar_ambiguous: True only when BOTH SL and TP were hit
    """
    if direction == 1:
        sl_hit = bar_low <= stop_loss_price
        tp_hit = bar_high >= take_profit_price
    else:
        sl_hit = bar_high >= stop_loss_price
        tp_hit = bar_low <= take_profit_price

    if sl_hit and tp_hit:
        # Ambiguous: both barriers touched in the same bar
        if policy == "optimistic":
            return take_profit_price, "take_profit_same_bar", True
        else:
            # conservative (default) — stop loss wins
            return stop_loss_price, "stop_loss_same_bar", True

    if sl_hit:
        return stop_loss_price, "stop_loss", False

    if tp_hit:
        return take_profit_price, "take_profit", False

    return None, None, False


# ---------------------------------------------------------------------------
# Signal and mark-to-market helpers
# ---------------------------------------------------------------------------

def compute_signal_probabilities(
    model: XGBoostMultiClassModel,
    rows: list[ProcessedRow],
    feature_names: list[str],
) -> list[dict[int, float]]:
    matrix = [[row.features[name] for name in feature_names] for row in rows]
    return model.predict_probabilities(matrix)


def _mark_to_market(
    position: SimulatedPosition,
    bar: ProcessedRow,
    instrument: ForexInstrument,
    config: BacktestConfig,
    aux_rates: dict[str, float] | None = None,
) -> float:
    """Unrealised P&L for an open position at the current bar's close."""
    exit_price = _exit_execution_price(bar.close, position.direction, instrument, config)
    gross_pnl = realized_pnl_usd(instrument, position.entry_price, exit_price, position.direction, position.volume_lots, aux_rates)
    commission_exit = _commission_usd(position.volume_lots, config)
    return gross_pnl - position.commission_entry_usd - commission_exit


# ---------------------------------------------------------------------------
# Aggregate trade metrics
# ---------------------------------------------------------------------------

def _trade_metrics(
    trades: list[TradeRecord],
    starting_balance: float,
    ending_balance: float,
    equity_curve: list[EquityPoint],
    total_steps: int,
) -> dict[str, object]:
    total_trades = len(trades)
    wins = [t for t in trades if t.net_pnl_usd > 0.0]
    losses = [t for t in trades if t.net_pnl_usd < 0.0]
    gross_profit = sum(t.net_pnl_usd for t in wins)
    gross_loss = -sum(t.net_pnl_usd for t in losses)
    net_pnl = ending_balance - starting_balance
    average_pnl = 0.0 if total_trades == 0 else net_pnl / total_trades
    profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0)
    win_rate = 0.0 if total_trades == 0 else len(wins) / total_trades

    peak = starting_balance
    max_drawdown = 0.0
    for point in equity_curve:
        peak = max(peak, point.equity)
        drawdown = peak - point.equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    exposure = (
        0.0
        if total_steps == 0
        else sum(1 for pt in equity_curve if pt.open_positions > 0) / total_steps
    )
    return {
        "starting_balance_usd": starting_balance,
        "ending_balance_usd": ending_balance,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "net_pnl_usd": net_pnl,
        "average_pnl_usd": average_pnl,
        "expectancy_usd": average_pnl,
        "max_drawdown_usd": max_drawdown,
        "profit_factor": profit_factor,
        "exposure": exposure,
        "return_pct": 0.0 if starting_balance == 0.0 else net_pnl / starting_balance,
    }


def summarize_trades_by_symbol(trades: list[TradeRecord]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for trade in trades:
        bucket = summary.setdefault(
            trade.symbol,
            {
                "symbol": trade.symbol,
                "timeframe": trade.timeframe,
                "total_trades": 0,
                "net_pnl_usd": 0.0,
                "wins": 0,
                "losses": 0,
            },
        )
        bucket["total_trades"] += 1
        bucket["net_pnl_usd"] += trade.net_pnl_usd
        if trade.net_pnl_usd > 0.0:
            bucket["wins"] += 1
        elif trade.net_pnl_usd < 0.0:
            bucket["losses"] += 1
    for bucket in summary.values():
        n = bucket["total_trades"]
        bucket["win_rate"] = 0.0 if n == 0 else bucket["wins"] / n
    return summary


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def run_backtest_engine(
    rows: list[ProcessedRow],
    probabilities: list[dict[int, float]],
    threshold: float,
    backtest: BacktestConfig,
    risk: RiskConfig,
    intrabar_policy: str = "conservative",
    aux_rates: dict[str, float] | None = None,
    exit_policy_config: ExitPolicyRuntimeConfig | None = None,
    dynamic_exit_config: DynamicExitConfig | None = None,
    symbol_exit_profiles: dict[str, SymbolExitProfile] | None = None,
    threshold_by_symbol: dict[str, float] | None = None,
) -> tuple[dict[str, object], list[TradeRecord], list[EquityPoint]]:
    """
    Core event-driven backtest loop.

    Parameters
    ----------
    rows:             Time-ordered ProcessedRow objects (single timeframe).
    probabilities:    Model output per row, aligned by index.
    threshold:        Minimum probability for a signal to be active.
    backtest:         Cost, sizing and stop configuration.
    risk:             Position limits and daily loss controls.
    intrabar_policy:  "conservative" (SL wins) or "optimistic" (TP wins)
                      when both barriers are hit on the same bar.
    aux_rates:        Auxiliary currency conversion rates for cross pairs.
                      Not needed for default symbols (EURUSD/GBPUSD/USDJPY/AUDUSD).

    Returns
    -------
    (metrics_dict, trades_list, equity_curve_list)
    """
    if intrabar_policy not in {"conservative", "optimistic"}:
        raise ValueError(
            f"intrabar_policy must be 'conservative' or 'optimistic', got {intrabar_policy!r}"
        )

    row_key_to_probability = {
        (row.timestamp.isoformat(), row.symbol, row.timeframe): prob
        for row, prob in zip(rows, probabilities, strict=False)
    }

    rows_by_symbol: dict[str, list[ProcessedRow]] = {}
    symbol_index_map: dict[tuple[str, str], int] = {}
    for row in rows:
        rows_by_symbol.setdefault(row.symbol, []).append(row)
    for symbol, series in rows_by_symbol.items():
        series.sort(key=lambda r: r.timestamp)
        for index, item in enumerate(series):
            symbol_index_map[(symbol, item.timestamp.isoformat())] = index

    pending_signals: dict[tuple[str, int], PendingSignal] = {}
    threshold_by_symbol = threshold_by_symbol or {}
    symbol_exit_profiles = symbol_exit_profiles or {}
    exit_policy_config = exit_policy_config or ExitPolicyRuntimeConfig()
    dynamic_exit_config = dynamic_exit_config or DynamicExitConfig()
    stop_policy, target_policy = build_exit_policies(
        exit_policy_config.stop_policy,
        exit_policy_config.target_policy,
    )
    positions: dict[str, SimulatedPosition] = {}
    instruments = {symbol: build_instrument(symbol, backtest) for symbol in rows_by_symbol}
    trades: list[TradeRecord] = []
    equity_curve: list[EquityPoint] = []
    balance = backtest.starting_balance_usd
    daily_realized: dict[str, float] = {}
    cooldown_until_index: dict[str, int] = {}
    blocked_entry_count = 0

    global_rows = sorted(rows, key=lambda r: (r.timestamp, r.symbol))
    total_steps = len(global_rows)

    for row in global_rows:
        series = rows_by_symbol[row.symbol]
        series_index = symbol_index_map[(row.symbol, row.timestamp.isoformat())]
        instrument = instruments[row.symbol]
        pending_signal = pending_signals.pop((row.symbol, series_index), None)
        current_day = row.timestamp.date().isoformat()

        # --- Entry logic ---
        if pending_signal is not None and row.symbol not in positions:
            can_open = (
                len(positions) < risk.max_open_positions
                and daily_realized.get(current_day, 0.0) > -risk.max_daily_loss_usd
                and cooldown_until_index.get(row.symbol, -1) < series_index
            )
            direction_ok = (pending_signal.direction == 1 and backtest.allow_long) or (
                pending_signal.direction == -1 and backtest.allow_short
            )
            if can_open and direction_ok:
                entry_price = _entry_execution_price(row.open, pending_signal.direction, instrument, backtest)
                symbol_profile = symbol_exit_profiles.get(row.symbol)
                stop_level = stop_policy.stop_loss_price(
                    row=row,
                    entry_price=entry_price,
                    direction=pending_signal.direction,
                    backtest=backtest,
                    risk=risk,
                    dynamic_config=dynamic_exit_config,
                    symbol_profile=symbol_profile,
                )
                target_level = target_policy.take_profit_price(
                    row=row,
                    entry_price=entry_price,
                    direction=pending_signal.direction,
                    backtest=backtest,
                    risk=risk,
                    dynamic_config=dynamic_exit_config,
                    symbol_profile=symbol_profile,
                )
                stop_price = stop_level.price
                take_profit_price = target_level.price
                volume_lots = calculate_position_size(
                    balance=balance,
                    risk_per_trade=risk.risk_per_trade,
                    entry_price=entry_price,
                    stop_loss_price=stop_price,
                    instrument=instrument,
                    aux_rates=aux_rates,
                )
                if volume_lots > 0.0:
                    positions[row.symbol] = SimulatedPosition(
                        symbol=row.symbol,
                        timeframe=row.timeframe,
                        direction=pending_signal.direction,
                        volume_lots=volume_lots,
                        entry_timestamp=row.timestamp,
                        entry_index=series_index,
                        signal_timestamp=pending_signal.generated_at,
                        signal_probability_long=pending_signal.probability_long,
                        signal_probability_short=pending_signal.probability_short,
                        entry_price=entry_price,
                        stop_loss_price=stop_price,
                        take_profit_price=take_profit_price,
                        stop_policy=stop_policy.name,
                        target_policy=target_policy.name,
                        stop_policy_details=stop_level.details,
                        target_policy_details=target_level.details,
                        commission_entry_usd=_commission_usd(volume_lots, backtest),
                    )
                else:
                    blocked_entry_count += 1

        # --- Exit logic ---
        active = positions.get(row.symbol)
        if active is not None:
            raw_exit_price, exit_reason, is_ambiguous = _resolve_intrabar_exit(
                direction=active.direction,
                bar_low=row.low,
                bar_high=row.high,
                stop_loss_price=active.stop_loss_price,
                take_profit_price=active.take_profit_price,
                policy=intrabar_policy,
            )

            if exit_reason is None:
                active.bars_held += 1
                if active.bars_held >= backtest.max_holding_bars:
                    raw_exit_price = row.close
                    exit_reason = "time_exit"
                    is_ambiguous = False
            else:
                active.bars_held += 1

            if exit_reason is not None and raw_exit_price is not None:
                exit_price = _exit_execution_price(raw_exit_price, active.direction, instrument, backtest)
                gross_pnl = realized_pnl_usd(
                    instrument,
                    active.entry_price,
                    exit_price,
                    active.direction,
                    active.volume_lots,
                    aux_rates,
                )
                commission_exit = _commission_usd(active.volume_lots, backtest)
                # BUG FIX: use series[entry_index].open (not entry_index + 1)
                entry_raw_price = series[active.entry_index].open
                spread_cost_usd, slippage_cost_usd = _estimate_cost_breakdown(
                    instrument,
                    entry_raw_price,
                    raw_exit_price,
                    active.direction,
                    active.volume_lots,
                    backtest,
                    aux_rates,
                )
                net_pnl = gross_pnl - active.commission_entry_usd - commission_exit
                balance += net_pnl
                daily_realized[current_day] = daily_realized.get(current_day, 0.0) + net_pnl
                if net_pnl < 0.0 and risk.cooldown_bars_after_loss > 0:
                    cooldown_until_index[row.symbol] = series_index + risk.cooldown_bars_after_loss
                trades.append(
                    TradeRecord(
                        symbol=row.symbol,
                        timeframe=row.timeframe,
                        direction=active.direction,
                        entry_timestamp=active.entry_timestamp.isoformat(),
                        exit_timestamp=row.timestamp.isoformat(),
                        signal_timestamp=active.signal_timestamp.isoformat(),
                        entry_price=active.entry_price,
                        exit_price=exit_price,
                        stop_loss_price=active.stop_loss_price,
                        take_profit_price=active.take_profit_price,
                        volume_lots=active.volume_lots,
                        gross_pnl_usd=gross_pnl,
                        net_pnl_usd=net_pnl,
                        total_commission_usd=active.commission_entry_usd + commission_exit,
                        spread_cost_usd=spread_cost_usd,
                        slippage_cost_usd=slippage_cost_usd,
                        exit_reason=exit_reason,
                        bars_held=active.bars_held,
                        probability_long=active.signal_probability_long,
                        probability_short=active.signal_probability_short,
                        stop_policy=active.stop_policy,
                        target_policy=active.target_policy,
                        stop_policy_details=active.stop_policy_details,
                        target_policy_details=active.target_policy_details,
                        is_intrabar_ambiguous=is_ambiguous,
                    )
                )
                positions.pop(row.symbol, None)

        # --- Signal generation (current bar) ---
        probability = row_key_to_probability[(row.timestamp.isoformat(), row.symbol, row.timeframe)]
        effective_threshold = threshold_by_symbol.get(row.symbol, threshold)
        signal = apply_probability_threshold([probability], effective_threshold)[0]
        if signal != 0 and row.symbol not in positions and series_index + 1 < len(series):
            if (signal == 1 and backtest.allow_long) or (signal == -1 and backtest.allow_short):
                pending_signals[(row.symbol, series_index + 1)] = PendingSignal(
                    direction=signal,
                    probability_long=probability.get(1, 0.0),
                    probability_short=probability.get(-1, 0.0),
                    generated_at=row.timestamp,
                )

        # --- Equity snapshot ---
        equity = balance
        for pos in positions.values():
            pos_instrument = instruments[pos.symbol]
            # Use the most recent available bar for this position's symbol
            pos_bar_index = min(pos.entry_index + pos.bars_held, len(rows_by_symbol[pos.symbol]) - 1)
            pos_bar = rows_by_symbol[pos.symbol][pos_bar_index]
            equity += _mark_to_market(pos, pos_bar, pos_instrument, backtest, aux_rates)

        equity_curve.append(
            EquityPoint(
                timestamp=row.timestamp.isoformat(),
                balance=balance,
                equity=equity,
                open_positions=len(positions),
            )
        )

    # --- End-of-data: close all remaining positions ---
    for symbol, position in list(positions.items()):
        series = rows_by_symbol[symbol]
        last_row = series[-1]
        instrument = instruments[symbol]
        exit_price = _exit_execution_price(last_row.close, position.direction, instrument, backtest)
        gross_pnl = realized_pnl_usd(
            instrument,
            position.entry_price,
            exit_price,
            position.direction,
            position.volume_lots,
            aux_rates,
        )
        commission_exit = _commission_usd(position.volume_lots, backtest)
        entry_raw_price = series[position.entry_index].open
        spread_cost_usd, slippage_cost_usd = _estimate_cost_breakdown(
            instrument,
            entry_raw_price,
            last_row.close,
            position.direction,
            position.volume_lots,
            backtest,
            aux_rates,
        )
        net_pnl = gross_pnl - position.commission_entry_usd - commission_exit
        balance += net_pnl
        trades.append(
            TradeRecord(
                symbol=position.symbol,
                timeframe=position.timeframe,
                direction=position.direction,
                entry_timestamp=position.entry_timestamp.isoformat(),
                exit_timestamp=last_row.timestamp.isoformat(),
                signal_timestamp=position.signal_timestamp.isoformat(),
                entry_price=position.entry_price,
                exit_price=exit_price,
                stop_loss_price=position.stop_loss_price,
                take_profit_price=position.take_profit_price,
                volume_lots=position.volume_lots,
                gross_pnl_usd=gross_pnl,
                net_pnl_usd=net_pnl,
                total_commission_usd=position.commission_entry_usd + commission_exit,
                spread_cost_usd=spread_cost_usd,
                slippage_cost_usd=slippage_cost_usd,
                exit_reason="end_of_data",
                bars_held=position.bars_held,
                probability_long=position.signal_probability_long,
                probability_short=position.signal_probability_short,
                stop_policy=position.stop_policy,
                target_policy=position.target_policy,
                stop_policy_details=position.stop_policy_details,
                target_policy_details=position.target_policy_details,
                is_intrabar_ambiguous=False,
            )
        )
        positions.pop(symbol, None)

    metrics = _trade_metrics(trades, backtest.starting_balance_usd, balance, equity_curve, total_steps)
    metrics["by_symbol"] = summarize_trades_by_symbol(trades)
    metrics["blocked_entry_count"] = blocked_entry_count
    metrics["intrabar_ambiguous_count"] = sum(1 for t in trades if t.is_intrabar_ambiguous)
    metrics["intrabar_policy"] = intrabar_policy
    return metrics, trades, equity_curve


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_trade_log(path: Path, trades: list[TradeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(trades[0]).keys()) if trades else [
        "symbol", "timeframe", "direction", "entry_timestamp", "exit_timestamp",
        "signal_timestamp", "entry_price", "exit_price", "stop_loss_price",
        "take_profit_price", "volume_lots", "gross_pnl_usd", "net_pnl_usd",
        "total_commission_usd", "spread_cost_usd", "slippage_cost_usd",
        "exit_reason", "bars_held", "probability_long", "probability_short",
        "stop_policy", "target_policy", "stop_policy_details", "target_policy_details",
        "is_intrabar_ambiguous",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            payload = asdict(trade)
            payload["stop_policy_details"] = json.dumps(trade.stop_policy_details or {}, sort_keys=True)
            payload["target_policy_details"] = json.dumps(trade.target_policy_details or {}, sort_keys=True)
            writer.writerow(payload)


def write_equity_curve(path: Path, equity_curve: list[EquityPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "balance", "equity", "open_positions"])
        writer.writeheader()
        for point in equity_curve:
            writer.writerow(asdict(point))


# ---------------------------------------------------------------------------
# Experiment reference loader
# ---------------------------------------------------------------------------

def _locate_experiment_reference(settings: Settings) -> ExperimentReference:
    if settings.backtest.experiment_run_dir:
        run_dir = Path(settings.backtest.experiment_run_dir)
    else:
        candidates = sorted(settings.data.runs_dir.glob("*_experiment"))
        if not candidates:
            raise FileNotFoundError("No hay runs de experimento disponibles")
        run_dir = candidates[-1]

    report_path = run_dir / "experiment_report.json"
    model_path = run_dir / "models" / "xgboost_model.json"
    if not report_path.exists():
        raise FileNotFoundError(f"No existe experiment_report.json en {run_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe xgboost_model.json en {run_dir}")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    xgb_section = payload.get("xgboost")
    if not isinstance(xgb_section, dict):
        raise FileNotFoundError("No existe metadata de XGBoost en el experiment_report")
    threshold = xgb_section["threshold"]["threshold"]
    split_summary = payload["split_summary"]
    test_summary = next(item for item in split_summary if item["name"] == "test")
    return ExperimentReference(
        run_dir=run_dir,
        model_path=model_path,
        report_path=report_path,
        threshold=float(threshold),
        threshold_metric=str(xgb_section["threshold"]["metric_name"]),
        threshold_value=float(xgb_section["threshold"]["metric_value"]),
        feature_names=list(payload["feature_names"]),
        test_start_timestamp=str(test_summary["start_timestamp"]),
        test_end_timestamp=str(test_summary["end_timestamp"]),
    )


def _filter_backtest_rows(
    dataset: ProcessedDataset,
    settings: Settings,
    reference: ExperimentReference,
) -> list[ProcessedRow]:
    test_start = datetime.fromisoformat(reference.test_start_timestamp)
    test_end = datetime.fromisoformat(reference.test_end_timestamp)
    rows = [
        row
        for row in dataset.rows
        if row.timeframe == settings.trading.primary_timeframe
        and row.symbol in set(settings.trading.symbols)
        and test_start <= row.timestamp <= test_end
    ]
    rows.sort(key=lambda r: (r.timestamp, r.symbol))
    return rows


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_backtest(settings: Settings, intrabar_policy_override: str | None = None) -> int:
    """
    Run a single-pass economic backtest over the test split.

    Parameters
    ----------
    settings:               Loaded settings (from config.py).
    intrabar_policy_override: Override the settings.backtest.intrabar_policy
                              value from the CLI. None = use settings.

    Returns
    -------
    Exit code (0 = success).
    """
    run_dir = build_run_directory(settings.data.runs_dir, "backtest")
    logger = configure_logging(run_dir, settings.logging.level)

    try:
        reference = _locate_experiment_reference(settings)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    try:
        dataset = load_processed_dataset(
            settings.experiment.processed_dataset_path,
            settings.experiment.processed_schema_path,
            settings.experiment.processed_manifest_path,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    rows = _filter_backtest_rows(dataset, settings, reference)
    symbol_profiles = load_symbol_strategy_profiles(settings)
    tradable_states = {"enabled", "caution"}
    rows = [
        row
        for row in rows
        if symbol_profiles.get(row.symbol) is None
        or (
            symbol_profiles[row.symbol].enabled_state in tradable_states
            and row_allowed_by_profile(symbol_profiles[row.symbol], row.timestamp, row.timeframe)
        )
    ]
    if len(rows) < 20:
        logger.error("No hay suficientes filas procesadas para backtest: %s", len(rows))
        return 2

    model = XGBoostMultiClassModel(settings.xgboost)
    try:
        model.load(reference.model_path)
    except RuntimeError as exc:
        logger.error(str(exc))
        return 3

    policy = intrabar_policy_override or settings.backtest.intrabar_policy

    probabilities = compute_signal_probabilities(model, rows, reference.feature_names)
    metrics, trades, equity_curve = run_backtest_engine(
        rows=rows,
        probabilities=probabilities,
        threshold=reference.threshold,
        backtest=settings.backtest,
        risk=settings.risk,
        intrabar_policy=policy,
        exit_policy_config=settings.exit_policy,
        dynamic_exit_config=settings.dynamic_exits,
        symbol_exit_profiles={symbol: profile.exit_profile for symbol, profile in symbol_profiles.items()},
        threshold_by_symbol={symbol: max(reference.threshold, profile.threshold) for symbol, profile in symbol_profiles.items()},
    )

    # Consistency check
    consistency = verify_engine_consistency(
        trades=trades,
        equity_curve=equity_curve,
        starting_balance=settings.backtest.starting_balance_usd,
    )
    if not consistency.is_clean:
        logger.warning(
            "Consistency check FAILED: errors=%s warnings=%s",
            consistency.error_count,
            consistency.warning_count,
        )
    else:
        logger.info(
            "Consistency check OK: checks_passed=%s warnings=%s",
            consistency.checks_passed,
            consistency.warning_count,
        )

    write_trade_log(run_dir / "trade_log.csv", trades)
    write_equity_curve(run_dir / "equity_curve.csv", equity_curve)

    report: dict[str, object] = {
        "dataset_manifest": dataset.manifest,
        "dataset_schema": dataset.schema,
        "experiment_reference": {
            "run_dir": str(reference.run_dir),
            "report_path": str(reference.report_path),
            "model_path": str(reference.model_path),
            "threshold": reference.threshold,
            "threshold_metric": reference.threshold_metric,
            "threshold_value": reference.threshold_value,
            "test_start_timestamp": reference.test_start_timestamp,
            "test_end_timestamp": reference.test_end_timestamp,
        },
        "backtest_config": {
            "starting_balance_usd": settings.backtest.starting_balance_usd,
            "spread_pips": settings.backtest.spread_pips,
            "slippage_pips": settings.backtest.slippage_pips,
            "commission_per_lot_per_side_usd": settings.backtest.commission_per_lot_per_side_usd,
            "contract_size": settings.backtest.contract_size,
            "min_lot": settings.backtest.min_lot,
            "lot_step": settings.backtest.lot_step,
            "max_lot": settings.backtest.max_lot,
            "use_atr_stops": settings.backtest.use_atr_stops,
            "fixed_stop_loss_pct": settings.backtest.fixed_stop_loss_pct,
            "fixed_take_profit_pct": settings.backtest.fixed_take_profit_pct,
            "max_holding_bars": settings.backtest.max_holding_bars,
            "intrabar_policy": policy,
            "stop_policy": settings.exit_policy.stop_policy,
            "target_policy": settings.exit_policy.target_policy,
        },
        "dynamic_exit_config": asdict(settings.dynamic_exits),
        "strategy_profiles": {symbol: asdict(profile) for symbol, profile in symbol_profiles.items()},
        "economic_model_notes": {
            "entry": "open of bar N+1 (one bar after signal bar N)",
            "exit_triggers": ["stop_loss", "take_profit", "time_exit", "end_of_data"],
            "intrabar_ambiguity_policy": policy,
            "costs_per_side": "commission + half-spread + slippage applied independently at entry and exit",
            "aux_rates_provided": False,
            "cross_pair_support": "Non-USD crosses require aux_rates; will block entry if unavailable",
        },
        "metrics": metrics,
        "consistency": consistency.to_dict(),
        "trade_count": len(trades),
    }
    write_json_report(run_dir, "backtest_report.json", report)
    logger.info(
        "backtest trades=%s ending_balance=%.2f intrabar_policy=%s consistency=%s run_dir=%s",
        len(trades),
        metrics["ending_balance_usd"],
        policy,
        "ok" if consistency.is_clean else f"ERRORS={consistency.error_count}",
        run_dir,
    )
    return 0
