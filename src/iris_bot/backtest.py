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
from typing import TypedDict

from iris_bot.backtest_analysis import (
    compute_signal_probabilities,
    mark_to_market as _mark_to_market,
    trade_metrics as _trade_metrics,
)
from iris_bot.backtest_pricing import (
    build_instrument,
    commission_usd as _commission_usd,
    entry_execution_price as _entry_execution_price,
    estimate_cost_breakdown as _estimate_cost_breakdown,
    exit_execution_price as _exit_execution_price,
    resolve_intrabar_exit as _resolve_intrabar_exit,
)
from iris_bot.config import BacktestConfig, DynamicExitConfig, ExitPolicyRuntimeConfig, RiskConfig, Settings
from iris_bot.consistency import verify_engine_consistency
from iris_bot.exits import SymbolExitProfile, build_exit_policies
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.processed_dataset import ProcessedDataset, ProcessedRow, load_processed_dataset
from iris_bot.risk import calculate_position_size, realized_pnl_usd
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


class TradeSummaryBucket(TypedDict):
    symbol: str
    timeframe: str
    total_trades: int
    net_pnl_usd: float
    wins: int
    losses: int
    win_rate: float


def summarize_trades_by_symbol(trades: list[TradeRecord]) -> dict[str, TradeSummaryBucket]:
    summary: dict[str, TradeSummaryBucket] = {}
    for trade in trades:
        default_bucket: TradeSummaryBucket = {
            "symbol": trade.symbol,
            "timeframe": trade.timeframe,
            "total_trades": 0,
            "net_pnl_usd": 0.0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
        }
        bucket = summary.setdefault(
            trade.symbol,
            default_bucket,
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
            raise FileNotFoundError("No experiment runs found")
        run_dir = candidates[-1]

    report_path = run_dir / "experiment_report.json"
    model_path = run_dir / "models" / "xgboost_model.json"
    if not report_path.exists():
        raise FileNotFoundError(f"experiment_report.json not found in {run_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"xgboost_model.json not found in {run_dir}")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    xgb_section = payload.get("xgboost")
    if not isinstance(xgb_section, dict):
        raise FileNotFoundError("XGBoost metadata missing from experiment_report")
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
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)

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
        logger.error("Insufficient processed rows for backtest: %s", len(rows))
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
