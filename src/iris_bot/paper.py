from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from iris_bot.backtest import (
    _commission_usd,
    _entry_execution_price,
    _estimate_cost_breakdown,
    _exit_execution_price,
    _filter_backtest_rows,
    _locate_experiment_reference,
    _mark_to_market,
    _resolve_intrabar_exit,
    build_instrument,
    compute_signal_probabilities,
)
from iris_bot.config import DynamicExitConfig, RiskConfig, Settings
from iris_bot.consistency import verify_engine_consistency
from iris_bot.exits import SymbolExitProfile, build_exit_policies
from iris_bot.governance import resolve_active_profiles
from iris_bot.logging_utils import build_run_directory, configure_logging
from iris_bot.operational import (
    AccountState,
    BrokerSyncStatus,
    ClosedPaperTrade,
    DailyLossTracker,
    ExitPolicyConfig,
    ExposureState,
    OperationalEvent,
    PaperEngineState,
    PaperPosition,
    PaperRunArtifacts,
    PendingIntent,
    ProcessingState,
    SessionStatus,
    new_session_id,
    write_operational_artifacts,
)
from iris_bot.processed_dataset import ProcessedRow, load_processed_dataset
from iris_bot.risk import calculate_position_size, realized_pnl_usd
from iris_bot.symbols import load_symbol_strategy_profiles, row_allowed_by_profile
from iris_bot.symbols import profile_trace as symbol_profile_trace
from iris_bot.thresholds import apply_probability_threshold
from iris_bot.xgb_model import XGBoostMultiClassModel

@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_timestamp: str


@dataclass(frozen=True)
class ExecutionDecision:
    accepted: bool
    reason: str
    details: dict[str, Any]


ExecutionValidator = Callable[[OrderIntent], ExecutionDecision]


def _update_exposure(state: PaperEngineState) -> None:
    state.exposure = ExposureState(
        open_positions=len(state.open_positions),
        gross_volume_lots=round(sum(pos.volume_lots for pos in state.open_positions.values()), 8),
        symbols=sorted(state.open_positions.keys()),
    )


def _add_event(
    events: list[OperationalEvent],
    event_type: str,
    row: ProcessedRow,
    status: str,
    reason: str,
    details: dict[str, Any],
) -> None:
    events.append(
        OperationalEvent(
            event_type=event_type,
            timestamp=row.timestamp.isoformat(),
            symbol=row.symbol,
            status=status,
            reason=reason,
            details=details,
        )
    )


def _blocked(state: PaperEngineState, reason: str) -> None:
    state.blocked_trades_summary[reason] = state.blocked_trades_summary.get(reason, 0) + 1


def _daily_realized(closed_positions: list[ClosedPaperTrade], day: str) -> float:
    return sum(
        trade.net_pnl_usd
        for trade in closed_positions
        if trade.exit_timestamp.startswith(day)
    )


def run_paper_engine(
    rows: list[ProcessedRow],
    probabilities: list[dict[int, float]],
    threshold: float,
    backtest: BacktestConfig,
    risk: RiskConfig,
    trading_symbols: tuple[str, ...],
    one_position_per_symbol: bool,
    allow_long: bool,
    allow_short: bool,
    mode: str,
    execution_validator: ExecutionValidator | None = None,
    exit_policy_config: ExitPolicyConfig | None = None,
    dynamic_exit_config: DynamicExitConfig | None = None,
    intrabar_policy: str = "conservative",
    aux_rates: dict[str, float] | None = None,
    initial_state: PaperEngineState | None = None,
    should_process_row: Callable[[PaperEngineState, ProcessedRow], tuple[bool, str]] | None = None,
    symbol_exit_profiles: dict[str, SymbolExitProfile] | None = None,
    symbol_strategy_profiles: dict[str, Any] | None = None,
    threshold_by_symbol: dict[str, float] | None = None,
) -> PaperRunArtifacts:
    exit_policy_config = exit_policy_config or ExitPolicyConfig()
    dynamic_exit_config = dynamic_exit_config or DynamicExitConfig()
    symbol_exit_profiles = symbol_exit_profiles or {}
    symbol_strategy_profiles = symbol_strategy_profiles or {}
    threshold_by_symbol = threshold_by_symbol or {}
    stop_policy, target_policy = build_exit_policies(
        exit_policy_config.stop_policy,
        exit_policy_config.target_policy,
    )
    state = initial_state or PaperEngineState(
        account_state=AccountState(
            balance_usd=backtest.starting_balance_usd,
            cash_usd=backtest.starting_balance_usd,
            equity_usd=backtest.starting_balance_usd,
        ),
        daily_loss_tracker=DailyLossTracker(None, 0.0, risk.max_daily_loss_usd, False),
        current_session_status=SessionStatus(
            session_id=new_session_id(mode),
            mode=mode,
            status="running",
            last_timestamp=None,
        ),
        broker_sync_status=BrokerSyncStatus(reconciliation_policy="hard_fail"),
        processing_state=ProcessingState(),
    )
    state.current_session_status.mode = mode
    state.current_session_status.status = "running"
    events: list[OperationalEvent] = []
    signal_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []

    row_key_to_probability = {
        (row.timestamp.isoformat(), row.symbol, row.timeframe): prob
        for row, prob in zip(rows, probabilities, strict=False)
    }
    rows_by_symbol: dict[str, list[ProcessedRow]] = {}
    symbol_index_map: dict[tuple[str, str], int] = {}
    for row in rows:
        rows_by_symbol.setdefault(row.symbol, []).append(row)
    for symbol, series in rows_by_symbol.items():
        series.sort(key=lambda item: item.timestamp)
        for index, item in enumerate(series):
            symbol_index_map[(symbol, item.timestamp.isoformat())] = index
    pending_signals: dict[tuple[str, int], dict[str, Any]] = {}

    for row in sorted(rows, key=lambda item: (item.timestamp, item.symbol)):
        timestamp_text = row.timestamp.isoformat()
        if should_process_row is not None:
            allowed, reason = should_process_row(state, row)
            if not allowed:
                _add_event(events, "signal_rejected", row, "blocked", reason, {"timestamp": timestamp_text})
                signal_rows.append(
                    {
                        "timestamp": timestamp_text,
                        "symbol": row.symbol,
                        "timeframe": row.timeframe,
                        "signal": 0,
                        "probability_long": 0.0,
                        "probability_short": 0.0,
                        "threshold": threshold_by_symbol.get(row.symbol, threshold),
                        "status": "blocked",
                        "reason": reason,
                    }
                )
                continue
        series = rows_by_symbol[row.symbol]
        series_index = symbol_index_map[(row.symbol, timestamp_text)]
        state.current_session_status.last_timestamp = timestamp_text
        current_day = row.timestamp.date().isoformat()
        state.daily_loss_tracker.current_day = current_day
        state.daily_loss_tracker.realized_pnl_usd = _daily_realized(state.closed_positions, current_day)
        state.daily_loss_tracker.blocked = (
            state.daily_loss_tracker.realized_pnl_usd <= -risk.max_daily_loss_usd
        )

        pending = pending_signals.pop((row.symbol, series_index), None)
        if pending is not None:
            state.pending_intents = [
                intent
                for intent in state.pending_intents
                if not (intent.symbol == row.symbol and intent.signal_timestamp == pending["generated_at"])
            ]
            details = {
                "signal": pending["signal"],
                "probability_long": pending["probability_long"],
                "probability_short": pending["probability_short"],
                "threshold": threshold_by_symbol.get(row.symbol, threshold),
            }
            details.update(pending["profile_trace"])
            rejected_reason: str | None = None
            if row.symbol not in trading_symbols:
                rejected_reason = "symbol_not_configured"
                _blocked(state, rejected_reason)
                _add_event(events, "symbol_blocked", row, "blocked", rejected_reason, details)
            elif one_position_per_symbol and row.symbol in state.open_positions:
                rejected_reason = "position_duplicate"
                _blocked(state, rejected_reason)
            elif len(state.open_positions) >= risk.max_open_positions:
                rejected_reason = "max_open_positions"
                _blocked(state, rejected_reason)
            elif state.daily_loss_tracker.blocked:
                rejected_reason = "max_daily_loss"
                _blocked(state, rejected_reason)
                _add_event(events, "max_daily_loss_block", row, "blocked", rejected_reason, details)
            elif state.cooldown_tracker.get(row.symbol, -1) >= series_index:
                rejected_reason = "cooldown_active"
                _blocked(state, rejected_reason)
                _add_event(events, "cooldown_active", row, "blocked", rejected_reason, details)
            elif pending["signal"] == 1 and not allow_long:
                rejected_reason = "long_blocked"
                _blocked(state, rejected_reason)
            elif pending["signal"] == -1 and not allow_short:
                rejected_reason = "short_blocked"
                _blocked(state, rejected_reason)

            if rejected_reason is None:
                instrument = build_instrument(row.symbol, backtest)
                entry_price = _entry_execution_price(row.open, pending["signal"], instrument, backtest)
                symbol_profile = symbol_exit_profiles.get(row.symbol)
                strategy_profile = symbol_strategy_profiles.get(row.symbol)
                profile_trace = symbol_profile_trace(strategy_profile)
                stop_level = stop_policy.stop_loss_price(
                    row=row,
                    entry_price=entry_price,
                    direction=pending["signal"],
                    backtest=backtest,
                    risk=risk,
                    dynamic_config=dynamic_exit_config,
                    symbol_profile=symbol_profile,
                )
                target_level = target_policy.take_profit_price(
                    row=row,
                    entry_price=entry_price,
                    direction=pending["signal"],
                    backtest=backtest,
                    risk=risk,
                    dynamic_config=dynamic_exit_config,
                    symbol_profile=symbol_profile,
                )
                stop_price = stop_level.price
                take_profit_price = target_level.price
                volume_lots = calculate_position_size(
                    balance=state.account_state.balance_usd,
                    risk_per_trade=risk.risk_per_trade,
                    entry_price=entry_price,
                    stop_loss_price=stop_price,
                    instrument=instrument,
                    aux_rates=aux_rates,
                )
                details.update(
                    {
                        "entry_price": entry_price,
                        "stop_loss_price": stop_price,
                        "take_profit_price": take_profit_price,
                        "volume_lots": volume_lots,
                        "stop_policy": stop_policy.name,
                        "target_policy": target_policy.name,
                        "stop_policy_details": stop_level.details,
                        "target_policy_details": target_level.details,
                        **profile_trace,
                    }
                )
                if volume_lots <= 0.0:
                    rejected_reason = "volume_invalid"
                    _blocked(state, rejected_reason)
                    _add_event(events, "volume_rejected", row, "blocked", rejected_reason, details)
                else:
                    if execution_validator is not None:
                        decision = execution_validator(
                            OrderIntent(
                                symbol=row.symbol,
                                side="buy" if pending["signal"] == 1 else "sell",
                                volume=volume_lots,
                                entry_price=entry_price,
                                stop_loss=stop_price,
                                take_profit=take_profit_price,
                                signal_timestamp=pending["generated_at"],
                            )
                        )
                        details["execution_validation"] = decision.details
                        if not decision.accepted:
                            rejected_reason = decision.reason
                            _blocked(state, rejected_reason)
                            event_type = "volume_rejected" if "volume" in rejected_reason else "symbol_blocked"
                            _add_event(events, event_type, row, "blocked", rejected_reason, details)
                        else:
                            _add_event(events, "order_simulated", row, "accepted", "dry_run", details)
                    if rejected_reason is None:
                        commission_entry = _commission_usd(volume_lots, backtest)
                        state.open_positions[row.symbol] = PaperPosition(
                            symbol=row.symbol,
                            timeframe=row.timeframe,
                            direction=pending["signal"],
                            entry_timestamp=timestamp_text,
                            signal_timestamp=pending["generated_at"],
                            entry_index=series_index,
                            volume_lots=volume_lots,
                            entry_price=entry_price,
                            stop_loss_price=stop_price,
                            take_profit_price=take_profit_price,
                            commission_entry_usd=commission_entry,
                            bars_held=0,
                            probability_long=pending["probability_long"],
                            probability_short=pending["probability_short"],
                            stop_policy=stop_policy.name,
                            target_policy=target_policy.name,
                            stop_policy_details=stop_level.details,
                            target_policy_details=target_level.details,
                            active_profile_id=profile_trace["active_profile_id"],
                            model_variant=profile_trace["model_variant"],
                            profile_source_run_id=profile_trace["profile_source_run_id"],
                            enablement_state=profile_trace["enablement_state"],
                            promotion_state=profile_trace["promotion_state"],
                        )
                        _update_exposure(state)
                        _add_event(
                            events,
                            "position_opened",
                            row,
                            "accepted",
                            "opened",
                            {
                                **details,
                                "portfolio_open_positions": state.exposure.open_positions,
                                "portfolio_balance_usd": state.account_state.balance_usd,
                            },
                        )

            if rejected_reason is not None:
                _add_event(events, "signal_rejected", row, "blocked", rejected_reason, details)
                signal_rows.append(
                    {
                        "timestamp": timestamp_text,
                        "symbol": row.symbol,
                        "timeframe": row.timeframe,
                        "signal": pending["signal"],
                        "probability_long": pending["probability_long"],
                        "probability_short": pending["probability_short"],
                        "threshold": threshold_by_symbol.get(row.symbol, threshold),
                        "status": "blocked",
                        "reason": rejected_reason,
                    }
                )

        active = state.open_positions.get(row.symbol)
        if active is not None:
            instrument = build_instrument(row.symbol, backtest)
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
            else:
                active.bars_held += 1
            if raw_exit_price is not None and exit_reason is not None:
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
                state.account_state.balance_usd += net_pnl
                state.account_state.cash_usd = state.account_state.balance_usd
                state.daily_loss_tracker.realized_pnl_usd = _daily_realized(
                    state.closed_positions, current_day
                ) + net_pnl
                state.daily_loss_tracker.blocked = (
                    state.daily_loss_tracker.realized_pnl_usd <= -risk.max_daily_loss_usd
                )
                if net_pnl < 0.0 and risk.cooldown_bars_after_loss > 0:
                    state.cooldown_tracker[row.symbol] = (
                        series_index + risk.cooldown_bars_after_loss
                    )
                trade = ClosedPaperTrade(
                    symbol=row.symbol,
                    timeframe=row.timeframe,
                    direction=active.direction,
                    entry_timestamp=active.entry_timestamp,
                    exit_timestamp=timestamp_text,
                    signal_timestamp=active.signal_timestamp,
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
                    probability_long=active.probability_long,
                    probability_short=active.probability_short,
                    stop_policy=active.stop_policy,
                    target_policy=active.target_policy,
                    stop_policy_details=active.stop_policy_details,
                    target_policy_details=active.target_policy_details,
                    is_intrabar_ambiguous=is_ambiguous,
                    active_profile_id=active.active_profile_id,
                    model_variant=active.model_variant,
                    profile_source_run_id=active.profile_source_run_id,
                    enablement_state=active.enablement_state,
                    promotion_state=active.promotion_state,
                )
                state.closed_positions.append(trade)
                del state.open_positions[row.symbol]
                _update_exposure(state)
                event_type = (
                    "stop_loss_hit"
                    if exit_reason.startswith("stop_loss")
                    else "take_profit_hit"
                    if exit_reason.startswith("take_profit")
                    else "position_closed"
                )
                _add_event(
                    events,
                    event_type,
                    row,
                    "closed",
                    exit_reason,
                    {
                        "net_pnl_usd": net_pnl,
                        "exit_price": exit_price,
                        "portfolio_balance_usd": state.account_state.balance_usd,
                        "portfolio_open_positions": state.exposure.open_positions,
                    },
                )

        probability = row_key_to_probability[(timestamp_text, row.symbol, row.timeframe)]
        effective_threshold = threshold_by_symbol.get(row.symbol, threshold)
        signal = apply_probability_threshold([probability], effective_threshold)[0]
        state.last_signal_per_symbol[row.symbol] = {
            "timestamp": timestamp_text,
            "signal": signal,
            "probability_long": probability.get(1, 0.0),
            "probability_short": probability.get(-1, 0.0),
            "threshold": effective_threshold,
        }
        if signal != 0:
            _add_event(
                events,
                "signal_generated",
                row,
                "observed",
                "threshold_passed",
                state.last_signal_per_symbol[row.symbol],
            )
            signal_rows.append(
                {
                    "timestamp": timestamp_text,
                    "symbol": row.symbol,
                    "timeframe": row.timeframe,
                    "signal": signal,
                    "probability_long": probability.get(1, 0.0),
                    "probability_short": probability.get(-1, 0.0),
                    "threshold": effective_threshold,
                    **symbol_profile_trace(symbol_strategy_profiles.get(row.symbol)),
                    "status": "generated",
                    "reason": "threshold_passed",
                }
            )
            if series_index + 1 < len(series):
                strategy_profile = symbol_strategy_profiles.get(row.symbol)
                pending_trace = symbol_profile_trace(strategy_profile)
                pending_signals[(row.symbol, series_index + 1)] = {
                    "signal": signal,
                    "probability_long": probability.get(1, 0.0),
                    "probability_short": probability.get(-1, 0.0),
                    "generated_at": timestamp_text,
                    "profile_trace": pending_trace,
                }
                state.pending_intents.append(
                    PendingIntent(
                        symbol=row.symbol,
                        created_at=timestamp_text,
                        signal_timestamp=timestamp_text,
                        side="buy" if signal == 1 else "sell",
                        volume_lots=0.0,
                        decision_context={
                            "threshold": effective_threshold,
                            "probability_long": probability.get(1, 0.0),
                            "probability_short": probability.get(-1, 0.0),
                        },
                        active_profile_id=pending_trace["active_profile_id"],
                        model_variant=pending_trace["model_variant"],
                        profile_source_run_id=pending_trace["profile_source_run_id"],
                        enablement_state=pending_trace["enablement_state"],
                        promotion_state=pending_trace["promotion_state"],
                    )
                )

        equity = state.account_state.balance_usd
        for symbol, position in state.open_positions.items():
            instrument = build_instrument(symbol, backtest)
            pos_series = rows_by_symbol[symbol]
            pos_index = min(position.entry_index + position.bars_held, len(pos_series) - 1)
            equity += _mark_to_market(position, pos_series[pos_index], instrument, backtest, aux_rates)
        state.account_state.equity_usd = equity

    for symbol, position in list(state.open_positions.items()):
        series = rows_by_symbol[symbol]
        last_row = series[-1]
        instrument = build_instrument(symbol, backtest)
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
        state.account_state.balance_usd += net_pnl
        state.account_state.cash_usd = state.account_state.balance_usd
        trade = ClosedPaperTrade(
            symbol=symbol,
            timeframe=position.timeframe,
            direction=position.direction,
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=last_row.timestamp.isoformat(),
            signal_timestamp=position.signal_timestamp,
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
            probability_long=position.probability_long,
            probability_short=position.probability_short,
            stop_policy=position.stop_policy,
            target_policy=position.target_policy,
            stop_policy_details=position.stop_policy_details,
            target_policy_details=position.target_policy_details,
            is_intrabar_ambiguous=False,
            active_profile_id=position.active_profile_id,
            model_variant=position.model_variant,
            profile_source_run_id=position.profile_source_run_id,
            enablement_state=position.enablement_state,
            promotion_state=position.promotion_state,
        )
        state.closed_positions.append(trade)
        del state.open_positions[symbol]
        _update_exposure(state)
        events.append(
            OperationalEvent(
                event_type="position_closed",
                timestamp=last_row.timestamp.isoformat(),
                symbol=symbol,
                status="closed",
                reason="end_of_data",
                details={
                    "exit_price": exit_price,
                    "net_pnl_usd": net_pnl,
                    "active_profile_id": position.active_profile_id,
                    "model_variant": position.model_variant,
                    "profile_source_run_id": position.profile_source_run_id,
                    "enablement_state": position.enablement_state,
                    "promotion_state": position.promotion_state,
                },
            )
        )
    state.account_state.equity_usd = state.account_state.balance_usd

    execution_rows.extend(
        {
            "timestamp": event.timestamp,
            "symbol": event.symbol,
            "event_type": event.event_type,
            "status": event.status,
            "reason": event.reason,
            "active_profile_id": event.details.get("active_profile_id", ""),
            "model_variant": event.details.get("model_variant", ""),
            "profile_source_run_id": event.details.get("profile_source_run_id", ""),
            "enablement_state": event.details.get("enablement_state", ""),
            "promotion_state": event.details.get("promotion_state", ""),
            "promotion_reason": event.details.get("promotion_reason", ""),
            "volume_lots": event.details.get("volume_lots", ""),
            "entry_price": event.details.get("entry_price", ""),
            "exit_price": event.details.get("exit_price", ""),
            "stop_loss_price": event.details.get("stop_loss_price", ""),
            "take_profit_price": event.details.get("take_profit_price", ""),
            "details_json": json.dumps(event.details, sort_keys=True),
        }
        for event in events
    )

    state.current_session_status.status = "completed"
    consistency = verify_engine_consistency(
        trades=state.closed_positions,
        equity_curve=[],
        starting_balance=backtest.starting_balance_usd,
    )
    daily_summary = {
        "session_id": state.current_session_status.session_id,
        "mode": mode,
        "balance_usd": state.account_state.balance_usd,
        "equity_usd": state.account_state.equity_usd,
        "realized_pnl_usd": sum(item.net_pnl_usd for item in state.closed_positions),
        "open_positions": len(state.open_positions),
        "closed_positions": len(state.closed_positions),
        "blocked_trades_summary": dict(state.blocked_trades_summary),
    }
    validation_report = {
        "is_valid": consistency.is_clean,
        "consistency": consistency.to_dict(),
            "state_checks": {
                "open_positions_serializable": True,
                "closed_positions_serializable": True,
                "dynamic_stop_policy_implemented": exit_policy_config.stop_policy == "atr_dynamic",
                "dynamic_target_policy_implemented": exit_policy_config.target_policy == "atr_dynamic",
            },
        }
    run_report = {
        "session": asdict(state.current_session_status),
        "account_state": asdict(state.account_state),
        "daily_loss_tracker": asdict(state.daily_loss_tracker),
        "exposure": asdict(state.exposure),
        "blocked_trades_summary": dict(state.blocked_trades_summary),
        "closed_trade_count": len(state.closed_positions),
        "event_count": len(events),
        "future_exit_policy_extension": asdict(exit_policy_config or ExitPolicyConfig()),
    }
    return PaperRunArtifacts(
        state=state,
        events=events,
        closed_trades=state.closed_positions,
        daily_summary=daily_summary,
        run_report=run_report,
        validation_report=validation_report,
        signal_rows=signal_rows,
        execution_rows=execution_rows,
    )


def load_paper_context(
    settings: Settings,
    allowed_profile_states: set[str] | None = None,
) -> tuple[Any, list[ProcessedRow], Any]:
    reference = _locate_experiment_reference(settings)
    dataset = load_processed_dataset(
        settings.experiment.processed_dataset_path,
        settings.experiment.processed_schema_path,
        settings.experiment.processed_manifest_path,
    )
    rows = _filter_backtest_rows(dataset, settings, reference)
    tradable_symbols = set(settings.trading.symbols)
    rows = [row for row in rows if row.symbol in tradable_symbols]
    if allowed_profile_states is None and settings.governance.require_active_profile:
        symbol_profiles, active_report = resolve_active_profiles(settings)
        invalid = {
            symbol: payload["reasons"]
            for symbol, payload in active_report["symbols"].items()
            if not payload["ok"] and symbol in set(settings.trading.symbols)
        }
        if invalid:
            raise RuntimeError(f"Perfiles activos invalidos para operacion: {invalid}")
    else:
        symbol_profiles = load_symbol_strategy_profiles(settings)
    allowed_states = allowed_profile_states or {"enabled", "caution"}
    rows = [
        row
        for row in rows
        if symbol_profiles.get(row.symbol) is None
        or (
            symbol_profiles[row.symbol].enabled_state in allowed_states
            and row_allowed_by_profile(symbol_profiles[row.symbol], row.timestamp, row.timeframe)
        )
    ]
    if not rows:
        raise FileNotFoundError("No hay filas procesadas para la ventana de paper trading")
    model = XGBoostMultiClassModel(settings.xgboost)
    model.load(reference.model_path)
    probabilities = compute_signal_probabilities(model, rows, reference.feature_names)
    return reference, rows, probabilities


def run_paper_session(
    settings: Settings,
    mode: str,
    execution_validator: ExecutionValidator | None = None,
) -> tuple[int, Path]:
    command_name = "paper" if mode == "paper" else "demo_dry"
    run_dir = build_run_directory(settings.data.runs_dir, command_name)
    logger = configure_logging(run_dir, settings.logging.level)
    try:
        reference, rows, probabilities = load_paper_context(settings)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error(str(exc))
        return 1, run_dir

    logger.info("running %s session rows=%s", mode, len(rows))
    if settings.governance.require_active_profile:
        symbol_profiles, active_report = resolve_active_profiles(settings)
        invalid = {
            symbol: payload["reasons"]
            for symbol, payload in active_report["symbols"].items()
            if not payload["ok"] and symbol in set(settings.trading.symbols)
        }
        if invalid:
            logger.error("Perfiles activos invalidos para operacion: %s", invalid)
            return 2, run_dir
    else:
        symbol_profiles = load_symbol_strategy_profiles(settings)
    threshold_by_symbol = {
        symbol: max(reference.threshold, profile.threshold)
        for symbol, profile in symbol_profiles.items()
    }
    artifacts = run_paper_engine(
        rows=rows,
        probabilities=probabilities,
        threshold=reference.threshold,
        backtest=settings.backtest,
        risk=settings.risk,
        trading_symbols=settings.trading.symbols,
        one_position_per_symbol=settings.trading.one_position_per_symbol,
        allow_long=settings.trading.allow_long and settings.backtest.allow_long,
        allow_short=settings.trading.allow_short and settings.backtest.allow_short,
        mode=mode,
        execution_validator=execution_validator,
        intrabar_policy=settings.backtest.intrabar_policy,
        exit_policy_config=ExitPolicyConfig(
            stop_policy=settings.exit_policy.stop_policy,
            target_policy=settings.exit_policy.target_policy,
        ),
        dynamic_exit_config=settings.dynamic_exits,
        symbol_exit_profiles={symbol: profile.exit_profile for symbol, profile in symbol_profiles.items()},
        symbol_strategy_profiles=symbol_profiles,
        threshold_by_symbol=threshold_by_symbol,
    )
    config_payload = {
        "risk": asdict(settings.risk),
        "trading": asdict(settings.trading),
        "backtest": asdict(settings.backtest),
        "mt5": asdict(settings.mt5),
        "exit_policy": asdict(settings.exit_policy),
        "dynamic_exits": asdict(settings.dynamic_exits),
        "strategy_profiles": {symbol: asdict(profile) for symbol, profile in symbol_profiles.items()},
        "experiment_reference": {
            "run_dir": str(reference.run_dir),
            "model_path": str(reference.model_path),
            "report_path": str(reference.report_path),
            "threshold": reference.threshold,
            "threshold_metric": reference.threshold_metric,
            "threshold_value": reference.threshold_value,
        },
    }
    write_operational_artifacts(run_dir, artifacts, config_payload)
    logger.info(
        "%s complete closed_trades=%s blocked=%s run_dir=%s",
        mode,
        len(artifacts.closed_trades),
        sum(artifacts.state.blocked_trades_summary.values()),
        run_dir,
    )
    return 0, run_dir
