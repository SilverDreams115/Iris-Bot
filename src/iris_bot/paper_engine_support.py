from __future__ import annotations

from typing import Any, Protocol, Sequence

from iris_bot.backtest_pricing import build_instrument, commission_usd, entry_execution_price
from iris_bot.config import BacktestConfig, RiskConfig
from iris_bot.operational import (
    AccountState,
    BrokerSyncStatus,
    DailyLossTracker,
    ExposureState,
    OperationalEvent,
    PaperEngineState,
    PaperPosition,
    ProcessingState,
    SessionStatus,
    new_session_id,
)
from iris_bot.paper_types import ExecutionValidator, OrderIntent, PaperSessionConfig
from iris_bot.processed_dataset import ProcessedRow
from iris_bot.risk import calculate_position_size
from iris_bot.symbols import profile_trace as symbol_profile_trace


def update_exposure(state: PaperEngineState) -> None:
    state.exposure = ExposureState(
        open_positions=len(state.open_positions),
        gross_volume_lots=round(sum(pos.volume_lots for pos in state.open_positions.values()), 8),
        symbols=sorted(state.open_positions.keys()),
    )


def add_event(
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


class ClosedPositionLike(Protocol):
    @property
    def net_pnl_usd(self) -> float: ...

    @property
    def exit_timestamp(self) -> str: ...


def blocked(state: PaperEngineState, reason: str) -> None:
    state.blocked_trades_summary[reason] = state.blocked_trades_summary.get(reason, 0) + 1


def daily_realized(closed_positions: Sequence[ClosedPositionLike], day: str) -> float:
    return sum(trade.net_pnl_usd for trade in closed_positions if trade.exit_timestamp.startswith(day))


def initialize_engine_state(
    backtest: BacktestConfig,
    risk: RiskConfig,
    mode: str,
    initial_state: PaperEngineState | None,
) -> PaperEngineState:
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
    return state


def check_entry_gates(
    state: PaperEngineState,
    row: ProcessedRow,
    pending: dict[str, Any],
    config: PaperSessionConfig,
    series_index: int,
    events: list[OperationalEvent],
    details: dict[str, Any],
) -> str | None:
    """Returns a rejection reason string, or None if entry is allowed."""
    if row.symbol not in config.trading_symbols:
        blocked(state, "symbol_not_configured")
        add_event(events, "symbol_blocked", row, "blocked", "symbol_not_configured", details)
        return "symbol_not_configured"
    if config.one_position_per_symbol and row.symbol in state.open_positions:
        blocked(state, "position_duplicate")
        return "position_duplicate"
    if len(state.open_positions) >= config.risk.max_open_positions:
        blocked(state, "max_open_positions")
        return "max_open_positions"
    if state.daily_loss_tracker.blocked:
        blocked(state, "max_daily_loss")
        add_event(events, "max_daily_loss_block", row, "blocked", "max_daily_loss", details)
        return "max_daily_loss"
    if state.cooldown_tracker.get(row.symbol, -1) >= series_index:
        blocked(state, "cooldown_active")
        add_event(events, "cooldown_active", row, "blocked", "cooldown_active", details)
        return "cooldown_active"
    if pending["signal"] == 1 and not config.allow_long:
        blocked(state, "long_blocked")
        return "long_blocked"
    if pending["signal"] == -1 and not config.allow_short:
        blocked(state, "short_blocked")
        return "short_blocked"
    return None


def execute_entry(
    state: PaperEngineState,
    row: ProcessedRow,
    pending: dict[str, Any],
    config: PaperSessionConfig,
    stop_policy: Any,
    target_policy: Any,
    events: list[OperationalEvent],
    signal_rows: list[dict[str, Any]],
    series_index: int,
    details: dict[str, Any],
) -> str | None:
    """Executes a pending entry signal. Returns a rejection reason, or None on success."""
    symbol_exit_profiles = config.symbol_exit_profiles or {}
    symbol_strategy_profiles = config.symbol_strategy_profiles or {}
    dynamic_exit_config = config.dynamic_exit_config
    execution_validator: ExecutionValidator | None = config.execution_validator

    instrument = build_instrument(row.symbol, config.backtest)
    entry_price = entry_execution_price(row.open, pending["signal"], instrument, config.backtest)
    symbol_profile = symbol_exit_profiles.get(row.symbol)
    strategy_profile = symbol_strategy_profiles.get(row.symbol)
    profile_trace = symbol_profile_trace(strategy_profile)
    stop_level = stop_policy.stop_loss_price(
        row=row,
        entry_price=entry_price,
        direction=pending["signal"],
        backtest=config.backtest,
        risk=config.risk,
        dynamic_config=dynamic_exit_config,
        symbol_profile=symbol_profile,
    )
    target_level = target_policy.take_profit_price(
        row=row,
        entry_price=entry_price,
        direction=pending["signal"],
        backtest=config.backtest,
        risk=config.risk,
        dynamic_config=dynamic_exit_config,
        symbol_profile=symbol_profile,
    )
    stop_price = stop_level.price
    take_profit_price = target_level.price
    volume_lots = calculate_position_size(
        balance=state.account_state.balance_usd,
        risk_per_trade=config.risk.risk_per_trade,
        entry_price=entry_price,
        stop_loss_price=stop_price,
        instrument=instrument,
        aux_rates=config.aux_rates,
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
        blocked(state, "volume_invalid")
        add_event(events, "volume_rejected", row, "blocked", "volume_invalid", details)
        return "volume_invalid"

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
            blocked(state, decision.reason)
            event_type = "volume_rejected" if "volume" in decision.reason else "symbol_blocked"
            add_event(events, event_type, row, "blocked", decision.reason, details)
            return decision.reason
        add_event(events, "order_simulated", row, "accepted", "dry_run", details)

    commission_entry = commission_usd(volume_lots, config.backtest)
    state.open_positions[row.symbol] = PaperPosition(
        symbol=row.symbol,
        timeframe=row.timeframe,
        direction=pending["signal"],
        entry_timestamp=row.timestamp.isoformat(),
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
    update_exposure(state)
    add_event(
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
    return None
