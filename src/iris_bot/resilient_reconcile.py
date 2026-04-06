from __future__ import annotations

import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Callable

from iris_bot.config import RecoveryConfig, ReconciliationConfig
from iris_bot.mt5 import MT5Client, OrderRequest
from iris_bot.operational import AlertRecord, PaperEngineState
from iris_bot.paper import ExecutionDecision, OrderIntent
from iris_bot.resilient_models import (
    BrokerEventDecision,
    BrokerPositionSnapshot,
    BrokerStateSnapshot,
    ReconciliationDiscrepancy,
    ReconciliationOutcome,
    ReconnectReport,
    RestoreReport,
    now_iso,
)


def classify_broker_event(payload: dict[str, Any]) -> BrokerEventDecision:
    reason = str(payload.get("reason", "")).lower()
    if "partial" in reason:
        return BrokerEventDecision("partial_fill", "log_only", False, False, payload)
    if "money" in reason:
        return BrokerEventDecision("not_enough_money", "blocked", False, True, payload)
    if "market" in reason and "closed" in reason:
        return BrokerEventDecision("market_closed", "blocked", False, True, payload)
    if "requote" in reason:
        return BrokerEventDecision("requote", "retry", True, False, payload)
    if "trading" in reason and "disabled" in reason:
        return BrokerEventDecision("trading_disabled", "blocked", False, True, payload)
    if "volume" in reason:
        return BrokerEventDecision("volume_invalid", "blocked", False, True, payload)
    if "filling" in reason:
        return BrokerEventDecision("filling_mode_rejected", "blocked", False, True, payload)
    if "communication" in reason or "connect" in reason:
        return BrokerEventDecision("communication_error", "retry", True, False, payload)
    if payload.get("accepted") is False:
        return BrokerEventDecision("order_rejected", "blocked", False, True, payload)
    return BrokerEventDecision("accepted", "log_only", False, False, payload)


def _check_balance_discrepancy(
    local_state: PaperEngineState,
    broker_state: BrokerStateSnapshot,
    config: ReconciliationConfig,
    compare: bool,
) -> list[ReconciliationDiscrepancy]:
    if not compare or broker_state.balance_usd is None:
        return []
    if abs(local_state.account_state.balance_usd - broker_state.balance_usd) <= config.price_tolerance:
        return []
    return [ReconciliationDiscrepancy(
        "price_mismatch",
        "critical",
        f"Balance mismatch: local={local_state.account_state.balance_usd} broker={broker_state.balance_usd}",
        {"local_balance_usd": local_state.account_state.balance_usd, "broker_balance_usd": broker_state.balance_usd},
    )]


def _detect_orphan_local(local_symbols: set[str], broker_symbols: set[str]) -> list[ReconciliationDiscrepancy]:
    return [
        ReconciliationDiscrepancy(
            "missing_in_broker",
            "critical",
            f"Position for {symbol!r} exists locally but is absent from broker snapshot",
            {"symbol": symbol},
        )
        for symbol in sorted(local_symbols - broker_symbols)
    ]


def _detect_orphan_broker(local_symbols: set[str], broker_symbols: set[str]) -> list[ReconciliationDiscrepancy]:
    return [
        ReconciliationDiscrepancy(
            "missing_in_local_state",
            "critical",
            f"Broker reports open position for {symbol!r} not found in local state",
            {"symbol": symbol},
        )
        for symbol in sorted(broker_symbols - local_symbols)
    ]


def _check_position_details(
    local_state: PaperEngineState,
    broker_by_symbol: dict[str, BrokerPositionSnapshot],
    config: ReconciliationConfig,
) -> list[ReconciliationDiscrepancy]:
    discrepancies: list[ReconciliationDiscrepancy] = []
    common = sorted(set(local_state.open_positions) & set(broker_by_symbol))
    for symbol in common:
        local = local_state.open_positions[symbol]
        broker = broker_by_symbol[symbol]
        local_side = "buy" if local.direction == 1 else "sell"
        if local_side != broker.side:
            discrepancies.append(ReconciliationDiscrepancy(
                "side_mismatch", "critical",
                f"Side mismatch for {symbol!r}: local={local_side} broker={broker.side}",
                {"symbol": symbol, "local": local_side, "broker": broker.side},
            ))
        if abs(local.volume_lots - broker.volume_lots) > config.volume_tolerance:
            discrepancies.append(ReconciliationDiscrepancy(
                "volume_mismatch", "critical",
                f"Volume mismatch for {symbol!r}: local={local.volume_lots} broker={broker.volume_lots}",
                {"symbol": symbol, "local": local.volume_lots, "broker": broker.volume_lots},
            ))
        if abs(local.entry_price - broker.price_open) > config.price_tolerance:
            discrepancies.append(ReconciliationDiscrepancy(
                "price_mismatch", "warning",
                f"Entry price mismatch for {symbol!r}: local={local.entry_price} broker={broker.price_open}",
                {"symbol": symbol, "local": local.entry_price, "broker": broker.price_open},
            ))
    return discrepancies


def _check_event_deduplication(local_state: PaperEngineState) -> list[ReconciliationDiscrepancy]:
    ids = local_state.processing_state.processed_event_ids
    if len(ids) == len(set(ids)):
        return []
    return [ReconciliationDiscrepancy(
        "duplicate_state", "critical",
        "Duplicate processed event IDs detected in local state",
        {"total": len(ids), "unique": len(set(ids))},
    )]


def _check_stale_state(local_state: PaperEngineState, broker_state: BrokerStateSnapshot) -> list[ReconciliationDiscrepancy]:
    if not local_state.current_session_status.last_timestamp or not broker_state.closed_trades:
        return []
    broker_close_times: list[datetime] = []
    for item in broker_state.closed_trades:
        raw_time = item.get("time")
        if isinstance(raw_time, (int, float)):
            broker_close_times.append(datetime.fromtimestamp(raw_time, tz=UTC).replace(tzinfo=None))
        elif isinstance(raw_time, str) and raw_time:
            try:
                broker_close_times.append(datetime.fromisoformat(raw_time))
            except ValueError:
                continue
    if not broker_close_times:
        return []
    latest_broker_close = max(broker_close_times)
    local_last = datetime.fromisoformat(local_state.current_session_status.last_timestamp)
    if latest_broker_close >= local_last:
        return []
    return [ReconciliationDiscrepancy(
        "stale_state", "warning",
        "Broker closed-trade history is older than local session timestamp",
        {"latest_broker_close": latest_broker_close.isoformat(), "local_last_timestamp": local_last.isoformat()},
    )]


def _apply_reconciliation_policy(
    config: ReconciliationConfig,
    broker_state: BrokerStateSnapshot,
    synced: dict[str, Any],
) -> str:
    if config.policy == "soft_resync":
        synced["latest_broker_snapshot"] = broker_state.to_dict()
        if broker_state.balance_usd is not None:
            synced["account_state"]["balance_usd"] = broker_state.balance_usd
            synced["account_state"]["cash_usd"] = broker_state.balance_usd
        if broker_state.equity_usd is not None:
            synced["account_state"]["equity_usd"] = broker_state.equity_usd
        return "soft_resync"
    if config.policy == "block":
        return "blocked"
    if config.policy == "hard_fail":
        return "hard_fail"
    return "log_only"


def reconcile_state(
    local_state: PaperEngineState,
    broker_state: BrokerStateSnapshot,
    config: ReconciliationConfig,
    compare_account_state: bool = True,
) -> ReconciliationOutcome:
    broker_by_symbol = {item.symbol: item for item in broker_state.positions}
    local_symbols = set(local_state.open_positions.keys())
    broker_symbols = set(broker_by_symbol.keys())

    discrepancies: list[ReconciliationDiscrepancy] = [
        *_check_balance_discrepancy(local_state, broker_state, config, compare_account_state),
        *_detect_orphan_local(local_symbols, broker_symbols),
        *_detect_orphan_broker(local_symbols, broker_symbols),
        *_check_position_details(local_state, broker_by_symbol, config),
        *_check_event_deduplication(local_state),
        *_check_stale_state(local_state, broker_state),
    ]

    synced = local_state.to_dict()
    critical = [item for item in discrepancies if item.severity == "critical"]
    action = _apply_reconciliation_policy(config, broker_state, synced) if critical else "log_only"
    return ReconciliationOutcome(not critical, action, discrepancies, synced)


def reconnect_mt5(client: MT5Client, recovery: RecoveryConfig) -> ReconnectReport:
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, recovery.reconnect_retries + 1):
        ok = client.connect()
        attempts.append({"attempt": attempt, "ok": ok, "timestamp": now_iso(), "last_error": client.last_error()})
        if ok:
            return ReconnectReport(True, "connected", attempts)
        if recovery.reconnect_backoff_seconds > 0:
            time.sleep(recovery.reconnect_backoff_seconds)
    return ReconnectReport(False, "blocked", attempts)


def broker_snapshot_from_mt5(snapshot: Any) -> BrokerStateSnapshot:
    account = snapshot.account_info
    positions = [
        BrokerPositionSnapshot(
            ticket=str(item.get("ticket", item.get("identifier", ""))),
            symbol=str(item.get("symbol", "")),
            side="buy" if int(item.get("type", 0)) in {0} else "sell",
            volume_lots=float(item.get("volume", item.get("volume_current", 0.0))),
            price_open=float(item.get("price_open", 0.0)),
            stop_loss=float(item.get("sl", 0.0)),
            take_profit=float(item.get("tp", 0.0)),
            time=str(item.get("time", "")),
        )
        for item in snapshot.positions
    ]
    return BrokerStateSnapshot(
        connected=snapshot.connected,
        balance_usd=float(account.get("balance", 0.0)) if account else None,
        equity_usd=float(account.get("equity", 0.0)) if account else None,
        positions=positions,
        closed_trades=snapshot.closed_trades,
        pending_orders=snapshot.pending_orders,
        raw_account=account,
        scope_report=getattr(snapshot, "scope_report", {}),
    )


def build_operational_status(
    state: PaperEngineState,
    reconciliation: ReconciliationOutcome,
    restore: RestoreReport,
    alerts: list[AlertRecord],
) -> dict[str, Any]:
    return {
        "session_status": state.current_session_status.status,
        "broker_sync_status": asdict(state.broker_sync_status),
        "restore_ok": restore.ok,
        "restore_action": restore.action,
        "reconciliation_ok": reconciliation.ok,
        "reconciliation_action": reconciliation.action,
        "open_positions": len(state.open_positions),
        "closed_positions": len(state.closed_positions),
        "alerts_count": len(alerts),
        "blocked_reasons": list(state.blocked_reasons),
    }


def build_live_execution_validator(client: MT5Client) -> Callable[[OrderIntent], ExecutionDecision]:
    """Execution validator that ACTUALLY sends market orders to MT5 (demo or live account).

    Use this only when real order routing is intended. For shadow/dry-run validation
    without sending orders, use build_demo_execution_validator instead.
    """
    def validator(intent: OrderIntent) -> ExecutionDecision:
        result = client.send_market_order(
            OrderRequest(
                symbol=intent.symbol,
                side=intent.side,
                volume=intent.volume,
                stop_loss=intent.stop_loss,
                take_profit=intent.take_profit,
                price=intent.entry_price,
            )
        )
        decision = classify_broker_event(result.to_dict())
        return ExecutionDecision(
            accepted=result.accepted,
            reason="order_sent" if result.accepted else decision.classification,
            details={"broker_result": result.to_dict(), "broker_decision": asdict(decision)},
        )
    return validator


def build_demo_execution_validator(client: MT5Client) -> Callable[[OrderIntent], ExecutionDecision]:
    def validator(intent: OrderIntent) -> ExecutionDecision:
        result = client.dry_run_market_order(
            OrderRequest(
                symbol=intent.symbol,
                side=intent.side,
                volume=intent.volume,
                stop_loss=intent.stop_loss,
                take_profit=intent.take_profit,
                price=intent.entry_price,
            )
        )
        decision = classify_broker_event(result.to_dict())
        return ExecutionDecision(
            accepted=result.accepted,
            reason=decision.classification if result.accepted else decision.classification,
            details={"broker_result": result.to_dict(), "broker_decision": asdict(decision)},
        )
    return validator
