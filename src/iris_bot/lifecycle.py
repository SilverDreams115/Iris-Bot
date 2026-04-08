from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from iris_bot.artifacts import read_artifact_payload, wrap_artifact
from iris_bot.config import Settings
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.mt5 import MT5Client
from iris_bot.resilient import build_runtime_state_path, restore_runtime_state


@dataclass(frozen=True)
class LifecycleMismatch:
    category: str
    severity: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_iso_datetime(timestamp_text: str) -> datetime | None:
    if not timestamp_text:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _is_recent_trade(exit_timestamp: str, history_days: int) -> bool:
    parsed = _parse_iso_datetime(exit_timestamp)
    if parsed is None:
        return False
    return parsed >= datetime.now(tz=UTC) - timedelta(days=history_days)


def _parse_local_intents(state: Any, *, history_days: int) -> list[dict[str, Any]]:
    items = []
    for intent in state.pending_intents:
        items.append(
            {
                "symbol": intent.symbol,
                "created_at": intent.created_at,
                "signal_timestamp": intent.signal_timestamp,
                "side": intent.side,
                "volume_lots": intent.volume_lots,
                "active_profile_id": intent.active_profile_id,
                "promotion_state": intent.promotion_state,
            }
        )
    for symbol, position in state.open_positions.items():
        items.append(
            {
                "symbol": symbol,
                "created_at": position.entry_timestamp,
                "signal_timestamp": position.signal_timestamp,
                "side": "buy" if position.direction == 1 else "sell",
                "volume_lots": position.volume_lots,
                "active_profile_id": position.active_profile_id,
                "promotion_state": position.promotion_state,
                "local_position": True,
            }
        )
    for trade in state.closed_positions:
        if not _is_recent_trade(trade.exit_timestamp, history_days):
            continue
        items.append(
            {
                "symbol": trade.symbol,
                "created_at": trade.entry_timestamp,
                "signal_timestamp": trade.signal_timestamp,
                "side": "buy" if trade.direction == 1 else "sell",
                "volume_lots": trade.volume_lots,
                "active_profile_id": trade.active_profile_id,
                "promotion_state": trade.promotion_state,
                "local_closed_trade": True,
                "exit_timestamp": trade.exit_timestamp,
                "exit_reason": trade.exit_reason,
            }
        )
    return items


def reconcile_lifecycle_records(
    local_intents: list[dict[str, Any]],
    broker_trace: dict[str, Any],
    volume_tolerance: float = 1e-6,
) -> dict[str, Any]:
    mismatches: list[LifecycleMismatch] = []
    orders = broker_trace.get("orders", [])
    deals = broker_trace.get("deals", [])
    positions = broker_trace.get("positions", [])
    broker_symbols = {str(item.get("symbol", "")) for item in orders + deals + positions}
    local_symbols = {str(item.get("symbol", "")) for item in local_intents}

    order_tickets: list[str] = []
    for item in orders:
        order_tickets.append(str(item.get("ticket", item.get("order", ""))))
    if len(order_tickets) != len(set(order_tickets)):
        mismatches.append(LifecycleMismatch("duplicate_broker_event", "critical", "Duplicate broker order ticket detected", {}))

    deal_tickets: list[str] = []
    deals_by_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in deals:
        ticket = str(item.get("ticket", item.get("deal", "")))
        deal_tickets.append(ticket)
        order_key = str(item.get("order", item.get("position_id", item.get("position", ""))))
        deals_by_order[order_key].append(item)
    if len(deal_tickets) != len(set(deal_tickets)):
        mismatches.append(LifecycleMismatch("duplicate_broker_event", "critical", "Duplicate broker deal ticket detected", {}))

    for local in local_intents:
        symbol = str(local.get("symbol", ""))
        matching_orders = [item for item in orders if str(item.get("symbol", "")) == symbol]
        matching_deals = [item for item in deals if str(item.get("symbol", "")) == symbol]
        matching_positions = [item for item in positions if str(item.get("symbol", "")) == symbol]
        if not matching_orders and not matching_deals and not matching_positions:
            mismatches.append(
                LifecycleMismatch(
                    "local_intent_without_broker_evidence",
                    "critical",
                    "Local intent has no broker evidence in bot scope",
                    {"symbol": symbol, "signal_timestamp": local.get("signal_timestamp")},
                )
            )
            continue
        local_volume = float(local.get("volume_lots", 0.0) or 0.0)
        if matching_positions and local_volume > 0.0:
            broker_volume = sum(float(item.get("volume", item.get("volume_current", 0.0)) or 0.0) for item in matching_positions)
            if abs(local_volume - broker_volume) > volume_tolerance:
                mismatches.append(
                    LifecycleMismatch(
                        "position_mismatch",
                        "critical",
                        "Broker position volume differs from local intent/position",
                        {"symbol": symbol, "local_volume_lots": local_volume, "broker_volume_lots": broker_volume},
                    )
                )

    for symbol in sorted(broker_symbols - local_symbols):
        mismatches.append(
            LifecycleMismatch(
                "broker_event_without_local_intent",
                "critical",
                "Broker scope activity exists without local evidence",
                {"symbol": symbol},
            )
        )

    for order in orders:
        initial_volume = float(order.get("volume_initial", order.get("volume", 0.0)) or 0.0)
        current_volume = float(order.get("volume_current", initial_volume) or 0.0)
        if 0.0 < current_volume < initial_volume:
            mismatches.append(
                LifecycleMismatch(
                    "partial_fill_real",
                    "warning",
                    "Broker order indicates partial fill progression",
                    {"symbol": order.get("symbol"), "ticket": order.get("ticket", order.get("order")), "volume_initial": initial_volume, "volume_current": current_volume},
                )
            )
        order_key = str(order.get("ticket", order.get("order", "")))
        grouped_deals = deals_by_order.get(order_key, [])
        total_deal_volume = sum(float(item.get("volume", 0.0) or 0.0) for item in grouped_deals)
        if initial_volume > 0.0 and total_deal_volume - initial_volume > volume_tolerance:
            mismatches.append(
                LifecycleMismatch(
                    "fill_progression_inconsistency",
                    "critical",
                    "Aggregated deal volume exceeds broker order volume",
                    {"ticket": order_key, "order_volume": initial_volume, "deal_volume": total_deal_volume},
                )
            )

    critical_count = sum(1 for item in mismatches if item.severity == "critical")
    mismatch_counts: dict[str, int] = {}
    for item in mismatches:
        mismatch_counts[item.category] = mismatch_counts.get(item.category, 0) + 1
    return {
        "ok": critical_count == 0,
        "critical_mismatch_count": critical_count,
        "mismatch_counts": mismatch_counts,
        "mismatches": [item.to_dict() for item in mismatches],
    }


def run_lifecycle_reconciliation(
    settings: Settings,
    client_factory: Callable[[], MT5Client] | None = None,
) -> tuple[int, Path]:
    run_dir = build_run_directory(settings.data.runs_dir, "lifecycle_reconciliation")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    state_path = build_runtime_state_path(settings)
    local_state, restore = restore_runtime_state(state_path, require_clean=False)
    if local_state is None:
        logger.error("No se pudo restaurar estado local para lifecycle reconciliation")
        return 1, run_dir
    client = client_factory() if client_factory is not None else MT5Client(settings.mt5)
    if not client.connect():
        logger.error("No se pudo conectar MT5 para lifecycle reconciliation")
        return 2, run_dir
    try:
        broker_trace = client.broker_lifecycle_snapshot(settings.trading.symbols, settings.lifecycle.history_days)
    finally:
        client.shutdown()
    local_trace = _parse_local_intents(local_state, history_days=settings.lifecycle.history_days)
    reconciliation = reconcile_lifecycle_records(local_trace, broker_trace, volume_tolerance=settings.reconciliation.volume_tolerance)
    symbols: dict[str, Any] = {}
    for symbol in settings.trading.symbols:
        symbol_mismatches = [item for item in reconciliation["mismatches"] if item.get("details", {}).get("symbol") == symbol]
        symbols[symbol] = {
            "critical_mismatch_count": sum(1 for item in symbol_mismatches if item["severity"] == "critical"),
            "mismatch_categories": sorted({item["category"] for item in symbol_mismatches}),
        }
    payload = {
        **reconciliation,
        "symbols": symbols,
        "restore_state_report": restore.to_dict(),
        "scope_report": broker_trace.get("scope_report", {}),
    }
    write_json_report(run_dir, "lifecycle_reconciliation_report.json", wrap_artifact("lifecycle_reconciliation", payload))
    write_json_report(run_dir, "broker_event_trace.json", wrap_artifact("lifecycle_reconciliation", broker_trace))
    write_json_report(run_dir, "local_intent_trace.json", wrap_artifact("lifecycle_reconciliation", {"intents": local_trace}))
    write_json_report(run_dir, "mismatch_classification_report.json", wrap_artifact("lifecycle_reconciliation", {"mismatch_counts": reconciliation["mismatch_counts"]}))
    logger.info("lifecycle_reconciliation mismatches=%s run_dir=%s", len(reconciliation["mismatches"]), run_dir)
    return (0 if reconciliation["critical_mismatch_count"] <= settings.lifecycle.max_critical_mismatches else 2), run_dir


def lifecycle_audit_report(settings: Settings) -> tuple[int, Path]:
    candidates = sorted(settings.data.runs_dir.glob("*_lifecycle_reconciliation"))
    if not candidates:
        raise FileNotFoundError("No lifecycle reconciliation runs available")
    source = candidates[-1]
    out_dir = build_run_directory(settings.data.runs_dir, "lifecycle_audit_report")
    logger = configure_logging(out_dir, settings.logging.level, settings.logging.format)
    payload = read_artifact_payload(source / "lifecycle_reconciliation_report.json", expected_type="lifecycle_reconciliation")
    write_json_report(out_dir, "lifecycle_reconciliation_report.json", wrap_artifact("lifecycle_reconciliation", payload))
    logger.info("lifecycle_audit_report source=%s run_dir=%s", source, out_dir)
    return 0, out_dir
