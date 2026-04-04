from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from iris_bot.config import RecoveryConfig, ReconciliationConfig, SessionConfig, Settings
from iris_bot.logging_utils import build_run_directory, configure_logging
from iris_bot.mt5 import MT5Client, OrderRequest
from iris_bot.operational import (
    AccountState,
    AlertRecord,
    PaperEngineState,
    PaperRunArtifacts,
    PendingIntent,
    ProcessingState,
    atomic_write_json,
    write_operational_artifacts,
)
from iris_bot.paper import ExecutionDecision, OrderIntent, load_paper_context, run_paper_engine


@dataclass(frozen=True)
class BrokerPositionSnapshot:
    ticket: str
    symbol: str
    side: str
    volume_lots: float
    price_open: float
    stop_loss: float
    take_profit: float
    time: str


@dataclass(frozen=True)
class BrokerStateSnapshot:
    connected: bool
    balance_usd: float | None
    equity_usd: float | None
    positions: list[BrokerPositionSnapshot]
    closed_trades: list[dict[str, Any]]
    pending_orders: list[dict[str, Any]]
    raw_account: dict[str, Any]
    scope_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "balance_usd": self.balance_usd,
            "equity_usd": self.equity_usd,
            "positions": [asdict(item) for item in self.positions],
            "closed_trades": self.closed_trades,
            "pending_orders": self.pending_orders,
            "raw_account": self.raw_account,
            "scope_report": self.scope_report,
        }


@dataclass(frozen=True)
class ReconciliationDiscrepancy:
    category: str
    severity: str
    message: str
    details: dict[str, Any]


@dataclass(frozen=True)
class ReconciliationOutcome:
    ok: bool
    action: str
    discrepancies: list[ReconciliationDiscrepancy]
    synced_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "discrepancies": [asdict(item) for item in self.discrepancies],
            "synced_state": self.synced_state,
        }


@dataclass(frozen=True)
class RestoreReport:
    ok: bool
    action: str
    issues: list[str]
    restored: bool
    state_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconnectReport:
    ok: bool
    final_state: str
    attempts: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrokerEventDecision:
    classification: str
    action: str
    retryable: bool
    block_operation: bool
    details: dict[str, Any]


def now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def build_runtime_state_path(settings: Settings) -> Path:
    return settings.data.runtime_dir / settings.operational.persistence_state_filename


def emit_alert(alerts: list[AlertRecord], severity: str, category: str, message: str, details: dict[str, Any]) -> None:
    alerts.append(
        AlertRecord(
            timestamp=now_iso(),
            severity=severity,
            category=category,
            message=message,
            details=details,
        )
    )


def _fresh_state(starting_balance: float, mode: str) -> PaperEngineState:
    state = PaperEngineState(
        account_state=AccountState(starting_balance, starting_balance, starting_balance),
    )
    state.current_session_status.mode = mode
    state.current_session_status.status = "idle"
    return state


def persist_runtime_state(
    path: Path,
    state: PaperEngineState,
    latest_broker_sync_result: dict[str, Any],
) -> None:
    atomic_write_json(
        path,
        {
            "saved_at": now_iso(),
            "state": state.to_dict(),
            "latest_broker_sync_result": latest_broker_sync_result,
        },
    )


def restore_runtime_state(path: Path, require_clean: bool) -> tuple[PaperEngineState | None, RestoreReport]:
    if not path.exists():
        return None, RestoreReport(True, "log_only", ["state_missing"], False, str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        action = "blocked" if require_clean else "log_only"
        return None, RestoreReport(False, action, [f"state_corrupt:{exc}"], False, str(path))
    state_payload = payload.get("state")
    if not isinstance(state_payload, dict):
        action = "blocked" if require_clean else "log_only"
        return None, RestoreReport(False, action, ["state_missing_payload"], False, str(path))
    try:
        state = _state_from_dict(state_payload)
    except Exception as exc:  # noqa: BLE001
        action = "blocked" if require_clean else "log_only"
        return None, RestoreReport(False, action, [f"state_restore_failed:{exc}"], False, str(path))
    return state, RestoreReport(True, "soft_resync", [], True, str(path))


def _state_from_dict(payload: dict[str, Any]) -> PaperEngineState:
    from iris_bot.operational import (
        AccountState,
        BrokerSyncStatus,
        ClosedPaperTrade,
        DailyLossTracker,
        ExposureState,
        PaperPosition,
        SessionStatus,
    )

    state = PaperEngineState(
        account_state=AccountState(**payload["account_state"]),
        open_positions={key: PaperPosition(**value) for key, value in payload.get("open_positions", {}).items()},
        closed_positions=[ClosedPaperTrade(**item) for item in payload.get("closed_positions", [])],
        daily_loss_tracker=DailyLossTracker(**payload.get("daily_loss_tracker", {})),
        cooldown_tracker=payload.get("cooldown_tracker", {}),
        exposure=ExposureState(**payload.get("exposure", {})),
        last_signal_per_symbol=payload.get("last_signal_per_symbol", {}),
        current_session_status=SessionStatus(**payload.get("current_session_status", {})),
        blocked_trades_summary=payload.get("blocked_trades_summary", {}),
        blocked_reasons=payload.get("blocked_reasons", []),
        pending_intents=[PendingIntent(**item) for item in payload.get("pending_intents", [])],
        broker_sync_status=BrokerSyncStatus(**payload.get("broker_sync_status", {})),
        processing_state=ProcessingState(**payload.get("processing_state", {})),
        latest_broker_snapshot=payload.get("latest_broker_snapshot", {}),
    )
    return state


def is_session_allowed(timestamp: datetime, session: SessionConfig) -> tuple[bool, str]:
    if not session.enabled:
        return True, "session_control_disabled"
    if timestamp.weekday() not in session.allowed_weekdays:
        return False, "market_session_blocked_weekday"
    if not (session.allowed_start_hour_utc <= timestamp.hour <= session.allowed_end_hour_utc):
        return False, "market_session_blocked_hour"
    return True, "session_allowed"


def build_processing_event_id(symbol: str, timestamp_text: str, source_event_id: str | None = None) -> tuple[str, str]:
    if source_event_id:
        return f"event:{source_event_id}", "event_id"
    return f"fallback:{symbol}:{timestamp_text}", "timestamp_fallback"


def prevent_duplicate_processing(
    state: PaperEngineState,
    symbol: str,
    timestamp_text: str,
    source_event_id: str | None = None,
) -> bool:
    event_id, mode = build_processing_event_id(symbol, timestamp_text, source_event_id)
    if event_id in state.processing_state.processed_event_ids:
        return False
    previous = state.processing_state.last_processed_timestamp_by_symbol.get(symbol)
    if mode == "timestamp_fallback" and previous is not None and timestamp_text <= previous:
        return False
    state.processing_state.processed_event_ids.append(event_id)
    state.processing_state.last_processed_timestamp_by_symbol[symbol] = timestamp_text
    state.processing_state.idempotency_mode_counts[mode] = state.processing_state.idempotency_mode_counts.get(mode, 0) + 1
    return True


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


def reconcile_state(
    local_state: PaperEngineState,
    broker_state: BrokerStateSnapshot,
    config: ReconciliationConfig,
    compare_account_state: bool = True,
) -> ReconciliationOutcome:
    discrepancies: list[ReconciliationDiscrepancy] = []
    synced = local_state.to_dict()
    if compare_account_state and broker_state.balance_usd is not None and abs(local_state.account_state.balance_usd - broker_state.balance_usd) > config.price_tolerance:
        discrepancies.append(
            ReconciliationDiscrepancy(
                "price_mismatch",
                "critical",
                "Balance mismatch between local state and broker",
                {"local_balance_usd": local_state.account_state.balance_usd, "broker_balance_usd": broker_state.balance_usd},
            )
        )
    broker_by_symbol = {item.symbol: item for item in broker_state.positions}
    local_symbols = set(local_state.open_positions.keys())
    broker_symbols = set(broker_by_symbol.keys())
    for symbol in sorted(local_symbols - broker_symbols):
        discrepancies.append(
            ReconciliationDiscrepancy(
                "missing_in_broker",
                "critical",
                "Open position exists locally but not in broker snapshot",
                {"symbol": symbol},
            )
        )
    for symbol in sorted(broker_symbols - local_symbols):
        discrepancies.append(
            ReconciliationDiscrepancy(
                "missing_in_local_state",
                "critical",
                "Broker reports position missing in local state",
                {"symbol": symbol},
            )
        )
    for symbol in sorted(local_symbols & broker_symbols):
        local_position = local_state.open_positions[symbol]
        broker_position = broker_by_symbol[symbol]
        local_side = "buy" if local_position.direction == 1 else "sell"
        if local_side != broker_position.side:
            discrepancies.append(
                ReconciliationDiscrepancy("side_mismatch", "critical", "Side mismatch", {"symbol": symbol, "local": local_side, "broker": broker_position.side})
            )
        if abs(local_position.volume_lots - broker_position.volume_lots) > config.volume_tolerance:
            discrepancies.append(
                ReconciliationDiscrepancy("volume_mismatch", "critical", "Volume mismatch", {"symbol": symbol, "local": local_position.volume_lots, "broker": broker_position.volume_lots})
            )
        if abs(local_position.entry_price - broker_position.price_open) > config.price_tolerance:
            discrepancies.append(
                ReconciliationDiscrepancy("price_mismatch", "warning", "Entry price mismatch", {"symbol": symbol, "local": local_position.entry_price, "broker": broker_position.price_open})
            )
    if len(local_state.processing_state.processed_event_ids) != len(set(local_state.processing_state.processed_event_ids)):
        discrepancies.append(
            ReconciliationDiscrepancy("duplicate_state", "critical", "Duplicate processed event ids detected", {})
        )
    if local_state.current_session_status.last_timestamp and broker_state.closed_trades:
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
        if broker_close_times:
            latest_broker_close = max(broker_close_times)
            local_last = datetime.fromisoformat(local_state.current_session_status.last_timestamp)
            if latest_broker_close < local_last:
                discrepancies.append(
                    ReconciliationDiscrepancy("stale_state", "warning", "Broker closed trade history appears older than local session", {"latest_broker_close": latest_broker_close.isoformat(), "local_last_timestamp": local_last.isoformat()})
                )
    critical = [item for item in discrepancies if item.severity == "critical"]
    action = "log_only"
    if critical:
        if config.policy == "soft_resync":
            action = "soft_resync"
            synced["latest_broker_snapshot"] = broker_state.to_dict()
            if broker_state.balance_usd is not None:
                synced["account_state"]["balance_usd"] = broker_state.balance_usd
                synced["account_state"]["cash_usd"] = broker_state.balance_usd
            if broker_state.equity_usd is not None:
                synced["account_state"]["equity_usd"] = broker_state.equity_usd
        elif config.policy == "log_only":
            action = "log_only"
        elif config.policy == "block":
            action = "blocked"
        else:
            action = "hard_fail"
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


def build_operational_status(state: PaperEngineState, reconciliation: ReconciliationOutcome, restore: RestoreReport, alerts: list[AlertRecord]) -> dict[str, Any]:
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


def run_resilient_session(
    settings: Settings,
    mode: str,
    require_broker: bool,
    client_factory: Callable[[], MT5Client] | None = None,
    allowed_profile_states: set[str] | None = None,
) -> tuple[int, Path]:
    command_name = "paper_resilient" if mode == "paper" else "demo_dry_resilient"
    run_dir = build_run_directory(settings.data.runs_dir, command_name)
    logger = configure_logging(run_dir, settings.logging.level)
    alerts: list[AlertRecord] = []
    runtime_state_path = build_runtime_state_path(settings)
    restored_state, restore_report = restore_runtime_state(runtime_state_path, settings.recovery.require_state_restore_clean)
    if not restore_report.ok:
        emit_alert(alerts, "critical", "persistence_restore_failed", "State restore failed", restore_report.to_dict())
        if settings.recovery.require_state_restore_clean:
            empty_state = _fresh_state(settings.backtest.starting_balance_usd, mode)
            artifacts = PaperRunArtifacts(
                state=empty_state,
                events=[],
                closed_trades=[],
                daily_summary={},
                run_report={},
                validation_report={"ok": False, "issues": restore_report.issues},
                signal_rows=[],
                execution_rows=[],
                restore_state_report=restore_report.to_dict(),
                reconciliation_report={},
                operational_status={"status": "blocked"},
                alerts=alerts,
            )
            write_operational_artifacts(run_dir, artifacts, {"mode": mode})
            return 2, run_dir
    client = client_factory() if client_factory is not None else MT5Client(settings.mt5)
    reconnect_report = ReconnectReport(True, "not_required", [])
    broker_snapshot = BrokerStateSnapshot(False, None, None, [], [], [], {}, {})
    validator = None
    if require_broker:
        reconnect_report = reconnect_mt5(client, settings.recovery)
        for attempt in reconnect_report.attempts:
            emit_alert(
                alerts,
                "warning" if attempt["ok"] else "error",
                "reconnect_attempt",
                "MT5 reconnect attempt",
                attempt,
            )
        if not reconnect_report.ok:
            emit_alert(alerts, "critical", "reconnect_fail", "Unable to reconnect MT5", reconnect_report.to_dict())
            empty_state = restored_state or _fresh_state(settings.backtest.starting_balance_usd, mode)
            artifacts = PaperRunArtifacts(
                state=empty_state,
                events=[],
                closed_trades=[],
                daily_summary={},
                run_report={},
                validation_report={"ok": False, "issues": ["reconnect_failed"]},
                signal_rows=[],
                execution_rows=[],
                restore_state_report=restore_report.to_dict(),
                reconciliation_report={},
                operational_status={"status": "blocked", "reconnect": reconnect_report.to_dict()},
                alerts=alerts,
            )
            write_operational_artifacts(run_dir, artifacts, {"mode": mode})
            return 2, run_dir
        broker_snapshot = broker_snapshot_from_mt5(client.broker_state_snapshot(settings.trading.symbols))
        validator = build_demo_execution_validator(client) if mode == "demo_dry" else None

    try:
        if allowed_profile_states is None:
            reference, rows, probabilities = load_paper_context(settings)
        else:
            try:
                reference, rows, probabilities = load_paper_context(settings, allowed_profile_states=allowed_profile_states)
            except TypeError:
                reference, rows, probabilities = load_paper_context(settings)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error(str(exc))
        return 1, run_dir
    session_filtered_rows = []
    session_filtered_probabilities = []
    for row, probability in zip(rows, probabilities, strict=False):
        allowed, reason = is_session_allowed(row.timestamp, settings.session)
        if allowed:
            session_filtered_rows.append(row)
            session_filtered_probabilities.append(probability)
        else:
            emit_alert(alerts, "warning", "market_session_blocked", "Bar skipped by session control", {"timestamp": row.timestamp.isoformat(), "symbol": row.symbol, "reason": reason})
    base_state = restored_state or _fresh_state(settings.backtest.starting_balance_usd, mode)
    reconciliation = (
        reconcile_state(
            base_state,
            broker_snapshot,
            settings.reconciliation,
            compare_account_state=mode != "demo_dry",
        )
        if require_broker
        else ReconciliationOutcome(True, "log_only", [], base_state.to_dict())
    )
    if not reconciliation.ok:
        emit_alert(alerts, "critical", "broker_mismatch", "Critical broker/local mismatch", reconciliation.to_dict())
        if reconciliation.action in {"hard_fail", "blocked"}:
            base_state.blocked_reasons.append("critical_reconciliation_mismatch")
            artifacts = PaperRunArtifacts(
                state=base_state,
                events=[],
                closed_trades=base_state.closed_positions,
                daily_summary={},
                run_report={"reconnect": reconnect_report.to_dict()},
                validation_report={"ok": False, "issues": ["critical_reconciliation_mismatch"]},
                signal_rows=[],
                execution_rows=[],
                restore_state_report=restore_report.to_dict(),
                reconciliation_report=reconciliation.to_dict(),
                operational_status=build_operational_status(base_state, reconciliation, restore_report, alerts),
                alerts=alerts,
            )
            write_operational_artifacts(run_dir, artifacts, {"mode": mode, "experiment_reference": str(reference.run_dir)})
            return 3, run_dir
    artifacts = run_paper_engine(
        rows=session_filtered_rows,
        probabilities=session_filtered_probabilities,
        threshold=reference.threshold,
        backtest=settings.backtest,
        risk=settings.risk,
        trading_symbols=settings.trading.symbols,
        one_position_per_symbol=settings.trading.one_position_per_symbol,
        allow_long=settings.trading.allow_long and settings.backtest.allow_long,
        allow_short=settings.trading.allow_short and settings.backtest.allow_short,
        mode=mode,
            execution_validator=validator,
            initial_state=base_state,
            should_process_row=lambda state, row: (
                prevent_duplicate_processing(
                    state,
                    row.symbol,
                    row.timestamp.isoformat(),
                    str(row.features.get("event_id")) if "event_id" in row.features else None,
                ),
                "duplicate_event_prevented",
            ),
        )
    artifacts.state.latest_broker_snapshot = broker_snapshot.to_dict()
    artifacts.state.broker_sync_status.state = "connected" if broker_snapshot.connected else "degraded"
    artifacts.state.broker_sync_status.last_sync_timestamp = now_iso()
    artifacts.state.broker_sync_status.reconciliation_policy = settings.reconciliation.policy
    artifacts.state.broker_sync_status.critical_discrepancy_count = sum(1 for item in reconciliation.discrepancies if item.severity == "critical")
    if artifacts.state.blocked_trades_summary.get("max_daily_loss", 0) > 0:
        emit_alert(alerts, "warning", "max_daily_loss_triggered", "Max daily loss blocked entries", {"blocked": artifacts.state.blocked_trades_summary["max_daily_loss"]})
    repeated = sum(artifacts.state.blocked_trades_summary.values())
    if repeated >= settings.operational.repeated_rejection_alert_threshold:
        emit_alert(alerts, "warning", "repeated_rejections", "Repeated rejections detected", {"blocked_summary": artifacts.state.blocked_trades_summary})
    processed_event_ids = artifacts.state.processing_state.processed_event_ids
    if len(processed_event_ids) != len(set(processed_event_ids)):
        emit_alert(alerts, "critical", "duplicate_event_prevented", "Duplicate processed event ids detected", {"processed_event_ids": processed_event_ids})
    persist_runtime_state(runtime_state_path, artifacts.state, broker_snapshot.to_dict())
    artifacts.restore_state_report = restore_report.to_dict()
    artifacts.reconciliation_report = reconciliation.to_dict()
    artifacts.reconciliation_scope_report = broker_snapshot.scope_report if hasattr(broker_snapshot, "scope_report") else {}
    artifacts.idempotency_report = {
        "processed_event_ids_count": len(artifacts.state.processing_state.processed_event_ids),
        "idempotency_mode_counts": dict(artifacts.state.processing_state.idempotency_mode_counts),
        "uses_real_event_ids": artifacts.state.processing_state.idempotency_mode_counts.get("event_id", 0) > 0,
        "fallback_active": artifacts.state.processing_state.idempotency_mode_counts.get("timestamp_fallback", 0) > 0,
    }
    artifacts.operational_status = build_operational_status(artifacts.state, reconciliation, restore_report, alerts)
    artifacts.alerts = alerts
    artifacts.run_report["reconnect"] = reconnect_report.to_dict()
    config_payload = {
        "mode": mode,
        "runtime_state_path": str(runtime_state_path),
        "reconciliation": asdict(settings.reconciliation),
        "recovery": asdict(settings.recovery),
        "session": asdict(settings.session),
        "experiment_reference": {"run_dir": str(reference.run_dir), "threshold": reference.threshold},
    }
    write_operational_artifacts(run_dir, artifacts, config_payload)
    logger.info("%s resilient complete run_dir=%s alerts=%s", mode, run_dir, len(alerts))
    if require_broker:
        client.shutdown()
    return 0, run_dir


def run_restore_state_check(settings: Settings) -> tuple[int, Path]:
    run_dir = build_run_directory(settings.data.runs_dir, "restore_state_check")
    logger = configure_logging(run_dir, settings.logging.level)
    _, report = restore_runtime_state(build_runtime_state_path(settings), settings.recovery.require_state_restore_clean)
    artifacts = PaperRunArtifacts(
        state=_fresh_state(settings.backtest.starting_balance_usd, "restore-check"),
        events=[],
        closed_trades=[],
        daily_summary={},
        run_report={"restore_state_report": report.to_dict()},
        validation_report={"ok": report.ok, "issues": report.issues},
        signal_rows=[],
        execution_rows=[],
        restore_state_report=report.to_dict(),
        reconciliation_report={},
        operational_status={"restore_ok": report.ok, "restore_action": report.action},
        alerts=[],
    )
    write_operational_artifacts(run_dir, artifacts, {"mode": "restore-check"})
    logger.info("restore_state_check ok=%s action=%s", report.ok, report.action)
    return (0 if report.ok else 2), run_dir


def run_reconcile_state(settings: Settings, client_factory: Callable[[], MT5Client] | None = None) -> tuple[int, Path]:
    run_dir = build_run_directory(settings.data.runs_dir, "reconcile_state")
    logger = configure_logging(run_dir, settings.logging.level)
    restored_state, restore_report = restore_runtime_state(build_runtime_state_path(settings), settings.recovery.require_state_restore_clean)
    base_state = restored_state or _fresh_state(settings.backtest.starting_balance_usd, "reconcile")
    client = client_factory() if client_factory is not None else MT5Client(settings.mt5)
    reconnect_report = reconnect_mt5(client, settings.recovery)
    if not reconnect_report.ok:
        artifacts = PaperRunArtifacts(
            state=base_state,
            events=[],
            closed_trades=base_state.closed_positions,
            daily_summary={},
            run_report={"reconnect": reconnect_report.to_dict()},
            validation_report={"ok": False, "issues": ["reconnect_failed"]},
            signal_rows=[],
            execution_rows=[],
            restore_state_report=restore_report.to_dict(),
            reconciliation_report={},
            operational_status={"status": "blocked"},
            alerts=[],
        )
        write_operational_artifacts(run_dir, artifacts, {"mode": "reconcile"})
        return 2, run_dir
    broker_snapshot = broker_snapshot_from_mt5(client.broker_state_snapshot(settings.trading.symbols))
    reconciliation = reconcile_state(base_state, broker_snapshot, settings.reconciliation, compare_account_state=False)
    artifacts = PaperRunArtifacts(
        state=base_state,
        events=[],
        closed_trades=base_state.closed_positions,
        daily_summary={},
        run_report={"reconnect": reconnect_report.to_dict()},
        validation_report={"ok": reconciliation.ok, "issues": [] if reconciliation.ok else ["reconciliation_failed"]},
        signal_rows=[],
        execution_rows=[],
        restore_state_report=restore_report.to_dict(),
        reconciliation_report=reconciliation.to_dict(),
        operational_status=build_operational_status(base_state, reconciliation, restore_report, []),
        alerts=[],
    )
    write_operational_artifacts(run_dir, artifacts, {"mode": "reconcile"})
    client.shutdown()
    logger.info("reconcile_state ok=%s action=%s discrepancies=%s", reconciliation.ok, reconciliation.action, len(reconciliation.discrepancies))
    return (0 if reconciliation.ok else 3), run_dir


def run_operational_status(settings: Settings) -> tuple[int, Path]:
    run_dir = build_run_directory(settings.data.runs_dir, "operational_status")
    logger = configure_logging(run_dir, settings.logging.level)
    restored_state, restore_report = restore_runtime_state(build_runtime_state_path(settings), settings.recovery.require_state_restore_clean)
    state = restored_state or _fresh_state(settings.backtest.starting_balance_usd, "operational-status")
    reconciliation = ReconciliationOutcome(True, "log_only", [], state.to_dict())
    status = build_operational_status(state, reconciliation, restore_report, [])
    artifacts = PaperRunArtifacts(
        state=state,
        events=[],
        closed_trades=state.closed_positions,
        daily_summary={},
        run_report={"operational_status": status},
        validation_report={"ok": restore_report.ok, "issues": restore_report.issues},
        signal_rows=[],
        execution_rows=[],
        restore_state_report=restore_report.to_dict(),
        reconciliation_report=reconciliation.to_dict(),
        operational_status=status,
        alerts=[],
    )
    write_operational_artifacts(run_dir, artifacts, {"mode": "operational-status"})
    logger.info("operational_status restore_ok=%s open_positions=%s", restore_report.ok, len(state.open_positions))
    return 0, run_dir
