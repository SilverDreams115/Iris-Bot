from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from iris_bot.config import Settings
from iris_bot.logging_utils import build_run_directory, configure_logging
from iris_bot.mt5 import MT5Client
from iris_bot.operational import (
    AlertRecord,
    PaperRunArtifacts,
    write_operational_artifacts,
)
from iris_bot.paper import load_paper_context, run_paper_engine, PaperSessionConfig
from iris_bot.resilient_models import (
    BrokerPositionSnapshot,
    BrokerStateSnapshot,
    ReconciliationDiscrepancy,
    ReconciliationOutcome,
    ReconnectReport,
    RestoreReport,
    build_runtime_state_path,
    emit_alert,
    now_iso,
)
from iris_bot.resilient_reconcile import (
    broker_snapshot_from_mt5,
    build_demo_execution_validator,
    build_live_execution_validator,
    build_operational_status,
    classify_broker_event,
    reconcile_state,
    reconnect_mt5,
)
from iris_bot.resilient_state import (
    build_processing_event_id,
    fresh_state,
    is_session_allowed,
    persist_runtime_state,
    prevent_duplicate_processing,
    restore_runtime_state,
)

__all__ = [
    # Public API re-exported from sub-modules for backward compatibility
    "BrokerPositionSnapshot",
    "BrokerStateSnapshot",
    "ReconciliationDiscrepancy",
    "ReconciliationOutcome",
    "ReconnectReport",
    "RestoreReport",
    "build_runtime_state_path",
    "emit_alert",
    "now_iso",
    "broker_snapshot_from_mt5",
    "build_demo_execution_validator",
    "build_live_execution_validator",
    "build_operational_status",
    "classify_broker_event",
    "reconcile_state",
    "reconnect_mt5",
    "build_processing_event_id",
    "fresh_state",
    "is_session_allowed",
    "persist_runtime_state",
    "prevent_duplicate_processing",
    "restore_runtime_state",
    "run_resilient_session",
    "run_restore_state_check",
    "run_reconcile_state",
    "run_operational_status",
]


def run_resilient_session(
    settings: Settings,
    mode: str,
    require_broker: bool,
    client_factory: Callable[[], MT5Client] | None = None,
    allowed_profile_states: set[str] | None = None,
) -> tuple[int, Path]:
    command_name = "paper_resilient" if mode == "paper" else "demo_dry_resilient"
    run_dir = build_run_directory(settings.data.runs_dir, command_name)
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    alerts: list[AlertRecord] = []
    runtime_state_path = build_runtime_state_path(settings)
    restored_state, restore_report = restore_runtime_state(runtime_state_path, settings.recovery.require_state_restore_clean)
    if not restore_report.ok:
        emit_alert(alerts, "critical", "persistence_restore_failed", "State restore failed", restore_report.to_dict())
        if settings.recovery.require_state_restore_clean:
            empty_state = fresh_state(settings.backtest.starting_balance_usd, mode)
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
            empty_state = restored_state or fresh_state(settings.backtest.starting_balance_usd, mode)
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
    base_state = restored_state or fresh_state(settings.backtest.starting_balance_usd, mode)
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
    resilient_config = PaperSessionConfig(
        mode=mode,
        threshold=reference.threshold,
        trading_symbols=settings.trading.symbols,
        one_position_per_symbol=settings.trading.one_position_per_symbol,
        allow_long=settings.trading.allow_long and settings.backtest.allow_long,
        allow_short=settings.trading.allow_short and settings.backtest.allow_short,
        backtest=settings.backtest,
        risk=settings.risk,
        execution_validator=validator,
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
    artifacts = run_paper_engine(resilient_config, session_filtered_rows, session_filtered_probabilities, initial_state=base_state)
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
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    _, report = restore_runtime_state(build_runtime_state_path(settings), settings.recovery.require_state_restore_clean)
    artifacts = PaperRunArtifacts(
        state=fresh_state(settings.backtest.starting_balance_usd, "restore-check"),
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
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    restored_state, restore_report = restore_runtime_state(build_runtime_state_path(settings), settings.recovery.require_state_restore_clean)
    base_state = restored_state or fresh_state(settings.backtest.starting_balance_usd, "reconcile")
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
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    restored_state, restore_report = restore_runtime_state(build_runtime_state_path(settings), settings.recovery.require_state_restore_clean)
    state = restored_state or fresh_state(settings.backtest.starting_balance_usd, "operational-status")
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
