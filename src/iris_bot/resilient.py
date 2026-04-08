from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

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
    validate_restored_state_invariants,
)

def _blocked_run_artifacts(
    state: Any,
    restore_report: Any,
    alerts: list[AlertRecord],
    *,
    closed_trades: list | None = None,
    run_report: dict | None = None,
    validation_issues: list[str],
    operational_status: dict,
    reconciliation_report: dict | None = None,
) -> PaperRunArtifacts:
    """Factory for early-exit blocked-run artifacts (restore/reconnect/reconcile failures)."""
    return PaperRunArtifacts(
        state=state,
        events=[],
        closed_trades=closed_trades if closed_trades is not None else [],
        equity_curve_rows=[],
        daily_summary={},
        run_report=run_report or {},
        validation_report={"ok": False, "issues": validation_issues},
        signal_rows=[],
        execution_rows=[],
        restore_state_report=restore_report.to_dict(),
        reconciliation_report=reconciliation_report or {},
        operational_status=operational_status,
        alerts=alerts,
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
    "validate_restored_state_invariants",
    "run_resilient_session",
    "run_restore_state_check",
    "run_restore_safety_drill",
    "run_reconcile_state",
    "run_reconciliation_drills",
    "run_recovery_drills",
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
            artifacts = _blocked_run_artifacts(
                empty_state, restore_report, alerts,
                validation_issues=restore_report.issues,
                operational_status={"status": "blocked"},
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
            artifacts = _blocked_run_artifacts(
                empty_state, restore_report, alerts,
                run_report={"reconnect": reconnect_report.to_dict()},
                validation_issues=["reconnect_failed"],
                operational_status={"status": "blocked", "reconnect": reconnect_report.to_dict()},
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
            artifacts = _blocked_run_artifacts(
                base_state, restore_report, alerts,
                closed_trades=base_state.closed_positions,
                run_report={"reconnect": reconnect_report.to_dict()},
                validation_issues=["critical_reconciliation_mismatch"],
                operational_status=build_operational_status(base_state, reconciliation, restore_report, alerts),
                reconciliation_report=reconciliation.to_dict(),
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
        equity_curve_rows=[],
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
            equity_curve_rows=[],
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
        equity_curve_rows=[],
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
        equity_curve_rows=[],
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


# ---------------------------------------------------------------------------
# BLOQUE 1 — Restore/Restart Safety Drill
# ---------------------------------------------------------------------------

def run_restore_safety_drill(settings: Settings) -> tuple[int, dict[str, Any]]:
    """Drill that explicitly validates all restore/restart invariants.

    Runs entirely in-memory (no broker required). Generates a structured
    auditable report. Returns (exit_code, drill_report).
    exit_code: 0=all_pass, 1=some_fail
    """
    from iris_bot.operational import DailyLossTracker, PaperPosition, PendingIntent, BrokerSyncStatus, SessionStatus

    runtime_state_path = build_runtime_state_path(settings)
    checks: dict[str, dict[str, Any]] = {}

    # ── Build a state with all significant fields populated ──────────────────
    state = fresh_state(settings.backtest.starting_balance_usd, "restore-drill")
    state.open_positions["EURUSD"] = PaperPosition(
        symbol="EURUSD", timeframe="M15", direction=1,
        entry_timestamp="2026-01-01T00:15:00", signal_timestamp="2026-01-01T00:00:00",
        entry_index=1, volume_lots=0.1, entry_price=1.1000,
        stop_loss_price=1.0980, take_profit_price=1.1040,
        commission_entry_usd=1.0, bars_held=2,
        probability_long=0.8, probability_short=0.05,
        stop_policy="static", target_policy="static",
    )
    state.pending_intents.append(PendingIntent(
        symbol="GBPUSD", created_at="2026-01-01T00:00:00",
        signal_timestamp="2026-01-01T00:00:00", side="sell", volume_lots=0.05,
    ))
    state.daily_loss_tracker = DailyLossTracker("2026-01-01", -20.0, 50.0, False)
    state.processing_state.processed_event_ids.extend(["evt:001", "evt:002"])
    state.processing_state.last_processed_timestamp_by_symbol["EURUSD"] = "2026-01-01T00:15:00"
    state.current_session_status = SessionStatus("sess-drill-001", "restore-drill", "running", "2026-01-01T00:15:00")

    # ── Persist ──────────────────────────────────────────────────────────────
    try:
        persist_runtime_state(runtime_state_path, state, {})
        checks["persist"] = {"ok": True, "reason": "state_persisted"}
    except Exception as exc:  # noqa: BLE001
        checks["persist"] = {"ok": False, "reason": f"persist_failed:{exc}"}

    # ── Restore ──────────────────────────────────────────────────────────────
    restored, report = restore_runtime_state(runtime_state_path, require_clean=True)
    checks["restore"] = {"ok": report.ok, "action": report.action, "issues": report.issues, "restored": report.restored}
    if restored is None:
        checks["all_invariants"] = {"ok": False, "reason": "restore_returned_none"}
        return 1, _build_drill_report("restore_safety", checks)

    # ── Invariant 1: Open positions preserved ────────────────────────────────
    pos_ok = "EURUSD" in restored.open_positions
    pos_data = restored.open_positions.get("EURUSD")
    checks["open_positions_preserved"] = {
        "ok": pos_ok and pos_data is not None and pos_data.volume_lots == 0.1,
        "expected_symbol": "EURUSD",
        "found": pos_ok,
        "volume_match": pos_data.volume_lots == 0.1 if pos_data else False,
    }

    # ── Invariant 2: Pending intents preserved ───────────────────────────────
    pi_ok = len(restored.pending_intents) == 1 and restored.pending_intents[0].symbol == "GBPUSD"
    checks["pending_intents_preserved"] = {
        "ok": pi_ok,
        "expected_count": 1,
        "found_count": len(restored.pending_intents),
        "symbol_match": restored.pending_intents[0].symbol == "GBPUSD" if restored.pending_intents else False,
    }

    # ── Invariant 3: Processing state preserved (no duplicate decisions) ─────
    evt_ids = restored.processing_state.processed_event_ids
    proc_ok = set(evt_ids) == {"evt:001", "evt:002"} and len(evt_ids) == len(set(evt_ids))
    checks["processing_state_preserved"] = {
        "ok": proc_ok,
        "event_ids": evt_ids,
        "no_duplicates": len(evt_ids) == len(set(evt_ids)),
    }

    # ── Invariant 4: Daily loss tracker preserved ────────────────────────────
    dl = restored.daily_loss_tracker
    dl_ok = dl.current_day == "2026-01-01" and dl.realized_pnl_usd == -20.0 and not dl.blocked
    checks["daily_loss_preserved"] = {"ok": dl_ok, "current_day": dl.current_day, "pnl": dl.realized_pnl_usd}

    # ── Invariant 5: New events can be processed without re-processing old ───
    allow_new = prevent_duplicate_processing(restored, "EURUSD", "2026-01-01T00:30:00")
    block_old = not prevent_duplicate_processing(restored, "EURUSD", "2026-01-01T00:15:00")
    checks["idempotency_after_restore"] = {
        "ok": allow_new and block_old,
        "new_event_allowed": allow_new,
        "old_event_blocked": block_old,
    }

    # ── Invariant 6: Session lineage preserved ───────────────────────────────
    sess = restored.current_session_status
    sess_ok = sess.session_id == "sess-drill-001" and sess.mode == "restore-drill"
    checks["session_lineage_preserved"] = {
        "ok": sess_ok,
        "session_id": sess.session_id,
        "mode": sess.mode,
    }

    # ── Invariant 7: Structural invariant validation passes ──────────────────
    struct_issues = validate_restored_state_invariants(restored)
    checks["structural_invariants_clean"] = {
        "ok": len(struct_issues) == 0,
        "issues": struct_issues,
    }

    all_pass = all(v.get("ok", False) for v in checks.values())
    report_payload = _build_drill_report("restore_safety", checks)
    return (0 if all_pass else 1), report_payload


def _build_drill_report(drill_name: str, checks: dict[str, Any]) -> dict[str, Any]:
    failed = [k for k, v in checks.items() if not v.get("ok", False)]
    return {
        "drill": drill_name,
        "ok": len(failed) == 0,
        "failed_checks": failed,
        "checks": checks,
        "generated_at": now_iso(),
    }


# ---------------------------------------------------------------------------
# BLOQUE 2 — Reconciliation Drills
# ---------------------------------------------------------------------------

def run_reconciliation_drills(settings: Settings) -> tuple[int, dict[str, Any]]:
    """Six reconciliation scenarios covering all required cases.

    All in-memory. Returns (exit_code, drill_report).
    exit_code: 0=all_expected_outcomes_correct, 1=unexpected_outcome
    """
    from iris_bot.config import ReconciliationConfig
    from iris_bot.operational import AccountState, PaperEngineState, PaperPosition, DailyLossTracker, BrokerSyncStatus, ProcessingState, SessionStatus

    cfg_hard = ReconciliationConfig(policy="hard_fail", price_tolerance=0.0001, volume_tolerance=0.000001)

    def _base_state() -> PaperEngineState:
        s = PaperEngineState(account_state=AccountState(1000.0, 1000.0, 1000.0))
        s.open_positions["EURUSD"] = PaperPosition(
            symbol="EURUSD", timeframe="M15", direction=1,
            entry_timestamp="2026-01-01T00:15:00", signal_timestamp="2026-01-01T00:00:00",
            entry_index=1, volume_lots=0.1, entry_price=1.1000,
            stop_loss_price=1.0980, take_profit_price=1.1040,
            commission_entry_usd=1.0, bars_held=1,
            probability_long=0.8, probability_short=0.05,
            stop_policy="static", target_policy="static",
        )
        return s

    def _broker_pos(symbol: str, side: str = "buy", volume: float = 0.1, price: float = 1.1000) -> BrokerPositionSnapshot:
        return BrokerPositionSnapshot(ticket="t1", symbol=symbol, side=side, volume_lots=volume, price_open=price, stop_loss=1.0980, take_profit=1.1040, time="2026-01-01T00:15:00")

    scenarios: dict[str, dict[str, Any]] = {}

    # ── Scenario 1: Local open, broker empty → missing_in_broker (critical) ─
    s1_state = _base_state()
    s1_broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [], [], [], {})
    s1_out = reconcile_state(s1_state, s1_broker, cfg_hard)
    scenarios["local_open_broker_empty"] = {
        "ok": not s1_out.ok and any(d.category == "missing_in_broker" for d in s1_out.discrepancies),
        "expected_ok": False,
        "expected_category": "missing_in_broker",
        "actual_ok": s1_out.ok,
        "actual_action": s1_out.action,
        "discrepancies": [asdict(d) for d in s1_out.discrepancies],
    }

    # ── Scenario 2: Broker open, local empty → missing_in_local_state ────────
    s2_state = PaperEngineState(account_state=AccountState(1000.0, 1000.0, 1000.0))
    s2_broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [_broker_pos("EURUSD")], [], [], {})
    s2_out = reconcile_state(s2_state, s2_broker, cfg_hard)
    scenarios["broker_open_local_empty"] = {
        "ok": not s2_out.ok and any(d.category == "missing_in_local_state" for d in s2_out.discrepancies),
        "expected_ok": False,
        "expected_category": "missing_in_local_state",
        "actual_ok": s2_out.ok,
        "actual_action": s2_out.action,
        "discrepancies": [asdict(d) for d in s2_out.discrepancies],
    }

    # ── Scenario 3: Volume divergence → volume_mismatch (critical) ───────────
    s3_state = _base_state()
    s3_broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [_broker_pos("EURUSD", volume=0.5)], [], [], {})
    s3_out = reconcile_state(s3_state, s3_broker, cfg_hard)
    scenarios["volume_divergence"] = {
        "ok": not s3_out.ok and any(d.category == "volume_mismatch" for d in s3_out.discrepancies),
        "expected_ok": False,
        "expected_category": "volume_mismatch",
        "actual_ok": s3_out.ok,
        "actual_action": s3_out.action,
        "discrepancies": [asdict(d) for d in s3_out.discrepancies],
    }

    # ── Scenario 4: Price within tolerance → no critical discrepancy ─────────
    s4_state = _base_state()
    s4_broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [_broker_pos("EURUSD", price=1.10005)], [], [], {})
    s4_out = reconcile_state(s4_state, s4_broker, cfg_hard)
    scenarios["price_within_tolerance"] = {
        "ok": s4_out.ok and not any(d.severity == "critical" and d.category == "price_mismatch" for d in s4_out.discrepancies),
        "expected_ok": True,
        "expected_no_critical_price_mismatch": True,
        "actual_ok": s4_out.ok,
        "actual_action": s4_out.action,
        "discrepancies": [asdict(d) for d in s4_out.discrepancies],
    }

    # ── Scenario 5: Price beyond tolerance → price_mismatch (warning) ────────
    s5_state = _base_state()
    s5_broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [_broker_pos("EURUSD", price=1.1050)], [], [], {})
    s5_out = reconcile_state(s5_state, s5_broker, cfg_hard)
    scenarios["price_beyond_tolerance"] = {
        "ok": any(d.category == "price_mismatch" for d in s5_out.discrepancies),
        "expected_price_mismatch_present": True,
        "actual_ok": s5_out.ok,
        "actual_action": s5_out.action,
        "discrepancies": [asdict(d) for d in s5_out.discrepancies],
    }

    # ── Scenario 6: Stale local state (broker closed trades older than local) ─
    s6_state = _base_state()
    s6_state.current_session_status = SessionStatus("s6", "test", "running", "2026-06-01T12:00:00")
    s6_broker = BrokerStateSnapshot(True, 1000.0, 1000.0, [_broker_pos("EURUSD")], [{"time": "2026-01-01T00:01:00"}], [], {})
    s6_out = reconcile_state(s6_state, s6_broker, cfg_hard)
    scenarios["stale_state_detection"] = {
        "ok": any(d.category == "stale_state" for d in s6_out.discrepancies),
        "expected_stale_state_warning": True,
        "actual_ok": s6_out.ok,
        "discrepancies": [asdict(d) for d in s6_out.discrepancies],
    }

    all_pass = all(v.get("ok", False) for v in scenarios.values())
    failed = [k for k, v in scenarios.items() if not v.get("ok", False)]
    report = {
        "drill": "reconciliation_drills",
        "ok": all_pass,
        "scenarios_total": len(scenarios),
        "scenarios_passed": len(scenarios) - len(failed),
        "failed_scenarios": failed,
        "scenarios": scenarios,
        "generated_at": now_iso(),
    }
    return (0 if all_pass else 1), report


# ---------------------------------------------------------------------------
# BLOQUE 3 — Recovery After Disconnect Drills
# ---------------------------------------------------------------------------

def run_recovery_drills(settings: Settings, base_client_factory: Callable[[], MT5Client] | None = None) -> tuple[int, dict[str, Any]]:
    """Four disconnect/reconnect recovery scenarios.

    Uses FakeClient-style factories. No real broker needed.
    Returns (exit_code, drill_report).
    """
    from iris_bot.config import RecoveryConfig, ReconciliationConfig
    from iris_bot.mt5 import BrokerSnapshot

    scenarios: dict[str, dict[str, Any]] = {}

    # ── Scenario 1: Clean disconnect + reconnect (no state degradation) ──────
    class _CleanClient(MT5Client):
        def __init__(self) -> None:
            super().__init__(settings.mt5)
            self._calls = 0
        def connect(self) -> bool:
            self._calls += 1
            self._connected = self._calls >= 2
            return self._connected
        def last_error(self) -> object:
            return (1, "Success") if self._connected else (500, "Initial disconnect")
        def broker_state_snapshot(self, symbols: tuple[str, ...]) -> BrokerSnapshot:
            return BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [], [], [])

    recovery_cfg = RecoveryConfig(reconnect_retries=3, reconnect_backoff_seconds=0.0, require_state_restore_clean=True)
    r1 = reconnect_mt5(_CleanClient(), recovery_cfg)
    scenarios["clean_disconnect_reconnect"] = {
        "ok": r1.ok and r1.final_state == "connected",
        "expected_final_state": "connected",
        "actual_final_state": r1.final_state,
        "attempts": len(r1.attempts),
        "first_attempt_ok": r1.attempts[0]["ok"] if r1.attempts else None,
    }

    # ── Scenario 2: Reconnect with broker state changed (new position appeared)
    class _ChangedBrokerClient(MT5Client):
        def __init__(self) -> None:
            super().__init__(settings.mt5)
        def connect(self) -> bool:
            self._connected = True
            return True
        def last_error(self) -> object:
            return (1, "Success")
        def broker_state_snapshot(self, symbols: tuple[str, ...]) -> BrokerSnapshot:
            # Returns a position that wasn't in local state
            new_pos = {"ticket": 42, "symbol": "USDJPY", "type": 0, "volume": 0.1, "price_open": 145.0, "sl": 144.0, "tp": 146.0, "time": 1735689600}
            return BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [new_pos], [], [])

    r2_client = _ChangedBrokerClient()
    r2_reconnect = reconnect_mt5(r2_client, recovery_cfg)
    r2_snapshot = broker_snapshot_from_mt5(r2_client.broker_state_snapshot(("USDJPY",)))
    # Local state has no positions; broker has USDJPY → should detect missing_in_local_state
    r2_local = fresh_state(settings.backtest.starting_balance_usd, "recovery-drill")
    r2_reconcile = reconcile_state(r2_local, r2_snapshot, ReconciliationConfig(policy="hard_fail", price_tolerance=0.001, volume_tolerance=0.001))
    scenarios["reconnect_broker_state_changed"] = {
        "ok": r2_reconnect.ok and any(d.category == "missing_in_local_state" for d in r2_reconcile.discrepancies),
        "reconnect_ok": r2_reconnect.ok,
        "discrepancy_detected": any(d.category == "missing_in_local_state" for d in r2_reconcile.discrepancies),
        "discrepancies": [asdict(d) for d in r2_reconcile.discrepancies],
    }

    # ── Scenario 3: Reconnect with prior artifacts (state restore + reconnect)
    runtime_path = build_runtime_state_path(settings)
    prior_state = fresh_state(settings.backtest.starting_balance_usd, "recovery-drill")
    prior_state.processing_state.processed_event_ids.append("prior-evt-001")
    persist_runtime_state(runtime_path, prior_state, {})
    restored3, restore3_report = restore_runtime_state(runtime_path, require_clean=True)

    class _SimpleClient(MT5Client):
        def __init__(self) -> None:
            super().__init__(settings.mt5)
        def connect(self) -> bool:
            self._connected = True
            return True
        def last_error(self) -> object:
            return (1, "Success")
        def broker_state_snapshot(self, symbols: tuple[str, ...]) -> BrokerSnapshot:
            return BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [], [], [])

    r3_reconnect = reconnect_mt5(_SimpleClient(), recovery_cfg)
    prior_events_preserved = restored3 is not None and "prior-evt-001" in restored3.processing_state.processed_event_ids
    scenarios["reconnect_with_prior_artifacts"] = {
        "ok": restore3_report.ok and r3_reconnect.ok and prior_events_preserved,
        "restore_ok": restore3_report.ok,
        "reconnect_ok": r3_reconnect.ok,
        "prior_events_preserved": prior_events_preserved,
    }

    # ── Scenario 4: Reconnect + reconcile chain (full sequence) ──────────────
    r4_local = restored3 or fresh_state(settings.backtest.starting_balance_usd, "recovery-drill")
    r4_snapshot = broker_snapshot_from_mt5(_SimpleClient().broker_state_snapshot(()))
    r4_reconcile = reconcile_state(r4_local, r4_snapshot, ReconciliationConfig(policy="log_only", price_tolerance=0.001, volume_tolerance=0.001))
    r4_status = build_operational_status(r4_local, r4_reconcile, restore3_report, [])
    scenarios["reconnect_reconcile_chain"] = {
        "ok": r4_reconcile.ok and r4_status["restore_ok"] and r4_status["reconciliation_ok"],
        "reconcile_ok": r4_reconcile.ok,
        "restore_ok": r4_status["restore_ok"],
        "reconnect_ok": r3_reconnect.ok,
        "operational_status": r4_status,
    }

    all_pass = all(v.get("ok", False) for v in scenarios.values())
    failed = [k for k, v in scenarios.items() if not v.get("ok", False)]
    report = {
        "drill": "recovery_drills",
        "ok": all_pass,
        "scenarios_total": len(scenarios),
        "scenarios_passed": len(scenarios) - len(failed),
        "failed_scenarios": failed,
        "scenarios": scenarios,
        "generated_at": now_iso(),
    }
    return (0 if all_pass else 1), report
