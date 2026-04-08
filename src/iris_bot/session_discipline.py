"""Session discipline and runbook for demo-guarded operation.

Provides:
- Structured runbook (machine-readable + human-readable)
- SessionStartupCheck: validates pre-run conditions before session start
- abort/hold/continue criteria as explicit data

Design: the runbook is not documentation—it is executable discipline.
Each step has a check function that can be evaluated programmatically.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_bot.config import Settings
from iris_bot.kill_switch import is_kill_switch_active, is_no_trade_mode_active
from iris_bot.operational import PaperEngineState
from iris_bot.resilient import (
    build_runtime_state_path,
    restore_runtime_state,
    validate_restored_state_invariants,
)


__all__ = [
    "RunbookStep",
    "DemoSessionRunbook",
    "SessionStartupReport",
    "SessionDecision",
    "DemoSessionReview",
    "DemoSeriesReview",
    "session_startup_check",
    "generate_session_runbook",
    "evaluate_session_decision",
    "review_demo_session",
    "review_demo_series",
    "write_session_discipline_report",
]


@dataclass(frozen=True)
class RunbookStep:
    phase: str          # "startup" | "pre_run" | "restart" | "mismatch" | "disconnect" | "close" | "post_run"
    step_id: str
    title: str
    action: str
    criteria: str       # what makes this step pass
    on_failure: str     # "abort" | "hold" | "continue"


@dataclass(frozen=True)
class DemoSessionRunbook:
    """Machine-readable runbook for demo-guarded session operation."""

    version: str
    mode: str
    steps: list[RunbookStep]
    abort_criteria: list[str]
    hold_criteria: list[str]
    continue_criteria: list[str]
    post_run_checklist: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "steps": [asdict(s) for s in self.steps],
            "abort_criteria": self.abort_criteria,
            "hold_criteria": self.hold_criteria,
            "continue_criteria": self.continue_criteria,
            "post_run_checklist": self.post_run_checklist,
        }


@dataclass(frozen=True)
class SessionStartupReport:
    """Result of session startup check."""

    ok: bool
    decision: str       # "proceed" | "abort" | "hold"
    passed_checks: list[str]
    failed_checks: list[str]
    warnings: list[str]
    details: dict[str, Any]
    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionDecision:
    """abort / hold / continue decision with explicit reason."""

    action: str         # "abort" | "hold" | "continue"
    reason: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DemoSessionReview:
    """Explicit post-run evaluation for a single demo session."""

    session_id: str
    session_series_id: str
    classification: str
    operationally_healthy: bool
    had_abort: bool
    had_reconcile_mismatch: bool
    had_recovery: bool
    had_restore: bool
    had_kill_switch_or_breaker: bool
    evidence_complete: bool
    valid_for_forward_validation: bool
    blockers: list[str]
    warnings: list[str]
    recommendation: str
    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DemoSeriesReview:
    """Explicit post-run evaluation for a demo forward-validation series."""

    session_series_id: str
    classification: str
    sessions_total: int
    successful_sessions: int
    valid_forward_sessions: int
    evidence_complete_sessions: int
    repeated_critical_failures: bool
    repeated_degrading_recovery: bool
    blockers: list[str]
    warnings: list[str]
    recommendation: str
    valid_forward_series: bool
    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_session_runbook(mode: str = "demo_guarded") -> DemoSessionRunbook:
    """Generate the standard demo-guarded session runbook."""
    steps = [
        # ── Startup ──────────────────────────────────────────────────────────
        RunbookStep("startup", "S01", "Verify official suite is green",
                    "Run make check / ruff / mypy / pytest",
                    "All quality commands return exit code 0",
                    "abort"),
        RunbookStep("startup", "S02", "Verify demo_execution_readiness gate",
                    "Run demo-execution-readiness command",
                    "decision == ready_for_demo_guarded",
                    "abort"),
        RunbookStep("startup", "S03", "Restore runtime state",
                    "Call restore_runtime_state on runtime_state.json",
                    "report.ok == True or state_missing (fresh start)",
                    "abort"),
        RunbookStep("startup", "S04", "Validate restored state invariants",
                    "Call validate_restored_state_invariants on restored state",
                    "No structural violations found",
                    "abort"),
        RunbookStep("startup", "S05", "Verify kill switch not active",
                    "Check state.blocked_reasons for kill_switch: tags",
                    "No kill_switch: tag present",
                    "abort"),
        # ── Pre-run ───────────────────────────────────────────────────────────
        RunbookStep("pre_run", "P01", "Confirm mode is paper or demo_dry (never live)",
                    "Verify session mode != live",
                    "mode in {paper, demo_dry, demo_guarded}",
                    "abort"),
        RunbookStep("pre_run", "P02", "Verify no open orphan positions",
                    "Cross-check local open_positions with broker snapshot",
                    "reconcile_state.ok == True or discrepancies are warnings only",
                    "hold"),
        RunbookStep("pre_run", "P03", "Confirm max_daily_loss not triggered",
                    "Check daily_loss_tracker.blocked",
                    "daily_loss_tracker.blocked == False",
                    "hold"),
        # ── Restart handling ─────────────────────────────────────────────────
        RunbookStep("restart", "R01", "Restore state from runtime_state.json",
                    "Call restore_runtime_state",
                    "report.ok == True",
                    "abort"),
        RunbookStep("restart", "R02", "Validate state invariants post-restore",
                    "Call validate_restored_state_invariants",
                    "Empty issues list",
                    "abort"),
        RunbookStep("restart", "R03", "Check for duplicate event IDs",
                    "Verify processed_event_ids has no duplicates",
                    "len(ids) == len(set(ids))",
                    "abort"),
        RunbookStep("restart", "R04", "Reconnect broker if required",
                    "Call reconnect_mt5 with configured retries",
                    "reconnect_report.ok == True",
                    "hold"),
        # ── Mismatch handling ─────────────────────────────────────────────────
        RunbookStep("mismatch", "M01", "Detect reconciliation discrepancies",
                    "Call reconcile_state",
                    "All discrepancies classified by severity",
                    "continue"),
        RunbookStep("mismatch", "M02", "Critical discrepancy → halt and audit",
                    "Check for severity=critical discrepancies",
                    "No critical discrepancies, or action != blocked",
                    "abort"),
        RunbookStep("mismatch", "M03", "Warning discrepancies → log and monitor",
                    "Log warning discrepancies to alerts_log.jsonl",
                    "Alerts written with correct severity",
                    "continue"),
        # ── Disconnect handling ───────────────────────────────────────────────
        RunbookStep("disconnect", "D01", "Detect broker disconnection",
                    "Check MT5Client.ensure_connection()",
                    "Connection restored within reconnect_retries attempts",
                    "hold"),
        RunbookStep("disconnect", "D02", "Reconcile after reconnect",
                    "Fetch broker_state_snapshot and call reconcile_state",
                    "No new critical discrepancies introduced by disconnect",
                    "abort"),
        # ── Session close ─────────────────────────────────────────────────────
        RunbookStep("close", "C01", "Persist final runtime state",
                    "Call persist_runtime_state",
                    "State written atomically without errors",
                    "abort"),
        RunbookStep("close", "C02", "Write all operational artifacts",
                    "Call write_operational_artifacts",
                    "All required artifact files exist in run_dir",
                    "abort"),
        RunbookStep("close", "C03", "Shutdown broker client",
                    "Call MT5Client.shutdown()",
                    "No exceptions; connection cleanly closed",
                    "continue"),
        # ── Post-run review ───────────────────────────────────────────────────
        RunbookStep("post_run", "PR01", "Review alerts_log.jsonl",
                    "Read all alerts from run",
                    "No unreviewed critical alerts",
                    "hold"),
        RunbookStep("post_run", "PR02", "Review execution_journal.csv",
                    "Check all execution events are expected",
                    "No unexpected execution events",
                    "hold"),
        RunbookStep("post_run", "PR03", "Review reconciliation_report.json",
                    "Confirm reconciliation outcome",
                    "reconciliation_report.ok == True or discrepancies reviewed",
                    "hold"),
    ]

    abort_criteria = [
        "kill_switch is active",
        "state restore failed with require_clean=True",
        "critical reconciliation discrepancy with action=blocked or hard_fail",
        "official quality suite failed",
        "demo_execution_readiness gate returned not_ready_for_demo",
        "structural state invariants violated",
        "duplicate processed event IDs detected after restore",
    ]

    hold_criteria = [
        "broker reconnect is in progress (within retry window)",
        "daily_loss_tracker.blocked == True",
        "warning reconciliation discrepancies pending review",
        "no_trade_mode is active but kill_switch is not",
        "post-run critical alerts unreviewed",
    ]

    continue_criteria = [
        "restore_report.ok == True",
        "reconcile_state.ok == True",
        "reconnect_report.ok == True",
        "daily_loss_tracker.blocked == False",
        "no kill_switch: tags in blocked_reasons",
        "no warning or critical alerts in current cycle",
    ]

    post_run_checklist = [
        "Review alerts_log.jsonl for unexpected events",
        "Verify execution_journal.csv matches expected signal activity",
        "Confirm reconciliation_report.json shows ok=True or review discrepancies",
        "Verify restore_state_report.json shows ok=True",
        "Review operational_status.json for degraded states",
        "Archive run_dir for audit trail",
        "Update soak cycle evidence if applicable",
    ]

    return DemoSessionRunbook(
        version="1.0",
        mode=mode,
        steps=steps,
        abort_criteria=abort_criteria,
        hold_criteria=hold_criteria,
        continue_criteria=continue_criteria,
        post_run_checklist=post_run_checklist,
    )


def session_startup_check(settings: Settings) -> SessionStartupReport:
    """Evaluates pre-run conditions before starting a demo-guarded session.

    Does NOT connect to MT5. Does NOT send orders. Read-only.
    Returns a SessionStartupReport with decision: proceed | abort | hold.
    """
    passed: list[str] = []
    failed: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    # ── Check 1: Runtime state restore ───────────────────────────────────────
    runtime_path = build_runtime_state_path(settings)
    restored, restore_report = restore_runtime_state(runtime_path, require_clean=True)
    details["restore"] = restore_report.to_dict()
    if restore_report.ok:
        # ok=True covers both clean restore and state_missing (fresh start)
        passed.append("state_restore")
    else:
        # ok=False means corrupt / unreadable / invariant violation → abort
        failed.append("state_restore")

    # ── Check 2: Structural invariants ───────────────────────────────────────
    if restored is not None:
        invariant_issues = validate_restored_state_invariants(restored)
        details["invariant_issues"] = invariant_issues
        if invariant_issues:
            failed.append("state_structural_invariants")
        else:
            passed.append("state_structural_invariants")
    else:
        passed.append("state_structural_invariants")  # fresh state, no invariant to check
        details["invariant_issues"] = []

    # ── Check 3: Kill switch not active ──────────────────────────────────────
    if restored is not None and is_kill_switch_active(restored):
        failed.append("kill_switch_not_active")
        details["kill_switch_active"] = True
        details["blocked_reasons"] = list(restored.blocked_reasons)
    else:
        passed.append("kill_switch_not_active")
        details["kill_switch_active"] = False

    # ── Check 4: No-trade mode ────────────────────────────────────────────────
    if restored is not None and is_no_trade_mode_active(restored):
        warnings.append("no_trade_mode_active")
        details["no_trade_mode_active"] = True
    else:
        details["no_trade_mode_active"] = False

    # ── Check 5: Daily loss not blocked ──────────────────────────────────────
    if restored is not None and restored.daily_loss_tracker.blocked:
        warnings.append("daily_loss_blocked")
        details["daily_loss_blocked"] = True
    else:
        details["daily_loss_blocked"] = False

    # ── Determine decision ────────────────────────────────────────────────────
    if failed:
        decision = "abort"
    elif warnings:
        decision = "hold"
    else:
        decision = "proceed"

    return SessionStartupReport(
        ok=decision == "proceed",
        decision=decision,
        passed_checks=passed,
        failed_checks=failed,
        warnings=warnings,
        details=details,
        evaluated_at=datetime.now(tz=UTC).isoformat(),
    )


def evaluate_session_decision(
    state: PaperEngineState,
    reconcile_ok: bool,
    reconnect_ok: bool,
    critical_alerts: int,
) -> SessionDecision:
    """Evaluate abort/hold/continue for an in-progress session."""
    if is_kill_switch_active(state):
        return SessionDecision("abort", "kill_switch_active", {"blocked_reasons": list(state.blocked_reasons)})
    if not reconnect_ok:
        return SessionDecision("hold", "broker_reconnect_pending", {"reconnect_ok": False})
    if not reconcile_ok:
        return SessionDecision("abort", "critical_reconciliation_failure", {"reconcile_ok": False})
    if critical_alerts > 0:
        return SessionDecision("abort", "unresolved_critical_alerts", {"critical_alerts": critical_alerts})
    if is_no_trade_mode_active(state) or state.daily_loss_tracker.blocked:
        return SessionDecision("hold", "no_trade_mode_or_daily_loss", {
            "no_trade_mode": is_no_trade_mode_active(state),
            "daily_loss_blocked": state.daily_loss_tracker.blocked,
        })
    return SessionDecision("continue", "all_clear", {})


def review_demo_session(
    *,
    session_id: str,
    session_series_id: str,
    session_evidence: dict[str, Any],
    startup_report: SessionStartupReport | None = None,
) -> DemoSessionReview:
    """Evaluate whether a completed session counts toward forward validation."""
    blockers: list[str] = []
    warnings: list[str] = []
    divergence_summary = dict(session_evidence.get("divergence_summary", {}))
    recovery_summary = dict(session_evidence.get("restore_recovery_summary", {}))
    artifact_paths = dict(session_evidence.get("artifact_paths", {}))
    final_state = dict(session_evidence.get("final_state_summary", {}))

    preflight_ok = bool(session_evidence.get("preflight_ok", False))
    if not preflight_ok:
        blockers.append("preflight_failed")

    had_abort = startup_report is not None and startup_report.decision == "abort"
    if had_abort:
        blockers.append("startup_abort")

    had_reconcile_mismatch = int(divergence_summary.get("divergence_events", 0) or 0) > 0
    if had_reconcile_mismatch:
        warnings.append("broker_local_divergence_detected")

    had_recovery = int(recovery_summary.get("recovery_events", 0) or 0) > 0
    if had_recovery:
        warnings.append("recovery_applied")

    had_restore = int(recovery_summary.get("restore_events", 0) or 0) > 0
    had_kill_switch_or_breaker = bool(final_state.get("kill_switch_active", False)) or bool(final_state.get("circuit_breaker_triggered", False))
    if had_kill_switch_or_breaker:
        blockers.append("kill_switch_or_circuit_breaker_triggered")

    required_artifacts = {
        "preflight_report",
        "execution_report",
        "reconciliation_report",
        "session_evidence",
    }
    evidence_complete = required_artifacts.issubset(set(artifact_paths.keys()))
    if not evidence_complete:
        blockers.append("incomplete_evidence")

    operationally_healthy = preflight_ok and not had_abort and not had_kill_switch_or_breaker
    if had_reconcile_mismatch and not had_kill_switch_or_breaker:
        operationally_healthy = False

    if blockers:
        classification = "failed"
        recommendation = "abort"
    elif warnings:
        classification = "caution"
        recommendation = "hold"
    else:
        classification = "healthy"
        recommendation = "continue"

    valid_for_forward_validation = classification in {"healthy", "caution"} and evidence_complete and preflight_ok and not had_abort and not had_kill_switch_or_breaker

    return DemoSessionReview(
        session_id=session_id,
        session_series_id=session_series_id,
        classification=classification,
        operationally_healthy=operationally_healthy,
        had_abort=had_abort,
        had_reconcile_mismatch=had_reconcile_mismatch,
        had_recovery=had_recovery,
        had_restore=had_restore,
        had_kill_switch_or_breaker=had_kill_switch_or_breaker,
        evidence_complete=evidence_complete,
        valid_for_forward_validation=valid_for_forward_validation,
        blockers=blockers,
        warnings=warnings,
        recommendation=recommendation,
        evaluated_at=datetime.now(tz=UTC).isoformat(),
    )


def review_demo_series(
    *,
    session_series_id: str,
    session_reviews: list[dict[str, Any]],
    aggregate_counts: dict[str, int],
    target_sessions: int,
) -> DemoSeriesReview:
    """Evaluate whether a demo series is valid as prolonged forward evidence."""
    blockers: list[str] = []
    warnings: list[str] = []
    sessions_total = len(session_reviews)
    successful_sessions = int(aggregate_counts.get("successful_sessions", 0) or 0)
    valid_forward_sessions = sum(1 for review in session_reviews if bool(review.get("valid_for_forward_validation", False)))
    evidence_complete_sessions = sum(1 for review in session_reviews if bool(review.get("evidence_complete", False)))
    aborted_sessions = int(aggregate_counts.get("aborted_sessions", 0) or 0)
    reconcile_mismatch_sessions = int(aggregate_counts.get("sessions_with_divergence", 0) or 0)
    recovery_sessions = int(aggregate_counts.get("sessions_with_recovery", 0) or 0)
    kill_switch_events = int(aggregate_counts.get("kill_switch_events", 0) or 0)
    breaker_triggers = int(aggregate_counts.get("circuit_breaker_triggers", 0) or 0)

    repeated_critical_failures = aborted_sessions >= 2 or kill_switch_events >= 2 or breaker_triggers >= 2
    repeated_degrading_recovery = reconcile_mismatch_sessions >= 2 or recovery_sessions >= 2

    if sessions_total < max(target_sessions, 1):
        blockers.append("insufficient_sessions_in_series")
    if successful_sessions <= 0:
        blockers.append("no_successful_sessions")
    if evidence_complete_sessions != sessions_total:
        blockers.append("series_evidence_incomplete")
    if repeated_critical_failures:
        blockers.append("repeated_critical_failures")
    if repeated_degrading_recovery:
        warnings.append("repeated_reconcile_or_recovery_degradation")
    if valid_forward_sessions < max(1, min(target_sessions, sessions_total)):
        warnings.append("not_all_sessions_count_as_forward_valid")

    if blockers:
        classification = "failed"
        recommendation = "abort"
    elif warnings:
        classification = "caution"
        recommendation = "hold"
    else:
        classification = "healthy"
        recommendation = "continue"

    valid_forward_series = not blockers and successful_sessions >= max(1, target_sessions) and valid_forward_sessions >= max(1, target_sessions)

    return DemoSeriesReview(
        session_series_id=session_series_id,
        classification=classification,
        sessions_total=sessions_total,
        successful_sessions=successful_sessions,
        valid_forward_sessions=valid_forward_sessions,
        evidence_complete_sessions=evidence_complete_sessions,
        repeated_critical_failures=repeated_critical_failures,
        repeated_degrading_recovery=repeated_degrading_recovery,
        blockers=blockers,
        warnings=warnings,
        recommendation=recommendation,
        valid_forward_series=valid_forward_series,
        evaluated_at=datetime.now(tz=UTC).isoformat(),
    )


def write_session_discipline_report(path: Path, report: SessionStartupReport) -> None:
    """Write session startup report as JSON to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
