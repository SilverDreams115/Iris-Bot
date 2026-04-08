"""Prolonged serious demo validation gate and runbook."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from iris_bot.artifacts import build_artifact_provenance, read_artifact_payload, wrap_artifact
from iris_bot.config import Settings
from iris_bot.demo_session_series import build_series_runtime_paths, load_series_registry
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.serious_demo_gate import generate_serious_demo_control_report


__all__ = [
    "build_controlled_execution_runbook",
    "generate_prolonged_serious_demo_report",
    "prolonged_serious_demo_gate",
    "demo_forward_runbook_command",
    "generate_demo_serious_validated_report",
    "demo_serious_validated_gate",
]

# Minimum thresholds for demo_serious_validated
_MIN_COMPLETED_VALID_SERIES = 2
_MIN_TOTAL_SUCCESSFUL_SESSIONS = 6
_MAX_REJECTED_RATIO = 0.20  # orders_rejected / orders_sent


MIN_FORWARD_SESSIONS = 3


def _latest_series_payload(settings: Settings) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    registry = load_series_registry(settings)
    series = dict(registry.get("series", {}))
    if not series:
        return registry, None
    selected = None
    for candidate in series.values():
        if selected is None:
            selected = dict(candidate)
            continue
        if str(candidate.get("started_at", "")) > str(selected.get("started_at", "")):
            selected = dict(candidate)
    if selected is None:
        return registry, None
    series_artifact = build_series_runtime_paths(settings)["artifacts"] / f"{selected['session_series_id']}.json"
    payload = read_artifact_payload(series_artifact, expected_type="demo_session_series") if series_artifact.exists() else None
    return registry, {"state": selected, "artifact_path": str(series_artifact), "payload": payload}


def build_controlled_execution_runbook(settings: Settings) -> dict[str, Any]:
    """Short operational runbook for prolonged demo execution."""
    return {
        "mode": "demo_serio_prolongado",
        "guardrails": [
            "never_open_live_real",
            "demo_broker_only",
            "do_not_relax_gates",
            "keep_kill_switch_and_circuit_breaker_enforced",
        ],
        "steps": [
            {
                "step": "start_series",
                "command": "python -m iris_bot.main start-demo-forward-series",
                "success_criteria": "active session_series_id created and registry updated",
            },
            {
                "step": "validate_preconditions",
                "command": "python -m iris_bot.main demo-execution-preflight",
                "success_criteria": "preflight ok and demo account confirmed",
            },
            {
                "step": "run_controlled_session",
                "command": "python -m iris_bot.main run-demo-execution",
                "success_criteria": "session evidence and review written with correlated session_series_id",
            },
            {
                "step": "review_session",
                "command": "python -m iris_bot.main demo-forward-series-status",
                "success_criteria": "session classified healthy/caution/failed and evidence complete",
            },
            {
                "step": "decide_continue_hold_abort",
                "command": "inspect latest session review recommendation",
                "success_criteria": "continue only on healthy; hold on caution; abort on failed",
            },
            {
                "step": "close_series",
                "command": "python -m iris_bot.main close-demo-forward-series",
                "success_criteria": "series ended and aggregate evidence finalized",
            },
            {
                "step": "interpret_gate",
                "command": "python -m iris_bot.main prolonged-serious-demo-gate",
                "success_criteria": "decision, blockers, warnings and next_recommendation emitted",
            },
        ],
        "defaults": {
            "minimum_forward_sessions": MIN_FORWARD_SESSIONS,
            "target_symbol": settings.demo_execution.target_symbol,
            "auto_close_after_entry": settings.demo_execution.auto_close_after_entry,
        },
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }


def generate_prolonged_serious_demo_report(settings: Settings) -> dict[str, Any]:
    """Determine readiness for prolonged serious demo validation."""
    serious_demo_report = generate_serious_demo_control_report(settings)
    registry, latest_series = _latest_series_payload(settings)
    blockers: list[str] = []
    warnings: list[str] = []
    checks: dict[str, dict[str, Any]] = {}

    serious_ok = serious_demo_report.get("decision") == "ready_for_controlled_serious_demo"
    checks["serious_demo_control_gate"] = {
        "ok": serious_ok,
        "decision": serious_demo_report.get("decision"),
        "reason": "ok" if serious_ok else "serious_demo_control_gate_not_green",
    }
    if not serious_ok:
        blockers.append("serious_demo_control_gate_not_green")

    if latest_series is None:
        checks["forward_series_exists"] = {"ok": False, "reason": "no_forward_series_evidence"}
        blockers.append("no_forward_series_evidence")
        series_payload = None
        series_state = {}
    else:
        series_payload = latest_series.get("payload")
        series_state = dict(latest_series.get("state", {}))
        checks["forward_series_exists"] = {
            "ok": series_payload is not None,
            "series_artifact_path": latest_series.get("artifact_path"),
            "reason": "ok" if series_payload is not None else "series_artifact_missing",
        }
        if series_payload is None:
            blockers.append("series_artifact_missing")

    minimum_sessions_ok = False
    evidence_complete_ok = False
    valid_series_ok = False
    repeated_critical_ok = False
    repeated_degrading_ok = False
    discipline_ok = False
    if series_payload is not None:
        aggregate_counts = dict(series_payload.get("aggregate_counts", {}))
        session_reviews = list(series_payload.get("session_reviews", []))
        successful_sessions = int(aggregate_counts.get("successful_sessions", 0) or 0)
        evidence_complete_sessions = sum(1 for review in session_reviews if bool(review.get("evidence_complete", False)))
        repeated_critical_failures = successful_sessions < MIN_FORWARD_SESSIONS and (
            int(aggregate_counts.get("aborted_sessions", 0) or 0) >= 2
            or int(aggregate_counts.get("kill_switch_events", 0) or 0) >= 2
            or int(aggregate_counts.get("circuit_breaker_triggers", 0) or 0) >= 2
        )
        repeated_degrading = (
            int(aggregate_counts.get("sessions_with_divergence", 0) or 0) >= 2
            or int(aggregate_counts.get("sessions_with_recovery", 0) or 0) >= 2
        )
        minimum_sessions_ok = successful_sessions >= MIN_FORWARD_SESSIONS
        evidence_complete_ok = evidence_complete_sessions == len(session_reviews) and len(session_reviews) > 0
        valid_series_ok = bool((series_state.get("latest_series_review") or {}).get("valid_forward_series", False))
        repeated_critical_ok = not repeated_critical_failures
        repeated_degrading_ok = not repeated_degrading
        discipline_ok = all(str(review.get("recommendation", "")) in {"continue", "hold", "abort"} for review in session_reviews) and len(session_reviews) > 0
        checks["minimum_successful_sessions"] = {
            "ok": minimum_sessions_ok,
            "successful_sessions": successful_sessions,
            "required": MIN_FORWARD_SESSIONS,
            "reason": "ok" if minimum_sessions_ok else "insufficient_successful_demo_sessions",
        }
        checks["evidence_complete"] = {
            "ok": evidence_complete_ok,
            "evidence_complete_sessions": evidence_complete_sessions,
            "session_reviews": len(session_reviews),
            "reason": "ok" if evidence_complete_ok else "incomplete_session_or_series_evidence",
        }
        checks["valid_forward_series"] = {
            "ok": valid_series_ok,
            "review": series_state.get("latest_series_review"),
            "reason": "ok" if valid_series_ok else "latest_series_not_valid_for_forward",
        }
        checks["no_repeated_critical_failures"] = {
            "ok": repeated_critical_ok,
            "reason": "ok" if repeated_critical_ok else "repeated_critical_failures_detected",
        }
        checks["no_repeated_degrading_reconcile_or_recovery"] = {
            "ok": repeated_degrading_ok,
            "reason": "ok" if repeated_degrading_ok else "repeated_degrading_reconcile_or_recovery",
        }
        checks["session_discipline"] = {
            "ok": discipline_ok,
            "reason": "ok" if discipline_ok else "session_discipline_incomplete",
        }

        if not minimum_sessions_ok:
            blockers.append("insufficient_successful_demo_sessions")
        if not evidence_complete_ok:
            blockers.append("incomplete_session_or_series_evidence")
        if not valid_series_ok:
            blockers.append("latest_series_not_valid_for_forward")
        if not repeated_critical_ok:
            blockers.append("repeated_critical_failures_detected")
        if not repeated_degrading_ok:
            warnings.append("repeated_degrading_reconcile_or_recovery")
        if not discipline_ok:
            warnings.append("session_discipline_incomplete")
    else:
        for name, reason in [
            ("minimum_successful_sessions", "no_series_payload"),
            ("evidence_complete", "no_series_payload"),
            ("valid_forward_series", "no_series_payload"),
            ("no_repeated_critical_failures", "no_series_payload"),
            ("no_repeated_degrading_reconcile_or_recovery", "no_series_payload"),
            ("session_discipline", "no_series_payload"),
        ]:
            checks[name] = {"ok": False, "reason": reason}
        blockers.extend(
            [
                "insufficient_successful_demo_sessions",
                "incomplete_session_or_series_evidence",
                "latest_series_not_valid_for_forward",
            ]
        )

    if blockers:
        decision = "not_ready_for_prolonged_serious_demo"
        next_recommendation = "build_real_demo_forward_evidence_until_the_blockers_clear"
    elif warnings:
        decision = "prolonged_serious_demo_with_reservations"
        next_recommendation = "continue_only_under_hold_discipline_and_review_degrading_patterns"
    else:
        decision = "ready_for_prolonged_serious_demo"
        next_recommendation = "continue_controlled_demo_series_and_monitor_forward_metrics"

    return {
        "decision": decision,
        "blockers": blockers,
        "warnings": warnings,
        "reasons": blockers + warnings,
        "next_recommendation": next_recommendation,
        "checks": checks,
        "series_registry": registry,
        "latest_series_state": series_state,
        "latest_series_payload": series_payload,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }


def generate_demo_serious_validated_report(settings: Settings) -> dict[str, Any]:
    """
    Gate: demo_serious_validated.

    Requires:
    - Prerequisite gates green (serious-demo-control-gate, prolonged-serious-demo-gate)
    - ≥ 2 completed valid series with full evidence
    - ≥ 6 total successful sessions across all valid series
    - 0 cumulative kill switch events
    - 0 cumulative circuit breaker triggers
    - 0 cumulative aborted sessions
    - orders_rejected / orders_sent < 20% cumulative
    - 0 sessions_with_divergence cumulative

    Decisions:
    - demo_serious_validated
    - demo_validated_with_reservations  (warnings only)
    - not_yet_demo_validated            (blockers present)
    """
    registry = load_series_registry(settings)
    series_map = dict(registry.get("series", {}))
    paths = build_series_runtime_paths(settings)

    blockers: list[str] = []
    warnings: list[str] = []
    checks: dict[str, dict[str, Any]] = {}

    # --- Prerequisite: serious-demo-control-gate ---
    serious_report = generate_serious_demo_control_report(settings)
    serious_ok = serious_report.get("decision") == "ready_for_controlled_serious_demo"
    checks["serious_demo_control_gate"] = {
        "ok": serious_ok,
        "decision": serious_report.get("decision"),
        "reason": "ok" if serious_ok else "serious_demo_control_gate_not_green",
    }
    if not serious_ok:
        blockers.append("serious_demo_control_gate_not_green")

    # --- Prerequisite: prolonged-serious-demo-gate (any completed valid series) ---
    # We check independently whether at least one valid series ever met prolonged
    # gate criteria — the latest series may be partial/active.
    any_series_met_prolonged = any(
        int(dict(s.get("aggregate_counts", {})).get("successful_sessions", 0) or 0) >= MIN_FORWARD_SESSIONS
        and str(s.get("status", "")) == "completed"
        for s in series_map.values()
    )
    checks["prolonged_serious_demo_gate"] = {
        "ok": any_series_met_prolonged,
        "reason": "ok" if any_series_met_prolonged else "no_completed_series_met_prolonged_gate",
        "note": "passes if any completed series has >= MIN_FORWARD_SESSIONS successful sessions",
    }
    if not any_series_met_prolonged:
        blockers.append("no_completed_series_met_prolonged_gate")

    # --- Collect all completed series with ≥ 3 sessions ---
    completed_valid: list[dict[str, Any]] = []
    series_details: list[dict[str, Any]] = []

    for sid, series in series_map.items():
        status = str(series.get("status", ""))
        session_ids = list(series.get("session_ids", []))
        counts = dict(series.get("aggregate_counts", {}))
        successful = int(counts.get("successful_sessions", 0) or 0)
        artifact_path = paths["artifacts"] / f"{sid}.json"
        artifact_ok = artifact_path.exists()
        is_complete_valid = (
            status == "completed"
            and successful >= MIN_FORWARD_SESSIONS
            and artifact_ok
        )
        if is_complete_valid:
            completed_valid.append(series)
        series_details.append({
            "session_series_id": sid,
            "status": status,
            "sessions": len(session_ids),
            "successful_sessions": successful,
            "is_complete_valid": is_complete_valid,
            "artifact_ok": artifact_ok,
            "started_at": series.get("started_at", ""),
        })

    completed_count = len(completed_valid)
    checks["completed_valid_series"] = {
        "ok": completed_count >= _MIN_COMPLETED_VALID_SERIES,
        "completed_valid_series": completed_count,
        "required": _MIN_COMPLETED_VALID_SERIES,
        "series_details": series_details,
        "reason": "ok" if completed_count >= _MIN_COMPLETED_VALID_SERIES
        else f"insufficient_completed_valid_series:{completed_count}<{_MIN_COMPLETED_VALID_SERIES}",
    }
    if completed_count < _MIN_COMPLETED_VALID_SERIES:
        blockers.append(f"insufficient_completed_valid_series:{completed_count}<{_MIN_COMPLETED_VALID_SERIES}")

    # --- Cumulative counts across ALL completed valid series ---
    total_successful = 0
    total_sent = 0
    total_rejected = 0
    total_kill_switch = 0
    total_circuit_breaker = 0
    total_aborted = 0
    total_divergence_sessions = 0

    for series in completed_valid:
        c = dict(series.get("aggregate_counts", {}))
        total_successful += int(c.get("successful_sessions", 0) or 0)
        total_sent += int(c.get("orders_sent", 0) or 0)
        total_rejected += int(c.get("orders_rejected", 0) or 0)
        total_kill_switch += int(c.get("kill_switch_events", 0) or 0)
        total_circuit_breaker += int(c.get("circuit_breaker_triggers", 0) or 0)
        total_aborted += int(c.get("aborted_sessions", 0) or 0)
        total_divergence_sessions += int(c.get("sessions_with_divergence", 0) or 0)

    checks["total_successful_sessions"] = {
        "ok": total_successful >= _MIN_TOTAL_SUCCESSFUL_SESSIONS,
        "total_successful_sessions": total_successful,
        "required": _MIN_TOTAL_SUCCESSFUL_SESSIONS,
        "reason": "ok" if total_successful >= _MIN_TOTAL_SUCCESSFUL_SESSIONS
        else f"insufficient_total_sessions:{total_successful}<{_MIN_TOTAL_SUCCESSFUL_SESSIONS}",
    }
    if total_successful < _MIN_TOTAL_SUCCESSFUL_SESSIONS:
        blockers.append(f"insufficient_total_sessions:{total_successful}<{_MIN_TOTAL_SUCCESSFUL_SESSIONS}")

    checks["no_critical_failures_cumulative"] = {
        "ok": total_kill_switch == 0 and total_circuit_breaker == 0 and total_aborted == 0,
        "kill_switch_events": total_kill_switch,
        "circuit_breaker_triggers": total_circuit_breaker,
        "aborted_sessions": total_aborted,
        "reason": "ok" if (total_kill_switch == 0 and total_circuit_breaker == 0 and total_aborted == 0)
        else "critical_failures_detected",
    }
    if total_kill_switch > 0 or total_circuit_breaker > 0 or total_aborted > 0:
        blockers.append("critical_failures_detected")

    rejected_ratio = (total_rejected / total_sent) if total_sent > 0 else 0.0
    checks["orders_rejected_ratio"] = {
        "ok": rejected_ratio < _MAX_REJECTED_RATIO,
        "orders_sent": total_sent,
        "orders_rejected": total_rejected,
        "rejected_ratio": round(rejected_ratio, 4),
        "threshold": _MAX_REJECTED_RATIO,
        "reason": "ok" if rejected_ratio < _MAX_REJECTED_RATIO else "high_order_rejection_rate",
    }
    if rejected_ratio >= _MAX_REJECTED_RATIO:
        blockers.append("high_order_rejection_rate")

    checks["no_broker_divergence_cumulative"] = {
        "ok": total_divergence_sessions == 0,
        "sessions_with_divergence": total_divergence_sessions,
        "reason": "ok" if total_divergence_sessions == 0 else "broker_divergence_detected",
    }
    if total_divergence_sessions > 0:
        warnings.append("broker_divergence_detected")

    # --- Temporal diversity warning ---
    start_times = sorted(str(s.get("started_at", "")) for s in completed_valid if s.get("started_at"))
    temporal_span_hours: float | None = None
    if len(start_times) >= 2:
        import datetime as _dt_mod
        try:
            t0 = _dt_mod.datetime.fromisoformat(start_times[0])
            t1 = _dt_mod.datetime.fromisoformat(start_times[-1])
            temporal_span_hours = round(abs((t1 - t0).total_seconds()) / 3600.0, 2)
        except ValueError:
            pass
    temporal_diverse = temporal_span_hours is not None and temporal_span_hours >= 1.0
    checks["temporal_diversity"] = {
        "ok": temporal_diverse,
        "span_hours": temporal_span_hours,
        "min_span_hours_for_ok": 1.0,
        "series_count": completed_count,
        "reason": "ok" if temporal_diverse else "series_within_same_hour",
    }
    if not temporal_diverse:
        warnings.append("limited_temporal_diversity:same_day_operation")

    # --- Final decision ---
    cumulative_summary = {
        "completed_valid_series": completed_count,
        "total_successful_sessions": total_successful,
        "total_orders_sent": total_sent,
        "total_orders_rejected": total_rejected,
        "total_kill_switch_events": total_kill_switch,
        "total_circuit_breaker_triggers": total_circuit_breaker,
        "total_aborted_sessions": total_aborted,
        "total_divergence_sessions": total_divergence_sessions,
        "temporal_span_hours": temporal_span_hours,
    }

    if blockers:
        decision = "not_yet_demo_validated"
        next_recommendation = "resolve_blockers_before_considering_scaling"
    elif warnings:
        decision = "demo_validated_with_reservations"
        next_recommendation = "accumulate_multi_day_evidence_before_opening_second_symbol"
    else:
        decision = "demo_serious_validated"
        next_recommendation = "consider_controlled_expansion_to_second_approved_symbol"

    return {
        "decision": decision,
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
        "cumulative_summary": cumulative_summary,
        "next_recommendation": next_recommendation,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }


def demo_serious_validated_gate(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "demo_serious_validated")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    report = generate_demo_serious_validated_report(settings)
    write_json_report(
        run_dir,
        "demo_serious_validated_report.json",
        wrap_artifact(
            "demo_serious_validated",
            report,
            provenance=build_artifact_provenance(
                run_dir=run_dir,
                policy_version="demo_serious_validated_v1",
                policy_source=str(settings.project_root),
                correlation_keys={"command": "demo-serious-validated"},
            ),
        ),
    )
    logger.info(
        "demo_serious_validated decision=%s blockers=%s warnings=%s run_dir=%s",
        report["decision"],
        report["blockers"],
        report["warnings"],
        run_dir,
    )
    if report["decision"] == "demo_serious_validated":
        return 0
    if report["decision"] == "demo_validated_with_reservations":
        return 1
    return 2


def demo_forward_runbook_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "demo_forward_runbook")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    report = build_controlled_execution_runbook(settings)
    write_json_report(
        run_dir,
        "demo_forward_runbook.json",
        wrap_artifact(
            "demo_forward_runbook",
            report,
            provenance=build_artifact_provenance(
                run_dir=run_dir,
                policy_version="demo_forward_runbook_v1",
                policy_source=str(settings.project_root),
                correlation_keys={"command": "demo_forward_runbook"},
            ),
        ),
    )
    logger.info("demo_forward_runbook generated run_dir=%s", run_dir)
    return 0


def prolonged_serious_demo_gate(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "prolonged_serious_demo_gate")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    report = generate_prolonged_serious_demo_report(settings)
    write_json_report(
        run_dir,
        "prolonged_serious_demo_gate_report.json",
        wrap_artifact(
            "prolonged_serious_demo_gate",
            report,
            provenance=build_artifact_provenance(
                run_dir=run_dir,
                policy_version="prolonged_serious_demo_gate_v1",
                policy_source=str(settings.project_root),
                correlation_keys={"command": "prolonged_serious_demo_gate"},
            ),
        ),
    )
    logger.info(
        "prolonged_serious_demo_gate decision=%s blockers=%s warnings=%s run_dir=%s",
        report["decision"],
        report["blockers"],
        report["warnings"],
        run_dir,
    )
    if report["decision"] == "ready_for_prolonged_serious_demo":
        return 0
    if report["decision"] == "prolonged_serious_demo_with_reservations":
        return 1
    return 2
