"""Serious demo control gate — IRIS-Bot.

Evaluates whether the system is ready for controlled serious demo execution
(real demo broker, real order flow, strictly controlled).

Decision values:
    "not_ready_for_serious_demo"          — one or more required checks fail
    "ready_for_controlled_serious_demo"   — all checks pass, no reservations
    "serious_demo_with_reservations"      — required checks pass, warnings present

This gate does NOT send orders. Does NOT connect to MT5. Read-only.
"""
from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from typing import Any

from iris_bot.artifacts import build_artifact_provenance, wrap_artifact
from iris_bot.config import Settings
from iris_bot.demo_operational_readiness import generate_demo_operational_readiness_report
from iris_bot.demo_readiness import generate_demo_execution_readiness_report
from iris_bot.kill_switch import build_default_circuit_breaker_conditions, is_kill_switch_active
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.resilient import fresh_state


__all__ = [
    "generate_serious_demo_control_report",
    "serious_demo_control_gate",
]

_REQUIRED_DEMO_MODULES = [
    "iris_bot.demo_execution",
    "iris_bot.demo_session_guard",
    "iris_bot.demo_forward_evidence",
    "iris_bot.demo_session_series",
    "iris_bot.kill_switch",
    "iris_bot.session_discipline",
]


def _check_module_imports() -> dict[str, Any]:
    results: dict[str, bool] = {}
    failed: list[str] = []
    for module in _REQUIRED_DEMO_MODULES:
        try:
            importlib.import_module(module)
            results[module] = True
        except ImportError as exc:
            results[module] = False
            failed.append(f"{module}:{exc}")
    ok = len(failed) == 0
    return {"ok": ok, "modules": results, "failed": failed, "reason": "ok" if ok else "required_modules_not_importable"}


def _check_operational_gate(settings: Settings) -> dict[str, Any]:
    """Required: demo_operational_readiness must be credible_guarded."""
    try:
        report = generate_demo_operational_readiness_report(settings)
        ok = report.get("decision") == "credible_guarded"
        return {
            "ok": ok,
            "decision": report.get("decision"),
            "failed_checks": report.get("failed_required_checks", []),
            "reason": "ok" if ok else "demo_operational_readiness_not_credible_guarded",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"demo_operational_readiness_exception:{exc}"}


def _check_demo_execution_gate(settings: Settings) -> dict[str, Any]:
    """Required: demo_execution_readiness must already be ready_for_demo_guarded."""
    try:
        report = generate_demo_execution_readiness_report(settings)
        ok = report.get("decision") == "ready_for_demo_guarded"
        return {
            "ok": ok,
            "decision": report.get("decision"),
            "failed_checks": report.get("failed_required_checks", []),
            "reason": "ok" if ok else "demo_execution_readiness_not_ready_for_demo_guarded",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"demo_execution_readiness_exception:{exc}"}


def _check_kill_switch_clean(settings: Settings) -> dict[str, Any]:
    """Required: kill switch infrastructure functional; fresh state has no kill switch."""
    try:
        state = fresh_state(settings.backtest.starting_balance_usd, "serious_demo_gate_check")
        clean = not is_kill_switch_active(state)
        conditions = build_default_circuit_breaker_conditions()
        all_have_names = all(c.name for c in conditions)
        ok = clean and all_have_names and len(conditions) >= 3
        return {
            "ok": ok,
            "kill_switch_on_fresh_state": not clean,
            "circuit_breaker_conditions": len(conditions),
            "reason": "ok" if ok else "kill_switch_infrastructure_issue",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"kill_switch_check_exception:{exc}"}


def _check_session_guard_infrastructure() -> dict[str, Any]:
    """Required: demo_session_guard module functional."""
    try:
        from iris_bot.demo_session_guard import DemoSessionLimits, run_demo_session_precheck  # noqa: F401
        limits = DemoSessionLimits()
        ok = limits.max_active_positions >= 1
        return {"ok": ok, "reason": "ok" if ok else "session_guard_infrastructure_issue"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"session_guard_exception:{exc}"}


def _check_forward_evidence_infrastructure() -> dict[str, Any]:
    """Required: demo_forward_evidence module functional and evidence is serializable."""
    try:
        from iris_bot.demo_forward_evidence import build_demo_session_evidence
        evidence = build_demo_session_evidence(
            session_id="gate_check",
            symbol="EURUSD",
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-01T01:00:00Z",
            preflight_report={"ok": True, "checks": {}},
        )
        serializable = bool(json.loads(json.dumps(evidence.to_dict())))
        ok = serializable and evidence.session_id == "gate_check"
        return {"ok": ok, "reason": "ok" if ok else "forward_evidence_infrastructure_issue"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"forward_evidence_exception:{exc}"}


def _check_preflight_hardened() -> dict[str, Any]:
    """Warning check: preflight has kill_switch + circuit_breaker + correct nesting."""
    try:
        import inspect
        from iris_bot.demo_execution import demo_execution_preflight_payload
        src = inspect.getsource(demo_execution_preflight_payload)
        has_kill_switch = "kill_switch_active" in src
        has_circuit_breaker = "circuit_breaker_triggered" in src
        has_state_nesting = 'runtime_state.get("state"' in src
        ok = has_kill_switch and has_circuit_breaker and has_state_nesting
        return {
            "ok": ok,
            "kill_switch_check_present": has_kill_switch,
            "circuit_breaker_check_present": has_circuit_breaker,
            "correct_state_nesting": has_state_nesting,
            "reason": "ok" if ok else "preflight_not_fully_hardened",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"preflight_hardening_check_exception:{exc}"}


def generate_serious_demo_control_report(settings: Settings) -> dict[str, Any]:
    """Generate the serious demo control report.

    Required checks (failure → not_ready_for_serious_demo):
    - module_imports
    - demo_execution_gate (must be ready_for_demo_guarded)
    - operational_gate (must be credible_guarded)
    - kill_switch_clean
    - session_guard_infrastructure
    - forward_evidence_infrastructure

    Warning checks (failure → serious_demo_with_reservations):
    - preflight_hardened
    """
    required: dict[str, dict[str, Any]] = {
        "module_imports": _check_module_imports(),
        "demo_execution_gate": _check_demo_execution_gate(settings),
        "operational_gate": _check_operational_gate(settings),
        "kill_switch_clean": _check_kill_switch_clean(settings),
        "session_guard_infrastructure": _check_session_guard_infrastructure(),
        "forward_evidence_infrastructure": _check_forward_evidence_infrastructure(),
    }
    warning: dict[str, dict[str, Any]] = {
        "preflight_hardened": _check_preflight_hardened(),
    }

    failed_required = [k for k, v in required.items() if not v.get("ok", False)]
    failed_warnings = [k for k, v in warning.items() if not v.get("ok", False)]

    if failed_required:
        decision = "not_ready_for_serious_demo"
    elif failed_warnings:
        decision = "serious_demo_with_reservations"
    else:
        decision = "ready_for_controlled_serious_demo"

    return {
        "decision": decision,
        "ready_for_serious_demo": decision != "not_ready_for_serious_demo",
        "failed_required_checks": failed_required,
        "failed_warning_checks": failed_warnings,
        "blocking_reasons": [required[k].get("reason", "unknown") for k in failed_required],
        "reservation_reasons": [warning[k].get("reason", "unknown") for k in failed_warnings],
        "checks": {**required, **warning},
        "phase": "demo_serio_controlado",
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }


def serious_demo_control_gate(settings: Settings) -> int:
    """Command: generates serious_demo_control_report.json.

    Returns:
        0 if ready_for_controlled_serious_demo
        1 if serious_demo_with_reservations
        2 if not_ready_for_serious_demo
    """
    run_dir = build_run_directory(settings.data.runs_dir, "serious_demo_control_gate")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)

    report = generate_serious_demo_control_report(settings)
    write_json_report(
        run_dir,
        "serious_demo_control_report.json",
        wrap_artifact(
            "serious_demo_control",
            report,
            provenance=build_artifact_provenance(
                run_dir=run_dir,
                policy_version="demo_serio_v1",
                policy_source=str(settings.project_root),
                correlation_keys={"command": "serious_demo_control_gate"},
                references={"phase": "demo_serio_controlado"},
            ),
        ),
    )

    decision = report["decision"]
    logger.info(
        "serious_demo_control_gate decision=%s failed_required=%s reservations=%s run_dir=%s",
        decision, report["failed_required_checks"], report["failed_warning_checks"], run_dir,
    )
    if decision == "ready_for_controlled_serious_demo":
        return 0
    if decision == "serious_demo_with_reservations":
        return 1
    return 2
