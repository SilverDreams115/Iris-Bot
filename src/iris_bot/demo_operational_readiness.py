"""Demo Operational Readiness Gate — IRIS-Bot.

Evaluates whether the system is operationally credible for demo-guarded
execution. More stringent than demo_readiness.py: it verifies that the
operational resilience infrastructure (restore, reconcile, recovery,
soak, kill switch, session discipline) is correctly implemented and
structurally sound.

Decision values:
    "credible_guarded"         — all required checks pass
    "not_yet_operationally_credible" — one or more required checks fail

This gate does NOT send orders. Does NOT connect to MT5. Read-only.
"""
from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_bot.artifacts import build_artifact_provenance, wrap_artifact
from iris_bot.config import Settings
from iris_bot.kill_switch import (
    activate_kill_switch,
    build_default_circuit_breaker_conditions,
    circuit_breaker_check,
    is_kill_switch_active,
)
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.resilient import (
    fresh_state,
    restore_runtime_state,
    run_reconciliation_drills,
    run_recovery_drills,
    run_restore_safety_drill,
    validate_restored_state_invariants,
)
from iris_bot.resilient_state import STATE_SCHEMA_VERSION
from iris_bot.session_discipline import (
    generate_session_runbook,
    session_startup_check,
)
from iris_bot.soak import DemoGuardedSoakSummary


_REQUIRED_BLOQUE_MODULES = [
    "iris_bot.resilient_state",
    "iris_bot.resilient_reconcile",
    "iris_bot.resilient",
    "iris_bot.kill_switch",
    "iris_bot.session_discipline",
    "iris_bot.soak",
]


def _check_restore_safety_infrastructure(settings: Settings) -> dict[str, Any]:
    """BLOQUE 1: Verify restore/restart safety infrastructure is correct."""
    checks: dict[str, Any] = {}
    issues: list[str] = []

    # Schema version constant must be set
    checks["schema_version_defined"] = STATE_SCHEMA_VERSION > 0
    if not checks["schema_version_defined"]:
        issues.append("schema_version_not_defined")

    # validate_restored_state_invariants must work on a clean state
    try:
        state = fresh_state(settings.backtest.starting_balance_usd, "check")
        inv_issues = validate_restored_state_invariants(state)
        checks["invariant_validator_functional"] = inv_issues == []
        if inv_issues:
            issues.append(f"invariant_validator_reports_issues_on_fresh_state:{inv_issues}")
    except Exception as exc:  # noqa: BLE001
        checks["invariant_validator_functional"] = False
        issues.append(f"invariant_validator_exception:{exc}")

    # persist + restore round-trip works
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime_state.json"
            state = fresh_state(settings.backtest.starting_balance_usd, "check")
            from iris_bot.resilient import persist_runtime_state
            persist_runtime_state(path, state, {})
            restored, report = restore_runtime_state(path, require_clean=True)
            checks["persist_restore_round_trip"] = report.ok and restored is not None
            if not checks["persist_restore_round_trip"]:
                issues.append(f"persist_restore_failed:{report.issues}")
    except Exception as exc:  # noqa: BLE001
        checks["persist_restore_round_trip"] = False
        issues.append(f"persist_restore_exception:{exc}")

    ok = len(issues) == 0
    return {"ok": ok, "checks": checks, "issues": issues, "reason": "ok" if ok else "restore_safety_infrastructure_degraded"}


def _check_reconciliation_infrastructure(settings: Settings) -> dict[str, Any]:
    """BLOQUE 2: Verify all 6 reconciliation scenarios are covered."""
    try:
        exit_code, report = run_reconciliation_drills(settings)
        ok = exit_code == 0 and report.get("ok", False)
        return {
            "ok": ok,
            "scenarios_total": report.get("scenarios_total", 0),
            "scenarios_passed": report.get("scenarios_passed", 0),
            "failed_scenarios": report.get("failed_scenarios", []),
            "reason": "ok" if ok else "reconciliation_drill_scenarios_failed",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"reconciliation_drill_exception:{exc}"}


def _check_recovery_infrastructure(settings: Settings) -> dict[str, Any]:
    """BLOQUE 3: Verify all 4 disconnect/reconnect scenarios pass."""
    try:
        exit_code, report = run_recovery_drills(settings)
        ok = exit_code == 0 and report.get("ok", False)
        return {
            "ok": ok,
            "scenarios_total": report.get("scenarios_total", 0),
            "scenarios_passed": report.get("scenarios_passed", 0),
            "failed_scenarios": report.get("failed_scenarios", []),
            "reason": "ok" if ok else "recovery_drill_scenarios_failed",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"recovery_drill_exception:{exc}"}


def _check_restore_safety_drill(settings: Settings) -> dict[str, Any]:
    """BLOQUE 1 integrated drill: run all restore invariants."""
    try:
        exit_code, report = run_restore_safety_drill(settings)
        ok = exit_code == 0 and report.get("ok", False)
        return {
            "ok": ok,
            "failed_checks": report.get("failed_checks", []),
            "checks_total": len(report.get("checks", {})),
            "reason": "ok" if ok else "restore_safety_drill_failed",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"restore_safety_drill_exception:{exc}"}


def _check_soak_infrastructure(settings: Settings) -> dict[str, Any]:
    """BLOQUE 4: Verify demo-guarded soak infrastructure is importable and functional."""
    checks: dict[str, Any] = {}
    issues: list[str] = []

    # DemoGuardedSoakSummary must be importable
    try:
        s = DemoGuardedSoakSummary(
            cycles_requested=1, cycles_completed=1,
            restore_events=1, restore_failures=0,
            reconcile_events=1, reconcile_failures=0,
            blocked_trade_events=0, critical_alerts=0,
            warning_alerts=0, circuit_breaker_triggers=0,
            no_go_cycles=0, overall_decision="go",
        )
        checks["demo_guarded_soak_summary_functional"] = s.overall_decision == "go"
    except Exception as exc:  # noqa: BLE001
        checks["demo_guarded_soak_summary_functional"] = False
        issues.append(f"soak_summary_exception:{exc}")

    # run_demo_guarded_soak must be importable
    try:
        from iris_bot.soak import run_demo_guarded_soak  # noqa: F401
        checks["run_demo_guarded_soak_importable"] = True
    except ImportError as exc:
        checks["run_demo_guarded_soak_importable"] = False
        issues.append(f"run_demo_guarded_soak_not_importable:{exc}")

    ok = len(issues) == 0
    return {"ok": ok, "checks": checks, "issues": issues, "reason": "ok" if ok else "soak_infrastructure_degraded"}


def _check_kill_switch_infrastructure(settings: Settings) -> dict[str, Any]:
    """BLOQUE 5: Verify kill switch and circuit breaker are functional."""
    checks: dict[str, Any] = {}
    issues: list[str] = []

    # kill switch activation must work
    try:
        from iris_bot.operational import AlertRecord, AccountState, BrokerSyncStatus, DailyLossTracker, PaperEngineState
        state = PaperEngineState(
            account_state=AccountState(1000.0, 1000.0, 1000.0),
            broker_sync_status=BrokerSyncStatus(),
            daily_loss_tracker=DailyLossTracker("2026-01-01", 0.0, 50.0, False),
        )
        alerts: list[AlertRecord] = []
        activate_kill_switch(state, "infra_check", "manual", alerts)
        checks["kill_switch_activation"] = is_kill_switch_active(state) and len(alerts) == 1
        if not checks["kill_switch_activation"]:
            issues.append("kill_switch_activation_failed")
    except Exception as exc:  # noqa: BLE001
        checks["kill_switch_activation"] = False
        issues.append(f"kill_switch_exception:{exc}")

    # circuit breaker must work
    try:
        from iris_bot.operational import AlertRecord, AccountState, BrokerSyncStatus, DailyLossTracker, PaperEngineState
        state2 = PaperEngineState(
            account_state=AccountState(1000.0, 1000.0, 1000.0),
            broker_sync_status=BrokerSyncStatus(),
            daily_loss_tracker=DailyLossTracker("2026-01-01", 0.0, 50.0, False),
        )
        alerts2: list[AlertRecord] = []
        conditions = build_default_circuit_breaker_conditions()
        # clean state → no trigger
        result = circuit_breaker_check(state2, conditions, alerts2)
        checks["circuit_breaker_no_false_trigger"] = result is None
        if result is not None:
            issues.append("circuit_breaker_false_trigger_on_clean_state")
    except Exception as exc:  # noqa: BLE001
        checks["circuit_breaker_no_false_trigger"] = False
        issues.append(f"circuit_breaker_exception:{exc}")

    # KillSwitchReport serialisable
    try:
        from iris_bot.kill_switch import KillSwitchReport
        r = KillSwitchReport(
            event_type="kill_switch", triggered_by="manual", reason="test",
            triggered_at="2026-01-01T00:00:00Z", blocked_reasons_added=["kill_switch:test"],
            prior_blocked_reasons=[], condition_name="",
            state_snapshot={"open_positions": 0, "blocked_reasons": []},
        )
        checks["kill_switch_report_serialisable"] = bool(json.loads(json.dumps(r.to_dict())))
    except Exception as exc:  # noqa: BLE001
        checks["kill_switch_report_serialisable"] = False
        issues.append(f"kill_switch_report_exception:{exc}")

    ok = len(issues) == 0
    return {"ok": ok, "checks": checks, "issues": issues, "reason": "ok" if ok else "kill_switch_infrastructure_degraded"}


def _check_session_discipline_infrastructure(settings: Settings) -> dict[str, Any]:
    """BLOQUE 6: Verify runbook and session startup check are functional."""
    checks: dict[str, Any] = {}
    issues: list[str] = []

    # Runbook must be generated with required structure
    try:
        runbook = generate_session_runbook()
        phases = {s.phase for s in runbook.steps}
        required_phases = {"startup", "pre_run", "restart", "mismatch", "disconnect", "close", "post_run"}
        checks["runbook_phases_complete"] = required_phases.issubset(phases)
        if not checks["runbook_phases_complete"]:
            issues.append(f"runbook_missing_phases:{required_phases - phases}")
        checks["runbook_abort_criteria_present"] = len(runbook.abort_criteria) >= 5
        checks["runbook_serialisable"] = bool(json.loads(json.dumps(runbook.to_dict())))
    except Exception as exc:  # noqa: BLE001
        checks["runbook_phases_complete"] = False
        issues.append(f"runbook_exception:{exc}")

    # session_startup_check must work on fresh settings
    try:
        startup_report = session_startup_check(settings)
        checks["session_startup_check_functional"] = startup_report.decision in {"proceed", "abort", "hold"}
    except Exception as exc:  # noqa: BLE001
        checks["session_startup_check_functional"] = False
        issues.append(f"session_startup_check_exception:{exc}")

    ok = len(issues) == 0
    return {"ok": ok, "checks": checks, "issues": issues, "reason": "ok" if ok else "session_discipline_infrastructure_degraded"}


def _check_module_imports() -> dict[str, Any]:
    """Verify all required resilience modules are importable."""
    results: dict[str, bool] = {}
    failed: list[str] = []
    for module in _REQUIRED_BLOQUE_MODULES:
        try:
            importlib.import_module(module)
            results[module] = True
        except ImportError as exc:
            results[module] = False
            failed.append(f"{module}:{exc}")
    ok = len(failed) == 0
    return {"ok": ok, "modules": results, "failed": failed, "reason": "ok" if ok else "required_modules_not_importable"}


def generate_demo_operational_readiness_report(settings: Settings) -> dict[str, Any]:
    """Generate the demo operational readiness report.

    Required checks (failure → not_yet_operationally_credible):
    - module_imports
    - restore_safety_infrastructure
    - restore_safety_drill
    - reconciliation_infrastructure
    - recovery_infrastructure
    - soak_infrastructure
    - kill_switch_infrastructure
    - session_discipline_infrastructure
    """
    checks: dict[str, dict[str, Any]] = {
        "module_imports": _check_module_imports(),
        "restore_safety_infrastructure": _check_restore_safety_infrastructure(settings),
        "restore_safety_drill": _check_restore_safety_drill(settings),
        "reconciliation_infrastructure": _check_reconciliation_infrastructure(settings),
        "recovery_infrastructure": _check_recovery_infrastructure(settings),
        "soak_infrastructure": _check_soak_infrastructure(settings),
        "kill_switch_infrastructure": _check_kill_switch_infrastructure(settings),
        "session_discipline_infrastructure": _check_session_discipline_infrastructure(settings),
    }

    required_checks = list(checks.keys())
    failed_required = [k for k in required_checks if not checks[k].get("ok", False)]

    decision = "credible_guarded" if not failed_required else "not_yet_operationally_credible"
    blocking_reasons = [checks[k].get("reason", "unknown") for k in failed_required]

    return {
        "decision": decision,
        "operationally_credible": decision == "credible_guarded",
        "failed_required_checks": failed_required,
        "blocking_reasons": blocking_reasons,
        "checks": checks,
        "phase": "demo_guarded_operational_readiness",
        "schema_version": STATE_SCHEMA_VERSION,
        "bloque_coverage": {
            "bloque_1_restore_restart_safety": checks["restore_safety_infrastructure"]["ok"] and checks["restore_safety_drill"]["ok"],
            "bloque_2_reconciliation_drills": checks["reconciliation_infrastructure"]["ok"],
            "bloque_3_recovery_disconnect": checks["recovery_infrastructure"]["ok"],
            "bloque_4_soak_readiness": checks["soak_infrastructure"]["ok"],
            "bloque_5_kill_switch": checks["kill_switch_infrastructure"]["ok"],
            "bloque_6_session_discipline": checks["session_discipline_infrastructure"]["ok"],
        },
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }


def demo_operational_readiness(settings: Settings) -> int:
    """Command: generates demo_operational_readiness_report.json.

    Returns:
        0 if credible_guarded
        2 if not_yet_operationally_credible
    """
    run_dir = build_run_directory(settings.data.runs_dir, "demo_operational_readiness")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)

    report = generate_demo_operational_readiness_report(settings)
    write_json_report(
        run_dir,
        "demo_operational_readiness_report.json",
        wrap_artifact(
            "demo_operational_readiness",
            report,
            provenance=build_artifact_provenance(
                run_dir=run_dir,
                policy_version="demo_guarded_v1",
                policy_source=str(settings.project_root),
                correlation_keys={"command": "demo_operational_readiness"},
                references={"schema_version": str(report["schema_version"])},
            ),
        ),
    )

    decision = report["decision"]
    logger.info(
        "demo_operational_readiness decision=%s failed=%s run_dir=%s",
        decision, report["failed_required_checks"], run_dir,
    )
    return 0 if decision == "credible_guarded" else 2
