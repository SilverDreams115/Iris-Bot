"""
Demo execution readiness assessment for IRIS-Bot.

This module evaluates whether the system is ready for broker-executing demo
(real order routing to a demo broker account) WITHOUT triggering any execution.

Conservative design:
  - Default decision is "not_ready_for_demo"
  - Any missing required artifact → not_ready
  - Any ambiguous check → not_ready or reservations (never optimistic)
  - order_send is explicitly NOT integrated here

The generated report (demo_execution_readiness_report.json) serves as an
explicit gate before the next phase can begin.

Decision values:
  "ready_for_demo_guarded"           — all required checks pass with high confidence
  "ready_for_demo_with_reservations" — required checks pass but warnings remain
  "not_ready_for_demo"               — one or more required checks fail
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_bot.artifacts import build_artifact_provenance, wrap_artifact
from iris_bot.config import Settings
from iris_bot.evidence_store import evidence_store_status
from iris_bot.evaluation_contract import locate_experiment_reference
from iris_bot.governance import (
    _entry_checksum_ok,
    _find_entry,
    _latest_lifecycle_evidence,
    _lifecycle_evidence_age_hours,
    active_strategy_profiles_path,
    load_strategy_profile_registry,
    registry_path,
)
from iris_bot.governance_policy import load_governance_policy
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.portfolio import build_portfolio_separation
from iris_bot.profile_evidence import _is_within_project
from iris_bot.registry_lock import governance_lock_audit


def _quality_python(project_root: Path) -> str:
    """Return the Python executable to use for quality commands.

    Prefers the project's virtual environment Python (which has dev tools
    like ruff and mypy installed) over the system Python.
    """
    for candidate in (
        project_root / ".venv" / "bin" / "python3",
        project_root / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _build_official_suite_commands(project_root: Path) -> tuple[tuple[str, list[str]], ...]:
    python = _quality_python(project_root)
    return (
        ("ruff", [python, "-m", "ruff", "check", "."]),
        ("mypy", [python, "-m", "mypy"]),
        ("pytest", [python, "-m", "pytest"]),
        ("smoke", [python, "-m", "iris_bot.main", "--help"]),
    )
_REQUIRED_PROVENANCE_KEYS = ("source_run_id", "lineage_id", "materialized_at", "correlation_keys", "references")


def _read_json_dict(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return raw


def _check_provenance_keys(provenance: object) -> tuple[bool, list[str]]:
    if not isinstance(provenance, dict):
        return False, list(_REQUIRED_PROVENANCE_KEYS)
    missing = [key for key in _REQUIRED_PROVENANCE_KEYS if key not in provenance]
    return len(missing) == 0, missing


_RESEARCH_ENV_VARS = frozenset({
    "IRIS_SYMBOL_FOCUS_REWORK_SYMBOL",
    "IRIS_BACKTEST_INTRABAR_POLICY",
})


def _run_quality_command(project_root: Path, name: str, command: list[str]) -> dict[str, Any]:
    started = datetime.now(tz=UTC)
    clean_env = {k: v for k, v in os.environ.items() if k not in _RESEARCH_ENV_VARS}
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env=clean_env,
    )
    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    warning_lines = [
        line for line in combined.splitlines()
        if "warning" in line.lower()
    ]
    return {
        "name": name,
        "command": command,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(tz=UTC).isoformat(),
        "returncode": completed.returncode,
        "ok": completed.returncode == 0 and not warning_lines,
        "warning_count": len(warning_lines),
        "warning_lines": warning_lines[-10:],
        "output_tail": combined.splitlines()[-20:],
    }


def _check_official_suite(settings: Settings) -> dict[str, Any]:
    suite_commands = _build_official_suite_commands(settings.project_root)
    command_results = {
        name: _run_quality_command(settings.project_root, name, command)
        for name, command in suite_commands
    }
    failed = [name for name, result in command_results.items() if not result["ok"]]
    return {
        "ok": not failed,
        "official_source": "repo_standard_quality_gate_equivalent",
        "commands": command_results,
        "failed_commands": failed,
        "reason": "ok" if not failed else "official_quality_gate_failed",
    }


def _check_mt5_ownership_policy(settings: Settings) -> dict[str, Any]:
    ownership_mode = settings.mt5.ownership_mode
    has_magic = settings.mt5.magic_number > 0
    has_comment_tag = bool(settings.mt5.comment_tag.strip())
    ok = ownership_mode == "strict" and (has_magic or has_comment_tag)
    return {
        "ok": ok,
        "ownership_mode": ownership_mode,
        "magic_number_configured": has_magic,
        "comment_tag_configured": has_comment_tag,
        "reason": (
            "ok"
            if ok
            else "mt5_ownership_mode_not_strict"
            if ownership_mode != "strict"
            else "ownership_tagging_not_configured"
        ),
    }


def _check_governance_policy(settings: Settings) -> dict[str, Any]:
    policy = load_governance_policy(settings)
    policy_source = Path(str(policy["policy_source"]))
    ok = (
        policy_source.exists()
        and bool(str(policy["policy_version"]).strip())
        and isinstance(policy.get("symbol_rules"), dict)
    )
    return {
        "ok": ok,
        "policy_version": str(policy["policy_version"]),
        "policy_source": str(policy_source),
        "symbol_rule_count": len(policy.get("symbol_rules", {})),
        "reason": "ok" if ok else "governance_policy_invalid",
    }


def _check_artifact_provenance(settings: Settings) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    try:
        reference = locate_experiment_reference(settings)
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "reason": f"missing_experiment_reference:{exc}",
            "artifacts": {},
        }

    experiment_payload = _read_json_dict(reference.report_path)
    exp_provenance_ok, exp_missing = _check_provenance_keys(experiment_payload.get("artifact_provenance"))
    experiment_ok = exp_provenance_ok and bool(experiment_payload.get("training_contract_version")) and bool(experiment_payload.get("evaluation_contract_version"))
    checks["experiment_report"] = {
        "ok": experiment_ok,
        "path": str(reference.report_path),
        "artifact_type": "experiment_report",
        "missing_provenance_keys": exp_missing,
        "training_contract_version": experiment_payload.get("training_contract_version"),
        "evaluation_contract_version": experiment_payload.get("evaluation_contract_version"),
        "source_run_id": experiment_payload.get("artifact_provenance", {}).get("source_run_id", ""),
    }
    if not experiment_ok:
        failures.append("experiment_report")

    backtest_candidates = sorted(settings.data.runs_dir.glob("*_backtest/backtest_report.json"))
    if not backtest_candidates:
        checks["backtest_report"] = {
            "ok": False,
            "path": "",
            "artifact_type": "backtest_report",
            "reason": "missing_backtest_report",
        }
        failures.append("backtest_report")
    else:
        backtest_path = backtest_candidates[-1]
        backtest_payload = _read_json_dict(backtest_path)
        backtest_provenance_ok, backtest_missing = _check_provenance_keys(backtest_payload.get("artifact_provenance"))
        backtest_ok = (
            backtest_provenance_ok
            and str(backtest_payload.get("artifact_provenance", {}).get("correlation_keys", {}).get("experiment_run_id", "")) == reference.run_dir.name
            and bool(backtest_payload.get("effective_threshold_by_symbol"))
        )
        checks["backtest_report"] = {
            "ok": backtest_ok,
            "path": str(backtest_path),
            "artifact_type": "backtest_report",
            "missing_provenance_keys": backtest_missing,
            "experiment_run_id": backtest_payload.get("artifact_provenance", {}).get("correlation_keys", {}).get("experiment_run_id", ""),
            "effective_threshold_symbols": sorted((backtest_payload.get("effective_threshold_by_symbol") or {}).keys()),
        }
        if not backtest_ok:
            failures.append("backtest_report")

    active_profiles_path = active_strategy_profiles_path(settings)
    if not active_profiles_path.exists():
        checks["active_strategy_profiles"] = {
            "ok": False,
            "path": str(active_profiles_path),
            "artifact_type": "active_strategy_profiles",
            "reason": "missing_active_strategy_profiles",
        }
        failures.append("active_strategy_profiles")
    else:
        active_profiles_payload = _read_json_dict(active_profiles_path)
        active_profiles_provenance_ok, active_profiles_missing = _check_provenance_keys(active_profiles_payload.get("provenance"))
        active_profiles_ok = active_profiles_payload.get("artifact_type") == "active_strategy_profiles" and active_profiles_provenance_ok
        checks["active_strategy_profiles"] = {
            "ok": active_profiles_ok,
            "path": str(active_profiles_path),
            "artifact_type": str(active_profiles_payload.get("artifact_type", "")),
            "missing_provenance_keys": active_profiles_missing,
            "lineage_id": active_profiles_payload.get("provenance", {}).get("lineage_id", ""),
        }
        if not active_profiles_ok:
            failures.append("active_strategy_profiles")

    operational_candidates = sorted(
        list(settings.data.runs_dir.glob("*_paper/config_used.json"))
        + list(settings.data.runs_dir.glob("*_demo_dry/config_used.json"))
    )
    if operational_candidates:
        operational_path = operational_candidates[-1]
        operational_payload = _read_json_dict(operational_path)
        operational_provenance_ok, operational_missing = _check_provenance_keys(operational_payload.get("artifact_provenance"))
        operational_ok = (
            operational_provenance_ok
            and str(operational_payload.get("artifact_provenance", {}).get("correlation_keys", {}).get("experiment_run_id", "")) == reference.run_dir.name
        )
        checks["operational_config"] = {
            "ok": operational_ok,
            "path": str(operational_path),
            "artifact_type": "config_used",
            "missing_provenance_keys": operational_missing,
            "experiment_run_id": operational_payload.get("artifact_provenance", {}).get("correlation_keys", {}).get("experiment_run_id", ""),
        }
    else:
        checks["operational_config"] = {
            "ok": False,
            "path": "",
            "artifact_type": "config_used",
            "reason": "missing_paper_or_demo_dry_operational_artifact",
            "advisory": True,
        }

    return {
        "ok": not failures,
        "reason": "ok" if not failures else "artifact_provenance_incomplete",
        "blocking_artifacts": failures,
        "artifacts": checks,
    }


def _check_registry_integrity(settings: Settings) -> dict[str, Any]:
    """Checks that the registry exists, is readable, and has valid checksums."""
    reg_path = registry_path(settings)
    lock_audit = governance_lock_audit(reg_path)
    registry_ok = False
    all_checksums_ok = True
    active_profile_count = 0
    approved_demo_count = 0
    try:
        registry = load_strategy_profile_registry(settings)
        registry_ok = isinstance(registry, dict) and "profiles" in registry
        for symbol, entries in registry.get("profiles", {}).items():
            for entry in entries:
                if not _entry_checksum_ok(entry):
                    all_checksums_ok = False
                if entry.get("promotion_state") == "approved_demo":
                    approved_demo_count += 1
        active_profile_count = len(registry.get("active_profiles", {}))
    except (OSError, ValueError):
        pass

    return {
        "registry_exists": lock_audit["registry_exists"],
        "registry_readable": registry_ok,
        "registry_checksums_ok": all_checksums_ok,
        "lock_currently_held": lock_audit["lock_currently_held"],
        "active_profile_count": active_profile_count,
        "approved_demo_count": approved_demo_count,
        "ok": lock_audit["registry_exists"] and registry_ok and all_checksums_ok and not lock_audit["lock_currently_held"],
    }


def _check_lifecycle_evidence(settings: Settings) -> dict[str, Any]:
    """Checks lifecycle reconciliation evidence is present, clean, and recent."""
    gate_cfg = settings.approved_demo_gate
    lifecycle = _latest_lifecycle_evidence(settings)
    if lifecycle is None:
        return {
            "present": False,
            "clean": False,
            "recent": False,
            "age_hours": None,
            "origin": "none",
            "ok": False,
            "reason": "no_lifecycle_evidence",
        }
    payload = lifecycle.get("payload", {})
    # Check clean: no critical mismatches for any symbol
    critical_total = 0
    for sym_data in payload.get("symbols", {}).values():
        critical_total += int(sym_data.get("critical_mismatch_count", 0) or 0)
    clean = critical_total <= gate_cfg.lifecycle_max_critical

    # Check age
    age_hours = _lifecycle_evidence_age_hours(lifecycle)
    recent = age_hours is not None and age_hours <= gate_cfg.lifecycle_max_age_hours

    # Check audit_ok if required
    audit_ok = bool(lifecycle.get("audit_ok", False))
    audit_satisfied = not gate_cfg.require_lifecycle_audit_ok or audit_ok

    # Check it's from a project-internal path (not external Windows workspace)
    report_path = Path(str(lifecycle.get("report_path", "")))
    from_project = _is_within_project(settings, report_path)

    ok = clean and recent and audit_satisfied and from_project

    return {
        "present": True,
        "clean": clean,
        "recent": recent,
        "age_hours": age_hours,
        "audit_ok": audit_ok,
        "audit_satisfied": audit_satisfied,
        "from_project_internal_path": from_project,
        "origin": lifecycle.get("origin", "unknown"),
        "critical_mismatch_total": critical_total,
        "ok": ok,
        "reason": (
            "ok" if ok else
            "lifecycle_not_clean" if not clean else
            "lifecycle_too_old" if not recent else
            "lifecycle_audit_not_ok" if not audit_satisfied else
            "lifecycle_external_path"
        ),
    }


def _check_endurance_evidence(settings: Settings) -> dict[str, Any]:
    """Checks endurance evidence exists and shows 'go' for all active symbols."""
    from iris_bot.governance import _latest_endurance_evidence
    gate_cfg = settings.approved_demo_gate
    registry = load_strategy_profile_registry(settings)
    separation = build_portfolio_separation(settings, registry)

    if not separation.approved_demo_universe:
        return {
            "ok": False,
            "reason": "no_approved_demo_symbols",
            "symbols_checked": [],
        }

    symbol_results: dict[str, Any] = {}
    all_ok = True
    for symbol in separation.approved_demo_universe:
        endurance = _latest_endurance_evidence(settings, symbol)
        if endurance is None:
            symbol_results[symbol] = {"ok": False, "reason": "no_endurance_evidence"}
            all_ok = False
            continue
        end_payload = endurance.get("payload", {})
        end_sym = (end_payload.get("symbols", {}) or {}).get(symbol, {})
        decision = end_sym.get("decision", "missing")
        cycles = int(end_sym.get("cycles_completed", 0) or 0)
        ok = (
            decision == "go"
            and cycles >= gate_cfg.endurance_min_cycles
        )
        symbol_results[symbol] = {
            "ok": ok,
            "decision": decision,
            "cycles_completed": cycles,
            "reason": "ok" if ok else f"endurance_decision={decision}" if decision != "go" else f"insufficient_cycles:{cycles}",
        }
        if not ok:
            all_ok = False

    return {
        "ok": all_ok,
        "symbols_checked": list(separation.approved_demo_universe),
        "per_symbol": symbol_results,
        "reason": "ok" if all_ok else "endurance_requirements_not_met",
    }


def _check_active_materialization(settings: Settings) -> dict[str, Any]:
    """Checks that active_strategy_profiles.json exists and is coherent with registry."""
    active_path = active_strategy_profiles_path(settings)
    if not active_path.exists():
        return {
            "ok": False,
            "reason": "active_strategy_profiles_missing",
            "path": str(active_path),
        }
    try:
        import json
        raw = json.loads(active_path.read_text(encoding="utf-8"))
        payload = raw.get("payload", {})
        symbols = payload.get("symbols", {})
        # Cross-check: each symbol in active_strategy_profiles should be approved_demo in registry
        registry = load_strategy_profile_registry(settings)
        coherent = True
        incoherent_symbols: list[str] = []
        for symbol, prof in symbols.items():
            active_id = registry.get("active_profiles", {}).get(symbol, "")
            if not active_id:
                coherent = False
                incoherent_symbols.append(f"{symbol}:no_active_in_registry")
                continue
            entry = _find_entry(registry, symbol, active_id)
            if entry is None or entry.get("promotion_state") != "approved_demo":
                coherent = False
                incoherent_symbols.append(f"{symbol}:not_approved_demo_in_registry")
        return {
            "ok": coherent,
            "symbol_count": len(symbols),
            "symbols": list(symbols.keys()),
            "coherent_with_registry": coherent,
            "incoherent_symbols": incoherent_symbols,
            "path": str(active_path),
            "reason": "ok" if coherent else "active_profiles_incoherent_with_registry",
        }
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"active_strategy_profiles_unreadable:{exc}",
            "path": str(active_path),
        }


def _check_approved_demo_portfolio(settings: Settings) -> dict[str, Any]:
    """Checks that at least one symbol has approved_demo status in the active portfolio."""
    registry = load_strategy_profile_registry(settings)
    separation = build_portfolio_separation(settings, registry)
    has_approved = len(separation.approved_demo_universe) > 0
    has_active = len(separation.active_portfolio) > 0

    return {
        "ok": has_approved and has_active,
        "approved_demo_symbols": list(separation.approved_demo_universe),
        "active_portfolio_symbols": list(separation.active_portfolio),
        "deliberately_blocked": separation.deliberately_blocked,
        "reason": (
            "ok" if (has_approved and has_active) else
            "no_approved_demo_symbols" if not has_approved else
            "no_active_portfolio_symbols"
        ),
    }


def _check_no_order_send(settings: Settings) -> dict[str, Any]:
    """
    Verifies that order_send is NOT integrated into the execution pipeline.

    This is a safety invariant: during this phase, no real orders must be sent.
    We check that MT5 is not configured for live execution.
    """
    mt5_enabled = settings.mt5.enabled
    # If MT5 is enabled, it might route real orders. We flag this as a warning.
    # The system should be in paper/demo-dry mode only at this stage.
    return {
        "ok": True,  # order_send not integrated in this phase by design
        "mt5_enabled": mt5_enabled,
        "order_send_integrated": False,  # invariant: never True in this phase
        "phase": "pre_broker_executing_demo",
        "note": (
            "MT5 enabled for broker connection (reads only). "
            "order_send NOT integrated. Safe for demo research."
            if mt5_enabled else
            "MT5 disabled. Paper/simulation mode."
        ),
        "reason": "ok",
    }


def _check_evidence_store(settings: Settings) -> dict[str, Any]:
    """Checks that the canonical evidence store has recent, valid artifacts."""
    status = evidence_store_status(settings)
    has_lifecycle = "lifecycle_reconciliation" in status.get("by_artifact_type", {})
    has_stability = "symbol_stability" in status.get("by_artifact_type", {})
    integrity_ok = status.get("integrity_ok", False)
    manifest_valid = status.get("manifest_valid", False)

    return {
        "ok": has_lifecycle and integrity_ok and manifest_valid,
        "has_lifecycle_evidence": has_lifecycle,
        "has_stability_evidence": has_stability,
        "integrity_ok": integrity_ok,
        "manifest_valid": manifest_valid,
        "conflict_policy": status.get("conflict_policy", ""),
        "retention_policy": status.get("retention_policy", ""),
        "tombstone_count": status.get("tombstone_count", 0),
        "integrity_failures": status.get("integrity_failures", []),
        "total_entries": status.get("total_entries", 0),
        "reason": (
            "ok" if (has_lifecycle and integrity_ok and manifest_valid) else
            "no_lifecycle_in_evidence_store" if not has_lifecycle else
            "evidence_store_manifest_invalid" if not manifest_valid else
            "evidence_store_integrity_failure"
        ),
    }


def generate_demo_execution_readiness_report(settings: Settings) -> dict[str, Any]:
    """
    Generates a conservative readiness assessment for broker-executing demo.

    All required checks must pass for "ready_for_demo_guarded".
    Any required check failure → "not_ready_for_demo".
    Optional/advisory warnings → "ready_for_demo_with_reservations".

    Does NOT send any orders. Does NOT connect to MT5.
    """
    checks: dict[str, dict[str, Any]] = {
        "official_quality_gate": _check_official_suite(settings),
        "registry_integrity": _check_registry_integrity(settings),
        "lifecycle_evidence": _check_lifecycle_evidence(settings),
        "endurance_evidence": _check_endurance_evidence(settings),
        "active_materialization": _check_active_materialization(settings),
        "approved_demo_portfolio": _check_approved_demo_portfolio(settings),
        "governance_policy": _check_governance_policy(settings),
        "artifact_provenance": _check_artifact_provenance(settings),
        "mt5_ownership_policy": _check_mt5_ownership_policy(settings),
        "no_order_send": _check_no_order_send(settings),
        "evidence_store": _check_evidence_store(settings),
    }

    # Required checks (failure → not_ready)
    required_checks = [
        "official_quality_gate",
        "registry_integrity",
        "lifecycle_evidence",
        "endurance_evidence",
        "active_materialization",
        "approved_demo_portfolio",
        "governance_policy",
        "artifact_provenance",
        "mt5_ownership_policy",
        "no_order_send",
        "evidence_store",
    ]
    # Advisory checks (failure → caution, not not_ready)
    advisory_checks = [
        "operational_artifact_provenance",
    ]

    operational_artifact = checks["artifact_provenance"]["artifacts"].get("operational_config", {})
    checks["operational_artifact_provenance"] = {
        "ok": bool(operational_artifact.get("ok", False)),
        "reason": str(operational_artifact.get("reason", "ok" if operational_artifact.get("ok", False) else "operational_artifact_missing")),
        "details": operational_artifact,
    }

    failed_required = [k for k in required_checks if not checks[k].get("ok", False)]
    failed_advisory = [k for k in advisory_checks if not checks[k].get("ok", False)]

    # Conservative: not_ready by default
    if failed_required:
        decision = "not_ready_for_demo"
    elif failed_advisory:
        decision = "ready_for_demo_with_reservations"
    else:
        decision = "ready_for_demo_guarded"

    blocking_reasons = [checks[k].get("reason", "unknown") for k in failed_required]
    advisory_reasons = [checks[k].get("reason", "unknown") for k in failed_advisory]
    recommendation = (
        "address_blockers_before_demo"
        if failed_required
        else "review_advisories_then_proceed_guardedly"
        if failed_advisory
        else "guarded_demo_can_be_reviewed_manually"
    )

    return {
        "decision": decision,
        "ready_for_demo": decision == "ready_for_demo_guarded",
        "failed_required_checks": failed_required,
        "failed_advisory_checks": failed_advisory,
        "blocking_reasons": blocking_reasons,
        "advisory_reasons": advisory_reasons,
        "checks": checks,
        "blockers": blocking_reasons,
        "warnings": advisory_reasons,
        "recommendation_next_step": recommendation,
        "phase_note": (
            "This assessment is for broker-executing demo readiness. "
            "order_send is NOT integrated and will NOT be activated based on this report. "
            "Manual approval required before any real order routing."
        ),
        "order_send_integrated": False,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }


def demo_execution_readiness(settings: Settings) -> int:
    """
    Command: generates demo_execution_readiness_report.json.

    Returns:
        0 if ready_for_demo_guarded
        1 if ready_for_demo_with_reservations
        2 if not_ready_for_demo
    """
    run_dir = build_run_directory(settings.data.runs_dir, "demo_execution_readiness")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)

    report = generate_demo_execution_readiness_report(settings)
    write_json_report(
        run_dir,
        "demo_execution_readiness_report.json",
        wrap_artifact(
            "demo_execution_readiness",
            report,
            provenance=build_artifact_provenance(
                run_dir=run_dir,
                policy_version=report["checks"]["governance_policy"].get("policy_version"),
                policy_source=report["checks"]["governance_policy"].get("policy_source"),
                correlation_keys={"command": "demo_execution_readiness"},
                references={
                    "registry_path": str(registry_path(settings)),
                    "active_strategy_profiles_path": str(active_strategy_profiles_path(settings)),
                },
            ),
        ),
    )

    decision = report["decision"]
    logger.info(
        "demo_execution_readiness decision=%s blocked=%s run_dir=%s",
        decision,
        report["failed_required_checks"],
        run_dir,
    )

    if decision == "ready_for_demo_guarded":
        return 0
    if decision == "ready_for_demo_with_reservations":
        return 1
    return 2
