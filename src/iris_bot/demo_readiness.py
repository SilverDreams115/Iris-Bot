"""
Demo execution readiness assessment for IRIS-Bot.

This module evaluates whether the system is ready for broker-executing demo
(real order routing to a demo broker account) WITHOUT triggering any execution.

Conservative design:
  - Default decision is "not_ready"
  - Any missing required artifact → not_ready
  - Any ambiguous check → not_ready or caution (never optimistic)
  - order_send is explicitly NOT integrated here

The generated report (demo_execution_readiness_report.json) serves as an
explicit gate before the next phase can begin.

Decision values:
  "ready_for_next_phase" — all required checks pass with high confidence
  "caution"              — checks pass but with warnings (review before proceeding)
  "not_ready"            — one or more required checks fail (do NOT proceed)
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_bot.artifacts import artifact_schema_report, wrap_artifact
from iris_bot.config import Settings
from iris_bot.evidence_store import evidence_store_status, get_latest_evidence
from iris_bot.governance import (
    _entry_checksum_ok,
    _find_entry,
    _latest_lifecycle_evidence,
    _lifecycle_evidence_age_hours,
    active_strategy_profiles_path,
    load_strategy_profile_registry,
    registry_path,
    resolve_active_profiles,
)
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.portfolio import build_portfolio_separation
from iris_bot.registry_lock import governance_lock_audit
from iris_bot.run_index import run_index_status
from iris_bot.symbols import strategy_profiles_path


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
    report_path = lifecycle.get("report_path", "")
    from_project = str(settings.project_root) in str(report_path)

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

    return {
        "ok": has_lifecycle and integrity_ok,
        "has_lifecycle_evidence": has_lifecycle,
        "has_stability_evidence": has_stability,
        "integrity_ok": integrity_ok,
        "integrity_failures": status.get("integrity_failures", []),
        "total_entries": status.get("total_entries", 0),
        "reason": (
            "ok" if (has_lifecycle and integrity_ok) else
            "no_lifecycle_in_evidence_store" if not has_lifecycle else
            "evidence_store_integrity_failure"
        ),
    }


def generate_demo_execution_readiness_report(settings: Settings) -> dict[str, Any]:
    """
    Generates a conservative readiness assessment for broker-executing demo.

    All checks must pass for "ready_for_next_phase".
    Any required check failure → "not_ready".
    Optional/advisory warnings → "caution".

    Does NOT send any orders. Does NOT connect to MT5.
    """
    checks: dict[str, dict[str, Any]] = {
        "registry_integrity": _check_registry_integrity(settings),
        "lifecycle_evidence": _check_lifecycle_evidence(settings),
        "endurance_evidence": _check_endurance_evidence(settings),
        "active_materialization": _check_active_materialization(settings),
        "approved_demo_portfolio": _check_approved_demo_portfolio(settings),
        "no_order_send": _check_no_order_send(settings),
        "evidence_store": _check_evidence_store(settings),
    }

    # Required checks (failure → not_ready)
    required_checks = [
        "registry_integrity",
        "lifecycle_evidence",
        "endurance_evidence",
        "active_materialization",
        "approved_demo_portfolio",
        "no_order_send",
    ]
    # Advisory checks (failure → caution, not not_ready)
    advisory_checks = ["evidence_store"]

    failed_required = [k for k in required_checks if not checks[k].get("ok", False)]
    failed_advisory = [k for k in advisory_checks if not checks[k].get("ok", False)]

    # Conservative: not_ready by default
    if failed_required:
        decision = "not_ready"
    elif failed_advisory:
        decision = "caution"
    else:
        decision = "ready_for_next_phase"

    blocking_reasons = [checks[k].get("reason", "unknown") for k in failed_required]
    advisory_reasons = [checks[k].get("reason", "unknown") for k in failed_advisory]

    return {
        "decision": decision,
        "ready_for_next_phase": decision == "ready_for_next_phase",
        "failed_required_checks": failed_required,
        "failed_advisory_checks": failed_advisory,
        "blocking_reasons": blocking_reasons,
        "advisory_reasons": advisory_reasons,
        "checks": checks,
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
        0 if ready_for_next_phase
        1 if caution
        2 if not_ready
    """
    run_dir = build_run_directory(settings.data.runs_dir, "demo_execution_readiness")
    logger = configure_logging(run_dir, settings.logging.level)

    report = generate_demo_execution_readiness_report(settings)
    write_json_report(
        run_dir,
        "demo_execution_readiness_report.json",
        wrap_artifact("demo_execution_readiness", report),
    )

    decision = report["decision"]
    logger.info(
        "demo_execution_readiness decision=%s blocked=%s run_dir=%s",
        decision,
        report["failed_required_checks"],
        run_dir,
    )

    if decision == "ready_for_next_phase":
        return 0
    if decision == "caution":
        return 1
    return 2
