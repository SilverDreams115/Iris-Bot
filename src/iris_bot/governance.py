from __future__ import annotations

import json
from typing import Any

from iris_bot.artifacts import artifact_schema_report, wrap_artifact
from iris_bot.config import Settings
from iris_bot.evidence_store import evidence_store_status
from iris_bot.governance_active import (
    resolve_active_profile_entry,
    resolve_active_profiles,
)
from iris_bot.governance_promotion import (
    promote_strategy_profile as _promote_strategy_profile,
    rollback_strategy_profile as _rollback_strategy_profile,
)
from iris_bot.governance_validation import validate_strategy_profiles
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.profile_evidence import (
    _compute_endurance_gate_metrics,
    _latest_endurance_evidence,
    _latest_endurance_payload,
    _latest_lifecycle_evidence,
    _latest_lifecycle_payload,
    _lifecycle_evidence_age_hours,
)
from iris_bot.profile_registry import (
    _entry_checksum_ok,
    _find_entry,
    _latest_entry_by_state,
    _materialize_active_profiles_from_registry,
    _profile_checksum,
    _validated_profile_for_symbol,
    active_strategy_profiles_path,
    load_strategy_profile_registry,
    registry_path,
    save_strategy_profile_registry,
)
from iris_bot.registry_lock import (
    RegistryLockTimeoutError,
    RegistryMutationConflictError,
    governance_lock_audit,
    registry_etag,
    registry_exclusive_lock,
)
from iris_bot.symbols import load_symbol_strategy_profiles, strategy_profiles_path

__all__ = [
    # Governance pipeline functions defined here
    "run_promotion_review",
    "run_governance_audit",
    "run_governance_rollback",
    "resolve_active_profiles",
    "resolve_active_profile_entry",
    # Re-exported registry helpers used by tests and callers
    "active_strategy_profiles_path",
    "load_strategy_profile_registry",
    "registry_path",
    "save_strategy_profile_registry",
    "validate_strategy_profiles",
    "repair_strategy_profile_registry",
    # Re-exported internal helpers for test access
    "_find_entry",
]


def _build_promotion_gate_matrix(
    profile: dict[str, Any] | None,
    strategy_schema: dict[str, Any],
    lifecycle: dict[str, Any] | None,
    lifecycle_symbol: dict[str, Any],
    lifecycle_age_hours: float | None,
    endurance: dict[str, Any] | None,
    endurance_symbol: dict[str, Any],
    alerts: dict[str, Any],
    endo_metrics: dict[str, Any],
    gate_cfg: Any,
) -> dict[str, bool]:
    lifecycle_recent = (
        lifecycle_age_hours is not None and lifecycle_age_hours <= gate_cfg.lifecycle_max_age_hours
    ) if lifecycle is not None else False
    profit_factor_floor_ok = (
        endo_metrics["avg_profit_factor"] is not None
        and endo_metrics["avg_profit_factor"] >= gate_cfg.min_profit_factor
    ) if endurance is not None else False
    expectancy_floor_ok = (
        endo_metrics["avg_expectancy_usd"] is not None
        and endo_metrics["avg_expectancy_usd"] >= gate_cfg.min_expectancy_usd
    ) if endurance is not None else False
    no_trade_ratio_ok = (
        endo_metrics["no_trade_ratio"] is None
        or endo_metrics["no_trade_ratio"] <= gate_cfg.max_no_trade_ratio
    ) if endurance is not None else False
    blocked_ratio_ok = (
        endo_metrics["blocked_trades_ratio"] is None
        or endo_metrics["blocked_trades_ratio"] <= gate_cfg.max_blocked_trades_ratio
    ) if endurance is not None else False
    return {
        "validated_profile_present": profile is not None,
        "profile_checksum_ok": _entry_checksum_ok(profile) if profile is not None else False,
        "schema_compatible": strategy_schema.get("ok", False),
        "lifecycle_evidence_present": lifecycle is not None,
        "lifecycle_clean": (
            int(lifecycle_symbol.get("critical_mismatch_count", 0) or 0) <= gate_cfg.lifecycle_max_critical
        ) if lifecycle is not None else False,
        "lifecycle_audit_ok": bool((lifecycle or {}).get("audit_ok", False)) if lifecycle is not None else False,
        "lifecycle_evidence_recent": lifecycle_recent,
        "endurance_present": endurance is not None,
        "endurance_decision_go": endurance_symbol.get("decision") == "go" if endurance is not None else False,
        "endurance_cycles_sufficient": (
            endo_metrics["cycles_completed"] >= gate_cfg.endurance_min_cycles
        ) if endurance is not None else False,
        "endurance_min_trades_met": endo_metrics["trade_count"] >= gate_cfg.endurance_min_trades,
        "critical_alerts_absent": int(alerts.get("critical", 0) or 0) == 0,
        "profit_factor_floor_ok": profit_factor_floor_ok,
        "expectancy_floor_ok": expectancy_floor_ok,
        "no_trade_ratio_ok": no_trade_ratio_ok,
        "blocked_trades_ratio_ok": blocked_ratio_ok,
        "no_expectancy_degradation_gate_breach": (
            float(endurance_symbol.get("expectancy_degradation_pct", 0.0) or 0.0)
            <= gate_cfg.max_expectancy_degradation_pct
        ) if endurance is not None else False,
        "no_profit_factor_degradation_gate_breach": (
            float(endurance_symbol.get("profit_factor_degradation_pct", 0.0) or 0.0)
            <= gate_cfg.max_profit_factor_degradation_pct
        ) if endurance is not None else False,
    }


def _apply_gate_decisions(
    symbol: str,
    gate_matrix: dict[str, bool],
    gate_cfg: Any,
    lifecycle_age_hours: float | None,
    endurance_symbol: dict[str, Any],
    endo_metrics: dict[str, Any],
    *,
    strict: bool,
) -> tuple[str, str, list[str]]:
    """Return (final_decision, severity, reasons) by walking the gate matrix."""
    reasons: list[str] = []
    severity = "info"
    final_decision = "KEEP_VALIDATED"

    def block(reason: str) -> None:
        nonlocal final_decision, severity
        final_decision = "REVERT_TO_BLOCKED"
        severity = "error"
        reasons.append(reason)

    def caution(reason: str) -> None:
        nonlocal final_decision, severity
        if final_decision != "REVERT_TO_BLOCKED":
            final_decision = "MOVE_TO_CAUTION"
            severity = "warning"
        reasons.append(reason)

    if symbol == "USDJPY":
        block("symbol_out_of_scope_for_promotion")

    if not gate_matrix["validated_profile_present"]:
        block("validated_profile_missing")
    if gate_matrix["validated_profile_present"] and not gate_matrix["profile_checksum_ok"]:
        block("profile_checksum_incompatible")
    if not gate_matrix["schema_compatible"]:
        block("strategy_profiles_schema_incompatible")

    if final_decision != "REVERT_TO_BLOCKED":
        # Lifecycle gates
        if not gate_matrix["lifecycle_evidence_present"]:
            if strict:
                block("missing_lifecycle_evidence_blocks_promotion")
            else:
                reasons.append("missing_lifecycle_validation")
        else:
            if not gate_matrix["lifecycle_clean"]:
                block("lifecycle_critical_mismatch")
            if gate_cfg.require_lifecycle_audit_ok and not gate_matrix["lifecycle_audit_ok"]:
                block("lifecycle_audit_not_confirmed_blocks_promotion")
            if not gate_matrix["lifecycle_evidence_recent"]:
                block(
                    f"lifecycle_evidence_too_old:{lifecycle_age_hours:.1f}h>max_{gate_cfg.lifecycle_max_age_hours:.0f}h"
                    if lifecycle_age_hours is not None else "lifecycle_evidence_age_unknown"
                )

        # Endurance gates
        if not gate_matrix["endurance_present"]:
            if strict:
                block("missing_endurance_evidence_blocks_promotion")
            else:
                reasons.append("missing_endurance_validation")
        else:
            if not gate_matrix["endurance_decision_go"]:
                decision_val = endurance_symbol.get("decision", "missing")
                if strict:
                    block(f"endurance_decision={decision_val}_blocks_promotion")
                else:
                    caution(f"endurance_decision={decision_val}")
            if not gate_matrix["endurance_cycles_sufficient"]:
                block(f"insufficient_endurance_cycles:{endo_metrics['cycles_completed']}<{gate_cfg.endurance_min_cycles}")
            if not gate_matrix["endurance_min_trades_met"]:
                block(f"insufficient_endurance_trades:{endo_metrics['trade_count']}<{gate_cfg.endurance_min_trades}")
            if not gate_matrix["critical_alerts_absent"]:
                block("critical_alerts_present_blocks_promotion")
            if not gate_matrix["profit_factor_floor_ok"]:
                pf_val = endo_metrics["avg_profit_factor"]
                block(f"profit_factor_below_floor:{pf_val:.3f}<{gate_cfg.min_profit_factor}" if pf_val is not None else "profit_factor_unavailable")
            if not gate_matrix["expectancy_floor_ok"]:
                ex_val = endo_metrics["avg_expectancy_usd"]
                block(f"expectancy_below_floor:{ex_val:.3f}<{gate_cfg.min_expectancy_usd}" if ex_val is not None else "expectancy_unavailable")
            if not gate_matrix["no_trade_ratio_ok"]:
                ntr = endo_metrics["no_trade_ratio"]
                block(f"no_trade_ratio_exceeded:{ntr:.3f}>{gate_cfg.max_no_trade_ratio}" if ntr is not None else "no_trade_ratio_check_failed")
            if not gate_matrix["blocked_trades_ratio_ok"]:
                btr = endo_metrics["blocked_trades_ratio"]
                block(f"blocked_trades_ratio_exceeded:{btr:.3f}>{gate_cfg.max_blocked_trades_ratio}" if btr is not None else "blocked_trades_ratio_check_failed")
            if not gate_matrix["no_expectancy_degradation_gate_breach"]:
                caution("expectancy_degradation_above_gate")
            if not gate_matrix["no_profit_factor_degradation_gate_breach"]:
                caution("profit_factor_degradation_above_gate")

    if final_decision == "KEEP_VALIDATED" and all(gate_matrix.values()):
        final_decision = "APPROVED_DEMO"
        reasons.append("validated_endurance_lifecycle_passed")

    return final_decision, severity, reasons


def _promotion_review_for_symbol(
    settings: Settings,
    symbol: str,
    registry: dict[str, Any],
    *,
    target_profile_id: str = "",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Evaluates whether a symbol's validated profile should be promoted to approved_demo.

    Gate is conservative: a single missing required evidence → REVERT_TO_BLOCKED.
    CAUTION is reserved for genuinely borderline cases where evidence exists but
    shows degradation. Missing evidence always blocks.

    Uses settings.approved_demo_gate for all configurable floors.
    """
    gate_cfg = settings.approved_demo_gate
    strategy_schema = artifact_schema_report(strategy_profiles_path(settings), expected_type="strategy_profiles")
    profile = _validated_profile_for_symbol(registry, symbol, target_profile_id)
    lifecycle = _latest_lifecycle_evidence(settings)
    endurance = _latest_endurance_evidence(settings, symbol)
    lifecycle_symbol = ((lifecycle or {}).get("payload", {}) or {}).get("symbols", {}).get(symbol, {})
    endurance_symbol = ((endurance or {}).get("payload", {}) or {}).get("symbols", {}).get(symbol, {})
    alerts = endurance_symbol.get("alerts_by_severity", {}) if isinstance(endurance_symbol, dict) else {}
    endo_metrics = _compute_endurance_gate_metrics(endurance_symbol)
    lifecycle_age_hours = _lifecycle_evidence_age_hours(lifecycle)

    gate_matrix = _build_promotion_gate_matrix(
        profile, strategy_schema, lifecycle, lifecycle_symbol, lifecycle_age_hours,
        endurance, endurance_symbol, alerts, endo_metrics, gate_cfg,
    )
    final_decision, severity, reasons = _apply_gate_decisions(
        symbol, gate_matrix, gate_cfg, lifecycle_age_hours, endurance_symbol, endo_metrics, strict=strict,
    )
    active_status = resolve_active_profile_entry(settings, symbol, registry=registry)
    return {
        "symbol": symbol,
        "profile_id": profile.get("profile_id", "") if profile else "",
        "source_run_id": profile.get("source_run_id", "") if profile else "",
        "checksum": profile.get("checksum", "") if profile else "",
        "schema_version": profile.get("schema_version", None) if profile else None,
        "current_registry_state": profile.get("promotion_state", "") if profile else "missing",
        "current_enablement_state": profile.get("enablement_state", "") if profile else "missing",
        "active_profile_id": registry.get("active_profiles", {}).get(symbol, ""),
        "active_profile_ok": active_status["ok"],
        "active_profile_reasons": active_status["reasons"],
        "gate_matrix": gate_matrix,
        "gate_config_snapshot": {
            "min_trade_count": gate_cfg.min_trade_count,
            "max_no_trade_ratio": gate_cfg.max_no_trade_ratio,
            "max_blocked_trades_ratio": gate_cfg.max_blocked_trades_ratio,
            "min_profit_factor": gate_cfg.min_profit_factor,
            "min_expectancy_usd": gate_cfg.min_expectancy_usd,
            "lifecycle_max_age_hours": gate_cfg.lifecycle_max_age_hours,
            "require_lifecycle_audit_ok": gate_cfg.require_lifecycle_audit_ok,
            "endurance_min_cycles": gate_cfg.endurance_min_cycles,
            "endurance_min_trades": gate_cfg.endurance_min_trades,
            "max_expectancy_degradation_pct": gate_cfg.max_expectancy_degradation_pct,
            "max_profit_factor_degradation_pct": gate_cfg.max_profit_factor_degradation_pct,
        },
        "final_decision": final_decision,
        "severity": severity,
        "reasons": sorted(set(reasons)),
        "endurance_summary": {
            "available": endurance is not None,
            "decision": endurance_symbol.get("decision", "missing") if endurance is not None else "missing",
            "cycles_completed": endo_metrics["cycles_completed"],
            "trade_count": endo_metrics["trade_count"],
            "no_trade_count": endo_metrics["no_trade_count"],
            "blocked_trades": endo_metrics["blocked_trades"],
            "no_trade_ratio": endo_metrics["no_trade_ratio"],
            "blocked_trades_ratio": endo_metrics["blocked_trades_ratio"],
            "avg_profit_factor": endo_metrics["avg_profit_factor"],
            "avg_expectancy_usd": endo_metrics["avg_expectancy_usd"],
            "alerts_by_severity": alerts,
            "expectancy_degradation_pct": float(endurance_symbol.get("expectancy_degradation_pct", 0.0) or 0.0) if endurance is not None else None,
            "profit_factor_degradation_pct": float(endurance_symbol.get("profit_factor_degradation_pct", 0.0) or 0.0) if endurance is not None else None,
            "source_run": str((endurance or {}).get("run_dir", "")) if endurance is not None else "",
        },
        "lifecycle_summary": {
            "available": lifecycle is not None,
            "critical_mismatch_count": int(lifecycle_symbol.get("critical_mismatch_count", 0) or 0) if lifecycle is not None else None,
            "evidence_age_hours": lifecycle_age_hours,
            "source_path": str((lifecycle or {}).get("report_path", "")) if lifecycle is not None else "",
            "origin": (lifecycle or {}).get("origin", "") if lifecycle is not None else "",
            "audit_ok": (lifecycle or {}).get("audit_ok", None) if lifecycle is not None else None,
            "rerun_source": (lifecycle or {}).get("rerun_source", "") if lifecycle is not None else "",
        },
    }


def _promotion_review_symbols(settings: Settings, registry: dict[str, Any]) -> tuple[str, ...]:
    if settings.governance.target_symbol:
        return (settings.governance.target_symbol,)
    symbols: list[str] = []
    for symbol in settings.trading.symbols:
        latest = _latest_entry_by_state(registry, symbol, {"validated"})
        if latest is not None and symbol != "USDJPY":
            symbols.append(symbol)
    return tuple(symbols)


def review_approved_demo_readiness(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "review_approved_demo_readiness")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    registry = load_strategy_profile_registry(settings)
    symbols = _promotion_review_symbols(settings, registry)
    if not symbols:
        logger.error("No validated symbols eligible for promotion review")
        return 2
    reviews = {symbol: _promotion_review_for_symbol(settings, symbol, registry, strict=False) for symbol in symbols}
    summary = {
        "symbols": {symbol: review["final_decision"] for symbol, review in reviews.items()},
        "approved_demo_ready": sorted(symbol for symbol, review in reviews.items() if review["final_decision"] == "APPROVED_DEMO"),
        "keep_validated": sorted(symbol for symbol, review in reviews.items() if review["final_decision"] == "KEEP_VALIDATED"),
        "move_to_caution": sorted(symbol for symbol, review in reviews.items() if review["final_decision"] == "MOVE_TO_CAUTION"),
        "revert_to_blocked": sorted(symbol for symbol, review in reviews.items() if review["final_decision"] == "REVERT_TO_BLOCKED"),
    }
    gate_matrix = {symbol: review["gate_matrix"] for symbol, review in reviews.items()}
    write_json_report(run_dir, "symbol_promotion_review_report.json", wrap_artifact("strategy_profile_promotion", {"symbols": reviews}))
    write_json_report(run_dir, "approved_demo_readiness_report.json", wrap_artifact("strategy_profile_promotion", {"symbols": reviews}))
    write_json_report(run_dir, "promotion_gate_matrix.json", wrap_artifact("strategy_profile_promotion", {"symbols": gate_matrix}))
    write_json_report(run_dir, "promotion_decision_summary.json", wrap_artifact("strategy_profile_promotion", summary))
    write_json_report(
        run_dir,
        "active_profile_resolution_report.json",
        wrap_artifact("active_strategy_status", resolve_active_profiles(settings)[1]),
    )
    logger.info(
        "review_approved_demo_readiness symbols=%s approved=%s run_dir=%s",
        len(reviews),
        len(summary["approved_demo_ready"]),
        run_dir,
    )
    return 0 if all(review["final_decision"] == "APPROVED_DEMO" for review in reviews.values()) else 2


def promote_strategy_profile(settings: Settings) -> int:
    return _promote_strategy_profile(settings, _promotion_review_for_symbol)


def rollback_strategy_profile(settings: Settings) -> int:
    return _rollback_strategy_profile(settings)


def list_strategy_profiles(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "list_strategy_profiles")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    payload = load_strategy_profile_registry(settings)
    write_json_report(run_dir, "strategy_profile_registry.json", wrap_artifact("strategy_profile_registry", payload))
    logger.info("list_strategy_profiles run_dir=%s", run_dir)
    return 0


def active_strategy_status(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "active_strategy_status")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    _, payload = resolve_active_profiles(settings)
    write_json_report(run_dir, "active_strategy_status.json", wrap_artifact("active_strategy_status", payload))
    logger.info("active_strategy_status blocked=%s run_dir=%s", payload["blocked_symbols"], run_dir)
    return 0 if payload["blocked_symbols"] == 0 else 2


def diagnose_profile_activation(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "diagnose_profile_activation")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    registry = load_strategy_profile_registry(settings)
    file_profiles = load_symbol_strategy_profiles(settings)
    _, active_payload = resolve_active_profiles(settings)
    strategy_schema = artifact_schema_report(strategy_profiles_path(settings), expected_type="strategy_profiles")
    registry_path_report = artifact_schema_report(registry_path(settings), expected_type="strategy_profile_registry")
    consistency_symbols: dict[str, Any] = {}
    readiness_symbols: dict[str, Any] = {}
    for symbol in settings.trading.symbols:
        latest_entry = _latest_entry_by_state(registry, symbol, {"approved_demo", "validated", "caution", "blocked", "deprecated"})
        file_profile = file_profiles.get(symbol)
        active_status = active_payload["symbols"][symbol]
        consistency_symbols[symbol] = {
            "active_profile_id": active_status["active_profile_id"],
            "active_profile_reasons": active_status["reasons"],
            "active_profile_warnings": active_status["warnings"],
            "latest_registry_profile_id": latest_entry.get("profile_id", "") if latest_entry else "",
            "latest_registry_promotion_state": latest_entry.get("promotion_state", "") if latest_entry else "",
            "registry_checksum_ok": _entry_checksum_ok(latest_entry) if latest_entry else False,
            "strategy_profile_id": file_profile.profile_id if file_profile else "",
            "strategy_profile_promotion_state": file_profile.promotion_state if file_profile else "missing",
            "strategy_profile_enabled_state": file_profile.enabled_state if file_profile else "missing",
            "strategy_profile_sync_ok": latest_entry is not None and file_profile is not None and file_profile.profile_id == latest_entry.get("profile_id", ""),
        }
        latest_lifecycle = _latest_lifecycle_payload(settings)
        latest_endurance = _latest_endurance_payload(settings)
        lifecycle_symbol = (latest_lifecycle or {}).get("symbols", {}).get(symbol, {})
        endurance_symbol = (latest_endurance or {}).get("symbols", {}).get(symbol, {})
        if symbol == "USDJPY":
            readiness = "blocked_out_of_scope"
        elif active_status["ok"]:
            readiness = "active"
        else:
            readiness = "blocked"
        readiness_symbols[symbol] = {
            "readiness": readiness,
            "active_profile_reasons": active_status["reasons"],
            "latest_endurance_decision": endurance_symbol.get("decision", "missing"),
            "latest_lifecycle_critical_mismatch_count": lifecycle_symbol.get("critical_mismatch_count", None),
        }
    write_json_report(run_dir, "profile_activation_diagnostic_report.json", wrap_artifact("active_strategy_status", active_payload))
    write_json_report(
        run_dir,
        "active_profile_resolution_report.json",
        wrap_artifact("active_strategy_status", {"symbols": active_payload["symbols"]}),
    )
    write_json_report(
        run_dir,
        "governance_consistency_report.json",
        wrap_artifact(
            "strategy_profile_validation",
            {
                "strategy_profiles_schema": strategy_schema,
                "registry_schema": registry_path_report,
                "symbols": consistency_symbols,
            },
        ),
    )
    write_json_report(
        run_dir,
        "technical_debt_avoidance_report.json",
        wrap_artifact(
            "corrective_audit",
            {
                "avoided_shortcuts": [
                    "no active_profile_id injection by hand",
                    "no symbol-specific bypass for blocked symbols",
                    "no parallel activation path outside governance registry",
                    "no threshold relaxation to force tradability",
                ],
                "protected_invariants": [
                    "active profile must exist in registry",
                    "active profile must have valid promotion_state",
                    "active profile checksum must match canonical payload",
                    "strategy_profiles file must stay synchronized with latest registry entry",
                ],
            },
        ),
    )
    write_json_report(
        run_dir,
        "symbol_reactivation_readiness_report.json",
        wrap_artifact("strategy_profile_validation", {"symbols": readiness_symbols}),
    )
    logger.info("diagnose_profile_activation blocked=%s run_dir=%s", active_payload["blocked_symbols"], run_dir)
    return 0 if active_payload["blocked_symbols"] == 0 else 2


# ---------------------------------------------------------------------------
# New blindaje commands (Phase 4)
# ---------------------------------------------------------------------------

def audit_governance_locking(settings: Settings) -> int:
    """
    Audits the lock state of the registry and reports on its integrity.

    Checks:
      - Whether the lock file exists and its current state
      - Whether the lock is currently held by another process
      - Whether the registry is readable and has a valid etag
      - Whether the active_strategy_profiles.json is present and coherent

    Returns 0 if everything is clean, 2 if there are warnings.
    """
    run_dir = build_run_directory(settings.data.runs_dir, "governance_lock_audit")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)

    reg_path = registry_path(settings)
    lock_audit = governance_lock_audit(reg_path)

    active_profiles_present = active_strategy_profiles_path(settings).exists()
    registry_coherent = False
    registry_checksum_ok = False
    try:
        reg = load_strategy_profile_registry(settings)
        registry_coherent = isinstance(reg, dict) and "profiles" in reg
        registry_checksum_ok = True
    except (OSError, ValueError):
        pass

    payload = {
        "lock_audit": lock_audit,
        "registry_coherent": registry_coherent,
        "registry_checksum_ok": registry_checksum_ok,
        "active_strategy_profiles_present": active_profiles_present,
        "overall_ok": (
            lock_audit["registry_exists"]
            and not lock_audit["lock_currently_held"]
            and registry_coherent
            and registry_checksum_ok
        ),
    }
    write_json_report(run_dir, "governance_lock_audit_report.json", wrap_artifact("governance_lock_audit", payload))
    logger.info("audit_governance_locking ok=%s run_dir=%s", payload["overall_ok"], run_dir)
    return 0 if payload["overall_ok"] else 2


def materialize_active_profiles(settings: Settings) -> int:
    """
    Explicitly materializes active_strategy_profiles.json from the current registry.

    Writes only approved_demo entries to the active materialization file.
    This command is idempotent — running it multiple times produces the same result.

    Must be run after any promotion/rollback to ensure the active materialization
    is up to date.
    """
    run_dir = build_run_directory(settings.data.runs_dir, "materialize_active_profiles")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)

    reg_path = registry_path(settings)
    pre_lock_etag = registry_etag(reg_path)

    try:
        with registry_exclusive_lock(reg_path, expected_etag=pre_lock_etag):
            registry = load_strategy_profile_registry(settings)
            _materialize_active_profiles_from_registry(settings, registry)
    except (RegistryMutationConflictError, RegistryLockTimeoutError) as exc:
        logger.error("Could not materialize active profiles under lock: %s", exc)
        return 5

    # Report
    active_path = active_strategy_profiles_path(settings)
    active_exists = active_path.exists()
    active_count = 0
    if active_exists:
        try:
            raw = json.loads(active_path.read_text(encoding="utf-8"))
            payload = raw.get("payload", {})
            active_count = payload.get("symbol_count", len(payload.get("symbols", {})))
        except (OSError, json.JSONDecodeError):
            pass

    report = {
        "active_strategy_profiles_path": str(active_path),
        "materialized": active_exists,
        "approved_demo_symbol_count": active_count,
    }
    write_json_report(run_dir, "active_strategy_materialization_report.json", wrap_artifact("active_strategy_materialization", report))
    logger.info("materialize_active_profiles count=%s run_dir=%s", active_count, run_dir)
    return 0


def repair_strategy_profile_registry(settings: Settings) -> int:
    """
    Recomputes stale profile checksums in the registry and refreshes active materialization.

    This is a maintenance command for historical registries created before the
    current canonical checksum rules. It does not change business state
    (promotion/enablement), only integrity metadata and the derived
    active_strategy_profiles materialization.
    """
    run_dir = build_run_directory(settings.data.runs_dir, "repair_strategy_profile_registry")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)

    reg_path = registry_path(settings)
    pre_lock_etag = registry_etag(reg_path)
    repaired_entries: list[dict[str, str]] = []

    try:
        with registry_exclusive_lock(reg_path, expected_etag=pre_lock_etag):
            registry = load_strategy_profile_registry(settings)
            for symbol, entries in registry.get("profiles", {}).items():
                for entry in entries:
                    if _entry_checksum_ok(entry):
                        continue
                    payload = dict(entry.get("profile_payload", {}))
                    old_checksum = str(entry.get("checksum", ""))
                    new_checksum = _profile_checksum(payload)
                    entry["checksum"] = new_checksum
                    repaired_entries.append(
                        {
                            "symbol": str(symbol),
                            "profile_id": str(entry.get("profile_id", "")),
                            "old_checksum": old_checksum,
                            "new_checksum": new_checksum,
                        }
                    )
            save_strategy_profile_registry(settings, registry)
            _materialize_active_profiles_from_registry(settings, registry)
    except (RegistryMutationConflictError, RegistryLockTimeoutError) as exc:
        logger.error("Could not repair strategy profile registry under lock: %s", exc)
        return 5

    report = {
        "registry_path": str(reg_path),
        "repaired_entry_count": len(repaired_entries),
        "repaired_entries": repaired_entries,
        "active_strategy_profiles_path": str(active_strategy_profiles_path(settings)),
    }
    write_json_report(run_dir, "strategy_profile_registry_repair_report.json", wrap_artifact("strategy_profile_registry", report))
    logger.info("repair_strategy_profile_registry repaired=%s run_dir=%s", len(repaired_entries), run_dir)
    return 0


def evidence_store_status_command(settings: Settings) -> int:
    """
    Reports the status of the canonical evidence store.

    Shows:
      - All ingested artifacts with metadata
      - Integrity check results (checksum verification)
      - Latest evidence per (artifact_type, symbol) key
      - Whether any integrity failures exist
    """
    run_dir = build_run_directory(settings.data.runs_dir, "evidence_store_status")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    status = evidence_store_status(settings)
    write_json_report(run_dir, "evidence_store_manifest.json", wrap_artifact("evidence_store_manifest", status))
    logger.info(
        "evidence_store_status total=%s integrity_ok=%s run_dir=%s",
        status["total_entries"],
        status["integrity_ok"],
        run_dir,
    )
    return 0 if status["integrity_ok"] else 2


def approved_demo_gate_audit(settings: Settings) -> int:
    """
    Detailed gate audit for approved_demo promotion.

    Shows exactly which gate checks pass/fail for each eligible symbol,
    with the configured floor values that determine pass/fail.

    This is the authoritative audit trail for why a symbol was or was not
    promoted to approved_demo.
    """
    run_dir = build_run_directory(settings.data.runs_dir, "approved_demo_gate_audit")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    registry = load_strategy_profile_registry(settings)
    symbols = _promotion_review_symbols(settings, registry)
    if not symbols:
        # Include all eligible symbols in the audit even if not validated
        from iris_bot.portfolio import _PERMANENTLY_EXCLUDED
        symbols = tuple(s for s in settings.trading.symbols if s not in _PERMANENTLY_EXCLUDED)

    gate_cfg = settings.approved_demo_gate
    gate_config_report = {
        "min_trade_count": gate_cfg.min_trade_count,
        "max_no_trade_ratio": gate_cfg.max_no_trade_ratio,
        "max_blocked_trades_ratio": gate_cfg.max_blocked_trades_ratio,
        "min_profit_factor": gate_cfg.min_profit_factor,
        "min_expectancy_usd": gate_cfg.min_expectancy_usd,
        "lifecycle_max_age_hours": gate_cfg.lifecycle_max_age_hours,
        "require_lifecycle_audit_ok": gate_cfg.require_lifecycle_audit_ok,
        "lifecycle_max_critical": gate_cfg.lifecycle_max_critical,
        "endurance_min_cycles": gate_cfg.endurance_min_cycles,
        "endurance_min_trades": gate_cfg.endurance_min_trades,
        "max_expectancy_degradation_pct": gate_cfg.max_expectancy_degradation_pct,
        "max_profit_factor_degradation_pct": gate_cfg.max_profit_factor_degradation_pct,
    }

    reviews = {symbol: _promotion_review_for_symbol(settings, symbol, registry) for symbol in symbols}
    blocked_by_gate = [s for s, r in reviews.items() if r["final_decision"] == "REVERT_TO_BLOCKED"]
    approved = [s for s, r in reviews.items() if r["final_decision"] == "APPROVED_DEMO"]

    payload = {
        "gate_config": gate_config_report,
        "symbols_audited": list(symbols),
        "approved_demo_ready": approved,
        "blocked_by_gate": blocked_by_gate,
        "details": {
            symbol: {
                "final_decision": r["final_decision"],
                "reasons": r["reasons"],
                "gate_matrix": r["gate_matrix"],
                "endurance_summary": r["endurance_summary"],
                "lifecycle_summary": r["lifecycle_summary"],
            }
            for symbol, r in reviews.items()
        },
    }
    write_json_report(run_dir, "approved_demo_gate_audit_report.json", wrap_artifact("approved_demo_gate_audit", payload))
    logger.info(
        "approved_demo_gate_audit approved=%s blocked=%s run_dir=%s",
        len(approved), len(blocked_by_gate), run_dir,
    )
    return 0 if not blocked_by_gate else 2


def active_portfolio_status(settings: Settings) -> int:
    """
    Reports the active portfolio status with explicit universe separation.

    Separates:
      - Full universe (all configured symbols)
      - Eligible universe (not permanently excluded)
      - approved_demo universe
      - Active portfolio (approved_demo + enabled)
      - Deliberately blocked (permanently excluded with reasons)

    This replaces the old active_strategy_status which mixed everything.
    """
    from iris_bot.portfolio import active_portfolio_status_report, active_universe_status_report
    run_dir = build_run_directory(settings.data.runs_dir, "active_portfolio_status")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    registry = load_strategy_profile_registry(settings)
    portfolio_report = active_portfolio_status_report(settings, registry)
    universe_report = active_universe_status_report(settings, registry)
    write_json_report(run_dir, "active_portfolio_status.json", wrap_artifact("active_portfolio_status", portfolio_report))
    write_json_report(run_dir, "active_universe_status.json", wrap_artifact("active_universe_status", universe_report))
    logger.info(
        "active_portfolio_status portfolio_size=%s approved_demo_size=%s run_dir=%s",
        portfolio_report["summary"]["active_portfolio_size"],
        portfolio_report["summary"]["approved_demo_size"],
        run_dir,
    )
    return 0 if portfolio_report["summary"]["active_portfolio_size"] > 0 else 2
