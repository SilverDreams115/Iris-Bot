from __future__ import annotations

from typing import Any, Callable

from iris_bot.artifacts import wrap_artifact
from iris_bot.config import Settings
from iris_bot.demo_trade_audit import load_latest_demo_audit
from iris_bot.governance_active import resolve_active_profiles
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.profile_registry import (
    _build_rollback_snapshot,
    _find_entry,
    _latest_entry_by_state,
    _materialize_active_profiles_from_registry,
    _profile_checksum,
    _sync_strategy_profiles_file,
    _upsert_entry,
    _validated_profile_for_symbol,
    load_strategy_profile_registry,
    registry_path,
    save_strategy_profile_registry,
)
from iris_bot.registry_lock import (
    RegistryLockTimeoutError,
    RegistryMutationConflictError,
    registry_etag,
    registry_exclusive_lock,
)


def promote_strategy_profile(
    settings: Settings,
    promotion_review_for_symbol: Callable[..., dict[str, Any]],
) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "strategy_profile_promotion")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    symbol = settings.governance.target_symbol or settings.endurance.target_symbol
    if not symbol:
        logger.error("IRIS_GOVERNANCE_TARGET_SYMBOL is required")
        return 1

    reg_path = registry_path(settings)
    pre_lock_etag = registry_etag(reg_path)
    registry_snapshot = load_strategy_profile_registry(settings)
    review = promotion_review_for_symbol(
        settings,
        symbol,
        registry_snapshot,
        target_profile_id=settings.governance.target_profile_id,
    )
    profile_snapshot = _validated_profile_for_symbol(registry_snapshot, symbol, settings.governance.target_profile_id)
    if profile_snapshot is None:
        logger.error("No validated profile found for %s", symbol)
        return 2

    final_state = review["final_decision"]

    # Demo audit gate: block approved_demo promotion if audit shows unmatched broker
    # trades or P&L divergence beyond tolerance.  A missing audit is a warning, not
    # a hard block — audit infrastructure may not have run yet.
    demo_audit_gate: dict[str, Any] = {"checked": False, "ok": True, "reason": "audit_not_found"}
    if final_state == "APPROVED_DEMO":
        audit_report = load_latest_demo_audit(settings)
        if audit_report is not None:
            demo_audit_gate = {
                "checked": True,
                "ok": audit_report.get("ok", True),
                "fills_compared": audit_report.get("fills_compared", 0),
                "pnl_within_tolerance": audit_report.get("pnl_divergence", {}).get("within_tolerance", True),
                "slippage_within_tolerance": audit_report.get("slippage", {}).get("within_tolerance", True),
                "unmatched_broker_deals": len(audit_report.get("unmatched_broker_deals", [])),
            }
            if not audit_report.get("ok", True):
                final_state = "REVERT_TO_BLOCKED"
                review["reasons"].append("demo_audit_gate_failed")
                logger.warning(
                    "Demo audit gate failed for %s — promotion blocked. pnl_ok=%s slippage_ok=%s unmatched=%d",
                    symbol,
                    demo_audit_gate["pnl_within_tolerance"],
                    demo_audit_gate["slippage_within_tolerance"],
                    demo_audit_gate["unmatched_broker_deals"],
                )

    report_profile_id = profile_snapshot["profile_id"]
    report_active_id = ""
    report_final_state = "validated"

    try:
        with registry_exclusive_lock(reg_path, expected_etag=pre_lock_etag):
            registry = load_strategy_profile_registry(settings)
            profile = _validated_profile_for_symbol(registry, symbol, settings.governance.target_profile_id)
            if profile is None:
                logger.error("Validated profile not found in registry within lock: %s", symbol)
                return 3
            previous_active = registry.get("active_profiles", {}).get(symbol)

            if final_state == "APPROVED_DEMO":
                if previous_active is None or previous_active == profile["profile_id"]:
                    rollback_snapshot = _build_rollback_snapshot(profile)
                    _upsert_entry(registry, symbol, rollback_snapshot)
                    previous_active = rollback_snapshot["profile_id"]
                if previous_active and previous_active != profile["profile_id"]:
                    previous_entry = _find_entry(registry, symbol, previous_active)
                    if previous_entry is not None:
                        previous_entry["promotion_state"] = "deprecated"
                        previous_entry["promotion_reason"] = "replaced_by_new_approved_profile"
                profile["promotion_state"] = settings.governance.promotion_target_state
                profile["promotion_reason"] = "endurance_and_lifecycle_passed"
                profile["rollback_target"] = previous_active
                profile["profile_payload"]["promotion_state"] = settings.governance.promotion_target_state
                profile["profile_payload"]["promotion_reason"] = "endurance_and_lifecycle_passed"
                profile["profile_payload"]["rollback_target"] = previous_active
                profile["checksum"] = _profile_checksum(profile["profile_payload"])
                registry.setdefault("active_profiles", {})[symbol] = profile["profile_id"]
            elif final_state == "MOVE_TO_CAUTION":
                profile["promotion_state"] = "caution"
                profile["promotion_reason"] = ",".join(review["reasons"]) or "promotion_requires_caution"
                profile["profile_payload"]["promotion_state"] = "caution"
                profile["profile_payload"]["promotion_reason"] = profile["promotion_reason"]
                profile["checksum"] = _profile_checksum(profile["profile_payload"])
                _upsert_entry(registry, symbol, profile)
            elif final_state == "REVERT_TO_BLOCKED":
                profile["promotion_state"] = "blocked"
                profile["promotion_reason"] = ",".join(review["reasons"]) or "promotion_blocked"
                profile["profile_payload"]["promotion_state"] = "blocked"
                profile["profile_payload"]["promotion_reason"] = profile["promotion_reason"]
                profile["checksum"] = _profile_checksum(profile["profile_payload"])
                registry.get("active_profiles", {}).pop(symbol, None)
                _upsert_entry(registry, symbol, profile)

            save_strategy_profile_registry(settings, registry)
            _sync_strategy_profiles_file(settings, registry)
            _materialize_active_profiles_from_registry(settings, registry)

            report_profile_id = profile["profile_id"]
            report_active_id = registry.get("active_profiles", {}).get(symbol, "")
            report_final_state = profile["promotion_state"] if final_state != "KEEP_VALIDATED" else "validated"

    except RegistryMutationConflictError as exc:
        logger.error("Concurrent mutation conflict in promote_strategy_profile for %s: %s", symbol, exc)
        return 5
    except RegistryLockTimeoutError as exc:
        logger.error("Timeout acquiring registry lock for promote %s: %s", symbol, exc)
        return 6

    report = {
        "symbol": symbol,
        "profile_id": report_profile_id,
        "requested_state": settings.governance.promotion_target_state,
        "final_state": report_final_state,
        "review_decision": final_state,
        "promotion_reason": ",".join(review["reasons"]) if final_state in ("KEEP_VALIDATED",) else review.get("reasons", []),
        "rollback_target": profile_snapshot.get("rollback_target"),
        "active_profile_id": report_active_id,
        "gate_matrix": review["gate_matrix"],
        "gate_config_snapshot": review.get("gate_config_snapshot", {}),
        "endurance_summary": review["endurance_summary"],
        "lifecycle_summary": review["lifecycle_summary"],
        "demo_audit_gate": demo_audit_gate,
        "locking": {"lock_used": True, "etag_checked": True},
    }
    write_json_report(run_dir, "strategy_profile_promotion_report.json", wrap_artifact("strategy_profile_promotion", report))
    write_json_report(run_dir, "promotion_gate_matrix.json", wrap_artifact("strategy_profile_promotion", {"symbols": {symbol: review["gate_matrix"]}}))
    write_json_report(run_dir, "promotion_decision_summary.json", wrap_artifact("strategy_profile_promotion", {"symbols": {symbol: final_state}}))
    write_json_report(
        run_dir,
        "active_profile_resolution_report.json",
        wrap_artifact("active_strategy_status", resolve_active_profiles(settings)[1]),
    )
    write_json_report(run_dir, "active_strategy_status.json", wrap_artifact("active_strategy_status", resolve_active_profiles(settings)[1]))
    logger.info("strategy_profile_promotion symbol=%s state=%s decision=%s run_dir=%s", symbol, report_final_state, final_state, run_dir)
    return 0 if final_state == "APPROVED_DEMO" else 2


def rollback_strategy_profile(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "strategy_profile_rollback")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    symbol = settings.governance.target_symbol or settings.endurance.target_symbol
    if not symbol:
        logger.error("IRIS_GOVERNANCE_TARGET_SYMBOL is required")
        return 1

    reg_path = registry_path(settings)
    pre_lock_etag = registry_etag(reg_path)
    registry_snapshot = load_strategy_profile_registry(settings)
    current_id = registry_snapshot.get("active_profiles", {}).get(symbol)
    if not current_id:
        logger.error("No active profile for %s", symbol)
        return 2

    report_deprecated_id = ""
    report_restored_id = ""

    try:
        with registry_exclusive_lock(reg_path, expected_etag=pre_lock_etag):
            registry = load_strategy_profile_registry(settings)
            current_id = registry.get("active_profiles", {}).get(symbol)
            if not current_id:
                logger.error("No active profile for %s (within lock)", symbol)
                return 2
            current_entry = _find_entry(registry, symbol, current_id)
            if current_entry is None:
                logger.error("Active entry not found for %s", symbol)
                return 3
            rollback_target = current_entry.get("rollback_target")
            if not rollback_target:
                previous = _latest_entry_by_state(registry, symbol, {"approved_demo", "validated", "deprecated"})
                rollback_target = (
                    previous.get("profile_id")
                    if previous is not None and previous.get("profile_id") != current_id
                    else None
                )
            target_entry = _find_entry(registry, symbol, rollback_target) if rollback_target else None
            if target_entry is None:
                logger.error("No valid rollback_target for %s", symbol)
                return 4

            current_entry["promotion_state"] = "deprecated"
            current_entry["promotion_reason"] = "manual_rollback"
            current_entry["profile_payload"]["promotion_state"] = "deprecated"
            current_entry["profile_payload"]["promotion_reason"] = "manual_rollback"
            current_entry["checksum"] = _profile_checksum(current_entry["profile_payload"])

            target_entry["promotion_state"] = "approved_demo"
            target_entry["promotion_reason"] = "rollback_target_restored"
            target_entry["profile_payload"]["promotion_state"] = "approved_demo"
            target_entry["profile_payload"]["promotion_reason"] = "rollback_target_restored"
            target_entry["checksum"] = _profile_checksum(target_entry["profile_payload"])

            registry.setdefault("active_profiles", {})[symbol] = target_entry["profile_id"]
            save_strategy_profile_registry(settings, registry)
            _sync_strategy_profiles_file(settings, registry)
            _materialize_active_profiles_from_registry(settings, registry)

            report_deprecated_id = current_id
            report_restored_id = target_entry["profile_id"]

    except RegistryMutationConflictError as exc:
        logger.error("Concurrent mutation conflict in rollback for %s: %s", symbol, exc)
        return 5
    except RegistryLockTimeoutError as exc:
        logger.error("Timeout adquiriendo lock del registry para rollback %s: %s", symbol, exc)
        return 6

    report = {
        "symbol": symbol,
        "deprecated_profile_id": report_deprecated_id,
        "restored_profile_id": report_restored_id,
        "locking": {"lock_used": True, "etag_checked": True},
    }
    write_json_report(run_dir, "strategy_profile_promotion_report.json", wrap_artifact("strategy_profile_promotion", report))
    logger.info("strategy_profile_rollback symbol=%s restored=%s", symbol, report_restored_id)
    return 0
