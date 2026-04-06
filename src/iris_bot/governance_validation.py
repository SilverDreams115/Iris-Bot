from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from iris_bot.artifacts import artifact_schema_report, wrap_artifact
from iris_bot.config import Settings
from iris_bot.governance_active import strategy_validation_inputs
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.profile_registry import (
    _materialize_active_profiles_from_registry,
    _profile_checksum,
    _sync_strategy_profiles_file,
    _upsert_entry,
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
from iris_bot.symbols import load_symbol_strategy_profiles, strategy_profiles_path


def _build_validation_entries(
    settings: Settings,
    latest_validation: Any,
    enablement: dict[str, Any],
    leakage: dict[str, Any],
    comparison: dict[str, Any],
    strategy_schema: dict[str, Any],
    registry_snapshot: dict[str, Any],
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    profiles = load_symbol_strategy_profiles(settings)
    new_entries: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    payload: dict[str, Any] = {
        "symbols": {},
        "source_run_id": latest_validation.name,
        "leakage_safe": leakage.get("test_used_for_selection") is False,
    }

    for symbol, profile in sorted(profiles.items()):
        decision = enablement.get("symbols", {}).get(symbol, {})
        chosen_model = comparison.get("symbols", {}).get(symbol, {}).get("chosen_model", profile.model_variant)
        promotion_state = "blocked"
        reason = "blocked_by_default"
        enablement_state = "disabled"
        enabled = False
        if not strategy_schema.get("ok", False):
            promotion_state = "blocked"
            reason = "strategy_profiles_schema_incompatible"
        elif leakage.get("test_used_for_selection") is not False:
            promotion_state = "blocked"
            reason = "validation_leakage_detected"
        elif decision.get("state") == "enabled":
            promotion_state = "validated"
            reason = "strategy_validation_passed"
            enablement_state = "enabled"
            enabled = True
        elif decision.get("state") == "caution":
            promotion_state = "caution"
            reason = "strategy_validation_caution"
            enablement_state = "caution"
            enabled = True
        profile_payload = {
            **asdict(profile),
            "enabled_state": enablement_state,
            "enabled": enabled,
            "model_variant": chosen_model,
            "source_run_id": latest_validation.name,
            "promotion_state": promotion_state,
            "promotion_reason": reason,
            "rollback_target": registry_snapshot.get("active_profiles", {}).get(symbol),
        }
        checksum = _profile_checksum(profile_payload)
        profile_id = f"{symbol}-{latest_validation.name}-{checksum[:10]}"
        profile_payload["profile_id"] = profile_id
        entry = {
            "profile_id": profile_id,
            "artifact_type": "strategy_profile",
            "schema_version": strategy_schema.get("schema_version", 1),
            "checksum": checksum,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "source_run_id": latest_validation.name,
            "symbol": symbol,
            "model_variant": chosen_model,
            "enablement_state": enablement_state,
            "promotion_state": promotion_state,
            "promotion_reason": reason,
            "rollback_target": registry_snapshot.get("active_profiles", {}).get(symbol),
            "profile_payload": profile_payload,
        }
        new_entries.append(
            (
                symbol,
                entry,
                {
                    "profile_id": profile_id,
                    "from_state": "candidate",
                    "to_state": promotion_state,
                    "promotion_reason": reason,
                },
            )
        )
    return new_entries, payload


def validate_strategy_profiles(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "strategy_profile_validation")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        latest_validation, enablement, leakage, comparison = strategy_validation_inputs(settings)
        load_symbol_strategy_profiles(settings)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    strategy_schema = artifact_schema_report(strategy_profiles_path(settings), expected_type="strategy_profiles")
    reg_path = registry_path(settings)
    pre_lock_etag = registry_etag(reg_path)
    registry_snapshot = load_strategy_profile_registry(settings)
    new_entries, payload = _build_validation_entries(
        settings,
        latest_validation,
        enablement,
        leakage,
        comparison,
        strategy_schema,
        registry_snapshot,
    )

    try:
        with registry_exclusive_lock(reg_path, expected_etag=pre_lock_etag):
            registry = load_strategy_profile_registry(settings)
            for symbol, entry, summary in new_entries:
                rollback_target = registry.get("active_profiles", {}).get(symbol)
                entry["rollback_target"] = rollback_target
                entry["profile_payload"]["rollback_target"] = rollback_target
                _upsert_entry(registry, symbol, entry)
                payload["symbols"][symbol] = summary
            save_strategy_profile_registry(settings, registry)
            _sync_strategy_profiles_file(settings, registry)
            _materialize_active_profiles_from_registry(settings, registry)
    except RegistryMutationConflictError as exc:
        logger.error("Concurrent mutation conflict in validate_strategy_profiles: %s", exc)
        return 5
    except RegistryLockTimeoutError as exc:
        logger.error("Timeout adquiriendo lock del registry en validate_strategy_profiles: %s", exc)
        return 6

    write_json_report(run_dir, "strategy_profile_validation_report.json", wrap_artifact("strategy_profile_validation", payload))
    logger.info("strategy_profile_validation symbols=%s run_dir=%s", len(payload["symbols"]), run_dir)
    return 0
