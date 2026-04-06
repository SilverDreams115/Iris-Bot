from __future__ import annotations

from pathlib import Path
from typing import Any

from iris_bot.artifacts import artifact_schema_report, read_artifact_payload
from iris_bot.config import Settings
from iris_bot.profile_registry import (
    _entry_checksum_ok,
    _find_entry,
    _latest_run,
    _merge_symbol_profile,
    load_strategy_profile_registry,
)
from iris_bot.symbols import SymbolStrategyProfile, load_symbol_strategy_profiles, strategy_profiles_path


def resolve_active_profile_entry(
    settings: Settings,
    symbol: str,
    registry: dict[str, Any] | None = None,
    file_profiles: dict[str, SymbolStrategyProfile] | None = None,
) -> dict[str, Any]:
    registry_payload = registry or load_strategy_profile_registry(settings)
    strategy_profiles = file_profiles or load_symbol_strategy_profiles(settings)
    active_id = registry_payload.get("active_profiles", {}).get(symbol, "")
    entry = _find_entry(registry_payload, symbol, active_id) if active_id else None
    valid_states = {"approved_demo"} | ({"validated"} if settings.governance.allow_validated_fallback else set())
    reasons: list[str] = []
    warnings: list[str] = []
    if not active_id:
        reasons.append("no_active_profile_registered")
    elif entry is None:
        reasons.append("active_profile_missing_in_registry")
    else:
        if entry.get("promotion_state") not in valid_states:
            reasons.append(f"active_profile_state_invalid:{entry.get('promotion_state', '')}")
        if not _entry_checksum_ok(entry):
            reasons.append("registry_profile_checksum_mismatch")
        path_report = artifact_schema_report(strategy_profiles_path(settings), expected_type="strategy_profiles")
        if not path_report.get("ok", False):
            reasons.append("strategy_profiles_schema_incompatible")
        file_profile = strategy_profiles.get(symbol)
        if file_profile is not None and file_profile.profile_id and file_profile.profile_id != active_id:
            warnings.append("strategy_profiles_out_of_sync_with_registry")
    resolved_profile = None
    if entry is not None and not reasons:
        resolved_profile = _merge_symbol_profile(settings, symbol, dict(entry["profile_payload"]))
    file_profile = strategy_profiles.get(symbol)
    enablement_state = entry.get("enablement_state", file_profile.enabled_state if file_profile else "missing") if entry is not None else (file_profile.enabled_state if file_profile else "missing")
    model_variant = entry.get("model_variant", file_profile.model_variant if file_profile else "") if entry is not None else (file_profile.model_variant if file_profile else "")
    source_run_id = entry.get("source_run_id", file_profile.source_run_id if file_profile else "") if entry is not None else (file_profile.source_run_id if file_profile else "")
    return {
        "symbol": symbol,
        "ok": resolved_profile is not None,
        "operationally_valid": resolved_profile is not None,
        "active_profile_id": active_id,
        "profile_found": entry is not None,
        "promotion_state": entry.get("promotion_state", "") if entry else "",
        "enablement_state": enablement_state,
        "model_variant": model_variant,
        "source_run_id": source_run_id,
        "reasons": reasons,
        "warnings": warnings,
        "resolved_profile": resolved_profile,
    }


def resolve_active_profiles(settings: Settings) -> tuple[dict[str, SymbolStrategyProfile], dict[str, Any]]:
    registry = load_strategy_profile_registry(settings)
    file_profiles = load_symbol_strategy_profiles(settings)
    resolved: dict[str, SymbolStrategyProfile] = {}
    report_symbols: dict[str, Any] = {}
    blocked = 0
    for symbol in settings.trading.symbols:
        status = resolve_active_profile_entry(settings, symbol, registry=registry, file_profiles=file_profiles)
        report_symbols[symbol] = {key: value for key, value in status.items() if key != "resolved_profile"}
        if status["ok"] and status["resolved_profile"] is not None:
            resolved[symbol] = status["resolved_profile"]
        elif settings.governance.require_active_profile:
            blocked += 1
    return resolved, {"symbols": report_symbols, "blocked_symbols": blocked, "require_active_profile": settings.governance.require_active_profile}


def strategy_validation_inputs(settings: Settings) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    latest = _latest_run(settings, "strategy_validation")
    if latest is None:
        raise FileNotFoundError("No prior strategy_validation run found")
    enablement = read_artifact_payload(latest / "symbol_enablement_report.json", expected_type="symbol_enablement")
    leakage = read_artifact_payload(latest / "leakage_fix_report.json", expected_type="strategy_validation")
    comparison = read_artifact_payload(latest / "strategy_validation_report.json", expected_type="strategy_validation")
    return latest, enablement, leakage, comparison
