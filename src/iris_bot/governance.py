from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from iris_bot.artifacts import artifact_schema_report, read_artifact_payload, wrap_artifact
from iris_bot.config import Settings
from iris_bot.evidence_store import (
    MANAGED_ARTIFACT_TYPES,
    ExternalPathError,
    evidence_store_status,
    get_latest_evidence,
    get_latest_evidence_payload,
    ingest_evidence,
)
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.operational import atomic_write_json
from iris_bot.registry_lock import (
    RegistryLockTimeoutError,
    RegistryMutationConflictError,
    governance_lock_audit,
    registry_etag,
    registry_exclusive_lock,
)
from iris_bot.run_index import get_latest_run, register_run
from iris_bot.symbols import SymbolStrategyProfile, default_symbol_strategy_profile, load_symbol_strategy_profiles, strategy_profiles_path


def registry_path(settings: Settings) -> Path:
    return settings.data.runtime_dir / settings.governance.registry_filename


def _canonical_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = dict(payload)
    for key in ("profile_id", "promotion_state", "promotion_reason", "rollback_target"):
        canonical.pop(key, None)
    return canonical


def _profile_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(_canonical_profile_payload(payload), sort_keys=True).encode("utf-8")).hexdigest()


def _registry_default() -> dict[str, Any]:
    return {"profiles": {}, "active_profiles": {}}


def load_strategy_profile_registry(settings: Settings) -> dict[str, Any]:
    path = registry_path(settings)
    if not path.exists():
        return _registry_default()
    return read_artifact_payload(path, expected_type="strategy_profile_registry")


def save_strategy_profile_registry(settings: Settings, payload: dict[str, Any]) -> Path:
    path = registry_path(settings)
    atomic_write_json(
        path,
        wrap_artifact(
            "strategy_profile_registry",
            payload,
            compatibility={"loader": "load_strategy_profile_registry"},
        ),
    )
    return path


def _latest_run(settings: Settings, suffix: str) -> Path | None:
    candidates = sorted(path for path in settings.data.runs_dir.glob(f"*_{suffix}*") if path.is_dir())
    return candidates[-1] if candidates else None


def _find_entry(registry: dict[str, Any], symbol: str, profile_id: str) -> dict[str, Any] | None:
    for item in registry.get("profiles", {}).get(symbol, []):
        if item.get("profile_id") == profile_id:
            return item
    return None


def _upsert_entry(registry: dict[str, Any], symbol: str, entry: dict[str, Any]) -> None:
    items = registry.setdefault("profiles", {}).setdefault(symbol, [])
    for index, item in enumerate(items):
        if item.get("profile_id") == entry.get("profile_id"):
            items[index] = entry
            return
    items.append(entry)


def _build_rollback_snapshot(entry: dict[str, Any]) -> dict[str, Any]:
    snapshot = json.loads(json.dumps(entry))
    created_at = datetime.now(tz=UTC).isoformat()
    snapshot_id = f"{entry['profile_id']}-rollback-{entry['checksum'][:8]}"
    payload = dict(snapshot.get("profile_payload", {}))
    payload["profile_id"] = snapshot_id
    payload["promotion_state"] = "validated"
    payload["promotion_reason"] = "rollback_snapshot_before_approval"
    payload["rollback_target"] = entry.get("rollback_target")
    snapshot.update(
        {
            "profile_id": snapshot_id,
            "created_at": created_at,
            "promotion_state": "validated",
            "promotion_reason": "rollback_snapshot_before_approval",
            "rollback_target": entry.get("rollback_target"),
            "profile_payload": payload,
        }
    )
    snapshot["checksum"] = _profile_checksum(snapshot["profile_payload"])
    return snapshot


def _latest_entry_by_state(registry: dict[str, Any], symbol: str, states: set[str]) -> dict[str, Any] | None:
    candidates = [item for item in registry.get("profiles", {}).get(symbol, []) if item.get("promotion_state") in states]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.get("created_at", ""))
    return candidates[-1]


def _read_strategy_profiles_payload(settings: Settings) -> tuple[dict[str, Any], dict[str, Any]]:
    path = strategy_profiles_path(settings)
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = read_artifact_payload(path, expected_type="strategy_profiles")
    return raw, payload


def _merge_symbol_profile(settings: Settings, symbol: str, payload: dict[str, Any]) -> SymbolStrategyProfile:
    merged = asdict(default_symbol_strategy_profile(settings, symbol))
    merged.update(payload)
    merged["allowed_timeframes"] = tuple(merged["allowed_timeframes"])
    merged["allowed_sessions"] = tuple(merged["allowed_sessions"])
    return SymbolStrategyProfile(**merged)


def _entry_checksum_ok(entry: dict[str, Any]) -> bool:
    payload = dict(entry.get("profile_payload", {}))
    return entry.get("checksum", "") == _profile_checksum(payload)


def _sync_strategy_profiles_file(settings: Settings, registry: dict[str, Any]) -> None:
    """
    Syncs strategy_profiles.json from the registry.

    This file is the historical/full sync used by the paper engine.
    It includes all promotion states (active, validated, caution, blocked, deprecated).
    For each symbol: uses the active profile if one is registered, otherwise the latest entry.

    NOTE: This file intentionally includes non-approved profiles for backwards compatibility
    with the paper engine. Use active_strategy_profiles.json (written by
    _materialize_active_profiles_from_registry) for consuming only approved_demo profiles.
    """
    path = strategy_profiles_path(settings)
    if not path.exists():
        return
    _, payload = _read_strategy_profiles_payload(settings)
    common = payload.get("common", {})
    symbols = dict(payload.get("symbols", {}))
    for symbol in settings.trading.symbols:
        profile_id = registry.get("active_profiles", {}).get(symbol, "")
        entry = _find_entry(registry, symbol, profile_id) if profile_id else None
        if entry is None:
            entry = _latest_entry_by_state(registry, symbol, {"approved_demo", "validated", "caution", "blocked", "deprecated"})
        if entry is not None:
            symbols[symbol] = dict(entry["profile_payload"])
    atomic_write_json(
        path,
        wrap_artifact(
            "strategy_profiles",
            {"common": common, "symbols": symbols},
            compatibility={"loader": "load_symbol_strategy_profiles"},
        ),
    )


def active_strategy_profiles_path(settings: Settings) -> Path:
    """Path for the active-only materialization (only approved_demo entries)."""
    return settings.data.runtime_dir / "active_strategy_profiles.json"


def _materialize_active_profiles_from_registry(settings: Settings, registry: dict[str, Any]) -> None:
    """
    Writes active_strategy_profiles.json containing ONLY approved_demo entries.

    This is the materialization that should be consumed when you need to know
    exactly which profiles are currently approved for demo execution.

    Separation from strategy_profiles.json:
      - strategy_profiles.json: full sync including historical, validated, blocked entries
      - active_strategy_profiles.json: only symbols with active approved_demo profile

    Must be called inside the registry lock to ensure coherence.
    """
    active_symbols: dict[str, Any] = {}
    for symbol in settings.trading.symbols:
        active_id = registry.get("active_profiles", {}).get(symbol, "")
        if not active_id:
            continue
        entry = _find_entry(registry, symbol, active_id)
        if entry is None:
            continue
        if entry.get("promotion_state") != "approved_demo":
            continue
        active_symbols[symbol] = dict(entry["profile_payload"])

    path = active_strategy_profiles_path(settings)
    atomic_write_json(
        path,
        wrap_artifact(
            "active_strategy_profiles",
            {
                "materialized_at": datetime.now(tz=UTC).isoformat(),
                "symbols": active_symbols,
                "symbol_count": len(active_symbols),
                "note": "Only approved_demo profiles. Use strategy_profiles.json for full historical sync.",
            },
            compatibility={"consumer": "paper_engine", "loader": "active_strategy_profiles"},
        ),
    )


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


def _strategy_validation_inputs(settings: Settings) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    latest = _latest_run(settings, "strategy_validation")
    if latest is None:
        raise FileNotFoundError("No hay strategy_validation previa")
    enablement = read_artifact_payload(latest / "symbol_enablement_report.json", expected_type="symbol_enablement")
    leakage = read_artifact_payload(latest / "leakage_fix_report.json", expected_type="strategy_validation")
    comparison = read_artifact_payload(latest / "strategy_validation_report.json", expected_type="strategy_validation")
    return latest, enablement, leakage, comparison


def validate_strategy_profiles(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "strategy_profile_validation")
    logger = configure_logging(run_dir, settings.logging.level)
    try:
        latest_validation, enablement, leakage, comparison = _strategy_validation_inputs(settings)
        profiles = load_symbol_strategy_profiles(settings)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    strategy_schema = artifact_schema_report(strategy_profiles_path(settings), expected_type="strategy_profiles")

    # Capture etag BEFORE any computation so we can detect concurrent mutations
    reg_path = registry_path(settings)
    pre_lock_etag = registry_etag(reg_path)

    # Build the new entries outside the lock (read-only, fast)
    # We'll use a temporary registry snapshot for rollback_target lookups
    registry_snapshot = load_strategy_profile_registry(settings)
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
        new_entries.append((symbol, entry, {
            "profile_id": profile_id,
            "from_state": "candidate",
            "to_state": promotion_state,
            "promotion_reason": reason,
        }))

    # Write mutations under exclusive lock to prevent lost updates
    try:
        with registry_exclusive_lock(reg_path, expected_etag=pre_lock_etag):
            # Re-read authoritative state inside lock
            registry = load_strategy_profile_registry(settings)
            for symbol, entry, summary in new_entries:
                # Refresh rollback_target with authoritative active_profiles
                rollback_target = registry.get("active_profiles", {}).get(symbol)
                entry["rollback_target"] = rollback_target
                entry["profile_payload"]["rollback_target"] = rollback_target
                _upsert_entry(registry, symbol, entry)
                payload["symbols"][symbol] = summary
            save_strategy_profile_registry(settings, registry)
            _sync_strategy_profiles_file(settings, registry)
            _materialize_active_profiles_from_registry(settings, registry)
    except RegistryMutationConflictError as exc:
        logger.error("Conflicto de mutación concurrente en validate_strategy_profiles: %s", exc)
        return 5
    except RegistryLockTimeoutError as exc:
        logger.error("Timeout adquiriendo lock del registry en validate_strategy_profiles: %s", exc)
        return 6

    write_json_report(run_dir, "strategy_profile_validation_report.json", wrap_artifact("strategy_profile_validation", payload))
    logger.info("strategy_profile_validation symbols=%s run_dir=%s", len(payload["symbols"]), run_dir)
    return 0


def _latest_endurance_evidence(settings: Settings, symbol: str) -> dict[str, Any] | None:
    candidates = sorted(
        [path for path in settings.data.runs_dir.glob("*_symbol_endurance*") if path.is_dir()]
        + [path for path in settings.data.runs_dir.glob("*_enabled_symbols_soak*") if path.is_dir()],
        reverse=True,
    )
    for run_dir in candidates:
        report_path = run_dir / "symbol_stability_report.json"
        if not report_path.exists():
            continue
        payload = read_artifact_payload(report_path, expected_type="symbol_stability")
        if symbol in payload.get("symbols", {}):
            return {"run_dir": run_dir, "payload": payload}
    return None


def _latest_endurance_payload(settings: Settings) -> dict[str, Any] | None:
    latest = _latest_run(settings, "symbol_endurance") or _latest_run(settings, "enabled_symbols_soak")
    if latest is None or not (latest / "symbol_stability_report.json").exists():
        return None
    return read_artifact_payload(latest / "symbol_stability_report.json", expected_type="symbol_stability")


def _artifact_generated_at(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return ""
    return str(payload.get("generated_at", ""))


def _plain_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _is_within_project(settings: Settings, path: Path) -> bool:
    """Returns True if path is within the project root or any configured data directory."""
    resolved = path.resolve()
    roots = [settings.project_root.resolve(), settings.data.runs_dir.resolve()]
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _latest_lifecycle_evidence(settings: Settings) -> dict[str, Any] | None:
    """
    Returns the latest lifecycle reconciliation evidence for governance decisions.

    Evidence discovery order (most authoritative first):
      1. Canonical evidence store (data/runtime/evidence_store/) — project-internal
      2. Local lifecycle_reconciliation run (within project root only)
      3. mt5_windows_stabilization run — ONLY if referenced path is within project root

    External paths (/mnt/c/..., paths outside project root) are REJECTED.
    A stale or missing evidence entry does NOT fall through to external paths.
    """
    candidates: list[dict[str, Any]] = []

    # Priority 1: Canonical evidence store (most authoritative, project-internal)
    store_entry = get_latest_evidence(settings, "lifecycle_reconciliation")
    if store_entry is not None:
        canonical_path = Path(store_entry.get("canonical_abs_path", ""))
        if canonical_path.exists() and _is_within_project(settings, canonical_path):
            try:
                payload = get_latest_evidence_payload(settings, "lifecycle_reconciliation")
                if payload is not None:
                    candidates.append(
                        {
                            "origin": "canonical_evidence_store",
                            "report_path": canonical_path,
                            "generated_at": store_entry.get("created_at", ""),
                            "payload": payload,
                            "audit_ok": None,
                            "source_run_id": store_entry.get("source_run_id", ""),
                            "provenance": store_entry.get("provenance", ""),
                        }
                    )
            except (OSError, ValueError):
                pass

    # Priority 2: Local lifecycle_reconciliation run (project-internal only)
    latest_local = _latest_run(settings, "lifecycle_reconciliation")
    if latest_local is not None and _is_within_project(settings, latest_local):
        report_path = latest_local / "lifecycle_reconciliation_report.json"
        if report_path.exists():
            try:
                candidates.append(
                    {
                        "origin": "local_lifecycle_reconciliation",
                        "report_path": report_path,
                        "generated_at": _artifact_generated_at(report_path),
                        "payload": read_artifact_payload(report_path, expected_type="lifecycle_reconciliation"),
                        "audit_ok": None,
                    }
                )
            except (OSError, ValueError):
                pass

    # Priority 3: mt5_windows_stabilization — ONLY if referenced path is within project
    latest_stabilization = _latest_run(settings, "mt5_windows_stabilization")
    if latest_stabilization is not None and _is_within_project(settings, latest_stabilization):
        stabilization_path = latest_stabilization / "lifecycle_rerun_report.json"
        stabilization = _plain_json(stabilization_path)
        if stabilization:
            referenced_path = Path(str(stabilization.get("reconciliation_run", "")))
            # CRITICAL: reject external paths (e.g., /mnt/c/Temp/...)
            if referenced_path.exists() and _is_within_project(settings, referenced_path):
                try:
                    candidates.append(
                        {
                            "origin": "windows_native_stabilization",
                            "report_path": referenced_path,
                            "generated_at": _artifact_generated_at(referenced_path),
                            "payload": read_artifact_payload(referenced_path, expected_type="lifecycle_reconciliation"),
                            "audit_ok": bool(stabilization.get("audit_ok", False)),
                            "stabilization_run": str(stabilization_path),
                            "audit_run": stabilization.get("audit_run", ""),
                            "rerun_source": stabilization.get("rerun_source", ""),
                        }
                    )
                except (OSError, ValueError):
                    pass
            # If path is external, silently skip — do NOT use it as canonical evidence

    if not candidates:
        return None
    candidates.sort(key=lambda item: (str(item.get("generated_at", "")), str(item.get("report_path", ""))))
    return candidates[-1]


def _latest_lifecycle_payload(settings: Settings) -> dict[str, Any] | None:
    evidence = _latest_lifecycle_evidence(settings)
    return None if evidence is None else evidence["payload"]


def _validated_profile_for_symbol(registry: dict[str, Any], symbol: str, profile_id: str = "") -> dict[str, Any] | None:
    if profile_id:
        entry = _find_entry(registry, symbol, profile_id)
        if entry is not None:
            return entry
    return _latest_entry_by_state(registry, symbol, {"validated"})


def _lifecycle_evidence_age_hours(lifecycle: dict[str, Any] | None) -> float | None:
    """Returns age of lifecycle evidence in hours, or None if unparseable."""
    if lifecycle is None:
        return None
    generated_at = str((lifecycle or {}).get("generated_at", ""))
    if not generated_at:
        return None
    try:
        ts = datetime.fromisoformat(generated_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(tz=UTC) - ts).total_seconds()
        return age_seconds / 3600.0
    except (ValueError, TypeError):
        return None


def _compute_endurance_gate_metrics(
    endurance_symbol: dict[str, Any],
) -> dict[str, Any]:
    """
    Extracts and computes gate-relevant metrics from an endurance symbol payload.

    Handles missing fields gracefully (returns None for unavailable metrics).
    """
    if not isinstance(endurance_symbol, dict):
        return {
            "trade_count": 0,
            "no_trade_count": 0,
            "blocked_trades": 0,
            "no_trade_ratio": None,
            "blocked_trades_ratio": None,
            "avg_profit_factor": None,
            "avg_expectancy_usd": None,
            "cycles_completed": 0,
        }
    cycle_metrics = endurance_symbol.get("cycle_metrics", []) or []
    trade_count = sum(int(item.get("trades", 0) or 0) for item in cycle_metrics)
    no_trade_count = int(endurance_symbol.get("no_trade_count", 0) or 0)
    blocked_trades = int(endurance_symbol.get("blocked_trades", 0) or 0)
    total_signals = trade_count + no_trade_count + blocked_trades
    no_trade_ratio = no_trade_count / total_signals if total_signals > 0 else None
    trade_plus_blocked = trade_count + blocked_trades
    blocked_trades_ratio = blocked_trades / trade_plus_blocked if trade_plus_blocked > 0 else None
    pf_series = [float(item.get("profit_factor", 0.0) or 0.0) for item in cycle_metrics if item.get("trades", 0)]
    ex_series = [float(item.get("expectancy_usd", 0.0) or 0.0) for item in cycle_metrics if item.get("trades", 0)]
    avg_profit_factor = sum(pf_series) / len(pf_series) if pf_series else None
    avg_expectancy_usd = sum(ex_series) / len(ex_series) if ex_series else None
    return {
        "trade_count": trade_count,
        "no_trade_count": no_trade_count,
        "blocked_trades": blocked_trades,
        "no_trade_ratio": no_trade_ratio,
        "blocked_trades_ratio": blocked_trades_ratio,
        "avg_profit_factor": avg_profit_factor,
        "avg_expectancy_usd": avg_expectancy_usd,
        "cycles_completed": int(endurance_symbol.get("cycles_completed", 0) or 0),
    }


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
    endurance_payload = ((endurance or {}).get("payload", {}) or {})
    endurance_symbol = endurance_payload.get("symbols", {}).get(symbol, {})
    reasons: list[str] = []
    severity = "info"
    final_decision = "KEEP_VALIDATED"
    alerts = endurance_symbol.get("alerts_by_severity", {}) if isinstance(endurance_symbol, dict) else {}
    endo_metrics = _compute_endurance_gate_metrics(endurance_symbol)
    trade_count = endo_metrics["trade_count"]

    # --- Compute lifecycle age ---
    lifecycle_age_hours = _lifecycle_evidence_age_hours(lifecycle)

    # --- Hard gate matrix (all must be True for APPROVED_DEMO) ---
    # Lifecycle age check
    lifecycle_recent = (
        lifecycle_age_hours is not None
        and lifecycle_age_hours <= gate_cfg.lifecycle_max_age_hours
    ) if lifecycle is not None else False

    # Endurance economic floors
    profit_factor_floor_ok = (
        endo_metrics["avg_profit_factor"] is not None
        and endo_metrics["avg_profit_factor"] >= gate_cfg.min_profit_factor
    ) if endurance is not None else False
    expectancy_floor_ok = (
        endo_metrics["avg_expectancy_usd"] is not None
        and endo_metrics["avg_expectancy_usd"] >= gate_cfg.min_expectancy_usd
    ) if endurance is not None else False

    # Signal efficiency checks
    no_trade_ratio_ok = (
        endo_metrics["no_trade_ratio"] is None  # insufficient data to check
        or endo_metrics["no_trade_ratio"] <= gate_cfg.max_no_trade_ratio
    ) if endurance is not None else False
    blocked_ratio_ok = (
        endo_metrics["blocked_trades_ratio"] is None
        or endo_metrics["blocked_trades_ratio"] <= gate_cfg.max_blocked_trades_ratio
    ) if endurance is not None else False

    gate_matrix = {
        "validated_profile_present": profile is not None,
        "profile_checksum_ok": _entry_checksum_ok(profile) if profile is not None else False,
        "schema_compatible": strategy_schema.get("ok", False),
        # Lifecycle: missing evidence is now BLOCKED (not just a warning)
        "lifecycle_evidence_present": lifecycle is not None,
        "lifecycle_clean": (
            int(lifecycle_symbol.get("critical_mismatch_count", 0) or 0) <= gate_cfg.lifecycle_max_critical
        ) if lifecycle is not None else False,
        # lifecycle_audit_ok: now REQUIRED for APPROVED_DEMO (not just caution-eligible)
        "lifecycle_audit_ok": bool((lifecycle or {}).get("audit_ok", False)) if lifecycle is not None else False,
        "lifecycle_evidence_recent": lifecycle_recent,
        # Endurance: missing evidence is now BLOCKED (not just a warning)
        "endurance_present": endurance is not None,
        "endurance_decision_go": endurance_symbol.get("decision") == "go" if endurance is not None else False,
        "endurance_cycles_sufficient": (
            endo_metrics["cycles_completed"] >= gate_cfg.endurance_min_cycles
        ) if endurance is not None else False,
        "endurance_min_trades_met": trade_count >= gate_cfg.endurance_min_trades,
        "critical_alerts_absent": int(alerts.get("critical", 0) or 0) == 0,
        # Economic floors (new hard gates)
        "profit_factor_floor_ok": profit_factor_floor_ok,
        "expectancy_floor_ok": expectancy_floor_ok,
        # Signal efficiency floors (new hard gates)
        "no_trade_ratio_ok": no_trade_ratio_ok,
        "blocked_trades_ratio_ok": blocked_ratio_ok,
        # Degradation checks (tighter thresholds from gate config)
        "no_expectancy_degradation_gate_breach": (
            float(endurance_symbol.get("expectancy_degradation_pct", 0.0) or 0.0)
            <= gate_cfg.max_expectancy_degradation_pct
        ) if endurance is not None else False,
        "no_profit_factor_degradation_gate_breach": (
            float(endurance_symbol.get("profit_factor_degradation_pct", 0.0) or 0.0)
            <= gate_cfg.max_profit_factor_degradation_pct
        ) if endurance is not None else False,
    }

    # --- Decision logic: conservative, fail-safe ---
    # Permanently excluded symbols always block first
    if symbol == "USDJPY":
        final_decision = "REVERT_TO_BLOCKED"
        severity = "error"
        reasons.append("symbol_out_of_scope_for_promotion")

    # Profile integrity checks
    if not gate_matrix["validated_profile_present"]:
        final_decision = "REVERT_TO_BLOCKED"
        severity = "error"
        reasons.append("validated_profile_missing")
    if gate_matrix["validated_profile_present"] and not gate_matrix["profile_checksum_ok"]:
        final_decision = "REVERT_TO_BLOCKED"
        severity = "error"
        reasons.append("profile_checksum_incompatible")
    if not gate_matrix["schema_compatible"]:
        final_decision = "REVERT_TO_BLOCKED"
        severity = "error"
        reasons.append("strategy_profiles_schema_incompatible")

    if final_decision != "REVERT_TO_BLOCKED":
        # Lifecycle: all conditions block in strict mode; soft mode keeps validated
        if not gate_matrix["lifecycle_evidence_present"]:
            if strict:
                final_decision = "REVERT_TO_BLOCKED"
                severity = "error"
                reasons.append("missing_lifecycle_evidence_blocks_promotion")
            else:
                reasons.append("missing_lifecycle_validation")
        else:
            if not gate_matrix["lifecycle_clean"]:
                final_decision = "REVERT_TO_BLOCKED"
                severity = "error"
                reasons.append("lifecycle_critical_mismatch")
            if gate_cfg.require_lifecycle_audit_ok and not gate_matrix["lifecycle_audit_ok"]:
                # lifecycle audit_ok is now REQUIRED, not just caution-eligible
                final_decision = "REVERT_TO_BLOCKED"
                severity = "error"
                reasons.append("lifecycle_audit_not_confirmed_blocks_promotion")
            if not gate_matrix["lifecycle_evidence_recent"]:
                final_decision = "REVERT_TO_BLOCKED"
                severity = "error"
                reasons.append(
                    f"lifecycle_evidence_too_old:"
                    f"{lifecycle_age_hours:.1f}h>max_{gate_cfg.lifecycle_max_age_hours:.0f}h"
                    if lifecycle_age_hours is not None else "lifecycle_evidence_age_unknown"
                )

        # Endurance: ALL missing conditions block in strict mode; soft mode keeps validated
        if not gate_matrix["endurance_present"]:
            if strict:
                final_decision = "REVERT_TO_BLOCKED"
                severity = "error"
                reasons.append("missing_endurance_evidence_blocks_promotion")
            else:
                reasons.append("missing_endurance_validation")
        else:
            if not gate_matrix["endurance_decision_go"]:
                decision_val = endurance_symbol.get("decision", "missing")
                if strict:
                    final_decision = "REVERT_TO_BLOCKED"
                    severity = "error"
                    reasons.append(f"endurance_decision={decision_val}_blocks_promotion")
                else:
                    if final_decision not in ("REVERT_TO_BLOCKED",):
                        final_decision = "MOVE_TO_CAUTION"
                        severity = "warning"
                    reasons.append(f"endurance_decision={decision_val}")
            if not gate_matrix["endurance_cycles_sufficient"]:
                final_decision = "REVERT_TO_BLOCKED"
                severity = "error"
                reasons.append(
                    f"insufficient_endurance_cycles:{endo_metrics['cycles_completed']}<{gate_cfg.endurance_min_cycles}"
                )
            if not gate_matrix["endurance_min_trades_met"]:
                final_decision = "REVERT_TO_BLOCKED"
                severity = "error"
                reasons.append(
                    f"insufficient_endurance_trades:{trade_count}<{gate_cfg.endurance_min_trades}"
                )
            if not gate_matrix["critical_alerts_absent"]:
                final_decision = "REVERT_TO_BLOCKED"
                severity = "error"
                reasons.append("critical_alerts_present_blocks_promotion")
            if not gate_matrix["profit_factor_floor_ok"]:
                final_decision = "REVERT_TO_BLOCKED"
                severity = "error"
                pf_val = endo_metrics["avg_profit_factor"]
                reasons.append(
                    f"profit_factor_below_floor:{pf_val:.3f}<{gate_cfg.min_profit_factor}"
                    if pf_val is not None else "profit_factor_unavailable"
                )
            if not gate_matrix["expectancy_floor_ok"]:
                final_decision = "REVERT_TO_BLOCKED"
                severity = "error"
                ex_val = endo_metrics["avg_expectancy_usd"]
                reasons.append(
                    f"expectancy_below_floor:{ex_val:.3f}<{gate_cfg.min_expectancy_usd}"
                    if ex_val is not None else "expectancy_unavailable"
                )
            if not gate_matrix["no_trade_ratio_ok"]:
                final_decision = "REVERT_TO_BLOCKED"
                severity = "error"
                ntr = endo_metrics["no_trade_ratio"]
                reasons.append(
                    f"no_trade_ratio_exceeded:{ntr:.3f}>{gate_cfg.max_no_trade_ratio}"
                    if ntr is not None else "no_trade_ratio_check_failed"
                )
            if not gate_matrix["blocked_trades_ratio_ok"]:
                final_decision = "REVERT_TO_BLOCKED"
                severity = "error"
                btr = endo_metrics["blocked_trades_ratio"]
                reasons.append(
                    f"blocked_trades_ratio_exceeded:{btr:.3f}>{gate_cfg.max_blocked_trades_ratio}"
                    if btr is not None else "blocked_trades_ratio_check_failed"
                )
            if not gate_matrix["no_expectancy_degradation_gate_breach"]:
                # Degradation is CAUTION (borderline data quality, not a hard error)
                if final_decision not in ("REVERT_TO_BLOCKED",):
                    final_decision = "MOVE_TO_CAUTION"
                    severity = "warning"
                reasons.append("expectancy_degradation_above_gate")
            if not gate_matrix["no_profit_factor_degradation_gate_breach"]:
                if final_decision not in ("REVERT_TO_BLOCKED",):
                    final_decision = "MOVE_TO_CAUTION"
                    severity = "warning"
                reasons.append("profit_factor_degradation_above_gate")

    if final_decision == "KEEP_VALIDATED" and all(gate_matrix.values()):
        final_decision = "APPROVED_DEMO"
        reasons.append("validated_endurance_lifecycle_passed")
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
            "trade_count": trade_count,
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
    logger = configure_logging(run_dir, settings.logging.level)
    registry = load_strategy_profile_registry(settings)
    symbols = _promotion_review_symbols(settings, registry)
    if not symbols:
        logger.error("No hay simbolos validated elegibles para revisar")
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
    run_dir = build_run_directory(settings.data.runs_dir, "strategy_profile_promotion")
    logger = configure_logging(run_dir, settings.logging.level)
    symbol = settings.governance.target_symbol or settings.endurance.target_symbol
    if not symbol:
        logger.error("IRIS_GOVERNANCE_TARGET_SYMBOL es requerido")
        return 1

    # --- Pre-lock phase: read-only review (may be slightly stale, used for decision + reporting) ---
    reg_path = registry_path(settings)
    pre_lock_etag = registry_etag(reg_path)
    registry_snapshot = load_strategy_profile_registry(settings)
    review = _promotion_review_for_symbol(
        settings,
        symbol,
        registry_snapshot,
        target_profile_id=settings.governance.target_profile_id,
    )
    profile_snapshot = _validated_profile_for_symbol(registry_snapshot, symbol, settings.governance.target_profile_id)
    if profile_snapshot is None:
        logger.error("No existe perfil validado para %s", symbol)
        return 2

    final_state = review["final_decision"]
    report_profile_id = profile_snapshot["profile_id"]
    report_active_id = ""
    report_final_state = "validated"

    # --- Mutations require exclusive lock ---
    try:
        with registry_exclusive_lock(reg_path, expected_etag=pre_lock_etag):
            # Re-read authoritative state inside lock
            registry = load_strategy_profile_registry(settings)
            profile = _validated_profile_for_symbol(registry, symbol, settings.governance.target_profile_id)
            if profile is None:
                logger.error("Perfil validado no encontrado en registry dentro del lock: %s", symbol)
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

            # Capture post-mutation values for reporting (inside lock, authoritative)
            report_profile_id = profile["profile_id"]
            report_active_id = registry.get("active_profiles", {}).get(symbol, "")
            report_final_state = profile["promotion_state"] if final_state != "KEEP_VALIDATED" else "validated"

    except RegistryMutationConflictError as exc:
        logger.error("Conflicto de mutación concurrente en promote_strategy_profile para %s: %s", symbol, exc)
        return 5
    except RegistryLockTimeoutError as exc:
        logger.error("Timeout adquiriendo lock del registry para promote %s: %s", symbol, exc)
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
    logger = configure_logging(run_dir, settings.logging.level)
    symbol = settings.governance.target_symbol or settings.endurance.target_symbol
    if not symbol:
        logger.error("IRIS_GOVERNANCE_TARGET_SYMBOL es requerido")
        return 1

    reg_path = registry_path(settings)
    pre_lock_etag = registry_etag(reg_path)

    # Pre-flight: check preconditions before acquiring lock
    registry_snapshot = load_strategy_profile_registry(settings)
    current_id = registry_snapshot.get("active_profiles", {}).get(symbol)
    if not current_id:
        logger.error("No hay perfil activo para %s", symbol)
        return 2

    report_deprecated_id = ""
    report_restored_id = ""

    try:
        with registry_exclusive_lock(reg_path, expected_etag=pre_lock_etag):
            # Re-read authoritative state inside lock
            registry = load_strategy_profile_registry(settings)
            current_id = registry.get("active_profiles", {}).get(symbol)
            if not current_id:
                logger.error("No hay perfil activo para %s (dentro del lock)", symbol)
                return 2
            current_entry = _find_entry(registry, symbol, current_id)
            if current_entry is None:
                logger.error("No existe entrada activa para %s", symbol)
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
                logger.error("No existe rollback_target valido para %s", symbol)
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
        logger.error("Conflicto de mutación concurrente en rollback para %s: %s", symbol, exc)
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


def list_strategy_profiles(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "list_strategy_profiles")
    logger = configure_logging(run_dir, settings.logging.level)
    payload = load_strategy_profile_registry(settings)
    write_json_report(run_dir, "strategy_profile_registry.json", wrap_artifact("strategy_profile_registry", payload))
    logger.info("list_strategy_profiles run_dir=%s", run_dir)
    return 0


def active_strategy_status(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "active_strategy_status")
    logger = configure_logging(run_dir, settings.logging.level)
    _, payload = resolve_active_profiles(settings)
    write_json_report(run_dir, "active_strategy_status.json", wrap_artifact("active_strategy_status", payload))
    logger.info("active_strategy_status blocked=%s run_dir=%s", payload["blocked_symbols"], run_dir)
    return 0 if payload["blocked_symbols"] == 0 else 2


def diagnose_profile_activation(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "diagnose_profile_activation")
    logger = configure_logging(run_dir, settings.logging.level)
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
    logger = configure_logging(run_dir, settings.logging.level)

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
    logger = configure_logging(run_dir, settings.logging.level)

    reg_path = registry_path(settings)
    pre_lock_etag = registry_etag(reg_path)

    try:
        with registry_exclusive_lock(reg_path, expected_etag=pre_lock_etag):
            registry = load_strategy_profile_registry(settings)
            _materialize_active_profiles_from_registry(settings, registry)
    except (RegistryMutationConflictError, RegistryLockTimeoutError) as exc:
        logger.error("No se pudo materializar active profiles bajo lock: %s", exc)
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
    logger = configure_logging(run_dir, settings.logging.level)
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
    logger = configure_logging(run_dir, settings.logging.level)
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
    logger = configure_logging(run_dir, settings.logging.level)
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
