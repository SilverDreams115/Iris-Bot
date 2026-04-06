"""
Registry I/O and entry helpers for the strategy profile registry.

Internal module used by governance.py — do not import directly from outside the package.
Public API lives in iris_bot.governance.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from iris_bot.artifacts import read_artifact_payload, wrap_artifact
from iris_bot.config import Settings
from iris_bot.operational import atomic_write_json
from iris_bot.symbols import SymbolStrategyProfile, default_symbol_strategy_profile, strategy_profiles_path


# ---------------------------------------------------------------------------
# Registry path helpers
# ---------------------------------------------------------------------------

def registry_path(settings: Settings) -> Path:
    return settings.data.runtime_dir / settings.governance.registry_filename


def active_strategy_profiles_path(settings: Settings) -> Path:
    """Path for the active-only materialization (only approved_demo entries)."""
    return settings.data.runtime_dir / "active_strategy_profiles.json"


# ---------------------------------------------------------------------------
# Checksum and default helpers
# ---------------------------------------------------------------------------

def _canonical_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = dict(payload)
    for key in ("profile_id", "promotion_state", "promotion_reason", "rollback_target"):
        canonical.pop(key, None)
    return canonical


def _profile_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_canonical_profile_payload(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _registry_default() -> dict[str, Any]:
    return {"profiles": {}, "active_profiles": {}}


def _json_dict(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Expected JSON object")
    return cast(dict[str, Any], raw)


def _entry_checksum_ok(entry: dict[str, Any]) -> bool:
    payload = dict(entry.get("profile_payload", {}))
    checksum = entry.get("checksum", "")
    return isinstance(checksum, str) and checksum == _profile_checksum(payload)


# ---------------------------------------------------------------------------
# Registry load / save
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Entry helpers
# ---------------------------------------------------------------------------

def _find_entry(registry: dict[str, Any], symbol: str, profile_id: str) -> dict[str, Any] | None:
    for item in registry.get("profiles", {}).get(symbol, []):
        if item.get("profile_id") == profile_id:
            return cast(dict[str, Any], item)
    return None


def _upsert_entry(registry: dict[str, Any], symbol: str, entry: dict[str, Any]) -> None:
    items = registry.setdefault("profiles", {}).setdefault(symbol, [])
    for index, item in enumerate(items):
        if item.get("profile_id") == entry.get("profile_id"):
            items[index] = entry
            return
    items.append(entry)


def _latest_entry_by_state(
    registry: dict[str, Any], symbol: str, states: set[str]
) -> dict[str, Any] | None:
    candidates = [
        item for item in registry.get("profiles", {}).get(symbol, [])
        if item.get("promotion_state") in states
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.get("created_at", ""))
    return cast(dict[str, Any], candidates[-1])


def _build_rollback_snapshot(entry: dict[str, Any]) -> dict[str, Any]:
    snapshot = _json_dict(json.loads(json.dumps(entry)))
    created_at = datetime.now(tz=UTC).isoformat()
    snapshot_id = f"{entry['profile_id']}-rollback-{entry['checksum'][:8]}"
    payload = dict(snapshot.get("profile_payload", {}))
    payload["profile_id"] = snapshot_id
    payload["promotion_state"] = "validated"
    payload["promotion_reason"] = "rollback_snapshot_before_approval"
    payload["rollback_target"] = entry.get("rollback_target")
    snapshot.update({
        "profile_id": snapshot_id,
        "created_at": created_at,
        "promotion_state": "validated",
        "promotion_reason": "rollback_snapshot_before_approval",
        "rollback_target": entry.get("rollback_target"),
        "profile_payload": payload,
    })
    snapshot["checksum"] = _profile_checksum(snapshot["profile_payload"])
    return snapshot


def _validated_profile_for_symbol(
    registry: dict[str, Any], symbol: str, profile_id: str = ""
) -> dict[str, Any] | None:
    if profile_id:
        entry = _find_entry(registry, symbol, profile_id)
        if entry is not None:
            return entry
    return _latest_entry_by_state(registry, symbol, {"validated"})


# ---------------------------------------------------------------------------
# Strategy profiles file sync and materialization
# ---------------------------------------------------------------------------

def _read_strategy_profiles_payload(
    settings: Settings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = strategy_profiles_path(settings)
    raw = _json_dict(json.loads(path.read_text(encoding="utf-8")))
    payload = read_artifact_payload(path, expected_type="strategy_profiles")
    return raw, payload


def _merge_symbol_profile(
    settings: Settings, symbol: str, payload: dict[str, Any]
) -> SymbolStrategyProfile:
    merged = asdict(default_symbol_strategy_profile(settings, symbol))
    merged.update(payload)
    merged["allowed_timeframes"] = tuple(merged["allowed_timeframes"])
    merged["allowed_sessions"] = tuple(merged["allowed_sessions"])
    merged.pop("enabled", None)  # derived property — not a constructor argument
    return SymbolStrategyProfile(**merged)


def _sync_strategy_profiles_file(settings: Settings, registry: dict[str, Any]) -> None:
    """
    Syncs strategy_profiles.json from the registry (full historical sync).
    For each symbol: uses the active profile if registered, otherwise latest entry.
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
            entry = _latest_entry_by_state(
                registry, symbol, {"approved_demo", "validated", "caution", "blocked", "deprecated"}
            )
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


def _materialize_active_profiles_from_registry(
    settings: Settings, registry: dict[str, Any]
) -> None:
    """Writes active_strategy_profiles.json with ONLY approved_demo entries."""
    active_symbols: dict[str, Any] = {}
    for symbol in settings.trading.symbols:
        active_id = registry.get("active_profiles", {}).get(symbol, "")
        if not active_id:
            continue
        entry = _find_entry(registry, symbol, active_id)
        if entry is None or entry.get("promotion_state") != "approved_demo":
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


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def _latest_run(settings: Settings, suffix: str) -> Path | None:
    candidates = sorted(
        path for path in settings.data.runs_dir.glob(f"*_{suffix}*") if path.is_dir()
    )
    return candidates[-1] if candidates else None
