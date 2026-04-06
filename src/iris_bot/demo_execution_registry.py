from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_bot.artifacts import read_artifact_payload, wrap_artifact
from iris_bot.config import Settings
from iris_bot.operational import atomic_write_json


def demo_execution_registry_path(settings: Settings) -> Path:
    return settings.data.runtime_dir / settings.demo_execution.registry_filename


def default_demo_execution_registry() -> dict[str, Any]:
    return {
        "symbols": {},
        "active_symbol": "",
        "gate_open": False,
        "last_updated_at": None,
    }


def load_demo_execution_registry(settings: Settings) -> dict[str, Any]:
    path = demo_execution_registry_path(settings)
    if not path.exists():
        return default_demo_execution_registry()
    return read_artifact_payload(path, expected_type="demo_execution_registry")


def save_demo_execution_registry(settings: Settings, payload: dict[str, Any]) -> Path:
    path = demo_execution_registry_path(settings)
    payload = dict(payload)
    payload["last_updated_at"] = datetime.now(tz=UTC).isoformat()
    atomic_write_json(
        path,
        wrap_artifact(
            "demo_execution_registry",
            payload,
            compatibility={"loader": "load_demo_execution_registry"},
        ),
    )
    return path


def activate_demo_execution_symbol(settings: Settings, symbol: str) -> tuple[bool, dict[str, Any]]:
    registry = load_demo_execution_registry(settings)
    symbols = dict(registry.get("symbols", {}))
    entry = dict(symbols.get(symbol, {}))
    if not entry:
        return False, {"reason": "symbol_missing_from_demo_execution_registry", "symbol": symbol}
    if entry.get("decision") != "APPROVED_FOR_DEMO_EXECUTION":
        return False, {"reason": "symbol_not_approved_for_demo_execution", "symbol": symbol}
    for item_symbol, item_entry in symbols.items():
        item_copy = dict(item_entry)
        item_copy["active_for_demo_execution"] = item_symbol == symbol
        symbols[item_symbol] = item_copy
    registry["symbols"] = symbols
    registry["active_symbol"] = symbol
    registry["gate_open"] = True
    save_demo_execution_registry(settings, registry)
    return True, {"reason": "activated", "symbol": symbol}
