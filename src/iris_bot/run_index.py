"""
Canonical run index for IRIS-Bot.

Replaces the fragile glob + lexicographic-sort heuristic (_latest_run) with an
explicit, append-only index file that is updated after every relevant run.

The index is the authoritative source for:
  - Latest lifecycle_reconciliation run
  - Latest symbol_endurance / enabled_symbols_soak run per symbol
  - Latest strategy_profile_promotion run per symbol

Design:
  - Single JSON file: data/runtime/run_index.json
  - Append-only entries (never mutate past entries)
  - Atomic writes via os.replace
  - Queried by (run_type, symbol) → most recent entry

The glob fallback in governance.py is kept for backwards compatibility with runs
that predate the index, but new runs register themselves and future queries use
the index first.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from iris_bot.config import Settings


_INDEX_FILENAME = "run_index.json"
_SCHEMA_VERSION = 1


def run_index_path(settings: Settings) -> Path:
    return settings.data.runtime_dir / _INDEX_FILENAME


def _coerce_index(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"schema_version": _SCHEMA_VERSION, "entries": []}
    return cast(dict[str, Any], raw)


def _load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": _SCHEMA_VERSION, "entries": []}
    try:
        raw = _coerce_index(json.loads(path.read_text(encoding="utf-8")))
        if not isinstance(raw.get("entries"), list):
            return {"schema_version": _SCHEMA_VERSION, "entries": []}
        return raw
    except (OSError, json.JSONDecodeError, ValueError):
        return {"schema_version": _SCHEMA_VERSION, "entries": []}


def _save_index(path: Path, index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def register_run(
    settings: Settings,
    run_id: str,
    run_type: str,
    run_dir: Path,
    symbol: str | None = None,
    artifact_types: list[str] | None = None,
    status: str = "completed",
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Registers a completed run in the canonical run index.

    Must be called after the run writes all its artifacts, so that the index
    entry is only created when the run is actually complete.
    """
    path = run_index_path(settings)
    index = _load_index(path)
    entry: dict[str, Any] = {
        "run_id": run_id,
        "run_type": run_type,
        "run_dir": str(run_dir),
        "symbol": symbol,
        "artifact_types": artifact_types or [],
        "status": status,
        "registered_at": datetime.now(tz=UTC).isoformat(),
    }
    if metadata:
        entry["metadata"] = metadata
    index["entries"].append(entry)
    _save_index(path, index)


def get_latest_run(
    settings: Settings,
    run_type: str,
    symbol: str | None = None,
) -> dict[str, Any] | None:
    """
    Returns the most recently registered run entry for a given run_type.

    If symbol is provided, filters to runs that either match the symbol or
    have symbol=None (global runs).

    Returns None if no matching run is found in the index (caller should
    fall back to glob heuristic for pre-index runs).
    """
    path = run_index_path(settings)
    index = _load_index(path)
    entries = [
        e for e in index["entries"]
        if e.get("run_type") == run_type
        and e.get("status") == "completed"
        and (symbol is None or e.get("symbol") in (symbol, None))
    ]
    if not entries:
        return None
    # Sort by registered_at (ISO 8601 string sorts correctly for UTC timestamps)
    entries.sort(key=lambda e: e.get("registered_at", ""))
    return cast(dict[str, Any], entries[-1])


def get_all_runs(
    settings: Settings,
    run_type: str | None = None,
    symbol: str | None = None,
) -> list[dict[str, Any]]:
    """Returns all indexed runs, optionally filtered by type and/or symbol."""
    path = run_index_path(settings)
    index = _load_index(path)
    entries = index["entries"]
    if run_type is not None:
        entries = [e for e in entries if e.get("run_type") == run_type]
    if symbol is not None:
        entries = [e for e in entries if e.get("symbol") in (symbol, None)]
    return sorted(entries, key=lambda e: e.get("registered_at", ""))


def run_index_status(settings: Settings) -> dict[str, Any]:
    """Returns a summary of the run index for audit purposes."""
    path = run_index_path(settings)
    index = _load_index(path)
    entries = index["entries"]
    by_type: dict[str, int] = {}
    for e in entries:
        rt = e.get("run_type", "unknown")
        by_type[rt] = by_type.get(rt, 0) + 1
    latest_per_type: dict[str, str] = {}
    for rt in by_type:
        matches = [e for e in entries if e.get("run_type") == rt]
        if matches:
            matches.sort(key=lambda e: e.get("registered_at", ""))
            latest_per_type[rt] = matches[-1].get("registered_at", "")
    return {
        "index_path": str(path),
        "index_exists": path.exists(),
        "total_entries": len(entries),
        "by_run_type": by_type,
        "latest_per_type": latest_per_type,
    }
