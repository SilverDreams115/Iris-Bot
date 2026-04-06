"""
Canonical evidence store for IRIS-Bot governance artifacts.

Problem solved:
  Previously, lifecycle reconciliation evidence could reference external paths
  like /mnt/c/Temp/... (Windows workspace) which are not durable parts of the
  project. If that path disappears, governance decisions lose their evidence trail.

Solution:
  A canonical evidence store inside the project at data/runtime/evidence_store/.
  Critical governance artifacts are materialized (copied with full metadata) into
  this store when they are produced. The store has a manifest that is the
  authoritative index for evidence discovery — no more glob heuristics.

Evidence types managed by this store:
  - lifecycle_reconciliation  (per-symbol or global)
  - symbol_stability          (per-symbol endurance results)

Manifest format:
  data/runtime/evidence_store/evidence_store_manifest.json

Each entry in the manifest has:
  - entry_id: unique identifier
  - artifact_type: which kind of artifact
  - symbol: which symbol (or null for global)
  - checksum: SHA-256 of the artifact payload
  - created_at: ISO 8601 UTC timestamp of original artifact
  - ingested_at: ISO 8601 UTC timestamp of store ingestion
  - source_run_id: run directory name that produced the artifact
  - source_host: hostname (or "unknown")
  - provenance: human-readable origin description
  - canonical_path: path within the project to the stored artifact

External paths (starting with /mnt/ or outside project root) are NEVER stored
as canonical evidence. Attempts to ingest from such paths raise ExternalPathError.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from iris_bot.config import Settings


_MANIFEST_FILENAME = "evidence_store_manifest.json"
_SCHEMA_VERSION = 1

# Artifact types this store manages (others are rejected)
MANAGED_ARTIFACT_TYPES = frozenset({"lifecycle_reconciliation", "symbol_stability"})


class ExternalPathError(Exception):
    """Raised when an artifact path is outside the project boundary."""


class UnknownArtifactTypeError(Exception):
    """Raised when an artifact type is not managed by the evidence store."""


def evidence_store_dir(settings: Settings) -> Path:
    return settings.data.runtime_dir / "evidence_store"


def _manifest_path(settings: Settings) -> Path:
    return evidence_store_dir(settings) / _MANIFEST_FILENAME


def _coerce_json_dict(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Expected JSON object")
    return cast(dict[str, Any], raw)


def _load_manifest(settings: Settings) -> dict[str, Any]:
    path = _manifest_path(settings)
    if not path.exists():
        return {"schema_version": _SCHEMA_VERSION, "entries": []}
    try:
        raw = _coerce_json_dict(json.loads(path.read_text(encoding="utf-8")))
        if not isinstance(raw.get("entries"), list):
            return {"schema_version": _SCHEMA_VERSION, "entries": []}
        return raw
    except (OSError, json.JSONDecodeError, ValueError):
        return {"schema_version": _SCHEMA_VERSION, "entries": []}


def _save_manifest(settings: Settings, manifest: dict[str, Any]) -> None:
    store_dir = evidence_store_dir(settings)
    store_dir.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(settings)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _is_external_path(settings: Settings, path: Path) -> bool:
    """Returns True if the path is outside the project boundary."""
    try:
        path.resolve().relative_to(settings.project_root.resolve())
        return False
    except ValueError:
        return True


def _checksum_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry_id(artifact_type: str, symbol: str | None, source_run_id: str) -> str:
    tag = symbol or "global"
    return f"{artifact_type}__{tag}__{source_run_id}"


def ingest_evidence(
    settings: Settings,
    source_path: Path,
    artifact_type: str,
    source_run_id: str,
    symbol: str | None = None,
    provenance: str = "local_run",
    source_host: str | None = None,
) -> dict[str, Any]:
    """
    Materializes an artifact from a run directory into the canonical evidence store.

    - source_path must be WITHIN the project root (no external Windows paths)
    - artifact_type must be in MANAGED_ARTIFACT_TYPES
    - An existing entry for the same (artifact_type, symbol, source_run_id) is
      replaced (idempotent for the same run)
    - Returns the manifest entry that was created

    Raises:
        ExternalPathError: if source_path is outside project root
        UnknownArtifactTypeError: if artifact_type is not managed
        FileNotFoundError: if source_path does not exist
    """
    if artifact_type not in MANAGED_ARTIFACT_TYPES:
        raise UnknownArtifactTypeError(
            f"Artifact type {artifact_type!r} is not managed by the evidence store. "
            f"Managed types: {sorted(MANAGED_ARTIFACT_TYPES)}"
        )
    if not source_path.exists():
        raise FileNotFoundError(f"Source artifact does not exist: {source_path}")
    if _is_external_path(settings, source_path):
        raise ExternalPathError(
            f"Cannot ingest evidence from external path: {source_path}. "
            "Evidence must originate from within the project root. "
            "External Windows workspace paths (/mnt/c/...) are not accepted as canonical evidence."
        )

    # Auto-expire stale entries before ingesting, if configured.
    max_age = settings.operational.evidence_max_age_days
    if max_age is not None:
        expire_evidence(settings, max_age)

    store_dir = evidence_store_dir(settings)
    store_dir.mkdir(parents=True, exist_ok=True)

    entry_id = _entry_id(artifact_type, symbol, source_run_id)
    # Use a stable filename based on entry_id
    dest_filename = f"{entry_id}.json"
    dest_path = store_dir / dest_filename

    # Copy atomically
    tmp_dest = dest_path.with_suffix(".json.tmp")
    shutil.copy2(str(source_path), str(tmp_dest))
    os.replace(tmp_dest, dest_path)

    checksum = _checksum_file(dest_path)

    # Parse created_at from artifact envelope if available
    created_at = datetime.now(tz=UTC).isoformat()
    try:
        raw = json.loads(dest_path.read_text(encoding="utf-8"))
        if "generated_at" in raw:
            created_at = str(raw["generated_at"])
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    entry: dict[str, Any] = {
        "entry_id": entry_id,
        "artifact_type": artifact_type,
        "symbol": symbol,
        "checksum": checksum,
        "created_at": created_at,
        "ingested_at": datetime.now(tz=UTC).isoformat(),
        "source_run_id": source_run_id,
        "source_host": source_host or platform.node() or "unknown",
        "provenance": provenance,
        "canonical_path": str(dest_path.relative_to(settings.project_root)),
        "canonical_abs_path": str(dest_path),
    }

    # Update manifest atomically: replace existing entry or append
    manifest = _load_manifest(settings)
    entries = [e for e in manifest["entries"] if e.get("entry_id") != entry_id]
    entries.append(entry)
    manifest["entries"] = entries
    _save_manifest(settings, manifest)

    return entry


def get_latest_evidence(
    settings: Settings,
    artifact_type: str,
    symbol: str | None = None,
) -> dict[str, Any] | None:
    """
    Returns the latest evidence manifest entry for artifact_type/symbol.

    Uses created_at for ordering (most recent artifact, not most recently ingested).
    Returns None if no matching evidence exists in the store.
    """
    manifest = _load_manifest(settings)
    entries = [
        e for e in manifest["entries"]
        if e.get("artifact_type") == artifact_type
        and (symbol is None or e.get("symbol") == symbol)
    ]
    if not entries:
        return None
    entries.sort(key=lambda e: e.get("created_at", ""))
    return cast(dict[str, Any], entries[-1])


def get_latest_evidence_payload(
    settings: Settings,
    artifact_type: str,
    symbol: str | None = None,
) -> dict[str, Any] | None:
    """
    Returns the deserialized payload of the latest evidence artifact.

    Verifies checksum before returning. Returns None on any error.
    """
    entry = get_latest_evidence(settings, artifact_type, symbol)
    if entry is None:
        return None
    path = Path(entry.get("canonical_abs_path", ""))
    if not path.exists():
        return None
    # Verify integrity
    current_checksum = _checksum_file(path)
    if current_checksum != entry.get("checksum", ""):
        return None
    try:
        raw = _coerce_json_dict(json.loads(path.read_text(encoding="utf-8")))
        # Unwrap artifact envelope if present
        if "payload" in raw and "artifact_type" in raw:
            return cast(dict[str, Any], raw["payload"])
        return raw
    except (OSError, json.JSONDecodeError):
        return None


def expire_evidence(
    settings: Settings,
    max_age_days: float,
    reference_time: datetime | None = None,
) -> list[str]:
    """
    Removes manifest entries (and their canonical files) older than ``max_age_days``.

    Age is measured from ``created_at`` (original artifact timestamp).
    Returns the list of entry_ids that were expired.
    """
    if max_age_days <= 0:
        raise ValueError("max_age_days must be positive")

    now = reference_time or datetime.now(tz=UTC)
    manifest = _load_manifest(settings)
    kept: list[dict[str, Any]] = []
    expired_ids: list[str] = []

    for entry in manifest["entries"]:
        created_raw = entry.get("created_at", "")
        try:
            ts = datetime.fromisoformat(created_raw)
            if ts.tzinfo is None:
                from datetime import timezone
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (now - ts).total_seconds() / 86400.0
        except (ValueError, TypeError):
            age_days = 0.0  # unparseable timestamp → keep entry

        if age_days > max_age_days:
            expired_ids.append(entry.get("entry_id", ""))
            # Remove physical file if it still exists
            abs_path = Path(entry.get("canonical_abs_path", ""))
            if abs_path.exists():
                try:
                    abs_path.unlink()
                except OSError:
                    pass
        else:
            kept.append(entry)

    if expired_ids:
        manifest["entries"] = kept
        _save_manifest(settings, manifest)

    return expired_ids


def detect_contradictions(settings: Settings) -> list[dict[str, Any]]:
    """
    Detects conflicting evidence entries for the same (artifact_type, symbol) key.

    Two entries are contradictory when they share the same key but differ in checksum,
    which can indicate that two independent runs produced incompatible artifacts.

    Returns a list of contradiction records, each containing:
      - key: the (artifact_type, symbol) identifier
      - entry_ids: the conflicting entry IDs
      - checksums: the distinct checksum values
    """
    manifest = _load_manifest(settings)
    by_key: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest["entries"]:
        key = f"{entry.get('artifact_type', '')}::{entry.get('symbol') or 'global'}"
        by_key.setdefault(key, []).append(entry)

    contradictions: list[dict[str, Any]] = []
    for key, entries in sorted(by_key.items()):
        distinct_checksums = {e.get("checksum", "") for e in entries}
        if len(distinct_checksums) > 1:
            contradictions.append({
                "key": key,
                "entry_ids": [e.get("entry_id") for e in entries],
                "checksums": sorted(distinct_checksums),
                "count": len(entries),
            })
    return contradictions


def query_evidence_by_date_range(
    settings: Settings,
    artifact_type: str,
    start: datetime,
    end: datetime,
    symbol: str | None = None,
) -> list[dict[str, Any]]:
    """
    Returns manifest entries for ``artifact_type`` whose ``created_at`` falls
    within [start, end] (inclusive).  Optionally filtered by symbol.

    Results are ordered by ``created_at`` ascending.
    """
    manifest = _load_manifest(settings)
    results: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        if entry.get("artifact_type") != artifact_type:
            continue
        if symbol is not None and entry.get("symbol") != symbol:
            continue
        created_raw = entry.get("created_at", "")
        try:
            ts = datetime.fromisoformat(created_raw)
            if ts.tzinfo is None:
                from datetime import timezone
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        if start <= ts <= end:
            results.append(entry)
    results.sort(key=lambda e: e.get("created_at", ""))
    return results


def evidence_store_status(settings: Settings) -> dict[str, Any]:
    """
    Returns a full audit report of the evidence store.

    Includes:
      - Total entries by artifact_type
      - Latest entry per (artifact_type, symbol)
      - Integrity check for all entries (checksum verification)
      - Whether any external paths were rejected (checked via manifest provenance)
    """
    manifest = _load_manifest(settings)
    entries = manifest["entries"]

    by_type: dict[str, int] = {}
    integrity_failures: list[str] = []
    latest_per_key: dict[str, dict[str, Any]] = {}

    for entry in entries:
        at = entry.get("artifact_type", "unknown")
        sym = entry.get("symbol")
        by_type[at] = by_type.get(at, 0) + 1

        # Check integrity
        abs_path = Path(entry.get("canonical_abs_path", ""))
        if abs_path.exists():
            current = _checksum_file(abs_path)
            if current != entry.get("checksum", ""):
                integrity_failures.append(entry.get("entry_id", "?"))
        else:
            integrity_failures.append(f"missing:{entry.get('entry_id', '?')}")

        # Track latest per key
        key = f"{at}__{sym or 'global'}"
        existing = latest_per_key.get(key)
        if existing is None or entry.get("created_at", "") > existing.get("created_at", ""):
            latest_per_key[key] = entry

    return {
        "store_dir": str(evidence_store_dir(settings)),
        "manifest_path": str(_manifest_path(settings)),
        "total_entries": len(entries),
        "by_artifact_type": by_type,
        "integrity_failures": integrity_failures,
        "integrity_ok": len(integrity_failures) == 0,
        "latest_per_key": {
            key: {
                "entry_id": e.get("entry_id"),
                "created_at": e.get("created_at"),
                "source_run_id": e.get("source_run_id"),
                "source_host": e.get("source_host"),
                "provenance": e.get("provenance"),
            }
            for key, e in sorted(latest_per_key.items())
        },
    }
