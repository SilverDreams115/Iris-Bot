"""
Evidence discovery and gate metric computation for governance decisions.

Internal module used by governance.py — do not import directly from outside the package.
Public API lives in iris_bot.governance.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from iris_bot.artifacts import read_artifact_payload
from iris_bot.config import Settings
from iris_bot.evidence_store import get_latest_evidence, get_latest_evidence_payload
from iris_bot.profile_registry import _latest_run


# ---------------------------------------------------------------------------
# Path guard
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evidence discovery
# ---------------------------------------------------------------------------

def _latest_endurance_evidence(
    settings: Settings, symbol: str
) -> dict[str, Any] | None:
    """Returns the most recent endurance run that contains data for the given symbol."""
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


def _latest_lifecycle_evidence(settings: Settings) -> dict[str, Any] | None:
    """
    Returns the latest lifecycle reconciliation evidence for governance decisions.

    Evidence discovery order (most authoritative first):
      1. Canonical evidence store (data/runtime/evidence_store/) — project-internal
      2. Local lifecycle_reconciliation run (within project root only)
      3. mt5_windows_stabilization run — ONLY if referenced path is within project root

    External paths (/mnt/c/..., paths outside project root) are REJECTED.
    """
    candidates: list[dict[str, Any]] = []

    # Priority 1: Canonical evidence store
    store_entry = get_latest_evidence(settings, "lifecycle_reconciliation")
    if store_entry is not None:
        canonical_path = Path(store_entry.get("canonical_abs_path", ""))
        if canonical_path.exists() and _is_within_project(settings, canonical_path):
            try:
                payload = get_latest_evidence_payload(settings, "lifecycle_reconciliation")
                if payload is not None:
                    candidates.append({
                        "origin": "canonical_evidence_store",
                        "report_path": canonical_path,
                        "generated_at": store_entry.get("created_at", ""),
                        "payload": payload,
                        "audit_ok": None,
                        "source_run_id": store_entry.get("source_run_id", ""),
                        "provenance": store_entry.get("provenance", ""),
                    })
            except (OSError, ValueError):
                pass

    # Priority 2: Local lifecycle_reconciliation run
    latest_local = _latest_run(settings, "lifecycle_reconciliation")
    if latest_local is not None and _is_within_project(settings, latest_local):
        report_path = latest_local / "lifecycle_reconciliation_report.json"
        if report_path.exists():
            try:
                candidates.append({
                    "origin": "local_lifecycle_reconciliation",
                    "report_path": report_path,
                    "generated_at": _artifact_generated_at(report_path),
                    "payload": read_artifact_payload(report_path, expected_type="lifecycle_reconciliation"),
                    "audit_ok": None,
                })
            except (OSError, ValueError):
                pass

    # Priority 3: mt5_windows_stabilization — ONLY if referenced path is within project
    latest_stabilization = _latest_run(settings, "mt5_windows_stabilization")
    if latest_stabilization is not None and _is_within_project(settings, latest_stabilization):
        stabilization_path = latest_stabilization / "lifecycle_rerun_report.json"
        stabilization = _plain_json(stabilization_path)
        if stabilization:
            referenced_path = Path(str(stabilization.get("reconciliation_run", "")))
            if referenced_path.exists() and _is_within_project(settings, referenced_path):
                try:
                    candidates.append({
                        "origin": "windows_native_stabilization",
                        "report_path": referenced_path,
                        "generated_at": _artifact_generated_at(referenced_path),
                        "payload": read_artifact_payload(referenced_path, expected_type="lifecycle_reconciliation"),
                        "audit_ok": bool(stabilization.get("audit_ok", False)),
                        "stabilization_run": str(stabilization_path),
                        "audit_run": stabilization.get("audit_run", ""),
                        "rerun_source": stabilization.get("rerun_source", ""),
                    })
                except (OSError, ValueError):
                    pass

    if not candidates:
        return None
    candidates.sort(key=lambda item: (str(item.get("generated_at", "")), str(item.get("report_path", ""))))
    return candidates[-1]


def _latest_lifecycle_payload(settings: Settings) -> dict[str, Any] | None:
    evidence = _latest_lifecycle_evidence(settings)
    return None if evidence is None else evidence["payload"]


# ---------------------------------------------------------------------------
# Evidence age
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Gate metrics computation
# ---------------------------------------------------------------------------

def _compute_endurance_gate_metrics(endurance_symbol: dict[str, Any]) -> dict[str, Any]:
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
