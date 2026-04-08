"""Forward-validation session series runtime registry."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_bot.artifacts import read_artifact_payload, wrap_artifact
from iris_bot.config import Settings
from iris_bot.demo_forward_evidence import build_demo_series_evidence, write_demo_series_evidence
from iris_bot.durable_io import durable_write_json
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.session_discipline import review_demo_series


__all__ = [
    "DemoSessionSeriesState",
    "build_series_runtime_paths",
    "load_series_registry",
    "start_demo_session_series",
    "ensure_active_demo_session_series",
    "record_demo_session_result",
    "close_demo_session_series",
    "start_demo_session_series_command",
    "demo_forward_series_status_command",
    "close_demo_session_series_command",
]


def _resolve_series_path(settings: Settings, stored_path: str) -> Path:
    """Resolve a stored evidence/review path to an absolute Path for the current environment.

    Paths are now stored relative to project_root.  Older entries may contain
    absolute WSL paths (e.g. ``/home/silver/.../runs/...``).  On Windows the
    workspace root is different, so absolute WSL paths are remapped heuristically
    by finding the first ``runs`` or ``data`` ancestor and resolving relative to
    the current project_root.
    """
    p = Path(stored_path)
    if not p.is_absolute():
        return settings.project_root / p
    if p.exists():
        return p
    # Heuristic: find the first "runs" or "data" segment and treat everything
    # from there as relative to the current project root.
    parts = p.parts
    for i, part in enumerate(parts):
        if part in ("runs", "data"):
            candidate = settings.project_root / Path(*parts[i:])
            if candidate.exists():
                return candidate
    return p  # unchanged — will produce a clear FileNotFoundError


DEFAULT_TARGET_SESSIONS = 3


@dataclass(frozen=True)
class DemoSessionSeriesState:
    session_series_id: str
    symbol: str
    status: str
    started_at: str
    ended_at: str | None
    target_sessions: int
    session_ids: list[str] = field(default_factory=list)
    session_evidence_paths: list[str] = field(default_factory=list)
    session_review_paths: list[str] = field(default_factory=list)
    aggregate_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_series_id": self.session_series_id,
            "symbol": self.symbol,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "target_sessions": self.target_sessions,
            "session_ids": self.session_ids,
            "session_evidence_paths": self.session_evidence_paths,
            "session_review_paths": self.session_review_paths,
            "aggregate_counts": self.aggregate_counts,
        }


def build_series_runtime_paths(settings: Settings) -> dict[str, Path]:
    base = settings.data.runtime_dir / "demo_forward_validation"
    return {
        "base": base,
        "registry": base / "demo_session_series_registry.json",
        "artifacts": base / "series_artifacts",
    }


def _default_registry() -> dict[str, Any]:
    return {
        "active_series_id": "",
        "series": {},
        "last_updated_at": None,
    }


def load_series_registry(settings: Settings) -> dict[str, Any]:
    path = build_series_runtime_paths(settings)["registry"]
    if not path.exists():
        return _default_registry()
    return read_artifact_payload(path, expected_type="demo_session_series")


def _save_series_registry(settings: Settings, registry: dict[str, Any]) -> Path:
    path = build_series_runtime_paths(settings)["registry"]
    payload = dict(registry)
    payload["last_updated_at"] = datetime.now(tz=UTC).isoformat()
    durable_write_json(path, wrap_artifact("demo_session_series", payload))
    return path


def _new_series_id(symbol: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{symbol.lower()}_{stamp}_forward_series"


def _empty_counts() -> dict[str, int]:
    return {
        "successful_sessions": 0,
        "aborted_sessions": 0,
        "hold_sessions": 0,
        "orders_sent": 0,
        "orders_rejected": 0,
        "reconciliations": 0,
        "recoveries": 0,
        "restore_events": 0,
        "circuit_breaker_triggers": 0,
        "kill_switch_events": 0,
        "sessions_with_divergence": 0,
        "sessions_with_recovery": 0,
    }


def start_demo_session_series(
    settings: Settings,
    *,
    symbol: str,
    target_sessions: int = DEFAULT_TARGET_SESSIONS,
) -> DemoSessionSeriesState:
    registry = load_series_registry(settings)
    series_id = _new_series_id(symbol)
    state = DemoSessionSeriesState(
        session_series_id=series_id,
        symbol=symbol,
        status="active",
        started_at=datetime.now(tz=UTC).isoformat(),
        ended_at=None,
        target_sessions=max(target_sessions, 1),
        aggregate_counts=_empty_counts(),
    )
    series = dict(registry.get("series", {}))
    series[series_id] = state.to_dict()
    registry["series"] = series
    registry["active_series_id"] = series_id
    _save_series_registry(settings, registry)
    return state


def ensure_active_demo_session_series(settings: Settings, *, symbol: str) -> DemoSessionSeriesState:
    registry = load_series_registry(settings)
    active_id = str(registry.get("active_series_id", ""))
    series = dict(registry.get("series", {}))
    if active_id:
        active = dict(series.get(active_id, {}))
        if active and active.get("symbol") == symbol and active.get("status") == "active":
            return DemoSessionSeriesState(
                session_series_id=str(active["session_series_id"]),
                symbol=str(active["symbol"]),
                status=str(active["status"]),
                started_at=str(active["started_at"]),
                ended_at=active.get("ended_at"),
                target_sessions=int(active.get("target_sessions", DEFAULT_TARGET_SESSIONS) or DEFAULT_TARGET_SESSIONS),
                session_ids=list(active.get("session_ids", [])),
                session_evidence_paths=list(active.get("session_evidence_paths", [])),
                session_review_paths=list(active.get("session_review_paths", [])),
                aggregate_counts=dict(active.get("aggregate_counts", _empty_counts())),
            )
    return start_demo_session_series(settings, symbol=symbol)


def record_demo_session_result(
    settings: Settings,
    *,
    session_series_id: str,
    session_id: str,
    session_evidence_path: Path,
    session_review_path: Path,
    session_evidence_payload: dict[str, Any],
    session_review_payload: dict[str, Any],
) -> dict[str, Any]:
    registry = load_series_registry(settings)
    series_map = dict(registry.get("series", {}))
    current = dict(series_map.get(session_series_id, {}))
    if not current:
        raise ValueError(f"unknown_session_series_id:{session_series_id}")

    counts = dict(_empty_counts())
    counts.update(current.get("aggregate_counts", {}))
    review_classification = str(session_review_payload.get("classification", "failed"))
    if review_classification == "healthy":
        counts["successful_sessions"] += 1
    elif review_classification == "caution":
        counts["hold_sessions"] += 1
    else:
        counts["aborted_sessions"] += 1
    counts["orders_sent"] += int((session_evidence_payload.get("execution_summary") or {}).get("orders_sent", 0) or 0)
    counts["orders_rejected"] += int((session_evidence_payload.get("execution_summary") or {}).get("orders_rejected", 0) or 0)
    counts["reconciliations"] += int((session_evidence_payload.get("divergence_summary") or {}).get("reconcile_events", 0) or 0)
    counts["recoveries"] += int((session_evidence_payload.get("restore_recovery_summary") or {}).get("recovery_events", 0) or 0)
    counts["restore_events"] += int((session_evidence_payload.get("restore_recovery_summary") or {}).get("restore_events", 0) or 0)
    divergence_events = int((session_evidence_payload.get("divergence_summary") or {}).get("divergence_events", 0) or 0)
    recovery_events = int((session_evidence_payload.get("restore_recovery_summary") or {}).get("recovery_events", 0) or 0)
    final_state = dict(session_evidence_payload.get("final_state_summary", {}))
    if divergence_events > 0:
        counts["sessions_with_divergence"] += 1
    if recovery_events > 0:
        counts["sessions_with_recovery"] += 1
    if bool(final_state.get("circuit_breaker_triggered", False)):
        counts["circuit_breaker_triggers"] += 1
    if bool(final_state.get("kill_switch_active", False)):
        counts["kill_switch_events"] += 1

    session_ids = list(current.get("session_ids", []))
    session_evidence_paths = list(current.get("session_evidence_paths", []))
    session_review_paths = list(current.get("session_review_paths", []))
    if session_id not in session_ids:
        session_ids.append(session_id)
        try:
            session_evidence_paths.append(str(session_evidence_path.relative_to(settings.project_root)))
            session_review_paths.append(str(session_review_path.relative_to(settings.project_root)))
        except ValueError:
            session_evidence_paths.append(str(session_evidence_path))
            session_review_paths.append(str(session_review_path))

    current.update(
        {
            "session_ids": session_ids,
            "session_evidence_paths": session_evidence_paths,
            "session_review_paths": session_review_paths,
            "aggregate_counts": counts,
        }
    )

    session_payloads = [read_artifact_payload(_resolve_series_path(settings, path), expected_type="demo_session_evidence") for path in session_evidence_paths]
    review_payloads = [read_artifact_payload(_resolve_series_path(settings, path), expected_type="demo_session_review") for path in session_review_paths]
    series_review = review_demo_series(
        session_series_id=session_series_id,
        session_reviews=review_payloads,
        aggregate_counts=counts,
        target_sessions=int(current.get("target_sessions", DEFAULT_TARGET_SESSIONS) or DEFAULT_TARGET_SESSIONS),
    )
    current["latest_series_review"] = series_review.to_dict()

    if len(session_ids) >= int(current.get("target_sessions", DEFAULT_TARGET_SESSIONS) or DEFAULT_TARGET_SESSIONS):
        current["status"] = "completed"
        current["ended_at"] = datetime.now(tz=UTC).isoformat()

    series_map[session_series_id] = current
    registry["series"] = series_map
    if current.get("status") != "active" and registry.get("active_series_id") == session_series_id:
        registry["active_series_id"] = ""
    _save_series_registry(settings, registry)

    paths = build_series_runtime_paths(settings)
    artifact_path = paths["artifacts"] / f"{session_series_id}.json"
    series_evidence = build_demo_series_evidence(
        session_series_id=session_series_id,
        symbol=str(current.get("symbol", "")),
        status=str(current.get("status", "active")),
        started_at=str(current.get("started_at", "")),
        ended_at=current.get("ended_at"),
        target_sessions=int(current.get("target_sessions", DEFAULT_TARGET_SESSIONS) or DEFAULT_TARGET_SESSIONS),
        session_payloads=session_payloads,
        session_reviews=review_payloads,
        aggregate_counts=counts,
    )
    write_demo_series_evidence(artifact_path, series_evidence)
    return {
        "series_state": current,
        "series_evidence_path": str(artifact_path),
        "series_review": series_review.to_dict(),
    }


def close_demo_session_series(settings: Settings, *, session_series_id: str, status: str = "closed") -> dict[str, Any]:
    registry = load_series_registry(settings)
    series_map = dict(registry.get("series", {}))
    current = dict(series_map.get(session_series_id, {}))
    if not current:
        raise ValueError(f"unknown_session_series_id:{session_series_id}")
    current["status"] = status
    current["ended_at"] = datetime.now(tz=UTC).isoformat()
    series_map[session_series_id] = current
    registry["series"] = series_map
    if registry.get("active_series_id") == session_series_id:
        registry["active_series_id"] = ""
    _save_series_registry(settings, registry)
    return current


def start_demo_session_series_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "start_demo_forward_series")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    symbol = settings.demo_execution.target_symbol
    if not symbol:
        write_json_report(run_dir, "demo_session_series_status.json", wrap_artifact("demo_session_series", {"ok": False, "reason": "demo_execution_target_symbol_missing"}))
        logger.error("start_demo_forward_series blocked: demo_execution_target_symbol_missing")
        return 2
    state = start_demo_session_series(settings, symbol=symbol)
    write_json_report(run_dir, "demo_session_series_status.json", wrap_artifact("demo_session_series", {"ok": True, "series": state.to_dict()}))
    logger.info("start_demo_forward_series session_series_id=%s symbol=%s run_dir=%s", state.session_series_id, symbol, run_dir)
    return 0


def demo_forward_series_status_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "demo_forward_series_status")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    registry = load_series_registry(settings)
    latest = None
    for candidate in dict(registry.get("series", {})).values():
        if latest is None or str(candidate.get("started_at", "")) > str(latest.get("started_at", "")):
            latest = dict(candidate)
    payload = {
        "ok": True,
        "active_series_id": registry.get("active_series_id", ""),
        "latest_series": latest,
        "registry": registry,
    }
    write_json_report(run_dir, "demo_session_series_status.json", wrap_artifact("demo_session_series", payload))
    logger.info("demo_forward_series_status active_series_id=%s run_dir=%s", registry.get("active_series_id", ""), run_dir)
    return 0


def close_demo_session_series_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "close_demo_forward_series")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    registry = load_series_registry(settings)
    active_id = str(registry.get("active_series_id", ""))
    if not active_id:
        write_json_report(run_dir, "demo_session_series_status.json", wrap_artifact("demo_session_series", {"ok": False, "reason": "no_active_demo_session_series"}))
        logger.error("close_demo_forward_series blocked: no_active_demo_session_series")
        return 2
    state = close_demo_session_series(settings, session_series_id=active_id, status="closed")
    write_json_report(run_dir, "demo_session_series_status.json", wrap_artifact("demo_session_series", {"ok": True, "series": state}))
    logger.info("close_demo_forward_series session_series_id=%s run_dir=%s", active_id, run_dir)
    return 0
