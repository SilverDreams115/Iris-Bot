"""Demo forward evidence consolidation.

Builds per-session and per-series evidence artifacts linking preflight,
signals, execution, reconciliation, and recovery into an auditable trail for
serious demo execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from iris_bot.artifacts import wrap_artifact
from iris_bot.durable_io import durable_write_json


__all__ = [
    "DemoForwardMetrics",
    "DemoSessionEvidence",
    "DemoSeriesEvidence",
    "build_demo_session_evidence",
    "build_demo_series_evidence",
    "write_demo_session_evidence",
    "write_demo_series_evidence",
]


@dataclass(frozen=True)
class DemoForwardMetrics:
    """Minimal forward metrics kept reconstructible from session evidence."""

    sessions_counted: int
    realized_pnl_usd: float
    average_position_lifetime_seconds: float
    block_rate: float
    reject_rate: float
    divergence_rate: float
    recovery_rate: float
    reconcile_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DemoSessionEvidence:
    """Auditable evidence artifact for a single demo execution session."""

    session_id: str
    session_series_id: str
    series_position: int
    symbol: str
    start_time: str
    end_time: str
    preflight_ok: bool
    preflight_checks: dict[str, Any]
    signal_summary: dict[str, Any]
    execution_summary: dict[str, Any]
    trade_summary: dict[str, Any]
    performance_summary: dict[str, Any]
    divergence_summary: dict[str, Any]
    restore_recovery_summary: dict[str, Any]
    session_decision_log: list[dict[str, Any]]
    final_state_summary: dict[str, Any]
    artifact_paths: dict[str, str]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DemoSeriesEvidence:
    """Auditable evidence artifact for a forward-validation series."""

    session_series_id: str
    symbol: str
    status: str
    started_at: str
    ended_at: str | None
    target_sessions: int
    session_ids: list[str]
    session_reviews: list[dict[str, Any]]
    aggregate_counts: dict[str, int]
    signal_summary: dict[str, Any]
    execution_summary: dict[str, Any]
    trade_summary: dict[str, Any]
    divergence_summary: dict[str, Any]
    restore_recovery_summary: dict[str, Any]
    forward_metrics: dict[str, Any]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _count_true(items: list[dict[str, Any]], key: str) -> int:
    return sum(1 for item in items if bool(item.get(key, False)))


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _build_forward_metrics(session_payloads: list[dict[str, Any]]) -> DemoForwardMetrics:
    lifetime_values = [
        float((payload.get("performance_summary") or {}).get("position_lifetime_seconds", 0.0) or 0.0)
        for payload in session_payloads
    ]
    block_events = sum(int((payload.get("signal_summary") or {}).get("blocked_signals", 0) or 0) for payload in session_payloads)
    signals = sum(int((payload.get("signal_summary") or {}).get("signals_generated", 0) or 0) for payload in session_payloads)
    rejected = sum(int((payload.get("execution_summary") or {}).get("orders_rejected", 0) or 0) for payload in session_payloads)
    orders_sent = sum(int((payload.get("execution_summary") or {}).get("orders_sent", 0) or 0) for payload in session_payloads)
    divergence_events = sum(int((payload.get("divergence_summary") or {}).get("divergence_events", 0) or 0) for payload in session_payloads)
    reconcile_events = sum(int((payload.get("divergence_summary") or {}).get("reconcile_events", 0) or 0) for payload in session_payloads)
    recovery_events = sum(int((payload.get("restore_recovery_summary") or {}).get("recovery_events", 0) or 0) for payload in session_payloads)
    restore_events = sum(int((payload.get("restore_recovery_summary") or {}).get("restore_events", 0) or 0) for payload in session_payloads)
    realized_pnl = sum(float((payload.get("performance_summary") or {}).get("realized_pnl_usd", 0.0) or 0.0) for payload in session_payloads)
    return DemoForwardMetrics(
        sessions_counted=len(session_payloads),
        realized_pnl_usd=round(realized_pnl, 8),
        average_position_lifetime_seconds=round(_safe_mean(lifetime_values), 4),
        block_rate=round(block_events / signals, 6) if signals > 0 else 0.0,
        reject_rate=round(rejected / orders_sent, 6) if orders_sent > 0 else 0.0,
        divergence_rate=round(divergence_events / reconcile_events, 6) if reconcile_events > 0 else 0.0,
        recovery_rate=round(recovery_events / restore_events, 6) if restore_events > 0 else 0.0,
        reconcile_rate=round(reconcile_events / len(session_payloads), 6) if session_payloads else 0.0,
    )


def build_demo_session_evidence(
    *,
    session_id: str,
    symbol: str,
    start_time: str,
    end_time: str,
    preflight_report: dict[str, Any],
    session_series_id: str = "",
    series_position: int = 0,
    trades_opened: int = 0,
    trades_closed: int = 0,
    signals_evaluated: int = 0,
    no_trade_signals: int = 0,
    blocked_signals: int = 0,
    orders_sent: int = 0,
    orders_rejected: int = 0,
    session_decision_log: list[dict[str, Any]] | None = None,
    final_state_summary: dict[str, Any] | None = None,
    signal_summary: dict[str, Any] | None = None,
    execution_summary: dict[str, Any] | None = None,
    trade_summary: dict[str, Any] | None = None,
    performance_summary: dict[str, Any] | None = None,
    divergence_summary: dict[str, Any] | None = None,
    restore_recovery_summary: dict[str, Any] | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> DemoSessionEvidence:
    """Build a DemoSessionEvidence artifact from session execution data."""
    signal_payload = {
        "signals_generated": signals_evaluated,
        "no_trade_signals": no_trade_signals,
        "blocked_signals": blocked_signals,
    }
    signal_payload.update(signal_summary or {})
    execution_payload = {
        "orders_sent": orders_sent,
        "orders_rejected": orders_rejected,
        "decisions_executed": max(orders_sent - orders_rejected, 0),
    }
    execution_payload.update(execution_summary or {})
    trade_payload = {
        "trades_opened": trades_opened,
        "trades_closed": trades_closed,
    }
    trade_payload.update(trade_summary or {})
    performance_payload = {
        "realized_pnl_usd": 0.0,
        "position_lifetime_seconds": 0.0,
    }
    performance_payload.update(performance_summary or {})
    divergence_payload = {
        "divergence_events": 0,
        "reconcile_events": 0,
        "broker_local_divergences": [],
    }
    divergence_payload.update(divergence_summary or {})
    restore_payload = {
        "restore_events": 0,
        "recovery_events": 0,
        "recovery_details": [],
    }
    restore_payload.update(restore_recovery_summary or {})
    return DemoSessionEvidence(
        session_id=session_id,
        session_series_id=session_series_id,
        series_position=series_position,
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        preflight_ok=bool(preflight_report.get("ok", False)),
        preflight_checks=dict(preflight_report.get("checks", {})),
        signal_summary=signal_payload,
        execution_summary=execution_payload,
        trade_summary=trade_payload,
        performance_summary=performance_payload,
        divergence_summary=divergence_payload,
        restore_recovery_summary=restore_payload,
        session_decision_log=session_decision_log or [],
        final_state_summary=final_state_summary or {},
        artifact_paths=dict(artifact_paths or {}),
        generated_at=datetime.now(tz=UTC).isoformat(),
    )


def build_demo_series_evidence(
    *,
    session_series_id: str,
    symbol: str,
    status: str,
    started_at: str,
    ended_at: str | None,
    target_sessions: int,
    session_payloads: list[dict[str, Any]],
    session_reviews: list[dict[str, Any]],
    aggregate_counts: dict[str, int],
) -> DemoSeriesEvidence:
    """Aggregate session evidence into a forward-validation series artifact."""
    signal_summary = {
        "signals_generated": sum(int((payload.get("signal_summary") or {}).get("signals_generated", 0) or 0) for payload in session_payloads),
        "no_trade_signals": sum(int((payload.get("signal_summary") or {}).get("no_trade_signals", 0) or 0) for payload in session_payloads),
        "blocked_signals": sum(int((payload.get("signal_summary") or {}).get("blocked_signals", 0) or 0) for payload in session_payloads),
    }
    execution_summary = {
        "orders_sent": sum(int((payload.get("execution_summary") or {}).get("orders_sent", 0) or 0) for payload in session_payloads),
        "orders_rejected": sum(int((payload.get("execution_summary") or {}).get("orders_rejected", 0) or 0) for payload in session_payloads),
        "decisions_executed": sum(int((payload.get("execution_summary") or {}).get("decisions_executed", 0) or 0) for payload in session_payloads),
    }
    trade_summary = {
        "trades_opened": sum(int((payload.get("trade_summary") or {}).get("trades_opened", 0) or 0) for payload in session_payloads),
        "trades_closed": sum(int((payload.get("trade_summary") or {}).get("trades_closed", 0) or 0) for payload in session_payloads),
        "positions_with_lifetime": _count_true(session_payloads, "performance_summary"),
    }
    divergence_summary = {
        "divergence_events": sum(int((payload.get("divergence_summary") or {}).get("divergence_events", 0) or 0) for payload in session_payloads),
        "reconcile_events": sum(int((payload.get("divergence_summary") or {}).get("reconcile_events", 0) or 0) for payload in session_payloads),
        "sessions_with_divergence": sum(1 for payload in session_payloads if int((payload.get("divergence_summary") or {}).get("divergence_events", 0) or 0) > 0),
    }
    restore_recovery_summary = {
        "restore_events": sum(int((payload.get("restore_recovery_summary") or {}).get("restore_events", 0) or 0) for payload in session_payloads),
        "recovery_events": sum(int((payload.get("restore_recovery_summary") or {}).get("recovery_events", 0) or 0) for payload in session_payloads),
        "sessions_with_recovery": sum(1 for payload in session_payloads if int((payload.get("restore_recovery_summary") or {}).get("recovery_events", 0) or 0) > 0),
    }
    forward_metrics = _build_forward_metrics(session_payloads).to_dict()
    return DemoSeriesEvidence(
        session_series_id=session_series_id,
        symbol=symbol,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        target_sessions=target_sessions,
        session_ids=[str(payload.get("session_id", "")) for payload in session_payloads],
        session_reviews=session_reviews,
        aggregate_counts=dict(aggregate_counts),
        signal_summary=signal_summary,
        execution_summary=execution_summary,
        trade_summary=trade_summary,
        divergence_summary=divergence_summary,
        restore_recovery_summary=restore_recovery_summary,
        forward_metrics=forward_metrics,
        generated_at=datetime.now(tz=UTC).isoformat(),
    )


def write_demo_session_evidence(path: Path, evidence: DemoSessionEvidence) -> None:
    """Write demo session evidence artifact as versioned JSON."""
    durable_write_json(path, wrap_artifact("demo_session_evidence", evidence.to_dict()))


def write_demo_series_evidence(path: Path, evidence: DemoSeriesEvidence) -> None:
    """Write demo series evidence artifact as versioned JSON."""
    durable_write_json(path, wrap_artifact("demo_session_series", evidence.to_dict()))
