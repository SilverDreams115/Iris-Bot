"""Kill switch, circuit breaker, and no-trade mode for IRIS-Bot.

Design principles:
- Any activation is irreversible within the session (no auto-reset).
- All transitions are auditable (KillSwitchReport written to disk).
- kill_switch > no_trade in severity; circuit_breaker may trigger either.
- No execution is sent to MT5 via this module.
- Blocked state is persisted via state.blocked_reasons.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from iris_bot.operational import AlertRecord, PaperEngineState


__all__ = [
    "KillSwitchReport",
    "CircuitBreakerCondition",
    "activate_kill_switch",
    "activate_no_trade_mode",
    "circuit_breaker_check",
    "write_kill_switch_report",
    "is_kill_switch_active",
    "is_no_trade_mode_active",
    "build_default_circuit_breaker_conditions",
]


@dataclass(frozen=True)
class KillSwitchReport:
    """Auditable artifact produced by every kill-switch or no-trade activation."""

    event_type: str          # "kill_switch" | "no_trade_mode" | "circuit_breaker"
    triggered_by: str        # "manual" | "circuit_breaker" | "max_daily_loss" | ...
    reason: str
    triggered_at: str
    blocked_reasons_added: list[str]
    prior_blocked_reasons: list[str]
    condition_name: str      # circuit-breaker condition name, or "" if manual
    state_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CircuitBreakerCondition:
    """A named condition that triggers kill-switch or no-trade when true."""

    name: str
    check: Callable[[PaperEngineState], bool]
    severity: str   # "kill_switch" | "no_trade"
    reason: str


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _emit_alert(
    alerts: list[AlertRecord],
    severity: str,
    category: str,
    message: str,
    details: dict[str, Any],
) -> None:
    alerts.append(AlertRecord(
        timestamp=_now_iso(),
        severity=severity,
        category=category,
        message=message,
        details=details,
    ))


def is_kill_switch_active(state: PaperEngineState) -> bool:
    return any(r.startswith("kill_switch:") for r in state.blocked_reasons)


def is_no_trade_mode_active(state: PaperEngineState) -> bool:
    return any(r.startswith("no_trade_mode:") or r.startswith("kill_switch:") for r in state.blocked_reasons)


def activate_kill_switch(
    state: PaperEngineState,
    reason: str,
    triggered_by: str,
    alerts: list[AlertRecord],
    condition_name: str = "",
) -> KillSwitchReport:
    """Activate kill switch. Adds blocked_reason, emits critical alert.

    Idempotent: if already active, still produces a report but does not
    add a second identical blocked_reason.
    """
    prior = list(state.blocked_reasons)
    tag = f"kill_switch:{reason}"
    added: list[str] = []
    if tag not in state.blocked_reasons:
        state.blocked_reasons.append(tag)
        added.append(tag)

    report = KillSwitchReport(
        event_type="kill_switch",
        triggered_by=triggered_by,
        reason=reason,
        triggered_at=_now_iso(),
        blocked_reasons_added=added,
        prior_blocked_reasons=prior,
        condition_name=condition_name,
        state_snapshot={
            "open_positions": len(state.open_positions),
            "blocked_reasons": list(state.blocked_reasons),
        },
    )
    _emit_alert(
        alerts, "critical", "kill_switch_activated",
        f"Kill switch activated: {reason}",
        {"triggered_by": triggered_by, "condition": condition_name, "blocked_reasons": list(state.blocked_reasons)},
    )
    return report


def activate_no_trade_mode(
    state: PaperEngineState,
    reason: str,
    triggered_by: str,
    alerts: list[AlertRecord],
    condition_name: str = "",
) -> KillSwitchReport:
    """Activate no-trade mode (less severe than kill switch).

    Blocks new entries but does not imply immediate shutdown.
    Idempotent.
    """
    prior = list(state.blocked_reasons)
    tag = f"no_trade_mode:{reason}"
    added: list[str] = []
    if tag not in state.blocked_reasons:
        state.blocked_reasons.append(tag)
        added.append(tag)

    report = KillSwitchReport(
        event_type="no_trade_mode",
        triggered_by=triggered_by,
        reason=reason,
        triggered_at=_now_iso(),
        blocked_reasons_added=added,
        prior_blocked_reasons=prior,
        condition_name=condition_name,
        state_snapshot={
            "open_positions": len(state.open_positions),
            "blocked_reasons": list(state.blocked_reasons),
        },
    )
    _emit_alert(
        alerts, "warning", "no_trade_mode_activated",
        f"No-trade mode activated: {reason}",
        {"triggered_by": triggered_by, "condition": condition_name},
    )
    return report


def circuit_breaker_check(
    state: PaperEngineState,
    conditions: list[CircuitBreakerCondition],
    alerts: list[AlertRecord],
) -> KillSwitchReport | None:
    """Evaluate circuit breaker conditions in order.

    Returns the first KillSwitchReport if any condition triggers, else None.
    """
    for cond in conditions:
        if cond.check(state):
            if cond.severity == "kill_switch":
                return activate_kill_switch(
                    state, cond.reason, "circuit_breaker", alerts, cond.name
                )
            if cond.severity == "no_trade":
                return activate_no_trade_mode(
                    state, cond.reason, "circuit_breaker", alerts, cond.name
                )
    return None


def write_kill_switch_report(path: Path, report: KillSwitchReport) -> None:
    """Write kill switch report as JSON to path (auditable artifact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def build_default_circuit_breaker_conditions(
    max_critical_discrepancies: int = 0,
    max_blocked_reasons: int = 3,
) -> list[CircuitBreakerCondition]:
    """Returns the standard set of circuit breaker conditions for demo-guarded operation."""
    return [
        CircuitBreakerCondition(
            name="critical_broker_discrepancy",
            check=lambda s: s.broker_sync_status.critical_discrepancy_count > max_critical_discrepancies,
            severity="kill_switch",
            reason="critical_broker_discrepancy_count_exceeded",
        ),
        CircuitBreakerCondition(
            name="accumulating_blocks",
            check=lambda s: len(s.blocked_reasons) > max_blocked_reasons,
            severity="no_trade",
            reason="too_many_blocked_reasons_accumulated",
        ),
        CircuitBreakerCondition(
            name="max_daily_loss_blocked",
            check=lambda s: s.daily_loss_tracker.blocked,
            severity="no_trade",
            reason="max_daily_loss_reached",
        ),
    ]
