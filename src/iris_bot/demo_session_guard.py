"""Demo session gating for controlled serious demo execution.

Evaluates whether it is safe to start a new demo execution cycle.
Does NOT connect to MT5. Read-only state evaluation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_bot.kill_switch import is_kill_switch_active, is_no_trade_mode_active
from iris_bot.operational import PaperEngineState


__all__ = [
    "DemoSessionLimits",
    "DemoSessionGatingReport",
    "run_demo_session_precheck",
    "write_demo_session_gating_report",
]


@dataclass(frozen=True)
class DemoSessionLimits:
    max_active_positions: int = 1
    max_daily_loss_pct: float = 2.0


@dataclass(frozen=True)
class DemoSessionGatingReport:
    """Result of demo session pre-execution gate check."""

    ok: bool
    decision: str         # "proceed" | "abort" | "hold"
    passed_checks: list[str]
    failed_checks: list[str]
    warnings: list[str]
    details: dict[str, Any]
    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_demo_session_precheck(
    state: PaperEngineState,
    limits: DemoSessionLimits | None = None,
) -> DemoSessionGatingReport:
    """Evaluate pre-session conditions before starting a serious demo cycle.

    Does NOT connect to MT5. Does NOT send orders. State-only evaluation.
    Returns DemoSessionGatingReport with decision: proceed | abort | hold.
    """
    if limits is None:
        limits = DemoSessionLimits()

    passed: list[str] = []
    failed: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    # Check 1: Kill switch not active (abort if active)
    if is_kill_switch_active(state):
        failed.append("kill_switch_not_active")
        details["kill_switch_active"] = True
        details["blocked_reasons"] = list(state.blocked_reasons)
    else:
        passed.append("kill_switch_not_active")
        details["kill_switch_active"] = False

    # Check 2: No-trade mode (hold if active, but only if kill switch is not already blocking)
    if is_no_trade_mode_active(state) and not is_kill_switch_active(state):
        warnings.append("no_trade_mode_active")
        details["no_trade_mode_active"] = True
    else:
        details["no_trade_mode_active"] = is_no_trade_mode_active(state)

    # Check 3: Daily loss not blocked (hold if blocked)
    if state.daily_loss_tracker.blocked:
        warnings.append("daily_loss_blocked")
        details["daily_loss_blocked"] = True
    else:
        details["daily_loss_blocked"] = False

    # Check 4: Active positions within limit (abort if at or above limit)
    open_positions = len(state.open_positions)
    if open_positions >= limits.max_active_positions:
        failed.append("max_active_positions_exceeded")
    else:
        passed.append("max_active_positions_within_limit")
    details["open_positions"] = open_positions
    details["max_active_positions"] = limits.max_active_positions

    # Determine decision
    if failed:
        decision = "abort"
    elif warnings:
        decision = "hold"
    else:
        decision = "proceed"

    return DemoSessionGatingReport(
        ok=decision == "proceed",
        decision=decision,
        passed_checks=passed,
        failed_checks=failed,
        warnings=warnings,
        details=details,
        evaluated_at=datetime.now(tz=UTC).isoformat(),
    )


def write_demo_session_gating_report(path: Path, report: DemoSessionGatingReport) -> None:
    """Write demo session gating report as JSON to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
