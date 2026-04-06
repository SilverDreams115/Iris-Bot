"""
Engine consistency validation for IRIS-Bot backtest output.

This module provides verify_engine_consistency(), which audits that
trade_log, equity_curve and aggregate metrics are internally coherent.

It is intentionally decoupled from the backtest engine so it can be
used both by run_backtest() and run_walkforward_economic_backtest().

Checks performed
----------------
Per-trade:
  1.  PnL math:         net_pnl ≈ gross_pnl - total_commission
  2.  Timestamp order:  signal_ts <= entry_ts <= exit_ts
  3.  No duplicates:    (symbol, entry_timestamp) pair is unique
  4.  Direction:        must be 1 (long) or -1 (short)
  5.  Volume:           volume_lots > 0
  6.  bars_held:        >= 1
  7.  Commission:       total_commission_usd >= 0
  8.  exit_reason:      must be a known exit type

Global:
  9.  Final balance:    equity_curve[-1].balance ≈ starting_balance + Σ net_pnl
  10. open_positions:   always >= 0 in equity curve
  11. Balance warning:  balance < 0 emits a warning (margin-call territory)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from iris_bot.backtest import EquityPoint


class ConsistencyTrade(Protocol):
    @property
    def symbol(self) -> str: ...

    @property
    def signal_timestamp(self) -> str: ...

    @property
    def entry_timestamp(self) -> str: ...

    @property
    def exit_timestamp(self) -> str: ...

    @property
    def direction(self) -> int: ...

    @property
    def volume_lots(self) -> float: ...

    @property
    def bars_held(self) -> int: ...

    @property
    def total_commission_usd(self) -> float: ...

    @property
    def exit_reason(self) -> str: ...

    @property
    def gross_pnl_usd(self) -> float: ...

    @property
    def net_pnl_usd(self) -> float: ...


@dataclass
class ConsistencyViolation:
    severity: str       # "error" | "warning"
    trade_index: int | None
    message: str


@dataclass
class ConsistencyReport:
    violations: list[ConsistencyViolation] = field(default_factory=list)
    checks_passed: int = 0

    @property
    def is_clean(self) -> bool:
        """True if there are zero ERROR-level violations."""
        return not any(v.severity == "error" for v in self.violations)

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")

    def to_dict(self) -> dict[str, object]:
        return {
            "is_clean": self.is_clean,
            "checks_passed": self.checks_passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "violations": [
                {
                    "severity": v.severity,
                    "trade_index": v.trade_index,
                    "message": v.message,
                }
                for v in self.violations
            ],
        }


_VALID_EXIT_REASONS: frozenset[str] = frozenset({
    "stop_loss",
    "take_profit",
    "stop_loss_same_bar",
    "take_profit_same_bar",
    "time_exit",
    "end_of_data",
})


def verify_engine_consistency(
    trades: Sequence[ConsistencyTrade],
    equity_curve: Sequence[EquityPoint],
    starting_balance: float,
    tolerance: float = 0.01,
) -> ConsistencyReport:
    """
    Audit internal consistency of backtest engine output.

    Parameters
    ----------
    trades:           List of TradeRecord objects from run_backtest_engine.
    equity_curve:     List of EquityPoint objects from run_backtest_engine.
    starting_balance: The initial account balance used in the backtest.
    tolerance:        Absolute tolerance for floating-point comparisons.
                      Default 0.01 USD covers fp rounding across many trades.

    Returns
    -------
    ConsistencyReport with violations (errors/warnings) and checks_passed count.
    A report with is_clean=True means all error-level checks passed.
    """
    report = ConsistencyReport()

    def ok() -> None:
        report.checks_passed += 1

    def error(idx: int | None, msg: str) -> None:
        report.violations.append(ConsistencyViolation("error", idx, msg))

    def warn(idx: int | None, msg: str) -> None:
        report.violations.append(ConsistencyViolation("warning", idx, msg))

    seen_entries: set[tuple[str, str]] = set()

    for idx, trade in enumerate(trades):

        # 1. PnL math: net_pnl must equal gross_pnl - total_commission
        expected_net = trade.gross_pnl_usd - trade.total_commission_usd
        diff = abs(expected_net - trade.net_pnl_usd)
        if diff > tolerance:
            error(
                idx,
                f"PnL math violation: net={trade.net_pnl_usd:.6f} "
                f"!= gross-comm={expected_net:.6f} (diff={diff:.6f})",
            )
        else:
            ok()

        # 2. Timestamp ordering: signal_ts <= entry_ts <= exit_ts
        try:
            signal_ts = datetime.fromisoformat(trade.signal_timestamp)
            entry_ts = datetime.fromisoformat(trade.entry_timestamp)
            exit_ts = datetime.fromisoformat(trade.exit_timestamp)

            if signal_ts > entry_ts:
                error(
                    idx,
                    f"Timestamp: signal {trade.signal_timestamp} > entry {trade.entry_timestamp}",
                )
            else:
                ok()

            if entry_ts > exit_ts:
                error(
                    idx,
                    f"Timestamp: entry {trade.entry_timestamp} > exit {trade.exit_timestamp}",
                )
            else:
                ok()

        except ValueError as exc:
            error(idx, f"Timestamp parse error: {exc}")

        # 3. No duplicate positions (same symbol opened at the same bar)
        key = (trade.symbol, trade.entry_timestamp)
        if key in seen_entries:
            error(idx, f"Duplicate position: {trade.symbol} @ {trade.entry_timestamp}")
        else:
            seen_entries.add(key)
            ok()

        # 4. Direction must be 1 (long) or -1 (short)
        if trade.direction not in {1, -1}:
            error(idx, f"Invalid direction: {trade.direction} (must be 1 or -1)")
        else:
            ok()

        # 5. Volume strictly positive
        if trade.volume_lots <= 0.0:
            error(idx, f"volume_lots={trade.volume_lots:.6f} must be > 0")
        else:
            ok()

        # 6. At least one bar was held
        if trade.bars_held < 1:
            error(idx, f"bars_held={trade.bars_held} < 1")
        else:
            ok()

        # 7. Commission non-negative
        if trade.total_commission_usd < 0.0:
            error(idx, f"total_commission_usd={trade.total_commission_usd:.6f} < 0")
        else:
            ok()

        # 8. exit_reason must be a known value
        if trade.exit_reason not in _VALID_EXIT_REASONS:
            warn(idx, f"Unknown exit_reason: {trade.exit_reason!r}")
        else:
            ok()

    # --- Global checks against equity curve ---

    if equity_curve:

        # 9. Final balance consistency
        expected_final = starting_balance + sum(t.net_pnl_usd for t in trades)
        actual_final = equity_curve[-1].balance
        abs_tol = max(tolerance, abs(expected_final) * tolerance)
        if abs(actual_final - expected_final) > abs_tol:
            error(
                None,
                f"Final balance mismatch: equity_curve[-1].balance={actual_final:.4f} "
                f"!= starting_balance + Σnet_pnl={expected_final:.4f} "
                f"(diff={abs(actual_final - expected_final):.4f})",
            )
        else:
            ok()

        # 10. open_positions must always be >= 0
        neg_pos_step = next(
            (i for i, pt in enumerate(equity_curve) if pt.open_positions < 0),
            None,
        )
        if neg_pos_step is not None:
            error(
                None,
                f"Equity curve step {neg_pos_step}: "
                f"open_positions={equity_curve[neg_pos_step].open_positions} < 0",
            )
        else:
            ok()

        # 11. Balance < 0 is a warning (margin-call territory)
        neg_balance_step = next(
            (i for i, pt in enumerate(equity_curve) if pt.balance < 0.0),
            None,
        )
        if neg_balance_step is not None:
            warn(
                None,
                f"Equity curve step {neg_balance_step}: "
                f"balance={equity_curve[neg_balance_step].balance:.4f} < 0 "
                f"(margin-call territory)",
            )

    return report
