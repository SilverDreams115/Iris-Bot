from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_bot.config import Settings
from iris_bot.operational import AlertRecord


@dataclass(frozen=True)
class BrokerPositionSnapshot:
    ticket: str
    symbol: str
    side: str
    volume_lots: float
    price_open: float
    stop_loss: float
    take_profit: float
    time: str


@dataclass(frozen=True)
class BrokerStateSnapshot:
    connected: bool
    balance_usd: float | None
    equity_usd: float | None
    positions: list[BrokerPositionSnapshot]
    closed_trades: list[dict[str, Any]]
    pending_orders: list[dict[str, Any]]
    raw_account: dict[str, Any]
    scope_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "balance_usd": self.balance_usd,
            "equity_usd": self.equity_usd,
            "positions": [asdict(item) for item in self.positions],
            "closed_trades": self.closed_trades,
            "pending_orders": self.pending_orders,
            "raw_account": self.raw_account,
            "scope_report": self.scope_report,
        }


@dataclass(frozen=True)
class ReconciliationDiscrepancy:
    category: str
    severity: str
    message: str
    details: dict[str, Any]


@dataclass(frozen=True)
class ReconciliationOutcome:
    ok: bool
    action: str
    discrepancies: list[ReconciliationDiscrepancy]
    synced_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "discrepancies": [asdict(item) for item in self.discrepancies],
            "synced_state": self.synced_state,
        }


@dataclass(frozen=True)
class RestoreReport:
    ok: bool
    action: str
    issues: list[str]
    restored: bool
    state_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconnectReport:
    ok: bool
    final_state: str
    attempts: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrokerEventDecision:
    classification: str
    action: str
    retryable: bool
    block_operation: bool
    details: dict[str, Any]


def now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def build_runtime_state_path(settings: Settings) -> Path:
    return settings.data.runtime_dir / settings.operational.persistence_state_filename


def emit_alert(
    alerts: list[AlertRecord],
    severity: str,
    category: str,
    message: str,
    details: dict[str, Any],
) -> None:
    alerts.append(
        AlertRecord(
            timestamp=now_iso(),
            severity=severity,
            category=category,
            message=message,
            details=details,
        )
    )
