from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_bot.durable_io import durable_write_json, durable_write_text

@dataclass(frozen=True)
class ExitPolicyConfig:
    stop_policy: str = "static"
    target_policy: str = "static"
    notes: str = (
        "Static and ATR-based dynamic exits are auditables. Trailing, break-even "
        "and adaptive exits remain intentionally out of scope."
    )


@dataclass
class OperationalEvent:
    event_type: str
    timestamp: str
    symbol: str
    status: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, str]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "status": self.status,
            "reason": self.reason,
            "details_json": json.dumps(self.details, sort_keys=True),
        }


@dataclass
class PaperPosition:
    symbol: str
    timeframe: str
    direction: int
    entry_timestamp: str
    signal_timestamp: str
    entry_index: int
    volume_lots: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    commission_entry_usd: float
    bars_held: int
    probability_long: float
    probability_short: float
    stop_policy: str
    target_policy: str
    stop_policy_details: dict[str, Any] = field(default_factory=dict)
    target_policy_details: dict[str, Any] = field(default_factory=dict)
    active_profile_id: str = ""
    model_variant: str = ""
    profile_source_run_id: str = ""
    enablement_state: str = ""
    promotion_state: str = ""


@dataclass
class ClosedPaperTrade:
    symbol: str
    timeframe: str
    direction: int
    entry_timestamp: str
    exit_timestamp: str
    signal_timestamp: str
    entry_price: float
    exit_price: float
    stop_loss_price: float
    take_profit_price: float
    volume_lots: float
    gross_pnl_usd: float
    net_pnl_usd: float
    total_commission_usd: float
    spread_cost_usd: float
    slippage_cost_usd: float
    exit_reason: str
    bars_held: int
    probability_long: float
    probability_short: float
    stop_policy: str
    target_policy: str
    stop_policy_details: dict[str, Any] = field(default_factory=dict)
    target_policy_details: dict[str, Any] = field(default_factory=dict)
    is_intrabar_ambiguous: bool = False
    active_profile_id: str = ""
    model_variant: str = ""
    profile_source_run_id: str = ""
    enablement_state: str = ""
    promotion_state: str = ""


@dataclass
class AccountState:
    balance_usd: float
    cash_usd: float
    equity_usd: float


@dataclass
class DailyLossTracker:
    current_day: str | None
    realized_pnl_usd: float
    loss_limit_usd: float
    blocked: bool


@dataclass
class ExposureState:
    open_positions: int
    gross_volume_lots: float
    symbols: list[str]


@dataclass
class SessionStatus:
    session_id: str
    mode: str
    status: str
    last_timestamp: str | None


@dataclass
class BrokerSyncStatus:
    state: str = "unknown"
    last_sync_timestamp: str | None = None
    reconciliation_policy: str = "hard_fail"
    critical_discrepancy_count: int = 0


@dataclass
class ProcessingState:
    last_processed_timestamp_by_symbol: dict[str, str] = field(default_factory=dict)
    processed_event_ids: list[str] = field(default_factory=list)
    idempotency_mode_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class PendingIntent:
    symbol: str
    created_at: str
    signal_timestamp: str
    side: str
    volume_lots: float
    decision_context: dict[str, Any] = field(default_factory=dict)
    active_profile_id: str = ""
    model_variant: str = ""
    profile_source_run_id: str = ""
    enablement_state: str = ""
    promotion_state: str = ""


@dataclass
class AlertRecord:
    timestamp: str
    severity: str
    category: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperEngineState:
    account_state: AccountState
    open_positions: dict[str, PaperPosition] = field(default_factory=dict)
    closed_positions: list[ClosedPaperTrade] = field(default_factory=list)
    daily_loss_tracker: DailyLossTracker = field(
        default_factory=lambda: DailyLossTracker(None, 0.0, 0.0, False)
    )
    cooldown_tracker: dict[str, int] = field(default_factory=dict)
    exposure: ExposureState = field(default_factory=lambda: ExposureState(0, 0.0, []))
    last_signal_per_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_session_status: SessionStatus = field(
        default_factory=lambda: SessionStatus("", "paper", "idle", None)
    )
    blocked_trades_summary: dict[str, int] = field(default_factory=dict)
    blocked_reasons: list[str] = field(default_factory=list)
    pending_intents: list[PendingIntent] = field(default_factory=list)
    broker_sync_status: BrokerSyncStatus = field(default_factory=BrokerSyncStatus)
    processing_state: ProcessingState = field(default_factory=ProcessingState)
    latest_broker_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_state": asdict(self.account_state),
            "open_positions": {key: asdict(value) for key, value in self.open_positions.items()},
            "closed_positions": [asdict(item) for item in self.closed_positions],
            "daily_loss_tracker": asdict(self.daily_loss_tracker),
            "cooldown_tracker": dict(self.cooldown_tracker),
            "exposure": asdict(self.exposure),
            "last_signal_per_symbol": self.last_signal_per_symbol,
            "current_session_status": asdict(self.current_session_status),
            "blocked_trades_summary": dict(self.blocked_trades_summary),
            "blocked_reasons": list(self.blocked_reasons),
            "pending_intents": [asdict(item) for item in self.pending_intents],
            "broker_sync_status": asdict(self.broker_sync_status),
            "processing_state": asdict(self.processing_state),
            "latest_broker_snapshot": self.latest_broker_snapshot,
        }


@dataclass
class PaperRunArtifacts:
    state: PaperEngineState
    events: list[OperationalEvent]
    closed_trades: list[ClosedPaperTrade]
    equity_curve_rows: list[dict[str, Any]]
    daily_summary: dict[str, Any]
    run_report: dict[str, Any]
    validation_report: dict[str, Any]
    signal_rows: list[dict[str, Any]]
    execution_rows: list[dict[str, Any]]
    reconciliation_report: dict[str, Any] = field(default_factory=dict)
    reconciliation_scope_report: dict[str, Any] = field(default_factory=dict)
    restore_state_report: dict[str, Any] = field(default_factory=dict)
    idempotency_report: dict[str, Any] = field(default_factory=dict)
    operational_status: dict[str, Any] = field(default_factory=dict)
    alerts: list[AlertRecord] = field(default_factory=list)


def new_session_id(prefix: str) -> str:
    return f"{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}_{prefix}"


def write_events_csv(path: Path, events: list[OperationalEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    with buffer:
        writer = csv.DictWriter(
            buffer,
            fieldnames=["event_type", "timestamp", "symbol", "status", "reason", "details_json"],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(event.to_row())
        durable_write_text(path, buffer.getvalue())


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    with buffer:
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        durable_write_text(path, buffer.getvalue())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    durable_write_json(path, payload)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    durable_write_json(path, payload)


def write_alerts_jsonl(path: Path, alerts: list[AlertRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    durable_write_text(path, "".join(f"{json.dumps(asdict(alert), sort_keys=True)}\n" for alert in alerts))


def write_operational_artifacts(
    run_dir: Path,
    artifacts: PaperRunArtifacts,
    config_payload: dict[str, Any],
) -> None:
    write_json(run_dir / "config_used.json", config_payload)
    write_rows_csv(
        run_dir / "signal_log.csv",
        artifacts.signal_rows,
        [
            "timestamp",
            "symbol",
            "timeframe",
            "signal",
            "probability_long",
            "probability_short",
            "threshold",
            "active_profile_id",
            "model_variant",
            "profile_source_run_id",
            "enablement_state",
            "promotion_state",
            "promotion_reason",
            "status",
            "reason",
        ],
    )
    write_rows_csv(
        run_dir / "execution_journal.csv",
        artifacts.execution_rows,
        [
            "timestamp",
            "symbol",
            "event_type",
            "status",
            "reason",
            "active_profile_id",
            "model_variant",
            "profile_source_run_id",
            "enablement_state",
            "promotion_state",
            "promotion_reason",
            "volume_lots",
            "entry_price",
            "exit_price",
            "stop_loss_price",
            "take_profit_price",
            "details_json",
        ],
    )
    write_rows_csv(
        run_dir / "equity_curve.csv",
        artifacts.equity_curve_rows,
        ["timestamp", "balance", "equity", "open_positions"],
    )
    closed_payload = []
    for trade in artifacts.closed_trades:
        payload = asdict(trade)
        payload["stop_policy_details"] = json.dumps(trade.stop_policy_details, sort_keys=True)
        payload["target_policy_details"] = json.dumps(trade.target_policy_details, sort_keys=True)
        closed_payload.append(payload)
    fieldnames = list(closed_payload[0].keys()) if closed_payload else [
        "symbol",
        "timeframe",
        "direction",
        "entry_timestamp",
        "exit_timestamp",
        "signal_timestamp",
        "entry_price",
        "exit_price",
        "stop_loss_price",
        "take_profit_price",
        "volume_lots",
        "gross_pnl_usd",
        "net_pnl_usd",
        "total_commission_usd",
        "spread_cost_usd",
        "slippage_cost_usd",
        "exit_reason",
        "bars_held",
        "probability_long",
        "probability_short",
        "stop_policy",
        "target_policy",
        "stop_policy_details",
        "target_policy_details",
        "is_intrabar_ambiguous",
    ]
    write_rows_csv(run_dir / "closed_trades.csv", closed_payload, fieldnames)
    atomic_write_json(run_dir / "open_positions_snapshot.json", artifacts.state.to_dict())
    atomic_write_json(run_dir / "daily_summary.json", artifacts.daily_summary)
    atomic_write_json(run_dir / "reconciliation_report.json", artifacts.reconciliation_report)
    atomic_write_json(run_dir / "reconciliation_scope_report.json", artifacts.reconciliation_scope_report)
    atomic_write_json(run_dir / "restore_state_report.json", artifacts.restore_state_report)
    atomic_write_json(run_dir / "idempotency_report.json", artifacts.idempotency_report)
    atomic_write_json(run_dir / "operational_status.json", artifacts.operational_status)
    atomic_write_json(run_dir / "run_report.json", artifacts.run_report)
    atomic_write_json(run_dir / "validation_report.json", artifacts.validation_report)
    write_alerts_jsonl(run_dir / "alerts_log.jsonl", artifacts.alerts)
