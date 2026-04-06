from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from iris_bot.config import SessionConfig
from iris_bot.operational import (
    AccountState,
    PaperEngineState,
    PendingIntent,
    ProcessingState,
    atomic_write_json,
)
from iris_bot.resilient_models import RestoreReport, now_iso


def fresh_state(starting_balance: float, mode: str) -> PaperEngineState:
    state = PaperEngineState(
        account_state=AccountState(starting_balance, starting_balance, starting_balance),
    )
    state.current_session_status.mode = mode
    state.current_session_status.status = "idle"
    return state


def persist_runtime_state(
    path: Path,
    state: PaperEngineState,
    latest_broker_sync_result: dict[str, Any],
) -> None:
    atomic_write_json(
        path,
        {
            "saved_at": now_iso(),
            "state": state.to_dict(),
            "latest_broker_sync_result": latest_broker_sync_result,
        },
    )


def restore_runtime_state(path: Path, require_clean: bool) -> tuple[PaperEngineState | None, RestoreReport]:
    if not path.exists():
        return None, RestoreReport(True, "log_only", ["state_missing"], False, str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        action = "blocked" if require_clean else "log_only"
        return None, RestoreReport(False, action, [f"state_corrupt:{exc}"], False, str(path))
    state_payload = payload.get("state")
    if not isinstance(state_payload, dict):
        action = "blocked" if require_clean else "log_only"
        return None, RestoreReport(False, action, ["state_missing_payload"], False, str(path))
    try:
        state = state_from_dict(state_payload)
    except Exception as exc:  # noqa: BLE001
        action = "blocked" if require_clean else "log_only"
        return None, RestoreReport(False, action, [f"state_restore_failed:{exc}"], False, str(path))
    return state, RestoreReport(True, "soft_resync", [], True, str(path))


def state_from_dict(payload: dict[str, Any]) -> PaperEngineState:
    from iris_bot.operational import (
        BrokerSyncStatus,
        ClosedPaperTrade,
        DailyLossTracker,
        ExposureState,
        PaperPosition,
        SessionStatus,
    )

    return PaperEngineState(
        account_state=AccountState(**payload["account_state"]),
        open_positions={key: PaperPosition(**value) for key, value in payload.get("open_positions", {}).items()},
        closed_positions=[ClosedPaperTrade(**item) for item in payload.get("closed_positions", [])],
        daily_loss_tracker=DailyLossTracker(**payload.get("daily_loss_tracker", {})),
        cooldown_tracker=payload.get("cooldown_tracker", {}),
        exposure=ExposureState(**payload.get("exposure", {})),
        last_signal_per_symbol=payload.get("last_signal_per_symbol", {}),
        current_session_status=SessionStatus(**payload.get("current_session_status", {})),
        blocked_trades_summary=payload.get("blocked_trades_summary", {}),
        blocked_reasons=payload.get("blocked_reasons", []),
        pending_intents=[PendingIntent(**item) for item in payload.get("pending_intents", [])],
        broker_sync_status=BrokerSyncStatus(**payload.get("broker_sync_status", {})),
        processing_state=ProcessingState(**payload.get("processing_state", {})),
        latest_broker_snapshot=payload.get("latest_broker_snapshot", {}),
    )


def is_session_allowed(timestamp: datetime, session: SessionConfig) -> tuple[bool, str]:
    if not session.enabled:
        return True, "session_control_disabled"
    if timestamp.weekday() not in session.allowed_weekdays:
        return False, "market_session_blocked_weekday"
    if not (session.allowed_start_hour_utc <= timestamp.hour <= session.allowed_end_hour_utc):
        return False, "market_session_blocked_hour"
    return True, "session_allowed"


def build_processing_event_id(symbol: str, timestamp_text: str, source_event_id: str | None = None) -> tuple[str, str]:
    if source_event_id:
        return f"event:{source_event_id}", "event_id"
    return f"fallback:{symbol}:{timestamp_text}", "timestamp_fallback"


def prevent_duplicate_processing(
    state: PaperEngineState,
    symbol: str,
    timestamp_text: str,
    source_event_id: str | None = None,
) -> bool:
    event_id, mode = build_processing_event_id(symbol, timestamp_text, source_event_id)
    if event_id in state.processing_state.processed_event_ids:
        return False
    previous = state.processing_state.last_processed_timestamp_by_symbol.get(symbol)
    if mode == "timestamp_fallback" and previous is not None and timestamp_text <= previous:
        return False
    state.processing_state.processed_event_ids.append(event_id)
    state.processing_state.last_processed_timestamp_by_symbol[symbol] = timestamp_text
    state.processing_state.idempotency_mode_counts[mode] = state.processing_state.idempotency_mode_counts.get(mode, 0) + 1
    return True
