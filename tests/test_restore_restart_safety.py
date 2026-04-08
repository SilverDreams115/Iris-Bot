"""BLOQUE 1 — Restore/Restart Safety Drills.

These tests validate that the system can restart without losing operational
integrity. Each test covers one explicit invariant. All tests generate
auditable reports (via run_restore_safety_drill) or validate in-memory.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from iris_bot.config import (
    BacktestConfig,
    OperationalConfig,
    RecoveryConfig,
    ReconciliationConfig,
    RiskConfig,
    SessionConfig,
    Settings,
)
from iris_bot.operational import (
    AccountState,
    BrokerSyncStatus,
    DailyLossTracker,
    PaperEngineState,
    PaperPosition,
    PendingIntent,
    ProcessingState,
    SessionStatus,
)
from iris_bot.resilient import (
    build_runtime_state_path,
    persist_runtime_state,
    prevent_duplicate_processing,
    restore_runtime_state,
    run_restore_safety_drill,
    validate_restored_state_invariants,
)


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "data", replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"))
    object.__setattr__(settings, "recovery", RecoveryConfig(reconnect_retries=2, reconnect_backoff_seconds=0.0, require_state_restore_clean=True))
    object.__setattr__(settings, "reconciliation", ReconciliationConfig(policy="hard_fail", price_tolerance=0.0001, volume_tolerance=0.000001))
    object.__setattr__(settings, "session", SessionConfig(enabled=True, allowed_weekdays=(0, 1, 2, 3, 4), allowed_start_hour_utc=0, allowed_end_hour_utc=23))
    object.__setattr__(settings, "operational", OperationalConfig(repeated_rejection_alert_threshold=1, persistence_state_filename="runtime_state.json"))
    object.__setattr__(settings, "backtest", BacktestConfig(use_atr_stops=False, fixed_stop_loss_pct=0.002, fixed_take_profit_pct=0.004, max_holding_bars=5))
    object.__setattr__(settings, "risk", RiskConfig(max_daily_loss_usd=50.0))
    return settings


def _state_with_full_fields() -> PaperEngineState:
    state = PaperEngineState(
        account_state=AccountState(1000.0, 1000.0, 1000.0),
        open_positions={
            "EURUSD": PaperPosition(
                symbol="EURUSD", timeframe="M15", direction=1,
                entry_timestamp="2026-01-01T00:15:00", signal_timestamp="2026-01-01T00:00:00",
                entry_index=1, volume_lots=0.1, entry_price=1.1000,
                stop_loss_price=1.0980, take_profit_price=1.1040,
                commission_entry_usd=1.0, bars_held=2,
                probability_long=0.8, probability_short=0.05,
                stop_policy="static", target_policy="static",
            )
        },
        daily_loss_tracker=DailyLossTracker("2026-01-01", -15.0, 50.0, False),
        current_session_status=SessionStatus("sess-001", "paper", "running", "2026-01-01T00:15:00"),
        broker_sync_status=BrokerSyncStatus(),
        processing_state=ProcessingState({"EURUSD": "2026-01-01T00:15:00"}, ["evt:001", "evt:002"]),
    )
    state.pending_intents.append(PendingIntent(
        symbol="GBPUSD", created_at="2026-01-01T00:10:00",
        signal_timestamp="2026-01-01T00:10:00", side="sell", volume_lots=0.05,
    ))
    return state


# ── Invariant 1: Open positions preserved after restart ─────────────────────

def test_open_positions_preserved_after_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = _state_with_full_fields()
    persist_runtime_state(build_runtime_state_path(settings), state, {})
    restored, report = restore_runtime_state(build_runtime_state_path(settings), require_clean=True)
    assert report.ok
    assert restored is not None
    assert "EURUSD" in restored.open_positions
    pos = restored.open_positions["EURUSD"]
    assert pos.volume_lots == 0.1
    assert pos.direction == 1
    assert pos.entry_price == 1.1000


# ── Invariant 2: Pending intents preserved ──────────────────────────────────

def test_pending_intents_preserved_after_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = _state_with_full_fields()
    persist_runtime_state(build_runtime_state_path(settings), state, {})
    restored, _ = restore_runtime_state(build_runtime_state_path(settings), require_clean=True)
    assert restored is not None
    assert len(restored.pending_intents) == 1
    assert restored.pending_intents[0].symbol == "GBPUSD"
    assert restored.pending_intents[0].side == "sell"


# ── Invariant 3: Processing state prevents duplicate decisions ───────────────

def test_no_duplicate_decisions_after_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = _state_with_full_fields()
    persist_runtime_state(build_runtime_state_path(settings), state, {})
    restored, _ = restore_runtime_state(build_runtime_state_path(settings), require_clean=True)
    assert restored is not None
    # Previously processed events must be blocked
    assert not prevent_duplicate_processing(restored, "EURUSD", "2026-01-01T00:15:00")
    # New events must be allowed
    assert prevent_duplicate_processing(restored, "EURUSD", "2026-01-01T00:30:00")


# ── Invariant 4: Daily loss tracker (blocked state) survives restart ─────────

def test_daily_loss_blocked_persists_across_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = _state_with_full_fields()
    state.daily_loss_tracker.blocked = True
    state.daily_loss_tracker.realized_pnl_usd = -55.0
    persist_runtime_state(build_runtime_state_path(settings), state, {})
    restored, _ = restore_runtime_state(build_runtime_state_path(settings), require_clean=True)
    assert restored is not None
    assert restored.daily_loss_tracker.blocked is True
    assert restored.daily_loss_tracker.realized_pnl_usd == -55.0


# ── Invariant 5: Session lineage preserved ──────────────────────────────────

def test_session_lineage_preserved_after_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = _state_with_full_fields()
    persist_runtime_state(build_runtime_state_path(settings), state, {})
    restored, _ = restore_runtime_state(build_runtime_state_path(settings), require_clean=True)
    assert restored is not None
    assert restored.current_session_status.session_id == "sess-001"
    assert restored.current_session_status.mode == "paper"


# ── Invariant 6: Corrupt state → blocked cleanly ────────────────────────────

def test_corrupt_state_blocks_restore(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = build_runtime_state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken json", encoding="utf-8")
    restored, report = restore_runtime_state(path, require_clean=True)
    assert restored is None
    assert report.ok is False
    assert report.action == "blocked"


# ── Invariant 7: Structural invariant violations are caught ─────────────────

def test_structural_invariant_validation_catches_violation() -> None:
    state = _state_with_full_fields()
    # Inject duplicate event IDs
    state.processing_state.processed_event_ids.append("evt:001")
    issues = validate_restored_state_invariants(state)
    assert any("duplicate_processed_event_ids" in issue for issue in issues)


# ── Invariant 8: Clean state passes structural invariant check ──────────────

def test_clean_state_passes_invariant_validation() -> None:
    state = _state_with_full_fields()
    issues = validate_restored_state_invariants(state)
    assert issues == []


# ── Invariant 9: Schema version stored and validated ────────────────────────

def test_schema_version_present_in_persisted_state(tmp_path: Path) -> None:
    import json
    settings = _settings(tmp_path)
    state = _state_with_full_fields()
    path = build_runtime_state_path(settings)
    persist_runtime_state(path, state, {})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.get("schema_version") == 1


# ── Integrated drill: run_restore_safety_drill generates full report ─────────

def test_restore_safety_drill_all_pass(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    exit_code, report = run_restore_safety_drill(settings)
    assert exit_code == 0, f"Drill failed: {report.get('failed_checks')}"
    assert report["ok"] is True
    assert report["failed_checks"] == []
    # All invariant checks must be present
    expected_checks = {
        "persist",
        "restore",
        "open_positions_preserved",
        "pending_intents_preserved",
        "processing_state_preserved",
        "daily_loss_preserved",
        "idempotency_after_restore",
        "session_lineage_preserved",
        "structural_invariants_clean",
    }
    assert expected_checks.issubset(report["checks"].keys())
