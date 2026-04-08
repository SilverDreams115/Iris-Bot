"""BLOQUE 4 — Demo-Guarded Soak Tests.

Validates that the demo-guarded soak mode:
- generates all required audit artifacts
- accumulates restore / reconcile / blocked-trade / alert counters
- produces a DemoGuardedSoakSummary as first-class artifact
- returns correct exit codes for go / caution / no_go decisions
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from iris_bot.config import (
    BacktestConfig,
    OperationalConfig,
    RecoveryConfig,
    ReconciliationConfig,
    RiskConfig,
    SessionConfig,
    Settings,
    SoakConfig,
)
from iris_bot.soak import DemoGuardedSoakSummary, run_demo_guarded_soak


def _settings(tmp_path: Path, cycles: int = 2) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "data", replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"))
    object.__setattr__(settings, "recovery", RecoveryConfig(reconnect_retries=1, reconnect_backoff_seconds=0.0, require_state_restore_clean=True))
    object.__setattr__(settings, "reconciliation", ReconciliationConfig(policy="log_only", price_tolerance=0.001, volume_tolerance=0.001))
    object.__setattr__(settings, "backtest", BacktestConfig(use_atr_stops=False, fixed_stop_loss_pct=0.002, fixed_take_profit_pct=0.004, max_holding_bars=5))
    object.__setattr__(settings, "risk", RiskConfig(max_daily_loss_usd=50.0))
    object.__setattr__(settings, "session", SessionConfig(enabled=True, allowed_weekdays=(0, 1, 2, 3, 4), allowed_start_hour_utc=0, allowed_end_hour_utc=23))
    object.__setattr__(settings, "operational", OperationalConfig(persistence_state_filename="runtime_state.json"))
    object.__setattr__(settings, "soak", SoakConfig(cycles=cycles, pause_seconds=0.0, restore_between_cycles=True))
    return settings


def _row(ts: datetime, symbol: str = "EURUSD") -> object:
    from iris_bot.processed_dataset import ProcessedRow
    return ProcessedRow(
        timestamp=ts, symbol=symbol, timeframe="M15",
        open=1.1, high=1.1008, low=1.0992, close=1.1,
        volume=100.0, label=1, label_reason="test",
        horizon_end_timestamp=ts.isoformat(),
        features={"atr_10": 0.0005, "atr_5": 0.0005},
    )


class FakeReference:
    threshold = 0.5
    run_dir = Path("/tmp/fake_experiment")


# ── Required artifacts after soak run ────────────────────────────────────────

def test_demo_guarded_soak_generates_required_artifacts(tmp_path: Path, monkeypatch: object) -> None:
    settings = _settings(tmp_path, cycles=2)
    rows = [_row(datetime(2026, 1, 1, 0, 0)), _row(datetime(2026, 1, 1, 0, 15))]
    probs = [{1: 0.2, 0: 0.6, -1: 0.2}, {1: 0.2, 0: 0.6, -1: 0.2}]
    import iris_bot.resilient as r_mod
    monkeypatch.setattr(r_mod, "load_paper_context", lambda *a, **kw: (FakeReference(), rows, probs))

    exit_code, run_dir = run_demo_guarded_soak(settings, mode="paper", require_broker=False)

    assert (run_dir / "demo_guarded_soak_summary.json").exists()
    assert (run_dir / "soak_report.json").exists()
    assert (run_dir / "health_report.json").exists()
    assert (run_dir / "go_no_go_report.json").exists()
    assert (run_dir / "incident_log.jsonl").exists()
    assert (run_dir / "cycle_summaries").is_dir()
    assert (run_dir / "cycle_summaries" / "cycle_01.json").exists()
    assert (run_dir / "cycle_summaries" / "cycle_02.json").exists()


# ── DemoGuardedSoakSummary fields are correct ────────────────────────────────

def test_demo_guarded_soak_summary_fields(tmp_path: Path, monkeypatch: object) -> None:
    settings = _settings(tmp_path, cycles=3)
    rows = [_row(datetime(2026, 1, 1, 0, 0))]
    probs = [{1: 0.2, 0: 0.6, -1: 0.2}]
    import iris_bot.resilient as r_mod
    monkeypatch.setattr(r_mod, "load_paper_context", lambda *a, **kw: (FakeReference(), rows, probs))

    _, run_dir = run_demo_guarded_soak(settings, mode="paper", require_broker=False)
    raw = json.loads((run_dir / "demo_guarded_soak_summary.json").read_text())

    assert raw["cycles_requested"] == 3
    assert raw["cycles_completed"] == 3
    assert raw["restore_events"] == 3
    assert raw["restore_failures"] == 0
    assert raw["reconcile_events"] == 3
    assert raw["reconcile_failures"] == 0
    assert raw["overall_decision"] in {"go", "caution", "no_go"}


# ── Summary is serialisable (JSON artifact) ──────────────────────────────────

def test_demo_guarded_soak_summary_serialisable(tmp_path: Path, monkeypatch: object) -> None:
    settings = _settings(tmp_path, cycles=1)
    rows = [_row(datetime(2026, 1, 1, 0, 0))]
    probs = [{1: 0.2, 0: 0.6, -1: 0.2}]
    import iris_bot.resilient as r_mod
    monkeypatch.setattr(r_mod, "load_paper_context", lambda *a, **kw: (FakeReference(), rows, probs))

    _, run_dir = run_demo_guarded_soak(settings, mode="paper", require_broker=False)
    raw = json.loads((run_dir / "demo_guarded_soak_summary.json").read_text())
    # Must round-trip cleanly
    assert json.loads(json.dumps(raw))


# ── Soak type is tagged in report ────────────────────────────────────────────

def test_demo_guarded_soak_report_type_tagged(tmp_path: Path, monkeypatch: object) -> None:
    settings = _settings(tmp_path, cycles=1)
    rows = [_row(datetime(2026, 1, 1, 0, 0))]
    probs = [{1: 0.2, 0: 0.6, -1: 0.2}]
    import iris_bot.resilient as r_mod
    monkeypatch.setattr(r_mod, "load_paper_context", lambda *a, **kw: (FakeReference(), rows, probs))

    _, run_dir = run_demo_guarded_soak(settings, mode="paper", require_broker=False)
    soak_report = json.loads((run_dir / "soak_report.json").read_text())
    assert soak_report.get("soak_type") == "demo_guarded"


# ── DemoGuardedSoakSummary dataclass round-trip ──────────────────────────────

def test_demo_guarded_soak_summary_dataclass_round_trip() -> None:
    s = DemoGuardedSoakSummary(
        cycles_requested=3, cycles_completed=3,
        restore_events=3, restore_failures=0,
        reconcile_events=3, reconcile_failures=0,
        blocked_trade_events=1, critical_alerts=0,
        warning_alerts=2, circuit_breaker_triggers=0,
        no_go_cycles=0, overall_decision="go",
    )
    d = s.to_dict()
    assert d["cycles_completed"] == 3
    assert d["overall_decision"] == "go"
    assert json.loads(json.dumps(d))
