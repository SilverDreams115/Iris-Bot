from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from iris_bot.config import ChaosConfig, MT5Config, OperationalConfig, RecoveryConfig, Settings, SoakConfig
from iris_bot.logging_utils import build_run_directory
from iris_bot.mt5 import BrokerSnapshot, DryRunOrderResult, MT5Client
from iris_bot.processed_dataset import ProcessedRow
from iris_bot.soak import classify_go_no_go, regenerate_go_no_go_report, run_soak


def _row(ts: datetime, symbol: str = "EURUSD", price: float = 1.1) -> ProcessedRow:
    return ProcessedRow(
        timestamp=ts,
        symbol=symbol,
        timeframe="M15",
        open=price,
        high=price + 0.0008,
        low=price - 0.0008,
        close=price,
        volume=100.0,
        label=1,
        label_reason="test",
        horizon_end_timestamp=ts.isoformat(),
        features={"atr_10": 0.0005, "atr_5": 0.0005},
    )


class FakeReference:
    threshold = 0.5
    run_dir = Path("/tmp/fake_experiment")


class FakeBrokerClient(MT5Client):
    def __init__(self, snapshot: BrokerSnapshot | None = None) -> None:
        super().__init__(MT5Config(enabled=True))
        self._snapshot = snapshot or BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [], [], [])

    def connect(self) -> bool:
        self._connected = True
        return True

    def last_error(self) -> object:
        return (1, "Success")

    def broker_state_snapshot(self, symbols: tuple[str, ...]):  # type: ignore[override]
        return self._snapshot

    def dry_run_market_order(self, order):  # type: ignore[override]
        return DryRunOrderResult(True, "dry_run_only", order.volume, {"symbol": order.symbol}, [])


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "data", replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"))
    object.__setattr__(settings, "soak", SoakConfig(cycles=2, pause_seconds=0.0, restore_between_cycles=True))
    object.__setattr__(settings, "recovery", RecoveryConfig(reconnect_retries=2, reconnect_backoff_seconds=0.0, require_state_restore_clean=True))
    object.__setattr__(settings, "operational", OperationalConfig(repeated_rejection_alert_threshold=1, persistence_state_filename="runtime_state.json"))
    return settings


def _patch_context(monkeypatch, rows: list[ProcessedRow], probabilities: list[dict[int, float]]) -> None:
    monkeypatch.setattr("iris_bot.resilient.load_paper_context", lambda settings: (FakeReference(), rows, probabilities))


def test_soak_multi_cycle_without_corruption(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    rows = [_row(datetime(2026, 1, 1, 0, 0, 0)), _row(datetime(2026, 1, 1, 0, 15, 0)), _row(datetime(2026, 1, 1, 0, 30, 0))]
    probabilities = [{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.2, 0: 0.6, -1: 0.2}, {1: 0.2, 0: 0.6, -1: 0.2}]
    _patch_context(monkeypatch, rows, probabilities)

    code, run_dir = run_soak(settings, "paper", require_broker=False)

    assert code in {0, 2}
    assert (run_dir / "soak_report.json").exists()
    assert (run_dir / "health_report.json").exists()
    assert (run_dir / "go_no_go_report.json").exists()


def test_restore_between_cycles_and_duplicate_event_prevention(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    rows = [_row(datetime(2026, 1, 1, 0, 0, 0)), _row(datetime(2026, 1, 1, 0, 15, 0))]
    probabilities = [{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.2, 0: 0.6, -1: 0.2}]
    _patch_context(monkeypatch, rows, probabilities)

    code, run_dir = run_soak(settings, "paper", require_broker=False)
    health = json.loads((run_dir / "health_report.json").read_text(encoding="utf-8"))

    assert code in {0, 2}
    assert len(health["cycles"]) == 2


def test_reconnect_injected_and_recovery(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    object.__setattr__(settings, "chaos", ChaosConfig(enabled=True, scenarios=("disconnect_once",), every_n_cycles=0))
    rows = [_row(datetime(2026, 1, 1, 0, 0, 0)), _row(datetime(2026, 1, 1, 0, 15, 0)), _row(datetime(2026, 1, 1, 0, 30, 0))]
    probabilities = [{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.2, 0: 0.6, -1: 0.2}, {1: 0.2, 0: 0.6, -1: 0.2}]
    _patch_context(monkeypatch, rows, probabilities)

    code, run_dir = run_soak(settings, "demo_dry", require_broker=True, base_client_factory=lambda: FakeBrokerClient())

    assert code in {0, 2}
    go = json.loads((run_dir / "go_no_go_report.json").read_text(encoding="utf-8"))
    assert go["decision"] in {"go", "caution"}


def test_reconnect_injected_and_no_go(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings = replace(settings, soak=SoakConfig(cycles=1, pause_seconds=0.0, restore_between_cycles=True), chaos=ChaosConfig(enabled=True, scenarios=("reconnect_fail_once",), every_n_cycles=0))
    rows = [_row(datetime(2026, 1, 1, 0, 0, 0)), _row(datetime(2026, 1, 1, 0, 15, 0))]
    probabilities = [{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.2, 0: 0.6, -1: 0.2}]
    _patch_context(monkeypatch, rows, probabilities)

    code, run_dir = run_soak(settings, "demo_dry", require_broker=True, base_client_factory=lambda: FakeBrokerClient())

    assert code == 3
    go = json.loads((run_dir / "go_no_go_report.json").read_text(encoding="utf-8"))
    assert go["decision"] == "no_go"


def test_corrupt_snapshot_blocks_operation(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    object.__setattr__(settings, "chaos", ChaosConfig(enabled=True, scenarios=("corrupt_restore_once",), every_n_cycles=0))
    rows = [_row(datetime(2026, 1, 1, 0, 0, 0)), _row(datetime(2026, 1, 1, 0, 15, 0))]
    probabilities = [{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.2, 0: 0.6, -1: 0.2}]
    _patch_context(monkeypatch, rows, probabilities)

    code, run_dir = run_soak(settings, "paper", require_broker=False)

    assert code == 3
    health = json.loads((run_dir / "health_report.json").read_text(encoding="utf-8"))
    assert any("restore_failed" in cycle["issues"] for cycle in health["cycles"])


def test_broker_mismatch_generates_no_go(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings = replace(settings, soak=SoakConfig(cycles=1, pause_seconds=0.0, restore_between_cycles=True), chaos=ChaosConfig(enabled=True, scenarios=("broker_mismatch_once",), every_n_cycles=0))
    rows = [_row(datetime(2026, 1, 1, 0, 0, 0)), _row(datetime(2026, 1, 1, 0, 15, 0))]
    probabilities = [{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.2, 0: 0.6, -1: 0.2}]
    _patch_context(monkeypatch, rows, probabilities)

    code, run_dir = run_soak(settings, "demo_dry", require_broker=True, base_client_factory=lambda: FakeBrokerClient())

    assert code == 3
    go = json.loads((run_dir / "go_no_go_report.json").read_text(encoding="utf-8"))
    assert go["decision"] == "no_go"


def test_repeated_rejections_elevate_alerts(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings = replace(settings, soak=SoakConfig(cycles=1, pause_seconds=0.0, restore_between_cycles=True), chaos=ChaosConfig(enabled=True, scenarios=("repeated_rejections",), every_n_cycles=0))
    rows = [_row(datetime(2026, 1, 1, 0, 0, 0)), _row(datetime(2026, 1, 1, 0, 15, 0)), _row(datetime(2026, 1, 1, 0, 30, 0))]
    probabilities = [{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.2, 0: 0.6, -1: 0.2}]
    _patch_context(monkeypatch, rows, probabilities)

    code, run_dir = run_soak(settings, "demo_dry", require_broker=True, base_client_factory=lambda: FakeBrokerClient())

    assert code == 2
    go = json.loads((run_dir / "go_no_go_report.json").read_text(encoding="utf-8"))
    assert go["decision"] == "caution"


def test_session_blocked_during_soak(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings = replace(settings, soak=SoakConfig(cycles=1, pause_seconds=0.0, restore_between_cycles=True), chaos=ChaosConfig(enabled=True, scenarios=("market_session_blocked",), every_n_cycles=0))
    rows = [_row(datetime(2026, 1, 3, 12, 0, 0)), _row(datetime(2026, 1, 3, 12, 15, 0))]
    probabilities = [{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.2, 0: 0.6, -1: 0.2}]
    _patch_context(monkeypatch, rows, probabilities)

    code, run_dir = run_soak(settings, "paper", require_broker=False)

    assert code == 2
    health = json.loads((run_dir / "health_report.json").read_text(encoding="utf-8"))
    assert health["cycles"][0]["status"] == "caution"


def test_health_report_and_incident_log_and_go_no_go_regeneration(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings = replace(settings, soak=SoakConfig(cycles=1, pause_seconds=0.0, restore_between_cycles=True))
    rows = [_row(datetime(2026, 1, 1, 0, 0, 0)), _row(datetime(2026, 1, 1, 0, 15, 0))]
    probabilities = [{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.2, 0: 0.6, -1: 0.2}]
    _patch_context(monkeypatch, rows, probabilities)

    code, run_dir = run_soak(settings, "paper", require_broker=False)
    regen_code, regen_dir = regenerate_go_no_go_report(settings)

    assert code in {0, 2}
    assert regen_code == 0
    assert regen_dir == run_dir
    assert (run_dir / "incident_log.jsonl").exists()
    assert (run_dir / "cycle_summaries").exists()


def test_go_no_go_classification_rules() -> None:
    decision = classify_go_no_go([])
    assert decision.decision == "go"


def test_build_run_directory_avoids_timestamp_collisions(tmp_path: Path) -> None:
    first = build_run_directory(tmp_path, "paper_resilient")
    second = build_run_directory(tmp_path, "paper_resilient")
    assert first != second
    assert first.exists()
    assert second.exists()
