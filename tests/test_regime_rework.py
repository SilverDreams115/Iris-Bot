from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from iris_bot.data import Bar, write_bars


pytest.importorskip("xgboost")


def _write_symbol_bars(raw_path: Path, *, counts_by_timeframe: dict[str, int], drift_scale: float) -> None:
    all_bars: list[Bar] = []
    start = datetime(2025, 6, 1, 0, 0, 0)
    symbols = ("GBPUSD", "EURUSD")
    base_prices = {"GBPUSD": 1.2700, "EURUSD": 1.0950}
    timeframe_minutes = {"M15": 15, "H1": 60}
    for symbol_index, symbol in enumerate(symbols):
        base_price = base_prices[symbol]
        for timeframe, count in counts_by_timeframe.items():
            price = base_price + symbol_index * 0.01
            step = drift_scale if symbol == "GBPUSD" else drift_scale * 0.65
            for index in range(count):
                regime = index % 36
                if regime < 18:
                    price += step
                elif regime < 24:
                    price -= step * 0.50
                else:
                    price += step * 0.15
                wick = step * (2.4 if regime < 18 else 1.5)
                spread = step * 0.65
                all_bars.append(
                    Bar(
                        timestamp=start + timedelta(minutes=timeframe_minutes[timeframe] * index),
                        symbol=symbol,
                        timeframe=timeframe,
                        open=max(price - spread, 0.0001),
                        high=max(price + wick, 0.0001),
                        low=max(price - wick, 0.0001),
                        close=max(price, 0.0001),
                        volume=float(100 + (index % 50)),
                    )
                )
    write_bars(raw_path, all_bars)


def _scoped_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    runtime_dir = tmp_path / "data" / "runtime"
    runs_dir = tmp_path / "runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    env_defaults = {
        "IRIS_TIMEFRAMES": "M15,H1",
        "IRIS_PRIMARY_TIMEFRAME": "M15",
        "IRIS_WF_TRAIN_WINDOW": "100",
        "IRIS_WF_VALIDATION_WINDOW": "40",
        "IRIS_WF_TEST_WINDOW": "40",
        "IRIS_WF_STEP": "40",
        "IRIS_XGB_NUM_BOOST_ROUND": "18",
        "IRIS_XGB_EARLY_STOPPING_ROUNDS": "5",
        "IRIS_MT5_HISTORY_BARS": "120",
    }
    for key, value in env_defaults.items():
        if key not in os.environ:
            monkeypatch.setenv(key, value)

    from iris_bot.config import load_settings

    settings = load_settings()
    object.__setattr__(settings.data, "raw_dir", raw_dir)
    object.__setattr__(settings.data, "processed_dir", processed_dir)
    object.__setattr__(settings.data, "runtime_dir", runtime_dir)
    object.__setattr__(settings.data, "runs_dir", runs_dir)
    object.__setattr__(settings.experiment, "_processed_dir", processed_dir)
    return settings


def _latest_run_dir(settings, suffix: str) -> Path:
    candidates = sorted(settings.data.runs_dir.glob(f"*_{suffix}"))
    assert candidates
    return candidates[-1]


def test_fetch_extended_history_writes_audited_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot import regime_rework

    settings = _scoped_settings(tmp_path, monkeypatch)
    _write_symbol_bars(settings.data.raw_dataset_path, counts_by_timeframe={"M15": 80, "H1": 80}, drift_scale=0.00030)

    class FakeMT5Client:
        def __init__(self, _config) -> None:
            self.connected = False

        def connect(self) -> bool:
            self.connected = True
            return True

        def shutdown(self) -> None:
            self.connected = False

        def fetch_historical_bars(self, symbol: str, timeframe: str, count: int) -> list[Bar]:
            assert count >= 120
            path = tmp_path / f"{symbol}_{timeframe}.csv"
            _write_symbol_bars(path, counts_by_timeframe={timeframe: 140}, drift_scale=0.00035)
            bars = [bar for bar in regime_rework.load_bars(path) if bar.symbol == symbol and bar.timeframe == timeframe]
            duplicate = bars[20]
            invalid = Bar(
                timestamp=bars[30].timestamp + timedelta(minutes=15 if timeframe == "M15" else 60),
                symbol=symbol,
                timeframe=timeframe,
                open=1.0,
                high=0.9,
                low=1.1,
                close=1.0,
                volume=1.0,
            )
            return bars + [duplicate, invalid]

    monkeypatch.setattr(regime_rework, "MT5Client", FakeMT5Client)
    assert regime_rework.run_fetch_extended_history(settings) == 0

    report_path = _latest_run_dir(settings, "fetch_extended_history") / "expanded_history_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))["payload"]
    assert Path(payload["extended_dataset"]["dataset_path"]).exists()
    assert payload["focus_symbol"] == "GBPUSD"
    assert payload["secondary_symbol"] == "EURUSD"
    for summary in payload["extended_dataset"]["series"]:
        assert summary["bars_downloaded"] >= summary["bars"]
        assert summary["duplicates_removed"] >= 1


def test_audit_regime_features_reports_explicit_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.regime_rework import EXTENDED_HISTORY_DATASET_NAME, run_audit_regime_features

    settings = _scoped_settings(tmp_path, monkeypatch)
    _write_symbol_bars(settings.data.raw_dir / EXTENDED_HISTORY_DATASET_NAME, counts_by_timeframe={"M15": 220, "H1": 220}, drift_scale=0.00032)

    assert run_audit_regime_features(settings) == 0
    report_path = _latest_run_dir(settings, "audit_regime_features") / "regime_feature_diagnostic_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))["payload"]
    assert payload["added_regime_features"] == [
        "adx_14",
        "trend_regime_flag",
        "high_volatility_regime_flag",
        "low_volatility_regime_flag",
    ]
    assert payload["diagnostics_by_symbol"]["GBPUSD"]["row_count"] > 0


def test_run_regime_aware_rework_generates_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.regime_rework import (
        EXTENDED_HISTORY_DATASET_NAME,
        run_compare_regime_experiments,
        run_evaluate_regime_demo_candidate,
        run_regime_aware_rework,
    )

    settings = _scoped_settings(tmp_path, monkeypatch)
    _write_symbol_bars(settings.data.raw_dataset_path, counts_by_timeframe={"M15": 240, "H1": 240}, drift_scale=0.00028)
    _write_symbol_bars(settings.data.raw_dir / EXTENDED_HISTORY_DATASET_NAME, counts_by_timeframe={"M15": 420, "H1": 420}, drift_scale=0.00034)

    assert run_regime_aware_rework(settings) == 0
    run_dir = _latest_run_dir(settings, "run_regime_aware_rework")
    focus = json.loads((run_dir / "symbol_focus_rework_report.json").read_text(encoding="utf-8"))["payload"]
    secondary = json.loads((run_dir / "symbol_secondary_comparison_report.json").read_text(encoding="utf-8"))["payload"]
    candidate = json.loads((run_dir / "demo_execution_candidate_report.json").read_text(encoding="utf-8"))["payload"]

    assert focus["symbol"] == "GBPUSD"
    assert focus["decision"] in {
        "REJECT_FOR_DEMO_EXECUTION",
        "IMPROVED_BUT_NOT_ENOUGH",
        "CANDIDATE_FOR_DEMO_EXECUTION",
    }
    assert secondary["symbol"] == "EURUSD"
    assert candidate["approved_for_demo_execution_exists"] is False

    assert run_compare_regime_experiments(settings) == 0
    assert run_evaluate_regime_demo_candidate(settings) == 0

    compare_payload = json.loads((_latest_run_dir(settings, "compare_regime_experiments") / "regime_aware_experiment_matrix_report.json").read_text(encoding="utf-8"))["payload"]
    evaluate_payload = json.loads((_latest_run_dir(settings, "evaluate_demo_candidate") / "demo_execution_candidate_report.json").read_text(encoding="utf-8"))["payload"]
    assert compare_payload["focus_symbol"] == "GBPUSD"
    assert evaluate_payload["approved_for_demo_execution_exists"] is False
