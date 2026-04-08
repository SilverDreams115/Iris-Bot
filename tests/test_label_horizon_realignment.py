from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from iris_bot.artifacts import read_artifact_payload
from iris_bot.data import Bar, write_bars


pytest.importorskip("xgboost")


def _write_dataset(raw_path: Path, *, bars_per_symbol: int) -> None:
    all_bars: list[Bar] = []
    start = datetime(2025, 1, 1, 0, 0, 0)
    for symbol_index, symbol in enumerate(("GBPUSD", "EURUSD")):
        price = 1.2700 if symbol == "GBPUSD" else 1.0950
        scale = 0.00028 if symbol == "GBPUSD" else 0.00022
        for index in range(bars_per_symbol):
            phase = (index + symbol_index * 5) % 36
            if phase < 12:
                price += scale * 1.1
            elif phase < 22:
                price -= scale * 0.75
            else:
                price += scale * 0.20
            wick = scale * (2.5 if phase < 12 else 1.6)
            all_bars.append(
                Bar(
                    timestamp=start + timedelta(minutes=15 * index),
                    symbol=symbol,
                    timeframe="M15",
                    open=max(price - scale * 0.45, 0.0001),
                    high=max(price + wick, 0.0001),
                    low=max(price - wick, 0.0001),
                    close=max(price, 0.0001),
                    volume=float(100 + (index % 30)),
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

    _write_dataset(raw_dir / "market.csv", bars_per_symbol=280)

    env_defaults = {
        "IRIS_TIMEFRAMES": "M15",
        "IRIS_PRIMARY_TIMEFRAME": "M15",
        "IRIS_WF_TRAIN_WINDOW": "100",
        "IRIS_WF_VALIDATION_WINDOW": "40",
        "IRIS_WF_TEST_WINDOW": "40",
        "IRIS_WF_STEP": "40",
        "IRIS_XGB_NUM_BOOST_ROUND": "16",
        "IRIS_XGB_EARLY_STOPPING_ROUNDS": "4",
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


def test_audit_trade_duration_and_timeout_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.label_horizon_realignment import run_audit_timeout_impact, run_audit_trade_duration

    settings = _scoped_settings(tmp_path, monkeypatch)
    assert run_audit_trade_duration(settings) == 0
    assert run_audit_timeout_impact(settings) == 0

    duration = read_artifact_payload(
        _latest_run_dir(settings, "audit_trade_duration") / "trade_duration_distribution_report.json",
        expected_type="trade_duration_distribution_report",
    )
    timeout = read_artifact_payload(
        _latest_run_dir(settings, "audit_timeout_impact") / "timeout_label_impact_report.json",
        expected_type="timeout_label_impact_report",
    )
    assert duration["focus_symbol"] == "GBPUSD"
    assert duration["symbols"]["GBPUSD"]["trade_duration_summary"]["trade_count"] >= 0
    assert "baseline_actual" in timeout["symbols"]["GBPUSD"]


def test_run_label_horizon_realignment_generates_matrix_and_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.label_horizon_realignment import (
        run_compare_exit_alignment,
        run_evaluate_label_horizon_candidate,
        run_label_horizon_realignment,
    )

    settings = _scoped_settings(tmp_path, monkeypatch)
    assert run_label_horizon_realignment(settings) == 0

    run_dir = _latest_run_dir(settings, "run_label_horizon_realignment")
    matrix = read_artifact_payload(run_dir / "label_horizon_exit_matrix_report.json", expected_type="label_horizon_exit_matrix_report")
    candidate = read_artifact_payload(run_dir / "demo_execution_candidate_report.json", expected_type="demo_execution_candidate_report")
    threshold = read_artifact_payload(run_dir / "threshold_utility_report.json", expected_type="threshold_utility_report")

    assert matrix["focus_symbol"] == "GBPUSD"
    assert matrix["selected_focus_variant"]
    assert candidate["approved_for_demo_execution_exists"] is False
    assert "GBPUSD" in threshold["symbols"]

    assert run_compare_exit_alignment(settings) == 0
    assert run_evaluate_label_horizon_candidate(settings) == 0

    compare = read_artifact_payload(
        _latest_run_dir(settings, "compare_exit_alignment") / "label_horizon_exit_matrix_report.json",
        expected_type="label_horizon_exit_matrix_report",
    )
    evaluated = read_artifact_payload(
        _latest_run_dir(settings, "evaluate_label_horizon_candidate") / "demo_execution_candidate_report.json",
        expected_type="demo_execution_candidate_report",
    )
    assert compare["focus_symbol"] == "GBPUSD"
    assert evaluated["approved_for_demo_execution_exists"] is False
