from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from iris_bot.artifacts import read_artifact_payload
from iris_bot.data import Bar, write_bars


pytest.importorskip("xgboost")


def _write_dataset(raw_path: Path, *, bars_per_symbol: int = 320) -> None:
    all_bars: list[Bar] = []
    start = datetime(2025, 2, 1, 0, 0, 0)
    for symbol_index, symbol in enumerate(("GBPUSD", "EURUSD")):
        price = 1.2700 if symbol == "GBPUSD" else 1.0950
        scale = 0.00030 if symbol == "GBPUSD" else 0.00022
        for index in range(bars_per_symbol):
            phase = (index + symbol_index * 7) % 40
            if phase < 14:
                price += scale * 1.05
            elif phase < 24:
                price -= scale * 0.72
            else:
                price += scale * 0.18
            wick = scale * (2.4 if phase < 14 else 1.7)
            all_bars.append(
                Bar(
                    timestamp=start + timedelta(minutes=15 * index),
                    symbol=symbol,
                    timeframe="M15",
                    open=max(price - scale * 0.40, 0.0001),
                    high=max(price + wick, 0.0001),
                    low=max(price - wick, 0.0001),
                    close=max(price, 0.0001),
                    volume=float(100 + (index % 25)),
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
    _write_dataset(raw_dir / "market.csv")

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


def test_audit_exit_lifecycle_writes_diagnostic_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.exit_lifecycle_realignment import run_audit_exit_lifecycle

    settings = _scoped_settings(tmp_path, monkeypatch)
    assert run_audit_exit_lifecycle(settings) == 0
    payload = read_artifact_payload(
        _latest_run_dir(settings, "audit_exit_lifecycle") / "exit_lifecycle_diagnostic_report.json",
        expected_type="exit_lifecycle_diagnostic_report",
    )
    assert payload["focus_symbol"] == "GBPUSD"
    gbp = payload["symbols"]["GBPUSD"]
    assert gbp["variant_id"] == "baseline_h12_actual"
    assert "trade_duration_summary" in gbp
    assert "exit_reason_metrics" in gbp


def test_run_h12_exit_realignment_generates_matrix_and_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.exit_lifecycle_realignment import (
        run_compare_h12_exit_variants,
        run_evaluate_gbpusd_demo_candidate,
        run_h12_exit_realignment,
    )

    settings = _scoped_settings(tmp_path, monkeypatch)
    assert run_h12_exit_realignment(settings) == 0
    run_dir = _latest_run_dir(settings, "run_h12_exit_realignment")

    matrix = read_artifact_payload(run_dir / "h12_exit_variant_matrix_report.json", expected_type="h12_exit_variant_matrix_report")
    reduction = read_artifact_payload(
        run_dir / "timeout_timeexit_reduction_report.json",
        expected_type="timeout_timeexit_reduction_report",
    )
    preservation = read_artifact_payload(
        run_dir / "trade_count_preservation_report.json",
        expected_type="trade_count_preservation_report",
    )
    candidate = read_artifact_payload(
        run_dir / "demo_execution_candidate_report.json",
        expected_type="demo_execution_candidate_report",
    )

    assert matrix["focus_symbol"] == "GBPUSD"
    assert matrix["selected_focus_variant"]
    assert "overall_focus_conclusion" in matrix
    assert "baseline_h12_actual" in reduction["focus_reduction"]
    assert "GBPUSD" in preservation["symbols"]
    assert candidate["approved_for_demo_execution_exists"] is False

    assert run_compare_h12_exit_variants(settings) == 0
    assert run_evaluate_gbpusd_demo_candidate(settings) == 0

    compact = read_artifact_payload(
        _latest_run_dir(settings, "compare_h12_exit_variants") / "h12_exit_variant_matrix_report.json",
        expected_type="h12_exit_variant_matrix_report",
    )
    evaluated = read_artifact_payload(
        _latest_run_dir(settings, "evaluate_gbpusd_demo_candidate") / "demo_execution_candidate_report.json",
        expected_type="demo_execution_candidate_report",
    )
    assert compact["focus_symbol"] == "GBPUSD"
    assert evaluated["approved_for_demo_execution_exists"] is False
