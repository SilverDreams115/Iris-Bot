from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from iris_bot.artifacts import read_artifact_payload
from iris_bot.data import Bar, write_bars


pytest.importorskip("xgboost")


def _write_edge_dataset(raw_path: Path, *, bars_per_symbol: int, drift_scale: float) -> None:
    all_bars: list[Bar] = []
    start = datetime(2025, 1, 1, 0, 0, 0)
    base_prices = {"GBPUSD": 1.2700, "EURUSD": 1.0950}
    for symbol_index, symbol in enumerate(("GBPUSD", "EURUSD")):
        price = base_prices[symbol] + symbol_index * 0.01
        for index in range(bars_per_symbol):
            regime = index % 48
            if regime < 18:
                price += drift_scale * (1.2 if symbol == "GBPUSD" else 0.8)
            elif regime < 30:
                price -= drift_scale * (0.9 if symbol == "GBPUSD" else 0.7)
            else:
                price += drift_scale * 0.15
            wick = drift_scale * (2.8 if regime < 18 else 1.6)
            spread = drift_scale * 0.60
            all_bars.append(
                Bar(
                    timestamp=start + timedelta(minutes=15 * index),
                    symbol=symbol,
                    timeframe="M15",
                    open=max(price - spread, 0.0001),
                    high=max(price + wick, 0.0001),
                    low=max(price - wick, 0.0001),
                    close=max(price, 0.0001),
                    volume=float(100 + (index % 40)),
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

    _write_edge_dataset(raw_dir / "market.csv", bars_per_symbol=280, drift_scale=0.00030)
    _write_edge_dataset(raw_dir / "market_extended.csv", bars_per_symbol=420, drift_scale=0.00034)

    env_defaults = {
        "IRIS_TIMEFRAMES": "M15",
        "IRIS_PRIMARY_TIMEFRAME": "M15",
        "IRIS_WF_TRAIN_WINDOW": "100",
        "IRIS_WF_VALIDATION_WINDOW": "40",
        "IRIS_WF_TEST_WINDOW": "40",
        "IRIS_WF_STEP": "40",
        "IRIS_XGB_NUM_BOOST_ROUND": "18",
        "IRIS_XGB_EARLY_STOPPING_ROUNDS": "5",
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


def test_audit_edge_baseline_generates_consistent_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.edge_diagnostics import run_audit_edge_baseline

    settings = _scoped_settings(tmp_path, monkeypatch)
    assert run_audit_edge_baseline(settings) == 0

    payload = read_artifact_payload(
        _latest_run_dir(settings, "audit_edge_baseline") / "baseline_edge_diagnostic_report.json",
        expected_type="baseline_edge_diagnostic_report",
    )
    assert payload["focus_symbol"] == "GBPUSD"
    assert payload["secondary_symbol"] == "EURUSD"
    gbp = payload["baseline_symbol_reports"]["GBPUSD"]
    assert gbp["symbol"] == "GBPUSD"
    assert gbp["split_metrics"]["test"]["row_count"] > 0
    assert "performance_by_regime" in gbp
    assert "walk_forward" in gbp


def test_label_noise_and_class_separability_reports_stay_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.edge_diagnostics import run_audit_class_separability, run_audit_label_noise

    settings = _scoped_settings(tmp_path, monkeypatch)
    assert run_audit_label_noise(settings) == 0
    assert run_audit_class_separability(settings) == 0

    label_payload = read_artifact_payload(
        _latest_run_dir(settings, "audit_label_noise") / "label_noise_report.json",
        expected_type="label_noise_report",
    )
    separability_payload = read_artifact_payload(
        _latest_run_dir(settings, "audit_class_separability") / "class_separability_report.json",
        expected_type="class_separability_report",
    )

    for symbol in ("GBPUSD", "EURUSD"):
        assert 0.0 <= label_payload["symbols"][symbol]["timeout_ratio"] <= 1.0
        assert 0.0 <= label_payload["symbols"][symbol]["error_share_timeout_labels"] <= 1.0
        stability = separability_payload["symbols"][symbol]["walk_forward"]["threshold_stability"]
        assert stability["min"] <= stability["mean"] <= stability["max"]


def test_audit_edge_hypotheses_generates_ranked_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.edge_diagnostics import FINAL_EDGE_DECISIONS, run_audit_edge_hypotheses

    settings = _scoped_settings(tmp_path, monkeypatch)
    assert run_audit_edge_hypotheses(settings) == 0

    run_dir = _latest_run_dir(settings, "audit_edge_hypotheses")
    expected_files = {
        "baseline_edge_diagnostic_report.json",
        "label_noise_report.json",
        "horizon_exit_alignment_report.json",
        "regime_value_report.json",
        "class_separability_report.json",
        "costly_error_analysis_report.json",
        "hypothesis_matrix_report.json",
        "edge_diagnosis_recommendation_report.json",
    }
    assert expected_files.issubset({path.name for path in run_dir.iterdir() if path.is_file()})

    matrix = read_artifact_payload(run_dir / "hypothesis_matrix_report.json", expected_type="hypothesis_matrix_report")
    recommendation = read_artifact_payload(
        run_dir / "edge_diagnosis_recommendation_report.json",
        expected_type="edge_diagnosis_recommendation_report",
    )
    scores = [item["support_score"] for item in matrix["hypotheses"]]
    assert scores == sorted(scores, reverse=True)
    assert matrix["final_conservative_decision"] in FINAL_EDGE_DECISIONS
    assert recommendation["final_conservative_decision"] == matrix["final_conservative_decision"]
