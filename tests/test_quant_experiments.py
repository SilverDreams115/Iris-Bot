import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from iris_bot.data import Bar, write_bars
from iris_bot.quant_experiments import (
    audit_effective_config_payload,
    audit_symbol_context_payload,
    run_experiment_matrix,
)


pytest.importorskip("xgboost")


def _write_multi_symbol_dataset(raw_path: Path, bars_per_symbol: int = 180) -> None:
    symbols = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")
    all_bars: list[Bar] = []
    start = datetime(2026, 1, 1, 0, 0, 0)
    base_prices = {
        "EURUSD": 1.1000,
        "GBPUSD": 1.2700,
        "USDJPY": 145.00,
        "AUDUSD": 0.6700,
    }
    for symbol_index, symbol in enumerate(symbols):
        price = base_prices[symbol]
        scale = 0.00035 if symbol != "USDJPY" else 0.035
        for index in range(bars_per_symbol):
            phase = (index + symbol_index) % 9
            if phase in (0, 1, 2, 3):
                price += scale
            elif phase in (4, 5):
                price -= scale * 0.75
            else:
                price += scale * 0.15
            spread = 0.00015 if symbol != "USDJPY" else 0.015
            all_bars.append(
                Bar(
                    timestamp=start + timedelta(minutes=15 * index),
                    symbol=symbol,
                    timeframe="M15",
                    open=price - spread,
                    high=price + spread * 3,
                    low=price - spread * 3,
                    close=price,
                    volume=100 + ((index + symbol_index) % 30),
                )
            )
    write_bars(raw_path, all_bars)


def _scoped_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    runs_dir = tmp_path / "runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    _write_multi_symbol_dataset(raw_dir / "market.csv")

    if "IRIS_PRIMARY_TIMEFRAME" not in os.environ:
        monkeypatch.setenv("IRIS_PRIMARY_TIMEFRAME", "M15")
    if "IRIS_WALK_FORWARD_ENABLED" not in os.environ:
        monkeypatch.setenv("IRIS_WALK_FORWARD_ENABLED", "true")
    if "IRIS_WF_TRAIN_WINDOW" not in os.environ:
        monkeypatch.setenv("IRIS_WF_TRAIN_WINDOW", "120")
    if "IRIS_WF_VALIDATION_WINDOW" not in os.environ:
        monkeypatch.setenv("IRIS_WF_VALIDATION_WINDOW", "40")
    if "IRIS_WF_TEST_WINDOW" not in os.environ:
        monkeypatch.setenv("IRIS_WF_TEST_WINDOW", "40")
    if "IRIS_WF_STEP" not in os.environ:
        monkeypatch.setenv("IRIS_WF_STEP", "40")
    if "IRIS_XGB_NUM_BOOST_ROUND" not in os.environ:
        monkeypatch.setenv("IRIS_XGB_NUM_BOOST_ROUND", "40")
    if "IRIS_XGB_EARLY_STOPPING_ROUNDS" not in os.environ:
        monkeypatch.setenv("IRIS_XGB_EARLY_STOPPING_ROUNDS", "8")

    from iris_bot.config import load_settings

    settings = load_settings()
    object.__setattr__(settings.data, "raw_dir", raw_dir)
    object.__setattr__(settings.data, "processed_dir", processed_dir)
    object.__setattr__(settings.data, "runs_dir", runs_dir)
    object.__setattr__(settings.experiment, "_processed_dir", processed_dir)
    return settings


def test_audit_effective_config_reports_runtime_value_and_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_XGB_NUM_BOOST_ROUND", "123")
    settings = _scoped_settings(tmp_path, monkeypatch)

    payload = audit_effective_config_payload(settings)

    tracked = payload["tracked_variables"]["IRIS_XGB_NUM_BOOST_ROUND"]
    assert tracked["effective_value"] == 123
    assert tracked["source"] == "process_env"
    assert tracked["in_process_env"] is True


def test_audit_symbol_context_detects_global_symbol_mixing_without_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _scoped_settings(tmp_path, monkeypatch)

    payload = audit_symbol_context_payload(settings)

    assert payload["model_sees_explicit_symbol_feature"] is False
    assert payload["model_sees_explicit_timeframe_feature"] is False
    assert payload["training_is_global_mixed_across_symbols"] is True
    assert payload["primary_timeframe_only"] is True
    assert payload["per_symbol_rows"]["EURUSD"] > 0


def test_run_experiment_matrix_writes_controlled_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _scoped_settings(tmp_path, monkeypatch)

    exit_code = run_experiment_matrix(settings)

    assert exit_code == 0
    run_dirs = sorted(settings.data.runs_dir.glob("*_experiment_matrix"))
    assert run_dirs
    run_dir = run_dirs[-1]
    matrix_report = json.loads((run_dir / "experiment_matrix_report.json").read_text(encoding="utf-8"))
    recommendation = json.loads((run_dir / "experiment_recommendation_report.json").read_text(encoding="utf-8"))
    xgb_report = json.loads((run_dir / "xgb_tuning_comparison_report.json").read_text(encoding="utf-8"))
    per_symbol = json.loads((run_dir / "per_symbol_experiment_report.json").read_text(encoding="utf-8"))

    assert matrix_report["baseline_experiment_id"] == "exp0_baseline"
    assert "exp2_labels_asymmetric" in matrix_report["experiments"]
    assert "exp4_symbol_context" in matrix_report["experiments"]
    assert set(xgb_report["variants"]) == {"exp3_xgb_mcw_5", "exp3_xgb_mcw_8", "exp3_xgb_mcw_20"}
    assert recommendation["final_decision"] in {
        "KEEP_BASELINE",
        "ADOPT_CONFIG_FIX_ONLY",
        "ADOPT_LABEL_CHANGE",
        "ADOPT_XGB_TUNING",
        "ADOPT_SYMBOL_CONTEXT_CHANGE",
        "ADOPT_COMBINED_CHANGE",
        "REQUIRE_STRUCTURAL_REWORK",
    }
    assert "exp0_baseline" in per_symbol
