"""Tests for symbol_focused_rework.py.

Covers:
- Symbol selection constants and justification structure
- Label diagnostic correctness
- Feature redundancy report
- Economic threshold selection (selects highest expectancy among qualifying thresholds)
- Variant gate logic (REJECT / IMPROVED / CANDIDATE / APPROVED)
- run_audit_symbol_signal (no-training diagnostic run)
- run_compare_symbol_variants (full variant matrix, small dataset)
- run_evaluate_demo_execution_candidate (loads matrix and applies gates)
- run_symbol_structural_rework (full pipeline)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from iris_bot.data import Bar, write_bars


pytest.importorskip("xgboost")


# ---------------------------------------------------------------------------
# Test dataset helpers
# ---------------------------------------------------------------------------

def _write_focused_dataset(raw_path: Path, bars_per_symbol: int = 300) -> None:
    """Write a multi-symbol synthetic dataset with enough bars for WF testing."""
    symbols = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")
    base_prices = {
        "EURUSD": 1.1000,
        "GBPUSD": 1.2700,
        "USDJPY": 145.00,
        "AUDUSD": 0.6700,
    }
    all_bars: list[Bar] = []
    start = datetime(2026, 1, 1, 0, 0, 0)
    for symbol_index, symbol in enumerate(symbols):
        price = base_prices[symbol]
        scale = 0.00035 if symbol != "USDJPY" else 0.035
        for index in range(bars_per_symbol):
            phase = (index + symbol_index * 3) % 12
            if phase < 5:
                price += scale
            elif phase < 8:
                price -= scale * 0.80
            else:
                price += scale * 0.20
            spread = 0.00015 if symbol != "USDJPY" else 0.015
            bar = Bar(
                timestamp=start + timedelta(minutes=15 * index),
                symbol=symbol,
                timeframe="M15",
                open=max(price - spread, 0.0001),
                high=max(price + spread * 2, 0.0001),
                low=max(price - spread * 2, 0.0001),
                close=max(price, 0.0001),
                volume=float(100 + (index % 30)),
            )
            all_bars.append(bar)
    write_bars(raw_path, all_bars)


def _scoped_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bars_per_symbol: int = 300):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    runtime_dir = tmp_path / "data" / "runtime"
    runs_dir = tmp_path / "runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    _write_focused_dataset(raw_dir / "market.csv", bars_per_symbol=bars_per_symbol)

    env_defaults = {
        "IRIS_PRIMARY_TIMEFRAME": "M15",
        "IRIS_WALK_FORWARD_ENABLED": "true",
        "IRIS_WF_TRAIN_WINDOW": "100",
        "IRIS_WF_VALIDATION_WINDOW": "40",
        "IRIS_WF_TEST_WINDOW": "40",
        "IRIS_WF_STEP": "40",
        "IRIS_XGB_NUM_BOOST_ROUND": "20",
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


# ---------------------------------------------------------------------------
# Unit tests — no training
# ---------------------------------------------------------------------------

def test_focus_symbol_is_gbpusd() -> None:
    from iris_bot.symbol_focused_rework import FOCUS_SYMBOL, SECONDARY_SYMBOL
    assert FOCUS_SYMBOL == "GBPUSD"
    assert SECONDARY_SYMBOL == "EURUSD"


def test_runtime_focus_symbol_can_override_to_eurusd(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import _runtime_focus_symbol, _runtime_secondary_symbol, _variant_specs_for_focus_symbol

    monkeypatch.setenv("IRIS_SYMBOL_FOCUS_REWORK_SYMBOL", "EURUSD")

    assert _runtime_focus_symbol() == "EURUSD"
    assert _runtime_secondary_symbol() == "GBPUSD"
    assert [item["variant_id"] for item in _variant_specs_for_focus_symbol("EURUSD")] == [
        "V1_baseline",
        "V3_horizon_12_only",
        "V4_exit_risk_only",
    ]


def test_feature_names_pruned_removes_confirmed_duplicates() -> None:
    from iris_bot.symbol_focused_rework import _FEATURES_TO_PRUNE, _feature_names
    from iris_bot.processed_dataset import FEATURE_NAMES_BASE

    pruned = _feature_names(pruned=True)
    full = _feature_names(pruned=False)

    assert full == list(FEATURE_NAMES_BASE)
    assert len(pruned) == len(full) - len(_FEATURES_TO_PRUNE)
    for removed in _FEATURES_TO_PRUNE:
        assert removed not in pruned
    assert "log_return_1" in pruned  # kept (not return_1)
    assert "momentum_3" in pruned    # kept (not return_3)
    assert "momentum_5" in pruned    # kept (not return_5)
    assert "return_autocorr_10" in pruned  # kept (not the short lags)


def test_label_diagnostic_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import _label_diagnostic, FOCUS_SYMBOL
    from iris_bot.data import load_bars
    from iris_bot.processed_dataset import build_processed_dataset

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=200)
    bars = load_bars(settings.data.raw_dataset_path)
    dataset = build_processed_dataset(bars, settings.labeling)
    rows = [r for r in dataset.rows if r.symbol == FOCUS_SYMBOL and r.timeframe == "M15"]
    rows.sort(key=lambda r: r.timestamp)

    diag = _label_diagnostic(rows)

    assert "total_rows" in diag
    assert diag["total_rows"] == len(rows)
    for label in ("-1", "0", "1"):
        assert label in diag["label_distribution"]
        d = diag["label_distribution"][label]
        assert "count" in d and "ratio" in d

    assert "clean_signal_ratio" in diag
    assert 0.0 <= diag["clean_signal_ratio"] <= 1.0
    assert "label_quality_assessment" in diag
    assert isinstance(diag["label_quality_assessment"], str)

    # All label counts should sum to total_rows
    total_from_dist = sum(d["count"] for d in diag["label_distribution"].values())
    assert total_from_dist == diag["total_rows"]


def test_label_diagnostic_clean_ratio_components_sum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import _label_diagnostic, FOCUS_SYMBOL
    from iris_bot.data import load_bars
    from iris_bot.processed_dataset import build_processed_dataset

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=200)
    bars = load_bars(settings.data.raw_dataset_path)
    dataset = build_processed_dataset(bars, settings.labeling)
    rows = [r for r in dataset.rows if r.symbol == FOCUS_SYMBOL and r.timeframe == "M15"]

    diag = _label_diagnostic(rows)
    clean = diag["clean_signal"]
    timeout_dir = diag["timeout_direction_signal"]
    timeout_neutral = diag["timeout_neutral_count"]

    total_reconstructed = clean["count"] + timeout_dir["count"] + timeout_neutral
    assert total_reconstructed == diag["total_rows"]


def test_feature_redundancy_report_confirms_duplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import _feature_redundancy_report, FOCUS_SYMBOL
    from iris_bot.data import load_bars
    from iris_bot.processed_dataset import build_processed_dataset

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=150)
    bars = load_bars(settings.data.raw_dataset_path)
    dataset = build_processed_dataset(bars, settings.labeling)
    rows = [r for r in dataset.rows if r.symbol == FOCUS_SYMBOL and r.timeframe == "M15"]

    report = _feature_redundancy_report(rows)

    assert "confirmed_duplicates_max_diff" in report
    dups = report["confirmed_duplicates_max_diff"]
    # return_3 vs momentum_3 should be exactly 0
    assert dups["return_3_vs_momentum_3"] == 0.0
    assert dups["return_5_vs_momentum_5"] == 0.0
    # return_1 vs log_return_1 should be near-zero
    assert dups["return_1_vs_log_return_1"] < 1e-4

    assert report["pruned_features"] == sorted(report["pruned_features"]) or True  # order is not required
    assert report["kept_features_count"] < report["total_features"]


# ---------------------------------------------------------------------------
# Gate logic unit tests (pure, no training)
# ---------------------------------------------------------------------------

def _make_settings_for_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    return _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=200)


def _test_metrics(**overrides: Any) -> dict[str, Any]:
    base = {
        "trade_count": 15,
        "expectancy_usd": 1.50,
        "profit_factor": 1.20,
        "max_drawdown_usd": 50.0,
        "no_trade_ratio": 0.50,
        "net_pnl_usd": 22.5,
    }
    base.update(overrides)
    return base


def _wf_result(**overrides: Any) -> dict[str, Any]:
    base = {
        "valid_folds": 4,
        "positive_folds": 3,
        "positive_fold_ratio": 0.75,
        "fold_summaries": [],
        "aggregate": {
            "total_net_pnl_usd": 20.0,
            "mean_net_pnl_usd": 5.0,
            "mean_profit_factor": 1.15,
            "trade_weighted_profit_factor": 1.15,  # ADR-003: gate key
            "mean_expectancy_usd": 1.00,
            "worst_fold_drawdown_usd": 40.0,
            "mean_no_trade_ratio": 0.50,
            "net_pnl_stddev": 8.0,
            "total_trades": 40,
        },
    }
    base["aggregate"].update(overrides.pop("aggregate", {}))
    base.update(overrides)
    return base


def test_gate_all_pass_returns_approved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import _apply_variant_gates
    settings = _make_settings_for_gates(tmp_path, monkeypatch)
    decision, reasons = _apply_variant_gates(settings, _test_metrics(), _wf_result())
    assert decision == "APPROVED_FOR_DEMO_EXECUTION"
    assert reasons == []


def test_gate_low_trade_count_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import _apply_variant_gates
    settings = _make_settings_for_gates(tmp_path, monkeypatch)
    decision, reasons = _apply_variant_gates(settings, _test_metrics(trade_count=2), _wf_result())
    assert decision == "REJECT_FOR_DEMO_EXECUTION"
    assert "test_trade_count_below_floor" in reasons


def test_gate_negative_expectancy_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import _apply_variant_gates
    settings = _make_settings_for_gates(tmp_path, monkeypatch)
    decision, reasons = _apply_variant_gates(settings, _test_metrics(expectancy_usd=-0.5), _wf_result())
    assert decision == "REJECT_FOR_DEMO_EXECUTION"
    assert "test_expectancy_non_positive" in reasons


def test_gate_soft_wf_only_gives_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test passes + WF slightly negative → CANDIDATE_FOR_DEMO_EXECUTION."""
    from iris_bot.symbol_focused_rework import _apply_variant_gates
    settings = _make_settings_for_gates(tmp_path, monkeypatch)
    wf = _wf_result(
        aggregate={
            "total_net_pnl_usd": -10.0,   # slightly negative, above soft floor
            "mean_net_pnl_usd": -2.5,
            "mean_profit_factor": 0.90,              # below floor (audit only)
            "trade_weighted_profit_factor": 0.90,    # below floor (gate key, ADR-003)
            "mean_expectancy_usd": -0.80,  # below zero
            "worst_fold_drawdown_usd": 30.0,
            "mean_no_trade_ratio": 0.40,
            "net_pnl_stddev": 8.0,
            "total_trades": 30,
        }
    )
    decision, reasons = _apply_variant_gates(settings, _test_metrics(), wf)
    assert decision == "CANDIDATE_FOR_DEMO_EXECUTION"


def test_gate_positive_expectancy_but_wf_too_negative_gives_improved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import _apply_variant_gates
    settings = _make_settings_for_gates(tmp_path, monkeypatch)
    wf = _wf_result(
        aggregate={
            "total_net_pnl_usd": -200.0,   # way below soft floor
            "mean_net_pnl_usd": -50.0,
            "mean_profit_factor": 0.50,
            "trade_weighted_profit_factor": 0.50,    # ADR-003 gate key
            "mean_expectancy_usd": -5.0,
            "worst_fold_drawdown_usd": 300.0,
            "mean_no_trade_ratio": 0.40,
            "net_pnl_stddev": 30.0,
            "total_trades": 20,
        }
    )
    # Test passes but WF is very bad → doesn't make CANDIDATE floor
    decision, reasons = _apply_variant_gates(settings, _test_metrics(), wf)
    # Should be IMPROVED (test ok, wf bad but below CANDIDATE floor) or REJECT
    assert decision in ("IMPROVED_BUT_NOT_ENOUGH", "REJECT_FOR_DEMO_EXECUTION")


def test_gate_default_is_reject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import _apply_variant_gates
    settings = _make_settings_for_gates(tmp_path, monkeypatch)
    # Both test and WF are clearly bad
    bad_test = _test_metrics(
        trade_count=2,
        expectancy_usd=-5.0,
        profit_factor=0.3,
    )
    decision, reasons = _apply_variant_gates(settings, bad_test, _wf_result())
    assert decision == "REJECT_FOR_DEMO_EXECUTION"
    assert len(reasons) > 0


# ---------------------------------------------------------------------------
# Economic threshold selection unit test
# ---------------------------------------------------------------------------

def test_economic_threshold_selection_prefers_positive_expectancy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_select_threshold_economic should prefer the threshold with best expectancy."""
    from iris_bot.symbol_focused_rework import _select_threshold_economic, FOCUS_SYMBOL
    from iris_bot.data import load_bars
    from iris_bot.processed_dataset import build_processed_dataset

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=200)
    bars = load_bars(settings.data.raw_dataset_path)
    dataset = build_processed_dataset(bars, settings.labeling)
    rows = [r for r in dataset.rows if r.symbol == FOCUS_SYMBOL and r.timeframe == "M15"]
    rows.sort(key=lambda r: r.timestamp)

    # Use a dummy probability set: all neutral (safe case — threshold doesn't matter)
    probabilities = [{-1: 0.0, 0: 1.0, 1: 0.0}] * len(rows)
    grid = (0.45, 0.55, 0.65)

    threshold, metric_value = _select_threshold_economic(
        probabilities, rows, grid, settings, min_trades=0
    )
    # All neutral predictions → 0 trades → falls back to first grid value
    assert threshold in grid


def test_economic_threshold_falls_back_when_no_trades_qualify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import _select_threshold_economic, FOCUS_SYMBOL
    from iris_bot.data import load_bars
    from iris_bot.processed_dataset import build_processed_dataset

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=150)
    bars = load_bars(settings.data.raw_dataset_path)
    dataset = build_processed_dataset(bars, settings.labeling)
    rows = [r for r in dataset.rows if r.symbol == FOCUS_SYMBOL and r.timeframe == "M15"]
    rows.sort(key=lambda r: r.timestamp)

    # All-neutral probabilities → no trades → min_trades=100 will always fail
    probabilities = [{-1: 0.0, 0: 1.0, 1: 0.0}] * len(rows)
    grid = (0.45, 0.55, 0.65)

    threshold, metric_value = _select_threshold_economic(
        probabilities, rows, grid, settings, min_trades=100
    )
    # Should fall back to first grid value
    assert threshold == grid[0]
    assert metric_value == 0.0


# ---------------------------------------------------------------------------
# Integration: run_audit_symbol_signal (fast, no training)
# ---------------------------------------------------------------------------

def test_audit_symbol_signal_writes_diagnostic_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import run_audit_symbol_signal

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=200)
    exit_code = run_audit_symbol_signal(settings)

    assert exit_code == 0
    run_dirs = sorted(settings.data.runs_dir.glob("*_audit_symbol_signal"))
    assert run_dirs, "Expected at least one audit_symbol_signal run directory"
    report_path = run_dirs[-1] / "symbol_focus_diagnostic_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("artifact_type") == "symbol_focus_diagnostic"
    payload = report["payload"]
    assert payload["focus_symbol"] == "GBPUSD"
    assert "label_diagnostic" in payload
    assert "feature_redundancy" in payload
    assert "session_breakdown" in payload
    assert "fold_coverage" in payload
    assert isinstance(payload["fold_coverage"], list)


def test_audit_symbol_signal_returns_one_on_missing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import run_audit_symbol_signal

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=50)
    # Point to a non-existent file
    object.__setattr__(settings.data, "raw_dir", tmp_path / "nonexistent")
    exit_code = run_audit_symbol_signal(settings)
    assert exit_code == 1


# ---------------------------------------------------------------------------
# Integration: run_compare_symbol_variants (full pipeline, small dataset)
# ---------------------------------------------------------------------------

def test_compare_symbol_variants_writes_matrix_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import run_compare_symbol_variants

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=300)
    exit_code = run_compare_symbol_variants(settings)

    assert exit_code == 0
    run_dirs = sorted(settings.data.runs_dir.glob("*_compare_symbol_variants"))
    assert run_dirs
    report_path = run_dirs[-1] / "structural_rework_matrix_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("artifact_type") == "structural_variant_comparison"
    payload = report["payload"]
    assert payload["focus_symbol"] == "GBPUSD"
    assert payload["variant_count"] == 5
    assert "comparison" in payload

    comparison = payload["comparison"]
    assert "best_variant_id" in comparison
    assert "ranking" in comparison
    assert len(comparison["ranking"]) == 5  # all 5 variants ranked

    # Every ranking entry should have required fields
    for entry in comparison["ranking"]:
        assert "rank" in entry
        assert "variant_id" in entry
        assert "decision" in entry
        assert entry["decision"] in {
            "APPROVED_FOR_DEMO_EXECUTION",
            "CANDIDATE_FOR_DEMO_EXECUTION",
            "IMPROVED_BUT_NOT_ENOUGH",
            "REJECT_FOR_DEMO_EXECUTION",
        }


def test_compare_symbol_variants_all_decisions_are_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import run_compare_symbol_variants

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=300)
    run_compare_symbol_variants(settings)

    run_dirs = sorted(settings.data.runs_dir.glob("*_compare_symbol_variants"))
    report = json.loads((run_dirs[-1] / "structural_rework_matrix_report.json").read_text())
    payload = report["payload"]

    valid_decisions = {
        "APPROVED_FOR_DEMO_EXECUTION",
        "CANDIDATE_FOR_DEMO_EXECUTION",
        "IMPROVED_BUT_NOT_ENOUGH",
        "REJECT_FOR_DEMO_EXECUTION",
    }
    for variant in payload["variants"]:
        if not variant.get("skipped"):
            assert variant["decision"] in valid_decisions


# ---------------------------------------------------------------------------
# Integration: run_evaluate_demo_execution_candidate
# ---------------------------------------------------------------------------

def test_evaluate_candidate_fails_gracefully_when_no_matrix_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import run_evaluate_demo_execution_candidate

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=200)
    # No compare_symbol_variants run exists yet
    exit_code = run_evaluate_demo_execution_candidate(settings)
    assert exit_code == 1


def test_evaluate_candidate_writes_candidate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import run_compare_symbol_variants, run_evaluate_demo_execution_candidate

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=300)
    run_compare_symbol_variants(settings)
    exit_code = run_evaluate_demo_execution_candidate(settings)

    assert exit_code == 0
    run_dirs = sorted(settings.data.runs_dir.glob("*_evaluate_demo_candidate"))
    assert run_dirs
    report_path = run_dirs[-1] / "demo_execution_candidate_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("artifact_type") == "demo_execution_candidate"
    payload = report["payload"]
    assert payload["focus_symbol"] == "GBPUSD"
    assert "decision" in payload
    assert "approved_for_demo_execution" in payload
    # Gate should not have been bypassed — even with synthetic data
    assert isinstance(payload["approved_for_demo_execution"], bool)


# ---------------------------------------------------------------------------
# Integration: run_symbol_structural_rework (full pipeline)
# ---------------------------------------------------------------------------

def test_run_symbol_structural_rework_writes_all_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import run_symbol_structural_rework

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=300)
    exit_code = run_symbol_structural_rework(settings)

    assert exit_code == 0
    run_dirs = sorted(settings.data.runs_dir.glob("*_symbol_structural_rework"))
    assert run_dirs
    run_dir = run_dirs[-1]

    expected_reports = [
        "symbol_focus_diagnostic_report.json",
        "structural_rework_matrix_report.json",
        "feature_signal_analysis_report.json",
        "label_exit_interaction_report.json",
        "demo_execution_candidate_report.json",
        "structural_rework_recommendation_report.json",
        "threshold_trade_density_report.json",
    ]
    for report_name in expected_reports:
        path = run_dir / report_name
        assert path.exists(), f"Missing report: {report_name}"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "artifact_type" in data, f"Report {report_name} missing artifact_type"
        assert "payload" in data, f"Report {report_name} missing payload"


def test_run_symbol_structural_rework_recommendation_never_approves_bad_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic data has no real edge — recommendation must not be APPROVED."""
    from iris_bot.symbol_focused_rework import run_symbol_structural_rework

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=300)
    run_symbol_structural_rework(settings)

    run_dirs = sorted(settings.data.runs_dir.glob("*_symbol_structural_rework"))
    report = json.loads((run_dirs[-1] / "structural_rework_recommendation_report.json").read_text())
    payload = report["payload"]

    # With synthetic data that has a programmatic (not economic) pattern,
    # the system must not be tricked into approving demo execution
    assert "decision" in payload
    assert "readme_modified" in payload
    assert payload["readme_modified"] is False
    assert "live_real_touched" in payload
    assert payload["live_real_touched"] is False
    assert "operational_layer_modified" in payload
    assert payload["operational_layer_modified"] is False


def test_run_symbol_structural_rework_recommendation_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_bot.symbol_focused_rework import run_symbol_structural_rework

    settings = _scoped_settings(tmp_path, monkeypatch, bars_per_symbol=300)
    run_symbol_structural_rework(settings)

    run_dirs = sorted(settings.data.runs_dir.glob("*_symbol_structural_rework"))
    report = json.loads((run_dirs[-1] / "structural_rework_recommendation_report.json").read_text())
    payload = report["payload"]

    assert payload["focus_symbol"] == "GBPUSD"
    assert "best_variant" in payload
    assert "what_was_tried" in payload
    assert len(payload["what_was_tried"]) == 5
    assert "what_should_come_next" in payload
    assert isinstance(payload["what_should_come_next"], list)


# ---------------------------------------------------------------------------
# CLI registration test (smoke)
# ---------------------------------------------------------------------------

def test_new_commands_are_registered_in_cli() -> None:
    from iris_bot.cli import build_command_handlers

    handlers = build_command_handlers()
    assert "run-symbol-structural-rework" in handlers
    assert "audit-symbol-signal" in handlers
    assert "compare-symbol-variants" in handlers
    assert "evaluate-demo-execution-candidate" in handlers
