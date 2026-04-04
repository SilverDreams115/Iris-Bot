import dataclasses
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from iris_bot.backtest import run_backtest_engine
from iris_bot.artifacts import read_artifact_payload
from iris_bot.config import DynamicExitConfig, ExitPolicyRuntimeConfig, load_settings
from iris_bot.data import Bar, load_bars, write_bars
from iris_bot.exits import ATRDynamicStopPolicy, ATRDynamicTargetPolicy, SymbolExitProfile
from iris_bot.processed_dataset import ProcessedRow, build_processed_dataset, write_processed_dataset
from iris_bot.symbol_research import build_symbol_profiles_payload, run_symbol_research
from iris_bot.symbol_validation import audit_strategy_block_causes, build_symbol_profiles, run_strategy_validation, symbol_go_no_go
from iris_bot.symbols import load_symbol_strategy_profiles, row_allowed_by_profile, write_symbol_strategy_profiles


pytest.importorskip("xgboost")


def _make_bars(symbol: str, start: datetime, count: int, trend: float, amplitude: float) -> list[Bar]:
    bars: list[Bar] = []
    price = 1.1000 if symbol.endswith("USD") else 150.0
    for index in range(count):
        swing = amplitude if index % 6 in (0, 1, 2) else -amplitude * 0.6
        price += trend + swing
        bars.append(
            Bar(
                timestamp=start + timedelta(minutes=15 * index),
                symbol=symbol,
                timeframe="M15",
                open=price - amplitude * 0.4,
                high=price + amplitude * 1.6,
                low=price - amplitude * 1.4,
                close=price,
                volume=100 + (index % 15),
                spread=8 + (index % 3),
            )
        )
    return bars


def _make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IRIS_PRIMARY_TIMEFRAME", "M15")
    monkeypatch.setenv("IRIS_WF_TRAIN_WINDOW", "40")
    monkeypatch.setenv("IRIS_WF_VALIDATION_WINDOW", "15")
    monkeypatch.setenv("IRIS_WF_TEST_WINDOW", "15")
    monkeypatch.setenv("IRIS_WF_STEP", "15")
    monkeypatch.setenv("IRIS_XGB_NUM_BOOST_ROUND", "8")
    monkeypatch.setenv("IRIS_XGB_EARLY_STOPPING_ROUNDS", "3")
    monkeypatch.setenv("IRIS_STRATEGY_MIN_SYMBOL_ROWS", "60")
    monkeypatch.setenv("IRIS_STRATEGY_MIN_VALIDATION_TRADES", "1")
    monkeypatch.setenv("IRIS_STRATEGY_MIN_PROFIT_FACTOR", "0.0")
    monkeypatch.setenv("IRIS_STRATEGY_MIN_EXPECTANCY_USD", "-1000")
    settings = load_settings()
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    runs_dir = tmp_path / "runs"
    runtime_dir = tmp_path / "data" / "runtime"
    for path in (raw_dir, processed_dir, runs_dir, runtime_dir):
        path.mkdir(parents=True, exist_ok=True)
    object.__setattr__(settings.data, "raw_dir", raw_dir)
    object.__setattr__(settings.data, "processed_dir", processed_dir)
    object.__setattr__(settings.data, "runs_dir", runs_dir)
    object.__setattr__(settings.data, "runtime_dir", runtime_dir)
    object.__setattr__(settings.experiment, "_processed_dir", processed_dir)
    return settings


def _write_dataset(settings, bars: list[Bar]) -> None:
    write_bars(settings.data.raw_dataset_path, bars)
    processed = build_processed_dataset(bars, settings.labeling)
    write_processed_dataset(
        processed,
        settings.experiment.processed_dataset_path,
        settings.experiment.processed_manifest_path,
        settings.experiment.processed_schema_path,
    )


def _sample_processed_row(timestamp: datetime, symbol: str = "EURUSD") -> ProcessedRow:
    return ProcessedRow(
        timestamp=timestamp,
        symbol=symbol,
        timeframe="M15",
        open=1.1000,
        high=1.1025,
        low=1.0980,
        close=1.1010,
        volume=100.0,
        label=1,
        label_reason="test",
        horizon_end_timestamp=(timestamp + timedelta(minutes=60)).isoformat(),
        features={
            "return_1": 0.001,
            "return_3": 0.002,
            "return_5": 0.003,
            "log_return_1": 0.001,
            "rolling_volatility_5": 0.0008,
            "rolling_volatility_10": 0.0012,
            "atr_5": 0.0015,
            "atr_10": 0.0020,
            "range_ratio": 0.002,
            "body_ratio": 0.5,
            "upper_wick_ratio": 0.25,
            "lower_wick_ratio": 0.25,
            "distance_to_sma_5": 0.001,
            "distance_to_sma_10": 0.001,
            "momentum_3": 0.002,
            "momentum_5": 0.003,
            "volume_zscore_5": 0.5,
            "session_asia": 0.0,
            "session_london": 1.0,
            "session_new_york": 0.0,
        },
    )


def test_symbol_research_builds_profiles_from_mt5_like_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    bars = _make_bars("EURUSD", datetime(2026, 1, 1), 120, trend=0.00015, amplitude=0.0004)
    _write_dataset(settings, bars)
    aggregate, reports = build_symbol_profiles_payload(settings)
    assert aggregate["profile_count"] == 1
    assert reports[0]["profile"]["spread"]["method"] == "mt5_rates_spread"
    assert reports[0]["profile"]["valid_bars"] == 120


def test_run_symbol_research_writes_symbol_profile_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    bars = _make_bars("EURUSD", datetime(2026, 1, 1), 120, trend=0.00015, amplitude=0.0004)
    _write_dataset(settings, bars)
    assert run_symbol_research(settings) == 0
    reports = list(settings.data.runs_dir.glob("*_symbol_research/EURUSD/M15/symbol_profile.json"))
    assert reports


def test_dynamic_stop_policy_respects_min_max_bounds() -> None:
    row = _sample_processed_row(datetime(2026, 1, 1, 8, 0, 0))
    level = ATRDynamicStopPolicy().stop_loss_price(
        row=row,
        entry_price=1.1000,
        direction=1,
        backtest=load_settings().backtest,
        risk=load_settings().risk,
        dynamic_config=DynamicExitConfig(min_stop_loss_pct=0.0010, max_stop_loss_pct=0.0012),
        symbol_profile=SymbolExitProfile(stop_policy="atr_dynamic", stop_atr_multiplier=4.0),
    )
    assert 1.1000 * 0.0010 <= level.distance <= 1.1000 * 0.0012


def test_dynamic_target_policy_respects_min_max_bounds() -> None:
    row = _sample_processed_row(datetime(2026, 1, 1, 8, 0, 0))
    level = ATRDynamicTargetPolicy().take_profit_price(
        row=row,
        entry_price=1.1000,
        direction=1,
        backtest=load_settings().backtest,
        risk=load_settings().risk,
        dynamic_config=DynamicExitConfig(min_take_profit_pct=0.0015, max_take_profit_pct=0.0018),
        symbol_profile=SymbolExitProfile(target_policy="atr_dynamic", target_atr_multiplier=6.0),
    )
    assert 1.1000 * 0.0015 <= level.distance <= 1.1000 * 0.0018


def test_write_and_load_strategy_profiles_support_symbol_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(
        settings,
        common_profile={"threshold": 0.60, "allowed_sessions": ["london"]},
        symbol_profiles={"EURUSD": {"threshold": 0.70, "enabled_state": "caution", "enabled": False}},
    )
    profiles = load_symbol_strategy_profiles(settings)
    assert profiles["EURUSD"].threshold == 0.70
    assert profiles["EURUSD"].enabled_state == "caution"


def test_row_allowed_by_profile_blocks_disallowed_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    profile = load_symbol_strategy_profiles(settings)["EURUSD"]
    custom = dataclasses.replace(profile, allowed_sessions=("asia",))
    assert row_allowed_by_profile(custom, datetime(2026, 1, 1, 10, 0, 0), "M15") is False


def test_backtest_engine_records_dynamic_exit_trace() -> None:
    start = datetime(2026, 1, 1, 8, 0, 0)
    rows = [_sample_processed_row(start + timedelta(minutes=15 * index)) for index in range(5)]
    probabilities = [
        {1: 0.90, 0: 0.05, -1: 0.05},
        {1: 0.90, 0: 0.05, -1: 0.05},
        {1: 0.05, 0: 0.05, -1: 0.90},
        {1: 0.05, 0: 0.05, -1: 0.90},
        {1: 0.05, 0: 0.90, -1: 0.05},
    ]
    metrics, trades, _ = run_backtest_engine(
        rows=rows,
        probabilities=probabilities,
        threshold=0.60,
        backtest=load_settings().backtest,
        risk=load_settings().risk,
        exit_policy_config=ExitPolicyRuntimeConfig(stop_policy="atr_dynamic", target_policy="atr_dynamic"),
        dynamic_exit_config=load_settings().dynamic_exits,
        symbol_exit_profiles={"EURUSD": SymbolExitProfile(stop_policy="atr_dynamic", target_policy="atr_dynamic")},
    )
    assert metrics["total_trades"] >= 1
    assert trades[0].stop_policy == "atr_dynamic"
    assert trades[0].stop_policy_details is not None


def test_spread_roundtrip_in_raw_dataset(tmp_path: Path) -> None:
    bars = _make_bars("EURUSD", datetime(2026, 1, 1), 5, trend=0.0001, amplitude=0.0003)
    path = tmp_path / "market.csv"
    write_bars(path, bars)
    loaded = load_bars(path)
    assert loaded[0].spread == bars[0].spread


def test_run_strategy_validation_writes_reports_and_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    bars = _make_bars("EURUSD", datetime(2026, 1, 1), 140, trend=0.00015, amplitude=0.0004)
    bars += _make_bars("GBPUSD", datetime(2026, 1, 1), 140, trend=0.00002, amplitude=0.0002)
    _write_dataset(settings, bars)
    assert run_strategy_validation(settings) == 0
    assert list(settings.data.runs_dir.glob("*_strategy_validation/strategy_validation_report.json"))
    assert (settings.data.runtime_dir / settings.strategy.profiles_filename).exists()


def test_build_symbol_profiles_emits_runtime_profile_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {"threshold": 0.60}, {"EURUSD": {"threshold": 0.65}})
    assert build_symbol_profiles(settings) == 0
    assert list(settings.data.runs_dir.glob("*_build_symbol_profiles/strategy_profiles.json"))


def test_symbol_go_no_go_uses_latest_validation_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    bars = _make_bars("EURUSD", datetime(2026, 1, 1), 140, trend=0.00015, amplitude=0.0004)
    _write_dataset(settings, bars)
    assert run_strategy_validation(settings) == 0
    assert symbol_go_no_go(settings) == 0
    assert list(settings.data.runs_dir.glob("*_symbol_go_no_go/symbol_enablement_report.json"))


def test_strategy_validation_persists_enablement_decisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    bars = _make_bars("EURUSD", datetime(2026, 1, 1), 140, trend=0.00015, amplitude=0.0004)
    bars += _make_bars("GBPUSD", datetime(2026, 1, 1), 140, trend=-0.00010, amplitude=0.00045)
    _write_dataset(settings, bars)
    assert run_strategy_validation(settings) == 0
    payload = read_artifact_payload(settings.data.runtime_dir / settings.strategy.profiles_filename, expected_type="strategy_profiles")
    assert "symbols" in payload


def test_strategy_validation_generates_threshold_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    bars = _make_bars("EURUSD", datetime(2026, 1, 1), 140, trend=0.00015, amplitude=0.0004)
    _write_dataset(settings, bars)
    assert run_strategy_validation(settings) == 0
    report = list(settings.data.runs_dir.glob("*_strategy_validation/threshold_report.json"))[0]
    payload = read_artifact_payload(report, expected_type="threshold_report")
    assert "EURUSD" in payload["symbols"]


def test_strategy_validation_generates_dynamic_exit_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    bars = _make_bars("EURUSD", datetime(2026, 1, 1), 140, trend=0.00015, amplitude=0.0004)
    _write_dataset(settings, bars)
    assert run_strategy_validation(settings) == 0
    report = list(settings.data.runs_dir.glob("*_strategy_validation/dynamic_exit_report.json"))[0]
    payload = read_artifact_payload(report, expected_type="dynamic_exit_report")
    assert payload["symbols"]["EURUSD"]["preferred_exit_policy"] in {"static", "atr_dynamic"}


def test_strategy_validation_generates_model_comparison_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    bars = _make_bars("EURUSD", datetime(2026, 1, 1), 140, trend=0.00015, amplitude=0.0004)
    _write_dataset(settings, bars)
    assert run_strategy_validation(settings) == 0
    report = list(settings.data.runs_dir.glob("*_strategy_validation/model_comparison_report.json"))[0]
    payload = read_artifact_payload(report, expected_type="model_comparison")
    assert payload["symbols"]["EURUSD"]["chosen_model"] in {"global_model", "symbol_model"}


def test_strategy_validation_blocks_symbol_when_data_is_insufficient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    object.__setattr__(settings, "strategy", dataclasses.replace(settings.strategy, min_symbol_rows=200))
    bars = _make_bars("EURUSD", datetime(2026, 1, 1), 90, trend=0.00015, amplitude=0.0004)
    _write_dataset(settings, bars)
    assert run_strategy_validation(settings) == 0
    report = list(settings.data.runs_dir.glob("*_strategy_validation/symbol_enablement_report.json"))[0]
    payload = read_artifact_payload(report, expected_type="symbol_enablement")
    assert payload["symbols"]["EURUSD"]["state"] == "disabled"


def test_strategy_validation_writes_block_diagnostic_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    bars = _make_bars("EURUSD", datetime(2026, 1, 1), 140, trend=0.00015, amplitude=0.0004)
    _write_dataset(settings, bars)

    assert audit_strategy_block_causes(settings) == 0

    gate_report = list(settings.data.runs_dir.glob("*_strategy_validation/gate_failure_matrix.json"))[0]
    threshold_report = list(settings.data.runs_dir.glob("*_strategy_validation/threshold_sensitivity_report.json"))[0]
    recommendation_report = list(settings.data.runs_dir.glob("*_strategy_validation/symbol_recommendation_report.json"))[0]

    gate_payload = read_artifact_payload(gate_report, expected_type="strategy_validation_audit")
    threshold_payload = read_artifact_payload(threshold_report, expected_type="strategy_validation_audit")
    recommendation_payload = read_artifact_payload(recommendation_report, expected_type="strategy_validation_audit")

    assert "profit_factor_below_floor" in gate_payload["symbols"]["EURUSD"]["reasons"]
    assert "static" in threshold_payload["symbols"]["EURUSD"]["threshold_grid"]
    assert recommendation_payload["symbols"]["EURUSD"]["recommendation"] in {
        "KEEP_BLOCKED",
        "MOVE_TO_CAUTION",
        "KEEP_VALIDATED",
        "CANDIDATE_FOR_APPROVED_DEMO",
    }
