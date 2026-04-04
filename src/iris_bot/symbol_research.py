from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from typing import Any

from iris_bot.baselines import MomentumSignBaseline
from iris_bot.backtest import run_backtest_engine
from iris_bot.config import Settings
from iris_bot.data import Bar, group_bars, load_bars
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.processed_dataset import ProcessedDataset, ProcessedRow, load_processed_dataset
from iris_bot.sessions import canonical_session_name, session_definition_report


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _build_symbol_profile(
    bars: list[Bar],
    processed_rows: list[ProcessedRow],
    settings: Settings,
) -> dict[str, Any]:
    close_returns = [
        (bars[index].close - bars[index - 1].close) / bars[index - 1].close
        for index in range(1, len(bars))
        if bars[index - 1].close != 0.0
    ]
    candle_ranges = [bar.high - bar.low for bar in bars]
    atr_like = [
        (bar.high - bar.low) / bar.close
        for bar in bars
        if bar.close != 0.0
    ]
    spread_values = [bar.spread for bar in bars if bar.spread > 0.0]
    if spread_values:
        spread_method = "mt5_rates_spread"
        average_spread = _mean(spread_values)
    else:
        spread_method = "backtest_config_proxy"
        average_spread = settings.backtest.spread_pips

    session_buckets: dict[str, list[float]] = defaultdict(list)
    for bar, ret in zip(bars[1:], close_returns, strict=False):
        session_buckets[canonical_session_name(bar.timestamp)].append(ret)

    movement_efficiency = [
        abs(bar.close - bar.open) / (bar.high - bar.low)
        for bar in bars
        if (bar.high - bar.low) > 0.0
    ]
    noise_ratio = 1.0 - _mean(movement_efficiency) if movement_efficiency else 1.0

    setup_frequency = (
        0.0
        if not processed_rows
        else sum(1 for row in processed_rows if row.label != 0) / len(processed_rows)
    )

    baseline_metrics: dict[str, Any]
    if len(processed_rows) >= 20:
        feature_dicts = [row.features for row in processed_rows]
        scores = MomentumSignBaseline().score(feature_dicts)
        probabilities = []
        for score in scores:
            if score > 0.0:
                probabilities.append({-1: 0.05, 0: 0.15, 1: 0.80})
            elif score < 0.0:
                probabilities.append({-1: 0.80, 0: 0.15, 1: 0.05})
            else:
                probabilities.append({-1: 0.10, 0: 0.80, 1: 0.10})
        baseline_metrics, _, _ = run_backtest_engine(
            rows=processed_rows,
            probabilities=probabilities,
            threshold=max(settings.threshold.grid[0], 0.45),
            backtest=settings.backtest,
            risk=settings.risk,
            intrabar_policy=settings.backtest.intrabar_policy,
        )
    else:
        baseline_metrics = {
            "total_trades": 0,
            "expectancy_usd": 0.0,
            "max_drawdown_usd": 0.0,
            "net_pnl_usd": 0.0,
        }

    aptitude = "suitable"
    reasons: list[str] = []
    if len(bars) < settings.strategy.min_symbol_rows:
        aptitude = "insufficient_data"
        reasons.append("insufficient_bars")
    if setup_frequency <= 0.05:
        aptitude = "low_setup_density"
        reasons.append("low_setup_density")
    if baseline_metrics.get("expectancy_usd", 0.0) <= 0.0:
        reasons.append("baseline_expectancy_non_positive")

    return {
        "range": {
            "start_timestamp": bars[0].timestamp.isoformat() if bars else None,
            "end_timestamp": bars[-1].timestamp.isoformat() if bars else None,
        },
        "valid_bars": len(bars),
        "valid_processed_rows": len(processed_rows),
        "spread": {
            "average": average_spread,
            "method": spread_method,
            "samples": len(spread_values),
        },
        "volatility": {
            "mean_abs_return": _mean([abs(value) for value in close_returns]),
            "std_return": _std(close_returns),
            "atr_typical": _mean(atr_like),
        },
        "sessions": {
            session: {
                "bars": len(values),
                "mean_return": _mean(values),
                "std_return": _std(values),
            }
            for session, values in sorted(session_buckets.items())
        },
        "returns_distribution": {
            "count": len(close_returns),
            "mean": _mean(close_returns),
            "std": _std(close_returns),
            "p05": _percentile(close_returns, 0.05),
            "p50": _percentile(close_returns, 0.50),
            "p95": _percentile(close_returns, 0.95),
            "positive_ratio": 0.0 if not close_returns else sum(1 for value in close_returns if value > 0.0) / len(close_returns),
        },
        "noise_vs_movement": {
            "movement_efficiency": _mean(movement_efficiency),
            "noise_ratio": noise_ratio,
        },
        "setup_frequency": {
            "valid_setup_ratio": setup_frequency,
            "label_distribution": {
                "-1": sum(1 for row in processed_rows if row.label == -1),
                "0": sum(1 for row in processed_rows if row.label == 0),
                "1": sum(1 for row in processed_rows if row.label == 1),
            },
        },
        "baseline_momentum_backtest": baseline_metrics,
        "aptitude": aptitude,
        "aptitude_reasons": reasons,
    }


def build_symbol_profiles_payload(settings: Settings) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bars = load_bars(settings.data.raw_dataset_path)
    if not bars:
        raise FileNotFoundError(f"No hay dataset crudo en {settings.data.raw_dataset_path}")
    dataset = load_processed_dataset(
        settings.experiment.processed_dataset_path,
        settings.experiment.processed_schema_path,
        settings.experiment.processed_manifest_path,
    )
    grouped_bars = group_bars(bars)
    processed_by_key: dict[tuple[str, str], list[ProcessedRow]] = defaultdict(list)
    for row in dataset.rows:
        processed_by_key[(row.symbol, row.timeframe)].append(row)

    symbol_reports: list[dict[str, Any]] = []
    for (symbol, timeframe), series in sorted(grouped_bars.items()):
        processed_rows = processed_by_key.get((symbol, timeframe), [])
        symbol_reports.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "mt5_raw_dataset",
                "profile": _build_symbol_profile(series, processed_rows, settings),
            }
        )
    aggregate = {
        "symbols": sorted({item["symbol"] for item in symbol_reports}),
        "timeframes": sorted({item["timeframe"] for item in symbol_reports}),
        "profile_count": len(symbol_reports),
        "session_definition": session_definition_report(),
        "aptitude_summary": {
            "suitable": sum(1 for item in symbol_reports if item["profile"]["aptitude"] == "suitable"),
            "insufficient_data": sum(1 for item in symbol_reports if item["profile"]["aptitude"] == "insufficient_data"),
            "low_setup_density": sum(1 for item in symbol_reports if item["profile"]["aptitude"] == "low_setup_density"),
        },
        "raw_dataset_path": str(settings.data.raw_dataset_path),
        "processed_dataset_path": str(settings.experiment.processed_dataset_path),
    }
    return aggregate, symbol_reports


def run_symbol_research(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "symbol_research")
    logger = configure_logging(run_dir, settings.logging.level)
    try:
        aggregate, symbol_reports = build_symbol_profiles_payload(settings)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    for item in symbol_reports:
        symbol_dir = run_dir / item["symbol"] / item["timeframe"]
        symbol_dir.mkdir(parents=True, exist_ok=True)
        write_json_report(symbol_dir, "symbol_profile.json", item["profile"])

    write_json_report(run_dir, "symbol_research_report.json", {"aggregate": aggregate, "profiles": symbol_reports})
    logger.info("symbol_research profiles=%s run_dir=%s", len(symbol_reports), run_dir)
    return 0
