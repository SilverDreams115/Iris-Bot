from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from iris_bot.artifacts import wrap_artifact
from iris_bot.backtest import TradeRecord, run_backtest_engine
from iris_bot.config import LabelingConfig, Settings
from iris_bot.data import Bar, load_bars, write_bars
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.mt5 import MT5Client
from iris_bot.processed_dataset import FEATURE_NAMES_BASE, ProcessedRow, build_processed_dataset
from iris_bot.runtime_provenance import load_runtime_provenance_from_env
from iris_bot.sessions import session_flags
from iris_bot.splits import temporal_train_validation_test_split
from iris_bot.thresholds import apply_probability_threshold
from iris_bot.validation import TIMEFRAME_DELTAS, validate_bars
from iris_bot.walk_forward import generate_walk_forward_windows
from iris_bot.xgb_model import XGBoostMultiClassModel


FOCUS_SYMBOL = "GBPUSD"
SECONDARY_SYMBOL = "EURUSD"
OUT_OF_FOCUS_SYMBOLS = ("AUDUSD", "USDJPY")
TARGET_SYMBOLS = (FOCUS_SYMBOL, SECONDARY_SYMBOL)

EXTENDED_HISTORY_DATASET_NAME = "market_extended.csv"
EXTENDED_HISTORY_MULTIPLIER = 4
EXTENDED_HISTORY_MIN_BARS = 6000
REGIME_FEATURE_NAMES = (
    "adx_14",
    "trend_regime_flag",
    "high_volatility_regime_flag",
    "low_volatility_regime_flag",
)
REGIME_STATE_FEATURES = ("atr_regime_percentile",) + REGIME_FEATURE_NAMES
REGIME_ADX_THRESHOLD = 25.0
HIGH_VOLATILITY_THRESHOLD = 0.67
LOW_VOLATILITY_THRESHOLD = 0.33
MIN_TRAIN_ROWS = 30
MIN_VALIDATION_ROWS = 10
MIN_TEST_ROWS = 5
MIN_TRADES_FOR_THRESHOLD = 3
SOFT_WF_PNL_FLOOR = -5.0
MAX_WF_PNL_STDDEV_FOR_CANDIDATE = 35.0


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    hypothesis: str
    dataset_mode: str
    feature_mode: str
    label_tp_pct: float
    label_sl_pct: float
    label_horizon_bars: int


VARIANT_SPECS: tuple[VariantSpec, ...] = (
    VariantSpec(
        variant_id="baseline_symbol_specific_actual",
        hypothesis="Baseline actual del simbolo en muestra actual, sin capa explicita adicional de regimen.",
        dataset_mode="current",
        feature_mode="baseline",
        label_tp_pct=0.0020,
        label_sl_pct=0.0020,
        label_horizon_bars=8,
    ),
    VariantSpec(
        variant_id="baseline_plus_expanded_sample",
        hypothesis="Misma especificacion base, pero con historico ampliado auditado para reducir fragilidad de muestra.",
        dataset_mode="extended",
        feature_mode="baseline",
        label_tp_pct=0.0020,
        label_sl_pct=0.0020,
        label_horizon_bars=8,
    ),
    VariantSpec(
        variant_id="baseline_plus_explicit_regime",
        hypothesis="Misma muestra actual, pero con ADX + estados de volatilidad explicitos para hacer visible el regimen.",
        dataset_mode="current",
        feature_mode="regime",
        label_tp_pct=0.0020,
        label_sl_pct=0.0020,
        label_horizon_bars=8,
    ),
    VariantSpec(
        variant_id="expanded_sample_plus_explicit_regime",
        hypothesis="Rework estructural principal: mas muestra y contexto explicito de regimen sin relajar gates.",
        dataset_mode="extended",
        feature_mode="regime",
        label_tp_pct=0.0020,
        label_sl_pct=0.0020,
        label_horizon_bars=8,
    ),
    VariantSpec(
        variant_id="v3_asymmetric_tp_plus_expanded_regime",
        hypothesis="Mejor variante previa conocida combinada con muestra ampliada y regimen explicito, manteniendo salida conservadora ya conocida.",
        dataset_mode="extended",
        feature_mode="regime",
        label_tp_pct=0.0030,
        label_sl_pct=0.0020,
        label_horizon_bars=8,
    ),
)


def _extended_dataset_path(settings: Settings) -> Path:
    return settings.data.raw_dir / EXTENDED_HISTORY_DATASET_NAME


def _extended_metadata_path(settings: Settings) -> Path:
    return settings.data.raw_dir / f"{EXTENDED_HISTORY_DATASET_NAME}.metadata.json"


def _filter_target_bars(bars: list[Bar]) -> list[Bar]:
    allowed_symbols = set(TARGET_SYMBOLS)
    return [bar for bar in bars if bar.symbol in allowed_symbols]


def _target_history_bars(settings: Settings) -> int:
    return max(settings.mt5.history_bars * EXTENDED_HISTORY_MULTIPLIER, EXTENDED_HISTORY_MIN_BARS)


def _timeframe_delta(timeframe: str) -> timedelta | None:
    return TIMEFRAME_DELTAS.get(timeframe)


def _is_valid_bar(bar: Bar) -> bool:
    return bar.low <= min(bar.open, bar.close) and bar.high >= max(bar.open, bar.close) and bar.low <= bar.high


def _clean_series(series: list[Bar]) -> tuple[list[Bar], dict[str, int]]:
    sorted_series = sorted(series, key=lambda item: item.timestamp)
    cleaned: list[Bar] = []
    duplicate_timestamps = 0
    invalid_ohlc = 0
    seen: set[str] = set()
    for bar in sorted_series:
        ts_key = bar.timestamp.isoformat()
        if ts_key in seen:
            duplicate_timestamps += 1
            continue
        seen.add(ts_key)
        if not _is_valid_bar(bar):
            invalid_ohlc += 1
            continue
        cleaned.append(bar)
    return cleaned, {
        "duplicates_removed": duplicate_timestamps,
        "invalid_ohlc_removed": invalid_ohlc,
    }


def _series_audit(series: list[Bar], symbol: str, timeframe: str) -> dict[str, Any]:
    expected_delta = _timeframe_delta(timeframe)
    gap_count = 0
    max_gap_bars = 0.0
    if expected_delta is not None:
        for previous, current in zip(series[:-1], series[1:], strict=False):
            delta = current.timestamp - previous.timestamp
            if delta > expected_delta:
                gap_count += 1
                max_gap_bars = max(max_gap_bars, delta.total_seconds() / expected_delta.total_seconds())
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": len(series),
        "start_timestamp": series[0].timestamp.isoformat() if series else None,
        "end_timestamp": series[-1].timestamp.isoformat() if series else None,
        "gap_count": gap_count,
        "max_gap_bars": max_gap_bars,
    }


def _baseline_history_snapshot(settings: Settings) -> dict[str, Any]:
    bars = _filter_target_bars(load_bars(settings.data.raw_dataset_path))
    grouped: dict[tuple[str, str], list[Bar]] = defaultdict(list)
    for bar in bars:
        grouped[(bar.symbol, bar.timeframe)].append(bar)
    summaries = [
        _series_audit(sorted(series, key=lambda item: item.timestamp), symbol, timeframe)
        for (symbol, timeframe), series in sorted(grouped.items())
    ]
    return {
        "dataset_path": str(settings.data.raw_dataset_path),
        "total_rows": len(bars),
        "series": summaries,
    }


def _fetch_and_clean_extended_bars(settings: Settings) -> tuple[list[Bar], dict[str, Any]]:
    client = MT5Client(settings.mt5)
    if not client.connect():
        raise RuntimeError("No se pudo conectar a MetaTrader 5 para ampliar la muestra.")
    history_bars = _target_history_bars(settings)
    raw_series: dict[tuple[str, str], list[Bar]] = {}
    try:
        for symbol in TARGET_SYMBOLS:
            for timeframe in settings.trading.timeframes:
                raw_series[(symbol, timeframe)] = client.fetch_historical_bars(symbol, timeframe, history_bars)
    finally:
        client.shutdown()

    cleaned_bars: list[Bar] = []
    audits: list[dict[str, Any]] = []
    for (symbol, timeframe), series in sorted(raw_series.items()):
        cleaned, cleanup = _clean_series(series)
        audits.append(
            {
                **_series_audit(cleaned, symbol, timeframe),
                "bars_requested": history_bars,
                "bars_downloaded": len(series),
                **cleanup,
            }
        )
        cleaned_bars.extend(cleaned)
    return cleaned_bars, {
        "history_bars_requested": history_bars,
        "series": audits,
    }


def run_fetch_extended_history(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "fetch_extended_history")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    logger.info("ampliando muestra solo para symbols=%s", ",".join(TARGET_SYMBOLS))
    try:
        baseline = _baseline_history_snapshot(settings)
        bars, fetched = _fetch_and_clean_extended_bars(settings)
    except Exception as exc:  # pragma: no cover - external MT5 dependency
        logger.error(str(exc))
        return 1

    validation_report = validate_bars(bars)
    write_bars(_extended_dataset_path(settings), bars)
    metadata = {
        "source": "mt5",
        "history_bars_requested": fetched["history_bars_requested"],
        "row_count": len(bars),
        "symbols": list(TARGET_SYMBOLS),
        "timeframes": list(settings.trading.timeframes),
        "run_dir": str(run_dir),
    }
    _extended_metadata_path(settings).write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    report = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "out_of_focus_symbols": list(OUT_OF_FOCUS_SYMBOLS),
        "environment_provenance": load_runtime_provenance_from_env(),
        "baseline_dataset": baseline,
        "extended_dataset": {
            "dataset_path": str(_extended_dataset_path(settings)),
            "metadata_path": str(_extended_metadata_path(settings)),
            "total_rows": len(bars),
            "validation": validation_report.to_dict(),
            "series": fetched["series"],
        },
    }
    write_json_report(run_dir, "expanded_history_report.json", wrap_artifact("expanded_history_report", report))
    logger.info("dataset_extendido=%s rows=%s", _extended_dataset_path(settings), len(bars))
    return 0 if validation_report.is_valid else 2


def _dataset_path_for_mode(settings: Settings, dataset_mode: str) -> Path:
    if dataset_mode == "current":
        return settings.data.raw_dataset_path
    if dataset_mode == "extended":
        return _extended_dataset_path(settings)
    raise ValueError(f"dataset_mode desconocido: {dataset_mode}")


def _feature_names(feature_mode: str) -> list[str]:
    if feature_mode == "baseline":
        return [name for name in FEATURE_NAMES_BASE if name not in REGIME_FEATURE_NAMES]
    if feature_mode == "regime":
        return list(FEATURE_NAMES_BASE)
    raise ValueError(f"feature_mode desconocido: {feature_mode}")


def _build_symbol_rows(settings: Settings, *, dataset_mode: str, symbol: str, label_cfg: LabelingConfig) -> list[ProcessedRow]:
    dataset_path = _dataset_path_for_mode(settings, dataset_mode)
    bars = load_bars(dataset_path)
    if not bars:
        raise FileNotFoundError(f"No hay barras en {dataset_path}")
    dataset = build_processed_dataset([bar for bar in bars if bar.symbol in TARGET_SYMBOLS], label_cfg)
    rows = [row for row in dataset.rows if row.symbol == symbol and row.timeframe == settings.trading.primary_timeframe]
    rows.sort(key=lambda row: row.timestamp)
    return rows


def _economic_weights(rows: list[ProcessedRow], cap: float = 3.0) -> list[float]:
    atrs = [row.features.get("atr_5", 0.0) for row in rows]
    median = sorted(atrs)[len(atrs) // 2] if atrs else 0.0
    if median <= 0.0:
        return [1.0] * len(rows)
    return [min(atr / median, cap) for atr in atrs]


def _select_threshold_economic(
    settings: Settings,
    rows: list[ProcessedRow],
    probabilities: list[dict[int, float]],
) -> tuple[float, dict[str, Any]]:
    best_threshold = settings.threshold.grid[0]
    best_expectancy = float("-inf")
    candidates: list[dict[str, Any]] = []
    for threshold in settings.threshold.grid:
        metrics, _, _ = run_backtest_engine(
            rows=rows,
            probabilities=probabilities,
            threshold=threshold,
            backtest=settings.backtest,
            risk=settings.risk,
            intrabar_policy=settings.backtest.intrabar_policy,
            exit_policy_config=settings.exit_policy,
            dynamic_exit_config=settings.dynamic_exits,
        )
        trades = int(metrics.get("total_trades", 0) or 0)
        expectancy = float(metrics.get("expectancy_usd", 0.0) or 0.0)
        candidates.append(
            {
                "threshold": threshold,
                "trade_count": trades,
                "expectancy_usd": expectancy,
                "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
            }
        )
        if trades < MIN_TRADES_FOR_THRESHOLD:
            continue
        if expectancy > best_expectancy:
            best_expectancy = expectancy
            best_threshold = threshold
    if best_expectancy == float("-inf"):
        best_expectancy = 0.0
    return best_threshold, {
        "metric_name": "economic_expectancy",
        "metric_value": best_expectancy,
        "candidates": candidates,
    }


def _train_model(
    settings: Settings,
    train_rows: list[ProcessedRow],
    validation_rows: list[ProcessedRow],
    feature_names: list[str],
) -> tuple[XGBoostMultiClassModel, float, dict[str, Any]]:
    train_matrix = [[row.features[name] for name in feature_names] for row in train_rows]
    validation_matrix = [[row.features[name] for name in feature_names] for row in validation_rows]
    model = XGBoostMultiClassModel(settings.xgboost)
    model.fit(
        train_matrix,
        [row.label for row in train_rows],
        validation_matrix,
        [row.label for row in validation_rows],
        feature_names=feature_names,
        sample_weights=_economic_weights(train_rows),
    )
    validation_probabilities = model.predict_probabilities(validation_matrix)
    threshold, threshold_report = _select_threshold_economic(settings, validation_rows, validation_probabilities)
    return model, threshold, threshold_report


def _trade_context_index(rows: list[ProcessedRow]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        index[row.timestamp.isoformat()] = {
            "session": _session_name(row),
            "timeframe": row.timeframe,
            "regime": _regime_bucket(row),
        }
    return index


def _trade_metrics_from_subset(trades: list[TradeRecord]) -> dict[str, Any]:
    total_trades = len(trades)
    gross_profit = sum(trade.net_pnl_usd for trade in trades if trade.net_pnl_usd > 0.0)
    gross_loss = -sum(trade.net_pnl_usd for trade in trades if trade.net_pnl_usd < 0.0)
    net_pnl = sum(trade.net_pnl_usd for trade in trades)
    expectancy = net_pnl / total_trades if total_trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0)
    return {
        "trade_count": total_trades,
        "net_pnl_usd": net_pnl,
        "expectancy_usd": expectancy,
        "profit_factor": profit_factor,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
    }


def _evaluate_predictions(
    settings: Settings,
    rows: list[ProcessedRow],
    probabilities: list[dict[int, float]],
    threshold: float,
) -> tuple[dict[str, Any], list[TradeRecord]]:
    metrics, trades, _ = run_backtest_engine(
        rows=rows,
        probabilities=probabilities,
        threshold=threshold,
        backtest=settings.backtest,
        risk=settings.risk,
        intrabar_policy=settings.backtest.intrabar_policy,
        exit_policy_config=settings.exit_policy,
        dynamic_exit_config=settings.dynamic_exits,
    )
    predictions = apply_probability_threshold(probabilities, threshold)
    no_trade_ratio = Counter(predictions).get(0, 0) / len(predictions) if predictions else 0.0
    return {
        "row_count": len(rows),
        "trade_count": int(metrics["total_trades"]),
        "net_pnl_usd": float(metrics["net_pnl_usd"]),
        "expectancy_usd": float(metrics["expectancy_usd"]),
        "profit_factor": float(metrics["profit_factor"]),
        "max_drawdown_usd": float(metrics["max_drawdown_usd"]),
        "no_trade_ratio": no_trade_ratio,
        "threshold": threshold,
    }, trades


def _session_name(row: ProcessedRow) -> str:
    if row.features.get("session_london", 0.0) >= 1.0:
        return "london"
    if row.features.get("session_new_york", 0.0) >= 1.0:
        return "new_york"
    if row.features.get("session_asia", 0.0) >= 1.0:
        return "asia"
    asia, london, new_york = session_flags(row.timestamp)
    if london:
        return "london"
    if new_york:
        return "new_york"
    if asia:
        return "asia"
    return "off_session"


def _regime_bucket(row: ProcessedRow) -> str:
    adx = float(row.features.get("adx_14", 0.0))
    atr_percentile = float(row.features.get("atr_regime_percentile", 0.5))
    trend_state = "trend" if adx >= REGIME_ADX_THRESHOLD else "range"
    if atr_percentile >= HIGH_VOLATILITY_THRESHOLD:
        volatility_state = "high_vol"
    elif atr_percentile <= LOW_VOLATILITY_THRESHOLD:
        volatility_state = "low_vol"
    else:
        volatility_state = "mid_vol"
    return f"{trend_state}_{volatility_state}"


def _breakdown_from_trades(
    rows: list[ProcessedRow],
    probabilities: list[dict[int, float]],
    threshold: float,
    trades: list[TradeRecord],
    key_name: str,
) -> dict[str, Any]:
    predictions = apply_probability_threshold(probabilities, threshold)
    row_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if key_name == "regime":
            key = _regime_bucket(row)
        elif key_name == "session":
            key = _session_name(row)
        elif key_name == "timeframe":
            key = row.timeframe
        else:
            raise ValueError(f"breakdown desconocido: {key_name}")
        row_groups[key].append(index)

    trade_context = _trade_context_index(rows)
    trade_groups: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        context = trade_context.get(trade.signal_timestamp, {})
        group_key = str(context.get(key_name, "unknown"))
        trade_groups[group_key].append(trade)

    breakdown: dict[str, Any] = {}
    for group, indexes in sorted(row_groups.items()):
        row_count = len(indexes)
        no_trade_ratio = sum(1 for idx in indexes if predictions[idx] == 0) / row_count if row_count else 0.0
        trade_subset = trade_groups.get(group, [])
        trade_metrics = _trade_metrics_from_subset(trade_subset)
        breakdown[group] = {
            "row_count": row_count,
            "trade_density": trade_metrics["trade_count"] / row_count if row_count else 0.0,
            "no_trade_ratio": no_trade_ratio,
            **trade_metrics,
        }
    return breakdown


def _walk_forward_report(
    settings: Settings,
    rows: list[ProcessedRow],
    feature_names: list[str],
) -> dict[str, Any]:
    windows = generate_walk_forward_windows(
        total_rows=len(rows),
        train_window=settings.walk_forward.train_window,
        validation_window=settings.walk_forward.validation_window,
        test_window=settings.walk_forward.test_window,
        step=settings.walk_forward.step,
    )
    fold_summaries: list[dict[str, Any]] = []
    for window in windows:
        train_rows = rows[window.train_start : window.train_end]
        validation_rows = rows[window.validation_start : window.validation_end]
        test_rows = rows[window.test_start : window.test_end]
        if len(train_rows) < MIN_TRAIN_ROWS or len(validation_rows) < MIN_VALIDATION_ROWS or len(test_rows) < MIN_TEST_ROWS:
            fold_summaries.append({"fold_index": window.fold_index, "skipped": True, "reason": "insufficient_rows"})
            continue
        model, threshold, threshold_report = _train_model(settings, train_rows, validation_rows, feature_names)
        test_matrix = [[row.features[name] for name in feature_names] for row in test_rows]
        probabilities = model.predict_probabilities(test_matrix)
        metrics, trades = _evaluate_predictions(settings, test_rows, probabilities, threshold)
        fold_summaries.append(
            {
                "fold_index": window.fold_index,
                "skipped": False,
                "threshold": threshold,
                "threshold_report": threshold_report,
                "best_iteration": model.best_iteration,
                "best_score": model.best_score,
                **metrics,
                "session_breakdown": _breakdown_from_trades(test_rows, probabilities, threshold, trades, "session"),
                "timeframe_breakdown": _breakdown_from_trades(test_rows, probabilities, threshold, trades, "timeframe"),
                "regime_breakdown": _breakdown_from_trades(test_rows, probabilities, threshold, trades, "regime"),
            }
        )
    valid = [fold for fold in fold_summaries if not fold.get("skipped")]
    net_pnls = [float(fold["net_pnl_usd"]) for fold in valid]
    profits = [float(fold["profit_factor"]) for fold in valid]
    trades = [int(fold["trade_count"]) for fold in valid]
    no_trade_ratios = [float(fold["no_trade_ratio"]) for fold in valid]
    drawdowns = [float(fold["max_drawdown_usd"]) for fold in valid]
    positive_folds = sum(1 for fold in valid if float(fold["net_pnl_usd"]) > 0.0)
    return {
        "total_folds": len(fold_summaries),
        "valid_folds": len(valid),
        "positive_folds": positive_folds,
        "positive_fold_ratio": positive_folds / len(valid) if valid else 0.0,
        "fold_summaries": fold_summaries,
        "aggregate": {
            "total_net_pnl_usd": sum(net_pnls),
            "mean_profit_factor": mean(profits) if profits else 0.0,
            "mean_no_trade_ratio": mean(no_trade_ratios) if no_trade_ratios else 0.0,
            "worst_fold_drawdown_usd": max(drawdowns) if drawdowns else 0.0,
            "net_pnl_stddev": pstdev(net_pnls) if len(net_pnls) > 1 else 0.0,
            "total_trades": sum(trades),
        },
    }


def _decision(settings: Settings, test_metrics: dict[str, Any], walk_forward: dict[str, Any], regime_delta: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if int(test_metrics["trade_count"]) < settings.strategy.min_validation_trades:
        reasons.append("test_trade_count_below_floor")
    if float(test_metrics["expectancy_usd"]) <= 0.0:
        reasons.append("test_expectancy_non_positive")
    if float(test_metrics["profit_factor"]) < settings.strategy.min_profit_factor:
        reasons.append("test_profit_factor_below_floor")
    if float(test_metrics["max_drawdown_usd"]) > settings.strategy.max_drawdown_usd:
        reasons.append("test_drawdown_above_floor")
    if float(test_metrics["no_trade_ratio"]) > settings.approved_demo_gate.max_no_trade_ratio:
        reasons.append("test_no_trade_ratio_above_floor")

    aggregate = walk_forward["aggregate"]
    if int(walk_forward["valid_folds"]) == 0:
        reasons.append("walk_forward_missing")
    if float(aggregate["total_net_pnl_usd"]) <= 0.0:
        reasons.append("walk_forward_non_positive")
    if float(aggregate["mean_profit_factor"]) < settings.strategy.min_profit_factor:
        reasons.append("walk_forward_profit_factor_below_floor")
    if float(aggregate["worst_fold_drawdown_usd"]) > settings.strategy.max_drawdown_usd:
        reasons.append("walk_forward_drawdown_above_floor")
    if float(aggregate["mean_no_trade_ratio"]) > settings.approved_demo_gate.max_no_trade_ratio:
        reasons.append("walk_forward_no_trade_ratio_above_floor")
    if float(walk_forward["positive_fold_ratio"]) < settings.strategy.min_positive_walkforward_ratio:
        reasons.append("walk_forward_positive_fold_ratio_below_floor")
    if float(aggregate["net_pnl_stddev"]) > MAX_WF_PNL_STDDEV_FOR_CANDIDATE:
        reasons.append("walk_forward_instability_above_floor")
    if regime_delta <= 0.0:
        reasons.append("regime_context_not_helpful")

    if reasons:
        improved_reasons = {
            "walk_forward_non_positive",
            "walk_forward_profit_factor_below_floor",
            "walk_forward_positive_fold_ratio_below_floor",
            "regime_context_not_helpful",
        }
        if set(reasons).issubset(improved_reasons) and float(aggregate["total_net_pnl_usd"]) >= SOFT_WF_PNL_FLOOR:
            return "IMPROVED_BUT_NOT_ENOUGH", reasons
        return "REJECT_FOR_DEMO_EXECUTION", reasons
    return "CANDIDATE_FOR_DEMO_EXECUTION", []


def _dataset_snapshot(settings: Settings, dataset_mode: str, symbol: str, label_cfg: LabelingConfig) -> dict[str, Any]:
    dataset_path = _dataset_path_for_mode(settings, dataset_mode)
    bars = [bar for bar in load_bars(dataset_path) if bar.symbol in TARGET_SYMBOLS]
    validation = validate_bars(bars)
    dataset = build_processed_dataset(bars, label_cfg)
    rows = [row for row in dataset.rows if row.symbol == symbol and row.timeframe == settings.trading.primary_timeframe]
    return {
        "dataset_path": str(dataset_path),
        "raw_rows_total": len(bars),
        "primary_timeframe_rows": len(rows),
        "dataset_validation": validation.to_dict(),
        "dataset_manifest": dataset.manifest,
    }


def _run_variant(settings: Settings, symbol: str, variant: VariantSpec) -> dict[str, Any]:
    label_cfg = replace(
        settings.labeling,
        take_profit_pct=variant.label_tp_pct,
        stop_loss_pct=variant.label_sl_pct,
        horizon_bars=variant.label_horizon_bars,
    )
    rows = _build_symbol_rows(settings, dataset_mode=variant.dataset_mode, symbol=symbol, label_cfg=label_cfg)
    if len(rows) < MIN_TRAIN_ROWS + MIN_VALIDATION_ROWS + MIN_TEST_ROWS:
        raise RuntimeError(f"Filas insuficientes para {symbol} {variant.variant_id}: {len(rows)}")
    split = temporal_train_validation_test_split(
        rows,
        settings.split.train_ratio,
        settings.split.validation_ratio,
        settings.split.test_ratio,
    )
    feature_names = _feature_names(variant.feature_mode)
    model, threshold, threshold_report = _train_model(settings, split.train, split.validation, feature_names)
    test_matrix = [[row.features[name] for name in feature_names] for row in split.test]
    test_probabilities = model.predict_probabilities(test_matrix)
    test_metrics, trades = _evaluate_predictions(settings, split.test, test_probabilities, threshold)
    walk_forward = _walk_forward_report(settings, rows, feature_names)
    regime_breakdown = _breakdown_from_trades(split.test, test_probabilities, threshold, trades, "regime")
    range_expectancy = mean(
        item["expectancy_usd"] for key, item in regime_breakdown.items() if key.startswith("range_") and item["trade_count"] > 0
    ) if any(key.startswith("range_") and item["trade_count"] > 0 for key, item in regime_breakdown.items()) else 0.0
    trend_expectancy = mean(
        item["expectancy_usd"] for key, item in regime_breakdown.items() if key.startswith("trend_") and item["trade_count"] > 0
    ) if any(key.startswith("trend_") and item["trade_count"] > 0 for key, item in regime_breakdown.items()) else 0.0
    decision, reasons = _decision(settings, test_metrics, walk_forward, trend_expectancy - range_expectancy)
    return {
        "symbol": symbol,
        "variant_id": variant.variant_id,
        "hypothesis": variant.hypothesis,
        "dataset_mode": variant.dataset_mode,
        "feature_mode": variant.feature_mode,
        "labeling": {
            "take_profit_pct": variant.label_tp_pct,
            "stop_loss_pct": variant.label_sl_pct,
            "horizon_bars": variant.label_horizon_bars,
        },
        "dataset_snapshot": _dataset_snapshot(settings, variant.dataset_mode, symbol, label_cfg),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "threshold": threshold,
        "threshold_report": threshold_report,
        "best_iteration": model.best_iteration,
        "best_score": model.best_score,
        "test_metrics": test_metrics,
        "walk_forward": walk_forward,
        "performance_by_session": _breakdown_from_trades(split.test, test_probabilities, threshold, trades, "session"),
        "performance_by_timeframe": _breakdown_from_trades(split.test, test_probabilities, threshold, trades, "timeframe"),
        "performance_by_regime": regime_breakdown,
        "decision": decision,
        "decision_reasons": reasons,
        "regime_effect_summary": {
            "trend_expectancy_usd": trend_expectancy,
            "range_expectancy_usd": range_expectancy,
            "trend_minus_range_expectancy_usd": trend_expectancy - range_expectancy,
        },
    }


def _latest_run_dir(settings: Settings, suffix: str) -> Path | None:
    candidates = sorted(settings.data.runs_dir.glob(f"*_{suffix}"))
    return candidates[-1] if candidates else None


def _write_rework_reports(run_dir: Path, payload: dict[str, Any]) -> None:
    report_map = {
        "regime_feature_diagnostic_report.json": payload["regime_feature_diagnostic_report"],
        "regime_aware_experiment_matrix_report.json": payload["regime_aware_experiment_matrix_report"],
        "per_regime_performance_report.json": payload["per_regime_performance_report"],
        "symbol_focus_rework_report.json": payload["symbol_focus_rework_report"],
        "symbol_secondary_comparison_report.json": payload["symbol_secondary_comparison_report"],
        "demo_execution_candidate_report.json": payload["demo_execution_candidate_report"],
        "structural_rework_recommendation_report.json": payload["structural_rework_recommendation_report"],
    }
    for filename, report in report_map.items():
        write_json_report(run_dir, filename, report)


def _regime_feature_diagnostic(settings: Settings) -> dict[str, Any]:
    dataset_path = _extended_dataset_path(settings) if _extended_dataset_path(settings).exists() else settings.data.raw_dataset_path
    bars = load_bars(dataset_path)
    if not bars:
        raise FileNotFoundError(f"No hay barras en {dataset_path}")
    dataset = build_processed_dataset([bar for bar in bars if bar.symbol in TARGET_SYMBOLS], settings.labeling)
    summaries: dict[str, Any] = {}
    for symbol in TARGET_SYMBOLS:
        rows = [row for row in dataset.rows if row.symbol == symbol and row.timeframe == settings.trading.primary_timeframe]
        regime_counts = Counter(_regime_bucket(row) for row in rows)
        adx_values = [float(row.features.get("adx_14", 0.0)) for row in rows]
        atr_percentiles = [float(row.features.get("atr_regime_percentile", 0.5)) for row in rows]
        summaries[symbol] = {
            "row_count": len(rows),
            "adx_14_mean": mean(adx_values) if adx_values else 0.0,
            "adx_14_p95": sorted(adx_values)[int(len(adx_values) * 0.95)] if adx_values else 0.0,
            "atr_regime_percentile_mean": mean(atr_percentiles) if atr_percentiles else 0.0,
            "trend_regime_ratio": sum(1 for row in rows if row.features.get("trend_regime_flag", 0.0) >= 1.0) / len(rows) if rows else 0.0,
            "high_volatility_ratio": sum(1 for row in rows if row.features.get("high_volatility_regime_flag", 0.0) >= 1.0) / len(rows) if rows else 0.0,
            "low_volatility_ratio": sum(1 for row in rows if row.features.get("low_volatility_regime_flag", 0.0) >= 1.0) / len(rows) if rows else 0.0,
            "regime_distribution": dict(sorted(regime_counts.items())),
        }
    payload = {
        "dataset_path": str(dataset_path),
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "environment_provenance": load_runtime_provenance_from_env(),
        "baseline_feature_count": len(_feature_names("baseline")),
        "regime_feature_count": len(_feature_names("regime")),
        "added_regime_features": list(REGIME_FEATURE_NAMES),
        "diagnostics_by_symbol": summaries,
    }
    return wrap_artifact("regime_feature_diagnostic_report", payload)


def run_audit_regime_features(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "audit_regime_features")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        report = _regime_feature_diagnostic(settings)
    except Exception as exc:
        logger.error(str(exc))
        return 1
    write_json_report(run_dir, "regime_feature_diagnostic_report.json", report)
    logger.info("diagnostico_regimen=%s", run_dir / "regime_feature_diagnostic_report.json")
    return 0


def run_regime_aware_rework(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "run_regime_aware_rework")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    if not _extended_dataset_path(settings).exists():
        logger.error("Falta dataset ampliado en %s. Ejecuta fetch-extended-history primero.", _extended_dataset_path(settings))
        return 1

    try:
        regime_feature_report = _regime_feature_diagnostic(settings)
        results_by_symbol = {
            symbol: [_run_variant(settings, symbol, variant) for variant in VARIANT_SPECS]
            for symbol in TARGET_SYMBOLS
        }
    except Exception as exc:
        logger.error(str(exc))
        return 1

    focus_variants = results_by_symbol[FOCUS_SYMBOL]
    secondary_variants = results_by_symbol[SECONDARY_SYMBOL]
    best_focus = max(
        focus_variants,
        key=lambda item: (
            1 if item["decision"] == "CANDIDATE_FOR_DEMO_EXECUTION" else 0,
            1 if item["decision"] == "IMPROVED_BUT_NOT_ENOUGH" else 0,
            float(item["walk_forward"]["aggregate"]["total_net_pnl_usd"]),
            float(item["test_metrics"]["profit_factor"]),
        ),
    )
    best_secondary = max(
        secondary_variants,
        key=lambda item: (
            1 if item["decision"] == "CANDIDATE_FOR_DEMO_EXECUTION" else 0,
            1 if item["decision"] == "IMPROVED_BUT_NOT_ENOUGH" else 0,
            float(item["walk_forward"]["aggregate"]["total_net_pnl_usd"]),
            float(item["test_metrics"]["profit_factor"]),
        ),
    )

    matrix_payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "out_of_focus_symbols": list(OUT_OF_FOCUS_SYMBOLS),
        "environment_provenance": load_runtime_provenance_from_env(),
        "variants": [asdict(variant) for variant in VARIANT_SPECS],
        "results_by_symbol": results_by_symbol,
        "selected_focus_variant": best_focus["variant_id"],
        "selected_secondary_variant": best_secondary["variant_id"],
    }
    per_regime_payload = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "environment_provenance": load_runtime_provenance_from_env(),
        "per_regime_performance": {
            FOCUS_SYMBOL: {item["variant_id"]: item["performance_by_regime"] for item in focus_variants},
            SECONDARY_SYMBOL: {item["variant_id"]: item["performance_by_regime"] for item in secondary_variants},
        },
    }
    focus_report = {
        "symbol": FOCUS_SYMBOL,
        "environment_provenance": load_runtime_provenance_from_env(),
        "decision": best_focus["decision"],
        "selected_variant": best_focus["variant_id"],
        "selected_variant_result": best_focus,
        "recommendation": "Mantener foco en GBPUSD solo si la mejora estructural es reproducible; no promover operativamente en esta fase.",
    }
    secondary_report = {
        "symbol": SECONDARY_SYMBOL,
        "environment_provenance": load_runtime_provenance_from_env(),
        "decision": best_secondary["decision"],
        "selected_variant": best_secondary["variant_id"],
        "selected_variant_result": best_secondary,
        "comparison_vs_focus_total_wf_pnl_usd": float(best_secondary["walk_forward"]["aggregate"]["total_net_pnl_usd"]) - float(best_focus["walk_forward"]["aggregate"]["total_net_pnl_usd"]),
    }
    candidate_report = {
        "environment_provenance": load_runtime_provenance_from_env(),
        "focus_symbol": {
            "symbol": FOCUS_SYMBOL,
            "decision": best_focus["decision"],
            "variant_id": best_focus["variant_id"],
            "has_real_candidate": best_focus["decision"] == "CANDIDATE_FOR_DEMO_EXECUTION",
            "reasons": best_focus["decision_reasons"],
        },
        "secondary_symbol": {
            "symbol": SECONDARY_SYMBOL,
            "decision": best_secondary["decision"],
            "variant_id": best_secondary["variant_id"],
            "has_real_candidate": best_secondary["decision"] == "CANDIDATE_FOR_DEMO_EXECUTION",
            "reasons": best_secondary["decision_reasons"],
        },
        "approved_for_demo_execution_exists": False,
    }
    recommendation_report = {
        "focus_symbol": FOCUS_SYMBOL,
        "secondary_symbol": SECONDARY_SYMBOL,
        "out_of_focus_symbols": list(OUT_OF_FOCUS_SYMBOLS),
        "environment_provenance": load_runtime_provenance_from_env(),
        "structural_recommendation": (
            "Aceptar solo evidencia de candidato cuantitativo; si no mejora walk-forward, PF, estabilidad y regimen a la vez, conservar REJECT_FOR_DEMO_EXECUTION."
        ),
        "next_step": (
            "Si GBPUSD queda en IMPROVED_BUT_NOT_ENOUGH, extender mas historia M15/H1 y revisar si el edge sigue concentrado en un subconjunto de regimenes."
        ),
    }

    wrapped_payload = {
        "regime_feature_diagnostic_report": regime_feature_report,
        "regime_aware_experiment_matrix_report": wrap_artifact("regime_aware_experiment_matrix_report", matrix_payload),
        "per_regime_performance_report": wrap_artifact("per_regime_performance_report", per_regime_payload),
        "symbol_focus_rework_report": wrap_artifact("symbol_focus_rework_report", focus_report),
        "symbol_secondary_comparison_report": wrap_artifact("symbol_secondary_comparison_report", secondary_report),
        "demo_execution_candidate_report": wrap_artifact("demo_execution_candidate_report", candidate_report),
        "structural_rework_recommendation_report": wrap_artifact("structural_rework_recommendation_report", recommendation_report),
    }
    _write_rework_reports(run_dir, wrapped_payload)
    logger.info("run_regime_aware_rework=%s", run_dir)
    return 0


def _load_required_report(path: Path, expected_type: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != expected_type:
        raise ValueError(f"artifact_type inesperado en {path}: {payload.get('artifact_type')}")
    data = payload.get("payload")
    if not isinstance(data, dict):
        raise ValueError(f"payload invalido en {path}")
    return data


def run_compare_regime_experiments(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "compare_regime_experiments")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    source_dir = _latest_run_dir(settings, "run_regime_aware_rework")
    if source_dir is None:
        logger.error("No hay corrida previa de run-regime-aware-rework")
        return 1
    try:
        matrix = _load_required_report(source_dir / "regime_aware_experiment_matrix_report.json", "regime_aware_experiment_matrix_report")
    except Exception as exc:
        logger.error(str(exc))
        return 1
    compact = {
        "source_run_dir": str(source_dir),
        "focus_symbol": matrix["focus_symbol"],
        "secondary_symbol": matrix["secondary_symbol"],
        "selected_focus_variant": matrix["selected_focus_variant"],
        "selected_secondary_variant": matrix["selected_secondary_variant"],
        "focus_variant_scores": {
            item["variant_id"]: {
                "decision": item["decision"],
                "total_wf_net_pnl_usd": item["walk_forward"]["aggregate"]["total_net_pnl_usd"],
                "test_profit_factor": item["test_metrics"]["profit_factor"],
                "trend_minus_range_expectancy_usd": item["regime_effect_summary"]["trend_minus_range_expectancy_usd"],
            }
            for item in matrix["results_by_symbol"][FOCUS_SYMBOL]
        },
        "secondary_variant_scores": {
            item["variant_id"]: {
                "decision": item["decision"],
                "total_wf_net_pnl_usd": item["walk_forward"]["aggregate"]["total_net_pnl_usd"],
                "test_profit_factor": item["test_metrics"]["profit_factor"],
                "trend_minus_range_expectancy_usd": item["regime_effect_summary"]["trend_minus_range_expectancy_usd"],
            }
            for item in matrix["results_by_symbol"][SECONDARY_SYMBOL]
        },
    }
    write_json_report(run_dir, "regime_aware_experiment_matrix_report.json", wrap_artifact("regime_aware_experiment_matrix_report", compact))
    logger.info("comparacion_regimen=%s", run_dir / "regime_aware_experiment_matrix_report.json")
    return 0


def run_evaluate_regime_demo_candidate(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "evaluate_demo_candidate")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    source_dir = _latest_run_dir(settings, "run_regime_aware_rework")
    if source_dir is None:
        logger.error("No hay corrida previa de run-regime-aware-rework")
        return 1
    try:
        focus = _load_required_report(source_dir / "symbol_focus_rework_report.json", "symbol_focus_rework_report")
        secondary = _load_required_report(source_dir / "symbol_secondary_comparison_report.json", "symbol_secondary_comparison_report")
    except Exception as exc:
        logger.error(str(exc))
        return 1
    report = {
        "focus_symbol": {
            "symbol": focus["symbol"],
            "decision": focus["decision"],
            "selected_variant": focus["selected_variant"],
        },
        "secondary_symbol": {
            "symbol": secondary["symbol"],
            "decision": secondary["decision"],
            "selected_variant": secondary["selected_variant"],
        },
        "candidate_exists": focus["decision"] == "CANDIDATE_FOR_DEMO_EXECUTION" or secondary["decision"] == "CANDIDATE_FOR_DEMO_EXECUTION",
        "approved_for_demo_execution_exists": False,
    }
    write_json_report(run_dir, "demo_execution_candidate_report.json", wrap_artifact("demo_execution_candidate_report", report))
    logger.info("demo_candidate=%s", run_dir / "demo_execution_candidate_report.json")
    return 0
