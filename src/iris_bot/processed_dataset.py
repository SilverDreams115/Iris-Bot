from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from math import isfinite, log, sqrt
from pathlib import Path

from iris_bot.config import LabelingConfig
from iris_bot.data import Bar, group_bars
from iris_bot.labels import build_label
from iris_bot.sessions import session_flags


FEATURE_NAMES_BASE = [
    "return_1",
    "return_3",
    "return_5",
    "log_return_1",
    "rolling_volatility_5",
    "rolling_volatility_10",
    "atr_5",
    "atr_10",
    "range_ratio",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "distance_to_sma_5",
    "distance_to_sma_10",
    "momentum_3",
    "momentum_5",
    "volume_zscore_5",
    "session_asia",
    "session_london",
    "session_new_york",
]


@dataclass(frozen=True)
class ProcessedRow:
    timestamp: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    label: int
    label_reason: str
    horizon_end_timestamp: str
    features: dict[str, float]


@dataclass(frozen=True)
class ProcessedDataset:
    rows: list[ProcessedRow]
    feature_names: list[str]
    label_mode: str
    schema: dict[str, object]
    manifest: dict[str, object]


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else numerator / denominator


def _rolling_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rolling_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _rolling_mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return sqrt(variance)


def _compute_feature_row(series: list[Bar], index: int) -> dict[str, float]:
    current = series[index]
    close_1 = series[index - 1].close
    close_3 = series[index - 3].close
    close_5 = series[index - 5].close
    last_5 = series[index - 4 : index + 1]
    last_10 = series[index - 9 : index + 1]
    returns_5 = [
        _safe_div(last_5[position].close - last_5[position - 1].close, last_5[position - 1].close)
        for position in range(1, len(last_5))
    ]
    returns_10 = [
        _safe_div(last_10[position].close - last_10[position - 1].close, last_10[position - 1].close)
        for position in range(1, len(last_10))
    ]
    atr_5 = _rolling_mean([bar.high - bar.low for bar in last_5])
    atr_10 = _rolling_mean([bar.high - bar.low for bar in last_10])
    sma_5 = _rolling_mean([bar.close for bar in last_5])
    sma_10 = _rolling_mean([bar.close for bar in last_10])
    volume_window = [bar.volume for bar in last_5]
    volume_mean = _rolling_mean(volume_window)
    volume_std = _rolling_std(volume_window)
    candle_range = current.high - current.low
    body = abs(current.close - current.open)
    upper_wick = current.high - max(current.open, current.close)
    lower_wick = min(current.open, current.close) - current.low
    session_asia, session_london, session_new_york = session_flags(current.timestamp)

    features = {
        "return_1": _safe_div(current.close - close_1, close_1),
        "return_3": _safe_div(current.close - close_3, close_3),
        "return_5": _safe_div(current.close - close_5, close_5),
        "log_return_1": 0.0 if close_1 <= 0.0 or current.close <= 0.0 else log(current.close / close_1),
        "rolling_volatility_5": _rolling_std(returns_5),
        "rolling_volatility_10": _rolling_std(returns_10),
        "atr_5": _safe_div(atr_5, current.close),
        "atr_10": _safe_div(atr_10, current.close),
        "range_ratio": _safe_div(candle_range, current.close),
        "body_ratio": _safe_div(body, candle_range),
        "upper_wick_ratio": _safe_div(upper_wick, candle_range),
        "lower_wick_ratio": _safe_div(lower_wick, candle_range),
        "distance_to_sma_5": _safe_div(current.close - sma_5, sma_5),
        "distance_to_sma_10": _safe_div(current.close - sma_10, sma_10),
        "momentum_3": current.close - close_3,
        "momentum_5": current.close - close_5,
        "volume_zscore_5": 0.0 if volume_std == 0.0 else (current.volume - volume_mean) / volume_std,
        "session_asia": session_asia,
        "session_london": session_london,
        "session_new_york": session_new_york,
    }
    for value in features.values():
        if not isfinite(value):
            raise ValueError("Feature no finita detectada")
    return features


def build_processed_dataset(bars: list[Bar], labeling: LabelingConfig) -> ProcessedDataset:
    grouped = group_bars(bars)
    rows: list[ProcessedRow] = []
    series_summaries: list[dict[str, object]] = []
    warmup = 10

    for (symbol, timeframe), series in sorted(grouped.items()):
        usable_rows = 0
        for index in range(warmup - 1, len(series)):
            label = build_label(series, index, labeling)
            if label is None:
                continue
            features = _compute_feature_row(series, index)
            current = series[index]
            rows.append(
                ProcessedRow(
                    timestamp=current.timestamp,
                    symbol=symbol,
                    timeframe=timeframe,
                    open=current.open,
                    high=current.high,
                    low=current.low,
                    close=current.close,
                    volume=current.volume,
                    label=label.label,
                    label_reason=label.label_reason,
                    horizon_end_timestamp=label.horizon_end_timestamp,
                    features=features,
                )
            )
            usable_rows += 1
        if series:
            series_summaries.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "raw_rows": len(series),
                    "usable_rows": usable_rows,
                    "start_timestamp": series[0].timestamp.isoformat(),
                    "end_timestamp": series[-1].timestamp.isoformat(),
                }
            )

    rows.sort(key=lambda item: (item.timestamp, item.symbol, item.timeframe))
    schema = {
        "raw_columns": ["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume", "spread"],
        "feature_columns": FEATURE_NAMES_BASE,
        "label_columns": ["label", "label_reason", "horizon_end_timestamp"],
        "label_mode": labeling.mode,
    }
    manifest = {
        "row_count": len(rows),
        "feature_count": len(FEATURE_NAMES_BASE),
        "label_mode": labeling.mode,
        "series": series_summaries,
        "symbols": sorted({row.symbol for row in rows}),
        "timeframes": sorted({row.timeframe for row in rows}),
        "start_timestamp": rows[0].timestamp.isoformat() if rows else None,
        "end_timestamp": rows[-1].timestamp.isoformat() if rows else None,
        "class_balance": {
            "-1": sum(1 for row in rows if row.label == -1),
            "0": sum(1 for row in rows if row.label == 0),
            "1": sum(1 for row in rows if row.label == 1),
        },
    }
    return ProcessedDataset(rows=rows, feature_names=FEATURE_NAMES_BASE.copy(), label_mode=labeling.mode, schema=schema, manifest=manifest)


def write_processed_dataset(dataset: ProcessedDataset, dataset_path: Path, manifest_path: Path, schema_path: Path) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "symbol",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
        *dataset.feature_names,
        "label",
        "label_reason",
        "horizon_end_timestamp",
    ]
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset.rows:
            payload = {
                "timestamp": row.timestamp.isoformat(),
                "symbol": row.symbol,
                "timeframe": row.timeframe,
                "open": f"{row.open:.10f}",
                "high": f"{row.high:.10f}",
                "low": f"{row.low:.10f}",
                "close": f"{row.close:.10f}",
                "volume": f"{row.volume:.10f}",
                "label": row.label,
                "label_reason": row.label_reason,
                "horizon_end_timestamp": row.horizon_end_timestamp,
            }
            payload.update({name: f"{row.features[name]:.10f}" for name in dataset.feature_names})
            writer.writerow(payload)

    manifest_path.write_text(json.dumps(dataset.manifest, indent=2, sort_keys=True), encoding="utf-8")
    schema_path.write_text(json.dumps(dataset.schema, indent=2, sort_keys=True), encoding="utf-8")


def load_processed_dataset(dataset_path: Path, schema_path: Path, manifest_path: Path) -> ProcessedDataset:
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset procesado: {dataset_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names = list(schema["feature_columns"])
    rows: list[ProcessedRow] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                ProcessedRow(
                    timestamp=datetime.fromisoformat(raw["timestamp"]),
                    symbol=raw["symbol"],
                    timeframe=raw["timeframe"],
                    open=float(raw["open"]),
                    high=float(raw["high"]),
                    low=float(raw["low"]),
                    close=float(raw["close"]),
                    volume=float(raw["volume"]),
                    label=int(raw["label"]),
                    label_reason=raw["label_reason"],
                    horizon_end_timestamp=raw["horizon_end_timestamp"],
                    features={name: float(raw[name]) for name in feature_names},
                )
            )
    return ProcessedDataset(rows=rows, feature_names=feature_names, label_mode=str(schema["label_mode"]), schema=schema, manifest=manifest)
