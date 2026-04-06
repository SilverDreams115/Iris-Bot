from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, log, sqrt
from pathlib import Path
from typing import cast

from iris_bot.config import LabelingConfig
from iris_bot.data import Bar, group_bars
from iris_bot.labels import build_label
from iris_bot.sessions import session_flags


# ---------------------------------------------------------------------------
# Feature registry — single source of truth for feature names and ordering.
#
# Changes here affect the CSV schema, all model training, and all inference.
# Update tests after any addition or rename.
# ---------------------------------------------------------------------------
FEATURE_NAMES_BASE = [
    # --- Returns & momentum (all normalized as fractional returns, cross-pair comparable) ---
    "return_1",                    # (close[t] - close[t-1]) / close[t-1]
    "return_3",                    # (close[t] - close[t-3]) / close[t-3]
    "return_5",                    # (close[t] - close[t-5]) / close[t-5]
    "log_return_1",                # log(close[t] / close[t-1])
    "momentum_3",                  # (close[t] - close[t-3]) / close[t-3]  ← normalized (was absolute pips)
    "momentum_5",                  # (close[t] - close[t-5]) / close[t-5]  ← normalized (was absolute pips)
    # --- Volatility ---
    "rolling_volatility_5",        # std-dev of last 5 one-bar returns
    "rolling_volatility_10",       # std-dev of last 10 one-bar returns
    "atr_5",                       # mean(high-low) over 5 bars, normalized by close
    "atr_10",                      # mean(high-low) over 10 bars, normalized by close
    "parkinson_volatility_10",     # Parkinson estimator (uses high/low) over 10 bars
    # --- Candle structure ---
    "range_ratio",                 # (high - low) / close
    "body_ratio",                  # |close - open| / (high - low)
    "upper_wick_ratio",            # upper wick / (high - low)
    "lower_wick_ratio",            # lower wick / (high - low)
    # --- Trend & efficiency ---
    "distance_to_sma_5",           # (close - sma5) / sma5
    "distance_to_sma_10",          # (close - sma10) / sma10
    "efficiency_ratio_10",         # Kaufman ER: net_move / sum_of_moves, 10 bars
    "efficiency_ratio_50",         # Kaufman ER over 50 bars — regime: trending vs choppy
    # --- Mean-reversion & autocorrelation ---
    "return_autocorr_10",          # lag-1 autocorr of 10 one-bar returns
    "return_autocorr_3",           # lag-3 autocorr of 10 one-bar returns
    "return_autocorr_5",           # lag-5 autocorr of 10 one-bar returns
    "variance_ratio_hurst_proxy",  # VR(5) over 50 bars: >1 trending, <1 mean-reverting
    # --- Volume ---
    "volume_zscore_20",            # (vol - mean_20) / std_20  ← window 20 (was 5: too noisy)
    "volume_percentile_20",        # rank of current volume within last 20 bars
    # --- Regime ---
    "atr_regime_percentile",       # current ATR_10 as percentile of 50-bar ATR_10 history
    # --- Sessions ---
    "session_asia",
    "session_london",
    "session_new_york",
    # --- Cross-symbol (CurrencyStrengthMatrix) ---
    # All three default to 0.0 / 0.5 when only one symbol is in the dataset.
    "cross_momentum_agreement",    # fraction of other pairs with same return_1 sign
    "usd_strength_index",          # aggregate USD direction from non-current pairs
    "currency_strength_rank",      # return_1 rank among all pairs: 0=weakest, 1=strongest
]

# Approximate DXY composition weights per pair (sign encodes USD direction).
# EURUSD/GBPUSD/AUDUSD up → USD weak (negative contribution).
# USDJPY up → USD strong (positive contribution).
_USD_DIRECTION_WEIGHTS: dict[str, float] = {
    "EURUSD": -1.0,
    "GBPUSD": -1.0,
    "AUDUSD": -1.0,
    "USDJPY": +1.0,
}


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


# ---------------------------------------------------------------------------
# Pure numeric helpers
# ---------------------------------------------------------------------------

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


def _autocorrelation_lag_n(values: list[float], lag: int) -> float:
    """Pearson autocorrelation at the specified lag."""
    if len(values) < lag + 1:
        return 0.0
    current = values[lag:]
    previous = values[:-lag]
    current_mean = _rolling_mean(current)
    previous_mean = _rolling_mean(previous)
    numerator = sum(
        (a - current_mean) * (b - previous_mean)
        for a, b in zip(current, previous, strict=False)
    )
    denominator = len(current) * _rolling_std(current) * _rolling_std(previous)
    return 0.0 if denominator == 0.0 else numerator / denominator


def _volume_percentile(window: list[float], current_value: float) -> float:
    if not window:
        return 0.0
    return sum(1 for v in window if v <= current_value) / len(window)


def _parkinson_volatility(window: list[Bar]) -> float:
    if not window:
        return 0.0
    normalizer = 4.0 * log(2.0)
    estimators = [
        (log(bar.high / bar.low) ** 2) / normalizer
        for bar in window
        if bar.high > 0.0 and bar.low > 0.0 and bar.high >= bar.low
    ]
    return sqrt(_rolling_mean(estimators)) if estimators else 0.0


def _efficiency_ratio(bars: list[Bar]) -> float:
    """Kaufman Efficiency Ratio: net directional move / sum of all bar-to-bar moves."""
    if len(bars) < 2:
        return 0.0
    net_move = abs(bars[-1].close - bars[0].close)
    total_path = sum(
        abs(bars[i].close - bars[i - 1].close) for i in range(1, len(bars))
    )
    return _safe_div(net_move, total_path)


def _variance_ratio_hurst_proxy(bars: list[Bar], lag: int = 5) -> float:
    """
    Hurst exponent proxy via Lo-MacKinlay variance ratio at lag k.

    VR(k) = Var(k-period log-returns) / (k * Var(1-period log-returns))

    Interpretation:
      VR > 1  → trending (positive autocorrelation, Hurst > 0.5)
      VR < 1  → mean-reverting (negative autocorrelation, Hurst < 0.5)
      VR ≈ 1  → random walk (Hurst ≈ 0.5)
    """
    if len(bars) < lag + 1:
        return 1.0  # Not enough data: default to random-walk
    closes = [bar.close for bar in bars]
    r1 = [
        _safe_div(closes[i] - closes[i - 1], closes[i - 1])
        for i in range(1, len(closes))
    ]
    r_k = [
        _safe_div(closes[i] - closes[i - lag], closes[i - lag])
        for i in range(lag, len(closes))
    ]
    var_r1 = _rolling_std(r1) ** 2
    var_rk = _rolling_std(r_k) ** 2
    return _safe_div(var_rk, lag * var_r1)


def _atr_regime_percentile(bars: list[Bar]) -> float:
    """
    Percentile of the most recent ATR_10 within the rolling ATR_10 distribution.

    Returns:
      0.0 = current period has the lowest volatility in the lookback window
      1.0 = current period has the highest volatility
    """
    if len(bars) < 10:
        return 0.5  # Insufficient history: neutral
    atrs: list[float] = []
    for i in range(9, len(bars)):
        window = bars[i - 9 : i + 1]
        atrs.append(_rolling_mean([b.high - b.low for b in window]))
    return _volume_percentile(atrs, atrs[-1])


def _cross_symbol_features(
    symbol: str,
    current_return_1: float,
    cross_returns: dict[str, float],
) -> dict[str, float]:
    """
    Compute CurrencyStrengthMatrix features from other pairs at the same timestamp.

    All inputs are return_1 values (fractional, no look-ahead).
    Falls back to neutral defaults when no cross-pair data is available.
    """
    others = {sym: ret for sym, ret in cross_returns.items() if sym != symbol}
    if not others:
        return {
            "cross_momentum_agreement": 0.0,
            "usd_strength_index": 0.0,
            "currency_strength_rank": 0.5,
        }

    # Fraction of other pairs with the same directional sign as the current pair.
    current_sign = 1 if current_return_1 > 1e-10 else (-1 if current_return_1 < -1e-10 else 0)
    if current_sign != 0:
        agrees = sum(
            1 for r in others.values()
            if (1 if r > 1e-10 else (-1 if r < -1e-10 else 0)) == current_sign
        )
        cross_momentum_agreement = agrees / len(others)
    else:
        cross_momentum_agreement = 0.0

    # USD strength index: aggregate USD direction excluding the current pair.
    # Uses approximate DXY composition weights.
    usd_contributions = [
        _USD_DIRECTION_WEIGHTS[sym] * ret
        for sym, ret in others.items()
        if sym in _USD_DIRECTION_WEIGHTS
    ]
    usd_strength_index = _rolling_mean(usd_contributions) if usd_contributions else 0.0

    # Currency strength rank: normalized rank of this pair's return among all pairs.
    all_returns = sorted(list(others.values()) + [current_return_1])
    n = len(all_returns)
    # Find rank of current_return_1 (use midpoint when tied).
    indices = [i for i, v in enumerate(all_returns) if v == current_return_1]
    rank_normalized = _rolling_mean([i / max(n - 1, 1) for i in indices])

    return {
        "cross_momentum_agreement": cross_momentum_agreement,
        "usd_strength_index": usd_strength_index,
        "currency_strength_rank": rank_normalized,
    }


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_feature_row(
    series: list[Bar],
    index: int,
    cross_returns: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute all features for series[index].

    All features use only bars at index or earlier — zero look-ahead.

    Parameters
    ----------
    series:        Time-ordered bars for a single (symbol, timeframe).
    index:         Current bar index (must be >= warmup - 1).
    cross_returns: Optional map of {other_symbol: return_1} at the same timestamp,
                   used for CurrencyStrengthMatrix features.
    """
    current = series[index]
    close_1 = series[index - 1].close
    close_3 = series[index - 3].close
    close_5 = series[index - 5].close
    close_10 = series[index - 10].close

    last_5  = series[index - 4  : index + 1]
    last_10 = series[index - 9  : index + 1]
    last_11 = series[index - 10 : index + 1]
    last_20 = series[index - 19 : index + 1]
    # Regime windows: use max available up to 51 bars (gracefully degrade when shorter).
    last_51 = series[max(0, index - 50) : index + 1]
    last_50_regime = series[max(0, index - 49) : index + 1]  # for ATR percentile

    returns_5 = [
        _safe_div(last_5[i].close - last_5[i - 1].close, last_5[i - 1].close)
        for i in range(1, len(last_5))
    ]
    returns_10 = [
        _safe_div(last_10[i].close - last_10[i - 1].close, last_10[i - 1].close)
        for i in range(1, len(last_10))
    ]
    returns_11 = [
        _safe_div(last_11[i].close - last_11[i - 1].close, last_11[i - 1].close)
        for i in range(1, len(last_11))
    ]

    atr_5  = _rolling_mean([b.high - b.low for b in last_5])
    atr_10 = _rolling_mean([b.high - b.low for b in last_10])
    sma_5  = _rolling_mean([b.close for b in last_5])
    sma_10 = _rolling_mean([b.close for b in last_10])

    volume_window_20 = [b.volume for b in last_20]
    volume_mean_20   = _rolling_mean(volume_window_20)
    volume_std_20    = _rolling_std(volume_window_20)

    candle_range = current.high - current.low
    body        = abs(current.close - current.open)
    upper_wick  = current.high - max(current.open, current.close)
    lower_wick  = min(current.open, current.close) - current.low
    session_asia, session_london, session_new_york = session_flags(current.timestamp)

    return_1 = _safe_div(current.close - close_1, close_1)

    features: dict[str, float] = {
        # Returns & momentum
        "return_1":   return_1,
        "return_3":   _safe_div(current.close - close_3, close_3),
        "return_5":   _safe_div(current.close - close_5, close_5),
        "log_return_1": 0.0 if close_1 <= 0.0 or current.close <= 0.0 else log(current.close / close_1),
        "momentum_3": _safe_div(current.close - close_3, close_3),  # normalized (was absolute pips)
        "momentum_5": _safe_div(current.close - close_5, close_5),  # normalized (was absolute pips)
        # Volatility
        "rolling_volatility_5":    _rolling_std(returns_5),
        "rolling_volatility_10":   _rolling_std(returns_10),
        "atr_5":                   _safe_div(atr_5, current.close),
        "atr_10":                  _safe_div(atr_10, current.close),
        "parkinson_volatility_10": _parkinson_volatility(last_10),
        # Candle structure
        "range_ratio":       _safe_div(candle_range, current.close),
        "body_ratio":        _safe_div(body, candle_range),
        "upper_wick_ratio":  _safe_div(upper_wick, candle_range),
        "lower_wick_ratio":  _safe_div(lower_wick, candle_range),
        # Trend & efficiency
        "distance_to_sma_5":   _safe_div(current.close - sma_5, sma_5),
        "distance_to_sma_10":  _safe_div(current.close - sma_10, sma_10),
        "efficiency_ratio_10": _efficiency_ratio(last_11),
        "efficiency_ratio_50": _efficiency_ratio(last_51),
        # Autocorrelation
        "return_autocorr_10": _autocorrelation_lag_n(returns_11, lag=1),
        "return_autocorr_3":  _autocorrelation_lag_n(returns_11, lag=3),
        "return_autocorr_5":  _autocorrelation_lag_n(returns_11, lag=5),
        # Hurst proxy
        "variance_ratio_hurst_proxy": _variance_ratio_hurst_proxy(last_51, lag=5),
        # Volume
        "volume_zscore_20":    0.0 if volume_std_20 == 0.0 else (current.volume - volume_mean_20) / volume_std_20,
        "volume_percentile_20": _volume_percentile(volume_window_20, current.volume),
        # Regime
        "atr_regime_percentile": _atr_regime_percentile(last_50_regime),
        # Sessions
        "session_asia":     session_asia,
        "session_london":   session_london,
        "session_new_york": session_new_york,
        # Cross-symbol (filled below)
        **_cross_symbol_features(
            current.symbol,
            return_1,
            cross_returns if cross_returns is not None else {},
        ),
    }

    for name, value in features.items():
        if not isfinite(value):
            raise ValueError(f"Non-finite feature detected: {name}={value!r}")
    return features


# ---------------------------------------------------------------------------
# Cross-symbol returns index
# ---------------------------------------------------------------------------

def _build_cross_returns_index(
    grouped: dict[tuple[str, str], list[Bar]],
) -> dict[str, dict[str, float]]:
    """
    Build a timestamp → {symbol: return_1} lookup for CurrencyStrengthMatrix features.

    Covers every bar across all (symbol, timeframe) groups.
    Only 1-bar fractional returns are stored (no look-ahead).
    """
    index: dict[str, dict[str, float]] = {}
    for (symbol, _timeframe), series in grouped.items():
        for i in range(1, len(series)):
            r1 = _safe_div(series[i].close - series[i - 1].close, series[i - 1].close)
            ts = series[i].timestamp.isoformat()
            index.setdefault(ts, {})[symbol] = r1
    return index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_processed_dataset(bars: list[Bar], labeling: LabelingConfig) -> ProcessedDataset:
    grouped = group_bars(bars)
    cross_returns_index = _build_cross_returns_index(grouped)

    rows: list[ProcessedRow] = []
    series_summaries: list[dict[str, object]] = []
    warmup = 20  # Minimum history required for base features (10-bar window + 10 bars).
                 # Regime features (50-bar) degrade gracefully when fewer bars are available.

    for (symbol, timeframe), series in sorted(grouped.items()):
        usable_rows = 0
        for index in range(warmup - 1, len(series)):
            label = build_label(series, index, labeling)
            if label is None:
                continue
            ts_iso = series[index].timestamp.isoformat()
            cross_returns = cross_returns_index.get(ts_iso, {})
            features = _compute_feature_row(series, index, cross_returns=cross_returns)
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

    rows.sort(key=lambda r: (r.timestamp, r.symbol, r.timeframe))
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
            "0":  sum(1 for row in rows if row.label == 0),
            "1":  sum(1 for row in rows if row.label == 1),
        },
    }
    return ProcessedDataset(
        rows=rows,
        feature_names=FEATURE_NAMES_BASE.copy(),
        label_mode=labeling.mode,
        schema=cast(dict[str, object], schema),
        manifest=manifest,
    )


def write_processed_dataset(
    dataset: ProcessedDataset,
    dataset_path: Path,
    manifest_path: Path,
    schema_path: Path,
) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "symbol", "timeframe",
        "open", "high", "low", "close", "volume",
        *dataset.feature_names,
        "label", "label_reason", "horizon_end_timestamp",
    ]
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset.rows:
            payload: dict[str, object] = {
                "timestamp":              row.timestamp.isoformat(),
                "symbol":                 row.symbol,
                "timeframe":              row.timeframe,
                "open":                   f"{row.open:.10f}",
                "high":                   f"{row.high:.10f}",
                "low":                    f"{row.low:.10f}",
                "close":                  f"{row.close:.10f}",
                "volume":                 f"{row.volume:.10f}",
                "label":                  row.label,
                "label_reason":           row.label_reason,
                "horizon_end_timestamp":  row.horizon_end_timestamp,
            }
            payload.update({name: f"{row.features[name]:.10f}" for name in dataset.feature_names})
            writer.writerow(payload)

    manifest_path.write_text(json.dumps(dataset.manifest, indent=2, sort_keys=True), encoding="utf-8")
    schema_path.write_text(json.dumps(dataset.schema, indent=2, sort_keys=True), encoding="utf-8")


def load_processed_dataset(
    dataset_path: Path,
    schema_path: Path,
    manifest_path: Path,
) -> ProcessedDataset:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")
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
    return ProcessedDataset(
        rows=rows,
        feature_names=feature_names,
        label_mode=str(schema["label_mode"]),
        schema=schema,
        manifest=manifest,
    )
