from __future__ import annotations

from dataclasses import dataclass

from iris_bot.data import Bar


@dataclass(frozen=True)
class FeatureVector:
    symbol: str
    timeframe: str
    close: float
    ret_1: float
    ret_3: float
    range_ratio: float
    volume_zscore: float
    atr_3: float
    trend_gap: float
    target: int
    next_return: float

    def as_inputs(self) -> list[float]:
        return [
            self.ret_1,
            self.ret_3,
            self.range_ratio,
            self.volume_zscore,
            self.atr_3,
            self.trend_gap,
        ]


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else numerator / denominator


def build_feature_vectors(bars: list[Bar]) -> list[FeatureVector]:
    vectors: list[FeatureVector] = []
    for index in range(3, len(bars) - 1):
        current = bars[index]
        prev_1 = bars[index - 1]
        prev_3 = bars[index - 3]
        next_bar = bars[index + 1]
        avg_volume = sum(item.volume for item in bars[index - 3 : index + 1]) / 4
        tr_values = [(item.high - item.low) for item in bars[index - 2 : index + 1]]
        atr_3 = sum(tr_values) / len(tr_values)
        ret_1 = _safe_div(current.close - prev_1.close, prev_1.close)
        ret_3 = _safe_div(current.close - prev_3.close, prev_3.close)
        midpoint = (current.high + current.low) / 2
        trend_gap = _safe_div(current.close - midpoint, midpoint)
        target = 1 if next_bar.close > current.close else 0
        vectors.append(
            FeatureVector(
                symbol=current.symbol,
                timeframe=current.timeframe,
                close=current.close,
                ret_1=ret_1,
                ret_3=ret_3,
                range_ratio=_safe_div(current.high - current.low, current.close),
                volume_zscore=_safe_div(current.volume - avg_volume, avg_volume),
                atr_3=_safe_div(atr_3, current.close),
                trend_gap=trend_gap,
                target=target,
                next_return=_safe_div(next_bar.close - current.close, current.close),
            )
        )
    return vectors
