from __future__ import annotations

from dataclasses import dataclass

from iris_bot.config import RiskConfig, TradingConfig
from iris_bot.features import FeatureVector
from iris_bot.model import LogisticClassifier


@dataclass(frozen=True)
class Signal:
    symbol: str
    timeframe: str
    direction: int
    confidence: float
    close: float
    atr_3: float


def train_model(rows: list[FeatureVector]) -> LogisticClassifier:
    model = LogisticClassifier()
    model.fit(rows)
    return model


def build_signal(
    row: FeatureVector,
    model: LogisticClassifier,
    risk: RiskConfig,
    trading: TradingConfig,
) -> Signal | None:
    confidence = model.predict_proba(row)
    if confidence >= risk.min_confidence and trading.allow_long:
        direction = 1
    elif confidence <= 1.0 - risk.min_confidence and trading.allow_short:
        direction = -1
    else:
        return None

    return Signal(
        symbol=row.symbol,
        timeframe=row.timeframe,
        direction=direction,
        confidence=confidence,
        close=row.close,
        atr_3=row.atr_3,
    )
