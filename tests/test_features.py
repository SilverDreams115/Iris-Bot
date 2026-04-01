from datetime import datetime, timedelta

from iris_bot.data import Bar
from iris_bot.features import build_feature_vectors


def test_build_feature_vectors_generates_targets() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    bars = [
        Bar(start + timedelta(minutes=index * 5), "EURUSD", "M5", 1.1 + index * 0.001, 1.101 + index * 0.001, 1.099 + index * 0.001, 1.1 + index * 0.001, 100 + index)
        for index in range(6)
    ]

    vectors = build_feature_vectors(bars)

    assert len(vectors) == 2
    assert vectors[0].symbol == "EURUSD"
    assert vectors[0].timeframe == "M5"
    assert vectors[0].target == 1
