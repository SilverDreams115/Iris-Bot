from datetime import datetime, timedelta

from iris_bot.config import LabelingConfig
from iris_bot.data import Bar
from iris_bot.labels import next_bar_direction_label, triple_barrier_label


def test_next_bar_direction_label_supports_no_trade() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    bars = [
        Bar(start, "EURUSD", "M5", 1.1, 1.2, 1.0, 1.1000, 100),
        Bar(start + timedelta(minutes=5), "EURUSD", "M5", 1.1, 1.2, 1.0, 1.1001, 100),
    ]

    outcome = next_bar_direction_label(bars, 0, LabelingConfig(mode="next_bar_direction", min_abs_return=0.001))

    assert outcome is not None
    assert outcome.label == 0


def test_triple_barrier_label_detects_long_hit() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    bars = [
        Bar(start, "EURUSD", "M5", 1.1000, 1.1010, 1.0990, 1.1000, 100),
        Bar(start + timedelta(minutes=5), "EURUSD", "M5", 1.1000, 1.1030, 1.0995, 1.1020, 100),
        Bar(start + timedelta(minutes=10), "EURUSD", "M5", 1.1020, 1.1040, 1.1010, 1.1030, 100),
    ]

    outcome = triple_barrier_label(
        bars,
        0,
        LabelingConfig(mode="triple_barrier", horizon_bars=2, take_profit_pct=0.002, stop_loss_pct=0.002),
    )

    assert outcome is not None
    assert outcome.label == 1
