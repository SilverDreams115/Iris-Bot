from datetime import datetime, timedelta

from iris_bot.config import LabelingConfig
from iris_bot.data import Bar
from iris_bot.processed_dataset import build_processed_dataset


def test_processed_dataset_features_do_not_use_future_bar() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    bars = []
    price = 1.1000
    for index in range(20):
        if index == 11:
            price += 0.0500
        else:
            price += 0.0005
        bars.append(
            Bar(
                timestamp=start + timedelta(minutes=5 * index),
                symbol="EURUSD",
                timeframe="M5",
                open=price - 0.0002,
                high=price + 0.0005,
                low=price - 0.0005,
                close=price,
                volume=100 + index,
            )
        )

    dataset = build_processed_dataset(bars, LabelingConfig(mode="next_bar_direction"))
    first_row = dataset.rows[0]

    expected_previous_close = bars[8].close
    expected_current_close = bars[9].close
    expected_return_1 = (expected_current_close - expected_previous_close) / expected_previous_close
    assert round(first_row.features["return_1"], 10) == round(expected_return_1, 10)

