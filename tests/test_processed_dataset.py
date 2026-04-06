from datetime import datetime, timedelta
from math import isfinite

from iris_bot.config import LabelingConfig
from iris_bot.data import Bar
from iris_bot.processed_dataset import FEATURE_NAMES_BASE, build_processed_dataset


def test_processed_dataset_features_do_not_use_future_bar() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    bars = []
    price = 1.1000
    for index in range(30):
        if index == 21:
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

    expected_previous_close = bars[18].close
    expected_current_close = bars[19].close
    expected_return_1 = (expected_current_close - expected_previous_close) / expected_previous_close
    assert round(first_row.features["return_1"], 10) == round(expected_return_1, 10)


def test_processed_dataset_exposes_new_regime_features_and_warmup_20() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    bars = []
    price = 1.1000
    for index in range(30):
        price += 0.0006 if index % 4 in (0, 1) else -0.0002
        bars.append(
            Bar(
                timestamp=start + timedelta(minutes=15 * index),
                symbol="EURUSD",
                timeframe="M15",
                open=price - 0.0003,
                high=price + 0.0008 + (index * 0.00001),
                low=price - 0.0007,
                close=price,
                volume=100 + index,
            )
        )

    dataset = build_processed_dataset(bars, LabelingConfig(mode="next_bar_direction"))

    assert dataset.feature_names == FEATURE_NAMES_BASE
    assert dataset.manifest["feature_count"] == len(FEATURE_NAMES_BASE)
    assert dataset.rows
    assert dataset.rows[0].timestamp == bars[19].timestamp
    for feature_name in (
        "parkinson_volatility_10",
        "return_autocorr_10",
        "efficiency_ratio_10",
        "volume_percentile_20",
    ):
        assert feature_name in dataset.rows[0].features
        assert isfinite(dataset.rows[0].features[feature_name])


def test_efficiency_ratio_and_volume_percentile_stay_in_expected_bounds() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    bars = []
    price = 1.2000
    volumes = [50, 60, 55, 65, 70, 68, 72, 80, 78, 82, 85, 88, 90, 95, 92, 96, 100, 104, 108, 120, 125, 130]
    for index, volume in enumerate(volumes):
        price += 0.0005
        bars.append(
            Bar(
                timestamp=start + timedelta(minutes=15 * index),
                symbol="EURUSD",
                timeframe="M15",
                open=price - 0.0002,
                high=price + 0.0004,
                low=price - 0.0003,
                close=price,
                volume=float(volume),
            )
        )

    dataset = build_processed_dataset(bars, LabelingConfig(mode="next_bar_direction"))
    row = dataset.rows[0]

    assert 0.0 <= row.features["efficiency_ratio_10"] <= 1.0
    assert 0.0 <= row.features["volume_percentile_20"] <= 1.0
