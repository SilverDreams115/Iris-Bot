from dataclasses import dataclass
from datetime import datetime, timedelta

from iris_bot.splits import temporal_train_validation_test_split


@dataclass(frozen=True)
class Row:
    timestamp: datetime


def test_temporal_split_keeps_order() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [Row(start + timedelta(minutes=index)) for index in range(10)]

    split = temporal_train_validation_test_split(rows, 0.6, 0.2, 0.2)

    assert split.train[-1].timestamp <= split.validation[0].timestamp
    assert split.validation[-1].timestamp <= split.test[0].timestamp
