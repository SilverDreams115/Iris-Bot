from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Protocol, TypeVar


@dataclass(frozen=True)
class SplitSummary:
    name: str
    count: int
    start_timestamp: str | None
    end_timestamp: str | None


class TimestampedRow(Protocol):
    @property
    def timestamp(self) -> datetime: ...


RowT = TypeVar("RowT", bound=TimestampedRow)


@dataclass(frozen=True)
class TemporalSplit(Generic[RowT]):
    train: list[RowT]
    validation: list[RowT]
    test: list[RowT]
    summaries: list[SplitSummary]


def _extract_timestamp(item: TimestampedRow) -> datetime:
    return item.timestamp


def temporal_train_validation_test_split(
    rows: list[RowT],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> TemporalSplit[RowT]:
    if abs((train_ratio + validation_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    if len(rows) < 5:
        raise ValueError("Insufficient rows for temporal split")

    ordered = sorted(rows, key=_extract_timestamp)
    total = len(ordered)
    train_end = max(1, int(total * train_ratio))
    validation_end = train_end + max(1, int(total * validation_ratio))
    if validation_end >= total:
        validation_end = total - 1

    train_rows = ordered[:train_end]
    validation_rows = ordered[train_end:validation_end]
    test_rows = ordered[validation_end:]
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("El split temporal produjo un segmento vacio")
    if train_rows[-1].timestamp > validation_rows[0].timestamp:
        raise ValueError("El split temporal mezcla train y validation")
    if validation_rows[-1].timestamp > test_rows[0].timestamp:
        raise ValueError("El split temporal mezcla validation y test")

    return TemporalSplit(
        train=train_rows,
        validation=validation_rows,
        test=test_rows,
        summaries=[
            SplitSummary("train", len(train_rows), train_rows[0].timestamp.isoformat(), train_rows[-1].timestamp.isoformat()),
            SplitSummary("validation", len(validation_rows), validation_rows[0].timestamp.isoformat(), validation_rows[-1].timestamp.isoformat()),
            SplitSummary("test", len(test_rows), test_rows[0].timestamp.isoformat(), test_rows[-1].timestamp.isoformat()),
        ],
    )
