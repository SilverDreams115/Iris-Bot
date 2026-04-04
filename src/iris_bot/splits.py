from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SplitSummary:
    name: str
    count: int
    start_timestamp: str | None
    end_timestamp: str | None


@dataclass(frozen=True)
class TemporalSplit:
    train: list[object]
    validation: list[object]
    test: list[object]
    summaries: list[SplitSummary]


def _extract_timestamp(item: object) -> datetime:
    return getattr(item, "timestamp")


def temporal_train_validation_test_split(
    rows: list[object],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> TemporalSplit:
    if abs((train_ratio + validation_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("Las proporciones de split deben sumar 1.0")
    if len(rows) < 5:
        raise ValueError("No hay suficientes filas para split temporal")

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
