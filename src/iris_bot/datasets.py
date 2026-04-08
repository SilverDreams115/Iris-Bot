from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path

from iris_bot.data import Bar, write_bars
from iris_bot.durable_io import durable_write_json


@dataclass(frozen=True)
class DatasetManifest:
    created_at: str
    dataset_path: str
    row_count: int
    symbols: list[str]
    timeframes: list[str]
    source: str
    history_bars_requested: int
    extra: dict[str, str]


def write_dataset_bundle(
    dataset_path: Path,
    metadata_path: Path,
    bars: list[Bar],
    source: str,
    history_bars_requested: int,
    extra: dict[str, str] | None = None,
) -> DatasetManifest:
    write_bars(dataset_path, bars)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest(
        created_at=datetime.now(tz=UTC).isoformat(),
        dataset_path=str(dataset_path),
        row_count=len(bars),
        symbols=sorted({bar.symbol for bar in bars}),
        timeframes=sorted({bar.timeframe for bar in bars}),
        source=source,
        history_bars_requested=history_bars_requested,
        extra=extra or {},
    )
    durable_write_json(metadata_path, asdict(manifest))
    return manifest
