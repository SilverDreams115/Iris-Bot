from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_bars(csv_path: Path) -> list[Bar]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    bars: list[Bar] = []
    for row in rows:
        bars.append(
            Bar(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
            )
        )
    return bars


def group_bars(bars: list[Bar]) -> dict[tuple[str, str], list[Bar]]:
    grouped: dict[tuple[str, str], list[Bar]] = {}
    for bar in bars:
        grouped.setdefault((bar.symbol, bar.timeframe), []).append(bar)

    for series in grouped.values():
        series.sort(key=lambda item: item.timestamp)

    return grouped


def write_bars(csv_path: Path, bars: list[Bar]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"])
        for bar in sorted(bars, key=lambda item: (item.symbol, item.timeframe, item.timestamp)):
            writer.writerow(
                [
                    bar.timestamp.isoformat(),
                    bar.symbol,
                    bar.timeframe,
                    f"{bar.open:.10f}",
                    f"{bar.high:.10f}",
                    f"{bar.low:.10f}",
                    f"{bar.close:.10f}",
                    f"{bar.volume:.2f}",
                ]
            )
