from datetime import datetime
from pathlib import Path

from iris_bot.data import Bar, load_bars, write_bars


def test_write_bars_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "market.csv"
    bars = [
        Bar(datetime(2026, 1, 1, 0, 0, 0), "EURUSD", "M5", 1.1, 1.2, 1.0, 1.15, 100),
        Bar(datetime(2026, 1, 1, 0, 5, 0), "EURUSD", "M5", 1.15, 1.25, 1.1, 1.2, 110),
    ]

    write_bars(csv_path, bars)
    loaded = load_bars(csv_path)

    assert len(loaded) == 2
    assert loaded[1].close == 1.2
