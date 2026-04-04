import json
from datetime import datetime
from pathlib import Path

from iris_bot.data import Bar
from iris_bot.datasets import write_dataset_bundle


def test_write_dataset_bundle_creates_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "raw" / "market.csv"
    metadata_path = tmp_path / "raw" / "market.csv.metadata.json"
    bars = [
        Bar(datetime(2026, 1, 1, 0, 0, 0), "EURUSD", "M5", 1.1, 1.2, 1.0, 1.15, 100),
    ]

    manifest = write_dataset_bundle(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        bars=bars,
        source="unit-test",
        history_bars_requested=123,
        extra={"case": "roundtrip"},
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert manifest.row_count == 1
    assert payload["source"] == "unit-test"
    assert payload["symbols"] == ["EURUSD"]
