from datetime import datetime, timedelta

from iris_bot.data import Bar
from iris_bot.validation import validate_bars


def test_validate_bars_detects_gap_and_duplicate() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    bars = [
        Bar(start, "EURUSD", "M5", 1.1, 1.2, 1.0, 1.15, 100),
        Bar(start, "EURUSD", "M5", 1.1, 1.2, 1.0, 1.15, 100),
        Bar(start + timedelta(minutes=15), "EURUSD", "M5", 1.15, 1.21, 1.11, 1.2, 110),
    ]

    report = validate_bars(bars)

    assert report.is_valid is False
    assert any(issue.code == "duplicate_timestamp" for issue in report.issues)
    assert any(issue.code == "gap_detected" for issue in report.issues)
