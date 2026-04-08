from datetime import datetime, timedelta

import pytest

from iris_bot.config import LabelingConfig
from iris_bot.data import Bar
from iris_bot.labels import build_label, next_bar_direction_label, triple_barrier_label


def test_next_bar_direction_label_supports_no_trade() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    bars = [
        Bar(start, "EURUSD", "M5", 1.1, 1.2, 1.0, 1.1000, 100),
        Bar(start + timedelta(minutes=5), "EURUSD", "M5", 1.1, 1.2, 1.0, 1.1001, 100),
    ]

    outcome = next_bar_direction_label(bars, 0, LabelingConfig(mode="next_bar_direction", min_abs_return=0.001))

    assert outcome is not None
    assert outcome.label == 0


def test_triple_barrier_label_detects_long_hit() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    bars = [
        Bar(start, "EURUSD", "M5", 1.1000, 1.1010, 1.0990, 1.1000, 100),
        Bar(start + timedelta(minutes=5), "EURUSD", "M5", 1.1000, 1.1030, 1.0995, 1.1020, 100),
        Bar(start + timedelta(minutes=10), "EURUSD", "M5", 1.1020, 1.1040, 1.1010, 1.1030, 100),
    ]

    outcome = triple_barrier_label(
        bars,
        0,
        LabelingConfig(mode="triple_barrier", horizon_bars=2, take_profit_pct=0.002, stop_loss_pct=0.002),
    )

    assert outcome is not None
    assert outcome.label == 1


# ---------------------------------------------------------------------------
# next_bar_direction_label — boundary and imbalance cases
# ---------------------------------------------------------------------------

def _bar(start: datetime, index: int, close: float) -> Bar:
    ts = start + timedelta(minutes=5 * index)
    return Bar(ts, "EURUSD", "M5", close, close + 0.001, close - 0.001, close, 100)


def test_next_bar_direction_label_long() -> None:
    start = datetime(2026, 1, 1)
    bars = [_bar(start, 0, 1.1000), _bar(start, 1, 1.1020)]
    outcome = next_bar_direction_label(bars, 0, LabelingConfig(mode="next_bar_direction", min_abs_return=0.0001))
    assert outcome is not None
    assert outcome.label == 1


def test_next_bar_direction_label_short() -> None:
    start = datetime(2026, 1, 1)
    bars = [_bar(start, 0, 1.1020), _bar(start, 1, 1.1000)]
    outcome = next_bar_direction_label(bars, 0, LabelingConfig(mode="next_bar_direction", min_abs_return=0.0001))
    assert outcome is not None
    assert outcome.label == -1


def test_next_bar_direction_label_returns_none_at_last_bar() -> None:
    start = datetime(2026, 1, 1)
    bars = [_bar(start, 0, 1.1000), _bar(start, 1, 1.1010)]
    assert next_bar_direction_label(bars, 1, LabelingConfig(mode="next_bar_direction")) is None


def test_next_bar_direction_label_all_same_close_no_trade() -> None:
    """All bars have the same close → all labels should be 0 (imbalance: only no-trade)."""
    start = datetime(2026, 1, 1)
    bars = [_bar(start, i, 1.1000) for i in range(5)]
    config = LabelingConfig(mode="next_bar_direction", min_abs_return=0.0001, allow_no_trade=True)
    labels = [next_bar_direction_label(bars, i, config) for i in range(len(bars) - 1)]
    assert all(o is not None and o.label == 0 for o in labels)


# ---------------------------------------------------------------------------
# triple_barrier_label — short hit, ambiguous, timeout, boundary
# ---------------------------------------------------------------------------

def test_triple_barrier_label_short_hit() -> None:
    start = datetime(2026, 1, 1)
    # entry close = 1.1000 → lower_barrier = 1.1000 * (1-0.002) = 1.0978
    # next bar low = 1.0970 < 1.0978 → stop loss triggered
    bars = [
        Bar(start, "EURUSD", "M5", 1.1000, 1.1005, 1.0995, 1.1000, 100),
        Bar(start + timedelta(minutes=5), "EURUSD", "M5", 1.0990, 1.0992, 1.0970, 1.0975, 100),
    ]
    outcome = triple_barrier_label(
        bars, 0, LabelingConfig(mode="triple_barrier", horizon_bars=1, take_profit_pct=0.05, stop_loss_pct=0.002)
    )
    assert outcome is not None
    assert outcome.label == -1
    assert outcome.label_reason == "triple_barrier_stop_loss"


def test_triple_barrier_label_ambiguous_allow_no_trade() -> None:
    """Both TP and SL hit on the same bar → label=0 when allow_no_trade=True."""
    start = datetime(2026, 1, 1)
    bars = [
        Bar(start, "EURUSD", "M5", 1.1000, 1.1010, 1.0990, 1.1000, 100),
        # Bar that simultaneously reaches TP (+0.001) and SL (-0.001)
        Bar(start + timedelta(minutes=5), "EURUSD", "M5", 1.1000, 1.1015, 1.0985, 1.1000, 100),
    ]
    config = LabelingConfig(mode="triple_barrier", horizon_bars=1, take_profit_pct=0.001, stop_loss_pct=0.001, allow_no_trade=True)
    outcome = triple_barrier_label(bars, 0, config)
    assert outcome is not None
    assert outcome.label == 0
    assert outcome.label_reason == "triple_barrier_ambiguous"


def test_triple_barrier_label_timeout_direction() -> None:
    """No barrier hit within horizon → label based on terminal close direction."""
    start = datetime(2026, 1, 1)
    bars = [
        Bar(start, "EURUSD", "M5", 1.1000, 1.1002, 1.0998, 1.1000, 100),
        Bar(start + timedelta(minutes=5), "EURUSD", "M5", 1.1001, 1.1003, 1.0999, 1.1005, 100),
        Bar(start + timedelta(minutes=10), "EURUSD", "M5", 1.1005, 1.1007, 1.1003, 1.1010, 100),
    ]
    config = LabelingConfig(mode="triple_barrier", horizon_bars=2, take_profit_pct=0.05, stop_loss_pct=0.05, min_abs_return=0.0001)
    outcome = triple_barrier_label(bars, 0, config)
    assert outcome is not None
    assert outcome.label == 1
    assert "timeout" in outcome.label_reason


def test_triple_barrier_label_timeout_can_be_filtered_to_neutral() -> None:
    start = datetime(2026, 1, 1)
    bars = [
        Bar(start, "EURUSD", "M5", 1.1000, 1.1002, 1.0998, 1.1000, 100),
        Bar(start + timedelta(minutes=5), "EURUSD", "M5", 1.1000, 1.1002, 1.0999, 1.1002, 100),
        Bar(start + timedelta(minutes=10), "EURUSD", "M5", 1.1002, 1.1004, 1.1000, 1.1004, 100),
    ]
    config = LabelingConfig(
        mode="triple_barrier",
        horizon_bars=2,
        take_profit_pct=0.0020,
        stop_loss_pct=0.0020,
        min_abs_return=0.0001,
        timeout_handling_mode="neutral_by_barrier_fraction",
        timeout_direction_min_barrier_fraction=0.50,
    )
    outcome = triple_barrier_label(bars, 0, config)
    assert outcome is not None
    assert outcome.label == 0
    assert outcome.label_reason == "triple_barrier_timeout_filtered_small_move"


def test_triple_barrier_label_returns_none_when_not_enough_bars() -> None:
    start = datetime(2026, 1, 1)
    bars = [
        Bar(start, "EURUSD", "M5", 1.1000, 1.1010, 1.0990, 1.1000, 100),
        Bar(start + timedelta(minutes=5), "EURUSD", "M5", 1.1000, 1.1020, 1.0990, 1.1010, 100),
    ]
    config = LabelingConfig(mode="triple_barrier", horizon_bars=5)
    assert triple_barrier_label(bars, 0, config) is None


# ---------------------------------------------------------------------------
# build_label dispatch
# ---------------------------------------------------------------------------

def test_build_label_dispatches_next_bar_direction() -> None:
    start = datetime(2026, 1, 1)
    bars = [_bar(start, 0, 1.1000), _bar(start, 1, 1.1010)]
    outcome = build_label(bars, 0, LabelingConfig(mode="next_bar_direction", min_abs_return=0.0001))
    assert outcome is not None


def test_build_label_dispatches_triple_barrier() -> None:
    start = datetime(2026, 1, 1)
    bars = [
        Bar(start, "EURUSD", "M5", 1.1000, 1.1010, 1.0990, 1.1000, 100),
        Bar(start + timedelta(minutes=5), "EURUSD", "M5", 1.1000, 1.1030, 1.0990, 1.1025, 100),
        Bar(start + timedelta(minutes=10), "EURUSD", "M5", 1.1025, 1.1035, 1.1020, 1.1030, 100),
    ]
    outcome = build_label(bars, 0, LabelingConfig(mode="triple_barrier", horizon_bars=2, take_profit_pct=0.002, stop_loss_pct=0.002))
    assert outcome is not None


def test_build_label_raises_for_unknown_mode() -> None:
    start = datetime(2026, 1, 1)
    bars = [_bar(start, 0, 1.1), _bar(start, 1, 1.2)]
    with pytest.raises(ValueError):
        build_label(bars, 0, LabelingConfig(mode="unsupported_mode"))
