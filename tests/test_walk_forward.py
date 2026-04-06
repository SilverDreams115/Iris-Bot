import pytest

from iris_bot.walk_forward import generate_walk_forward_windows


def test_generate_walk_forward_windows_returns_multiple_folds() -> None:
    windows = generate_walk_forward_windows(total_rows=600, train_window=200, validation_window=100, test_window=100, step=100)

    assert len(windows) == 3
    assert windows[0].train_end == 200
    assert windows[1].train_start == 100


def test_generate_walk_forward_windows_insufficient_data_returns_empty() -> None:
    """When total_rows < train+validation+test the function returns an empty list (no folds)."""
    windows = generate_walk_forward_windows(total_rows=50, train_window=100, validation_window=50, test_window=50, step=50)
    assert windows == []


def test_generate_walk_forward_windows_exactly_one_fold() -> None:
    """Exactly train+validation+test rows → exactly 1 window."""
    windows = generate_walk_forward_windows(total_rows=300, train_window=200, validation_window=50, test_window=50, step=50)
    assert len(windows) == 1
    assert windows[0].fold_index == 0
    assert windows[0].train_start == 0
    assert windows[0].test_end == 300


def test_generate_walk_forward_windows_non_overlapping_indices() -> None:
    """train_end == validation_start == test_start consistency check for every fold."""
    windows = generate_walk_forward_windows(total_rows=800, train_window=200, validation_window=100, test_window=100, step=100)
    for win in windows:
        assert win.train_end == win.validation_start
        assert win.validation_end == win.test_start


def test_generate_walk_forward_windows_zero_step_raises() -> None:
    with pytest.raises(ValueError):
        generate_walk_forward_windows(total_rows=500, train_window=200, validation_window=100, test_window=100, step=0)


def test_generate_walk_forward_windows_negative_window_raises() -> None:
    with pytest.raises(ValueError):
        generate_walk_forward_windows(total_rows=500, train_window=-1, validation_window=100, test_window=100, step=50)
