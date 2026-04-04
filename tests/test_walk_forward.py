from iris_bot.walk_forward import generate_walk_forward_windows


def test_generate_walk_forward_windows_returns_multiple_folds() -> None:
    windows = generate_walk_forward_windows(total_rows=600, train_window=200, validation_window=100, test_window=100, step=100)

    assert len(windows) == 3
    assert windows[0].train_end == 200
    assert windows[1].train_start == 100
