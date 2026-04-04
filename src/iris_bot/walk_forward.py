from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class WalkForwardWindow:
    fold_index: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def generate_walk_forward_windows(
    total_rows: int,
    train_window: int,
    validation_window: int,
    test_window: int,
    step: int,
) -> list[WalkForwardWindow]:
    if min(train_window, validation_window, test_window, step) <= 0:
        raise ValueError("Los parametros de walk-forward deben ser positivos")
    windows: list[WalkForwardWindow] = []
    cursor = 0
    fold_index = 0
    while cursor + train_window + validation_window + test_window <= total_rows:
        train_start = cursor
        train_end = cursor + train_window
        validation_start = train_end
        validation_end = validation_start + validation_window
        test_start = validation_end
        test_end = test_start + test_window
        windows.append(
            WalkForwardWindow(
                fold_index=fold_index,
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_index += 1
        cursor += step
    return windows
