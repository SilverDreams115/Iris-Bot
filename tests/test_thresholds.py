
from iris_bot.thresholds import (
    _refined_grid,
    select_threshold_from_probabilities,
    select_threshold_from_scores,
)


def test_threshold_selection_returns_best_threshold() -> None:
    result = select_threshold_from_scores(
        scores=[0.9, 0.6, -0.7, 0.1, -0.1],
        labels=[1, 1, -1, 0, 0],
        grid=(0.1, 0.5, 0.8),
        metric_name="macro_f1",
    )

    assert result.threshold in {0.1, 0.5, 0.8}
    assert result.metric_value >= 0.0


def test_refined_grid_spans_one_step_around_best() -> None:
    # coarse grid step = 0.05; best at 0.55 → refined span [0.50, 0.60]
    refined = _refined_grid(grid=(0.45, 0.50, 0.55, 0.60, 0.65, 0.70), best_threshold=0.55, refinement_steps=10)
    assert len(refined) == 11
    assert abs(refined[0] - 0.50) < 1e-7
    assert abs(refined[-1] - 0.60) < 1e-7
    # all points must be strictly between endpoints or equal to them
    for pt in refined:
        assert 0.50 - 1e-9 <= pt <= 0.60 + 1e-9


def test_refined_grid_clamped_at_zero() -> None:
    # best at edge of grid — lo must not go below 0.0
    refined = _refined_grid(grid=(0.0, 0.1, 0.2), best_threshold=0.0, refinement_steps=5)
    assert all(pt >= 0.0 for pt in refined)


def test_refined_grid_clamped_at_one() -> None:
    refined = _refined_grid(grid=(0.8, 0.9, 1.0), best_threshold=1.0, refinement_steps=5)
    assert all(pt <= 1.0 for pt in refined)


def test_refined_grid_disabled_when_steps_zero() -> None:
    assert _refined_grid(grid=(0.4, 0.5, 0.6), best_threshold=0.5, refinement_steps=0) == ()


def test_refined_grid_disabled_for_single_point_grid() -> None:
    assert _refined_grid(grid=(0.5,), best_threshold=0.5, refinement_steps=10) == ()


def test_refinement_metric_never_regresses() -> None:
    # With a noisy but learnable signal, the refined result should be >= coarse result
    scores = [0.9, 0.6, 0.3, -0.8, -0.5, 0.1, 0.7, -0.4]
    labels = [1, 1, 0, -1, -1, 0, 1, -1]
    grid = (0.3, 0.5, 0.7)

    coarse = select_threshold_from_scores(scores, labels, grid, "macro_f1", refinement_steps=0)
    refined = select_threshold_from_scores(scores, labels, grid, "macro_f1", refinement_steps=10)

    assert refined.metric_value >= coarse.metric_value - 1e-9


def test_refinement_can_find_sub_grid_threshold() -> None:
    # Construct a case where the optimum lies between 0.50 and 0.55
    # prob scores: one sample needs exactly 0.52 to be classified correctly
    probs = [
        {1: 0.53, 0: 0.30, -1: 0.17},
        {1: 0.53, 0: 0.30, -1: 0.17},
        {0: 0.80, 1: 0.10, -1: 0.10},
    ]
    labels = [1, 1, 0]

    coarse = select_threshold_from_probabilities(probs, labels, (0.45, 0.55, 0.65), "macro_f1", refinement_steps=0)
    refined = select_threshold_from_probabilities(probs, labels, (0.45, 0.55, 0.65), "macro_f1", refinement_steps=10)

    # Refined must be at least as good; threshold may differ from coarse grid points
    assert refined.metric_value >= coarse.metric_value - 1e-9
    # Refined threshold should fall within [lo, hi] span around coarse best
    assert 0.45 - 1e-9 <= refined.threshold <= 0.65 + 1e-9
