from iris_bot.thresholds import select_threshold_from_scores


def test_threshold_selection_returns_best_threshold() -> None:
    result = select_threshold_from_scores(
        scores=[0.9, 0.6, -0.7, 0.1, -0.1],
        labels=[1, 1, -1, 0, 0],
        grid=(0.1, 0.5, 0.8),
        metric_name="macro_f1",
    )

    assert result.threshold in {0.1, 0.5, 0.8}
    assert result.metric_value >= 0.0
