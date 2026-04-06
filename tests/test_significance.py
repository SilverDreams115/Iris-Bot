from __future__ import annotations

from datetime import datetime

from iris_bot.processed_dataset import ProcessedRow
from iris_bot.significance import (
    EvaluationResult,
    compute_deflated_sharpe_ratio,
    estimate_expected_maximum_sharpe_ratio,
    run_permutation_significance,
)


def _rows() -> list[ProcessedRow]:
    base = datetime(2026, 1, 1, 0, 0, 0)
    labels = [1, -1, 1, 0, -1, 1]
    rows: list[ProcessedRow] = []
    for index, label in enumerate(labels):
        rows.append(
            ProcessedRow(
                timestamp=base,
                symbol="EURUSD",
                timeframe="M15",
                open=1.1,
                high=1.2,
                low=1.0,
                close=1.1,
                volume=100.0,
                label=label,
                label_reason="test",
                horizon_end_timestamp=base.isoformat(),
                features={"x": float(index)},
            )
        )
    return rows


def test_run_permutation_significance_is_deterministic() -> None:
    rows = _rows()

    def evaluator(candidate_rows: list[ProcessedRow]) -> EvaluationResult:
        metric = sum((index + 1) * row.label for index, row in enumerate(candidate_rows))
        return EvaluationResult(
            metric_value=float(metric),
            aggregate={"score": float(metric)},
            valid_folds=2,
            skipped_folds=0,
        )

    first = run_permutation_significance(
        rows=rows,
        evaluator=evaluator,
        trials=8,
        seed=123,
        metric_name="score",
        higher_is_better=True,
        minimum_valid_folds=1,
    )
    second = run_permutation_significance(
        rows=rows,
        evaluator=evaluator,
        trials=8,
        seed=123,
        metric_name="score",
        higher_is_better=True,
        minimum_valid_folds=1,
    )

    assert first == second
    assert first["status"] == "completed"
    assert first["trials_used_in_null_distribution"] == 8
    assert first["p_value"] is not None
    assert first["percentile_rank"] is not None


def test_run_permutation_significance_skips_when_real_result_has_too_few_valid_folds() -> None:
    rows = _rows()

    def evaluator(candidate_rows: list[ProcessedRow]) -> EvaluationResult:
        metric = float(sum(row.label for row in candidate_rows))
        return EvaluationResult(
            metric_value=metric,
            aggregate={"score": metric},
            valid_folds=0,
            skipped_folds=3,
        )

    report = run_permutation_significance(
        rows=rows,
        evaluator=evaluator,
        trials=5,
        seed=7,
        metric_name="score",
        higher_is_better=True,
        minimum_valid_folds=1,
    )

    assert report["status"] == "skipped"
    assert report["trials_completed"] == 0
    assert report["p_value"] is None
    assert report["percentile_rank"] is None


def test_estimate_expected_maximum_sharpe_ratio_exceeds_mean_when_dispersion_exists() -> None:
    trial_sharpes = [-0.4, -0.1, 0.0, 0.2, 0.7]
    estimated = estimate_expected_maximum_sharpe_ratio(trial_sharpes)

    assert estimated is not None
    assert estimated > sum(trial_sharpes) / len(trial_sharpes)


def test_compute_deflated_sharpe_ratio_returns_completed_payload() -> None:
    observed_returns = [0.012, 0.009, -0.004, 0.015, 0.011, 0.003]
    trial_sharpes = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.15]

    payload = compute_deflated_sharpe_ratio(observed_returns, trial_sharpes)

    assert payload["status"] == "completed"
    assert payload["deflated_sharpe_ratio"] is not None
    assert 0.0 <= payload["deflated_sharpe_ratio"] <= 1.0
    assert payload["benchmark_sharpe_ratio"] is not None
