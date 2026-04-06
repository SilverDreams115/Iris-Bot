from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import NormalDist, mean, pstdev, stdev
from typing import Any, Protocol

from iris_bot.config import Settings
from iris_bot.processed_dataset import ProcessedRow
from iris_bot.wf_backtest import run_walkforward_economic_backtest

EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class EvaluationResult:
    metric_value: float
    aggregate: dict[str, object]
    valid_folds: int
    skipped_folds: int
    returns: list[float] = field(default_factory=list)
    sharpe_ratio: float | None = None


@dataclass(frozen=True)
class TrialResult:
    trial_index: int
    metric_value: float
    valid_folds: int
    skipped_folds: int
    sharpe_ratio: float | None = None


class Evaluator(Protocol):
    def __call__(self, rows: list[ProcessedRow]) -> EvaluationResult: ...


def _metric_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise KeyError(f"Metric {key!r} not found or not numeric in aggregate payload")


def _safe_sqrt(value: float) -> float | None:
    if value < 0.0:
        return None
    return math.sqrt(value)


def _sample_skewness(returns: list[float]) -> float | None:
    sample_length = len(returns)
    if sample_length < 3:
        return None
    avg = mean(returns)
    centered = [item - avg for item in returns]
    sample_std = stdev(returns)
    if sample_std <= 0.0:
        return 0.0
    third_moment = sum(item ** 3 for item in centered) / sample_length
    return third_moment / (sample_std ** 3)


def _sample_kurtosis(returns: list[float]) -> float | None:
    sample_length = len(returns)
    if sample_length < 4:
        return None
    avg = mean(returns)
    centered = [item - avg for item in returns]
    sample_std = stdev(returns)
    if sample_std <= 0.0:
        return 3.0
    fourth_moment = sum(item ** 4 for item in centered) / sample_length
    return fourth_moment / (sample_std ** 4)


def compute_sharpe_ratio(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    sample_std = stdev(returns)
    if sample_std <= 0.0:
        return None
    return mean(returns) / sample_std


def compute_probabilistic_sharpe_ratio(
    observed_sharpe_ratio: float,
    benchmark_sharpe_ratio: float,
    sample_length: int,
    skewness: float,
    kurtosis: float,
) -> float | None:
    if sample_length < 2:
        return None
    denominator_term = 1.0 - skewness * observed_sharpe_ratio + ((kurtosis - 1.0) / 4.0) * (observed_sharpe_ratio ** 2)
    denominator = _safe_sqrt(denominator_term)
    if denominator is None or denominator == 0.0:
        return None
    z_score = ((observed_sharpe_ratio - benchmark_sharpe_ratio) * math.sqrt(sample_length - 1)) / denominator
    return NormalDist().cdf(z_score)


def estimate_expected_maximum_sharpe_ratio(trial_sharpes: list[float]) -> float | None:
    if not trial_sharpes:
        return None
    if len(trial_sharpes) == 1:
        return trial_sharpes[0]
    sharpe_mean = mean(trial_sharpes)
    sharpe_std = stdev(trial_sharpes)
    trial_count = len(trial_sharpes)
    if sharpe_std <= 0.0:
        return sharpe_mean
    quantile_one = NormalDist().inv_cdf(1.0 - (1.0 / trial_count))
    quantile_two = NormalDist().inv_cdf(1.0 - (1.0 / (trial_count * math.e)))
    return sharpe_mean + sharpe_std * ((1.0 - EULER_MASCHERONI) * quantile_one + EULER_MASCHERONI * quantile_two)


def compute_deflated_sharpe_ratio(
    observed_returns: list[float],
    trial_sharpes: list[float],
) -> dict[str, object]:
    observed_sharpe = compute_sharpe_ratio(observed_returns)
    sample_length = len(observed_returns)
    skewness = _sample_skewness(observed_returns)
    kurtosis = _sample_kurtosis(observed_returns)
    benchmark_sharpe = estimate_expected_maximum_sharpe_ratio(trial_sharpes)

    if observed_sharpe is None:
        return {
            "status": "skipped",
            "reason": "observed_sharpe_ratio_unavailable",
            "sample_length": sample_length,
            "observed_sharpe_ratio": None,
            "benchmark_sharpe_ratio": benchmark_sharpe,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "trial_sharpe_count": len(trial_sharpes),
        }
    if skewness is None or kurtosis is None:
        return {
            "status": "skipped",
            "reason": "insufficient_return_samples_for_higher_moments",
            "sample_length": sample_length,
            "observed_sharpe_ratio": observed_sharpe,
            "benchmark_sharpe_ratio": benchmark_sharpe,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "trial_sharpe_count": len(trial_sharpes),
        }
    if benchmark_sharpe is None:
        return {
            "status": "skipped",
            "reason": "trial_sharpe_distribution_unavailable",
            "sample_length": sample_length,
            "observed_sharpe_ratio": observed_sharpe,
            "benchmark_sharpe_ratio": None,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "trial_sharpe_count": len(trial_sharpes),
        }

    dsr = compute_probabilistic_sharpe_ratio(
        observed_sharpe_ratio=observed_sharpe,
        benchmark_sharpe_ratio=benchmark_sharpe,
        sample_length=sample_length,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    return {
        "status": "completed" if dsr is not None else "skipped",
        "reason": None if dsr is not None else "psr_denominator_invalid",
        "sample_length": sample_length,
        "observed_sharpe_ratio": observed_sharpe,
        "benchmark_sharpe_ratio": benchmark_sharpe,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "trial_sharpe_count": len(trial_sharpes),
        "deflated_sharpe_ratio": dsr,
        "is_significant_at_95pct": dsr is not None and dsr >= 0.95,
    }


def _summarize_distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "stddev_population": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "stddev_population": pstdev(values),
    }


def _percentile_rank(real_value: float, null_values: list[float], higher_is_better: bool) -> float | None:
    if not null_values:
        return None
    if higher_is_better:
        favorable = sum(1 for value in null_values if value <= real_value)
    else:
        favorable = sum(1 for value in null_values if value >= real_value)
    return favorable / len(null_values)


def _empirical_p_value(real_value: float, null_values: list[float], higher_is_better: bool) -> float | None:
    if not null_values:
        return None
    if higher_is_better:
        exceedances = sum(1 for value in null_values if value >= real_value)
    else:
        exceedances = sum(1 for value in null_values if value <= real_value)
    return (exceedances + 1) / (len(null_values) + 1)


def _with_labels(rows: list[ProcessedRow], labels: list[int]) -> list[ProcessedRow]:
    if len(rows) != len(labels):
        raise ValueError(f"rows length {len(rows)} != labels length {len(labels)}")
    return [
        ProcessedRow(
            timestamp=row.timestamp,
            symbol=row.symbol,
            timeframe=row.timeframe,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
            label=labels[index],
            label_reason=row.label_reason,
            horizon_end_timestamp=row.horizon_end_timestamp,
            features=row.features,
        )
        for index, row in enumerate(rows)
    ]


def _extract_fold_returns(walkforward: dict[str, object]) -> list[float]:
    payload = walkforward.get("fold_summaries", [])
    if not isinstance(payload, list):
        return []
    returns: list[float] = []
    for item in payload:
        if not isinstance(item, dict) or item.get("skipped"):
            continue
        value = item.get("return_pct", 0.0)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            returns.append(float(value) / 100.0)
    return returns


def run_permutation_significance(
    *,
    rows: list[ProcessedRow],
    evaluator: Evaluator,
    trials: int,
    seed: int,
    metric_name: str,
    higher_is_better: bool,
    minimum_valid_folds: int,
) -> dict[str, object]:
    real_result = evaluator(rows)
    if real_result.valid_folds < minimum_valid_folds:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": (
                f"real_result.valid_folds={real_result.valid_folds} < "
                f"minimum_valid_folds={minimum_valid_folds}"
            ),
            "metric_name": metric_name,
            "higher_is_better": higher_is_better,
            "trials_requested": trials,
            "trials_completed": 0,
            "seed": seed,
            "real_result": {
                "metric_value": real_result.metric_value,
                "valid_folds": real_result.valid_folds,
                "skipped_folds": real_result.skipped_folds,
                "aggregate": real_result.aggregate,
                "returns": real_result.returns,
                "sharpe_ratio": real_result.sharpe_ratio,
            },
            "trial_results": [],
            "null_distribution_summary": _summarize_distribution([]),
            "trial_sharpe_ratio_summary": _summarize_distribution([]),
            "deflated_sharpe_ratio": {
                "status": "skipped",
                "reason": "real_result_below_minimum_valid_folds",
            },
            "p_value": None,
            "percentile_rank": None,
        }

    rng = random.Random(seed)
    original_labels = [row.label for row in rows]
    trial_results: list[TrialResult] = []
    null_values: list[float] = []
    trial_sharpes: list[float] = []

    for trial_index in range(trials):
        shuffled_labels = original_labels.copy()
        rng.shuffle(shuffled_labels)
        permuted_rows = _with_labels(rows, shuffled_labels)
        result = evaluator(permuted_rows)
        trial_results.append(
            TrialResult(
                trial_index=trial_index,
                metric_value=result.metric_value,
                valid_folds=result.valid_folds,
                skipped_folds=result.skipped_folds,
                sharpe_ratio=result.sharpe_ratio,
            )
        )
        if result.valid_folds >= minimum_valid_folds:
            null_values.append(result.metric_value)
            if result.sharpe_ratio is not None:
                trial_sharpes.append(result.sharpe_ratio)

    percentile_rank = _percentile_rank(real_result.metric_value, null_values, higher_is_better)
    p_value = _empirical_p_value(real_result.metric_value, null_values, higher_is_better)
    dsr_payload = compute_deflated_sharpe_ratio(real_result.returns, trial_sharpes)
    return {
        "enabled": True,
        "status": "completed",
        "metric_name": metric_name,
        "higher_is_better": higher_is_better,
        "trials_requested": trials,
        "trials_completed": len(trial_results),
        "trials_used_in_null_distribution": len(null_values),
        "seed": seed,
        "minimum_valid_folds": minimum_valid_folds,
        "permutation_method": "label_shuffle",
        "real_result": {
            "metric_value": real_result.metric_value,
            "valid_folds": real_result.valid_folds,
            "skipped_folds": real_result.skipped_folds,
            "aggregate": real_result.aggregate,
            "returns": real_result.returns,
            "sharpe_ratio": real_result.sharpe_ratio,
        },
        "trial_results": [
            {
                "trial_index": item.trial_index,
                "metric_value": item.metric_value,
                "valid_folds": item.valid_folds,
                "skipped_folds": item.skipped_folds,
                "sharpe_ratio": item.sharpe_ratio,
            }
            for item in trial_results
        ],
        "null_distribution_summary": _summarize_distribution(null_values),
        "trial_sharpe_ratio_summary": _summarize_distribution(trial_sharpes),
        "deflated_sharpe_ratio": dsr_payload,
        "p_value": p_value,
        "percentile_rank": percentile_rank,
        "is_significant_at_95pct": p_value is not None and p_value <= 0.05,
    }


def run_walkforward_permutation_significance(
    *,
    rows: list[ProcessedRow],
    feature_names: list[str],
    settings: Settings,
    logger: Any,
) -> dict[str, object]:
    metric_name = settings.significance.metric_name

    def evaluator(candidate_rows: list[ProcessedRow]) -> EvaluationResult:
        walkforward = run_walkforward_economic_backtest(
            rows=candidate_rows,
            feature_names=feature_names,
            settings=settings,
            run_dir=settings.data.runs_dir,
            logger=logger,
            persist_artifacts=False,
        )
        aggregate = walkforward["aggregate"]
        if not isinstance(aggregate, dict):
            raise ValueError("walk-forward aggregate payload must be a dict")
        returns = _extract_fold_returns(walkforward)
        return EvaluationResult(
            metric_value=_metric_float(aggregate, metric_name),
            aggregate=aggregate,
            valid_folds=int(walkforward["valid_folds"]),
            skipped_folds=int(walkforward["skipped_folds"]),
            returns=returns,
            sharpe_ratio=compute_sharpe_ratio(returns),
        )

    report = run_permutation_significance(
        rows=rows,
        evaluator=evaluator,
        trials=settings.significance.trials,
        seed=settings.significance.seed,
        metric_name=metric_name,
        higher_is_better=settings.significance.higher_is_better,
        minimum_valid_folds=settings.significance.minimum_valid_folds,
    )
    report["evaluation_mode"] = "walk_forward"
    return report
