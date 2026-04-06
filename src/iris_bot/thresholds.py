from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from iris_bot.metrics import classification_metrics


@dataclass(frozen=True)
class ThresholdSelectionResult:
    threshold: float
    metric_name: str
    metric_value: float


def _metric_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _refined_grid(grid: tuple[float, ...], best_threshold: float, refinement_steps: int) -> tuple[float, ...]:
    """Build a fine-grained grid of `refinement_steps + 1` points around best_threshold.

    The span is ± one coarse grid step from best_threshold, clamped to [0.0, 1.0].
    Returns an empty tuple when refinement is disabled (refinement_steps <= 0) or the
    coarse grid has fewer than 2 points (no step size to infer).
    """
    if refinement_steps <= 0 or len(grid) < 2:
        return ()
    grid_step = min(abs(grid[i + 1] - grid[i]) for i in range(len(grid) - 1))
    lo = max(0.0, best_threshold - grid_step)
    hi = min(1.0, best_threshold + grid_step)
    step = (hi - lo) / refinement_steps
    return tuple(round(lo + i * step, 8) for i in range(refinement_steps + 1))


def apply_score_threshold(scores: list[float], threshold: float) -> list[int]:
    predictions: list[int] = []
    for score in scores:
        if score >= threshold:
            predictions.append(1)
        elif score <= -threshold:
            predictions.append(-1)
        else:
            predictions.append(0)
    return predictions


def apply_probability_threshold(probabilities: list[dict[int, float]], threshold: float) -> list[int]:
    predictions: list[int] = []
    for row in probabilities:
        long_score = row.get(1, 0.0)
        short_score = row.get(-1, 0.0)
        neutral_score = row.get(0, 0.0)
        if long_score >= threshold and long_score > short_score and long_score >= neutral_score:
            predictions.append(1)
        elif short_score >= threshold and short_score > long_score and short_score >= neutral_score:
            predictions.append(-1)
        else:
            predictions.append(0)
    return predictions


def select_threshold_from_scores(
    scores: list[float],
    labels: list[int],
    grid: tuple[float, ...],
    metric_name: str,
    refinement_steps: int = 0,
) -> ThresholdSelectionResult:
    best_result = ThresholdSelectionResult(threshold=grid[0], metric_name=metric_name, metric_value=-1.0)
    for threshold in grid:
        predictions = apply_score_threshold(scores, threshold)
        metrics = classification_metrics(labels, predictions)
        metric_value = _metric_float(metrics, metric_name, -1.0)
        if metric_value > best_result.metric_value:
            best_result = ThresholdSelectionResult(threshold=threshold, metric_name=metric_name, metric_value=metric_value)
    for threshold in _refined_grid(grid, best_result.threshold, refinement_steps):
        predictions = apply_score_threshold(scores, threshold)
        metrics = classification_metrics(labels, predictions)
        metric_value = _metric_float(metrics, metric_name, -1.0)
        if metric_value > best_result.metric_value:
            best_result = ThresholdSelectionResult(threshold=threshold, metric_name=metric_name, metric_value=metric_value)
    return best_result


def select_threshold_from_probabilities(
    probabilities: list[dict[int, float]],
    labels: list[int],
    grid: tuple[float, ...],
    metric_name: str,
    refinement_steps: int = 0,
) -> ThresholdSelectionResult:
    best_result = ThresholdSelectionResult(threshold=grid[0], metric_name=metric_name, metric_value=-1.0)
    for threshold in grid:
        predictions = apply_probability_threshold(probabilities, threshold)
        metrics = classification_metrics(labels, predictions)
        metric_value = _metric_float(metrics, metric_name, -1.0)
        if metric_value > best_result.metric_value:
            best_result = ThresholdSelectionResult(threshold=threshold, metric_name=metric_name, metric_value=metric_value)
    for threshold in _refined_grid(grid, best_result.threshold, refinement_steps):
        predictions = apply_probability_threshold(probabilities, threshold)
        metrics = classification_metrics(labels, predictions)
        metric_value = _metric_float(metrics, metric_name, -1.0)
        if metric_value > best_result.metric_value:
            best_result = ThresholdSelectionResult(threshold=threshold, metric_name=metric_name, metric_value=metric_value)
    return best_result
