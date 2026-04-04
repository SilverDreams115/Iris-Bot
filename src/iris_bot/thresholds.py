from __future__ import annotations

from dataclasses import dataclass

from iris_bot.metrics import classification_metrics


@dataclass(frozen=True)
class ThresholdSelectionResult:
    threshold: float
    metric_name: str
    metric_value: float


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
) -> ThresholdSelectionResult:
    best_result = ThresholdSelectionResult(threshold=grid[0], metric_name=metric_name, metric_value=-1.0)
    for threshold in grid:
        predictions = apply_score_threshold(scores, threshold)
        metrics = classification_metrics(labels, predictions)
        metric_value = float(metrics[metric_name])
        if metric_value > best_result.metric_value:
            best_result = ThresholdSelectionResult(threshold=threshold, metric_name=metric_name, metric_value=metric_value)
    return best_result


def select_threshold_from_probabilities(
    probabilities: list[dict[int, float]],
    labels: list[int],
    grid: tuple[float, ...],
    metric_name: str,
) -> ThresholdSelectionResult:
    best_result = ThresholdSelectionResult(threshold=grid[0], metric_name=metric_name, metric_value=-1.0)
    for threshold in grid:
        predictions = apply_probability_threshold(probabilities, threshold)
        metrics = classification_metrics(labels, predictions)
        metric_value = float(metrics[metric_name])
        if metric_value > best_result.metric_value:
            best_result = ThresholdSelectionResult(threshold=threshold, metric_name=metric_name, metric_value=metric_value)
    return best_result
