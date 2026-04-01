from __future__ import annotations

import math

from iris_bot.features import FeatureVector


class LogisticClassifier:
    """Clasificador ligero sin dependencias externas."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 250) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] = []
        self.bias: float = 0.0

    def fit(self, rows: list[FeatureVector]) -> None:
        if not rows:
            self.weights = []
            self.bias = 0.0
            return

        feature_count = len(rows[0].as_inputs())
        self.weights = [0.0] * feature_count
        self.bias = 0.0
        sample_count = len(rows)

        for _ in range(self.epochs):
            gradient_w = [0.0] * feature_count
            gradient_b = 0.0
            for row in rows:
                prediction = self.predict_proba(row)
                error = prediction - row.target
                for index, value in enumerate(row.as_inputs()):
                    gradient_w[index] += error * value
                gradient_b += error

            for index in range(feature_count):
                self.weights[index] -= self.learning_rate * gradient_w[index] / sample_count
            self.bias -= self.learning_rate * gradient_b / sample_count

    def predict_proba(self, row: FeatureVector) -> float:
        score = self.bias
        for weight, value in zip(self.weights, row.as_inputs(), strict=False):
            score += weight * value
        score = max(min(score, 35.0), -35.0)
        return 1.0 / (1.0 + math.exp(-score))
