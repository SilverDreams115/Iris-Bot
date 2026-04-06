from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True)
class FeatureMatrix:
    rows: list[list[float]]
    labels: list[int]
    timestamps: list[str]
    feature_names: list[str]


@dataclass
class ManualStandardScaler:
    means: list[float] | None = None
    stds: list[float] | None = None

    def fit(self, rows: list[list[float]]) -> None:
        if not rows:
            self.means = []
            self.stds = []
            return
        width = len(rows[0])
        means: list[float] = []
        stds: list[float] = []
        for column_index in range(width):
            values = [row[column_index] for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = sqrt(variance) if variance > 0.0 else 1.0
            means.append(mean)
            stds.append(std)
        self.means = means
        self.stds = stds

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        if self.means is None or self.stds is None:
            raise RuntimeError("Scaler must be fitted before transform")
        transformed: list[list[float]] = []
        for row in rows:
            transformed.append(
                [
                    0.0 if std == 0.0 else (value - mean) / std
                    for value, mean, std in zip(row, self.means, self.stds, strict=False)
                ]
            )
        return transformed


def validate_feature_rows(rows: list[list[float]]) -> None:
    for row in rows:
        for value in row:
            if not isfinite(value):
                raise ValueError("Se detecto un valor no finito en la matriz de features")
