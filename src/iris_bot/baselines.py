from __future__ import annotations


class MomentumSignBaseline:
    """Benchmark minimo: usa momentum reciente como score direccional."""

    def score(self, feature_rows: list[dict[str, float]]) -> list[float]:
        return [row["momentum_3"] for row in feature_rows]
