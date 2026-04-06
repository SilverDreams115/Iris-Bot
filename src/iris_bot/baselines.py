from __future__ import annotations


class WeightedMomentumBaseline:
    """
    Directional baseline using a weighted combination of normalized momentum features.

    Uses three time-scale momentum signals (return_1, momentum_3, momentum_5) weighted
    toward the shorter horizon, which tends to predict next-bar direction more reliably
    than a single raw value.  All three features are already normalized fractional returns
    so they are cross-pair comparable without further scaling.

    Score > 0 → bullish bias, Score < 0 → bearish bias.
    """

    _WEIGHTS = (0.5, 0.3, 0.2)  # return_1, momentum_3, momentum_5

    def score(self, feature_rows: list[dict[str, float]]) -> list[float]:
        w1, w3, w5 = self._WEIGHTS
        return [
            w1 * row.get("return_1", 0.0)
            + w3 * row.get("momentum_3", 0.0)
            + w5 * row.get("momentum_5", 0.0)
            for row in feature_rows
        ]


# Back-compat alias — experiments.py and symbol_research.py import this name.
MomentumSignBaseline = WeightedMomentumBaseline
