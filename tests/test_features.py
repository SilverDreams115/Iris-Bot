"""
Tests for preprocessing / feature utilities.
Restores coverage lost when test_features.py was deleted.
"""
from __future__ import annotations

import math

import pytest

from iris_bot.preprocessing import ManualStandardScaler, validate_feature_rows


# ---------------------------------------------------------------------------
# ManualStandardScaler
# ---------------------------------------------------------------------------

def test_scaler_normalises_known_values() -> None:
    rows = [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]
    scaler = ManualStandardScaler()
    scaler.fit(rows)
    transformed = scaler.transform(rows)

    # middle value of each column should transform to 0.0
    assert transformed[1][0] == pytest.approx(0.0, abs=1e-9)
    assert transformed[1][1] == pytest.approx(0.0, abs=1e-9)


def test_scaler_zero_variance_column_does_not_divide_by_zero() -> None:
    rows = [[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]
    scaler = ManualStandardScaler()
    scaler.fit(rows)
    transformed = scaler.transform(rows)

    # constant column → mean=5, std→1 (fallback), result = (5-5)/1 = 0
    assert all(row[0] == pytest.approx(0.0) for row in transformed)


def test_scaler_single_row_does_not_crash() -> None:
    scaler = ManualStandardScaler()
    scaler.fit([[1.0, 2.0, 3.0]])
    result = scaler.transform([[1.0, 2.0, 3.0]])
    assert len(result) == 1
    assert all(math.isfinite(v) for v in result[0])


def test_scaler_empty_fit_produces_empty_transform() -> None:
    scaler = ManualStandardScaler()
    scaler.fit([])
    result = scaler.transform([])
    assert result == []


def test_scaler_transform_before_fit_raises() -> None:
    scaler = ManualStandardScaler()
    with pytest.raises(RuntimeError):
        scaler.transform([[1.0, 2.0]])


def test_scaler_fit_then_transform_new_rows() -> None:
    """Scaler fitted on training data should transform held-out rows correctly."""
    train = [[0.0], [2.0], [4.0]]  # mean=2, std=sqrt(8/3)≈1.633
    scaler = ManualStandardScaler()
    scaler.fit(train)

    out_of_sample = scaler.transform([[2.0]])  # value == mean → z=0
    assert out_of_sample[0][0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# validate_feature_rows
# ---------------------------------------------------------------------------

def test_validate_feature_rows_accepts_finite_values() -> None:
    validate_feature_rows([[1.0, 2.0], [3.0, -4.0]])  # should not raise


def test_validate_feature_rows_rejects_nan() -> None:
    with pytest.raises(ValueError):
        validate_feature_rows([[1.0, float("nan")]])


def test_validate_feature_rows_rejects_positive_inf() -> None:
    with pytest.raises(ValueError):
        validate_feature_rows([[float("inf"), 1.0]])


def test_validate_feature_rows_rejects_negative_inf() -> None:
    with pytest.raises(ValueError):
        validate_feature_rows([[1.0, float("-inf")]])


def test_validate_feature_rows_empty_is_valid() -> None:
    validate_feature_rows([])  # no rows → nothing to validate
