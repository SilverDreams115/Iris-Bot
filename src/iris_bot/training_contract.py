from __future__ import annotations

from statistics import mean
from typing import Any

from iris_bot.config import Settings
from iris_bot.contract_versions import TRAINING_CONTRACT_VERSION, contract_fingerprint
from iris_bot.processed_dataset import ProcessedRow


ECONOMIC_SAMPLE_WEIGHT_CAP = 3.0


def compute_economic_sample_weights(rows: list[ProcessedRow], cap: float = ECONOMIC_SAMPLE_WEIGHT_CAP) -> list[float]:
    """Weight train rows by ATR-relative economic significance."""
    atrs = [row.features.get("atr_5", 0.0) for row in rows]
    sorted_atrs = sorted(atrs)
    median_atr = sorted_atrs[len(sorted_atrs) // 2] if sorted_atrs else 0.0
    if median_atr <= 0.0:
        return [1.0] * len(rows)
    return [min(atr / median_atr, cap) for atr in atrs]


def summarize_economic_sample_weights(weights: list[float], cap: float = ECONOMIC_SAMPLE_WEIGHT_CAP) -> dict[str, Any]:
    if not weights:
        return {
            "enabled": True,
            "cap": cap,
            "min": 1.0,
            "max": 1.0,
            "mean": 1.0,
            "count": 0,
        }
    return {
        "enabled": True,
        "cap": cap,
        "min": min(weights),
        "max": max(weights),
        "mean": mean(weights),
        "count": len(weights),
    }


def build_training_contract(
    settings: Settings,
    feature_names: list[str],
    *,
    use_primary_timeframe_only: bool,
    feature_source: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": TRAINING_CONTRACT_VERSION,
        "feature_ordering": {
            "source": feature_source,
            "feature_names": feature_names,
        },
        "dataset_assumptions": {
            "use_primary_timeframe_only": use_primary_timeframe_only,
            "primary_timeframe": settings.trading.primary_timeframe,
            "label_mode": settings.labeling.mode,
        },
        "economic_sample_weighting": {
            "enabled": True,
            "method": "atr_5_relative_to_median",
            "cap": ECONOMIC_SAMPLE_WEIGHT_CAP,
        },
        "class_weighting": {
            "enabled": settings.xgboost.use_class_weights,
            "max_multiplier": settings.xgboost.class_weight_max_multiplier,
        },
        "probability_calibration": {
            "enabled": settings.xgboost.use_probability_calibration,
            "configured_method": settings.xgboost.probability_calibration_method,
            "temperature_range": {
                "min": settings.xgboost.calibration_min_temperature,
                "max": settings.xgboost.calibration_max_temperature,
                "step": settings.xgboost.calibration_temperature_step,
            },
        },
        "model": {
            "family": "xgboost_multiclass",
            "num_boost_round": settings.xgboost.num_boost_round,
            "early_stopping_rounds": settings.xgboost.early_stopping_rounds,
            "eta": settings.xgboost.eta,
            "max_depth": settings.xgboost.max_depth,
            "min_child_weight": settings.xgboost.min_child_weight,
            "subsample": settings.xgboost.subsample,
            "colsample_bytree": settings.xgboost.colsample_bytree,
            "reg_lambda": settings.xgboost.reg_lambda,
            "reg_alpha": settings.xgboost.reg_alpha,
            "seed": settings.xgboost.seed,
        },
    }
    payload["contract_hash"] = contract_fingerprint(payload)
    return payload
