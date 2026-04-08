from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, cast

from iris_bot.config import XGBoostConfig
from iris_bot.durable_io import durable_write_json


LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {value: key for key, value in LABEL_TO_CLASS.items()}


@dataclass(frozen=True)
class XGBoostModelMetadata:
    feature_names: list[str]
    num_boost_round: int
    best_iteration: int | None
    best_score: float | None
    classes: dict[str, int]
    class_weights: dict[str, float]
    probability_calibration: dict[str, Any]
    feature_importance: dict[str, float]


class XGBoostMultiClassModel:
    def __init__(self, config: XGBoostConfig) -> None:
        self.config = config
        self.booster: Any | None = None
        self.best_iteration: int | None = None
        self.best_score: float | None = None
        self._feature_names: list[str] = []
        self.class_weights: dict[int, float] = {}
        self.calibration_temperature: float = 1.0
        self.calibration_method: str = config.probability_calibration_method
        self.calibration_class_temperatures: dict[int, float] = {}
        self.calibration_validation_log_loss_before: float | None = None
        self.calibration_validation_log_loss_after: float | None = None
        self.calibration_comparison: dict[str, Any] = {}

    def _temperature_grid(self) -> list[float]:
        step = self.config.calibration_temperature_step
        minimum = self.config.calibration_min_temperature
        maximum = self.config.calibration_max_temperature
        if step <= 0.0 or minimum <= 0.0 or maximum < minimum:
            return [1.0]
        count = int(round((maximum - minimum) / step))
        grid = [minimum + step * index for index in range(count + 1)]
        if 1.0 not in grid and minimum <= 1.0 <= maximum:
            grid.append(1.0)
        return sorted(set(round(item, 10) for item in grid if item > 0.0))

    def _normalize_probabilities(self, rows: Any) -> list[list[float]]:
        normalized: list[list[float]] = []
        for row in rows:
            normalized.append([float(value) for value in row])
        return normalized

    def _apply_temperature_to_row(self, row: list[float], temperature: float) -> list[float]:
        if temperature == 1.0:
            return row
        safe_probs = [min(max(value, 1e-12), 1.0) for value in row]
        logits = [math.log(value) / temperature for value in safe_probs]
        max_logit = max(logits)
        exps = [math.exp(value - max_logit) for value in logits]
        denom = sum(exps)
        if denom <= 0.0:
            return row
        return [value / denom for value in exps]

    def _apply_classwise_temperature_to_row(self, row: list[float], temperatures: dict[int, float]) -> list[float]:
        if not temperatures:
            return row
        safe_probs = [min(max(value, 1e-12), 1.0) for value in row]
        logits = [math.log(value) / temperatures.get(index, 1.0) for index, value in enumerate(safe_probs)]
        max_logit = max(logits)
        exps = [math.exp(value - max_logit) for value in logits]
        denom = sum(exps)
        if denom <= 0.0:
            return row
        return [value / denom for value in exps]

    def _apply_calibration_to_row(self, row: list[float]) -> list[float]:
        if self.calibration_method == "classwise_temperature":
            return self._apply_classwise_temperature_to_row(row, self.calibration_class_temperatures)
        return self._apply_temperature_to_row(row, self.calibration_temperature)

    def _resolved_calibration_method(self, best_method: str) -> str:
        configured = self.config.probability_calibration_method
        if configured == "auto":
            return best_method
        if configured == "classwise_temperature":
            return "classwise_temperature"
        return "global_temperature"

    def _multiclass_log_loss(self, probabilities: list[list[float]], labels: list[int]) -> float:
        if not probabilities or not labels or len(probabilities) != len(labels):
            return float("inf")
        total = 0.0
        for row, label in zip(probabilities, labels, strict=False):
            klass = LABEL_TO_CLASS[label]
            prob = min(max(row[klass], 1e-12), 1.0)
            total -= math.log(prob)
        return total / len(labels)

    def _fit_global_temperature_calibration(
        self,
        validation_probabilities: list[list[float]],
        validation_labels: list[int],
    ) -> float:
        best_temperature = 1.0
        best_loss = self._multiclass_log_loss(validation_probabilities, validation_labels)
        for temperature in self._temperature_grid():
            calibrated = [self._apply_temperature_to_row(row, temperature) for row in validation_probabilities]
            loss = self._multiclass_log_loss(calibrated, validation_labels)
            if loss < best_loss:
                best_loss = loss
                best_temperature = temperature
        return best_temperature

    def _fit_classwise_temperature_calibration(
        self,
        validation_probabilities: list[list[float]],
        validation_labels: list[int],
    ) -> dict[int, float]:
        temperatures = {index: 1.0 for index in range(len(LABEL_TO_CLASS))}
        best_loss = self._multiclass_log_loss(validation_probabilities, validation_labels)
        for class_index in range(len(LABEL_TO_CLASS)):
            best_class_temperature = temperatures[class_index]
            for temperature in self._temperature_grid():
                candidate = dict(temperatures)
                candidate[class_index] = temperature
                calibrated = [
                    self._apply_classwise_temperature_to_row(row, candidate) for row in validation_probabilities
                ]
                loss = self._multiclass_log_loss(calibrated, validation_labels)
                if loss < best_loss:
                    best_loss = loss
                    best_class_temperature = temperature
            temperatures[class_index] = best_class_temperature
        return temperatures

    def _fit_probability_calibration(
        self,
        validation_probabilities: list[list[float]],
        validation_labels: list[int],
    ) -> None:
        if (
            not self.config.use_probability_calibration
            or len(validation_probabilities) == 0
            or len(validation_labels) == 0
        ):
            self.calibration_method = self.config.probability_calibration_method
            self.calibration_temperature = 1.0
            self.calibration_class_temperatures = {}
            self.calibration_validation_log_loss_before = None
            self.calibration_validation_log_loss_after = None
            self.calibration_comparison = {}
            return

        baseline_loss = self._multiclass_log_loss(
            validation_probabilities,
            validation_labels,
        )
        global_temperature = self._fit_global_temperature_calibration(
            validation_probabilities,
            validation_labels,
        )
        global_calibrated = [
            self._apply_temperature_to_row(row, global_temperature) for row in validation_probabilities
        ]
        global_loss = self._multiclass_log_loss(global_calibrated, validation_labels)
        class_temperatures = self._fit_classwise_temperature_calibration(
            validation_probabilities,
            validation_labels,
        )
        classwise_calibrated = [
            self._apply_classwise_temperature_to_row(row, class_temperatures) for row in validation_probabilities
        ]
        classwise_loss = self._multiclass_log_loss(classwise_calibrated, validation_labels)
        losses = {
            "uncalibrated": baseline_loss,
            "global_temperature": global_loss,
            "classwise_temperature": classwise_loss,
        }
        best_method = min(losses, key=losses.__getitem__)

        self.calibration_method = self._resolved_calibration_method(best_method)
        self.calibration_validation_log_loss_before = baseline_loss
        self.calibration_temperature = 1.0
        self.calibration_class_temperatures = {}

        if self.calibration_method == "classwise_temperature":
            self.calibration_class_temperatures = class_temperatures
            calibrated = classwise_calibrated
        elif self.calibration_method == "uncalibrated":
            calibrated = validation_probabilities
        else:
            self.calibration_method = "global_temperature"
            self.calibration_temperature = global_temperature
            calibrated = global_calibrated

        self.calibration_validation_log_loss_after = self._multiclass_log_loss(calibrated, validation_labels)
        self.calibration_comparison = {
            "validation_log_loss_by_method": losses,
            "best_method": best_method,
            "best_validation_log_loss": losses[best_method],
            "applied_method": self.calibration_method,
            "configured_method": self.config.probability_calibration_method,
            "applied_matches_best": self.calibration_method == best_method,
        }

    def _compute_class_weights(self, train_labels: list[int]) -> dict[int, float]:
        if not self.config.use_class_weights or not train_labels:
            return {}
        counts: dict[int, int] = {}
        for label in train_labels:
            counts[label] = counts.get(label, 0) + 1
        max_count = max(counts.values())
        weights: dict[int, float] = {}
        for label, count in counts.items():
            if count <= 0:
                continue
            raw_weight = max_count / count
            weights[label] = min(raw_weight, self.config.class_weight_max_multiplier)
        return weights

    def fit(
        self,
        train_rows: list[list[float]],
        train_labels: list[int],
        validation_rows: list[list[float]],
        validation_labels: list[int],
        feature_names: list[str] | None = None,
        sample_weights: list[float] | None = None,
    ) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed") from exc

        if not train_rows:
            raise ValueError("train_rows is empty")
        if len(train_rows) != len(train_labels):
            raise ValueError(
                f"train_rows length {len(train_rows)} != train_labels length {len(train_labels)}"
            )

        mapped_train = [LABEL_TO_CLASS[label] for label in train_labels]
        mapped_valid = [LABEL_TO_CLASS[label] for label in validation_labels]
        self.class_weights = self._compute_class_weights(train_labels)
        train_weights = [
            self.class_weights.get(label, 1.0) * (sample_weights[i] if sample_weights else 1.0)
            for i, label in enumerate(train_labels)
        ]

        dtrain = xgb.DMatrix(train_rows, label=mapped_train, feature_names=feature_names, weight=train_weights)
        dvalid = xgb.DMatrix(validation_rows, label=mapped_valid, feature_names=feature_names)

        params = {
            "objective": "multi:softprob",
            "num_class": len(LABEL_TO_CLASS),
            "eta": self.config.eta,
            "max_depth": self.config.max_depth,
            "min_child_weight": self.config.min_child_weight,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "lambda": self.config.reg_lambda,
            "alpha": self.config.reg_alpha,
            "seed": self.config.seed,
            "eval_metric": "mlogloss",
        }

        callbacks = [
            xgb.callback.EarlyStopping(
                rounds=self.config.early_stopping_rounds,
                save_best=True,
                maximize=False,
                data_name="validation",
            )
        ]

        self.booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.config.num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "validation")],
            callbacks=callbacks,
            verbose_eval=False,
        )
        self.best_iteration = getattr(self.booster, "best_iteration", None)
        self.best_score = getattr(self.booster, "best_score", None)
        self._feature_names = feature_names or []
        validation_probabilities = self._normalize_probabilities(self.booster.predict(dvalid))
        self._fit_probability_calibration(validation_probabilities, validation_labels)

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """Returns per-feature importance scores (gain by default)."""
        if self.booster is None:
            return {}
        try:
            return cast(dict[str, float], self.booster.get_score(importance_type=importance_type))
        except Exception:
            return {}

    def probability_calibration_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.config.use_probability_calibration,
            "method": self.calibration_method,
            "temperature": self.calibration_temperature,
            "class_temperatures": {
                str(CLASS_TO_LABEL[index]): value for index, value in self.calibration_class_temperatures.items()
            },
            "validation_log_loss_before": self.calibration_validation_log_loss_before,
            "validation_log_loss_after": self.calibration_validation_log_loss_after,
            "comparison": self.calibration_comparison,
        }

    def predict_probabilities(self, rows: list[list[float]]) -> list[dict[int, float]]:
        if self.booster is None:
            raise RuntimeError("XGBoost model has not been trained")
        import xgboost as xgb

        matrix = xgb.DMatrix(rows, feature_names=self._feature_names or None)
        probabilities = self._normalize_probabilities(self.booster.predict(matrix))
        results: list[dict[int, float]] = []
        for row in probabilities:
            calibrated = self._apply_calibration_to_row(row)
            results.append({CLASS_TO_LABEL[index]: float(value) for index, value in enumerate(calibrated)})
        return results

    def save(self, model_path: Path, metadata_path: Path, feature_names: list[str]) -> None:
        if self.booster is None:
            raise RuntimeError("No model to save")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(model_path)
        metadata = XGBoostModelMetadata(
            feature_names=feature_names,
            num_boost_round=self.config.num_boost_round,
            best_iteration=self.best_iteration,
            best_score=self.best_score,
            classes={str(label): klass for label, klass in LABEL_TO_CLASS.items()},
            class_weights={str(label): weight for label, weight in self.class_weights.items()},
            probability_calibration=self.probability_calibration_metadata(),
            feature_importance=self.feature_importance(),
        )
        durable_write_json(metadata_path, asdict(metadata))

    def load(self, model_path: Path) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed") from exc
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        booster = xgb.Booster()
        booster.load_model(model_path)
        self.booster = booster
        # Restore feature names from metadata if available
        metadata_path = model_path.with_name("xgboost_metadata.json")
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                self._feature_names = meta.get("feature_names", [])
                calibration = meta.get("probability_calibration", {})
                if isinstance(calibration, dict):
                    self.calibration_method = str(
                        calibration.get("method", self.config.probability_calibration_method)
                    )
                    self.calibration_temperature = float(calibration.get("temperature", 1.0) or 1.0)
                    class_temperatures = calibration.get("class_temperatures", {})
                    if isinstance(class_temperatures, dict):
                        self.calibration_class_temperatures = {
                            LABEL_TO_CLASS[int(label)]: float(value)
                            for label, value in class_temperatures.items()
                        }
                    before = calibration.get("validation_log_loss_before")
                    after = calibration.get("validation_log_loss_after")
                    self.calibration_validation_log_loss_before = float(before) if before is not None else None
                    self.calibration_validation_log_loss_after = float(after) if after is not None else None
                    comparison = calibration.get("comparison", {})
                    self.calibration_comparison = comparison if isinstance(comparison, dict) else {}
            except (OSError, json.JSONDecodeError, KeyError):
                pass
