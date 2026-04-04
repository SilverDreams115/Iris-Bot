from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from iris_bot.config import XGBoostConfig


LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {value: key for key, value in LABEL_TO_CLASS.items()}


@dataclass(frozen=True)
class XGBoostModelMetadata:
    feature_names: list[str]
    num_boost_round: int
    best_iteration: int | None
    classes: dict[str, int]


class XGBoostMultiClassModel:
    def __init__(self, config: XGBoostConfig) -> None:
        self.config = config
        self.booster = None
        self.best_iteration: int | None = None

    def fit(
        self,
        train_rows: list[list[float]],
        train_labels: list[int],
        validation_rows: list[list[float]],
        validation_labels: list[int],
    ) -> None:
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("xgboost no esta instalado") from exc

        dtrain = xgb.DMatrix(train_rows, label=[LABEL_TO_CLASS[label] for label in train_labels])
        dvalid = xgb.DMatrix(validation_rows, label=[LABEL_TO_CLASS[label] for label in validation_labels])
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
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
        self.booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.config.num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "validation")],
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=False,
        )
        self.best_iteration = getattr(self.booster, "best_iteration", None)

    def predict_probabilities(self, rows: list[list[float]]) -> list[dict[int, float]]:
        if self.booster is None:
            raise RuntimeError("El modelo XGBoost no ha sido entrenado")
        import xgboost as xgb  # type: ignore

        matrix = xgb.DMatrix(rows)
        probabilities = self.booster.predict(matrix)
        results: list[dict[int, float]] = []
        for row in probabilities:
            results.append({CLASS_TO_LABEL[index]: float(value) for index, value in enumerate(row)})
        return results

    def save(self, model_path: Path, metadata_path: Path, feature_names: list[str]) -> None:
        if self.booster is None:
            raise RuntimeError("No hay modelo para guardar")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(model_path)
        metadata = XGBoostModelMetadata(
            feature_names=feature_names,
            num_boost_round=self.config.num_boost_round,
            best_iteration=self.best_iteration,
            classes={str(label): klass for label, klass in LABEL_TO_CLASS.items()},
        )
        metadata_path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True), encoding="utf-8")

    def load(self, model_path: Path) -> None:
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("xgboost no esta instalado") from exc
        booster = xgb.Booster()
        booster.load_model(model_path)
        self.booster = booster
