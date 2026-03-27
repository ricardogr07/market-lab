from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True, frozen=True)
class ModelDefinition:
    name: str
    estimator_label: str
    builder: Callable[[], ClassifierMixin]
    score_column: str = "score"


def _logistic_regression() -> ClassifierMixin:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=7,
        ),
    )


def _random_forest() -> ClassifierMixin:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=7,
    )


def _gradient_boosting() -> ClassifierMixin:
    return GradientBoostingClassifier(random_state=7)


MODEL_REGISTRY: dict[str, ModelDefinition] = {
    "logistic_regression": ModelDefinition(
        name="logistic_regression",
        estimator_label="LogisticRegression",
        builder=_logistic_regression,
    ),
    "random_forest": ModelDefinition(
        name="random_forest",
        estimator_label="RandomForestClassifier",
        builder=_random_forest,
    ),
    "gradient_boosting": ModelDefinition(
        name="gradient_boosting",
        estimator_label="GradientBoostingClassifier",
        builder=_gradient_boosting,
    ),
}


def supported_model_names() -> tuple[str, ...]:
    return tuple(sorted(MODEL_REGISTRY))


def build_model_estimator(
    model_name: str,
    target_type: str,
) -> tuple[ModelDefinition, ClassifierMixin]:
    if target_type != "direction":
        raise ValueError(
            "train-models currently supports target.type='direction' only."
        )

    try:
        definition = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        supported = ", ".join(supported_model_names())
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {supported}"
        ) from exc

    return definition, definition.builder()


def predict_direction_scores(
    estimator: ClassifierMixin,
    features: pd.DataFrame,
) -> pd.Series:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Direction models must expose predict_proba().")

    probabilities = estimator.predict_proba(features)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Direction model predict_proba() must include two classes.")

    return pd.Series(probabilities[:, 1], index=features.index, dtype=float, name="score")
