from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALLOCATION_UTILITY_CLASS_WEIGHTS = {
    0: 0.0,
    1: 0.25,
    2: 0.50,
    3: 1.0,
}
REGIME_STATE_CLASS_WEIGHTS = {
    0: 0.0,
    1: 0.50,
    2: 1.0,
}


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


def _uses_logistic_l1_ratio_api() -> bool:
    penalty = inspect.signature(LogisticRegression).parameters.get("penalty")
    return penalty is not None and penalty.default == "deprecated"


def _logistic_l1_parameters(*, solver: str, max_iter: int) -> dict[str, object]:
    parameters: dict[str, object] = {
        "solver": solver,
        "max_iter": max_iter,
        "random_state": 7,
    }
    if _uses_logistic_l1_ratio_api():
        parameters["l1_ratio"] = 1.0
    else:
        parameters["penalty"] = "l1"
    return parameters


def _parallel_n_jobs() -> int:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    return -1


def _logistic_l1() -> ClassifierMixin:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(**_logistic_l1_parameters(solver="liblinear", max_iter=1000)),
    )


def _logistic_l1_multiclass() -> ClassifierMixin:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(**_logistic_l1_parameters(solver="saga", max_iter=2000)),
    )


def _random_forest() -> ClassifierMixin:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        n_jobs=_parallel_n_jobs(),
        random_state=7,
    )


def _extra_trees() -> ClassifierMixin:
    return ExtraTreesClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        n_jobs=_parallel_n_jobs(),
        random_state=7,
    )


def _gradient_boosting() -> ClassifierMixin:
    return GradientBoostingClassifier(random_state=7)


def _hist_gradient_boosting() -> ClassifierMixin:
    return HistGradientBoostingClassifier(random_state=7)


MODEL_REGISTRY: dict[str, ModelDefinition] = {
    "extra_trees": ModelDefinition(
        name="extra_trees",
        estimator_label="ExtraTreesClassifier",
        builder=_extra_trees,
    ),
    "logistic_l1": ModelDefinition(
        name="logistic_l1",
        estimator_label="LogisticRegression",
        builder=_logistic_l1,
    ),
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
    "hist_gradient_boosting": ModelDefinition(
        name="hist_gradient_boosting",
        estimator_label="HistGradientBoostingClassifier",
        builder=_hist_gradient_boosting,
    ),
}


def supported_model_names() -> tuple[str, ...]:
    return tuple(sorted(MODEL_REGISTRY))


def build_model_estimator(
    model_name: str,
    target_type: str,
) -> tuple[ModelDefinition, ClassifierMixin]:
    if target_type not in {"allocation_utility", "direction", "regime_state"}:
        raise ValueError(
            "train-models currently supports target.type='direction', "
            "'allocation_utility', or 'regime_state' only."
        )

    try:
        definition = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        supported = ", ".join(supported_model_names())
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {supported}"
        ) from exc

    if target_type in {"allocation_utility", "regime_state"} and model_name == "logistic_l1":
        return definition, _logistic_l1_multiclass()
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


def predict_allocation_utility_scores(
    estimator: ClassifierMixin,
    features: pd.DataFrame,
    *,
    class_weight_map: dict[int, float] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Allocation utility models must expose predict_proba().")
    if not hasattr(estimator, "classes_"):
        raise TypeError("Allocation utility models must expose fitted classes_.")

    probabilities = estimator.predict_proba(features)
    classes = [int(value) for value in estimator.classes_]
    weights = class_weight_map or ALLOCATION_UTILITY_CLASS_WEIGHTS
    probability_frame = pd.DataFrame(index=features.index)
    expected_allocation = pd.Series(0.0, index=features.index, dtype=float)
    for class_index, class_label in enumerate(classes):
        tier_weight = weights.get(class_label, 0.0)
        class_probabilities = pd.Series(
            probabilities[:, class_index],
            index=features.index,
            dtype=float,
        )
        expected_allocation = expected_allocation.add(class_probabilities * tier_weight)
        suffix = str(int(tier_weight * 100))
        probability_frame[f"prob_tier_{suffix}"] = class_probabilities

    for tier_weight in ALLOCATION_UTILITY_CLASS_WEIGHTS.values():
        suffix = str(int(tier_weight * 100))
        column = f"prob_tier_{suffix}"
        if column not in probability_frame.columns:
            probability_frame[column] = 0.0

    probability_frame = probability_frame[
        ["prob_tier_0", "prob_tier_25", "prob_tier_50", "prob_tier_100"]
    ]
    return expected_allocation.rename("score"), probability_frame


def predict_regime_state_scores(
    estimator: ClassifierMixin,
    features: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    return predict_allocation_utility_scores(
        estimator,
        features,
        class_weight_map=REGIME_STATE_CLASS_WEIGHTS,
    )
