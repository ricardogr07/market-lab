from __future__ import annotations

import pandas as pd
import pytest

from marketlab.models import (
    build_model_estimator,
    predict_direction_scores,
    supported_model_names,
)

FEATURES = pd.DataFrame(
    {
        "feature_a": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "feature_b": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    }
)
TARGET = pd.Series([0, 0, 0, 1, 1, 1], dtype=int)


def test_supported_model_names_cover_phase_two_defaults() -> None:
    assert supported_model_names() == (
        "gradient_boosting",
        "logistic_regression",
        "random_forest",
    )


@pytest.mark.parametrize("model_name", supported_model_names())
def test_direction_models_fit_and_score_probabilities(model_name: str) -> None:
    definition, estimator = build_model_estimator(model_name, "direction")
    estimator.fit(FEATURES, TARGET)
    scores = predict_direction_scores(estimator, FEATURES)

    assert definition.name == model_name
    assert definition.score_column == "score"
    assert len(scores) == len(FEATURES)
    assert scores.between(0.0, 1.0).all()


def test_unknown_model_name_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported model 'svm'"):
        build_model_estimator("svm", "direction")


def test_non_direction_target_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="target.type='direction' only"):
        build_model_estimator("logistic_regression", "return")
