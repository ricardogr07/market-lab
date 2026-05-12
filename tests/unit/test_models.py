from __future__ import annotations

import pandas as pd
import pytest

from marketlab.models import (
    build_model_estimator,
    predict_allocation_utility_scores,
    predict_direction_scores,
    predict_regime_state_scores,
    supported_model_names,
)

FEATURES = pd.DataFrame(
    {
        "feature_a": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "feature_b": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    }
)
TARGET = pd.Series([0, 0, 0, 1, 1, 1], dtype=int)


def test_supported_model_names_cover_lightweight_baseline_set() -> None:
    assert supported_model_names() == (
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
        "logistic_l1",
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


@pytest.mark.parametrize("model_name", supported_model_names())
def test_allocation_utility_models_fit_and_score_expected_allocation(
    model_name: str,
) -> None:
    target = pd.Series([0, 1, 1, 2, 3, 3], dtype=int)
    definition, estimator = build_model_estimator(model_name, "allocation_utility")
    estimator.fit(FEATURES, target)
    scores, probabilities = predict_allocation_utility_scores(estimator, FEATURES)

    assert definition.name == model_name
    assert definition.score_column == "score"
    assert len(scores) == len(FEATURES)
    assert scores.between(0.0, 1.0).all()
    assert list(probabilities.columns) == [
        "prob_tier_0",
        "prob_tier_25",
        "prob_tier_50",
        "prob_tier_100",
    ]
    assert probabilities.sum(axis=1).round(6).eq(1.0).all()


@pytest.mark.parametrize("model_name", supported_model_names())
def test_regime_state_models_fit_and_score_expected_allocation(
    model_name: str,
) -> None:
    target = pd.Series([0, 0, 1, 1, 2, 2], dtype=int)
    definition, estimator = build_model_estimator(model_name, "regime_state")
    estimator.fit(FEATURES, target)
    scores, probabilities = predict_regime_state_scores(estimator, FEATURES)

    assert definition.name == model_name
    assert len(scores) == len(FEATURES)
    assert scores.between(0.0, 1.0).all()
    assert list(probabilities.columns) == [
        "prob_tier_0",
        "prob_tier_25",
        "prob_tier_50",
        "prob_tier_100",
    ]
    assert probabilities.sum(axis=1).round(6).eq(1.0).all()


def test_unknown_model_name_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported model 'svm'"):
        build_model_estimator("svm", "direction")


def test_non_classifier_target_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="direction'.*allocation_utility'.*regime_state"):
        build_model_estimator("logistic_regression", "return")
