from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from marketlab.config import ModelSpec
from marketlab.evaluation.walk_forward import WalkForwardFold, slice_fold_rows
from marketlab.models.evaluation import (
    MODEL_METRICS_COLUMNS,
    RANKING_DIAGNOSTICS_COLUMNS,
    build_ranking_diagnostics,
    classification_metrics,
    summarize_ranking_diagnostics,
)
from marketlab.models.registry import build_model_estimator, predict_direction_scores

MODELING_METADATA_COLUMNS = {
    "symbol",
    "signal_date",
    "effective_date",
    "target_end_date",
    "forward_return",
    "target",
}


@dataclass(slots=True)
class TrainingOutputs:
    manifest: pd.DataFrame
    metrics: pd.DataFrame
    predictions: pd.DataFrame | None
    ranking_diagnostics: pd.DataFrame


def modeling_feature_columns(modeling_dataset: pd.DataFrame) -> list[str]:
    feature_columns = [
        column
        for column in modeling_dataset.columns
        if column not in MODELING_METADATA_COLUMNS
    ]
    if not feature_columns:
        raise RuntimeError("Modeling dataset does not contain feature columns.")
    return feature_columns


def _prediction_frame(
    model_name: str,
    fold_id: int,
    test_rows: pd.DataFrame,
    score_series: pd.Series,
    predicted_target: pd.Series,
) -> pd.DataFrame:
    prediction_frame = test_rows.loc[
        :,
        [
            "symbol",
            "signal_date",
            "effective_date",
            "target_end_date",
            "forward_return",
            "target",
        ],
    ].copy()
    prediction_frame.insert(0, "fold_id", fold_id)
    prediction_frame.insert(0, "model_name", model_name)
    prediction_frame["score"] = score_series.to_numpy()
    prediction_frame["predicted_target"] = predicted_target.to_numpy()
    return prediction_frame


def train_direction_models_on_folds(
    modeling_dataset: pd.DataFrame,
    folds: list[WalkForwardFold],
    model_specs: list[ModelSpec],
    target_type: str,
    run_dir: Path,
    save_predictions: bool,
    *,
    long_n: int,
    short_n: int,
) -> TrainingOutputs:
    feature_columns = modeling_feature_columns(modeling_dataset)
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    ranking_diagnostics_frames: list[pd.DataFrame] = []

    for spec in model_specs:
        for fold in folds:
            train_rows, test_rows = slice_fold_rows(modeling_dataset, fold)
            if train_rows.empty or test_rows.empty:
                raise RuntimeError(
                    f"Fold {fold.fold_id} unexpectedly has empty train or test rows."
                )

            train_target = train_rows["target"].astype(int)
            if train_target.nunique() < 2:
                raise ValueError(
                    f"Fold {fold.fold_id} for model {spec.name} has only one target class."
                )

            definition, estimator = build_model_estimator(spec.name, target_type)
            estimator.fit(train_rows[feature_columns], train_target)

            score_series = predict_direction_scores(estimator, test_rows[feature_columns])
            predicted_target = pd.Series(
                estimator.predict(test_rows[feature_columns]),
                index=test_rows.index,
                name="predicted_target",
                dtype=int,
            )

            model_path = model_dir / f"{spec.name}_fold_{fold.fold_id}.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(estimator, handle)

            manifest_rows.append(
                {
                    "model_name": spec.name,
                    "estimator_label": definition.estimator_label,
                    "fold_id": fold.fold_id,
                    "model_path": model_path.relative_to(run_dir).as_posix(),
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "label_cutoff": fold.label_cutoff,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "train_rows": fold.train_rows,
                    "test_rows": fold.test_rows,
                }
            )

            fold_ranking_diagnostics = build_ranking_diagnostics(
                model_name=spec.name,
                fold_id=fold.fold_id,
                predictions=test_rows.assign(score=score_series.to_numpy()),
                long_n=long_n,
                short_n=short_n,
            )
            ranking_diagnostics_frames.append(fold_ranking_diagnostics)

            metrics_rows.append(
                {
                    "model_name": spec.name,
                    "fold_id": fold.fold_id,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "label_cutoff": fold.label_cutoff,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "train_rows": fold.train_rows,
                    "test_rows": fold.test_rows,
                    **classification_metrics(
                        truth=test_rows["target"].astype(int),
                        predicted=predicted_target,
                        scores=score_series,
                    ),
                    **summarize_ranking_diagnostics(fold_ranking_diagnostics),
                }
            )

            if save_predictions:
                prediction_frames.append(
                    _prediction_frame(
                        model_name=spec.name,
                        fold_id=fold.fold_id,
                        test_rows=test_rows,
                        score_series=score_series,
                        predicted_target=predicted_target,
                    )
                )

    manifest = pd.DataFrame(manifest_rows).sort_values(["model_name", "fold_id"]).reset_index(
        drop=True
    )
    if metrics_rows:
        metrics = pd.DataFrame(metrics_rows).loc[:, MODEL_METRICS_COLUMNS]
        metrics = metrics.sort_values(["model_name", "fold_id"]).reset_index(drop=True)
    else:
        metrics = pd.DataFrame(columns=MODEL_METRICS_COLUMNS)

    if ranking_diagnostics_frames:
        ranking_diagnostics = pd.concat(ranking_diagnostics_frames, ignore_index=True)
        ranking_diagnostics = ranking_diagnostics.loc[:, RANKING_DIAGNOSTICS_COLUMNS].sort_values(
            ["model_name", "fold_id", "signal_date"]
        ).reset_index(drop=True)
    else:
        ranking_diagnostics = pd.DataFrame(columns=RANKING_DIAGNOSTICS_COLUMNS)

    predictions = None
    if save_predictions:
        predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
            ["model_name", "fold_id", "signal_date", "symbol"]
        )

    return TrainingOutputs(
        manifest=manifest,
        metrics=metrics,
        predictions=predictions,
        ranking_diagnostics=ranking_diagnostics,
    )
