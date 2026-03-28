from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from marketlab.backtest.engine import run_backtest
from marketlab.backtest.metrics import compute_strategy_metrics
from marketlab.config import ExperimentConfig
from marketlab.data.market import load_symbol_frames
from marketlab.data.panel import build_market_panel, load_panel_csv, save_panel_csv
from marketlab.evaluation import build_walk_forward_folds, folds_to_frame, slice_fold_rows
from marketlab.features.engineering import add_feature_set
from marketlab.models import train_direction_models_on_folds
from marketlab.rebalance import next_rebalance_effective_date
from marketlab.reports.markdown import write_markdown_report
from marketlab.reports.plots import plot_cumulative_returns, plot_drawdown
from marketlab.reports.summary import build_fold_summary, build_model_summary
from marketlab.strategies.buy_hold import generate_weights as buy_hold_weights
from marketlab.strategies.ranking import generate_weights as ranking_weights
from marketlab.strategies.sma import generate_weights as sma_weights
from marketlab.targets.weekly import build_weekly_modeling_dataset

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExperimentArtifacts:
    run_dir: Path
    panel_path: Path
    metrics_path: Path
    performance_path: Path
    report_path: Path | None
    cumulative_plot_path: Path | None
    drawdown_plot_path: Path | None
    fold_summary_path: Path | None
    model_summary_path: Path | None


@dataclass(slots=True)
class TrainModelsArtifacts:
    run_dir: Path
    panel_path: Path
    folds_path: Path
    model_manifest_path: Path
    metrics_path: Path | None
    predictions_path: Path | None
    fold_summary_path: Path
    model_summary_path: Path


def prepare_data(config: ExperimentConfig) -> tuple[pd.DataFrame, Path]:
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    if config.prepared_panel_path.exists():
        LOGGER.info("Loading prepared panel from %s", config.prepared_panel_path)
        return load_panel_csv(config.prepared_panel_path), config.prepared_panel_path

    LOGGER.info("Prepared panel not found. Building it from raw market data.")
    frames = load_symbol_frames(config)
    panel = build_market_panel(frames)
    panel_path = save_panel_csv(panel, config.prepared_panel_path)
    return panel, panel_path


def _run_dir(config: ExperimentConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output_dir / config.experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _persist_experiment_outputs(
    config: ExperimentConfig,
    panel_path: Path,
    performance: pd.DataFrame,
    run_dir: Path | None = None,
    model_summary: pd.DataFrame | None = None,
    fold_summary: pd.DataFrame | None = None,
) -> ExperimentArtifacts:
    artifact_run_dir = run_dir or _run_dir(config)
    metrics = compute_strategy_metrics(performance)

    metrics_path = artifact_run_dir / "metrics.csv"
    performance_path = artifact_run_dir / "performance.csv"
    metrics.to_csv(metrics_path, index=False)
    performance.to_csv(performance_path, index=False)

    model_summary_path: Path | None = None
    if model_summary is not None:
        model_summary_path = artifact_run_dir / "model_summary.csv"
        model_summary.to_csv(model_summary_path, index=False)

    fold_summary_path: Path | None = None
    if fold_summary is not None:
        fold_summary_path = artifact_run_dir / "fold_summary.csv"
        fold_summary.to_csv(fold_summary_path, index=False)

    report_path: Path | None = None
    if config.artifacts.save_report_md:
        report_path = write_markdown_report(
            config=config,
            metrics=metrics,
            performance=performance,
            path=artifact_run_dir / "report.md",
            model_summary=model_summary,
            fold_summary=fold_summary,
        )

    cumulative_plot_path: Path | None = None
    drawdown_plot_path: Path | None = None
    if config.artifacts.save_plots:
        cumulative_plot_path = plot_cumulative_returns(
            performance=performance,
            path=artifact_run_dir / "cumulative_returns.png",
        )
        drawdown_plot_path = plot_drawdown(
            performance=performance,
            path=artifact_run_dir / "drawdown.png",
        )

    return ExperimentArtifacts(
        run_dir=artifact_run_dir,
        panel_path=panel_path,
        metrics_path=metrics_path,
        performance_path=performance_path,
        report_path=report_path,
        cumulative_plot_path=cumulative_plot_path,
        drawdown_plot_path=drawdown_plot_path,
        fold_summary_path=fold_summary_path,
        model_summary_path=model_summary_path,
    )


def run_baselines(config: ExperimentConfig, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    featured = add_feature_set(
        panel=panel,
        return_windows=config.features.return_windows,
        ma_windows=config.features.ma_windows,
        vol_windows=config.features.vol_windows,
        momentum_window=config.features.momentum_window,
    )

    performance_frames: list[pd.DataFrame] = []
    if config.baselines.buy_hold:
        weights = buy_hold_weights(featured)
        performance_frames.append(
            run_backtest(
                panel=featured,
                weights=weights,
                cost_bps=config.portfolio.costs.bps_per_trade,
            )
        )

    if config.baselines.sma.enabled:
        weights = sma_weights(
            panel=featured,
            fast_window=config.baselines.sma.fast_window,
            slow_window=config.baselines.sma.slow_window,
        )
        if not weights.empty:
            performance_frames.append(
                run_backtest(
                    panel=featured,
                    weights=weights,
                    cost_bps=config.portfolio.costs.bps_per_trade,
                )
            )

    if not performance_frames:
        raise RuntimeError("No baseline strategies are enabled.")

    performance = pd.concat(performance_frames, ignore_index=True)
    metrics = compute_strategy_metrics(performance)
    return performance, metrics


def _shared_oos_dates(
    panel: pd.DataFrame,
    modeling_dataset: pd.DataFrame,
    folds: list,
    frequency: str,
) -> pd.Index:
    panel_dates = pd.Index(sorted(pd.to_datetime(panel["timestamp"]).drop_duplicates()))
    oos_dates = pd.Index([], dtype="datetime64[ns]")

    for fold in folds:
        _, test_rows = slice_fold_rows(modeling_dataset, fold)
        if test_rows.empty:
            continue

        start_date = pd.Timestamp(test_rows["effective_date"].min())
        boundary_date = next_rebalance_effective_date(
            panel,
            signal_date=pd.Timestamp(test_rows["signal_date"].max()),
            frequency=frequency,
        )
        if boundary_date is None:
            segment_dates = panel_dates[panel_dates >= start_date]
        else:
            segment_dates = panel_dates[
                (panel_dates >= start_date) & (panel_dates < pd.Timestamp(boundary_date))
            ]
        oos_dates = oos_dates.union(segment_dates)

    return pd.Index(sorted(oos_dates))


def _slice_and_rebase_performance(
    performance: pd.DataFrame,
    oos_dates: pd.Index,
) -> pd.DataFrame:
    sliced = performance.loc[performance["date"].isin(oos_dates)].copy()
    if sliced.empty:
        raise RuntimeError("Shared OOS window produced no performance rows.")

    frames: list[pd.DataFrame] = []
    for strategy, strategy_frame in sliced.groupby("strategy", sort=False):
        ordered = strategy_frame.sort_values("date").reset_index(drop=True)
        ordered["equity"] = (1.0 + ordered["net_return"]).cumprod()
        ordered["strategy"] = strategy
        frames.append(ordered)

    return pd.concat(frames, ignore_index=True)


def _run_ml_strategies(
    config: ExperimentConfig,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    performance_frames: list[pd.DataFrame] = []

    for _, model_predictions in predictions.groupby("model_name", sort=True):
        weights = ranking_weights(
            predictions=model_predictions,
            panel=panel,
            long_n=config.portfolio.ranking.long_n,
            short_n=config.portfolio.ranking.short_n,
            frequency=config.portfolio.ranking.rebalance_frequency,
            weighting=config.portfolio.ranking.weighting,
        )
        performance_frames.append(
            run_backtest(
                panel=panel,
                weights=weights,
                cost_bps=config.portfolio.costs.bps_per_trade,
            )
        )

    if not performance_frames:
        raise RuntimeError("No ML strategy performance was generated.")

    return pd.concat(performance_frames, ignore_index=True)


def train_models(config: ExperimentConfig) -> TrainModelsArtifacts:
    if not config.models:
        raise RuntimeError("No models are configured for train-models.")

    panel, panel_path = prepare_data(config)
    modeling_dataset = build_weekly_modeling_dataset(panel, config)
    if modeling_dataset.empty:
        raise RuntimeError("Weekly modeling dataset is empty.")

    folds = build_walk_forward_folds(
        modeling_dataset=modeling_dataset,
        walk_forward=config.evaluation.walk_forward,
        frequency=config.portfolio.ranking.rebalance_frequency,
    )
    if not folds:
        raise RuntimeError("No walk-forward folds are available for train-models.")

    run_dir = _run_dir(config)

    folds_path = run_dir / "folds.csv"
    folds_to_frame(folds).to_csv(folds_path, index=False)

    training_outputs = train_direction_models_on_folds(
        modeling_dataset=modeling_dataset,
        folds=folds,
        model_specs=config.models,
        target_type=config.target.type,
        run_dir=run_dir,
        save_predictions=config.artifacts.save_predictions,
    )

    model_manifest_path = run_dir / "model_manifest.csv"
    training_outputs.manifest.to_csv(model_manifest_path, index=False)

    metrics_path: Path | None = None
    if config.artifacts.save_metrics_csv:
        metrics_path = run_dir / "model_metrics.csv"
        training_outputs.metrics.to_csv(metrics_path, index=False)

    predictions_path: Path | None = None
    if config.artifacts.save_predictions and training_outputs.predictions is not None:
        predictions_path = run_dir / "predictions.csv"
        training_outputs.predictions.to_csv(predictions_path, index=False)

    model_summary = build_model_summary(
        model_metrics=training_outputs.metrics,
        model_manifest=training_outputs.manifest,
    )
    fold_summary = build_fold_summary(
        model_metrics=training_outputs.metrics,
        model_manifest=training_outputs.manifest,
    )
    model_summary_path = run_dir / "model_summary.csv"
    fold_summary_path = run_dir / "fold_summary.csv"
    model_summary.to_csv(model_summary_path, index=False)
    fold_summary.to_csv(fold_summary_path, index=False)

    return TrainModelsArtifacts(
        run_dir=run_dir,
        panel_path=panel_path,
        folds_path=folds_path,
        model_manifest_path=model_manifest_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        fold_summary_path=fold_summary_path,
        model_summary_path=model_summary_path,
    )


def backtest(config: ExperimentConfig) -> ExperimentArtifacts:
    panel, panel_path = prepare_data(config)
    performance, _ = run_baselines(config, panel)
    return _persist_experiment_outputs(
        config=config,
        panel_path=panel_path,
        performance=performance,
    )


def run_experiment(config: ExperimentConfig) -> ExperimentArtifacts:
    panel, panel_path = prepare_data(config)
    baseline_performance, _ = run_baselines(config, panel)

    if not config.models:
        return _persist_experiment_outputs(
            config=config,
            panel_path=panel_path,
            performance=baseline_performance,
        )

    modeling_dataset = build_weekly_modeling_dataset(panel, config)
    if modeling_dataset.empty:
        raise RuntimeError("Weekly modeling dataset is empty.")

    folds = build_walk_forward_folds(
        modeling_dataset=modeling_dataset,
        walk_forward=config.evaluation.walk_forward,
        frequency=config.portfolio.ranking.rebalance_frequency,
    )
    if not folds:
        raise RuntimeError("No walk-forward folds are available for run-experiment.")

    run_dir = _run_dir(config)
    training_outputs = train_direction_models_on_folds(
        modeling_dataset=modeling_dataset,
        folds=folds,
        model_specs=config.models,
        target_type=config.target.type,
        run_dir=run_dir,
        save_predictions=True,
    )
    if training_outputs.predictions is None or training_outputs.predictions.empty:
        raise RuntimeError("run-experiment requires fold predictions for ranking.")

    ml_performance = _run_ml_strategies(
        config=config,
        panel=panel,
        predictions=training_outputs.predictions,
    )
    oos_dates = _shared_oos_dates(
        panel=panel,
        modeling_dataset=modeling_dataset,
        folds=folds,
        frequency=config.portfolio.ranking.rebalance_frequency,
    )
    combined_performance = pd.concat(
        [baseline_performance, ml_performance],
        ignore_index=True,
    )
    oos_performance = _slice_and_rebase_performance(combined_performance, oos_dates)
    model_summary = build_model_summary(
        model_metrics=training_outputs.metrics,
        model_manifest=training_outputs.manifest,
    )
    fold_summary = build_fold_summary(
        model_metrics=training_outputs.metrics,
        model_manifest=training_outputs.manifest,
    )

    return _persist_experiment_outputs(
        config=config,
        panel_path=panel_path,
        performance=oos_performance,
        run_dir=run_dir,
        model_summary=model_summary,
        fold_summary=fold_summary,
    )
