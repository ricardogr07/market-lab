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
from marketlab.evaluation import (
    build_walk_forward_folds,
    folds_to_frame,
    slice_fold_rows,
)
from marketlab.evaluation.walk_forward import build_walk_forward_diagnostics
from marketlab.features.engineering import add_feature_set
from marketlab.models import train_direction_models_on_folds
from marketlab.rebalance import next_rebalance_effective_date
from marketlab.reports.analytics import (
    build_monthly_returns,
    build_strategy_summary,
    build_turnover_costs,
)
from marketlab.reports.markdown import write_markdown_report
from marketlab.reports.plots import (
    plot_calibration_curves,
    plot_cumulative_returns,
    plot_drawdown,
    plot_score_histograms,
    plot_threshold_sweeps,
    plot_turnover,
)
from marketlab.reports.summary import build_fold_summary, build_model_summary
from marketlab.strategies.allocation import generate_weights as allocation_weights
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
    strategy_summary_path: Path
    monthly_returns_path: Path
    turnover_costs_path: Path
    report_path: Path | None
    cumulative_plot_path: Path | None
    drawdown_plot_path: Path | None
    turnover_plot_path: Path | None
    calibration_curves_plot_path: Path | None
    score_histograms_plot_path: Path | None
    threshold_sweeps_plot_path: Path | None
    fold_diagnostics_path: Path | None
    ranking_diagnostics_path: Path | None
    calibration_diagnostics_path: Path | None
    score_histograms_path: Path | None
    threshold_diagnostics_path: Path | None
    fold_summary_path: Path | None
    model_summary_path: Path | None


@dataclass(slots=True)
class TrainModelsArtifacts:
    run_dir: Path
    panel_path: Path
    folds_path: Path
    fold_diagnostics_path: Path
    ranking_diagnostics_path: Path
    calibration_diagnostics_path: Path
    score_histograms_path: Path
    threshold_diagnostics_path: Path
    model_manifest_path: Path
    metrics_path: Path | None
    predictions_path: Path | None
    calibration_curves_plot_path: Path | None
    score_histograms_plot_path: Path | None
    threshold_sweeps_plot_path: Path | None
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


def _write_fold_diagnostics(run_dir: Path, fold_diagnostics: pd.DataFrame) -> Path:
    diagnostics_path = run_dir / "fold_diagnostics.csv"
    fold_diagnostics.to_csv(diagnostics_path, index=False)
    return diagnostics_path


def _persist_experiment_outputs(
    config: ExperimentConfig,
    panel_path: Path,
    performance: pd.DataFrame,
    run_dir: Path | None = None,
    model_summary: pd.DataFrame | None = None,
    fold_summary: pd.DataFrame | None = None,
    fold_diagnostics: pd.DataFrame | None = None,
    fold_diagnostics_path: Path | None = None,
    ranking_diagnostics: pd.DataFrame | None = None,
    ranking_diagnostics_path: Path | None = None,
    calibration_diagnostics: pd.DataFrame | None = None,
    calibration_diagnostics_path: Path | None = None,
    score_histograms: pd.DataFrame | None = None,
    score_histograms_path: Path | None = None,
    threshold_diagnostics: pd.DataFrame | None = None,
    threshold_diagnostics_path: Path | None = None,
) -> ExperimentArtifacts:
    artifact_run_dir = run_dir or _run_dir(config)
    metrics = compute_strategy_metrics(performance)
    strategy_summary = build_strategy_summary(performance)
    monthly_returns = build_monthly_returns(performance)
    turnover_costs = build_turnover_costs(performance)

    metrics_path = artifact_run_dir / "metrics.csv"
    performance_path = artifact_run_dir / "performance.csv"
    strategy_summary_path = artifact_run_dir / "strategy_summary.csv"
    monthly_returns_path = artifact_run_dir / "monthly_returns.csv"
    turnover_costs_path = artifact_run_dir / "turnover_costs.csv"
    metrics.to_csv(metrics_path, index=False)
    performance.to_csv(performance_path, index=False)
    strategy_summary.to_csv(strategy_summary_path, index=False)
    monthly_returns.to_csv(monthly_returns_path, index=False)
    turnover_costs.to_csv(turnover_costs_path, index=False)

    persisted_fold_diagnostics_path = fold_diagnostics_path
    if fold_diagnostics is not None and persisted_fold_diagnostics_path is None:
        persisted_fold_diagnostics_path = _write_fold_diagnostics(
            artifact_run_dir,
            fold_diagnostics,
        )

    persisted_ranking_diagnostics_path = ranking_diagnostics_path
    if ranking_diagnostics is not None and persisted_ranking_diagnostics_path is None:
        persisted_ranking_diagnostics_path = artifact_run_dir / "ranking_diagnostics.csv"
        ranking_diagnostics.to_csv(persisted_ranking_diagnostics_path, index=False)

    persisted_calibration_diagnostics_path = calibration_diagnostics_path
    if calibration_diagnostics is not None and persisted_calibration_diagnostics_path is None:
        persisted_calibration_diagnostics_path = artifact_run_dir / "calibration_diagnostics.csv"
        calibration_diagnostics.to_csv(persisted_calibration_diagnostics_path, index=False)

    persisted_score_histograms_path = score_histograms_path
    if score_histograms is not None and persisted_score_histograms_path is None:
        persisted_score_histograms_path = artifact_run_dir / "score_histograms.csv"
        score_histograms.to_csv(persisted_score_histograms_path, index=False)

    persisted_threshold_diagnostics_path = threshold_diagnostics_path
    if threshold_diagnostics is not None and persisted_threshold_diagnostics_path is None:
        persisted_threshold_diagnostics_path = artifact_run_dir / "threshold_diagnostics.csv"
        threshold_diagnostics.to_csv(persisted_threshold_diagnostics_path, index=False)

    model_summary_path: Path | None = None
    if model_summary is not None:
        model_summary_path = artifact_run_dir / "model_summary.csv"
        model_summary.to_csv(model_summary_path, index=False)

    fold_summary_path: Path | None = None
    if fold_summary is not None:
        fold_summary_path = artifact_run_dir / "fold_summary.csv"
        fold_summary.to_csv(fold_summary_path, index=False)

    cumulative_plot_path: Path | None = None
    drawdown_plot_path: Path | None = None
    turnover_plot_path: Path | None = None
    calibration_curves_plot_path: Path | None = None
    score_histograms_plot_path: Path | None = None
    threshold_sweeps_plot_path: Path | None = None
    if config.artifacts.save_plots:
        cumulative_plot_path = plot_cumulative_returns(
            performance=performance,
            path=artifact_run_dir / "cumulative_returns.png",
        )
        drawdown_plot_path = plot_drawdown(
            performance=performance,
            path=artifact_run_dir / "drawdown.png",
        )
        turnover_plot_path = plot_turnover(
            performance=performance,
            path=artifact_run_dir / "turnover.png",
        )
        if calibration_diagnostics is not None and not calibration_diagnostics.empty:
            calibration_curves_plot_path = plot_calibration_curves(
                calibration_diagnostics=calibration_diagnostics,
                path=artifact_run_dir / "calibration_curves.png",
            )
        if score_histograms is not None and not score_histograms.empty:
            score_histograms_plot_path = plot_score_histograms(
                score_histograms=score_histograms,
                path=artifact_run_dir / "score_histograms.png",
            )
        if threshold_diagnostics is not None and not threshold_diagnostics.empty:
            threshold_sweeps_plot_path = plot_threshold_sweeps(
                threshold_diagnostics=threshold_diagnostics,
                path=artifact_run_dir / "threshold_sweeps.png",
            )

    report_path: Path | None = None
    if config.artifacts.save_report_md:
        report_path = write_markdown_report(
            config=config,
            metrics=metrics,
            performance=performance,
            path=artifact_run_dir / "report.md",
            model_summary=model_summary,
            fold_summary=fold_summary,
            strategy_summary=strategy_summary,
            monthly_returns=monthly_returns,
            turnover_costs=turnover_costs,
            fold_diagnostics=fold_diagnostics,
            threshold_diagnostics=threshold_diagnostics,
            calibration_curves_plot_path=calibration_curves_plot_path,
            score_histograms_plot_path=score_histograms_plot_path,
            threshold_sweeps_plot_path=threshold_sweeps_plot_path,
        )

    return ExperimentArtifacts(
        run_dir=artifact_run_dir,
        panel_path=panel_path,
        metrics_path=metrics_path,
        performance_path=performance_path,
        strategy_summary_path=strategy_summary_path,
        monthly_returns_path=monthly_returns_path,
        turnover_costs_path=turnover_costs_path,
        report_path=report_path,
        cumulative_plot_path=cumulative_plot_path,
        drawdown_plot_path=drawdown_plot_path,
        turnover_plot_path=turnover_plot_path,
        calibration_curves_plot_path=calibration_curves_plot_path,
        score_histograms_plot_path=score_histograms_plot_path,
        threshold_sweeps_plot_path=threshold_sweeps_plot_path,
        fold_diagnostics_path=persisted_fold_diagnostics_path,
        ranking_diagnostics_path=persisted_ranking_diagnostics_path,
        calibration_diagnostics_path=persisted_calibration_diagnostics_path,
        score_histograms_path=persisted_score_histograms_path,
        threshold_diagnostics_path=persisted_threshold_diagnostics_path,
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

    if config.baselines.allocation.enabled:
        weights = allocation_weights(
            panel=featured,
            frequency=config.portfolio.ranking.rebalance_frequency,
            mode=config.baselines.allocation.mode,
            symbol_weights=config.baselines.allocation.symbol_weights,
            symbol_groups=config.data.symbol_groups,
            group_weights=config.baselines.allocation.group_weights,
        )
        if not weights.empty:
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
            mode=config.portfolio.ranking.mode,
            min_score_threshold=config.portfolio.ranking.min_score_threshold,
            cash_when_underfilled=config.portfolio.ranking.cash_when_underfilled,
            symbol_groups=config.data.symbol_groups,
            max_position_weight=config.portfolio.risk.max_position_weight,
            max_group_weight=config.portfolio.risk.max_group_weight,
            max_long_exposure=config.portfolio.risk.max_long_exposure,
            max_short_exposure=config.portfolio.risk.max_short_exposure,
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

    run_dir = _run_dir(config)
    fold_diagnostics = build_walk_forward_diagnostics(
        modeling_dataset=modeling_dataset,
        walk_forward=config.evaluation.walk_forward,
        frequency=config.portfolio.ranking.rebalance_frequency,
    )
    fold_diagnostics_path = _write_fold_diagnostics(run_dir, fold_diagnostics)
    folds = build_walk_forward_folds(
        modeling_dataset=modeling_dataset,
        walk_forward=config.evaluation.walk_forward,
        frequency=config.portfolio.ranking.rebalance_frequency,
    )
    if not folds:
        raise RuntimeError(
            f"No walk-forward folds are available for train-models. See {fold_diagnostics_path}."
        )

    folds_path = run_dir / "folds.csv"
    folds_to_frame(folds).to_csv(folds_path, index=False)

    training_outputs = train_direction_models_on_folds(
        modeling_dataset=modeling_dataset,
        folds=folds,
        model_specs=config.models,
        target_type=config.target.type,
        run_dir=run_dir,
        save_predictions=config.artifacts.save_predictions,
        mode=config.portfolio.ranking.mode,
        long_n=config.portfolio.ranking.long_n,
        short_n=config.portfolio.ranking.short_n,
    )

    model_manifest_path = run_dir / "model_manifest.csv"
    training_outputs.manifest.to_csv(model_manifest_path, index=False)

    metrics_path: Path | None = None
    if config.artifacts.save_metrics_csv:
        metrics_path = run_dir / "model_metrics.csv"
        training_outputs.metrics.to_csv(metrics_path, index=False)

    ranking_diagnostics_path = run_dir / "ranking_diagnostics.csv"
    training_outputs.ranking_diagnostics.to_csv(ranking_diagnostics_path, index=False)
    calibration_diagnostics_path = run_dir / "calibration_diagnostics.csv"
    training_outputs.calibration_diagnostics.to_csv(calibration_diagnostics_path, index=False)
    score_histograms_path = run_dir / "score_histograms.csv"
    training_outputs.score_histograms.to_csv(score_histograms_path, index=False)
    threshold_diagnostics_path = run_dir / "threshold_diagnostics.csv"
    training_outputs.threshold_diagnostics.to_csv(threshold_diagnostics_path, index=False)

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

    calibration_curves_plot_path: Path | None = None
    score_histograms_plot_path: Path | None = None
    threshold_sweeps_plot_path: Path | None = None
    if config.artifacts.save_plots:
        calibration_curves_plot_path = plot_calibration_curves(
            calibration_diagnostics=training_outputs.calibration_diagnostics,
            path=run_dir / "calibration_curves.png",
        )
        score_histograms_plot_path = plot_score_histograms(
            score_histograms=training_outputs.score_histograms,
            path=run_dir / "score_histograms.png",
        )
        threshold_sweeps_plot_path = plot_threshold_sweeps(
            threshold_diagnostics=training_outputs.threshold_diagnostics,
            path=run_dir / "threshold_sweeps.png",
        )

    return TrainModelsArtifacts(
        run_dir=run_dir,
        panel_path=panel_path,
        folds_path=folds_path,
        fold_diagnostics_path=fold_diagnostics_path,
        ranking_diagnostics_path=ranking_diagnostics_path,
        calibration_diagnostics_path=calibration_diagnostics_path,
        score_histograms_path=score_histograms_path,
        threshold_diagnostics_path=threshold_diagnostics_path,
        model_manifest_path=model_manifest_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        calibration_curves_plot_path=calibration_curves_plot_path,
        score_histograms_plot_path=score_histograms_plot_path,
        threshold_sweeps_plot_path=threshold_sweeps_plot_path,
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

    run_dir = _run_dir(config)
    fold_diagnostics = build_walk_forward_diagnostics(
        modeling_dataset=modeling_dataset,
        walk_forward=config.evaluation.walk_forward,
        frequency=config.portfolio.ranking.rebalance_frequency,
    )
    fold_diagnostics_path = _write_fold_diagnostics(run_dir, fold_diagnostics)
    folds = build_walk_forward_folds(
        modeling_dataset=modeling_dataset,
        walk_forward=config.evaluation.walk_forward,
        frequency=config.portfolio.ranking.rebalance_frequency,
    )
    if not folds:
        raise RuntimeError(
            f"No walk-forward folds are available for run-experiment. See {fold_diagnostics_path}."
        )

    training_outputs = train_direction_models_on_folds(
        modeling_dataset=modeling_dataset,
        folds=folds,
        model_specs=config.models,
        target_type=config.target.type,
        run_dir=run_dir,
        save_predictions=True,
        mode=config.portfolio.ranking.mode,
        long_n=config.portfolio.ranking.long_n,
        short_n=config.portfolio.ranking.short_n,
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
        fold_diagnostics=fold_diagnostics,
        fold_diagnostics_path=fold_diagnostics_path,
        ranking_diagnostics=training_outputs.ranking_diagnostics,
        calibration_diagnostics=training_outputs.calibration_diagnostics,
        score_histograms=training_outputs.score_histograms,
        threshold_diagnostics=training_outputs.threshold_diagnostics,
    )



