from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from marketlab.backtest.engine import BacktestResult, run_backtest_detailed
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
    build_benchmark_relative,
    build_cost_sensitivity,
    build_daily_exposure,
    build_group_exposure,
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
from marketlab.strategies.optimized import (
    MEAN_VARIANCE_STRATEGY_NAME,
)
from marketlab.strategies.optimized import (
    generate_weights as optimized_weights,
)
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
    cost_sensitivity_path: Path
    daily_exposure_path: Path
    group_exposure_path: Path | None
    benchmark_relative_path: Path | None
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


def _concat_backtest_results(results: list[BacktestResult]) -> BacktestResult:
    if not results:
        raise RuntimeError("No strategy backtest results were generated.")

    return BacktestResult(
        performance=pd.concat([result.performance for result in results], ignore_index=True),
        daily_holdings=pd.concat([result.daily_holdings for result in results], ignore_index=True),
        daily_cash=pd.concat([result.daily_cash for result in results], ignore_index=True),
    )


def _slice_backtest_result(
    backtest_result: BacktestResult,
    oos_dates: pd.Index,
) -> BacktestResult:
    return BacktestResult(
        performance=_slice_and_rebase_performance(backtest_result.performance, oos_dates),
        daily_holdings=backtest_result.daily_holdings.loc[
            backtest_result.daily_holdings["date"].isin(oos_dates)
        ].copy(),
        daily_cash=backtest_result.daily_cash.loc[
            backtest_result.daily_cash["date"].isin(oos_dates)
        ].copy(),
    )

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
    daily_holdings: pd.DataFrame,
    daily_cash: pd.DataFrame,
    symbol_groups: dict[str, str],
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
    daily_exposure = build_daily_exposure(daily_holdings, daily_cash)
    group_exposure = build_group_exposure(daily_holdings, symbol_groups)
    benchmark_relative = build_benchmark_relative(
        performance,
        config.evaluation.benchmark_strategy,
    )
    strategy_summary = build_strategy_summary(
        performance,
        daily_exposure=daily_exposure,
        group_exposure=group_exposure,
        benchmark_relative=benchmark_relative,
        benchmark_strategy=config.evaluation.benchmark_strategy,
    )
    monthly_returns = build_monthly_returns(performance)
    turnover_costs = build_turnover_costs(performance)
    cost_sensitivity = build_cost_sensitivity(
        performance,
        base_cost_bps=config.portfolio.costs.bps_per_trade,
        sensitivity_bps=config.evaluation.cost_sensitivity_bps,
    )

    metrics_path = artifact_run_dir / "metrics.csv"
    performance_path = artifact_run_dir / "performance.csv"
    strategy_summary_path = artifact_run_dir / "strategy_summary.csv"
    monthly_returns_path = artifact_run_dir / "monthly_returns.csv"
    turnover_costs_path = artifact_run_dir / "turnover_costs.csv"
    cost_sensitivity_path = artifact_run_dir / "cost_sensitivity.csv"
    daily_exposure_path = artifact_run_dir / "daily_exposure.csv"
    metrics.to_csv(metrics_path, index=False)
    performance.to_csv(performance_path, index=False)
    strategy_summary.to_csv(strategy_summary_path, index=False)
    monthly_returns.to_csv(monthly_returns_path, index=False)
    turnover_costs.to_csv(turnover_costs_path, index=False)
    cost_sensitivity.to_csv(cost_sensitivity_path, index=False)
    daily_exposure.to_csv(daily_exposure_path, index=False)

    group_exposure_path: Path | None = None
    if not group_exposure.empty:
        group_exposure_path = artifact_run_dir / "group_exposure.csv"
        group_exposure.to_csv(group_exposure_path, index=False)

    benchmark_relative_path: Path | None = None
    if not benchmark_relative.empty:
        benchmark_relative_path = artifact_run_dir / "benchmark_relative.csv"
        benchmark_relative.to_csv(benchmark_relative_path, index=False)

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
            cost_sensitivity=cost_sensitivity,
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
        cost_sensitivity_path=cost_sensitivity_path,
        daily_exposure_path=daily_exposure_path,
        group_exposure_path=group_exposure_path,
        benchmark_relative_path=benchmark_relative_path,
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


def _cash_only_weights(
    *,
    strategy_name: str,
    effective_date: pd.Timestamp,
    symbols: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": strategy_name,
            "effective_date": pd.Timestamp(effective_date),
            "symbol": symbols,
            "weight": [0.0] * len(symbols),
        }
    )


def run_baselines(config: ExperimentConfig, panel: pd.DataFrame) -> BacktestResult:
    featured = add_feature_set(
        panel=panel,
        return_windows=config.features.return_windows,
        ma_windows=config.features.ma_windows,
        vol_windows=config.features.vol_windows,
        momentum_window=config.features.momentum_window,
    )

    backtest_results: list[BacktestResult] = []
    if config.baselines.buy_hold:
        weights = buy_hold_weights(featured)
        backtest_results.append(
            run_backtest_detailed(
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
            backtest_results.append(
                run_backtest_detailed(
                    panel=featured,
                    weights=weights,
                    cost_bps=config.portfolio.costs.bps_per_trade,
                )
            )

    if config.baselines.optimized.enabled:
        weights = optimized_weights(
            panel=featured,
            symbols=config.data.symbols,
            method=config.baselines.optimized.method,
            lookback_days=config.baselines.optimized.lookback_days,
            frequency=config.baselines.optimized.rebalance_frequency,
            covariance_estimator=config.baselines.optimized.covariance_estimator,
            external_covariance_path=config.optimized_external_covariance_path,
            expected_return_source=config.baselines.optimized.expected_return_source,
            external_expected_returns_path=config.optimized_external_expected_returns_path,
            long_only=config.baselines.optimized.long_only,
            target_gross_exposure=config.baselines.optimized.target_gross_exposure,
            risk_aversion=config.baselines.optimized.risk_aversion,
            symbol_groups=config.data.symbol_groups,
            max_position_weight=config.portfolio.risk.max_position_weight,
            max_group_weight=config.portfolio.risk.max_group_weight,
        )
        if weights.empty and config.baselines.optimized.method == "mean_variance":
            weights = _cash_only_weights(
                strategy_name=MEAN_VARIANCE_STRATEGY_NAME,
                effective_date=pd.Timestamp(featured["timestamp"].min()),
                symbols=config.data.symbols,
            )
        if not weights.empty:
            backtest_results.append(
                run_backtest_detailed(
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
            backtest_results.append(
                run_backtest_detailed(
                    panel=featured,
                    weights=weights,
                    cost_bps=config.portfolio.costs.bps_per_trade,
                )
            )

    return _concat_backtest_results(backtest_results)

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
) -> BacktestResult:
    backtest_results: list[BacktestResult] = []

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
        backtest_results.append(
            run_backtest_detailed(
                panel=panel,
                weights=weights,
                cost_bps=config.portfolio.costs.bps_per_trade,
            )
        )

    return _concat_backtest_results(backtest_results)

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
    baseline_outputs = run_baselines(config, panel)
    return _persist_experiment_outputs(
        config=config,
        panel_path=panel_path,
        performance=baseline_outputs.performance,
        daily_holdings=baseline_outputs.daily_holdings,
        daily_cash=baseline_outputs.daily_cash,
        symbol_groups=config.data.symbol_groups,
    )


def run_experiment(config: ExperimentConfig) -> ExperimentArtifacts:
    panel, panel_path = prepare_data(config)
    baseline_outputs = run_baselines(config, panel)

    if not config.models:
        return _persist_experiment_outputs(
            config=config,
            panel_path=panel_path,
            performance=baseline_outputs.performance,
            daily_holdings=baseline_outputs.daily_holdings,
            daily_cash=baseline_outputs.daily_cash,
            symbol_groups=config.data.symbol_groups,
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

    ml_outputs = _run_ml_strategies(
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
    combined_outputs = _concat_backtest_results([baseline_outputs, ml_outputs])
    oos_outputs = _slice_backtest_result(combined_outputs, oos_dates)
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
        performance=oos_outputs.performance,
        daily_holdings=oos_outputs.daily_holdings,
        daily_cash=oos_outputs.daily_cash,
        symbol_groups=config.data.symbol_groups,
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








