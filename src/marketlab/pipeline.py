from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from marketlab.backtest.engine import (
    BacktestResult,
    run_backtest,
    run_backtest_detailed,
)
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
from marketlab.models.registry import build_model_estimator, predict_direction_scores
from marketlab.models.training import modeling_feature_columns
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
    plot_pattern_detection_windows,
    plot_pattern_detections,
    plot_pattern_price_overlay,
    plot_score_histograms,
    plot_signal_confirmations,
    plot_signal_performance_focus,
    plot_signal_price_overlay,
    plot_threshold_sweeps,
    plot_turnover,
)
from marketlab.reports.risk_diagnostics import (
    build_covariance_diagnostics,
    build_factor_diagnostics,
    load_factor_returns,
)
from marketlab.reports.summary import build_fold_summary, build_model_summary
from marketlab.strategies.allocation import generate_weights as allocation_weights
from marketlab.strategies.buy_hold import generate_weights as buy_hold_weights
from marketlab.strategies.chart_patterns import (
    generate_diagnostics as chart_pattern_diagnostics,
)
from marketlab.strategies.indicator_stack import (
    generate_diagnostics as indicator_stack_diagnostics,
)
from marketlab.strategies.optimized import (
    generate_black_litterman_output,
    generate_covariance_diagnostic_windows,
)
from marketlab.strategies.optimized import (
    generate_cash_only_weights as optimized_cash_only_weights,
)
from marketlab.strategies.optimized import (
    generate_weights as optimized_weights,
)
from marketlab.strategies.optimized import (
    is_executable_method as optimized_method_is_executable,
)
from marketlab.strategies.pattern_exit_overlay import (
    generate_diagnostics as pattern_exit_overlay_diagnostics,
)
from marketlab.strategies.pattern_exit_overlay import (
    generate_weights as pattern_exit_overlay_weights,
)
from marketlab.strategies.pattern_meta_label import (
    build_labels as pattern_meta_labels,
)
from marketlab.strategies.pattern_meta_label import (
    generate_meta_overlay_diagnostics,
    predict_exit_candidates,
)
from marketlab.strategies.pattern_meta_label import (
    generate_weights as pattern_meta_label_weights,
)
from marketlab.strategies.pattern_partial_exposure import (
    generate_diagnostics as pattern_partial_exposure_diagnostics,
)
from marketlab.strategies.pattern_partial_exposure import (
    generate_weights as pattern_partial_exposure_weights,
)
from marketlab.strategies.ranking import generate_weights as ranking_weights
from marketlab.strategies.sma import generate_weights as sma_weights
from marketlab.targets import build_modeling_dataset

LOGGER = logging.getLogger(__name__)
ML_TUNED_STRATEGY_NAME = "ml_indicator_tuned__long_only__cash"


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
    factor_diagnostics_path: Path | None = None
    covariance_diagnostics_path: Path | None = None
    black_litterman_assumptions_path: Path | None = None
    indicator_diagnostics_path: Path | None = None
    signal_price_overlay_plot_path: Path | None = None
    signal_confirmations_plot_path: Path | None = None
    signal_performance_focus_plot_path: Path | None = None
    pattern_diagnostics_path: Path | None = None
    pattern_exit_overlay_diagnostics_path: Path | None = None
    pattern_meta_labels_path: Path | None = None
    pattern_meta_predictions_path: Path | None = None
    pattern_meta_fold_diagnostics_path: Path | None = None
    pattern_meta_threshold_sweep_path: Path | None = None
    pattern_meta_tuning_candidates_path: Path | None = None
    pattern_meta_tuning_selections_path: Path | None = None
    pattern_partial_exposure_diagnostics_path: Path | None = None
    pattern_partial_threshold_sweep_path: Path | None = None
    ml_strategy_threshold_sweep_path: Path | None = None
    ml_strategy_tuning_candidates_path: Path | None = None
    ml_strategy_tuning_selections_path: Path | None = None
    pattern_price_overlay_plot_path: Path | None = None
    pattern_detections_plot_path: Path | None = None
    pattern_detection_windows_plot_path: Path | None = None
    pattern_performance_focus_plot_path: Path | None = None


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


def _slice_black_litterman_assumptions(
    assumptions: pd.DataFrame | None,
    oos_dates: pd.Index,
) -> pd.DataFrame | None:
    if assumptions is None:
        return None
    if assumptions.empty:
        return None

    sliced = assumptions.loc[
        pd.to_datetime(assumptions["effective_date"]).isin(oos_dates)
    ].copy()
    if sliced.empty:
        return None
    return sliced.reset_index(drop=True)


def _slice_covariance_diagnostics(
    diagnostics: pd.DataFrame | None,
    oos_dates: pd.Index,
) -> pd.DataFrame | None:
    if diagnostics is None:
        return None
    if diagnostics.empty:
        return None

    oos_index = pd.Index(pd.to_datetime(oos_dates)).sort_values()
    if oos_index.empty:
        return None

    effective_dates = pd.to_datetime(diagnostics["effective_date"])
    keep_mask = effective_dates.isin(oos_index)
    first_oos_date = pd.Timestamp(oos_index.min())

    # Keep the latest pre-OOS covariance window per strategy so the first
    # shared OOS returns still have the active optimizer matrix in diagnostics.
    for strategy, strategy_rows in diagnostics.groupby("strategy", sort=False):
        strategy_dates = pd.to_datetime(strategy_rows["effective_date"])
        prior_dates = strategy_dates.loc[strategy_dates < first_oos_date]
        if prior_dates.empty:
            continue
        active_date = pd.Timestamp(prior_dates.max())
        keep_mask = keep_mask | (
            (diagnostics["strategy"] == strategy)
            & (effective_dates == active_date)
        )

    sliced = diagnostics.loc[keep_mask].copy()
    if sliced.empty:
        return None
    return sliced.reset_index(drop=True)


def _slice_indicator_diagnostics(
    diagnostics: pd.DataFrame | None,
    oos_dates: pd.Index,
) -> pd.DataFrame | None:
    if diagnostics is None:
        return None
    if diagnostics.empty:
        return None

    sliced = diagnostics.loc[pd.to_datetime(diagnostics["effective_date"]).isin(oos_dates)].copy()
    if sliced.empty:
        return None
    return sliced.reset_index(drop=True)


def _slice_pattern_diagnostics(
    diagnostics: pd.DataFrame | None,
    oos_dates: pd.Index,
) -> pd.DataFrame | None:
    if diagnostics is None:
        return None
    if diagnostics.empty:
        return None

    sliced = diagnostics.loc[pd.to_datetime(diagnostics["effective_date"]).isin(oos_dates)].copy()
    if sliced.empty:
        return None
    return sliced.reset_index(drop=True)


def _walk_forward_frequency(config: ExperimentConfig) -> str:
    if config.portfolio.ranking.rebalance_frequency.lower() != "bar":
        return config.portfolio.ranking.rebalance_frequency
    interval = config.data.interval.lower()
    if interval.endswith("m"):
        return f"{interval[:-1]}min"
    return interval


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


def _filter_focus_frame(
    frame: pd.DataFrame,
    *,
    column: str,
    config: ExperimentConfig,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    timestamps = pd.to_datetime(frame[column])
    mask = pd.Series(True, index=frame.index)
    if config.evaluation.focus_start:
        mask &= timestamps >= pd.Timestamp(config.evaluation.focus_start)
    if config.evaluation.focus_end:
        mask &= timestamps <= pd.Timestamp(config.evaluation.focus_end)
    return frame.loc[mask].copy()


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
    covariance_diagnostics: pd.DataFrame | None = None,
    black_litterman_assumptions: pd.DataFrame | None = None,
    indicator_diagnostics: pd.DataFrame | None = None,
    pattern_diagnostics: pd.DataFrame | None = None,
    pattern_exit_overlay_diagnostics: pd.DataFrame | None = None,
    pattern_meta_labels_frame: pd.DataFrame | None = None,
    pattern_meta_predictions: pd.DataFrame | None = None,
    pattern_meta_fold_diagnostics: pd.DataFrame | None = None,
    pattern_meta_threshold_sweep: pd.DataFrame | None = None,
    pattern_meta_tuning_candidates: pd.DataFrame | None = None,
    pattern_meta_tuning_selections: pd.DataFrame | None = None,
    pattern_partial_exposure_diagnostics: pd.DataFrame | None = None,
    pattern_partial_threshold_sweep: pd.DataFrame | None = None,
    ml_strategy_threshold_sweep: pd.DataFrame | None = None,
    ml_strategy_tuning_candidates: pd.DataFrame | None = None,
    ml_strategy_tuning_selections: pd.DataFrame | None = None,
) -> ExperimentArtifacts:
    artifact_run_dir = run_dir or _run_dir(config)
    metrics = compute_strategy_metrics(
        performance,
        periods_per_year=config.evaluation.periods_per_year,
    )
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
        periods_per_year=config.evaluation.periods_per_year,
    )
    monthly_returns = build_monthly_returns(performance)
    turnover_costs = build_turnover_costs(performance)
    cost_sensitivity = build_cost_sensitivity(
        performance,
        base_cost_bps=config.portfolio.costs.bps_per_trade,
        sensitivity_bps=config.evaluation.cost_sensitivity_bps,
        periods_per_year=config.evaluation.periods_per_year,
    )
    factor_diagnostics: pd.DataFrame | None = None
    if config.factor_model_path is not None:
        factor_diagnostics = build_factor_diagnostics(
            performance,
            load_factor_returns(config.factor_model_path),
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

    factor_diagnostics_path: Path | None = None
    if factor_diagnostics is not None and not factor_diagnostics.empty:
        factor_diagnostics_path = artifact_run_dir / "factor_diagnostics.csv"
        factor_diagnostics.to_csv(
            factor_diagnostics_path,
            index=False,
            date_format="%Y-%m-%d",
        )

    covariance_diagnostics_path: Path | None = None
    if covariance_diagnostics is not None and not covariance_diagnostics.empty:
        covariance_diagnostics_path = artifact_run_dir / "covariance_diagnostics.csv"
        covariance_diagnostics.to_csv(
            covariance_diagnostics_path,
            index=False,
            date_format="%Y-%m-%d",
        )

    black_litterman_assumptions_path: Path | None = None
    if black_litterman_assumptions is not None:
        black_litterman_assumptions_path = artifact_run_dir / "black_litterman_assumptions.csv"
        black_litterman_assumptions.to_csv(
            black_litterman_assumptions_path,
            index=False,
            date_format="%Y-%m-%d",
        )

    indicator_diagnostics_path: Path | None = None
    if indicator_diagnostics is not None and not indicator_diagnostics.empty:
        indicator_diagnostics_path = artifact_run_dir / "indicator_diagnostics.csv"
        indicator_diagnostics.to_csv(indicator_diagnostics_path, index=False)

    pattern_diagnostics_path: Path | None = None
    if pattern_diagnostics is not None and not pattern_diagnostics.empty:
        pattern_diagnostics_path = artifact_run_dir / "pattern_diagnostics.csv"
        pattern_diagnostics.to_csv(pattern_diagnostics_path, index=False)

    pattern_exit_overlay_diagnostics_path: Path | None = None
    if (
        pattern_exit_overlay_diagnostics is not None
        and not pattern_exit_overlay_diagnostics.empty
    ):
        pattern_exit_overlay_diagnostics_path = (
            artifact_run_dir / "pattern_exit_overlay_diagnostics.csv"
        )
        pattern_exit_overlay_diagnostics.to_csv(
            pattern_exit_overlay_diagnostics_path,
            index=False,
        )

    pattern_meta_labels_path: Path | None = None
    if pattern_meta_labels_frame is not None:
        pattern_meta_labels_path = artifact_run_dir / "pattern_meta_labels.csv"
        pattern_meta_labels_frame.to_csv(pattern_meta_labels_path, index=False)

    pattern_meta_predictions_path: Path | None = None
    if pattern_meta_predictions is not None:
        pattern_meta_predictions_path = artifact_run_dir / "pattern_meta_predictions.csv"
        pattern_meta_predictions.to_csv(pattern_meta_predictions_path, index=False)

    pattern_meta_fold_diagnostics_path: Path | None = None
    if pattern_meta_fold_diagnostics is not None:
        pattern_meta_fold_diagnostics_path = (
            artifact_run_dir / "pattern_meta_fold_diagnostics.csv"
        )
        pattern_meta_fold_diagnostics.to_csv(pattern_meta_fold_diagnostics_path, index=False)

    pattern_meta_threshold_sweep_path: Path | None = None
    if pattern_meta_threshold_sweep is not None:
        pattern_meta_threshold_sweep_path = artifact_run_dir / "pattern_meta_threshold_sweep.csv"
        pattern_meta_threshold_sweep.to_csv(pattern_meta_threshold_sweep_path, index=False)

    pattern_meta_tuning_candidates_path: Path | None = None
    if pattern_meta_tuning_candidates is not None:
        pattern_meta_tuning_candidates_path = artifact_run_dir / "pattern_meta_tuning_candidates.csv"
        pattern_meta_tuning_candidates.to_csv(pattern_meta_tuning_candidates_path, index=False)

    pattern_meta_tuning_selections_path: Path | None = None
    if pattern_meta_tuning_selections is not None:
        pattern_meta_tuning_selections_path = artifact_run_dir / "pattern_meta_tuning_selections.csv"
        pattern_meta_tuning_selections.to_csv(pattern_meta_tuning_selections_path, index=False)

    pattern_partial_exposure_diagnostics_path: Path | None = None
    if (
        pattern_partial_exposure_diagnostics is not None
        and not pattern_partial_exposure_diagnostics.empty
    ):
        pattern_partial_exposure_diagnostics_path = (
            artifact_run_dir / "pattern_partial_exposure_diagnostics.csv"
        )
        pattern_partial_exposure_diagnostics.to_csv(
            pattern_partial_exposure_diagnostics_path,
            index=False,
        )

    pattern_partial_threshold_sweep_path: Path | None = None
    if pattern_partial_threshold_sweep is not None:
        pattern_partial_threshold_sweep_path = artifact_run_dir / "pattern_partial_threshold_sweep.csv"
        pattern_partial_threshold_sweep.to_csv(pattern_partial_threshold_sweep_path, index=False)

    ml_strategy_threshold_sweep_path: Path | None = None
    if ml_strategy_threshold_sweep is not None and not ml_strategy_threshold_sweep.empty:
        ml_strategy_threshold_sweep_path = artifact_run_dir / "ml_strategy_threshold_sweep.csv"
        ml_strategy_threshold_sweep.to_csv(ml_strategy_threshold_sweep_path, index=False)

    ml_strategy_tuning_candidates_path: Path | None = None
    if ml_strategy_tuning_candidates is not None:
        ml_strategy_tuning_candidates_path = artifact_run_dir / "ml_strategy_tuning_candidates.csv"
        ml_strategy_tuning_candidates.to_csv(
            ml_strategy_tuning_candidates_path,
            index=False,
        )

    ml_strategy_tuning_selections_path: Path | None = None
    if ml_strategy_tuning_selections is not None:
        ml_strategy_tuning_selections_path = artifact_run_dir / "ml_strategy_tuning_selections.csv"
        ml_strategy_tuning_selections.to_csv(
            ml_strategy_tuning_selections_path,
            index=False,
        )

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
    signal_price_overlay_plot_path: Path | None = None
    signal_confirmations_plot_path: Path | None = None
    signal_performance_focus_plot_path: Path | None = None
    pattern_price_overlay_plot_path: Path | None = None
    pattern_detections_plot_path: Path | None = None
    pattern_detection_windows_plot_path: Path | None = None
    pattern_performance_focus_plot_path: Path | None = None
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
    if (
        config.evaluation.visualize_signals
        and indicator_diagnostics is not None
        and not indicator_diagnostics.empty
    ):
        focused_diagnostics = _filter_focus_frame(
            indicator_diagnostics,
            column="timestamp",
            config=config,
        )
        focused_performance = _filter_focus_frame(
            performance,
            column="date",
            config=config,
        )
        if not focused_diagnostics.empty:
            signal_price_overlay_plot_path = plot_signal_price_overlay(
                diagnostics=focused_diagnostics,
                path=artifact_run_dir / "signal_price_overlay.png",
            )
            signal_confirmations_plot_path = plot_signal_confirmations(
                diagnostics=focused_diagnostics,
                path=artifact_run_dir / "signal_confirmations.png",
            )
        if not focused_performance.empty:
            signal_performance_focus_plot_path = plot_signal_performance_focus(
                performance=focused_performance,
                path=artifact_run_dir / "signal_performance_focus.png",
            )
    if (
        config.evaluation.visualize_signals
        and pattern_diagnostics is not None
        and not pattern_diagnostics.empty
    ):
        focused_patterns = _filter_focus_frame(
            pattern_diagnostics,
            column="timestamp",
            config=config,
        )
        focused_performance = _filter_focus_frame(
            performance,
            column="date",
            config=config,
        )
        if not focused_patterns.empty:
            pattern_price_overlay_plot_path = plot_pattern_price_overlay(
                diagnostics=focused_patterns,
                path=artifact_run_dir / "pattern_price_overlay.png",
            )
            pattern_detections_plot_path = plot_pattern_detections(
                diagnostics=focused_patterns,
                path=artifact_run_dir / "pattern_detections.png",
            )
            pattern_detection_windows_plot_path = plot_pattern_detection_windows(
                diagnostics=focused_patterns,
                path=artifact_run_dir / "pattern_detection_windows.png",
            )
        if not focused_performance.empty:
            pattern_performance_focus_plot_path = plot_signal_performance_focus(
                performance=focused_performance,
                path=artifact_run_dir / "pattern_performance_focus.png",
                strategy_names={"chart_patterns", "buy_hold"},
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
            pattern_meta_threshold_sweep=pattern_meta_threshold_sweep,
            pattern_meta_tuning_candidates=pattern_meta_tuning_candidates,
            pattern_meta_tuning_selections=pattern_meta_tuning_selections,
            pattern_partial_threshold_sweep=pattern_partial_threshold_sweep,
            ml_strategy_threshold_sweep=ml_strategy_threshold_sweep,
            ml_strategy_tuning_candidates=ml_strategy_tuning_candidates,
            ml_strategy_tuning_selections=ml_strategy_tuning_selections,
            fold_diagnostics=fold_diagnostics,
            threshold_diagnostics=threshold_diagnostics,
            calibration_curves_plot_path=calibration_curves_plot_path,
            score_histograms_plot_path=score_histograms_plot_path,
            threshold_sweeps_plot_path=threshold_sweeps_plot_path,
            factor_diagnostics=factor_diagnostics,
            factor_diagnostics_path=factor_diagnostics_path,
            covariance_diagnostics=covariance_diagnostics,
            covariance_diagnostics_path=covariance_diagnostics_path,
            black_litterman_assumptions_path=black_litterman_assumptions_path,
            indicator_diagnostics_path=indicator_diagnostics_path,
            signal_price_overlay_plot_path=signal_price_overlay_plot_path,
            signal_confirmations_plot_path=signal_confirmations_plot_path,
            signal_performance_focus_plot_path=signal_performance_focus_plot_path,
            pattern_diagnostics_path=pattern_diagnostics_path,
            pattern_exit_overlay_diagnostics_path=pattern_exit_overlay_diagnostics_path,
            pattern_meta_labels_path=pattern_meta_labels_path,
            pattern_meta_predictions_path=pattern_meta_predictions_path,
            pattern_meta_fold_diagnostics_path=pattern_meta_fold_diagnostics_path,
            pattern_meta_threshold_sweep_path=pattern_meta_threshold_sweep_path,
            pattern_meta_tuning_candidates_path=pattern_meta_tuning_candidates_path,
            pattern_meta_tuning_selections_path=pattern_meta_tuning_selections_path,
            pattern_partial_exposure_diagnostics_path=pattern_partial_exposure_diagnostics_path,
            pattern_partial_threshold_sweep_path=pattern_partial_threshold_sweep_path,
            ml_strategy_threshold_sweep_path=ml_strategy_threshold_sweep_path,
            ml_strategy_tuning_candidates_path=ml_strategy_tuning_candidates_path,
            ml_strategy_tuning_selections_path=ml_strategy_tuning_selections_path,
            pattern_price_overlay_plot_path=pattern_price_overlay_plot_path,
            pattern_detections_plot_path=pattern_detections_plot_path,
            pattern_detection_windows_plot_path=pattern_detection_windows_plot_path,
            pattern_performance_focus_plot_path=pattern_performance_focus_plot_path,
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
        factor_diagnostics_path=factor_diagnostics_path,
        covariance_diagnostics_path=covariance_diagnostics_path,
        black_litterman_assumptions_path=black_litterman_assumptions_path,
        indicator_diagnostics_path=indicator_diagnostics_path,
        signal_price_overlay_plot_path=signal_price_overlay_plot_path,
        signal_confirmations_plot_path=signal_confirmations_plot_path,
        signal_performance_focus_plot_path=signal_performance_focus_plot_path,
        pattern_diagnostics_path=pattern_diagnostics_path,
        pattern_exit_overlay_diagnostics_path=pattern_exit_overlay_diagnostics_path,
        pattern_meta_labels_path=pattern_meta_labels_path,
        pattern_meta_predictions_path=pattern_meta_predictions_path,
        pattern_meta_fold_diagnostics_path=pattern_meta_fold_diagnostics_path,
        pattern_meta_threshold_sweep_path=pattern_meta_threshold_sweep_path,
        pattern_meta_tuning_candidates_path=pattern_meta_tuning_candidates_path,
        pattern_meta_tuning_selections_path=pattern_meta_tuning_selections_path,
        pattern_partial_exposure_diagnostics_path=pattern_partial_exposure_diagnostics_path,
        pattern_partial_threshold_sweep_path=pattern_partial_threshold_sweep_path,
        ml_strategy_threshold_sweep_path=ml_strategy_threshold_sweep_path,
        ml_strategy_tuning_candidates_path=ml_strategy_tuning_candidates_path,
        ml_strategy_tuning_selections_path=ml_strategy_tuning_selections_path,
        pattern_price_overlay_plot_path=pattern_price_overlay_plot_path,
        pattern_detections_plot_path=pattern_detections_plot_path,
        pattern_detection_windows_plot_path=pattern_detection_windows_plot_path,
        pattern_performance_focus_plot_path=pattern_performance_focus_plot_path,
    )


def _run_black_litterman_baseline(
    panel: pd.DataFrame,
    *,
    symbols: list[str],
    lookback_days: int,
    frequency: str,
    covariance_estimator: str,
    external_covariance_path: Path | None,
    target_gross_exposure: float,
    risk_aversion: float,
    equilibrium_weights: dict[str, float],
    tau: float,
    views: list,
    symbol_groups: dict[str, str],
    max_position_weight: float | None,
    max_group_weight: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = generate_black_litterman_output(
        panel,
        symbols=symbols,
        lookback_days=lookback_days,
        frequency=frequency,
        covariance_estimator=covariance_estimator,
        external_covariance_path=external_covariance_path,
        long_only=True,
        target_gross_exposure=target_gross_exposure,
        risk_aversion=risk_aversion,
        symbol_groups=symbol_groups,
        max_position_weight=max_position_weight,
        max_group_weight=max_group_weight,
        equilibrium_weights=equilibrium_weights,
        tau=tau,
        views=views,
    )
    return output.weights, output.assumptions


def _build_pattern_meta_threshold_sweep(
    *,
    panel: pd.DataFrame,
    buy_hold_performance: pd.DataFrame | None,
    exit_overlay_diagnostics: pd.DataFrame,
    meta_predictions: pd.DataFrame,
    thresholds: list[float],
    cost_bps: float,
    periods_per_year: float,
    reentry_clear_bars: int,
    require_price_below_trend_for_exit: bool,
    bearish_confirmation_window_bars: int,
    min_cash_bars: int,
    exit_cooldown_bars: int,
    reentry_requires_price_above_trend: bool,
) -> pd.DataFrame:
    columns = [
        "threshold",
        "strategy",
        "cumulative_return",
        "annualized_return",
        "max_drawdown",
        "sharpe_like",
        "total_turnover",
        "cost_drag",
        "exit_count",
        "cash_bar_count",
        "average_exposure",
        "excess_cumulative_return",
    ]
    if not thresholds:
        return pd.DataFrame(columns=columns)

    benchmark_return = None
    if buy_hold_performance is not None and not buy_hold_performance.empty:
        benchmark_return = float(buy_hold_performance["net_return"].add(1.0).prod() - 1.0)

    rows: list[dict[str, object]] = []
    for threshold in sorted(set(float(value) for value in thresholds)):
        diagnostics = generate_meta_overlay_diagnostics(
            exit_overlay_diagnostics,
            meta_predictions,
            threshold=threshold,
            reentry_clear_bars=reentry_clear_bars,
            require_price_below_trend_for_exit=require_price_below_trend_for_exit,
            bearish_confirmation_window_bars=bearish_confirmation_window_bars,
            min_cash_bars=min_cash_bars,
            exit_cooldown_bars=exit_cooldown_bars,
            reentry_requires_price_above_trend=reentry_requires_price_above_trend,
            strategy_name=f"pattern_meta_label_exit_overlay_threshold_{threshold:g}",
        )
        weights = pattern_meta_label_weights(diagnostics)
        if weights.empty:
            continue
        target_weights = pd.to_numeric(diagnostics["target_weight"], errors="coerce").fillna(0.0)
        exit_count = 0
        for _, symbol_weights in diagnostics.assign(target_weight=target_weights).groupby(
            "symbol",
            sort=False,
        ):
            previous_weights = symbol_weights["target_weight"].shift(1).fillna(1.0)
            exit_count += int(
                (
                    previous_weights.gt(0.0)
                    & symbol_weights["target_weight"].le(0.0)
                ).sum()
            )
        result = run_backtest_detailed(
            panel=panel,
            weights=weights,
            cost_bps=cost_bps,
        )
        metrics = compute_strategy_metrics(
            result.performance,
            periods_per_year=periods_per_year,
        ).iloc[0]
        gross_cumulative_return = float(
            result.performance["gross_return"].add(1.0).prod() - 1.0
        )
        rows.append(
            {
                "threshold": threshold,
                "strategy": metrics["strategy"],
                "cumulative_return": metrics["cumulative_return"],
                "annualized_return": metrics["annualized_return"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_like": metrics["sharpe_like"],
                "total_turnover": metrics["total_turnover"],
                "cost_drag": gross_cumulative_return - float(metrics["cumulative_return"]),
                "exit_count": exit_count,
                "cash_bar_count": int(target_weights.le(0.0).sum()),
                "average_exposure": float(target_weights.mean()),
                "excess_cumulative_return": (
                    float(metrics["cumulative_return"]) - benchmark_return
                    if benchmark_return is not None
                    else pd.NA
                ),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _strategy_activity(diagnostics: pd.DataFrame) -> tuple[int, int, int, float]:
    if diagnostics.empty:
        return 0, 0, 0, 0.0
    target_weights = pd.to_numeric(diagnostics["target_weight"], errors="coerce").fillna(0.0)
    exit_count = 0
    for _, symbol_weights in diagnostics.assign(target_weight=target_weights).groupby(
        "symbol",
        sort=False,
    ):
        previous_weights = symbol_weights["target_weight"].shift(1).fillna(1.0)
        exit_count += int(
            (
                previous_weights.gt(symbol_weights["target_weight"])
                & previous_weights.gt(0.0)
            ).sum()
        )
    cash_bar_count = int(target_weights.le(0.0).sum())
    partial_bar_count = int((target_weights.gt(0.0) & target_weights.lt(1.0)).sum())
    average_exposure = float(target_weights.mean())
    return exit_count, cash_bar_count, partial_bar_count, average_exposure


def _performance_window(panel: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    dates = pd.Index(pd.to_datetime(panel["timestamp"]).drop_duplicates())
    return performance.loc[pd.to_datetime(performance["date"]).isin(dates)].copy()


def _metrics_row_for_result(
    *,
    result: BacktestResult,
    periods_per_year: float,
) -> pd.Series:
    return compute_strategy_metrics(
        result.performance,
        periods_per_year=periods_per_year,
    ).iloc[0]


def _build_pattern_meta_tuning_outputs(
    *,
    panel: pd.DataFrame,
    buy_hold_performance: pd.DataFrame | None,
    exit_overlay_diagnostics: pd.DataFrame,
    meta_predictions: pd.DataFrame,
    thresholds: list[float],
    cost_bps: float,
    periods_per_year: float,
    max_average_exposure_for_active: float,
    min_oos_exit_count: int,
    reentry_clear_bars: int,
    require_price_below_trend_for_exit: bool,
    bearish_confirmation_window_bars: int,
    min_cash_bars: int,
    exit_cooldown_bars: int,
    reentry_requires_price_above_trend: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_columns = [
        "fold_id",
        "threshold",
        "strategy",
        "cumulative_return",
        "max_drawdown",
        "total_turnover",
        "exit_count",
        "cash_bar_count",
        "partial_bar_count",
        "average_exposure",
        "benchmark_cumulative_return",
        "excess_cumulative_return",
        "drawdown_delta",
        "active_candidate",
        "passed_gate",
    ]
    selection_columns = [
        "fold_id",
        "selected_threshold",
        "selected_strategy",
        "passed_gate",
        "excess_cumulative_return",
        "drawdown_delta",
        "exit_count",
        "average_exposure",
    ]
    if meta_predictions.empty or exit_overlay_diagnostics.empty:
        return (
            pd.DataFrame(columns=candidate_columns),
            pd.DataFrame(columns=selection_columns),
        )

    thresholds = sorted(set(float(value) for value in thresholds))
    if not thresholds:
        return (
            pd.DataFrame(columns=candidate_columns),
            pd.DataFrame(columns=selection_columns),
        )

    rows: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for fold_id, fold_predictions in meta_predictions.groupby("fold_id", sort=True):
        test_dates = pd.Index(pd.to_datetime(fold_predictions["effective_date"]).drop_duplicates())
        fold_panel = panel.loc[pd.to_datetime(panel["timestamp"]).isin(test_dates)].copy()
        if fold_panel.empty:
            continue
        benchmark_return = 0.0
        benchmark_drawdown = 0.0
        if buy_hold_performance is not None and not buy_hold_performance.empty:
            benchmark_window = _performance_window(fold_panel, buy_hold_performance)
            if not benchmark_window.empty:
                benchmark_return = float(benchmark_window["net_return"].add(1.0).prod() - 1.0)
                benchmark_drawdown = float(
                    compute_strategy_metrics(
                        benchmark_window,
                        periods_per_year=periods_per_year,
                    ).iloc[0]["max_drawdown"]
                )
        fold_rows: list[dict[str, object]] = []
        for threshold in thresholds:
            diagnostics = generate_meta_overlay_diagnostics(
                exit_overlay_diagnostics,
                fold_predictions,
                threshold=threshold,
                reentry_clear_bars=reentry_clear_bars,
                require_price_below_trend_for_exit=require_price_below_trend_for_exit,
                bearish_confirmation_window_bars=bearish_confirmation_window_bars,
                min_cash_bars=min_cash_bars,
                exit_cooldown_bars=exit_cooldown_bars,
                reentry_requires_price_above_trend=reentry_requires_price_above_trend,
                strategy_name=f"pattern_meta_label_tuned_fold_{fold_id}_threshold_{threshold:g}",
            )
            diagnostics = diagnostics.loc[
                pd.to_datetime(diagnostics["effective_date"]).isin(test_dates)
            ].copy()
            weights = pattern_meta_label_weights(diagnostics)
            if weights.empty:
                continue
            result = run_backtest_detailed(panel=fold_panel, weights=weights, cost_bps=cost_bps)
            metrics = _metrics_row_for_result(result=result, periods_per_year=periods_per_year)
            exit_count, cash_bar_count, partial_bar_count, average_exposure = _strategy_activity(
                diagnostics
            )
            excess_return = float(metrics["cumulative_return"]) - benchmark_return
            drawdown_delta = float(metrics["max_drawdown"]) - benchmark_drawdown
            active_candidate = (
                exit_count >= min_oos_exit_count
                and average_exposure <= max_average_exposure_for_active
            )
            passed_gate = active_candidate and excess_return > 0.0 and drawdown_delta >= 0.0
            fold_row = {
                "fold_id": int(fold_id),
                "threshold": threshold,
                "strategy": metrics["strategy"],
                "cumulative_return": metrics["cumulative_return"],
                "max_drawdown": metrics["max_drawdown"],
                "total_turnover": metrics["total_turnover"],
                "exit_count": exit_count,
                "cash_bar_count": cash_bar_count,
                "partial_bar_count": partial_bar_count,
                "average_exposure": average_exposure,
                "benchmark_cumulative_return": benchmark_return,
                "excess_cumulative_return": excess_return,
                "drawdown_delta": drawdown_delta,
                "active_candidate": active_candidate,
                "passed_gate": passed_gate,
            }
            rows.append(fold_row)
            fold_rows.append(fold_row)
        if fold_rows:
            selected = sorted(
                fold_rows,
                key=lambda row: (
                    bool(row["passed_gate"]),
                    bool(row["active_candidate"]),
                    float(row["excess_cumulative_return"]),
                    float(row["drawdown_delta"]),
                ),
                reverse=True,
            )[0]
            selections.append(
                {
                    "fold_id": selected["fold_id"],
                    "selected_threshold": selected["threshold"],
                    "selected_strategy": selected["strategy"],
                    "passed_gate": selected["passed_gate"],
                    "excess_cumulative_return": selected["excess_cumulative_return"],
                    "drawdown_delta": selected["drawdown_delta"],
                    "exit_count": selected["exit_count"],
                    "average_exposure": selected["average_exposure"],
                }
            )

    return (
        pd.DataFrame(rows, columns=candidate_columns),
        pd.DataFrame(selections, columns=selection_columns),
    )


def _build_pattern_partial_threshold_sweep(
    *,
    panel: pd.DataFrame,
    buy_hold_performance: pd.DataFrame | None,
    exit_overlay_diagnostics: pd.DataFrame,
    meta_predictions: pd.DataFrame,
    partial_thresholds: list[float],
    full_thresholds: list[float],
    partial_weight: float,
    cost_bps: float,
    periods_per_year: float,
    reentry_clear_bars: int,
    require_price_below_trend_for_exit: bool,
    bearish_confirmation_window_bars: int,
    min_cash_bars: int,
    exit_cooldown_bars: int,
    reentry_requires_price_above_trend: bool,
) -> pd.DataFrame:
    columns = [
        "partial_threshold",
        "full_threshold",
        "strategy",
        "cumulative_return",
        "max_drawdown",
        "total_turnover",
        "cost_drag",
        "exit_count",
        "cash_bar_count",
        "partial_bar_count",
        "average_exposure",
        "excess_cumulative_return",
    ]
    if not partial_thresholds or not full_thresholds:
        return pd.DataFrame(columns=columns)

    benchmark_return = None
    if buy_hold_performance is not None and not buy_hold_performance.empty:
        benchmark_return = float(buy_hold_performance["net_return"].add(1.0).prod() - 1.0)

    rows: list[dict[str, object]] = []
    for partial_threshold in sorted(set(float(value) for value in partial_thresholds)):
        for full_threshold in sorted(set(float(value) for value in full_thresholds)):
            if full_threshold < partial_threshold:
                continue
            diagnostics = pattern_partial_exposure_diagnostics(
                exit_overlay_diagnostics,
                meta_predictions,
                partial_threshold=partial_threshold,
                full_threshold=full_threshold,
                partial_weight=partial_weight,
                reentry_clear_bars=reentry_clear_bars,
                require_price_below_trend_for_exit=require_price_below_trend_for_exit,
                bearish_confirmation_window_bars=bearish_confirmation_window_bars,
                min_cash_bars=min_cash_bars,
                exit_cooldown_bars=exit_cooldown_bars,
                reentry_requires_price_above_trend=reentry_requires_price_above_trend,
                strategy_name=(
                    "pattern_partial_exposure_overlay_"
                    f"partial_{partial_threshold:g}_full_{full_threshold:g}"
                ),
            )
            weights = pattern_partial_exposure_weights(diagnostics)
            if weights.empty:
                continue
            result = run_backtest_detailed(panel=panel, weights=weights, cost_bps=cost_bps)
            metrics = _metrics_row_for_result(result=result, periods_per_year=periods_per_year)
            gross_cumulative_return = float(
                result.performance["gross_return"].add(1.0).prod() - 1.0
            )
            exit_count, cash_bar_count, partial_bar_count, average_exposure = _strategy_activity(
                diagnostics
            )
            rows.append(
                {
                    "partial_threshold": partial_threshold,
                    "full_threshold": full_threshold,
                    "strategy": metrics["strategy"],
                    "cumulative_return": metrics["cumulative_return"],
                    "max_drawdown": metrics["max_drawdown"],
                    "total_turnover": metrics["total_turnover"],
                    "cost_drag": gross_cumulative_return - float(metrics["cumulative_return"]),
                    "exit_count": exit_count,
                    "cash_bar_count": cash_bar_count,
                    "partial_bar_count": partial_bar_count,
                    "average_exposure": average_exposure,
                    "excess_cumulative_return": (
                        float(metrics["cumulative_return"]) - benchmark_return
                        if benchmark_return is not None
                        else pd.NA
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def run_baselines(
    config: ExperimentConfig,
    panel: pd.DataFrame,
) -> tuple[
    BacktestResult,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    featured = add_feature_set(
        panel=panel,
        return_windows=config.features.return_windows,
        ma_windows=config.features.ma_windows,
        vol_windows=config.features.vol_windows,
        momentum_window=config.features.momentum_window,
    )

    backtest_results: list[BacktestResult] = []
    buy_hold_performance: pd.DataFrame | None = None
    if config.baselines.buy_hold:
        weights = buy_hold_weights(featured)
        buy_hold_result = run_backtest_detailed(
            panel=featured,
            weights=weights,
            cost_bps=config.portfolio.costs.bps_per_trade,
        )
        buy_hold_performance = buy_hold_result.performance
        backtest_results.append(buy_hold_result)

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

    black_litterman_assumptions: pd.DataFrame | None = None
    covariance_diagnostics: pd.DataFrame | None = None
    indicator_diagnostics: pd.DataFrame | None = None
    pattern_diagnostics: pd.DataFrame | None = None
    exit_overlay_diagnostics: pd.DataFrame | None = None
    meta_labels: pd.DataFrame | None = None
    meta_predictions: pd.DataFrame | None = None
    meta_fold_diagnostics: pd.DataFrame | None = None
    meta_threshold_sweep: pd.DataFrame | None = None
    meta_tuning_candidates: pd.DataFrame | None = None
    meta_tuning_selections: pd.DataFrame | None = None
    partial_exposure_diagnostics: pd.DataFrame | None = None
    partial_threshold_sweep: pd.DataFrame | None = None
    if config.baselines.optimized.enabled:
        optimized_method = config.baselines.optimized.method
        if optimized_method == "black_litterman":
            weights, black_litterman_assumptions = _run_black_litterman_baseline(
                featured,
                symbols=config.data.symbols,
                lookback_days=config.baselines.optimized.lookback_days,
                frequency=config.baselines.optimized.rebalance_frequency,
                covariance_estimator=config.baselines.optimized.covariance_estimator,
                external_covariance_path=config.optimized_external_covariance_path,
                target_gross_exposure=config.baselines.optimized.target_gross_exposure,
                risk_aversion=config.baselines.optimized.risk_aversion,
                equilibrium_weights=config.baselines.optimized.equilibrium_weights,
                tau=config.baselines.optimized.tau,
                views=config.baselines.optimized.views,
                symbol_groups=config.data.symbol_groups,
                max_position_weight=config.portfolio.risk.max_position_weight,
                max_group_weight=config.portfolio.risk.max_group_weight,
            )
        else:
            weights = optimized_weights(
                panel=featured,
                symbols=config.data.symbols,
                method=optimized_method,
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
        if optimized_method_is_executable(optimized_method):
            covariance_windows = generate_covariance_diagnostic_windows(
                featured,
                symbols=config.data.symbols,
                method=optimized_method,
                lookback_days=config.baselines.optimized.lookback_days,
                frequency=config.baselines.optimized.rebalance_frequency,
                covariance_estimator=config.baselines.optimized.covariance_estimator,
                external_covariance_path=config.optimized_external_covariance_path,
            )
            covariance_diagnostics = build_covariance_diagnostics(covariance_windows)
            if covariance_diagnostics.empty:
                covariance_diagnostics = None
        if weights.empty and optimized_method_is_executable(optimized_method):
            if optimized_method == "black_litterman":
                black_litterman_assumptions = None
            weights = optimized_cash_only_weights(
                optimized_method,
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
            frequency=config.portfolio.ranking.rebalance_frequency,
        )
        if not weights.empty:
            backtest_results.append(
                run_backtest_detailed(
                    panel=featured,
                    weights=weights,
                    cost_bps=config.portfolio.costs.bps_per_trade,
                )
            )

    if config.baselines.indicator_stack.enabled:
        indicator = config.baselines.indicator_stack
        indicator_diagnostics = indicator_stack_diagnostics(
            panel=featured,
            frequency=config.portfolio.ranking.rebalance_frequency,
            ema_fast_window=indicator.ema_fast_window,
            ema_slow_window=indicator.ema_slow_window,
            rsi_window=indicator.rsi_window,
            rsi_min=indicator.rsi_min,
            rsi_max=indicator.rsi_max,
            macd_fast_window=indicator.macd_fast_window,
            macd_slow_window=indicator.macd_slow_window,
            macd_signal_window=indicator.macd_signal_window,
            bollinger_window=indicator.bollinger_window,
            bollinger_std=indicator.bollinger_std,
            bollinger_mode=indicator.bollinger_mode,
            volume_window=indicator.volume_window,
            volume_multiplier=indicator.volume_multiplier,
            vwap_window=indicator.vwap_window,
            use_vwap=indicator.use_vwap,
            min_confirmations=indicator.min_confirmations,
        )
        weights = indicator_diagnostics.loc[
            :,
            ["strategy", "effective_date", "symbol", "target_weight"],
        ].rename(columns={"target_weight": "weight"})
        if not weights.empty:
            backtest_results.append(
                run_backtest_detailed(
                    panel=featured,
                    weights=weights,
                    cost_bps=config.portfolio.costs.bps_per_trade,
                )
            )

    pattern_baselines_enabled = (
        config.baselines.chart_patterns.enabled
        or config.baselines.pattern_exit_overlay.enabled
        or config.baselines.pattern_meta_label.enabled
        or config.baselines.pattern_partial_exposure_overlay.enabled
    )
    if pattern_baselines_enabled:
        chart_patterns = config.baselines.chart_patterns
        pattern_diagnostics = chart_pattern_diagnostics(
            panel=featured,
            frequency=config.portfolio.ranking.rebalance_frequency,
            lookback_bars=chart_patterns.lookback_bars,
            triangle_slope_min=chart_patterns.triangle_slope_min,
            level_tolerance_pct=chart_patterns.level_tolerance_pct,
            breakout_pct=chart_patterns.breakout_pct,
            rectangle_max_range_pct=chart_patterns.rectangle_max_range_pct,
            flag_pole_bars=chart_patterns.flag_pole_bars,
            flag_consolidation_bars=chart_patterns.flag_consolidation_bars,
            flag_min_pole_return=chart_patterns.flag_min_pole_return,
            flag_max_retrace_pct=chart_patterns.flag_max_retrace_pct,
            volume_window=chart_patterns.volume_window,
            volume_multiplier=chart_patterns.volume_multiplier,
            min_bullish_patterns=chart_patterns.min_bullish_patterns,
        )
        if config.baselines.chart_patterns.enabled:
            weights = pattern_diagnostics.loc[
                :,
                ["strategy", "effective_date", "symbol", "target_weight"],
            ].rename(columns={"target_weight": "weight"})
            if not weights.empty:
                backtest_results.append(
                    run_backtest_detailed(
                        panel=featured,
                        weights=weights,
                        cost_bps=config.portfolio.costs.bps_per_trade,
                    )
                )

    if config.baselines.pattern_exit_overlay.enabled and pattern_diagnostics is not None:
        overlay = config.baselines.pattern_exit_overlay
        exit_overlay_diagnostics = pattern_exit_overlay_diagnostics(
            panel=featured,
            pattern_diagnostics=pattern_diagnostics,
            min_bearish_patterns=overlay.min_bearish_patterns,
            min_bullish_reentry_patterns=overlay.min_bullish_reentry_patterns,
            trend_ema_window=overlay.trend_ema_window,
            reentry_clear_bars=overlay.reentry_clear_bars,
            require_price_below_trend_for_exit=overlay.require_price_below_trend_for_exit,
            bearish_confirmation_window_bars=overlay.bearish_confirmation_window_bars,
            min_cash_bars=overlay.min_cash_bars,
            exit_cooldown_bars=overlay.exit_cooldown_bars,
            reentry_requires_price_above_trend=overlay.reentry_requires_price_above_trend,
        )
        weights = pattern_exit_overlay_weights(exit_overlay_diagnostics)
        if not weights.empty:
            backtest_results.append(
                run_backtest_detailed(
                    panel=featured,
                    weights=weights,
                    cost_bps=config.portfolio.costs.bps_per_trade,
                )
            )

    if (
        config.baselines.pattern_meta_label.enabled
        and exit_overlay_diagnostics is not None
        and not exit_overlay_diagnostics.empty
    ):
        overlay = config.baselines.pattern_exit_overlay
        meta = config.baselines.pattern_meta_label
        meta_labels = pattern_meta_labels(
            panel=featured,
            overlay_diagnostics=exit_overlay_diagnostics,
            label_horizon_bars=meta.label_horizon_bars,
            cost_bps=config.portfolio.costs.bps_per_trade,
        )
        meta_predictions, meta_fold_diagnostics = predict_exit_candidates(
            meta_labels,
            walk_forward=config.evaluation.walk_forward,
            rebalance_frequency=config.portfolio.ranking.rebalance_frequency,
            model_names=meta.models,
            threshold=meta.exit_probability_threshold,
        )
        meta_overlay_diagnostics = generate_meta_overlay_diagnostics(
            exit_overlay_diagnostics,
            meta_predictions,
            threshold=meta.exit_probability_threshold,
            reentry_clear_bars=overlay.reentry_clear_bars,
            require_price_below_trend_for_exit=overlay.require_price_below_trend_for_exit,
            bearish_confirmation_window_bars=overlay.bearish_confirmation_window_bars,
            min_cash_bars=overlay.min_cash_bars,
            exit_cooldown_bars=overlay.exit_cooldown_bars,
            reentry_requires_price_above_trend=overlay.reentry_requires_price_above_trend,
        )
        weights = pattern_meta_label_weights(meta_overlay_diagnostics)
        if not weights.empty:
            backtest_results.append(
                run_backtest_detailed(
                    panel=featured,
                    weights=weights,
                    cost_bps=config.portfolio.costs.bps_per_trade,
                )
            )
        meta_threshold_sweep = _build_pattern_meta_threshold_sweep(
            panel=featured,
            buy_hold_performance=buy_hold_performance,
            exit_overlay_diagnostics=exit_overlay_diagnostics,
            meta_predictions=meta_predictions,
            thresholds=meta.exit_probability_threshold_grid,
            cost_bps=config.portfolio.costs.bps_per_trade,
            periods_per_year=config.evaluation.periods_per_year,
            reentry_clear_bars=overlay.reentry_clear_bars,
            require_price_below_trend_for_exit=overlay.require_price_below_trend_for_exit,
            bearish_confirmation_window_bars=overlay.bearish_confirmation_window_bars,
            min_cash_bars=overlay.min_cash_bars,
            exit_cooldown_bars=overlay.exit_cooldown_bars,
            reentry_requires_price_above_trend=overlay.reentry_requires_price_above_trend,
        )
        if meta.tuning_mode == "nested_walk_forward":
            meta_tuning_candidates, meta_tuning_selections = _build_pattern_meta_tuning_outputs(
                panel=featured,
                buy_hold_performance=buy_hold_performance,
                exit_overlay_diagnostics=exit_overlay_diagnostics,
                meta_predictions=meta_predictions,
                thresholds=meta.exit_probability_threshold_grid
                or [meta.exit_probability_threshold],
                cost_bps=config.portfolio.costs.bps_per_trade,
                periods_per_year=config.evaluation.periods_per_year,
                max_average_exposure_for_active=meta.max_average_exposure_for_active,
                min_oos_exit_count=meta.min_oos_exit_count,
                reentry_clear_bars=overlay.reentry_clear_bars,
                require_price_below_trend_for_exit=overlay.require_price_below_trend_for_exit,
                bearish_confirmation_window_bars=overlay.bearish_confirmation_window_bars,
                min_cash_bars=overlay.min_cash_bars,
                exit_cooldown_bars=overlay.exit_cooldown_bars,
                reentry_requires_price_above_trend=overlay.reentry_requires_price_above_trend,
            )

        partial = config.baselines.pattern_partial_exposure_overlay
        if partial.enabled:
            partial_thresholds = partial.partial_exit_probability_threshold_grid or [
                meta.exit_probability_threshold
            ]
            full_thresholds = partial.full_exit_probability_threshold_grid or [
                max(partial_thresholds)
            ]
            partial_exposure_diagnostics = pattern_partial_exposure_diagnostics(
                exit_overlay_diagnostics,
                meta_predictions,
                partial_threshold=min(partial_thresholds),
                full_threshold=max(full_thresholds),
                partial_weight=partial.partial_weight,
                reentry_clear_bars=overlay.reentry_clear_bars,
                require_price_below_trend_for_exit=overlay.require_price_below_trend_for_exit,
                bearish_confirmation_window_bars=overlay.bearish_confirmation_window_bars,
                min_cash_bars=overlay.min_cash_bars,
                exit_cooldown_bars=overlay.exit_cooldown_bars,
                reentry_requires_price_above_trend=overlay.reentry_requires_price_above_trend,
            )
            weights = pattern_partial_exposure_weights(partial_exposure_diagnostics)
            if not weights.empty:
                backtest_results.append(
                    run_backtest_detailed(
                        panel=featured,
                        weights=weights,
                        cost_bps=config.portfolio.costs.bps_per_trade,
                    )
                )
            partial_threshold_sweep = _build_pattern_partial_threshold_sweep(
                panel=featured,
                buy_hold_performance=buy_hold_performance,
                exit_overlay_diagnostics=exit_overlay_diagnostics,
                meta_predictions=meta_predictions,
                partial_thresholds=partial_thresholds,
                full_thresholds=full_thresholds,
                partial_weight=partial.partial_weight,
                cost_bps=config.portfolio.costs.bps_per_trade,
                periods_per_year=config.evaluation.periods_per_year,
                reentry_clear_bars=overlay.reentry_clear_bars,
                require_price_below_trend_for_exit=overlay.require_price_below_trend_for_exit,
                bearish_confirmation_window_bars=overlay.bearish_confirmation_window_bars,
                min_cash_bars=overlay.min_cash_bars,
                exit_cooldown_bars=overlay.exit_cooldown_bars,
                reentry_requires_price_above_trend=overlay.reentry_requires_price_above_trend,
            )

    return (
        _concat_backtest_results(backtest_results),
        black_litterman_assumptions,
        covariance_diagnostics,
        indicator_diagnostics,
        pattern_diagnostics,
        exit_overlay_diagnostics,
        meta_labels,
        meta_predictions,
        meta_fold_diagnostics,
        meta_threshold_sweep,
        meta_tuning_candidates,
        meta_tuning_selections,
        partial_exposure_diagnostics,
        partial_threshold_sweep,
    )

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


def _weight_activity(weights: pd.DataFrame) -> tuple[int, float]:
    if weights.empty:
        return 0, 0.0
    target_weights = weights.copy()
    target_weights["effective_date"] = pd.to_datetime(target_weights["effective_date"])
    target_weights["weight"] = pd.to_numeric(target_weights["weight"], errors="coerce").fillna(0.0)
    exposure_changes = 0
    average_exposures: list[float] = []
    for _, symbol_weights in target_weights.sort_values(["symbol", "effective_date"]).groupby(
        "symbol",
        sort=False,
    ):
        weights_series = symbol_weights["weight"].astype(float)
        previous_weights = weights_series.shift(1).fillna(0.0)
        exposure_changes += int(weights_series.ne(previous_weights).sum())
        average_exposures.extend(weights_series.abs().tolist())
    average_exposure = (
        float(pd.Series(average_exposures, dtype=float).mean()) if average_exposures else 0.0
    )
    return exposure_changes, average_exposure


def _build_ml_strategy_threshold_sweep(
    *,
    config: ExperimentConfig,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    oos_dates: pd.Index,
    comparison_metrics: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "model_name",
        "threshold",
        "strategy",
        "cumulative_return",
        "annualized_return",
        "max_drawdown",
        "sharpe_like",
        "total_turnover",
        "exposure_changes",
        "average_exposure",
        "buy_hold_cumulative_return",
        "excess_cumulative_return",
        "best_comparison_strategy",
        "best_comparison_cumulative_return",
        "passed_gate",
    ]
    sweep_config = config.evaluation.ml_strategy_threshold_sweep
    if (
        not sweep_config.enabled
        or predictions.empty
        or not sweep_config.thresholds
        or comparison_metrics.empty
    ):
        return pd.DataFrame(columns=columns)

    comparison_names = {
        "buy_hold",
        "chart_patterns",
        "pattern_exit_overlay",
        "pattern_meta_label_exit_overlay",
        "pattern_partial_exposure_overlay",
    }
    comparison_rows = comparison_metrics.loc[
        comparison_metrics["strategy"].astype(str).isin(comparison_names)
    ].copy()
    buy_hold_rows = comparison_rows.loc[
        comparison_rows["strategy"].astype(str) == "buy_hold"
    ]
    if buy_hold_rows.empty:
        return pd.DataFrame(columns=columns)

    buy_hold_return = float(buy_hold_rows.iloc[0]["cumulative_return"])
    best_comparison = comparison_rows.sort_values(
        ["cumulative_return", "strategy"],
        ascending=[False, True],
    ).iloc[0]

    rows: list[dict[str, object]] = []
    for model_name, model_predictions in predictions.groupby("model_name", sort=True):
        for threshold in sorted(set(float(value) for value in sweep_config.thresholds)):
            weights = ranking_weights(
                predictions=model_predictions,
                panel=panel,
                long_n=config.portfolio.ranking.long_n,
                short_n=config.portfolio.ranking.short_n,
                frequency=config.portfolio.ranking.rebalance_frequency,
                weighting=config.portfolio.ranking.weighting,
                mode=config.portfolio.ranking.mode,
                min_score_threshold=threshold,
                cash_when_underfilled=config.portfolio.ranking.cash_when_underfilled,
                symbol_groups=config.data.symbol_groups,
                max_position_weight=config.portfolio.risk.max_position_weight,
                max_group_weight=config.portfolio.risk.max_group_weight,
                max_long_exposure=config.portfolio.risk.max_long_exposure,
                max_short_exposure=config.portfolio.risk.max_short_exposure,
            )
            if weights.empty:
                continue
            result = _slice_backtest_result(
                run_backtest_detailed(
                    panel=panel,
                    weights=weights,
                    cost_bps=config.portfolio.costs.bps_per_trade,
                ),
                oos_dates,
            )
            metrics = compute_strategy_metrics(
                result.performance,
                periods_per_year=config.evaluation.periods_per_year,
            ).iloc[0]
            exposure_changes, average_exposure = _weight_activity(weights)
            excess_return = float(metrics["cumulative_return"]) - buy_hold_return
            passed_gate = (
                excess_return > 0.0
                and exposure_changes >= sweep_config.min_exposure_changes
                and average_exposure <= sweep_config.max_average_exposure_for_active
            )
            rows.append(
                {
                    "model_name": model_name,
                    "threshold": threshold,
                    "strategy": metrics["strategy"],
                    "cumulative_return": metrics["cumulative_return"],
                    "annualized_return": metrics["annualized_return"],
                    "max_drawdown": metrics["max_drawdown"],
                    "sharpe_like": metrics["sharpe_like"],
                    "total_turnover": metrics["total_turnover"],
                    "exposure_changes": exposure_changes,
                    "average_exposure": average_exposure,
                    "buy_hold_cumulative_return": buy_hold_return,
                    "excess_cumulative_return": excess_return,
                    "best_comparison_strategy": best_comparison["strategy"],
                    "best_comparison_cumulative_return": best_comparison[
                        "cumulative_return"
                    ],
                    "passed_gate": passed_gate,
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["passed_gate", "excess_cumulative_return", "model_name", "threshold"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def _prediction_frame_for_rows(
    *,
    model_name: str,
    fold_id: int,
    rows: pd.DataFrame,
    score_series: pd.Series,
    predicted_target: pd.Series,
) -> pd.DataFrame:
    prediction_frame = rows.loc[
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


def _score_direction_rows(
    *,
    model_name: str,
    target_type: str,
    train_rows: pd.DataFrame,
    score_rows: pd.DataFrame,
    feature_columns: list[str],
    fold_id: int,
) -> pd.DataFrame | None:
    train_target = train_rows["target"].astype(int)
    if train_target.nunique(dropna=True) < 2:
        return None

    _, estimator = build_model_estimator(model_name, target_type)
    estimator.fit(train_rows.loc[:, feature_columns], train_target)
    score_series = predict_direction_scores(estimator, score_rows.loc[:, feature_columns])
    predicted_target = pd.Series(
        estimator.predict(score_rows.loc[:, feature_columns]),
        index=score_rows.index,
        name="predicted_target",
        dtype=int,
    )
    return _prediction_frame_for_rows(
        model_name=model_name,
        fold_id=fold_id,
        rows=score_rows,
        score_series=score_series,
        predicted_target=predicted_target,
    )


def _window_dates_for_rows(
    *,
    panel: pd.DataFrame,
    rows: pd.DataFrame,
    frequency: str,
) -> pd.Index:
    if rows.empty:
        return pd.Index([], dtype="datetime64[ns]")

    panel_dates = pd.Index(sorted(pd.to_datetime(panel["timestamp"]).drop_duplicates()))
    start_date = pd.Timestamp(rows["effective_date"].min())
    boundary_date = next_rebalance_effective_date(
        panel,
        signal_date=pd.Timestamp(rows["signal_date"].max()),
        frequency=frequency,
    )
    if boundary_date is None:
        return panel_dates[panel_dates >= start_date]
    return panel_dates[(panel_dates >= start_date) & (panel_dates < pd.Timestamp(boundary_date))]


def _benchmark_metrics_for_window(
    *,
    buy_hold_performance: pd.DataFrame,
    window_dates: pd.Index,
    periods_per_year: float,
) -> pd.Series:
    benchmark_window = buy_hold_performance.loc[
        pd.to_datetime(buy_hold_performance["date"]).isin(window_dates)
    ].copy()
    if benchmark_window.empty:
        raise RuntimeError("ML strategy tuning produced no buy_hold benchmark rows.")
    return compute_strategy_metrics(
        _slice_and_rebase_performance(benchmark_window, window_dates),
        periods_per_year=periods_per_year,
    ).iloc[0]


def _cash_weights_for_rows(
    *,
    panel: pd.DataFrame,
    rows: pd.DataFrame,
    frequency: str,
    strategy_name: str,
) -> pd.DataFrame:
    symbols = sorted(panel["symbol"].drop_duplicates().tolist())
    effective_dates = pd.Index(pd.to_datetime(rows["effective_date"]).drop_duplicates())
    boundary_date = next_rebalance_effective_date(
        panel,
        signal_date=pd.Timestamp(rows["signal_date"].max()),
        frequency=frequency,
    )
    if boundary_date is not None:
        effective_dates = effective_dates.union(pd.Index([pd.Timestamp(boundary_date)]))

    return pd.DataFrame(
        [
            {
                "strategy": strategy_name,
                "effective_date": pd.Timestamp(effective_date),
                "symbol": symbol,
                "weight": 0.0,
            }
            for effective_date in sorted(effective_dates)
            for symbol in symbols
        ],
        columns=["strategy", "effective_date", "symbol", "weight"],
    )


def _weights_for_predictions(
    *,
    config: ExperimentConfig,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float,
    strategy_name: str | None = None,
) -> pd.DataFrame:
    weights = ranking_weights(
        predictions=predictions,
        panel=panel,
        long_n=config.portfolio.ranking.long_n,
        short_n=config.portfolio.ranking.short_n,
        frequency=config.portfolio.ranking.rebalance_frequency,
        weighting=config.portfolio.ranking.weighting,
        mode=config.portfolio.ranking.mode,
        min_score_threshold=threshold,
        cash_when_underfilled=config.portfolio.ranking.cash_when_underfilled,
        symbol_groups=config.data.symbol_groups,
        max_position_weight=config.portfolio.risk.max_position_weight,
        max_group_weight=config.portfolio.risk.max_group_weight,
        max_long_exposure=config.portfolio.risk.max_long_exposure,
        max_short_exposure=config.portfolio.risk.max_short_exposure,
    )
    if strategy_name is not None and not weights.empty:
        weights = weights.copy()
        weights["strategy"] = strategy_name
    return weights


def _build_ml_strategy_tuning_outputs(
    *,
    config: ExperimentConfig,
    panel: pd.DataFrame,
    modeling_dataset: pd.DataFrame,
    folds: list,
    buy_hold_performance: pd.DataFrame,
) -> tuple[BacktestResult | None, pd.DataFrame, pd.DataFrame]:
    candidate_columns = [
        "fold_id",
        "model_name",
        "threshold",
        "strategy",
        "validation_start",
        "validation_end",
        "inner_train_rows",
        "validation_rows",
        "cumulative_return",
        "max_drawdown",
        "sharpe_like",
        "total_turnover",
        "exposure_changes",
        "average_exposure",
        "buy_hold_cumulative_return",
        "excess_cumulative_return",
        "sharpe_like_delta",
        "drawdown_delta",
        "active_candidate",
        "passed_gate",
    ]
    selection_columns = [
        "fold_id",
        "selection_status",
        "selected_model_name",
        "selected_threshold",
        "selected_strategy",
        "validation_start",
        "validation_end",
        "inner_train_rows",
        "validation_rows",
        "passed_gate",
        "excess_cumulative_return",
        "sharpe_like_delta",
        "drawdown_delta",
        "exposure_changes",
        "average_exposure",
    ]
    tuning = config.evaluation.ml_strategy_tuning
    if not tuning.enabled or not config.models or not tuning.thresholds:
        return (
            None,
            pd.DataFrame(columns=candidate_columns),
            pd.DataFrame(columns=selection_columns),
        )

    feature_columns = modeling_feature_columns(modeling_dataset)
    frequency = config.portfolio.ranking.rebalance_frequency
    thresholds = sorted(set(float(value) for value in tuning.thresholds))
    candidate_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    selected_weight_frames: list[pd.DataFrame] = []
    panel_timestamps = pd.to_datetime(panel["timestamp"])

    for fold in folds:
        outer_train_rows, test_rows = slice_fold_rows(modeling_dataset, fold)
        LOGGER.info("Scoring ML strategy tuning fold %s", fold.fold_id)
        validation_start = pd.Timestamp(fold.train_end) - pd.DateOffset(
            months=tuning.validation_months
        )
        validation_rows = outer_train_rows.loc[
            outer_train_rows["signal_date"].ge(validation_start)
        ].copy()
        inner_train_rows = outer_train_rows.loc[
            outer_train_rows["signal_date"].lt(validation_start)
        ].copy()
        validation_dates = _window_dates_for_rows(
            panel=panel,
            rows=validation_rows,
            frequency=frequency,
        )
        validation_panel = panel.loc[panel_timestamps.isin(validation_dates)].copy()
        fold_candidates: list[dict[str, object]] = []

        if (
            len(inner_train_rows) > 0
            and len(validation_rows) >= tuning.min_validation_rows
            and not validation_dates.empty
        ):
            benchmark_metrics = _benchmark_metrics_for_window(
                buy_hold_performance=buy_hold_performance,
                window_dates=validation_dates,
                periods_per_year=config.evaluation.periods_per_year,
            )
            buy_hold_return = float(benchmark_metrics["cumulative_return"])
            buy_hold_sharpe = float(benchmark_metrics["sharpe_like"])
            buy_hold_drawdown = float(benchmark_metrics["max_drawdown"])

            for model_spec in config.models:
                LOGGER.info(
                    "Fitting ML strategy tuning candidate model=%s fold=%s",
                    model_spec.name,
                    fold.fold_id,
                )
                validation_predictions = _score_direction_rows(
                    model_name=model_spec.name,
                    target_type=config.target.type,
                    train_rows=inner_train_rows,
                    score_rows=validation_rows,
                    feature_columns=feature_columns,
                    fold_id=fold.fold_id,
                )
                if validation_predictions is None:
                    continue
                for threshold in thresholds:
                    weights = _weights_for_predictions(
                        config=config,
                        panel=panel,
                        predictions=validation_predictions,
                        threshold=threshold,
                    )
                    if weights.empty:
                        continue
                    performance = run_backtest(
                        panel=validation_panel,
                        weights=weights,
                        cost_bps=config.portfolio.costs.bps_per_trade,
                    )
                    metrics = compute_strategy_metrics(
                        performance,
                        periods_per_year=config.evaluation.periods_per_year,
                    ).iloc[0]
                    exposure_changes, average_exposure = _weight_activity(weights)
                    excess_return = float(metrics["cumulative_return"]) - buy_hold_return
                    sharpe_delta = float(metrics["sharpe_like"]) - buy_hold_sharpe
                    drawdown_delta = float(metrics["max_drawdown"]) - buy_hold_drawdown
                    active_candidate = (
                        exposure_changes >= tuning.min_exposure_changes
                        and average_exposure <= tuning.max_average_exposure_for_active
                    )
                    passed_gate = (
                        active_candidate
                        and excess_return > 0.0
                        and (sharpe_delta > 0.0 or drawdown_delta >= 0.0)
                    )
                    row = {
                        "fold_id": fold.fold_id,
                        "model_name": model_spec.name,
                        "threshold": threshold,
                        "strategy": metrics["strategy"],
                        "validation_start": validation_rows["signal_date"].min(),
                        "validation_end": validation_rows["signal_date"].max(),
                        "inner_train_rows": len(inner_train_rows),
                        "validation_rows": len(validation_rows),
                        "cumulative_return": metrics["cumulative_return"],
                        "max_drawdown": metrics["max_drawdown"],
                        "sharpe_like": metrics["sharpe_like"],
                        "total_turnover": metrics["total_turnover"],
                        "exposure_changes": exposure_changes,
                        "average_exposure": average_exposure,
                        "buy_hold_cumulative_return": buy_hold_return,
                        "excess_cumulative_return": excess_return,
                        "sharpe_like_delta": sharpe_delta,
                        "drawdown_delta": drawdown_delta,
                        "active_candidate": active_candidate,
                        "passed_gate": passed_gate,
                    }
                    candidate_rows.append(row)
                    fold_candidates.append(row)

        valid_candidates = [row for row in fold_candidates if row["passed_gate"]]
        if not valid_candidates:
            selected_weight_frames.append(
                _cash_weights_for_rows(
                    panel=panel,
                    rows=test_rows,
                    frequency=frequency,
                    strategy_name=ML_TUNED_STRATEGY_NAME,
                )
            )
            selection_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "selection_status": "no_valid_candidate",
                    "selected_model_name": pd.NA,
                    "selected_threshold": pd.NA,
                    "selected_strategy": ML_TUNED_STRATEGY_NAME,
                    "validation_start": (
                        validation_rows["signal_date"].min()
                        if not validation_rows.empty
                        else pd.NaT
                    ),
                    "validation_end": (
                        validation_rows["signal_date"].max()
                        if not validation_rows.empty
                        else pd.NaT
                    ),
                    "inner_train_rows": len(inner_train_rows),
                    "validation_rows": len(validation_rows),
                    "passed_gate": False,
                    "excess_cumulative_return": pd.NA,
                    "sharpe_like_delta": pd.NA,
                    "drawdown_delta": pd.NA,
                    "exposure_changes": 0,
                    "average_exposure": 0.0,
                }
            )
            continue

        selected = sorted(
            valid_candidates,
            key=lambda row: (
                float(row["excess_cumulative_return"]),
                float(row["sharpe_like_delta"]),
                float(row["drawdown_delta"]),
            ),
            reverse=True,
        )[0]
        selected_model = str(selected["model_name"])
        selected_threshold = float(selected["threshold"])
        selected_test_predictions = _score_direction_rows(
            model_name=selected_model,
            target_type=config.target.type,
            train_rows=outer_train_rows,
            score_rows=test_rows,
            feature_columns=feature_columns,
            fold_id=fold.fold_id,
        )
        if selected_test_predictions is None:
            selected_weight_frames.append(
                _cash_weights_for_rows(
                    panel=panel,
                    rows=test_rows,
                    frequency=frequency,
                    strategy_name=ML_TUNED_STRATEGY_NAME,
                )
            )
            selection_status = "no_valid_candidate"
        else:
            selected_weight_frames.append(
                _weights_for_predictions(
                    config=config,
                    panel=panel,
                    predictions=selected_test_predictions,
                    threshold=selected_threshold,
                    strategy_name=ML_TUNED_STRATEGY_NAME,
                )
            )
            selection_status = "selected"

        selection_rows.append(
            {
                "fold_id": fold.fold_id,
                "selection_status": selection_status,
                "selected_model_name": selected_model if selection_status == "selected" else pd.NA,
                "selected_threshold": (
                    selected_threshold if selection_status == "selected" else pd.NA
                ),
                "selected_strategy": ML_TUNED_STRATEGY_NAME,
                "validation_start": selected["validation_start"],
                "validation_end": selected["validation_end"],
                "inner_train_rows": selected["inner_train_rows"],
                "validation_rows": selected["validation_rows"],
                "passed_gate": bool(selected["passed_gate"]) and selection_status == "selected",
                "excess_cumulative_return": selected["excess_cumulative_return"],
                "sharpe_like_delta": selected["sharpe_like_delta"],
                "drawdown_delta": selected["drawdown_delta"],
                "exposure_changes": selected["exposure_changes"],
                "average_exposure": selected["average_exposure"],
            }
        )

    candidates = pd.DataFrame(candidate_rows, columns=candidate_columns)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["fold_id", "passed_gate", "excess_cumulative_return", "model_name", "threshold"],
            ascending=[True, False, False, True, True],
        ).reset_index(drop=True)
    selections = pd.DataFrame(selection_rows, columns=selection_columns)
    if not selections.empty:
        selections = selections.sort_values("fold_id").reset_index(drop=True)
    if not selected_weight_frames:
        return None, candidates, selections

    selected_weights = (
        pd.concat(selected_weight_frames, ignore_index=True)
        .sort_values(["effective_date", "symbol"])
        .drop_duplicates(["strategy", "effective_date", "symbol"], keep="last")
        .reset_index(drop=True)
    )
    tuned_result = run_backtest_detailed(
        panel=panel,
        weights=selected_weights,
        cost_bps=config.portfolio.costs.bps_per_trade,
    )
    return tuned_result, candidates, selections

def train_models(config: ExperimentConfig) -> TrainModelsArtifacts:
    if not config.models:
        raise RuntimeError("No models are configured for train-models.")

    panel, panel_path = prepare_data(config)
    modeling_dataset = build_modeling_dataset(panel, config)
    if modeling_dataset.empty:
        raise RuntimeError("Modeling dataset is empty.")

    run_dir = _run_dir(config)
    fold_diagnostics = build_walk_forward_diagnostics(
        modeling_dataset=modeling_dataset,
        walk_forward=config.evaluation.walk_forward,
        frequency=_walk_forward_frequency(config),
    )
    fold_diagnostics_path = _write_fold_diagnostics(run_dir, fold_diagnostics)
    folds = build_walk_forward_folds(
        modeling_dataset=modeling_dataset,
        walk_forward=config.evaluation.walk_forward,
        frequency=_walk_forward_frequency(config),
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
    (
        baseline_outputs,
        black_litterman_assumptions,
        covariance_diagnostics,
        indicator_diagnostics,
        pattern_diagnostics,
        pattern_exit_overlay_diagnostics_frame,
        pattern_meta_labels_frame,
        pattern_meta_predictions,
        pattern_meta_fold_diagnostics,
        pattern_meta_threshold_sweep,
        pattern_meta_tuning_candidates,
        pattern_meta_tuning_selections,
        pattern_partial_exposure_diagnostics,
        pattern_partial_threshold_sweep,
    ) = run_baselines(
        config,
        panel,
    )
    return _persist_experiment_outputs(
        config=config,
        panel_path=panel_path,
        performance=baseline_outputs.performance,
        daily_holdings=baseline_outputs.daily_holdings,
        daily_cash=baseline_outputs.daily_cash,
        symbol_groups=config.data.symbol_groups,
        covariance_diagnostics=covariance_diagnostics,
        black_litterman_assumptions=black_litterman_assumptions,
        indicator_diagnostics=indicator_diagnostics,
        pattern_diagnostics=pattern_diagnostics,
        pattern_exit_overlay_diagnostics=pattern_exit_overlay_diagnostics_frame,
        pattern_meta_labels_frame=pattern_meta_labels_frame,
        pattern_meta_predictions=pattern_meta_predictions,
        pattern_meta_fold_diagnostics=pattern_meta_fold_diagnostics,
        pattern_meta_threshold_sweep=pattern_meta_threshold_sweep,
        pattern_meta_tuning_candidates=pattern_meta_tuning_candidates,
        pattern_meta_tuning_selections=pattern_meta_tuning_selections,
        pattern_partial_exposure_diagnostics=pattern_partial_exposure_diagnostics,
        pattern_partial_threshold_sweep=pattern_partial_threshold_sweep,
    )


def run_experiment(config: ExperimentConfig) -> ExperimentArtifacts:
    panel, panel_path = prepare_data(config)
    (
        baseline_outputs,
        black_litterman_assumptions,
        covariance_diagnostics,
        indicator_diagnostics,
        pattern_diagnostics,
        pattern_exit_overlay_diagnostics_frame,
        pattern_meta_labels_frame,
        pattern_meta_predictions,
        pattern_meta_fold_diagnostics,
        pattern_meta_threshold_sweep,
        pattern_meta_tuning_candidates,
        pattern_meta_tuning_selections,
        pattern_partial_exposure_diagnostics,
        pattern_partial_threshold_sweep,
    ) = run_baselines(
        config,
        panel,
    )

    if not config.models:
        return _persist_experiment_outputs(
            config=config,
            panel_path=panel_path,
            performance=baseline_outputs.performance,
            daily_holdings=baseline_outputs.daily_holdings,
            daily_cash=baseline_outputs.daily_cash,
            symbol_groups=config.data.symbol_groups,
            covariance_diagnostics=covariance_diagnostics,
            black_litterman_assumptions=black_litterman_assumptions,
            indicator_diagnostics=indicator_diagnostics,
            pattern_diagnostics=pattern_diagnostics,
            pattern_exit_overlay_diagnostics=pattern_exit_overlay_diagnostics_frame,
            pattern_meta_labels_frame=pattern_meta_labels_frame,
            pattern_meta_predictions=pattern_meta_predictions,
            pattern_meta_fold_diagnostics=pattern_meta_fold_diagnostics,
            pattern_meta_threshold_sweep=pattern_meta_threshold_sweep,
            pattern_meta_tuning_candidates=pattern_meta_tuning_candidates,
            pattern_meta_tuning_selections=pattern_meta_tuning_selections,
            pattern_partial_exposure_diagnostics=pattern_partial_exposure_diagnostics,
            pattern_partial_threshold_sweep=pattern_partial_threshold_sweep,
        )

    modeling_dataset = build_modeling_dataset(panel, config)
    if modeling_dataset.empty:
        raise RuntimeError("Modeling dataset is empty.")

    run_dir = _run_dir(config)
    fold_diagnostics = build_walk_forward_diagnostics(
        modeling_dataset=modeling_dataset,
        walk_forward=config.evaluation.walk_forward,
        frequency=_walk_forward_frequency(config),
    )
    fold_diagnostics_path = _write_fold_diagnostics(run_dir, fold_diagnostics)
    folds = build_walk_forward_folds(
        modeling_dataset=modeling_dataset,
        walk_forward=config.evaluation.walk_forward,
        frequency=_walk_forward_frequency(config),
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
    buy_hold_performance = baseline_outputs.performance.loc[
        baseline_outputs.performance["strategy"].astype(str) == "buy_hold"
    ].copy()
    if config.evaluation.ml_strategy_tuning.enabled and buy_hold_performance.empty:
        raise RuntimeError("ML strategy tuning requires the buy_hold baseline.")
    ml_strategy_tuning_result: BacktestResult | None = None
    ml_strategy_tuning_candidates: pd.DataFrame | None = None
    ml_strategy_tuning_selections: pd.DataFrame | None = None
    if config.evaluation.ml_strategy_tuning.enabled:
        (
            ml_strategy_tuning_result,
            ml_strategy_tuning_candidates,
            ml_strategy_tuning_selections,
        ) = _build_ml_strategy_tuning_outputs(
            config=config,
            panel=panel,
            modeling_dataset=modeling_dataset,
            folds=folds,
            buy_hold_performance=buy_hold_performance,
        )
    oos_dates = _shared_oos_dates(
        panel=panel,
        modeling_dataset=modeling_dataset,
        folds=folds,
        frequency=config.portfolio.ranking.rebalance_frequency,
    )
    result_frames = [baseline_outputs, ml_outputs]
    if ml_strategy_tuning_result is not None:
        result_frames.append(ml_strategy_tuning_result)
    combined_outputs = _concat_backtest_results(result_frames)
    oos_outputs = _slice_backtest_result(combined_outputs, oos_dates)
    comparison_metrics = compute_strategy_metrics(
        oos_outputs.performance,
        periods_per_year=config.evaluation.periods_per_year,
    )
    ml_strategy_threshold_sweep = _build_ml_strategy_threshold_sweep(
        config=config,
        panel=panel,
        predictions=training_outputs.predictions,
        oos_dates=oos_dates,
        comparison_metrics=comparison_metrics,
    )
    black_litterman_assumptions = _slice_black_litterman_assumptions(
        black_litterman_assumptions,
        oos_dates,
    )
    covariance_diagnostics = _slice_covariance_diagnostics(
        covariance_diagnostics,
        oos_dates,
    )
    indicator_diagnostics = _slice_indicator_diagnostics(
        indicator_diagnostics,
        oos_dates,
    )
    pattern_diagnostics = _slice_pattern_diagnostics(
        pattern_diagnostics,
        oos_dates,
    )
    pattern_exit_overlay_diagnostics_frame = _slice_pattern_diagnostics(
        pattern_exit_overlay_diagnostics_frame,
        oos_dates,
    )
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
        covariance_diagnostics=covariance_diagnostics,
        black_litterman_assumptions=black_litterman_assumptions,
        indicator_diagnostics=indicator_diagnostics,
        pattern_diagnostics=pattern_diagnostics,
        pattern_exit_overlay_diagnostics=pattern_exit_overlay_diagnostics_frame,
        pattern_meta_labels_frame=pattern_meta_labels_frame,
        pattern_meta_predictions=pattern_meta_predictions,
        pattern_meta_fold_diagnostics=pattern_meta_fold_diagnostics,
        pattern_meta_threshold_sweep=pattern_meta_threshold_sweep,
        pattern_meta_tuning_candidates=pattern_meta_tuning_candidates,
        pattern_meta_tuning_selections=pattern_meta_tuning_selections,
        pattern_partial_exposure_diagnostics=pattern_partial_exposure_diagnostics,
        pattern_partial_threshold_sweep=pattern_partial_threshold_sweep,
        ml_strategy_threshold_sweep=ml_strategy_threshold_sweep,
        ml_strategy_tuning_candidates=ml_strategy_tuning_candidates,
        ml_strategy_tuning_selections=ml_strategy_tuning_selections,
    )








