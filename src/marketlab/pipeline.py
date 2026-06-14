from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from marketlab.backtest.engine import (
    BacktestResult,
    run_backtest,
    run_backtest_detailed,
)
from marketlab.backtest.metrics import compute_strategy_metrics
from marketlab.config import (
    AllocationScoreTransformConfig,
    AllocationUtilityProfileConfig,
    ExperimentConfig,
    RegimeParticipationPolicyConfig,
)
from marketlab.data.market import load_symbol_frames
from marketlab.data.panel import build_market_panel, load_panel_csv, save_panel_csv
from marketlab.evaluation import (
    WalkForwardFold,
    build_walk_forward_folds,
    folds_to_frame,
    slice_fold_rows,
)
from marketlab.evaluation.walk_forward import build_walk_forward_diagnostics
from marketlab.features.engineering import add_feature_set
from marketlab.models import train_direction_models_on_folds
from marketlab.models.registry import (
    build_model_estimator,
    predict_allocation_utility_scores,
    predict_direction_scores,
    predict_regime_state_scores,
)
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
    reprice_performance,
)
from marketlab.reports.markdown import write_markdown_report
from marketlab.reports.phase8_methodology import build_phase8_methodology_review
from marketlab.reports.phase8_summary import build_phase8_run_summary
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
from marketlab.strategies.rebalanced_partial import (
    generate_weights as rebalanced_partial_weights,
)
from marketlab.strategies.rebalanced_partial import (
    strategy_name_for_weight as rebalanced_partial_strategy_name_for_weight,
)
from marketlab.strategies.sma import generate_weights as sma_weights
from marketlab.strategies.static_partial import (
    generate_weights as static_partial_weights,
)
from marketlab.strategies.static_partial import (
    strategy_name_for_weight as static_partial_strategy_name_for_weight,
)
from marketlab.strategies.tiered_allocation import (
    RegimeParticipationPolicy,
    nearest_tier,
)
from marketlab.strategies.tiered_allocation import (
    generate_weights as tiered_allocation_weights,
)
from marketlab.targets import (
    apply_allocation_utility_profile,
    build_modeling_dataset,
    build_scoring_dataset,
)

LOGGER = logging.getLogger(__name__)
ML_TUNED_STRATEGY_NAME = "ml_indicator_tuned__long_only__cash"
BENCHMARK_SELECTION_FAILURE_REASONS = {
    "non_positive_buy_hold_excess",
    "non_positive_required_benchmark_excess",
    "non_positive_validation_cost_benchmark_excess",
}
SCORE_VALIDITY_FAILURE_REASONS = {
    "negative_score_forward_return_correlation",
}


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
    allocation_target_diagnostics_path: Path | None = None
    allocation_probability_diagnostics_path: Path | None = None
    feature_importance_path: Path | None = None
    regime_slice_diagnostics_path: Path | None = None
    strict_research_gate_path: Path | None = None
    phase8_run_summary_path: Path | None = None
    phase8_methodology_review_path: Path | None = None
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


@dataclass(slots=True)
class ModelScoreOutput:
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    calibration_status: str = "not_applicable"


@dataclass(frozen=True, slots=True)
class ShadowCandidateEvaluation:
    selection_source: str
    target_allocation: float
    raw_score: float | None
    selected_tier: float
    regime_classification: str
    diagnostics: dict[str, object]


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


def _last_percentile(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.rank(pct=True).iloc[-1])


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _btc_regime_masks(panel: pd.DataFrame, config: ExperimentConfig) -> dict[str, pd.Index]:
    if panel.empty:
        return {}
    working = (
        panel.sort_values(["symbol", "timestamp"])
        .drop_duplicates(["timestamp"])
        .assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))
        .reset_index(drop=True)
    )
    close = working["adj_close"].astype(float)
    returns = close.pct_change()
    trend_window = max(config.features.crypto_regime_trend_windows or [180])
    vol_window = config.features.crypto_regime_volatility_window
    pct_window = config.features.crypto_regime_percentile_window
    drawdown_window = config.features.crypto_regime_drawdown_window

    moving_average = close.rolling(trend_window, min_periods=trend_window).mean()
    trend_return = close.pct_change(trend_window)
    rolling_high = close.rolling(drawdown_window, min_periods=drawdown_window).max()
    drawdown = (close / rolling_high.replace(0.0, pd.NA)) - 1.0
    realized_vol = returns.rolling(vol_window, min_periods=vol_window).std(ddof=0)
    vol_percentile = realized_vol.rolling(
        pct_window,
        min_periods=min(vol_window, pct_window),
    ).apply(_last_percentile, raw=False)

    dates = pd.Index(working["timestamp"])
    bull = close.ge(moving_average) & trend_return.gt(0.0)
    bear = close.lt(moving_average) & (trend_return.lt(0.0) | drawdown.le(-0.20))
    high_volatility = vol_percentile.ge(0.80)
    sideways = ~(bull.fillna(False) | bear.fillna(False))
    last_date = pd.Timestamp(dates.max())
    recent_start = last_date - pd.DateOffset(
        months=config.evaluation.strict_research_gate.recent_window_months
    )

    return {
        "bull": dates[bull.fillna(False).to_numpy()],
        "bear": dates[bear.fillna(False).to_numpy()],
        "sideways": dates[sideways.fillna(False).to_numpy()],
        "high_volatility": dates[high_volatility.fillna(False).to_numpy()],
        "recent": dates[dates >= recent_start],
    }


def _with_completed_bar_gate_labels(
    rows: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    if rows.empty or "signal_date" not in rows.columns:
        return rows.copy()

    working = rows.copy()
    signal_dates = pd.to_datetime(working["signal_date"], errors="coerce")
    for slice_name, dates in _btc_regime_masks(panel, config).items():
        working[f"gate_{slice_name}"] = signal_dates.isin(pd.to_datetime(list(dates)))
    return working


def _regime_slice_diagnostics(
    *,
    config: ExperimentConfig,
    panel: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "slice_name",
        "strategy",
        "benchmark_strategy",
        "start_date",
        "end_date",
        "periods",
        "cumulative_return",
        "benchmark_cumulative_return",
        "active_return",
        "sharpe_like",
        "benchmark_sharpe_like",
        "max_drawdown",
        "benchmark_max_drawdown",
    ]
    gate = config.evaluation.strict_research_gate
    if not gate.enabled:
        return pd.DataFrame(columns=columns)
    strategies = set(performance["strategy"].astype(str).unique())
    if gate.strategy_name not in strategies or gate.benchmark_strategy not in strategies:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for slice_name, dates in _btc_regime_masks(panel, config).items():
        if len(dates) == 0:
            continue
        slice_performance = performance.loc[
            performance["strategy"].astype(str).isin(
                {gate.strategy_name, gate.benchmark_strategy}
            )
            & pd.to_datetime(performance["date"]).isin(dates)
        ].copy()
        if slice_performance.empty:
            continue
        metrics = compute_strategy_metrics(
            _slice_and_rebase_performance(slice_performance, pd.Index(dates)),
            periods_per_year=config.evaluation.periods_per_year,
        )
        strategy_rows = metrics.loc[metrics["strategy"].astype(str) == gate.strategy_name]
        benchmark_rows = metrics.loc[
            metrics["strategy"].astype(str) == gate.benchmark_strategy
        ]
        if strategy_rows.empty or benchmark_rows.empty:
            continue
        strategy_row = strategy_rows.iloc[0]
        benchmark_row = benchmark_rows.iloc[0]
        slice_dates = pd.to_datetime(slice_performance["date"])
        rows.append(
            {
                "slice_name": slice_name,
                "strategy": gate.strategy_name,
                "benchmark_strategy": gate.benchmark_strategy,
                "start_date": slice_dates.min(),
                "end_date": slice_dates.max(),
                "periods": int(slice_dates.nunique()),
                "cumulative_return": strategy_row["cumulative_return"],
                "benchmark_cumulative_return": benchmark_row["cumulative_return"],
                "active_return": float(strategy_row["cumulative_return"])
                - float(benchmark_row["cumulative_return"]),
                "sharpe_like": strategy_row["sharpe_like"],
                "benchmark_sharpe_like": benchmark_row["sharpe_like"],
                "max_drawdown": strategy_row["max_drawdown"],
                "benchmark_max_drawdown": benchmark_row["max_drawdown"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _gate_benchmark_strategies(config: ExperimentConfig) -> list[str]:
    gate = config.evaluation.strict_research_gate
    benchmark_names = [
        gate.benchmark_strategy,
        *[str(value) for value in gate.required_benchmark_strategies],
    ]
    return list(dict.fromkeys(name for name in benchmark_names if name.strip()))


def _selection_benchmark_strategies(config: ExperimentConfig) -> list[str]:
    tuning = config.evaluation.ml_strategy_tuning
    if tuning.objective in {
        "net_return_and_risk_vs_required_benchmarks",
        "net_return_risk_score_validity_vs_required_benchmarks",
    }:
        benchmark_names = [str(value) for value in tuning.selection_benchmark_strategies]
    else:
        benchmark_names = ["buy_hold"]
    return list(dict.fromkeys(name for name in benchmark_names if name.strip()))


def _condition_suffix(strategy_name: str) -> str:
    return (
        str(strategy_name)
        .strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _strict_research_gate(
    *,
    config: ExperimentConfig,
    strategy_summary: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    regime_slices: pd.DataFrame,
    ml_strategy_tuning_selections: pd.DataFrame | None,
    allocation_target_diagnostics: pd.DataFrame | None = None,
    allocation_probability_diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = ["condition", "passed", "observed", "required"]
    gate = config.evaluation.strict_research_gate
    if not gate.enabled:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []

    def _add(condition: str, passed: bool, observed: object, required: object) -> None:
        rows.append(
            {
                "condition": condition,
                "passed": bool(passed),
                "observed": observed,
                "required": required,
            }
        )

    def _finish() -> pd.DataFrame:
        output = pd.DataFrame(rows, columns=columns)
        output.loc[len(output)] = {
            "condition": "overall",
            "passed": bool(output["passed"].all()) if not output.empty else False,
            "observed": "",
            "required": "all conditions pass",
        }
        return output

    strategy_rows = strategy_summary.loc[
        strategy_summary["strategy"].astype(str) == gate.strategy_name
    ]
    available_strategies = sorted(strategy_summary["strategy"].astype(str).unique())
    if strategy_rows.empty:
        _add(
            "required_strategy_present",
            False,
            ", ".join(available_strategies),
            gate.strategy_name,
        )
        return _finish()
    strategy_row = strategy_rows.iloc[0]

    required_benchmarks = _gate_benchmark_strategies(config)
    missing_benchmarks = [
        name for name in required_benchmarks if name not in set(available_strategies)
    ]
    _add(
        "required_benchmark_strategies_present",
        not missing_benchmarks,
        ", ".join(missing_benchmarks) if missing_benchmarks else "present",
        ", ".join(required_benchmarks),
    )

    primary_benchmark_rows = strategy_summary.loc[
        strategy_summary["strategy"].astype(str) == gate.benchmark_strategy
    ]
    if primary_benchmark_rows.empty:
        _add(
            f"primary_benchmark_present_{_condition_suffix(gate.benchmark_strategy)}",
            False,
            ", ".join(available_strategies),
            gate.benchmark_strategy,
        )
    else:
        benchmark_row = primary_benchmark_rows.iloc[0]
        sharpe_delta = float(strategy_row["sharpe_like"]) - float(
            benchmark_row["sharpe_like"]
        )
        drawdown_delta = float(strategy_row["max_drawdown"]) - float(
            benchmark_row["max_drawdown"]
        )
        _add("sharpe_like_matches_or_improves", sharpe_delta >= 0.0, sharpe_delta, ">= 0")
        _add(
            "max_drawdown_matches_or_improves",
            drawdown_delta >= 0.0,
            drawdown_delta,
            ">= 0",
        )

    for benchmark_name in required_benchmarks:
        benchmark_rows = strategy_summary.loc[
            strategy_summary["strategy"].astype(str) == benchmark_name
        ]
        condition = f"net_cumulative_return_beats_{_condition_suffix(benchmark_name)}"
        if benchmark_rows.empty:
            _add(condition, False, "missing", benchmark_name)
            continue
        benchmark_row = benchmark_rows.iloc[0]
        return_delta = float(strategy_row["cumulative_return"]) - float(
            benchmark_row["cumulative_return"]
        )
        _add(condition, return_delta > 0.0, return_delta, "> 0")

    average_exposure = float(strategy_row["avg_gross_exposure"])
    _add(
        "average_exposure_in_range",
        gate.min_average_exposure <= average_exposure <= gate.max_average_exposure,
        average_exposure,
        f"{gate.min_average_exposure:g} to {gate.max_average_exposure:g}",
    )
    turnover_budget = config.evaluation.ml_strategy_tuning.max_annualized_turnover
    if turnover_budget is not None:
        if "avg_turnover" not in strategy_row.index:
            _add(
                "annualized_turnover_budget",
                False,
                "missing",
                f"<= {turnover_budget:g}",
            )
        else:
            annualized_turnover = (
                float(strategy_row["avg_turnover"]) * config.evaluation.periods_per_year
            )
            _add(
                "annualized_turnover_budget",
                annualized_turnover <= turnover_budget,
                annualized_turnover,
                f"<= {turnover_budget:g}",
            )

    def _add_cost_condition(condition: str, bps: float, benchmark_name: str) -> None:
        scenario = cost_sensitivity.loc[
            cost_sensitivity["strategy"].astype(str).isin(
                {gate.strategy_name, benchmark_name}
            )
            & cost_sensitivity["bps_per_trade"].astype(float).eq(float(bps))
        ]
        strategy_scenario = scenario.loc[scenario["strategy"].astype(str) == gate.strategy_name]
        benchmark_scenario = scenario.loc[scenario["strategy"].astype(str) == benchmark_name]
        if strategy_scenario.empty or benchmark_scenario.empty:
            _add(condition, False, "missing", f"{bps:g} bps scenario for {benchmark_name}")
            return
        cost_delta = float(strategy_scenario.iloc[0]["cumulative_return"]) - float(
            benchmark_scenario.iloc[0]["cumulative_return"]
        )
        _add(condition, cost_delta > 0.0, cost_delta, "> 0")

    _add_cost_condition("cost_gate_bps", gate.cost_gate_bps, gate.benchmark_strategy)
    _add_cost_condition(
        "acceptable_cost_bps",
        gate.acceptable_cost_bps,
        gate.benchmark_strategy,
    )
    for benchmark_name in required_benchmarks:
        suffix = _condition_suffix(benchmark_name)
        _add_cost_condition(
            f"cost_gate_bps_vs_{suffix}",
            gate.cost_gate_bps,
            benchmark_name,
        )
        _add_cost_condition(
            f"acceptable_cost_bps_vs_{suffix}",
            gate.acceptable_cost_bps,
            benchmark_name,
        )

    positive_slices = int(regime_slices["active_return"].gt(0.0).sum()) if not regime_slices.empty else 0
    _add(
        "positive_active_return_regime_slices",
        positive_slices >= gate.min_positive_regime_slices,
        positive_slices,
        f">= {gate.min_positive_regime_slices}",
    )

    selected_folds = 0
    total_folds = 0
    if ml_strategy_tuning_selections is not None and not ml_strategy_tuning_selections.empty:
        total_folds = int(len(ml_strategy_tuning_selections))
        selected_folds = int(
            ml_strategy_tuning_selections["selection_status"].astype(str).eq("selected").sum()
        )
    _add("multiple_selected_walk_forward_folds", selected_folds >= 2, selected_folds, ">= 2")
    selected_fraction = (selected_folds / total_folds) if total_folds > 0 else 0.0
    _add(
        "selected_walk_forward_fold_fraction",
        selected_fraction >= gate.min_selected_fold_fraction,
        selected_fraction,
        f">= {gate.min_selected_fold_fraction:g}",
    )

    for row in _partial_target_support_gate_rows(
        config=config,
        allocation_target_diagnostics=allocation_target_diagnostics,
    ):
        _add(
            str(row["condition"]),
            bool(row["passed"]),
            row["observed"],
            row["required"],
        )

    for row in _predicted_target_support_gate_rows(
        config=config,
        allocation_probability_diagnostics=allocation_probability_diagnostics,
    ):
        _add(
            str(row["condition"]),
            bool(row["passed"]),
            row["observed"],
            row["required"],
        )

    return _finish()


def _partial_target_support_gate_rows(
    *,
    config: ExperimentConfig,
    allocation_target_diagnostics: pd.DataFrame | None,
) -> list[dict[str, object]]:
    gate = config.evaluation.strict_research_gate
    required_weights = [float(value) for value in gate.required_partial_target_weights]
    if not _is_allocation_target(config.target.type) or not required_weights:
        return []

    rows: list[dict[str, object]] = []

    def _row(condition: str, passed: bool, observed: object, required: object) -> None:
        rows.append(
            {
                "condition": condition,
                "passed": bool(passed),
                "observed": observed,
                "required": required,
            }
        )

    if allocation_target_diagnostics is None or allocation_target_diagnostics.empty:
        for target_weight in required_weights:
            suffix = int(target_weight * 100)
            _row(
                f"partial_target_{suffix}_global_fraction",
                False,
                "missing",
                f">= {gate.min_partial_target_fraction:g}",
            )
            _row(
                f"partial_target_{suffix}_fold_fraction",
                False,
                "missing",
                f">= {gate.min_partial_target_fold_fraction:g}",
            )
        return rows

    diagnostics = allocation_target_diagnostics.copy()
    diagnostics["target_weight"] = pd.to_numeric(
        diagnostics["target_weight"], errors="coerce"
    )
    diagnostics["row_count"] = pd.to_numeric(
        diagnostics["row_count"], errors="coerce"
    ).fillna(0)

    if "scope" in diagnostics.columns:
        global_rows = diagnostics.loc[diagnostics["scope"].astype(str).eq("global")]
        fold_rows = diagnostics.loc[
            diagnostics["scope"].astype(str).eq("train_validation")
        ]
    else:
        global_rows = diagnostics
        fold_rows = diagnostics

    total_global_rows = float(global_rows["row_count"].sum())
    eligible_folds = sorted(str(value) for value in fold_rows["fold_id"].dropna().unique())
    for target_weight in required_weights:
        suffix = int(target_weight * 100)
        target_global_rows = global_rows.loc[
            global_rows["target_weight"].sub(target_weight).abs().le(1e-9)
        ]
        global_count = float(target_global_rows["row_count"].sum())
        global_fraction = (
            global_count / total_global_rows if total_global_rows > 0.0 else 0.0
        )
        _row(
            f"partial_target_{suffix}_global_fraction",
            global_fraction >= gate.min_partial_target_fraction,
            global_fraction,
            f">= {gate.min_partial_target_fraction:g}",
        )

        present_folds = {
            str(value)
            for value in fold_rows.loc[
                fold_rows["target_weight"].sub(target_weight).abs().le(1e-9)
                & fold_rows["row_count"].gt(0),
                "fold_id",
            ].dropna()
        }
        fold_fraction = (
            len(present_folds) / len(eligible_folds) if eligible_folds else 0.0
        )
        _row(
            f"partial_target_{suffix}_fold_fraction",
            fold_fraction >= gate.min_partial_target_fold_fraction,
            fold_fraction,
            f">= {gate.min_partial_target_fold_fraction:g}",
        )

    return rows


def _predicted_target_support_gate_rows(
    *,
    config: ExperimentConfig,
    allocation_probability_diagnostics: pd.DataFrame | None,
) -> list[dict[str, object]]:
    gate = config.evaluation.strict_research_gate
    required_weights = [float(value) for value in gate.required_predicted_target_weights]
    if not _is_allocation_target(config.target.type) or not required_weights:
        return []

    rows: list[dict[str, object]] = []

    def _row(condition: str, passed: bool, observed: object, required: object) -> None:
        rows.append(
            {
                "condition": condition,
                "passed": bool(passed),
                "observed": observed,
                "required": required,
            }
        )

    if allocation_probability_diagnostics is None or allocation_probability_diagnostics.empty:
        for target_weight in required_weights:
            suffix = int(target_weight * 100)
            _row(
                f"predicted_target_{suffix}_global_fraction",
                False,
                "missing",
                f">= {gate.min_predicted_target_fraction:g}",
            )
            _row(
                f"predicted_target_{suffix}_fold_fraction",
                False,
                "missing",
                f">= {gate.min_predicted_target_fold_fraction:g}",
            )
        return rows

    diagnostics = allocation_probability_diagnostics.copy()
    if "predicted_tier_weight" in diagnostics.columns:
        predicted_tier_weight = pd.to_numeric(
            diagnostics["predicted_tier_weight"],
            errors="coerce",
        )
    elif "score" in diagnostics.columns:
        predicted_tier_weight = pd.to_numeric(diagnostics["score"], errors="coerce").map(
            lambda value: nearest_tier(float(value)) if pd.notna(value) else pd.NA
        )
    else:
        predicted_tier_weight = pd.Series(pd.NA, index=diagnostics.index)
    diagnostics["predicted_tier_weight"] = predicted_tier_weight
    total_rows = int(diagnostics["predicted_tier_weight"].notna().sum())
    eligible_folds = sorted(str(value) for value in diagnostics["fold_id"].dropna().unique())

    for target_weight in required_weights:
        suffix = int(target_weight * 100)
        matches = diagnostics["predicted_tier_weight"].sub(target_weight).abs().le(1e-9)
        global_fraction = float(matches.sum() / total_rows) if total_rows > 0 else 0.0
        _row(
            f"predicted_target_{suffix}_global_fraction",
            global_fraction >= gate.min_predicted_target_fraction,
            global_fraction,
            f">= {gate.min_predicted_target_fraction:g}",
        )

        present_folds = {
            str(value)
            for value in diagnostics.loc[matches, "fold_id"].dropna().unique()
        }
        fold_fraction = (
            len(present_folds) / len(eligible_folds) if eligible_folds else 0.0
        )
        _row(
            f"predicted_target_{suffix}_fold_fraction",
            fold_fraction >= gate.min_predicted_target_fold_fraction,
            fold_fraction,
            f">= {gate.min_predicted_target_fold_fraction:g}",
        )

    return rows


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
    allocation_target_diagnostics: pd.DataFrame | None = None,
    allocation_probability_diagnostics: pd.DataFrame | None = None,
    feature_importance: pd.DataFrame | None = None,
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
    regime_slice_diagnostics = _regime_slice_diagnostics(
        config=config,
        panel=load_panel_csv(panel_path) if panel_path.exists() else pd.DataFrame(),
        performance=performance,
    )
    strict_research_gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=regime_slice_diagnostics,
        ml_strategy_tuning_selections=ml_strategy_tuning_selections,
        allocation_target_diagnostics=allocation_target_diagnostics,
        allocation_probability_diagnostics=allocation_probability_diagnostics,
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

    regime_slice_diagnostics_path: Path | None = None
    if not regime_slice_diagnostics.empty:
        regime_slice_diagnostics_path = artifact_run_dir / "regime_slice_diagnostics.csv"
        regime_slice_diagnostics.to_csv(regime_slice_diagnostics_path, index=False)

    strict_research_gate_path: Path | None = None
    if not strict_research_gate.empty:
        strict_research_gate_path = artifact_run_dir / "strict_research_gate.csv"
        strict_research_gate.to_csv(strict_research_gate_path, index=False)

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

    allocation_target_diagnostics_path: Path | None = None
    if allocation_target_diagnostics is not None:
        allocation_target_diagnostics_path = artifact_run_dir / "allocation_target_diagnostics.csv"
        allocation_target_diagnostics.to_csv(allocation_target_diagnostics_path, index=False)

    allocation_probability_diagnostics_path: Path | None = None
    if allocation_probability_diagnostics is not None:
        allocation_probability_diagnostics_path = (
            artifact_run_dir / "allocation_probability_diagnostics.csv"
        )
        allocation_probability_diagnostics.to_csv(
            allocation_probability_diagnostics_path,
            index=False,
        )

    feature_importance_path: Path | None = None
    if feature_importance is not None:
        feature_importance_path = artifact_run_dir / "feature_importance.csv"
        feature_importance.to_csv(feature_importance_path, index=False)

    model_summary_path: Path | None = None
    if model_summary is not None:
        model_summary_path = artifact_run_dir / "model_summary.csv"
        model_summary.to_csv(model_summary_path, index=False)

    fold_summary_path: Path | None = None
    if fold_summary is not None:
        fold_summary_path = artifact_run_dir / "fold_summary.csv"
        fold_summary.to_csv(fold_summary_path, index=False)

    phase8_run_summary: pd.DataFrame | None = None
    phase8_run_summary_path: Path | None = None
    phase8_methodology_review: pd.DataFrame | None = None
    phase8_methodology_review_path: Path | None = None
    if (
        config.evaluation.strict_research_gate.enabled
        or ml_strategy_tuning_candidates is not None
        or allocation_target_diagnostics is not None
        or allocation_probability_diagnostics is not None
    ):
        phase8_run_summary = build_phase8_run_summary(artifact_run_dir)
        phase8_run_summary_path = artifact_run_dir / "phase8_run_summary.csv"
        phase8_run_summary.to_csv(phase8_run_summary_path, index=False)
        phase8_methodology_review = build_phase8_methodology_review(artifact_run_dir)
        phase8_methodology_review_path = artifact_run_dir / "phase8_methodology_review.csv"
        phase8_methodology_review.to_csv(phase8_methodology_review_path, index=False)

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
            regime_slice_diagnostics=regime_slice_diagnostics,
            strict_research_gate=strict_research_gate,
            phase8_run_summary=phase8_run_summary,
            phase8_methodology_review=phase8_methodology_review,
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
            allocation_target_diagnostics=allocation_target_diagnostics,
            allocation_probability_diagnostics=allocation_probability_diagnostics,
            feature_importance=feature_importance,
            allocation_target_diagnostics_path=allocation_target_diagnostics_path,
            allocation_probability_diagnostics_path=allocation_probability_diagnostics_path,
            feature_importance_path=feature_importance_path,
            regime_slice_diagnostics_path=regime_slice_diagnostics_path,
            strict_research_gate_path=strict_research_gate_path,
            phase8_run_summary_path=phase8_run_summary_path,
            phase8_methodology_review_path=phase8_methodology_review_path,
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
        allocation_target_diagnostics_path=allocation_target_diagnostics_path,
        allocation_probability_diagnostics_path=allocation_probability_diagnostics_path,
        feature_importance_path=feature_importance_path,
        regime_slice_diagnostics_path=regime_slice_diagnostics_path,
        strict_research_gate_path=strict_research_gate_path,
        phase8_run_summary_path=phase8_run_summary_path,
        phase8_methodology_review_path=phase8_methodology_review_path,
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

    if config.baselines.partial_allocation_benchmarks.enabled:
        for target_weight in config.baselines.partial_allocation_benchmarks.weights:
            weights = static_partial_weights(
                featured,
                target_weight=float(target_weight),
                strategy_name=static_partial_strategy_name_for_weight(float(target_weight)),
            )
            if not weights.empty:
                backtest_results.append(
                    run_backtest_detailed(
                        panel=featured,
                        weights=weights,
                        cost_bps=config.portfolio.costs.bps_per_trade,
                    )
                )

    if config.baselines.rebalanced_partial_allocation_benchmarks.enabled:
        for target_weight in config.baselines.rebalanced_partial_allocation_benchmarks.weights:
            weights = rebalanced_partial_weights(
                featured,
                target_weight=float(target_weight),
                frequency=config.portfolio.ranking.rebalance_frequency,
                strategy_name=rebalanced_partial_strategy_name_for_weight(float(target_weight)),
            )
            if not weights.empty:
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
    optional_columns = [
        "forward_drawdown",
        "forward_realized_volatility",
        "target_weight",
        "allocation_utility_0",
        "allocation_utility_25",
        "allocation_utility_50",
        "allocation_utility_100",
    ]
    for column in optional_columns:
        if column in rows.columns:
            prediction_frame[column] = rows[column].to_numpy()
    prediction_frame.insert(0, "fold_id", fold_id)
    prediction_frame.insert(0, "model_name", model_name)
    prediction_frame["score"] = score_series.to_numpy()
    prediction_frame["predicted_target"] = predicted_target.to_numpy()
    regime_columns = [column for column in rows.columns if str(column).startswith("crypto_regime_")]
    for column in regime_columns:
        prediction_frame[column] = rows[column].to_numpy()
    gate_columns = [column for column in rows.columns if str(column).startswith("gate_")]
    for column in gate_columns:
        prediction_frame[column] = rows[column].to_numpy()
    return prediction_frame


def _estimator_feature_importance(
    *,
    estimator: Any,
    model_name: str,
    fold_id: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    fitted = estimator
    if hasattr(estimator, "steps") and estimator.steps:
        fitted = estimator.steps[-1][1]

    rows: list[dict[str, object]] = []
    if hasattr(fitted, "feature_importances_"):
        importances = list(getattr(fitted, "feature_importances_"))
        rows = [
            {
                "model_name": model_name,
                "fold_id": fold_id,
                "feature": feature,
                "importance_type": "feature_importance",
                "importance": float(importance),
                "signed_coefficient": pd.NA,
            }
            for feature, importance in zip(feature_columns, importances)
        ]
    elif hasattr(fitted, "coef_"):
        coefficients = getattr(fitted, "coef_")
        if getattr(coefficients, "ndim", 1) == 1:
            signed_values = list(coefficients)
            importance_values = [abs(float(value)) for value in signed_values]
        else:
            signed_values = [float(values.mean()) for values in coefficients.T]
            importance_values = [
                float(pd.Series(values).abs().mean()) for values in coefficients.T
            ]
        rows = [
            {
                "model_name": model_name,
                "fold_id": fold_id,
                "feature": feature,
                "importance_type": "coefficient",
                "importance": float(importance),
                "signed_coefficient": float(signed),
            }
            for feature, importance, signed in zip(
                feature_columns,
                importance_values,
                signed_values,
            )
        ]

    if not rows:
        return pd.DataFrame(
            columns=[
                "model_name",
                "fold_id",
                "feature",
                "importance_type",
                "importance",
                "signed_coefficient",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["importance", "feature"], ascending=[False, True])
        .head(25)
        .reset_index(drop=True)
    )


def _is_allocation_target(target_type: str) -> bool:
    return target_type in {"allocation_utility", "regime_state"}


def _allocation_profile_candidates(config: ExperimentConfig) -> list[AllocationUtilityProfileConfig]:
    if not _is_allocation_target(config.target.type):
        return [
            AllocationUtilityProfileConfig(
                name="not_applicable",
                drawdown_penalty=config.target.allocation_utility_drawdown_penalty,
                volatility_penalty=config.target.allocation_utility_volatility_penalty,
                risk_penalty_power=config.target.allocation_utility_risk_penalty_power,
            )
        ]
    profiles = config.evaluation.ml_strategy_tuning.allocation_utility_profiles
    if profiles:
        return list(profiles)
    return [
        AllocationUtilityProfileConfig(
            name="target_config",
            drawdown_penalty=config.target.allocation_utility_drawdown_penalty,
            volatility_penalty=config.target.allocation_utility_volatility_penalty,
            risk_penalty_power=config.target.allocation_utility_risk_penalty_power,
        )
    ]


def _apply_allocation_profile(
    rows: pd.DataFrame,
    *,
    config: ExperimentConfig,
    profile: AllocationUtilityProfileConfig,
) -> pd.DataFrame:
    if not _is_allocation_target(config.target.type):
        return rows.copy()
    return apply_allocation_utility_profile(
        rows,
        target_type=config.target.type,
        cost_bps=config.portfolio.costs.bps_per_trade,
        drawdown_penalty=float(profile.drawdown_penalty),
        volatility_penalty=float(profile.volatility_penalty),
        risk_penalty_power=float(profile.risk_penalty_power),
    )


def _prediction_weight_map(target_type: str) -> dict[int, float]:
    if target_type == "regime_state":
        return {0: 0.0, 1: 0.50, 2: 1.0}
    return {0: 0.0, 1: 0.25, 2: 0.50, 3: 1.0}


def _allocation_sample_weights(
    *,
    train_target: pd.Series,
    target_type: str,
    class_weighting: str,
    partial_class_multiplier: float,
) -> pd.Series | None:
    if not _is_allocation_target(target_type) or class_weighting == "none":
        return None

    counts = train_target.value_counts()
    if counts.empty:
        return None
    row_count = float(len(train_target))
    class_count = float(len(counts))
    weights = train_target.map(
        lambda value: row_count / (class_count * float(counts.loc[int(value)]))
    ).astype(float)

    if class_weighting == "balanced_partial_boost":
        partial_classes = {1} if target_type == "regime_state" else {1, 2}
        weights.loc[train_target.isin(partial_classes)] *= float(partial_class_multiplier)
    return weights


def _fit_estimator_with_sample_weights(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    sample_weights: pd.Series | None,
) -> None:
    if sample_weights is None:
        estimator.fit(features, target)
        return

    if hasattr(estimator, "steps") and estimator.steps:
        step_name = str(estimator.steps[-1][0])
        estimator.fit(
            features,
            target,
            **{f"{step_name}__sample_weight": sample_weights.to_numpy(dtype=float)},
        )
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Since .* does not appear to accept sample_weight.*",
            category=UserWarning,
        )
        estimator.fit(features, target, sample_weight=sample_weights.to_numpy(dtype=float))


def _calibration_status(
    *,
    train_target: pd.Series,
    target_type: str,
    calibration: str,
    calibration_cv: int,
) -> str:
    if not _is_allocation_target(target_type):
        return "not_applicable"
    if calibration == "none":
        return "not_requested"
    counts = train_target.value_counts()
    if train_target.nunique(dropna=True) < 2 or counts.empty or int(counts.min()) < calibration_cv:
        return "skipped_insufficient_class_support"
    return calibration


def _predicted_tier_support(
    *,
    predictions: pd.DataFrame,
    required_weights: list[float],
    min_fraction: float,
) -> tuple[bool, dict[float, float]]:
    if not required_weights:
        return True, {}
    if predictions.empty or "score" not in predictions.columns:
        return False, {float(weight): 0.0 for weight in required_weights}
    predicted_tiers = predictions["score"].map(lambda value: nearest_tier(float(value)))
    fractions = {
        float(weight): float(predicted_tiers.sub(float(weight)).abs().le(1e-9).mean())
        for weight in required_weights
    }
    return all(fraction >= min_fraction for fraction in fractions.values()), fractions


def _apply_allocation_score_policy(
    *,
    rows: pd.DataFrame,
    score_series: pd.Series,
    probability_frame: pd.DataFrame,
    allocation_score_policy: str,
    prob100_threshold: float,
    score_policy_repair_authorized: bool = False,
) -> tuple[pd.Series, pd.Series]:
    final_scores = score_series.astype(float).copy()
    triggered_100 = pd.Series(False, index=score_series.index, dtype=bool)
    if allocation_score_policy == "expected_allocation":
        return final_scores.rename("score"), triggered_100
    if allocation_score_policy not in {
        "bull_prob100_threshold",
        "gate_bull_prob100_threshold",
    }:
        raise ValueError(f"Unsupported allocation score policy: {allocation_score_policy}")

    required_columns = {"prob_tier_0", "prob_tier_100"}
    if not required_columns.issubset(probability_frame.columns):
        return final_scores.rename("score"), triggered_100

    prob_tier_100 = probability_frame["prob_tier_100"].astype(float)
    prob_tier_0 = probability_frame["prob_tier_0"].astype(float)
    if allocation_score_policy == "gate_bull_prob100_threshold":
        if not score_policy_repair_authorized:
            return final_scores.rename("score"), triggered_100
        eligible_rows = rows.get(
            "gate_bull",
            pd.Series(False, index=rows.index, dtype=bool),
        ).map(_truthy)
    else:
        eligible_rows = rows.apply(_allocation_regime_label, axis=1).eq("bull")
    triggered_100 = (
        eligible_rows
        & prob_tier_100.ge(float(prob100_threshold))
        & prob_tier_100.ge(prob_tier_0)
    )
    final_scores.loc[triggered_100] = 1.0
    return final_scores.rename("score"), triggered_100.rename("score_policy_triggered_100")


def _score_policy_repair_authorization(
    *,
    allocation_score_policy: str,
    validation_raw_score_forward_return_correlation: object,
) -> tuple[bool, str]:
    if allocation_score_policy != "gate_bull_prob100_threshold":
        return False, ""
    if pd.isna(validation_raw_score_forward_return_correlation):
        return False, "non_finite_validation_raw_score_forward_return_correlation"
    correlation = float(validation_raw_score_forward_return_correlation)
    if not math.isfinite(correlation):
        return False, "non_finite_validation_raw_score_forward_return_correlation"
    if correlation < 0.0:
        return False, "negative_validation_raw_score_forward_return_correlation"
    return True, ""


def _guarded_gate_bull_risk_off_override_authorization(
    *,
    enabled: bool,
    validation_raw_score_forward_return_correlation: object,
) -> tuple[bool, str]:
    if not enabled:
        return False, ""
    if pd.isna(validation_raw_score_forward_return_correlation):
        return False, "non_finite_validation_raw_score_forward_return_correlation"
    correlation = float(validation_raw_score_forward_return_correlation)
    if not math.isfinite(correlation):
        return False, "non_finite_validation_raw_score_forward_return_correlation"
    if correlation < 0.0:
        return False, "negative_validation_raw_score_forward_return_correlation"
    return True, ""


def _apply_allocation_score_transform(
    *,
    rows: pd.DataFrame,
    score_series: pd.Series,
    score_transform: AllocationScoreTransformConfig,
) -> tuple[pd.Series, pd.Series]:
    base_scores = pd.to_numeric(score_series, errors="coerce").fillna(0.0).astype(float)
    transformed = base_scores.copy()
    runtime_regime = rows.apply(_allocation_regime_label, axis=1)
    bull_rows = runtime_regime.eq("bull")
    transformed.loc[bull_rows] = (
        transformed.loc[bull_rows].mul(float(score_transform.bull_multiplier))
        + float(score_transform.bull_addend)
    )
    if score_transform.risk_off_score_cap is not None:
        risk_off_rows = runtime_regime.eq("risk_off")
        transformed.loc[risk_off_rows] = transformed.loc[risk_off_rows].clip(
            upper=float(score_transform.risk_off_score_cap)
        )
    if score_transform.non_bull_score_cap is not None:
        non_bull_rows = ~bull_rows
        transformed.loc[non_bull_rows] = transformed.loc[non_bull_rows].clip(
            upper=float(score_transform.non_bull_score_cap)
        )
    transformed = transformed.clip(lower=0.0, upper=1.0)
    applied = transformed.sub(base_scores).abs().gt(1e-12)
    return transformed.rename("score"), applied.rename("score_transform_applied")


def _score_transform_metadata(
    score_transform: AllocationScoreTransformConfig,
) -> dict[str, object]:
    return {
        "allocation_score_transform": score_transform.name,
        "score_transform_bull_multiplier": float(score_transform.bull_multiplier),
        "score_transform_bull_addend": float(score_transform.bull_addend),
        "score_transform_risk_off_score_cap": (
            float(score_transform.risk_off_score_cap)
            if score_transform.risk_off_score_cap is not None
            else pd.NA
        ),
        "score_transform_non_bull_score_cap": (
            float(score_transform.non_bull_score_cap)
            if score_transform.non_bull_score_cap is not None
            else pd.NA
        ),
    }


def _selected_score_transform_metadata(
    score_transform: AllocationScoreTransformConfig | None,
) -> dict[str, object]:
    if score_transform is None:
        return {
            "selected_allocation_score_transform": pd.NA,
            "selected_score_transform_bull_multiplier": pd.NA,
            "selected_score_transform_bull_addend": pd.NA,
            "selected_score_transform_risk_off_score_cap": pd.NA,
            "selected_score_transform_non_bull_score_cap": pd.NA,
        }
    return {
        "selected_allocation_score_transform": score_transform.name,
        "selected_score_transform_bull_multiplier": float(score_transform.bull_multiplier),
        "selected_score_transform_bull_addend": float(score_transform.bull_addend),
        "selected_score_transform_risk_off_score_cap": (
            float(score_transform.risk_off_score_cap)
            if score_transform.risk_off_score_cap is not None
            else pd.NA
        ),
        "selected_score_transform_non_bull_score_cap": (
            float(score_transform.non_bull_score_cap)
            if score_transform.non_bull_score_cap is not None
            else pd.NA
        ),
    }


def _score_model_rows(
    *,
    model_name: str,
    target_type: str,
    train_rows: pd.DataFrame,
    score_rows: pd.DataFrame,
    feature_columns: list[str],
    fold_id: int,
    allocation_class_weighting: str = "none",
    allocation_partial_class_weight_multiplier: float = 1.0,
    allocation_probability_calibration: str = "none",
    allocation_calibration_cv: int = 3,
    allocation_score_policy: str = "expected_allocation",
    allocation_score_policy_prob100_threshold: float = 0.20,
    score_policy_repair_authorized: bool = False,
    score_policy_repair_denied_reason: str = "",
    allocation_score_transform: AllocationScoreTransformConfig | None = None,
    utility_profile: AllocationUtilityProfileConfig | None = None,
) -> ModelScoreOutput | None:
    train_target = train_rows["target"].astype(int)
    if train_target.nunique(dropna=True) < 2:
        return None

    train_features = train_rows.loc[:, feature_columns]
    score_features = score_rows.loc[:, feature_columns]
    sample_weights = _allocation_sample_weights(
        train_target=train_target,
        target_type=target_type,
        class_weighting=allocation_class_weighting,
        partial_class_multiplier=allocation_partial_class_weight_multiplier,
    )

    _, base_estimator = build_model_estimator(model_name, target_type)
    _fit_estimator_with_sample_weights(
        base_estimator,
        train_features,
        train_target,
        sample_weights,
    )
    feature_importance = _estimator_feature_importance(
        estimator=base_estimator,
        model_name=model_name,
        fold_id=fold_id,
        feature_columns=feature_columns,
    )

    calibration_status = _calibration_status(
        train_target=train_target,
        target_type=target_type,
        calibration=allocation_probability_calibration,
        calibration_cv=int(allocation_calibration_cv),
    )
    estimator = base_estimator
    if calibration_status == "sigmoid":
        _, calibration_estimator = build_model_estimator(model_name, target_type)
        estimator = CalibratedClassifierCV(
            calibration_estimator,
            method="sigmoid",
            cv=int(allocation_calibration_cv),
        )
        _fit_estimator_with_sample_weights(
            estimator,
            train_features,
            train_target,
            sample_weights,
        )

    probability_frame: pd.DataFrame | None = None
    if target_type == "allocation_utility":
        score_series, probability_frame = predict_allocation_utility_scores(
            estimator,
            score_features,
        )
    elif target_type == "regime_state":
        score_series, probability_frame = predict_regime_state_scores(
            estimator,
            score_features,
        )
    else:
        score_series = predict_direction_scores(estimator, score_features)
    raw_expected_allocation_score = score_series.astype(float).rename(
        "raw_expected_allocation_score"
    )
    selected_score_transform = allocation_score_transform or AllocationScoreTransformConfig()
    score_policy_triggered_100 = pd.Series(False, index=score_series.index, dtype=bool)
    score_transform_applied = pd.Series(False, index=score_series.index, dtype=bool)
    if probability_frame is not None:
        score_series, score_policy_triggered_100 = _apply_allocation_score_policy(
            rows=score_rows,
            score_series=score_series,
            probability_frame=probability_frame,
            allocation_score_policy=allocation_score_policy,
            prob100_threshold=allocation_score_policy_prob100_threshold,
            score_policy_repair_authorized=score_policy_repair_authorized,
        )
        score_series, score_transform_applied = _apply_allocation_score_transform(
            rows=score_rows,
            score_series=score_series,
            score_transform=selected_score_transform,
        )
    predicted_target = pd.Series(
        estimator.predict(score_features),
        index=score_rows.index,
        name="predicted_target",
        dtype=int,
    )
    predictions = _prediction_frame_for_rows(
        model_name=model_name,
        fold_id=fold_id,
        rows=score_rows,
        score_series=score_series,
        predicted_target=predicted_target,
    )
    if probability_frame is not None:
        for column in probability_frame.columns:
            predictions[column] = probability_frame[column].to_numpy()
        predictions["allocation_score_policy"] = allocation_score_policy
        predictions["allocation_score_policy_prob100_threshold"] = float(
            allocation_score_policy_prob100_threshold
        )
        predictions["score_policy_repair_authorized"] = bool(
            score_policy_repair_authorized
        )
        predictions["score_policy_repair_denied_reason"] = (
            score_policy_repair_denied_reason
        )
        for column, value in _score_transform_metadata(selected_score_transform).items():
            predictions[column] = value
        predictions["raw_expected_allocation_score"] = raw_expected_allocation_score.to_numpy()
        predictions["final_allocation_score"] = predictions["score"].to_numpy()
        predictions["score_policy_triggered_100"] = score_policy_triggered_100.to_numpy(
            dtype=bool
        )
        predictions["score_transform_applied"] = score_transform_applied.to_numpy(
            dtype=bool
        )
        predictions["predicted_weight"] = predictions["predicted_target"].map(
            _prediction_weight_map(target_type)
        )
        predictions["predicted_tier_weight"] = predictions["score"].map(
            lambda value: nearest_tier(float(value))
        )
        predictions["calibration_status"] = calibration_status
        if utility_profile is not None:
            predictions["utility_profile"] = utility_profile.name
    return ModelScoreOutput(
        predictions=predictions,
        feature_importance=feature_importance,
        calibration_status=calibration_status,
    )


def _score_direction_rows(
    *,
    model_name: str,
    target_type: str,
    train_rows: pd.DataFrame,
    score_rows: pd.DataFrame,
    feature_columns: list[str],
    fold_id: int,
) -> pd.DataFrame | None:
    output = _score_model_rows(
        model_name=model_name,
        target_type=target_type,
        train_rows=train_rows,
        score_rows=score_rows,
        feature_columns=feature_columns,
        fold_id=fold_id,
    )
    return None if output is None else output.predictions


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


def _strategy_metrics_for_window(
    *,
    performance: pd.DataFrame,
    strategy_name: str,
    window_dates: pd.Index,
    periods_per_year: float,
) -> pd.Series:
    benchmark_window = performance.loc[
        performance["strategy"].astype(str).eq(strategy_name)
        & pd.to_datetime(performance["date"]).isin(window_dates)
    ].copy()
    if benchmark_window.empty:
        raise RuntimeError(
            f"ML strategy tuning produced no {strategy_name} benchmark rows."
        )
    return compute_strategy_metrics(
        _slice_and_rebase_performance(benchmark_window, window_dates),
        periods_per_year=periods_per_year,
    ).iloc[0]


def _benchmark_metrics_for_window(
    *,
    buy_hold_performance: pd.DataFrame,
    window_dates: pd.Index,
    periods_per_year: float,
) -> pd.Series:
    return _strategy_metrics_for_window(
        performance=buy_hold_performance,
        strategy_name="buy_hold",
        window_dates=window_dates,
        periods_per_year=periods_per_year,
    )


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
    tier_thresholds: tuple[float, float, float] | None = None,
    min_holding_period_bars: int = 0,
    hysteresis_margin: float = 0.0,
    direct_tiered: bool = False,
    regime_policy: RegimeParticipationPolicy | None = None,
) -> pd.DataFrame:
    if tier_thresholds is not None or direct_tiered:
        weights = tiered_allocation_weights(
            predictions=predictions,
            panel=panel,
            thresholds=tier_thresholds or (0.25, 0.50, 0.75),
            frequency=config.portfolio.ranking.rebalance_frequency,
            strategy_name=strategy_name,
            max_long_exposure=config.portfolio.risk.max_long_exposure,
            min_holding_period_bars=min_holding_period_bars,
            hysteresis_margin=hysteresis_margin,
            direct_scores=direct_tiered,
            regime_policy=regime_policy,
        )
        return weights

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


def _tier_threshold_sets(config: ExperimentConfig) -> list[tuple[float, float, float]]:
    tuning = config.evaluation.ml_strategy_tuning
    configured_sets = tuning.tier_threshold_sets or [tuning.tier_thresholds]
    unique_sets: list[tuple[float, float, float]] = []
    for threshold_set in configured_sets:
        if len(threshold_set) != 3:
            raise ValueError("ML tiered allocation threshold sets must contain three values.")
        resolved = tuple(float(value) for value in threshold_set)
        if resolved not in unique_sets:
            unique_sets.append(resolved)
    return unique_sets


def _rolling_train_bars_candidates(config: ExperimentConfig) -> list[int | None]:
    values = sorted(
        {
            int(value)
            for value in config.evaluation.ml_strategy_tuning.rolling_train_bars_grid
        }
    )
    return values or [None]


def _min_holding_period_candidates(config: ExperimentConfig) -> list[int]:
    values = sorted(
        {
            int(value)
            for value in config.evaluation.ml_strategy_tuning.min_holding_period_bars_grid
        }
    )
    return values or [0]


def _hysteresis_margin_candidates(config: ExperimentConfig) -> list[float]:
    values = sorted(
        {
            float(value)
            for value in config.evaluation.ml_strategy_tuning.hysteresis_margin_grid
        }
    )
    return values or [0.0]


def _regime_participation_policy_candidates(
    config: ExperimentConfig,
) -> list[RegimeParticipationPolicyConfig]:
    policies = config.evaluation.ml_strategy_tuning.regime_participation_policies
    return policies or [RegimeParticipationPolicyConfig()]


def _strategy_regime_policy(
    policy: RegimeParticipationPolicyConfig,
) -> RegimeParticipationPolicy:
    return RegimeParticipationPolicy(
        name=policy.name,
        bull_floor=float(policy.bull_floor),
        sideways_floor=float(policy.sideways_floor),
        bear_floor=float(policy.bear_floor),
        risk_off_cap=(
            float(policy.risk_off_cap)
            if policy.risk_off_cap is not None
            else None
        ),
        gate_bull_floor=(
            float(policy.gate_bull_floor)
            if policy.gate_bull_floor is not None
            else None
        ),
    )


def _no_candidate_fallback_regime_policy(
    config: ExperimentConfig,
) -> RegimeParticipationPolicyConfig | None:
    policy_name = config.evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy
    if policy_name is None:
        return None
    resolved_name = str(policy_name).strip()
    if not resolved_name:
        raise ValueError(
            "evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy must be non-empty when configured."
        )
    for policy in _regime_participation_policy_candidates(config):
        if policy.name == resolved_name:
            if policy.risk_off_cap is None:
                raise ValueError(
                    "evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy "
                    "requires the referenced policy to define risk_off_cap."
                )
            return policy
    allowed = ", ".join(
        sorted(policy.name for policy in _regime_participation_policy_candidates(config))
    )
    raise ValueError(
        "evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy "
        f"must reference one of regime_participation_policies: {allowed}."
    )


def _allocation_score_policy_prob100_threshold_candidates(
    config: ExperimentConfig,
) -> list[float]:
    tuning = config.evaluation.ml_strategy_tuning
    if (
        tuning.allocation_score_policy
        in {"bull_prob100_threshold", "gate_bull_prob100_threshold"}
        and tuning.allocation_score_policy_prob100_threshold_grid
    ):
        return sorted(
            {float(value) for value in tuning.allocation_score_policy_prob100_threshold_grid}
        )
    return [float(tuning.allocation_score_policy_prob100_threshold)]


def _selection_validation_cost_bps(config: ExperimentConfig) -> list[float]:
    configured = config.evaluation.ml_strategy_tuning.selection_validation_cost_bps
    return configured or [float(config.portfolio.costs.bps_per_trade)]


def _allocation_score_transform_candidates(
    config: ExperimentConfig,
) -> list[AllocationScoreTransformConfig]:
    transforms = config.evaluation.ml_strategy_tuning.allocation_score_transforms
    return transforms or [AllocationScoreTransformConfig()]


def _prediction_frame_with_allocation_score_policy(
    *,
    predictions: pd.DataFrame,
    allocation_score_policy: str,
    prob100_threshold: float,
    score_policy_repair_authorized: bool = False,
    score_policy_repair_denied_reason: str = "",
    guarded_gate_bull_risk_off_override_enabled: bool = False,
    guarded_gate_bull_risk_off_override_authorized: bool = False,
    guarded_gate_bull_risk_off_override_denied_reason: str = "",
    score_transform: AllocationScoreTransformConfig | None = None,
) -> pd.DataFrame:
    working = predictions.copy()
    selected_score_transform = score_transform or AllocationScoreTransformConfig()
    working["allocation_score_policy"] = allocation_score_policy
    working["allocation_score_policy_prob100_threshold"] = float(prob100_threshold)
    working["score_policy_repair_authorized"] = bool(score_policy_repair_authorized)
    working["score_policy_repair_denied_reason"] = score_policy_repair_denied_reason
    for column, value in _score_transform_metadata(selected_score_transform).items():
        working[column] = value
    probability_columns = {"prob_tier_0", "prob_tier_100"}
    if probability_columns.issubset(working.columns):
        raw_score = pd.Series(
            pd.to_numeric(
                working.get("raw_expected_allocation_score", working["score"]),
                errors="coerce",
            ).fillna(0.0),
            index=working.index,
            dtype=float,
        )
        score_series, triggered_100 = _apply_allocation_score_policy(
            rows=working,
            score_series=raw_score,
            probability_frame=working,
            allocation_score_policy=allocation_score_policy,
            prob100_threshold=prob100_threshold,
            score_policy_repair_authorized=score_policy_repair_authorized,
        )
        score_series, transform_applied = _apply_allocation_score_transform(
            rows=working,
            score_series=score_series,
            score_transform=selected_score_transform,
        )
        working["score"] = score_series.to_numpy()
        working["final_allocation_score"] = score_series.to_numpy()
        working["score_policy_triggered_100"] = triggered_100.to_numpy(dtype=bool)
        working["score_transform_applied"] = transform_applied.to_numpy(dtype=bool)
    else:
        working["score_policy_triggered_100"] = False
        working["score_transform_applied"] = False
        if "final_allocation_score" not in working.columns:
            working["final_allocation_score"] = working["score"]
    if "raw_expected_allocation_score" not in working.columns:
        working["raw_expected_allocation_score"] = working["score"]
    if "score" in working.columns:
        working["predicted_tier_weight"] = working["score"].map(
            lambda value: nearest_tier(float(value))
        )
    working["guarded_gate_bull_risk_off_override_authorized"] = bool(
        guarded_gate_bull_risk_off_override_authorized
    )
    working["guarded_gate_bull_risk_off_override_denied_reason"] = (
        guarded_gate_bull_risk_off_override_denied_reason
    )
    guarded_override_triggered = pd.Series(False, index=working.index, dtype=bool)
    if (
        guarded_gate_bull_risk_off_override_enabled
        and guarded_gate_bull_risk_off_override_authorized
    ):
        guarded_override_triggered = (
            working.get(
                "gate_bull",
                pd.Series(False, index=working.index, dtype=bool),
            ).map(_truthy)
            & working.apply(_allocation_regime_label, axis=1).eq("risk_off")
        )
    working["guarded_gate_bull_risk_off_override_triggered"] = (
        guarded_override_triggered.to_numpy(dtype=bool)
    )
    return working


def _regime_policy_weight_for_row(
    row: pd.Series,
    policy: RegimeParticipationPolicyConfig,
) -> float:
    regime = _allocation_regime_label(row)
    if regime == "risk_off":
        if policy.risk_off_cap is None:
            raise ValueError(
                "Deterministic regime fallback requires the referenced policy to define risk_off_cap."
            )
        weight = float(policy.risk_off_cap)
    elif regime == "bull":
        weight = float(policy.bull_floor)
    elif regime == "bear":
        weight = float(policy.bear_floor)
    else:
        weight = float(policy.sideways_floor)
    if policy.gate_bull_floor is not None and _truthy(row.get("gate_bull", False)):
        weight = max(weight, float(policy.gate_bull_floor))
    return weight


def _deterministic_regime_fallback_weights_for_rows(
    *,
    panel: pd.DataFrame,
    rows: pd.DataFrame,
    frequency: str,
    strategy_name: str,
    policy: RegimeParticipationPolicyConfig,
) -> pd.DataFrame:
    required_columns = {
        "signal_date",
        "effective_date",
        "crypto_regime_risk_off",
        "crypto_regime_trend_state",
    }
    missing_columns = sorted(required_columns - set(rows.columns))
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(
            "evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy "
            f"requires test rows with regime columns: {joined}."
        )

    symbols = sorted(panel["symbol"].drop_duplicates().tolist())
    if len(symbols) != 1:
        raise ValueError(
            "evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy "
            "is only supported for single-symbol allocation runs."
        )
    if rows.empty:
        return pd.DataFrame(columns=["strategy", "effective_date", "symbol", "weight"])

    working = rows.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce")
    working["effective_date"] = pd.to_datetime(working["effective_date"], errors="coerce")
    if working["signal_date"].isna().any() or working["effective_date"].isna().any():
        raise ValueError(
            "evaluation.ml_strategy_tuning.no_candidate_fallback_regime_policy "
            "requires valid signal_date and effective_date values."
        )

    weight_rows: list[dict[str, object]] = []
    for effective_date, date_rows in working.sort_values(
        ["effective_date", "signal_date"]
    ).groupby("effective_date", sort=True):
        selected_row = date_rows.iloc[-1]
        weight_rows.append(
            {
                "strategy": strategy_name,
                "effective_date": pd.Timestamp(effective_date),
                "symbol": symbols[0],
                "weight": _regime_policy_weight_for_row(selected_row, policy),
            }
        )

    boundary_date = next_rebalance_effective_date(
        panel,
        signal_date=pd.Timestamp(working["signal_date"].max()),
        frequency=frequency,
    )
    if boundary_date is not None:
        boundary_date = pd.Timestamp(boundary_date)
        effective_dates = {pd.Timestamp(row["effective_date"]) for row in weight_rows}
        if boundary_date not in effective_dates:
            weight_rows.append(
                {
                    "strategy": strategy_name,
                    "effective_date": boundary_date,
                    "symbol": symbols[0],
                    "weight": 0.0,
                }
            )

    return pd.DataFrame(
        weight_rows,
        columns=["strategy", "effective_date", "symbol", "weight"],
    ).sort_values(["effective_date", "symbol"]).reset_index(drop=True)


def _candidate_failure_reasons(
    *,
    active_candidate: bool,
    turnover_budget_ok: bool,
    predicted_support_ok: bool,
    benchmark_gate_ok: bool,
    benchmark_relative_selection: bool,
    missing_selection_benchmarks: list[str],
    benchmark_excess_values: list[float],
    validation_cost_benchmark_gate_ok: bool,
    risk_gate_ok: bool,
    score_validity_ok: bool = True,
) -> str:
    reasons: list[str] = []
    if not active_candidate:
        reasons.append("inactive_candidate")
    if not turnover_budget_ok:
        reasons.append("turnover_budget_exceeded")
    if not predicted_support_ok:
        reasons.append("insufficient_predicted_tier_support")
    if not benchmark_gate_ok:
        if benchmark_relative_selection:
            if missing_selection_benchmarks or not benchmark_excess_values:
                reasons.append("missing_selection_benchmark")
            else:
                reasons.append("non_positive_required_benchmark_excess")
        else:
            reasons.append("non_positive_buy_hold_excess")
    if not validation_cost_benchmark_gate_ok:
        reasons.append("non_positive_validation_cost_benchmark_excess")
    if not risk_gate_ok:
        reasons.append("risk_not_improved")
    if not score_validity_ok:
        reasons.append("negative_score_forward_return_correlation")
    return ";".join(reasons)


def _candidate_failure_reason_set(row: dict[str, object]) -> set[str]:
    value = row.get("failure_reasons", "")
    if pd.isna(value):
        return set()
    return {reason.strip() for reason in str(value).split(";") if reason.strip()}


def _benchmark_only_fallback_candidates(
    candidates: list[dict[str, object]],
    *,
    allow_score_validity_fallback: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    allowed_reasons = set(BENCHMARK_SELECTION_FAILURE_REASONS)
    if allow_score_validity_fallback:
        allowed_reasons.update(SCORE_VALIDITY_FAILURE_REASONS)
    for row in candidates:
        reasons = _candidate_failure_reason_set(row)
        if (
            not bool(row.get("passed_gate"))
            and reasons
            and reasons.issubset(allowed_reasons)
        ):
            rows.append(row)
    return rows


def _latest_training_rows(rows: pd.DataFrame, rolling_train_bars: int | None) -> pd.DataFrame:
    if rolling_train_bars is None or rows.empty:
        return rows.copy()

    working = rows.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"])
    dates = pd.Index(sorted(working["signal_date"].drop_duplicates()))
    selected_dates = dates[-int(rolling_train_bars):]
    return working.loc[working["signal_date"].isin(selected_dates)].copy()


def _annualized_turnover(
    *,
    performance: pd.DataFrame,
    total_turnover: float,
    periods_per_year: float,
) -> float:
    periods = max(1, int(pd.to_datetime(performance["date"]).nunique()))
    return float(total_turnover) / periods * periods_per_year


def _safe_correlation(left: pd.Series, right: pd.Series) -> object:
    frame = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 2:
        return pd.NA
    if frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return pd.NA
    value = float(frame["left"].corr(frame["right"]))
    return value if math.isfinite(value) else pd.NA


def _score_validity_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    if predictions.empty or "score" not in predictions.columns:
        return {
            "validation_score_forward_return_correlation": pd.NA,
            "validation_raw_score_forward_return_correlation": pd.NA,
            "validation_score_target_correlation": pd.NA,
        }
    target_column = "target_weight" if "target_weight" in predictions.columns else "target"
    raw_score_column = (
        "raw_expected_allocation_score"
        if "raw_expected_allocation_score" in predictions.columns
        else "score"
    )
    return {
        "validation_score_forward_return_correlation": (
            _safe_correlation(predictions["score"], predictions["forward_return"])
            if "forward_return" in predictions.columns
            else pd.NA
        ),
        "validation_raw_score_forward_return_correlation": (
            _safe_correlation(predictions[raw_score_column], predictions["forward_return"])
            if "forward_return" in predictions.columns
            else pd.NA
        ),
        "validation_score_target_correlation": (
            _safe_correlation(predictions["score"], predictions[target_column])
            if target_column in predictions.columns
            else pd.NA
        ),
    }


def _gate_bull_underexposure_metrics(
    *,
    predictions: pd.DataFrame,
    weights: pd.DataFrame,
) -> dict[str, object]:
    empty_metrics = {
        "validation_gate_bull_average_exposure": pd.NA,
        "validation_gate_bull_underexposed_positive_benchmark_fraction": pd.NA,
        "validation_gate_bull_underexposed_positive_benchmark_return_sum": pd.NA,
    }
    if (
        predictions.empty
        or weights.empty
        or "gate_bull" not in predictions.columns
        or "forward_return" not in predictions.columns
    ):
        return empty_metrics

    exposure = (
        weights.copy()
        .assign(effective_date=lambda frame: pd.to_datetime(frame["effective_date"]))
        .groupby("effective_date", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "candidate_exposure"})
    )
    working = predictions.copy()
    working["effective_date"] = pd.to_datetime(working["effective_date"], errors="coerce")
    working = working.merge(exposure, on="effective_date", how="left")
    gate_bull = working.loc[working["gate_bull"].map(_truthy)].copy()
    if gate_bull.empty:
        return empty_metrics

    gate_bull["candidate_exposure"] = (
        pd.to_numeric(gate_bull["candidate_exposure"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    forward_return = pd.to_numeric(gate_bull["forward_return"], errors="coerce")
    positive_benchmark = forward_return.gt(0.0)
    underexposed_positive = positive_benchmark & gate_bull["candidate_exposure"].lt(
        1.0 - 1e-9
    )
    positive_count = int(positive_benchmark.sum())
    return {
        "validation_gate_bull_average_exposure": float(
            gate_bull["candidate_exposure"].mean()
        ),
        "validation_gate_bull_underexposed_positive_benchmark_fraction": (
            float(underexposed_positive.sum() / positive_count)
            if positive_count
            else pd.NA
        ),
        "validation_gate_bull_underexposed_positive_benchmark_return_sum": (
            float(forward_return.loc[underexposed_positive].sum())
            if bool(underexposed_positive.any())
            else 0.0
        ),
    }


def _select_ml_strategy_candidate(candidates: list[dict[str, object]]) -> dict[str, object]:
    if not candidates:
        raise ValueError("ML strategy candidate selection requires at least one candidate.")

    def _selection_score(
        row: dict[str, object],
    ) -> tuple[float, float, float, float, float, float]:
        min_validation_cost_benchmark_excess = row.get(
            "min_selection_validation_cost_benchmark_excess_cumulative_return",
            pd.NA,
        )
        min_benchmark_excess = row.get("min_benchmark_excess_cumulative_return", pd.NA)
        if pd.isna(min_benchmark_excess):
            min_benchmark_excess = row["excess_cumulative_return"]
        if pd.isna(min_validation_cost_benchmark_excess):
            min_validation_cost_benchmark_excess = min_benchmark_excess
        return (
            float(min_validation_cost_benchmark_excess),
            float(min_benchmark_excess),
            float(row["excess_cumulative_return"]),
            float(row["drawdown_delta"]),
            float(row["sharpe_like_delta"]),
            -float(row["annualized_turnover"]),
        )

    return sorted(
        candidates,
        key=_selection_score,
        reverse=True,
    )[0]


def _select_ml_strategy_candidate_for_policy(
    candidates: list[dict[str, object]],
    *,
    selection_policy: str,
    allow_score_validity_fallback: bool = False,
) -> tuple[dict[str, object] | None, str]:
    valid_candidates = [row for row in candidates if row["passed_gate"]]
    if valid_candidates:
        return _select_ml_strategy_candidate(valid_candidates), "strict"
    if selection_policy == "best_active_fallback":
        fallback_candidates = _benchmark_only_fallback_candidates(candidates)
        if not fallback_candidates and allow_score_validity_fallback:
            fallback_candidates = _benchmark_only_fallback_candidates(
                candidates,
                allow_score_validity_fallback=True,
            )
        if fallback_candidates:
            return _select_ml_strategy_candidate(fallback_candidates), "best_active_fallback"
    return None, "none"


def _allocation_probability_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    probability_columns = [
        "prob_tier_0",
        "prob_tier_25",
        "prob_tier_50",
        "prob_tier_100",
    ]
    columns = [
        "model_name",
        "fold_id",
        "signal_date",
        "effective_date",
        "symbol",
        "runtime_regime",
        "gate_bull",
        "crypto_regime_risk_off",
        "crypto_regime_trend_state",
        "target",
        "target_weight",
        "predicted_target",
        "predicted_weight",
        "predicted_tier_weight",
        "allocation_score_policy",
        "allocation_score_policy_prob100_threshold",
        "selected_regime_gate_bull_floor",
        "allocation_score_transform",
        "score_transform_bull_multiplier",
        "score_transform_bull_addend",
        "score_transform_risk_off_score_cap",
        "score_transform_non_bull_score_cap",
        "raw_expected_allocation_score",
        "final_allocation_score",
        "score_policy_repair_authorized",
        "score_policy_repair_denied_reason",
        "score_policy_triggered_100",
        "guarded_gate_bull_risk_off_override_authorized",
        "guarded_gate_bull_risk_off_override_denied_reason",
        "guarded_gate_bull_risk_off_override_triggered",
        "score_transform_applied",
        "score",
        *probability_columns,
        "forward_return",
        "forward_drawdown",
        "forward_realized_volatility",
        "realized_utility",
        "utility_profile",
        "calibration_status",
        "fold_predicted_25_fraction",
        "fold_predicted_50_fraction",
        "fold_predicted_100_fraction",
        "fold_score_policy_repair_authorized_fraction",
        "fold_score_policy_triggered_100_fraction",
        "fold_guarded_gate_bull_risk_off_override_authorized_fraction",
        "fold_guarded_gate_bull_risk_off_override_triggered_fraction",
        "fold_score_transform_applied_fraction",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    working = predictions.copy()
    if "predicted_weight" not in working.columns:
        working["predicted_weight"] = working["predicted_target"].map(
            {0: 0.0, 1: 0.25, 2: 0.50, 3: 1.0}
        )
    if "predicted_tier_weight" not in working.columns:
        working["predicted_tier_weight"] = working["score"].map(
            lambda value: nearest_tier(float(value))
        )
    if "final_allocation_score" not in working.columns:
        working["final_allocation_score"] = working["score"]
    if "raw_expected_allocation_score" not in working.columns:
        working["raw_expected_allocation_score"] = working["score"]
    if "score_policy_triggered_100" not in working.columns:
        working["score_policy_triggered_100"] = False
    if "score_policy_repair_authorized" not in working.columns:
        working["score_policy_repair_authorized"] = False
    if "score_policy_repair_denied_reason" not in working.columns:
        working["score_policy_repair_denied_reason"] = ""
    if "guarded_gate_bull_risk_off_override_authorized" not in working.columns:
        working["guarded_gate_bull_risk_off_override_authorized"] = False
    if "guarded_gate_bull_risk_off_override_denied_reason" not in working.columns:
        working["guarded_gate_bull_risk_off_override_denied_reason"] = ""
    if "guarded_gate_bull_risk_off_override_triggered" not in working.columns:
        working["guarded_gate_bull_risk_off_override_triggered"] = False
    if "score_transform_applied" not in working.columns:
        working["score_transform_applied"] = False
    for column, value in _score_transform_metadata(AllocationScoreTransformConfig()).items():
        if column not in working.columns:
            working[column] = value
    if {"crypto_regime_risk_off", "crypto_regime_trend_state"}.issubset(
        working.columns
    ):
        working["runtime_regime"] = working.apply(_allocation_regime_label, axis=1)
    if "gate_bull" not in working.columns:
        working["gate_bull"] = False
    if "selected_regime_gate_bull_floor" not in working.columns:
        working["selected_regime_gate_bull_floor"] = pd.NA
    for column in probability_columns:
        if column not in working.columns:
            working[column] = 0.0

    fold_support = (
        working.groupby("fold_id")["predicted_tier_weight"]
        .agg(
            fold_predicted_25_fraction=lambda values: float(
                values.sub(0.25).abs().le(1e-9).mean()
            ),
            fold_predicted_50_fraction=lambda values: float(
                values.sub(0.50).abs().le(1e-9).mean()
            ),
            fold_predicted_100_fraction=lambda values: float(
                values.sub(1.0).abs().le(1e-9).mean()
            ),
        )
        .reset_index()
    )
    working = working.merge(fold_support, on="fold_id", how="left")
    repair_authorization = (
        working.groupby("fold_id")["score_policy_repair_authorized"]
        .agg(
            fold_score_policy_repair_authorized_fraction=lambda values: float(
                values.astype(bool).mean()
            )
        )
        .reset_index()
    )
    working = working.merge(repair_authorization, on="fold_id", how="left")
    trigger_support = (
        working.groupby("fold_id")["score_policy_triggered_100"]
        .agg(
            fold_score_policy_triggered_100_fraction=lambda values: float(
                values.astype(bool).mean()
            )
        )
        .reset_index()
    )
    working = working.merge(trigger_support, on="fold_id", how="left")
    guarded_override_authorization = (
        working.groupby("fold_id")["guarded_gate_bull_risk_off_override_authorized"]
        .agg(
            fold_guarded_gate_bull_risk_off_override_authorized_fraction=lambda values: float(
                values.astype(bool).mean()
            )
        )
        .reset_index()
    )
    working = working.merge(guarded_override_authorization, on="fold_id", how="left")
    guarded_override_trigger_support = (
        working.groupby("fold_id")["guarded_gate_bull_risk_off_override_triggered"]
        .agg(
            fold_guarded_gate_bull_risk_off_override_triggered_fraction=lambda values: float(
                values.astype(bool).mean()
            )
        )
        .reset_index()
    )
    working = working.merge(guarded_override_trigger_support, on="fold_id", how="left")
    transform_support = (
        working.groupby("fold_id")["score_transform_applied"]
        .agg(
            fold_score_transform_applied_fraction=lambda values: float(
                values.astype(bool).mean()
            )
        )
        .reset_index()
    )
    working = working.merge(transform_support, on="fold_id", how="left")

    utility_columns = {
        0.0: "allocation_utility_0",
        0.25: "allocation_utility_25",
        0.50: "allocation_utility_50",
        1.0: "allocation_utility_100",
    }
    realized_utilities: list[float] = []
    for _, row in working.iterrows():
        utility_column = utility_columns.get(float(row["predicted_tier_weight"]))
        realized_utilities.append(
            float(row[utility_column])
            if utility_column is not None and utility_column in working.columns
            else float("nan")
        )
    working["realized_utility"] = realized_utilities
    for column in columns:
        if column not in working.columns:
            working[column] = pd.NA
    return working.loc[:, columns].sort_values(
        ["fold_id", "signal_date", "symbol"]
    ).reset_index(drop=True)


def _allocation_regime_label(row: pd.Series) -> str:
    risk_off_value = row.get("crypto_regime_risk_off", 0)
    trend_state_value = row.get("crypto_regime_trend_state", 0)
    risk_off = 0 if pd.isna(risk_off_value) else int(risk_off_value)
    trend_state = 0 if pd.isna(trend_state_value) else int(trend_state_value)
    if risk_off == 1:
        return "risk_off"
    if trend_state > 0:
        return "bull"
    if trend_state < 0:
        return "bear"
    return "sideways"


def _allocation_target_diagnostics(
    *,
    modeling_dataset: pd.DataFrame,
    folds: list,
) -> pd.DataFrame:
    columns = [
        "fold_id",
        "scope",
        "regime",
        "target",
        "target_weight",
        "row_count",
        "row_fraction",
        "avg_forward_return",
        "avg_forward_drawdown",
        "avg_forward_realized_volatility",
    ]
    if modeling_dataset.empty or "target_weight" not in modeling_dataset.columns:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []

    def _append_distribution_rows(
        *,
        frame: pd.DataFrame,
        fold_id: object,
        scope: str,
        group_by_regime: bool,
    ) -> None:
        if frame.empty:
            return
        working = frame.copy()
        working["regime"] = (
            working.apply(_allocation_regime_label, axis=1)
            if group_by_regime
            else "all"
        )
        total_rows = len(working)
        for (regime, target, target_weight), group in working.groupby(
            ["regime", "target", "target_weight"],
            sort=True,
        ):
            rows.append(
                {
                    "fold_id": fold_id,
                    "scope": scope,
                    "regime": regime,
                    "target": int(target),
                    "target_weight": float(target_weight),
                    "row_count": int(len(group)),
                    "row_fraction": float(len(group) / total_rows),
                    "avg_forward_return": float(group["forward_return"].mean()),
                    "avg_forward_drawdown": float(group["forward_drawdown"].mean())
                    if "forward_drawdown" in group.columns
                    else pd.NA,
                    "avg_forward_realized_volatility": float(
                        group["forward_realized_volatility"].mean()
                    )
                    if "forward_realized_volatility" in group.columns
                    else pd.NA,
                }
            )

    _append_distribution_rows(
        frame=modeling_dataset,
        fold_id="all",
        scope="global",
        group_by_regime=False,
    )
    for fold in folds:
        train_rows, test_rows = slice_fold_rows(modeling_dataset, fold)
        _append_distribution_rows(
            frame=train_rows,
            fold_id=fold.fold_id,
            scope="train_validation",
            group_by_regime=False,
        )
        _append_distribution_rows(
            frame=test_rows,
            fold_id=fold.fold_id,
            scope="oos_test",
            group_by_regime=True,
        )
    return pd.DataFrame(rows, columns=columns)


def _build_ml_strategy_tuning_outputs(
    *,
    config: ExperimentConfig,
    panel: pd.DataFrame,
    modeling_dataset: pd.DataFrame,
    folds: list,
    buy_hold_performance: pd.DataFrame,
    baseline_performance: pd.DataFrame,
) -> tuple[
    BacktestResult | None,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    candidate_columns = [
        "fold_id",
        "model_name",
        "allocation_mode",
        "utility_profile",
        "utility_drawdown_penalty",
        "utility_volatility_penalty",
        "utility_risk_penalty_power",
        "allocation_class_weighting",
        "allocation_score_policy",
        "allocation_score_policy_prob100_threshold",
        "allocation_score_transform",
        "score_transform_bull_multiplier",
        "score_transform_bull_addend",
        "score_transform_risk_off_score_cap",
        "score_transform_non_bull_score_cap",
        "calibration_status",
        "rolling_train_bars",
        "min_holding_period_bars",
        "hysteresis_margin",
        "regime_policy",
        "regime_bull_floor",
        "regime_sideways_floor",
        "regime_bear_floor",
        "regime_risk_off_cap",
        "regime_gate_bull_floor",
        "threshold",
        "tier_min_threshold",
        "tier_half_threshold",
        "tier_full_threshold",
        "strategy",
        "validation_start",
        "validation_end",
        "inner_train_rows",
        "validation_rows",
        "cumulative_return",
        "max_drawdown",
        "sharpe_like",
        "total_turnover",
        "annualized_turnover",
        "exposure_changes",
        "average_exposure",
        "buy_hold_cumulative_return",
        "excess_cumulative_return",
        "selection_benchmark_strategies",
        "selection_benchmark_excess_cumulative_returns",
        "min_benchmark_excess_cumulative_return",
        "selection_validation_cost_bps",
        "selection_validation_cost_benchmark_excess_cumulative_returns",
        "min_selection_validation_cost_benchmark_excess_cumulative_return",
        "validation_predicted_25_fraction",
        "validation_predicted_50_fraction",
        "validation_predicted_100_fraction",
        "validation_score_forward_return_correlation",
        "validation_raw_score_forward_return_correlation",
        "validation_score_target_correlation",
        "validation_score_policy_repair_authorized",
        "score_policy_repair_authorized",
        "score_policy_repair_denied_reason",
        "validation_guarded_gate_bull_risk_off_override_authorized",
        "guarded_gate_bull_risk_off_override_authorized",
        "guarded_gate_bull_risk_off_override_denied_reason",
        "validation_gate_bull_average_exposure",
        "validation_gate_bull_underexposed_positive_benchmark_fraction",
        "validation_gate_bull_underexposed_positive_benchmark_return_sum",
        "validation_score_policy_triggered_100_fraction",
        "validation_guarded_gate_bull_risk_off_override_triggered_fraction",
        "validation_score_transform_applied_fraction",
        "min_validation_predicted_target_fraction",
        "sharpe_like_delta",
        "drawdown_delta",
        "active_candidate",
        "failure_reasons",
        "passed_gate",
    ]
    selection_columns = [
        "fold_id",
        "selection_status",
        "selection_policy",
        "selection_source",
        "allocation_mode",
        "selected_model_name",
        "selected_utility_profile",
        "selected_utility_drawdown_penalty",
        "selected_utility_volatility_penalty",
        "selected_utility_risk_penalty_power",
        "allocation_class_weighting",
        "allocation_score_policy",
        "allocation_score_policy_prob100_threshold",
        "selected_allocation_score_transform",
        "selected_score_transform_bull_multiplier",
        "selected_score_transform_bull_addend",
        "selected_score_transform_risk_off_score_cap",
        "selected_score_transform_non_bull_score_cap",
        "calibration_status",
        "selected_rolling_train_bars",
        "selected_min_holding_period_bars",
        "selected_hysteresis_margin",
        "selected_regime_policy",
        "selected_regime_bull_floor",
        "selected_regime_sideways_floor",
        "selected_regime_bear_floor",
        "selected_regime_risk_off_cap",
        "selected_regime_gate_bull_floor",
        "selected_threshold",
        "selected_tier_min_threshold",
        "selected_tier_half_threshold",
        "selected_tier_full_threshold",
        "selected_strategy",
        "validation_start",
        "validation_end",
        "inner_train_rows",
        "validation_rows",
        "passed_gate",
        "excess_cumulative_return",
        "selection_benchmark_strategies",
        "selection_benchmark_excess_cumulative_returns",
        "min_benchmark_excess_cumulative_return",
        "selection_validation_cost_bps",
        "selection_validation_cost_benchmark_excess_cumulative_returns",
        "min_selection_validation_cost_benchmark_excess_cumulative_return",
        "validation_predicted_25_fraction",
        "validation_predicted_50_fraction",
        "validation_predicted_100_fraction",
        "validation_score_forward_return_correlation",
        "validation_raw_score_forward_return_correlation",
        "validation_score_target_correlation",
        "validation_score_policy_repair_authorized",
        "score_policy_repair_authorized",
        "score_policy_repair_denied_reason",
        "validation_guarded_gate_bull_risk_off_override_authorized",
        "guarded_gate_bull_risk_off_override_authorized",
        "guarded_gate_bull_risk_off_override_denied_reason",
        "validation_gate_bull_average_exposure",
        "validation_gate_bull_underexposed_positive_benchmark_fraction",
        "validation_gate_bull_underexposed_positive_benchmark_return_sum",
        "validation_score_policy_triggered_100_fraction",
        "validation_guarded_gate_bull_risk_off_override_triggered_fraction",
        "validation_score_transform_applied_fraction",
        "min_validation_predicted_target_fraction",
        "sharpe_like_delta",
        "drawdown_delta",
        "annualized_turnover",
        "exposure_changes",
        "average_exposure",
        "selected_candidate_failure_reasons",
    ]
    allocation_probability_columns = [
        "model_name",
        "fold_id",
        "signal_date",
        "effective_date",
        "symbol",
        "runtime_regime",
        "gate_bull",
        "crypto_regime_risk_off",
        "crypto_regime_trend_state",
        "target",
        "target_weight",
        "predicted_target",
        "predicted_weight",
        "predicted_tier_weight",
        "allocation_score_policy",
        "allocation_score_policy_prob100_threshold",
        "selected_regime_gate_bull_floor",
        "allocation_score_transform",
        "score_transform_bull_multiplier",
        "score_transform_bull_addend",
        "score_transform_risk_off_score_cap",
        "score_transform_non_bull_score_cap",
        "raw_expected_allocation_score",
        "final_allocation_score",
        "score_policy_repair_authorized",
        "score_policy_repair_denied_reason",
        "score_policy_triggered_100",
        "guarded_gate_bull_risk_off_override_authorized",
        "guarded_gate_bull_risk_off_override_denied_reason",
        "guarded_gate_bull_risk_off_override_triggered",
        "score_transform_applied",
        "score",
        "prob_tier_0",
        "prob_tier_25",
        "prob_tier_50",
        "prob_tier_100",
        "forward_return",
        "forward_drawdown",
        "forward_realized_volatility",
        "realized_utility",
        "utility_profile",
        "calibration_status",
        "fold_predicted_25_fraction",
        "fold_predicted_50_fraction",
        "fold_predicted_100_fraction",
        "fold_score_policy_repair_authorized_fraction",
        "fold_score_policy_triggered_100_fraction",
        "fold_guarded_gate_bull_risk_off_override_authorized_fraction",
        "fold_guarded_gate_bull_risk_off_override_triggered_fraction",
        "fold_score_transform_applied_fraction",
    ]
    feature_importance_columns = [
        "model_name",
        "fold_id",
        "feature",
        "importance_type",
        "importance",
        "signed_coefficient",
    ]
    tuning = config.evaluation.ml_strategy_tuning
    allocation_mode = tuning.allocation_mode
    selection_benchmark_names = _selection_benchmark_strategies(config)
    benchmark_relative_selection = tuning.objective in {
        "net_return_and_risk_vs_required_benchmarks",
        "net_return_risk_score_validity_vs_required_benchmarks",
    }
    score_validity_selection = (
        tuning.objective == "net_return_risk_score_validity_vs_required_benchmarks"
    )
    threshold_candidates = sorted(set(float(value) for value in tuning.thresholds))
    tier_threshold_candidates = _tier_threshold_sets(config)
    utility_profiles = _allocation_profile_candidates(config)
    required_predicted_weights = [
        float(value)
        for value in config.evaluation.strict_research_gate.required_predicted_target_weights
    ]
    min_predicted_fraction = (
        config.evaluation.strict_research_gate.min_predicted_target_fraction
    )
    rolling_train_candidates = _rolling_train_bars_candidates(config)
    min_holding_candidates = (
        _min_holding_period_candidates(config)
        if allocation_mode in {"direct_tiered", "tiered"}
        else [0]
    )
    hysteresis_margin_candidates = (
        _hysteresis_margin_candidates(config)
        if allocation_mode in {"direct_tiered", "tiered"}
        else [0.0]
    )
    prob100_threshold_candidates = _allocation_score_policy_prob100_threshold_candidates(
        config
    )
    score_transform_candidates = _allocation_score_transform_candidates(config)
    regime_policy_candidates = (
        _regime_participation_policy_candidates(config)
        if allocation_mode in {"direct_tiered", "tiered"}
        else [RegimeParticipationPolicyConfig()]
    )
    if (
        not tuning.enabled
        or not config.models
        or (allocation_mode == "binary" and not threshold_candidates)
        or (allocation_mode == "tiered" and not tier_threshold_candidates)
    ):
        return (
            None,
            pd.DataFrame(columns=candidate_columns),
            pd.DataFrame(columns=selection_columns),
            pd.DataFrame(columns=allocation_probability_columns),
            pd.DataFrame(columns=feature_importance_columns),
            pd.DataFrame(columns=["strategy", "effective_date", "symbol", "weight"]),
        )

    feature_columns = modeling_feature_columns(modeling_dataset)
    modeling_dataset = _with_completed_bar_gate_labels(
        modeling_dataset,
        panel=panel,
        config=config,
    )
    frequency = config.portfolio.ranking.rebalance_frequency
    candidate_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    selected_weight_frames: list[pd.DataFrame] = []
    allocation_probability_frames: list[pd.DataFrame] = []
    feature_importance_frames: list[pd.DataFrame] = []
    panel_timestamps = pd.to_datetime(panel["timestamp"])
    fallback_regime_policy = _no_candidate_fallback_regime_policy(config)
    selection_validation_cost_bps = _selection_validation_cost_bps(config)
    baseline_performance_by_cost = {
        float(cost_bps): reprice_performance(baseline_performance, float(cost_bps))
        for cost_bps in {
            float(config.portfolio.costs.bps_per_trade),
            *selection_validation_cost_bps,
        }
    }

    def _deterministic_fallback_selection(
        *,
        fold_id: int,
        test_rows: pd.DataFrame,
        validation_rows: pd.DataFrame,
        inner_train_rows: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        if fallback_regime_policy is None:
            raise ValueError("Deterministic regime fallback requires a configured policy.")
        weights = _deterministic_regime_fallback_weights_for_rows(
            panel=panel,
            rows=test_rows,
            frequency=frequency,
            strategy_name=ML_TUNED_STRATEGY_NAME,
            policy=fallback_regime_policy,
        )
        exposure_changes, average_exposure = _weight_activity(weights)
        row = {
            "fold_id": fold_id,
            "selection_status": "selected",
            "selection_policy": tuning.selection_policy,
            "selection_source": "regime_policy_fallback",
            "allocation_mode": allocation_mode,
            "selected_model_name": pd.NA,
            "selected_utility_profile": pd.NA,
            "selected_utility_drawdown_penalty": pd.NA,
            "selected_utility_volatility_penalty": pd.NA,
            "selected_utility_risk_penalty_power": pd.NA,
            "allocation_class_weighting": tuning.allocation_class_weighting,
            "allocation_score_policy": tuning.allocation_score_policy,
            "allocation_score_policy_prob100_threshold": (
                tuning.allocation_score_policy_prob100_threshold
            ),
            **_selected_score_transform_metadata(None),
            "calibration_status": pd.NA,
            "selected_rolling_train_bars": pd.NA,
            "selected_min_holding_period_bars": pd.NA,
            "selected_hysteresis_margin": pd.NA,
            "selected_regime_policy": fallback_regime_policy.name,
            "selected_regime_bull_floor": float(fallback_regime_policy.bull_floor),
            "selected_regime_sideways_floor": float(
                fallback_regime_policy.sideways_floor
            ),
            "selected_regime_bear_floor": float(fallback_regime_policy.bear_floor),
            "selected_regime_risk_off_cap": float(fallback_regime_policy.risk_off_cap)
            if fallback_regime_policy.risk_off_cap is not None
            else pd.NA,
            "selected_regime_gate_bull_floor": (
                float(fallback_regime_policy.gate_bull_floor)
                if fallback_regime_policy.gate_bull_floor is not None
                else pd.NA
            ),
            "selected_threshold": pd.NA,
            "selected_tier_min_threshold": pd.NA,
            "selected_tier_half_threshold": pd.NA,
            "selected_tier_full_threshold": pd.NA,
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
            "selection_benchmark_strategies": ",".join(selection_benchmark_names),
            "selection_benchmark_excess_cumulative_returns": pd.NA,
            "min_benchmark_excess_cumulative_return": pd.NA,
            "selection_validation_cost_bps": ",".join(
                f"{cost_bps:g}" for cost_bps in selection_validation_cost_bps
            ),
            "selection_validation_cost_benchmark_excess_cumulative_returns": pd.NA,
            "min_selection_validation_cost_benchmark_excess_cumulative_return": pd.NA,
            "validation_predicted_25_fraction": pd.NA,
            "validation_predicted_50_fraction": pd.NA,
            "validation_predicted_100_fraction": pd.NA,
            "validation_score_forward_return_correlation": pd.NA,
            "validation_raw_score_forward_return_correlation": pd.NA,
            "validation_score_target_correlation": pd.NA,
            "validation_score_policy_repair_authorized": False,
            "score_policy_repair_authorized": False,
            "score_policy_repair_denied_reason": (
                "regime_policy_fallback_no_valid_candidate"
                if tuning.allocation_score_policy == "gate_bull_prob100_threshold"
                else ""
            ),
            "validation_guarded_gate_bull_risk_off_override_authorized": False,
            "guarded_gate_bull_risk_off_override_authorized": False,
            "guarded_gate_bull_risk_off_override_denied_reason": (
                "regime_policy_fallback_no_valid_candidate"
                if tuning.guarded_gate_bull_risk_off_override
                else ""
            ),
            "validation_gate_bull_average_exposure": pd.NA,
            "validation_gate_bull_underexposed_positive_benchmark_fraction": pd.NA,
            "validation_gate_bull_underexposed_positive_benchmark_return_sum": pd.NA,
            "validation_score_policy_triggered_100_fraction": pd.NA,
            "validation_guarded_gate_bull_risk_off_override_triggered_fraction": pd.NA,
            "validation_score_transform_applied_fraction": pd.NA,
            "min_validation_predicted_target_fraction": pd.NA,
            "sharpe_like_delta": pd.NA,
            "drawdown_delta": pd.NA,
            "annualized_turnover": pd.NA,
            "exposure_changes": exposure_changes,
            "average_exposure": average_exposure,
            "selected_candidate_failure_reasons": "regime_policy_fallback_no_valid_candidate",
        }
        return weights, row

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
            selection_benchmark_returns: dict[str, float] = {}
            missing_selection_benchmarks: list[str] = []
            for benchmark_name in selection_benchmark_names:
                try:
                    benchmark_window_metrics = _strategy_metrics_for_window(
                        performance=baseline_performance,
                        strategy_name=benchmark_name,
                        window_dates=validation_dates,
                        periods_per_year=config.evaluation.periods_per_year,
                    )
                except RuntimeError:
                    missing_selection_benchmarks.append(benchmark_name)
                    continue
                selection_benchmark_returns[benchmark_name] = float(
                    benchmark_window_metrics["cumulative_return"]
                )
            selection_benchmark_returns_by_cost: dict[float, dict[str, float]] = {}
            missing_selection_benchmarks_by_cost: dict[float, list[str]] = {}
            for cost_bps in selection_validation_cost_bps:
                cost_benchmark_returns: dict[str, float] = {}
                cost_missing_benchmarks: list[str] = []
                cost_performance = baseline_performance_by_cost[float(cost_bps)]
                for benchmark_name in selection_benchmark_names:
                    try:
                        benchmark_window_metrics = _strategy_metrics_for_window(
                            performance=cost_performance,
                            strategy_name=benchmark_name,
                            window_dates=validation_dates,
                            periods_per_year=config.evaluation.periods_per_year,
                        )
                    except RuntimeError:
                        cost_missing_benchmarks.append(benchmark_name)
                        continue
                    cost_benchmark_returns[benchmark_name] = float(
                        benchmark_window_metrics["cumulative_return"]
                    )
                selection_benchmark_returns_by_cost[float(cost_bps)] = (
                    cost_benchmark_returns
                )
                missing_selection_benchmarks_by_cost[float(cost_bps)] = (
                    cost_missing_benchmarks
                )

            for model_spec in config.models:
                for rolling_train_bars in rolling_train_candidates:
                    candidate_train_rows = _latest_training_rows(
                        inner_train_rows,
                        rolling_train_bars,
                    )
                    for utility_profile in utility_profiles:
                        profiled_candidate_train_rows = _apply_allocation_profile(
                            candidate_train_rows,
                            config=config,
                            profile=utility_profile,
                        )
                        profiled_validation_rows = _apply_allocation_profile(
                            validation_rows,
                            config=config,
                            profile=utility_profile,
                        )
                        LOGGER.info(
                            "Fitting ML strategy tuning candidate model=%s fold=%s rolling_train_bars=%s utility_profile=%s",
                            model_spec.name,
                            fold.fold_id,
                            rolling_train_bars or "full",
                            utility_profile.name,
                        )
                        validation_output = _score_model_rows(
                            model_name=model_spec.name,
                            target_type=config.target.type,
                            train_rows=profiled_candidate_train_rows,
                            score_rows=profiled_validation_rows,
                            feature_columns=feature_columns,
                            fold_id=fold.fold_id,
                            allocation_class_weighting=tuning.allocation_class_weighting,
                            allocation_partial_class_weight_multiplier=(
                                tuning.allocation_partial_class_weight_multiplier
                            ),
                            allocation_probability_calibration=(
                                tuning.allocation_probability_calibration
                            ),
                            allocation_calibration_cv=tuning.allocation_calibration_cv,
                            allocation_score_policy=tuning.allocation_score_policy,
                            allocation_score_policy_prob100_threshold=(
                                tuning.allocation_score_policy_prob100_threshold
                            ),
                            utility_profile=utility_profile,
                        )
                        if validation_output is None:
                            continue
                        base_validation_predictions = validation_output.predictions
                        base_score_validity_metrics = _score_validity_metrics(
                            base_validation_predictions
                        )
                        (
                            score_policy_repair_authorized,
                            score_policy_repair_denied_reason,
                        ) = _score_policy_repair_authorization(
                            allocation_score_policy=tuning.allocation_score_policy,
                            validation_raw_score_forward_return_correlation=(
                                base_score_validity_metrics[
                                    "validation_raw_score_forward_return_correlation"
                                ]
                            ),
                        )
                        (
                            guarded_gate_bull_risk_off_override_authorized,
                            guarded_gate_bull_risk_off_override_denied_reason,
                        ) = _guarded_gate_bull_risk_off_override_authorization(
                            enabled=tuning.guarded_gate_bull_risk_off_override,
                            validation_raw_score_forward_return_correlation=(
                                base_score_validity_metrics[
                                    "validation_raw_score_forward_return_correlation"
                                ]
                            ),
                        )
                        candidate_thresholds: list[
                            tuple[
                                float,
                                tuple[float, float, float] | None,
                                float,
                                AllocationScoreTransformConfig,
                            ]
                        ]
                        if allocation_mode == "tiered":
                            candidate_thresholds = [
                                (
                                    threshold_set[0],
                                    threshold_set,
                                    prob100_threshold,
                                    score_transform,
                                )
                                for threshold_set in tier_threshold_candidates
                                for prob100_threshold in prob100_threshold_candidates
                                for score_transform in score_transform_candidates
                            ]
                        elif allocation_mode == "direct_tiered":
                            candidate_thresholds = [
                                (0.0, None, prob100_threshold, score_transform)
                                for prob100_threshold in prob100_threshold_candidates
                                for score_transform in score_transform_candidates
                            ]
                        else:
                            candidate_thresholds = [
                                (threshold, None, prob100_threshold, score_transform)
                                for threshold in threshold_candidates
                                for prob100_threshold in prob100_threshold_candidates
                                for score_transform in score_transform_candidates
                            ]
                        for (
                            threshold,
                            tier_thresholds,
                            prob100_threshold,
                            score_transform,
                        ) in candidate_thresholds:
                            validation_predictions = (
                                _prediction_frame_with_allocation_score_policy(
                                    predictions=base_validation_predictions,
                                    allocation_score_policy=tuning.allocation_score_policy,
                                    prob100_threshold=prob100_threshold,
                                    score_policy_repair_authorized=(
                                        score_policy_repair_authorized
                                    ),
                                    score_policy_repair_denied_reason=(
                                        score_policy_repair_denied_reason
                                    ),
                                    guarded_gate_bull_risk_off_override_enabled=(
                                        tuning.guarded_gate_bull_risk_off_override
                                    ),
                                    guarded_gate_bull_risk_off_override_authorized=(
                                        guarded_gate_bull_risk_off_override_authorized
                                    ),
                                    guarded_gate_bull_risk_off_override_denied_reason=(
                                        guarded_gate_bull_risk_off_override_denied_reason
                                    ),
                                    score_transform=score_transform,
                                )
                            )
                            (
                                predicted_support_ok,
                                predicted_support_fractions,
                            ) = _predicted_tier_support(
                                predictions=validation_predictions,
                                required_weights=(
                                    required_predicted_weights
                                    if _is_allocation_target(config.target.type)
                                    else []
                                ),
                                min_fraction=min_predicted_fraction,
                            )
                            validation_predicted_25_fraction = (
                                predicted_support_fractions.get(0.25, pd.NA)
                            )
                            validation_predicted_50_fraction = (
                                predicted_support_fractions.get(0.50, pd.NA)
                            )
                            validation_predicted_100_fraction = float(
                                validation_predictions["score"]
                                .map(lambda value: nearest_tier(float(value)))
                                .sub(1.0)
                                .abs()
                                .le(1e-9)
                                .mean()
                            )
                            if "score_policy_triggered_100" in validation_predictions.columns:
                                validation_score_policy_triggered_100_fraction = float(
                                    validation_predictions[
                                        "score_policy_triggered_100"
                                    ].astype(bool).mean()
                                )
                            else:
                                validation_score_policy_triggered_100_fraction = pd.NA
                            validation_guarded_gate_bull_risk_off_override_triggered_fraction = float(
                                validation_predictions[
                                    "guarded_gate_bull_risk_off_override_triggered"
                                ]
                                .astype(bool)
                                .mean()
                            )
                            if "score_transform_applied" in validation_predictions.columns:
                                validation_score_transform_applied_fraction = float(
                                    validation_predictions[
                                        "score_transform_applied"
                                    ].astype(bool).mean()
                                )
                            else:
                                validation_score_transform_applied_fraction = pd.NA
                            predicted_fraction_values = [
                                value
                                for value in predicted_support_fractions.values()
                                if pd.notna(value)
                            ]
                            min_validation_predicted_target_fraction = (
                                min(predicted_fraction_values)
                                if predicted_fraction_values
                                else pd.NA
                            )
                            score_validity_metrics = _score_validity_metrics(
                                validation_predictions
                            )
                            score_forward_correlation = score_validity_metrics[
                                "validation_score_forward_return_correlation"
                            ]
                            score_validity_ok = not (
                                score_validity_selection
                                and pd.notna(score_forward_correlation)
                                and float(score_forward_correlation) < 0.0
                            )
                            for regime_policy in regime_policy_candidates:
                                strategy_regime_policy = _strategy_regime_policy(
                                    regime_policy
                                )
                                for min_holding_period_bars in min_holding_candidates:
                                    for hysteresis_margin in hysteresis_margin_candidates:
                                        weights = _weights_for_predictions(
                                            config=config,
                                            panel=panel,
                                            predictions=validation_predictions,
                                            threshold=threshold,
                                            tier_thresholds=tier_thresholds,
                                            min_holding_period_bars=min_holding_period_bars,
                                            hysteresis_margin=hysteresis_margin,
                                            direct_tiered=allocation_mode == "direct_tiered",
                                            regime_policy=strategy_regime_policy,
                                        )
                                        if weights.empty:
                                            continue
                                        performance_by_cost = {
                                            float(cost_bps): run_backtest(
                                                panel=validation_panel,
                                                weights=weights,
                                                cost_bps=float(cost_bps),
                                            )
                                            for cost_bps in {
                                                float(config.portfolio.costs.bps_per_trade),
                                                *selection_validation_cost_bps,
                                            }
                                        }
                                        performance = performance_by_cost[
                                            float(config.portfolio.costs.bps_per_trade)
                                        ]
                                        metrics = compute_strategy_metrics(
                                            performance,
                                            periods_per_year=config.evaluation.periods_per_year,
                                        ).iloc[0]
                                        total_turnover = float(metrics["total_turnover"])
                                        annualized_turnover = _annualized_turnover(
                                            performance=performance,
                                            total_turnover=total_turnover,
                                            periods_per_year=config.evaluation.periods_per_year,
                                        )
                                        exposure_changes, average_exposure = _weight_activity(
                                            weights
                                        )
                                        gate_bull_metrics = (
                                            _gate_bull_underexposure_metrics(
                                                predictions=validation_predictions,
                                                weights=weights,
                                            )
                                        )
                                        excess_return = (
                                            float(metrics["cumulative_return"])
                                            - buy_hold_return
                                        )
                                        benchmark_excess_returns = {
                                            name: float(metrics["cumulative_return"])
                                            - benchmark_return
                                            for name, benchmark_return in selection_benchmark_returns.items()
                                        }
                                        benchmark_excess_values = list(
                                            benchmark_excess_returns.values()
                                        )
                                        min_benchmark_excess_return = (
                                            min(benchmark_excess_values)
                                            if benchmark_excess_values
                                            else pd.NA
                                        )
                                        benchmark_excess_summary = ";".join(
                                            f"{name}:{benchmark_excess_returns[name]:.12g}"
                                            for name in selection_benchmark_names
                                            if name in benchmark_excess_returns
                                        )
                                        if missing_selection_benchmarks:
                                            missing_summary = ";".join(
                                                f"{name}:missing"
                                                for name in missing_selection_benchmarks
                                            )
                                            benchmark_excess_summary = ";".join(
                                                value
                                                for value in [
                                                    benchmark_excess_summary,
                                                    missing_summary,
                                                ]
                                                if value
                                            )
                                        validation_cost_benchmark_excess_values: list[float] = []
                                        validation_cost_benchmark_excess_summary_parts: list[
                                            str
                                        ] = []
                                        for cost_bps in selection_validation_cost_bps:
                                            cost_metrics = compute_strategy_metrics(
                                                performance_by_cost[float(cost_bps)],
                                                periods_per_year=(
                                                    config.evaluation.periods_per_year
                                                ),
                                            ).iloc[0]
                                            cost_excess_returns = {
                                                name: float(
                                                    cost_metrics["cumulative_return"]
                                                )
                                                - benchmark_return
                                                for name, benchmark_return in (
                                                    selection_benchmark_returns_by_cost[
                                                        float(cost_bps)
                                                    ].items()
                                                )
                                            }
                                            validation_cost_benchmark_excess_values.extend(
                                                cost_excess_returns.values()
                                            )
                                            validation_cost_benchmark_excess_summary_parts.extend(
                                                f"{float(cost_bps):g}:{name}:{cost_excess_returns[name]:.12g}"
                                                for name in selection_benchmark_names
                                                if name in cost_excess_returns
                                            )
                                            validation_cost_benchmark_excess_summary_parts.extend(
                                                f"{float(cost_bps):g}:{name}:missing"
                                                for name in missing_selection_benchmarks_by_cost[
                                                    float(cost_bps)
                                                ]
                                            )
                                        min_validation_cost_benchmark_excess_return = (
                                            min(
                                                validation_cost_benchmark_excess_values
                                            )
                                            if validation_cost_benchmark_excess_values
                                            else pd.NA
                                        )
                                        validation_cost_benchmark_gate_ok = (
                                            not any(
                                                missing_selection_benchmarks_by_cost.values()
                                            )
                                            and bool(
                                                validation_cost_benchmark_excess_values
                                            )
                                            and all(
                                                excess > 0.0
                                                for excess in (
                                                    validation_cost_benchmark_excess_values
                                                )
                                            )
                                        )
                                        sharpe_delta = (
                                            float(metrics["sharpe_like"]) - buy_hold_sharpe
                                        )
                                        drawdown_delta = (
                                            float(metrics["max_drawdown"])
                                            - buy_hold_drawdown
                                        )
                                        active_candidate = (
                                            exposure_changes >= tuning.min_exposure_changes
                                            and average_exposure
                                            >= tuning.min_average_exposure_for_active
                                            and average_exposure
                                            <= tuning.max_average_exposure_for_active
                                        )
                                        turnover_budget_ok = (
                                            tuning.max_annualized_turnover is None
                                            or annualized_turnover
                                            <= tuning.max_annualized_turnover
                                        )
                                        benchmark_gate_ok = (
                                            (
                                                not benchmark_relative_selection
                                                and excess_return > 0.0
                                            )
                                            or (
                                                benchmark_relative_selection
                                                and not missing_selection_benchmarks
                                                and bool(benchmark_excess_values)
                                                and all(
                                                    excess > 0.0
                                                    for excess in benchmark_excess_values
                                                )
                                            )
                                        )
                                        risk_gate_ok = (
                                            sharpe_delta > 0.0 or drawdown_delta >= 0.0
                                        )
                                        failure_reasons = _candidate_failure_reasons(
                                            active_candidate=active_candidate,
                                            turnover_budget_ok=turnover_budget_ok,
                                            predicted_support_ok=predicted_support_ok,
                                            benchmark_gate_ok=benchmark_gate_ok,
                                            benchmark_relative_selection=(
                                                benchmark_relative_selection
                                            ),
                                            missing_selection_benchmarks=(
                                                missing_selection_benchmarks
                                            ),
                                            benchmark_excess_values=benchmark_excess_values,
                                            validation_cost_benchmark_gate_ok=(
                                                validation_cost_benchmark_gate_ok
                                            ),
                                            risk_gate_ok=risk_gate_ok,
                                            score_validity_ok=score_validity_ok,
                                        )
                                        passed_gate = (
                                            active_candidate
                                            and turnover_budget_ok
                                            and predicted_support_ok
                                            and benchmark_gate_ok
                                            and validation_cost_benchmark_gate_ok
                                            and risk_gate_ok
                                            and score_validity_ok
                                        )
                                        row = {
                                            "fold_id": fold.fold_id,
                                            "model_name": model_spec.name,
                                            "allocation_mode": allocation_mode,
                                            "utility_profile": utility_profile.name,
                                            "utility_drawdown_penalty": float(
                                                utility_profile.drawdown_penalty
                                            ),
                                            "utility_volatility_penalty": float(
                                                utility_profile.volatility_penalty
                                            ),
                                            "utility_risk_penalty_power": float(
                                                utility_profile.risk_penalty_power
                                            ),
                                            "allocation_class_weighting": (
                                                tuning.allocation_class_weighting
                                            ),
                                            "allocation_score_policy": (
                                                tuning.allocation_score_policy
                                            ),
                                            "allocation_score_policy_prob100_threshold": (
                                                prob100_threshold
                                            ),
                                            **_score_transform_metadata(score_transform),
                                            "calibration_status": (
                                                validation_output.calibration_status
                                            ),
                                            "rolling_train_bars": (
                                                rolling_train_bars
                                                if rolling_train_bars is not None
                                                else pd.NA
                                            ),
                                            "min_holding_period_bars": (
                                                min_holding_period_bars
                                            ),
                                            "hysteresis_margin": hysteresis_margin,
                                            "regime_policy": regime_policy.name,
                                            "regime_bull_floor": float(
                                                regime_policy.bull_floor
                                            ),
                                            "regime_sideways_floor": float(
                                                regime_policy.sideways_floor
                                            ),
                                            "regime_bear_floor": float(
                                                regime_policy.bear_floor
                                            ),
                                            "regime_risk_off_cap": (
                                                float(regime_policy.risk_off_cap)
                                                if regime_policy.risk_off_cap is not None
                                                else pd.NA
                                            ),
                                            "regime_gate_bull_floor": (
                                                float(regime_policy.gate_bull_floor)
                                                if regime_policy.gate_bull_floor is not None
                                                else pd.NA
                                            ),
                                            "threshold": threshold,
                                            "tier_min_threshold": (
                                                tier_thresholds[0]
                                                if tier_thresholds is not None
                                                else pd.NA
                                            ),
                                            "tier_half_threshold": (
                                                tier_thresholds[1]
                                                if tier_thresholds is not None
                                                else pd.NA
                                            ),
                                            "tier_full_threshold": (
                                                tier_thresholds[2]
                                                if tier_thresholds is not None
                                                else pd.NA
                                            ),
                                            "strategy": metrics["strategy"],
                                            "validation_start": (
                                                validation_rows["signal_date"].min()
                                            ),
                                            "validation_end": (
                                                validation_rows["signal_date"].max()
                                            ),
                                            "inner_train_rows": len(
                                                profiled_candidate_train_rows
                                            ),
                                            "validation_rows": len(
                                                profiled_validation_rows
                                            ),
                                            "cumulative_return": metrics[
                                                "cumulative_return"
                                            ],
                                            "max_drawdown": metrics["max_drawdown"],
                                            "sharpe_like": metrics["sharpe_like"],
                                            "total_turnover": total_turnover,
                                            "annualized_turnover": annualized_turnover,
                                            "exposure_changes": exposure_changes,
                                            "average_exposure": average_exposure,
                                            "buy_hold_cumulative_return": buy_hold_return,
                                            "excess_cumulative_return": excess_return,
                                            "selection_benchmark_strategies": ",".join(
                                                selection_benchmark_names
                                            ),
                                            "selection_benchmark_excess_cumulative_returns": benchmark_excess_summary,
                                            "min_benchmark_excess_cumulative_return": (
                                                min_benchmark_excess_return
                                            ),
                                            "selection_validation_cost_bps": ",".join(
                                                f"{cost_bps:g}"
                                                for cost_bps in selection_validation_cost_bps
                                            ),
                                            "selection_validation_cost_benchmark_excess_cumulative_returns": ";".join(
                                                validation_cost_benchmark_excess_summary_parts
                                            ),
                                            "min_selection_validation_cost_benchmark_excess_cumulative_return": (
                                                min_validation_cost_benchmark_excess_return
                                            ),
                                            "validation_predicted_25_fraction": (
                                                validation_predicted_25_fraction
                                            ),
                                            "validation_predicted_50_fraction": (
                                                validation_predicted_50_fraction
                                            ),
                                            "validation_predicted_100_fraction": (
                                                validation_predicted_100_fraction
                                            ),
                                            **score_validity_metrics,
                                            "validation_score_policy_repair_authorized": (
                                                score_policy_repair_authorized
                                            ),
                                            "score_policy_repair_authorized": (
                                                score_policy_repair_authorized
                                            ),
                                            "score_policy_repair_denied_reason": (
                                                score_policy_repair_denied_reason
                                            ),
                                            "validation_guarded_gate_bull_risk_off_override_authorized": (
                                                guarded_gate_bull_risk_off_override_authorized
                                            ),
                                            "guarded_gate_bull_risk_off_override_authorized": (
                                                guarded_gate_bull_risk_off_override_authorized
                                            ),
                                            "guarded_gate_bull_risk_off_override_denied_reason": (
                                                guarded_gate_bull_risk_off_override_denied_reason
                                            ),
                                            **gate_bull_metrics,
                                            "validation_score_policy_triggered_100_fraction": (
                                                validation_score_policy_triggered_100_fraction
                                            ),
                                            "validation_guarded_gate_bull_risk_off_override_triggered_fraction": (
                                                validation_guarded_gate_bull_risk_off_override_triggered_fraction
                                            ),
                                            "validation_score_transform_applied_fraction": (
                                                validation_score_transform_applied_fraction
                                            ),
                                            "min_validation_predicted_target_fraction": (
                                                min_validation_predicted_target_fraction
                                            ),
                                            "sharpe_like_delta": sharpe_delta,
                                            "drawdown_delta": drawdown_delta,
                                            "active_candidate": active_candidate,
                                            "failure_reasons": failure_reasons,
                                            "passed_gate": passed_gate,
                                        }
                                        candidate_rows.append(row)
                                        fold_candidates.append(row)

        selected, selected_source = _select_ml_strategy_candidate_for_policy(
            fold_candidates,
            selection_policy=tuning.selection_policy,
            allow_score_validity_fallback=score_validity_selection,
        )
        if selected is None:
            if fallback_regime_policy is not None:
                fallback_weights, fallback_selection = _deterministic_fallback_selection(
                    fold_id=fold.fold_id,
                    test_rows=test_rows,
                    validation_rows=validation_rows,
                    inner_train_rows=inner_train_rows,
                )
                selected_weight_frames.append(fallback_weights)
                selection_rows.append(fallback_selection)
                continue
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
                    "selection_policy": tuning.selection_policy,
                    "selection_source": "none",
                    "allocation_mode": allocation_mode,
                    "selected_model_name": pd.NA,
                    "selected_utility_profile": pd.NA,
                    "selected_utility_drawdown_penalty": pd.NA,
                    "selected_utility_volatility_penalty": pd.NA,
                    "selected_utility_risk_penalty_power": pd.NA,
                    "allocation_class_weighting": tuning.allocation_class_weighting,
                    "allocation_score_policy": tuning.allocation_score_policy,
                    "allocation_score_policy_prob100_threshold": (
                        tuning.allocation_score_policy_prob100_threshold
                    ),
                    **_selected_score_transform_metadata(None),
                    "calibration_status": pd.NA,
                    "selected_rolling_train_bars": pd.NA,
                    "selected_min_holding_period_bars": pd.NA,
                    "selected_hysteresis_margin": pd.NA,
                    "selected_regime_policy": pd.NA,
                    "selected_regime_bull_floor": pd.NA,
                    "selected_regime_sideways_floor": pd.NA,
                    "selected_regime_bear_floor": pd.NA,
                    "selected_regime_risk_off_cap": pd.NA,
                    "selected_regime_gate_bull_floor": pd.NA,
                    "selected_threshold": pd.NA,
                    "selected_tier_min_threshold": pd.NA,
                    "selected_tier_half_threshold": pd.NA,
                    "selected_tier_full_threshold": pd.NA,
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
                    "selection_benchmark_strategies": ",".join(selection_benchmark_names),
                    "selection_benchmark_excess_cumulative_returns": pd.NA,
                    "min_benchmark_excess_cumulative_return": pd.NA,
                    "selection_validation_cost_bps": ",".join(
                        f"{cost_bps:g}" for cost_bps in selection_validation_cost_bps
                    ),
                    "selection_validation_cost_benchmark_excess_cumulative_returns": pd.NA,
                    "min_selection_validation_cost_benchmark_excess_cumulative_return": pd.NA,
                    "validation_predicted_25_fraction": pd.NA,
                    "validation_predicted_50_fraction": pd.NA,
                    "validation_predicted_100_fraction": pd.NA,
                    "validation_score_forward_return_correlation": pd.NA,
                    "validation_raw_score_forward_return_correlation": pd.NA,
                    "validation_score_target_correlation": pd.NA,
                    "validation_score_policy_repair_authorized": False,
                    "score_policy_repair_authorized": False,
                    "score_policy_repair_denied_reason": (
                        "no_valid_candidate"
                        if tuning.allocation_score_policy
                        == "gate_bull_prob100_threshold"
                        else ""
                    ),
                    "validation_guarded_gate_bull_risk_off_override_authorized": False,
                    "guarded_gate_bull_risk_off_override_authorized": False,
                    "guarded_gate_bull_risk_off_override_denied_reason": (
                        "no_valid_candidate"
                        if tuning.guarded_gate_bull_risk_off_override
                        else ""
                    ),
                    "validation_gate_bull_average_exposure": pd.NA,
                    "validation_gate_bull_underexposed_positive_benchmark_fraction": pd.NA,
                    "validation_gate_bull_underexposed_positive_benchmark_return_sum": pd.NA,
                    "validation_score_policy_triggered_100_fraction": pd.NA,
                    "validation_guarded_gate_bull_risk_off_override_triggered_fraction": pd.NA,
                    "validation_score_transform_applied_fraction": pd.NA,
                    "min_validation_predicted_target_fraction": pd.NA,
                    "sharpe_like_delta": pd.NA,
                    "drawdown_delta": pd.NA,
                    "annualized_turnover": pd.NA,
                    "exposure_changes": 0,
                    "average_exposure": 0.0,
                    "selected_candidate_failure_reasons": pd.NA,
                }
            )
            continue
        selected_model = str(selected["model_name"])
        selected_rolling_train_bars = (
            int(selected["rolling_train_bars"])
            if pd.notna(selected["rolling_train_bars"])
            else None
        )
        selected_min_holding_period_bars = int(selected["min_holding_period_bars"])
        selected_hysteresis_margin = float(selected["hysteresis_margin"])
        selected_threshold = float(selected["threshold"])
        selected_prob100_threshold = float(
            selected["allocation_score_policy_prob100_threshold"]
        )
        selected_score_policy_repair_authorized = bool(
            selected["validation_score_policy_repair_authorized"]
        )
        selected_score_policy_repair_denied_reason = (
            str(selected["score_policy_repair_denied_reason"])
            if pd.notna(selected["score_policy_repair_denied_reason"])
            else ""
        )
        selected_guarded_gate_bull_risk_off_override_authorized = bool(
            selected["validation_guarded_gate_bull_risk_off_override_authorized"]
        )
        selected_guarded_gate_bull_risk_off_override_denied_reason = (
            str(selected["guarded_gate_bull_risk_off_override_denied_reason"])
            if pd.notna(
                selected["guarded_gate_bull_risk_off_override_denied_reason"]
            )
            else ""
        )
        selected_score_transform = AllocationScoreTransformConfig(
            name=str(selected["allocation_score_transform"]),
            bull_multiplier=float(selected["score_transform_bull_multiplier"]),
            bull_addend=float(selected["score_transform_bull_addend"]),
            risk_off_score_cap=(
                float(selected["score_transform_risk_off_score_cap"])
                if pd.notna(selected["score_transform_risk_off_score_cap"])
                else None
            ),
            non_bull_score_cap=(
                float(selected["score_transform_non_bull_score_cap"])
                if pd.notna(selected["score_transform_non_bull_score_cap"])
                else None
            ),
        )
        selected_regime_policy = RegimeParticipationPolicyConfig(
            name=str(selected["regime_policy"]),
            bull_floor=float(selected["regime_bull_floor"]),
            sideways_floor=float(selected["regime_sideways_floor"]),
            bear_floor=float(selected["regime_bear_floor"]),
            risk_off_cap=(
                float(selected["regime_risk_off_cap"])
                if pd.notna(selected["regime_risk_off_cap"])
                else None
            ),
            gate_bull_floor=(
                float(selected["regime_gate_bull_floor"])
                if pd.notna(selected["regime_gate_bull_floor"])
                else None
            ),
        )
        selected_utility_profile = AllocationUtilityProfileConfig(
            name=str(selected["utility_profile"]),
            drawdown_penalty=float(selected["utility_drawdown_penalty"]),
            volatility_penalty=float(selected["utility_volatility_penalty"]),
            risk_penalty_power=float(selected["utility_risk_penalty_power"]),
        )
        selected_tier_thresholds: tuple[float, float, float] | None = None
        if allocation_mode == "tiered":
            selected_tier_thresholds = (
                float(selected["tier_min_threshold"]),
                float(selected["tier_half_threshold"]),
                float(selected["tier_full_threshold"]),
            )
        selected_train_rows = _apply_allocation_profile(
            _latest_training_rows(outer_train_rows, selected_rolling_train_bars),
            config=config,
            profile=selected_utility_profile,
        )
        selected_test_rows = _apply_allocation_profile(
            test_rows,
            config=config,
            profile=selected_utility_profile,
        )
        selected_test_output = _score_model_rows(
            model_name=selected_model,
            target_type=config.target.type,
            train_rows=selected_train_rows,
            score_rows=selected_test_rows,
            feature_columns=feature_columns,
            fold_id=fold.fold_id,
            allocation_class_weighting=tuning.allocation_class_weighting,
            allocation_partial_class_weight_multiplier=(
                tuning.allocation_partial_class_weight_multiplier
            ),
            allocation_probability_calibration=tuning.allocation_probability_calibration,
            allocation_calibration_cv=tuning.allocation_calibration_cv,
            allocation_score_policy=tuning.allocation_score_policy,
            allocation_score_policy_prob100_threshold=(
                selected_prob100_threshold
            ),
            score_policy_repair_authorized=(
                selected_score_policy_repair_authorized
            ),
            score_policy_repair_denied_reason=(
                selected_score_policy_repair_denied_reason
            ),
            allocation_score_transform=selected_score_transform,
            utility_profile=selected_utility_profile,
        )
        if selected_test_output is None:
            if fallback_regime_policy is not None:
                fallback_weights, fallback_selection = _deterministic_fallback_selection(
                    fold_id=fold.fold_id,
                    test_rows=test_rows,
                    validation_rows=validation_rows,
                    inner_train_rows=inner_train_rows,
                )
                selected_weight_frames.append(fallback_weights)
                selection_rows.append(fallback_selection)
                continue
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
            selected_test_predictions = _prediction_frame_with_allocation_score_policy(
                predictions=selected_test_output.predictions,
                allocation_score_policy=tuning.allocation_score_policy,
                prob100_threshold=selected_prob100_threshold,
                score_policy_repair_authorized=(
                    selected_score_policy_repair_authorized
                ),
                score_policy_repair_denied_reason=(
                    selected_score_policy_repair_denied_reason
                ),
                guarded_gate_bull_risk_off_override_enabled=(
                    tuning.guarded_gate_bull_risk_off_override
                ),
                guarded_gate_bull_risk_off_override_authorized=(
                    selected_guarded_gate_bull_risk_off_override_authorized
                ),
                guarded_gate_bull_risk_off_override_denied_reason=(
                    selected_guarded_gate_bull_risk_off_override_denied_reason
                ),
                score_transform=selected_score_transform,
            )
            selected_test_predictions["selected_regime_gate_bull_floor"] = (
                float(selected_regime_policy.gate_bull_floor)
                if selected_regime_policy.gate_bull_floor is not None
                else pd.NA
            )
            selected_weight_frames.append(
                _weights_for_predictions(
                    config=config,
                    panel=panel,
                    predictions=selected_test_predictions,
                    threshold=selected_threshold,
                    strategy_name=ML_TUNED_STRATEGY_NAME,
                    tier_thresholds=selected_tier_thresholds,
                    min_holding_period_bars=selected_min_holding_period_bars,
                    hysteresis_margin=selected_hysteresis_margin,
                    direct_tiered=allocation_mode == "direct_tiered",
                    regime_policy=_strategy_regime_policy(selected_regime_policy),
                )
            )
            if _is_allocation_target(config.target.type):
                allocation_probability_frames.append(
                    _allocation_probability_diagnostics(selected_test_predictions)
                )
                if not selected_test_output.feature_importance.empty:
                    feature_importance_frames.append(selected_test_output.feature_importance)
            selection_status = "selected"

        selection_rows.append(
            {
                "fold_id": fold.fold_id,
                "selection_status": selection_status,
                "selection_policy": tuning.selection_policy,
                "selection_source": (
                    selected_source if selection_status == "selected" else "none"
                ),
                "allocation_mode": allocation_mode,
                "selected_model_name": selected_model if selection_status == "selected" else pd.NA,
                "selected_utility_profile": (
                    selected_utility_profile.name
                    if selection_status == "selected"
                    else pd.NA
                ),
                "selected_utility_drawdown_penalty": (
                    float(selected_utility_profile.drawdown_penalty)
                    if selection_status == "selected"
                    else pd.NA
                ),
                "selected_utility_volatility_penalty": (
                    float(selected_utility_profile.volatility_penalty)
                    if selection_status == "selected"
                    else pd.NA
                ),
                "selected_utility_risk_penalty_power": (
                    float(selected_utility_profile.risk_penalty_power)
                    if selection_status == "selected"
                    else pd.NA
                ),
                "allocation_class_weighting": tuning.allocation_class_weighting,
                "allocation_score_policy": tuning.allocation_score_policy,
                "allocation_score_policy_prob100_threshold": (
                    selected_prob100_threshold
                    if selection_status == "selected"
                    else tuning.allocation_score_policy_prob100_threshold
                ),
                **_selected_score_transform_metadata(
                    selected_score_transform if selection_status == "selected" else None
                ),
                "calibration_status": (
                    selected_test_output.calibration_status
                    if selection_status == "selected" and selected_test_output is not None
                    else selected.get("calibration_status", pd.NA)
                ),
                "selected_rolling_train_bars": (
                    selected_rolling_train_bars
                    if selected_rolling_train_bars is not None and selection_status == "selected"
                    else pd.NA
                ),
                "selected_min_holding_period_bars": (
                    selected_min_holding_period_bars
                    if selection_status == "selected"
                    else pd.NA
                ),
                "selected_hysteresis_margin": (
                    selected_hysteresis_margin if selection_status == "selected" else pd.NA
                ),
                "selected_regime_policy": (
                    selected_regime_policy.name
                    if selection_status == "selected"
                    else pd.NA
                ),
                "selected_regime_bull_floor": (
                    float(selected_regime_policy.bull_floor)
                    if selection_status == "selected"
                    else pd.NA
                ),
                "selected_regime_sideways_floor": (
                    float(selected_regime_policy.sideways_floor)
                    if selection_status == "selected"
                    else pd.NA
                ),
                "selected_regime_bear_floor": (
                    float(selected_regime_policy.bear_floor)
                    if selection_status == "selected"
                    else pd.NA
                ),
                "selected_regime_risk_off_cap": (
                    float(selected_regime_policy.risk_off_cap)
                    if (
                        selection_status == "selected"
                        and selected_regime_policy.risk_off_cap is not None
                    )
                    else pd.NA
                ),
                "selected_regime_gate_bull_floor": (
                    float(selected_regime_policy.gate_bull_floor)
                    if (
                        selection_status == "selected"
                        and selected_regime_policy.gate_bull_floor is not None
                    )
                    else pd.NA
                ),
                "selected_threshold": (
                    selected_threshold if selection_status == "selected" else pd.NA
                ),
                "selected_tier_min_threshold": (
                    selected_tier_thresholds[0]
                    if selected_tier_thresholds is not None and selection_status == "selected"
                    else pd.NA
                ),
                "selected_tier_half_threshold": (
                    selected_tier_thresholds[1]
                    if selected_tier_thresholds is not None and selection_status == "selected"
                    else pd.NA
                ),
                "selected_tier_full_threshold": (
                    selected_tier_thresholds[2]
                    if selected_tier_thresholds is not None and selection_status == "selected"
                    else pd.NA
                ),
                "selected_strategy": ML_TUNED_STRATEGY_NAME,
                "validation_start": selected["validation_start"],
                "validation_end": selected["validation_end"],
                "inner_train_rows": selected["inner_train_rows"],
                "validation_rows": selected["validation_rows"],
                "passed_gate": bool(selected["passed_gate"]) and selection_status == "selected",
                "excess_cumulative_return": selected["excess_cumulative_return"],
                "selection_benchmark_strategies": selected[
                    "selection_benchmark_strategies"
                ],
                "selection_benchmark_excess_cumulative_returns": selected[
                    "selection_benchmark_excess_cumulative_returns"
                ],
                "min_benchmark_excess_cumulative_return": selected[
                    "min_benchmark_excess_cumulative_return"
                ],
                "selection_validation_cost_bps": selected[
                    "selection_validation_cost_bps"
                ],
                "selection_validation_cost_benchmark_excess_cumulative_returns": selected[
                    "selection_validation_cost_benchmark_excess_cumulative_returns"
                ],
                "min_selection_validation_cost_benchmark_excess_cumulative_return": selected[
                    "min_selection_validation_cost_benchmark_excess_cumulative_return"
                ],
                "validation_predicted_25_fraction": selected[
                    "validation_predicted_25_fraction"
                ],
                "validation_predicted_50_fraction": selected[
                    "validation_predicted_50_fraction"
                ],
                "validation_predicted_100_fraction": selected[
                    "validation_predicted_100_fraction"
                ],
                "validation_score_forward_return_correlation": selected[
                    "validation_score_forward_return_correlation"
                ],
                "validation_raw_score_forward_return_correlation": selected[
                    "validation_raw_score_forward_return_correlation"
                ],
                "validation_score_target_correlation": selected[
                    "validation_score_target_correlation"
                ],
                "validation_score_policy_repair_authorized": selected[
                    "validation_score_policy_repair_authorized"
                ],
                "score_policy_repair_authorized": (
                    selected_score_policy_repair_authorized
                    if selection_status == "selected"
                    else False
                ),
                "score_policy_repair_denied_reason": (
                    selected_score_policy_repair_denied_reason
                    if selection_status == "selected"
                    else ""
                ),
                "validation_guarded_gate_bull_risk_off_override_authorized": selected[
                    "validation_guarded_gate_bull_risk_off_override_authorized"
                ],
                "guarded_gate_bull_risk_off_override_authorized": (
                    selected_guarded_gate_bull_risk_off_override_authorized
                    if selection_status == "selected"
                    else False
                ),
                "guarded_gate_bull_risk_off_override_denied_reason": (
                    selected_guarded_gate_bull_risk_off_override_denied_reason
                    if selection_status == "selected"
                    else ""
                ),
                "validation_gate_bull_average_exposure": selected[
                    "validation_gate_bull_average_exposure"
                ],
                "validation_gate_bull_underexposed_positive_benchmark_fraction": selected[
                    "validation_gate_bull_underexposed_positive_benchmark_fraction"
                ],
                "validation_gate_bull_underexposed_positive_benchmark_return_sum": selected[
                    "validation_gate_bull_underexposed_positive_benchmark_return_sum"
                ],
                "validation_score_policy_triggered_100_fraction": selected[
                    "validation_score_policy_triggered_100_fraction"
                ],
                "validation_guarded_gate_bull_risk_off_override_triggered_fraction": selected[
                    "validation_guarded_gate_bull_risk_off_override_triggered_fraction"
                ],
                "validation_score_transform_applied_fraction": selected[
                    "validation_score_transform_applied_fraction"
                ],
                "min_validation_predicted_target_fraction": selected[
                    "min_validation_predicted_target_fraction"
                ],
                "sharpe_like_delta": selected["sharpe_like_delta"],
                "drawdown_delta": selected["drawdown_delta"],
                "annualized_turnover": selected["annualized_turnover"],
                "exposure_changes": selected["exposure_changes"],
                "average_exposure": selected["average_exposure"],
                "selected_candidate_failure_reasons": (
                    selected["failure_reasons"]
                    if selected_source == "best_active_fallback"
                    else ""
                ),
            }
        )

    candidates = pd.DataFrame(candidate_rows, columns=candidate_columns)
    if not candidates.empty:
        candidates = candidates.sort_values(
            [
                "fold_id",
                "passed_gate",
                "min_selection_validation_cost_benchmark_excess_cumulative_return",
                "min_benchmark_excess_cumulative_return",
                "excess_cumulative_return",
                "drawdown_delta",
                "sharpe_like_delta",
                "annualized_turnover",
                "model_name",
                "allocation_score_transform",
                "regime_policy",
                "threshold",
                "hysteresis_margin",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
        ).reset_index(drop=True)
    selections = pd.DataFrame(selection_rows, columns=selection_columns)
    if not selections.empty:
        selections = selections.sort_values("fold_id").reset_index(drop=True)
    if not selected_weight_frames:
        return (
            None,
            candidates,
            selections,
            pd.DataFrame(columns=allocation_probability_columns),
            pd.DataFrame(columns=feature_importance_columns),
            pd.DataFrame(columns=["strategy", "effective_date", "symbol", "weight"]),
        )

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
    allocation_probability_diagnostics = (
        pd.concat(allocation_probability_frames, ignore_index=True)
        if allocation_probability_frames
        else pd.DataFrame(columns=allocation_probability_columns)
    )
    feature_importance = (
        pd.concat(feature_importance_frames, ignore_index=True)
        if feature_importance_frames
        else pd.DataFrame(columns=feature_importance_columns)
    )
    return (
        tuned_result,
        candidates,
        selections,
        allocation_probability_diagnostics,
        feature_importance,
        selected_weights,
    )


def train_models(config: ExperimentConfig) -> TrainModelsArtifacts:
    if not config.models:
        raise RuntimeError("No models are configured for train-models.")
    if _is_allocation_target(config.target.type):
        raise RuntimeError(
            "train-models does not support allocation_utility or regime_state; use run-experiment."
        )

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

    buy_hold_performance = baseline_outputs.performance.loc[
        baseline_outputs.performance["strategy"].astype(str) == "buy_hold"
    ].copy()
    if config.evaluation.ml_strategy_tuning.enabled and buy_hold_performance.empty:
        raise RuntimeError("ML strategy tuning requires the buy_hold baseline.")

    training_outputs = None
    ml_outputs: BacktestResult | None = None
    model_summary: pd.DataFrame | None = None
    fold_summary: pd.DataFrame | None = None
    ml_strategy_threshold_sweep: pd.DataFrame | None = None
    ranking_diagnostics: pd.DataFrame | None = None
    calibration_diagnostics: pd.DataFrame | None = None
    score_histograms: pd.DataFrame | None = None
    threshold_diagnostics: pd.DataFrame | None = None

    if not _is_allocation_target(config.target.type):
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
        model_summary = build_model_summary(
            model_metrics=training_outputs.metrics,
            model_manifest=training_outputs.manifest,
        )
        fold_summary = build_fold_summary(
            model_metrics=training_outputs.metrics,
            model_manifest=training_outputs.manifest,
        )
        ranking_diagnostics = training_outputs.ranking_diagnostics
        calibration_diagnostics = training_outputs.calibration_diagnostics
        score_histograms = training_outputs.score_histograms
        threshold_diagnostics = training_outputs.threshold_diagnostics

    ml_strategy_tuning_result: BacktestResult | None = None
    ml_strategy_tuning_candidates: pd.DataFrame | None = None
    ml_strategy_tuning_selections: pd.DataFrame | None = None
    allocation_probability_diagnostics: pd.DataFrame | None = None
    feature_importance: pd.DataFrame | None = None
    if config.evaluation.ml_strategy_tuning.enabled:
        (
            ml_strategy_tuning_result,
            ml_strategy_tuning_candidates,
            ml_strategy_tuning_selections,
            allocation_probability_diagnostics,
            feature_importance,
            _,
        ) = _build_ml_strategy_tuning_outputs(
            config=config,
            panel=panel,
            modeling_dataset=modeling_dataset,
            folds=folds,
            buy_hold_performance=buy_hold_performance,
            baseline_performance=baseline_outputs.performance,
        )
    oos_dates = _shared_oos_dates(
        panel=panel,
        modeling_dataset=modeling_dataset,
        folds=folds,
        frequency=config.portfolio.ranking.rebalance_frequency,
    )
    result_frames = [baseline_outputs]
    if ml_outputs is not None:
        result_frames.append(ml_outputs)
    if ml_strategy_tuning_result is not None:
        result_frames.append(ml_strategy_tuning_result)
    combined_outputs = _concat_backtest_results(result_frames)
    oos_outputs = _slice_backtest_result(combined_outputs, oos_dates)
    if training_outputs is not None and training_outputs.predictions is not None:
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
    allocation_target_diagnostics = (
        _allocation_target_diagnostics(modeling_dataset=modeling_dataset, folds=folds)
        if _is_allocation_target(config.target.type)
        else None
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
        ranking_diagnostics=ranking_diagnostics,
        calibration_diagnostics=calibration_diagnostics,
        score_histograms=score_histograms,
        threshold_diagnostics=threshold_diagnostics,
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
        allocation_target_diagnostics=allocation_target_diagnostics,
        allocation_probability_diagnostics=allocation_probability_diagnostics,
        feature_importance=feature_importance,
    )


def evaluate_shadow_candidate(
    *,
    config: ExperimentConfig,
    panel: pd.DataFrame,
    signal_date: date,
) -> ShadowCandidateEvaluation:
    """Score one completed BTC bar through the frozen Phase 8 candidate path."""
    working_panel = panel.copy()
    working_panel["timestamp"] = pd.to_datetime(working_panel["timestamp"])
    working_panel = working_panel.loc[
        working_panel["timestamp"].dt.date <= signal_date
    ].copy()
    if working_panel.empty:
        raise RuntimeError("Shadow evaluation requires completed panel rows.")
    latest_panel_date = pd.Timestamp(working_panel["timestamp"].max()).date()
    if latest_panel_date != signal_date:
        raise RuntimeError(
            "Shadow evaluation panel must end on the completed signal date."
        )

    modeling_dataset = build_modeling_dataset(working_panel, config)
    scoring_dataset = build_scoring_dataset(working_panel, config)
    score_rows = scoring_dataset.loc[
        pd.to_datetime(scoring_dataset["signal_date"]).dt.date.eq(signal_date)
    ].copy()
    if len(score_rows) != 1:
        raise RuntimeError(
            "Shadow evaluation requires exactly one BTC scoring row for the signal date."
        )

    score_rows["target_end_date"] = pd.Timestamp(signal_date)
    score_rows["forward_return"] = 0.0
    score_rows["forward_drawdown"] = 0.0
    score_rows["forward_realized_volatility"] = 0.0
    score_rows["target"] = 0
    score_rows["target_weight"] = 0.0
    for suffix in (0, 25, 50, 100):
        score_rows[f"allocation_utility_{suffix}"] = 0.0

    label_cutoff = pd.Timestamp(signal_date)
    mature_rows = modeling_dataset.loc[
        pd.to_datetime(modeling_dataset["target_end_date"]).le(label_cutoff)
    ].copy()
    train_start = label_cutoff - pd.DateOffset(
        years=config.evaluation.walk_forward.train_years
    )
    mature_rows = mature_rows.loc[
        pd.to_datetime(mature_rows["signal_date"]).ge(train_start)
    ].copy()
    if mature_rows.empty:
        raise RuntimeError("Shadow evaluation has no mature training rows.")

    combined_dataset = pd.concat(
        [mature_rows, score_rows.reindex(columns=mature_rows.columns)],
        ignore_index=True,
    )
    combined_dataset = _with_completed_bar_gate_labels(
        combined_dataset,
        panel=working_panel,
        config=config,
    )
    mature_rows = combined_dataset.loc[
        pd.to_datetime(combined_dataset["signal_date"]).dt.date.ne(signal_date)
    ]
    fold = WalkForwardFold(
        fold_id=1,
        train_start=pd.Timestamp(mature_rows["signal_date"].min()),
        train_end=pd.Timestamp(mature_rows["signal_date"].max()),
        label_cutoff=label_cutoff,
        test_start=pd.Timestamp(signal_date),
        test_end=pd.Timestamp(signal_date),
        train_rows=len(mature_rows),
        test_rows=1,
    )

    baseline_outputs = run_baselines(config, working_panel)[0]
    buy_hold_performance = baseline_outputs.performance.loc[
        baseline_outputs.performance["strategy"].astype(str).eq("buy_hold")
    ].copy()
    (
        _,
        candidates,
        selections,
        probability_diagnostics,
        _,
        selected_weights,
    ) = _build_ml_strategy_tuning_outputs(
        config=config,
        panel=working_panel,
        modeling_dataset=combined_dataset,
        folds=[fold],
        buy_hold_performance=buy_hold_performance,
        baseline_performance=baseline_outputs.performance,
    )
    if selections.empty or selected_weights.empty:
        raise RuntimeError("Frozen shadow candidate did not produce a selection.")

    selection = selections.iloc[-1]
    selection_source = str(selection["selection_source"])
    if selection_source == "none":
        raise RuntimeError("Frozen shadow candidate produced no valid selection.")
    effective_date = pd.Timestamp(score_rows.iloc[0]["effective_date"])
    weight_rows = selected_weights.loc[
        pd.to_datetime(selected_weights["effective_date"]).eq(effective_date)
    ]
    if len(weight_rows) != 1:
        raise RuntimeError("Frozen shadow candidate did not produce one BTC allocation.")
    target_allocation = float(weight_rows.iloc[0]["weight"])

    scored = probability_diagnostics.loc[
        pd.to_datetime(probability_diagnostics["signal_date"]).dt.date.eq(signal_date)
    ]
    score_row = scored.iloc[-1] if not scored.empty else None
    raw_score = (
        float(score_row["raw_expected_allocation_score"])
        if score_row is not None
        and pd.notna(score_row.get("raw_expected_allocation_score"))
        else None
    )
    latest_row = combined_dataset.loc[
        pd.to_datetime(combined_dataset["signal_date"]).dt.date.eq(signal_date)
    ].iloc[-1]
    regime = (
        str(score_row["runtime_regime"])
        if score_row is not None and pd.notna(score_row.get("runtime_regime"))
        else _allocation_regime_label(latest_row)
    )
    diagnostics = {
        "candidate_count": int(len(candidates)),
        "selection": _json_mapping(selection.to_dict()),
        "prediction": _json_mapping(score_row.to_dict()) if score_row is not None else {},
    }
    return ShadowCandidateEvaluation(
        selection_source=selection_source,
        target_allocation=target_allocation,
        raw_score=raw_score,
        selected_tier=target_allocation,
        regime_classification=regime,
        diagnostics=diagnostics,
    )


def _json_mapping(values: dict[object, object]) -> dict[str, object]:
    return {str(key): _json_value(value) for key, value in values.items()}


def _json_value(value: object) -> object:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value








