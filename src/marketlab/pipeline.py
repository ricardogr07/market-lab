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
from marketlab.features.engineering import add_feature_set
from marketlab.reports.markdown import write_markdown_report
from marketlab.reports.plots import plot_cumulative_returns, plot_drawdown
from marketlab.strategies.buy_hold import generate_weights as buy_hold_weights
from marketlab.strategies.sma import generate_weights as sma_weights

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


def backtest(config: ExperimentConfig) -> ExperimentArtifacts:
    panel, panel_path = prepare_data(config)
    performance, metrics = run_baselines(config, panel)
    run_dir = _run_dir(config)

    metrics_path = run_dir / "metrics.csv"
    performance_path = run_dir / "performance.csv"
    metrics.to_csv(metrics_path, index=False)
    performance.to_csv(performance_path, index=False)

    report_path: Path | None = None
    if config.artifacts.save_report_md:
        report_path = write_markdown_report(
            config=config,
            metrics=metrics,
            performance=performance,
            path=run_dir / "report.md",
        )

    cumulative_plot_path: Path | None = None
    drawdown_plot_path: Path | None = None
    if config.artifacts.save_plots:
        cumulative_plot_path = plot_cumulative_returns(
            performance=performance,
            path=run_dir / "cumulative_returns.png",
        )
        drawdown_plot_path = plot_drawdown(
            performance=performance,
            path=run_dir / "drawdown.png",
        )

    return ExperimentArtifacts(
        run_dir=run_dir,
        panel_path=panel_path,
        metrics_path=metrics_path,
        performance_path=performance_path,
        report_path=report_path,
        cumulative_plot_path=cumulative_plot_path,
        drawdown_plot_path=drawdown_plot_path,
    )


def run_experiment(config: ExperimentConfig) -> ExperimentArtifacts:
    return backtest(config)
