from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketlab.config import ExperimentConfig
from marketlab.reports.markdown import write_markdown_report


def _base_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["alpha"],
            "cumulative_return": [0.10],
            "annualized_return": [0.12],
            "annualized_volatility": [0.20],
            "sharpe_like": [0.60],
            "max_drawdown": [-0.05],
            "hit_rate": [0.55],
            "avg_turnover": [99.0],
            "total_turnover": [999.0],
        }
    )


def _base_performance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "strategy": ["alpha", "alpha"],
            "gross_return": [0.01, 0.02],
            "net_return": [0.009, 0.018],
            "turnover": [1.0, 3.0],
            "equity": [1.009, 1.027162],
        }
    )


def _strategy_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["alpha"],
            "start_date": pd.to_datetime(["2024-01-02"]),
            "end_date": pd.to_datetime(["2024-01-03"]),
            "trading_days": [2],
            "final_equity": [1.027162],
            "gross_final_equity": [1.0302],
            "gross_cumulative_return": [0.0302],
            "cumulative_return": [0.027162],
            "cost_drag": [0.003038],
            "annualized_return": [0.12],
            "annualized_volatility": [0.20],
            "sharpe_like": [0.60],
            "max_drawdown": [-0.05],
            "hit_rate": [0.55],
            "avg_turnover": [99.0],
            "total_turnover": [999.0],
            "avg_cost_return": [0.50],
            "total_cost_return": [1.00],
            "avg_long_exposure": [0.60],
            "avg_short_exposure": [0.20],
            "avg_gross_exposure": [0.80],
            "avg_net_exposure": [0.40],
            "avg_cash_weight": [0.20],
            "avg_engine_cash_weight": [1.05],
            "avg_active_positions": [3.0],
            "max_position_weight": [0.40],
            "max_group_weight": [0.55],
            "benchmark_strategy": [""],
            "excess_cumulative_return": [float("nan")],
            "annualized_excess_return": [float("nan")],
            "tracking_error": [float("nan")],
            "information_ratio": [float("nan")],
            "correlation_to_benchmark": [float("nan")],
            "up_capture": [float("nan")],
            "down_capture": [float("nan")],
        }
    )


def _cost_sensitivity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["alpha", "alpha"],
            "bps_per_trade": [0.0, 10.0],
            "gross_cumulative_return": [0.0302, 0.0302],
            "cumulative_return": [0.0302, 0.027162],
            "cost_drag": [0.0, 0.003038],
            "final_equity": [1.0302, 1.027162],
            "annualized_return": [0.13, 0.12],
            "annualized_volatility": [0.19, 0.20],
            "sharpe_like": [0.68, 0.60],
            "max_drawdown": [-0.04, -0.05],
            "hit_rate": [0.55, 0.55],
            "avg_turnover": [2.0, 2.0],
            "total_turnover": [4.0, 4.0],
            "avg_cost_return": [0.0, 0.0015],
            "total_cost_return": [0.0, 0.003],
        }
    )


def test_write_markdown_report_turnover_section_uses_turnover_costs_input(tmp_path: Path) -> None:
    config = ExperimentConfig(experiment_name="markdown_fixture")
    turnover_costs = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "strategy": ["alpha", "alpha"],
            "turnover": [1.0, 3.0],
            "gross_return": [0.01, 0.02],
            "net_return": [0.009, 0.018],
            "cost_return": [0.001, 0.002],
        }
    )

    report_path = write_markdown_report(
        config=config,
        metrics=_base_metrics(),
        performance=_base_performance(),
        path=tmp_path / "report.md",
        strategy_summary=_strategy_summary(),
        turnover_costs=turnover_costs,
    )

    report_text = report_path.read_text(encoding="utf-8")
    turnover_section = report_text.split("## Turnover And Costs", maxsplit=1)[1].split("## ", maxsplit=1)[0]

    assert "## Turnover And Costs" in report_text
    assert "| strategy | avg_turnover | total_turnover | avg_cost_return | total_cost_return |" in turnover_section
    assert "| alpha | 2.0 | 4.0 | 0.0015 | 0.003 |" in turnover_section
    assert "999.0" not in turnover_section
    assert "0.5" not in turnover_section


def test_write_markdown_report_adds_cost_sensitivity_section(tmp_path: Path) -> None:
    config = ExperimentConfig(experiment_name="markdown_fixture")

    report_path = write_markdown_report(
        config=config,
        metrics=_base_metrics(),
        performance=_base_performance(),
        path=tmp_path / "report.md",
        strategy_summary=_strategy_summary(),
        turnover_costs=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "strategy": ["alpha", "alpha"],
                "turnover": [1.0, 3.0],
                "gross_return": [0.01, 0.02],
                "net_return": [0.009, 0.018],
                "cost_return": [0.001, 0.002],
            }
        ),
        cost_sensitivity=_cost_sensitivity(),
    )

    report_text = report_path.read_text(encoding="utf-8")
    cost_section = report_text.split("## Cost Sensitivity", maxsplit=1)[1]

    assert "## Cost Sensitivity" in report_text
    assert (
        "| strategy | bps_per_trade | cumulative_return | annualized_return | max_drawdown | cost_drag |"
        in cost_section
    )
    assert "| alpha | 0.0 | 0.0302 | 0.13 | -0.04 | 0.0 |" in cost_section
    assert "| alpha | 10.0 | 0.027162 | 0.12 | -0.05 | 0.003038 |" in cost_section
    assert "Zero-cost rows are theoretical gross-return baselines" in cost_section
    assert "Higher implementation cost can worsen return and drawdown" in cost_section


def test_write_markdown_report_adds_exposure_summary_section(tmp_path: Path) -> None:
    config = ExperimentConfig(experiment_name="markdown_fixture")

    report_path = write_markdown_report(
        config=config,
        metrics=_base_metrics(),
        performance=_base_performance(),
        path=tmp_path / "report.md",
        strategy_summary=_strategy_summary(),
    )

    report_text = report_path.read_text(encoding="utf-8")
    exposure_section = report_text.split("## Exposure Summary", maxsplit=1)[1]

    assert "## Exposure Summary" in report_text
    assert "| strategy | avg_long_exposure | avg_short_exposure | avg_gross_exposure | avg_net_exposure | avg_cash_weight | avg_engine_cash_weight | avg_active_positions | max_position_weight | max_group_weight |" in exposure_section
    assert "Lower drawdown can reflect lower gross exposure or more cash" in exposure_section
    assert "group_exposure.csv" in exposure_section


def test_write_markdown_report_walk_forward_diagnostics_section_uses_fold_diagnostics_input(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(experiment_name="markdown_fixture")
    fold_diagnostics = pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "fold_id": [1, pd.NA],
            "status": ["used", "skipped"],
            "skip_reasons": ["", "insufficient_train_rows;insufficient_test_positive_rate"],
            "train_start": pd.to_datetime(["2023-01-06", "2023-03-31"]),
            "train_end": pd.to_datetime(["2024-01-05", "2024-05-31"]),
            "label_cutoff": pd.to_datetime(["2024-01-12", "2024-06-07"]),
            "test_start": pd.to_datetime(["2024-01-12", "2024-06-07"]),
            "test_end": pd.to_datetime(["2024-04-05", "2024-07-12"]),
            "train_rows": [110, 80],
            "test_rows": [26, 8],
            "train_positive_rate": [0.5, 0.03],
            "test_positive_rate": [0.5, 0.01],
        }
    )

    report_path = write_markdown_report(
        config=config,
        metrics=_base_metrics(),
        performance=_base_performance(),
        path=tmp_path / "report.md",
        fold_diagnostics=fold_diagnostics,
    )

    report_text = report_path.read_text(encoding="utf-8")
    diagnostics_section = report_text.split("## Walk-Forward Diagnostics", maxsplit=1)[1]

    assert "## Walk-Forward Diagnostics" in report_text
    assert "- Used candidates: 1" in diagnostics_section
    assert "- Skipped candidates: 1" in diagnostics_section
    assert "| candidate_id | test_start | test_end | skip_reasons | train_rows | test_rows | train_positive_rate | test_positive_rate |" in diagnostics_section
    assert "insufficient_train_rows;insufficient_test_positive_rate" in diagnostics_section


def test_write_markdown_report_adds_ranking_aware_model_headline(tmp_path: Path) -> None:
    config = ExperimentConfig(experiment_name="markdown_fixture")
    model_summary = pd.DataFrame(
        {
            "model_name": ["random_forest", "logistic_regression"],
            "mean_roc_auc": [0.54, 0.57],
            "mean_top_bucket_return": [0.01, 0.03],
            "mean_top_bottom_spread": [0.03, 0.01],
        }
    )

    report_path = write_markdown_report(
        config=config,
        metrics=_base_metrics(),
        performance=_base_performance(),
        path=tmp_path / "report.md",
        model_summary=model_summary,
    )

    report_text = report_path.read_text(encoding="utf-8")

    assert "- Best model by mean ROC AUC: `logistic_regression` (0.570000)" in report_text
    assert "- Best model by mean top-bucket return: `logistic_regression` (0.030000)" in report_text
    assert "- Best model by mean top-bottom spread: `random_forest` (0.030000)" in report_text


def test_write_markdown_report_adds_calibration_section_and_plot_links(tmp_path: Path) -> None:
    config = ExperimentConfig(experiment_name="markdown_fixture")
    model_summary = pd.DataFrame(
        {
            "model_name": ["random_forest", "logistic_regression"],
            "mean_roc_auc": [0.54, 0.57],
            "mean_top_bottom_spread": [0.03, 0.01],
            "mean_ece": [0.08, 0.05],
            "mean_max_calibration_gap": [0.18, 0.12],
        }
    )
    threshold_diagnostics = pd.DataFrame(
        {
            "model_name": [
                "logistic_regression",
                "logistic_regression",
                "random_forest",
                "random_forest",
            ],
            "threshold": [0.25, 0.50, 0.25, 0.50],
            "f1": [0.60, 0.65, 0.55, 0.50],
            "balanced_accuracy": [0.58, 0.57, 0.56, 0.62],
        }
    )
    calibration_plot = tmp_path / "calibration_curves.png"
    hist_plot = tmp_path / "score_histograms.png"
    threshold_plot = tmp_path / "threshold_sweeps.png"
    for path in [calibration_plot, hist_plot, threshold_plot]:
        path.write_text("placeholder", encoding="utf-8")

    report_path = write_markdown_report(
        config=config,
        metrics=_base_metrics(),
        performance=_base_performance(),
        path=tmp_path / "report.md",
        model_summary=model_summary,
        threshold_diagnostics=threshold_diagnostics,
        calibration_curves_plot_path=calibration_plot,
        score_histograms_plot_path=hist_plot,
        threshold_sweeps_plot_path=threshold_plot,
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert "## Calibration And Threshold Diagnostics" in report_text
    assert "| model_name | mean_ece | mean_max_calibration_gap |" in report_text
    assert "| logistic_regression | 0.05 | 0.12 |" in report_text
    assert "| random_forest | 0.08 | 0.18 |" in report_text
    assert "| model_name | threshold_max_f1 | max_f1 | threshold_max_balanced_accuracy | max_balanced_accuracy |" in report_text
    assert "![Calibration Curves](calibration_curves.png)" in report_text
    assert "![Score Histograms](score_histograms.png)" in report_text
    assert "![Threshold Sweeps](threshold_sweeps.png)" in report_text



def test_write_markdown_report_adds_benchmark_relative_summary_section(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(experiment_name="markdown_fixture")
    strategy_summary = _strategy_summary()
    strategy_summary.loc[:, "benchmark_strategy"] = "buy_hold"
    strategy_summary.loc[:, "excess_cumulative_return"] = 0.02
    strategy_summary.loc[:, "annualized_excess_return"] = 0.15
    strategy_summary.loc[:, "tracking_error"] = 0.08
    strategy_summary.loc[:, "information_ratio"] = 0.75
    strategy_summary.loc[:, "correlation_to_benchmark"] = 0.92
    strategy_summary.loc[:, "up_capture"] = 0.88
    strategy_summary.loc[:, "down_capture"] = 0.70

    report_path = write_markdown_report(
        config=config,
        metrics=_base_metrics(),
        performance=_base_performance(),
        path=tmp_path / "report.md",
        strategy_summary=strategy_summary,
    )

    report_text = report_path.read_text(encoding="utf-8")

    assert "## Benchmark-Relative Summary" in report_text
    assert (
        "| strategy | benchmark_strategy | excess_cumulative_return | annualized_excess_return | tracking_error | information_ratio | correlation_to_benchmark | up_capture | down_capture |"
        in report_text
    )
    assert "benchmark_relative.csv" in report_text
    assert "active risk" in report_text
