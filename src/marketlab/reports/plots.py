from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _subplot_axes(model_names: list[str], *, height: float = 4.0) -> tuple[plt.Figure, list[plt.Axes]]:
    figure, axes = plt.subplots(
        len(model_names),
        1,
        figsize=(9, max(height * len(model_names), 4.0)),
        squeeze=False,
    )
    return figure, list(axes.flatten())


def plot_cumulative_returns(performance: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5))
    for strategy, frame in performance.groupby("strategy", sort=False):
        axis.plot(frame["date"], frame["equity"], label=strategy)

    axis.set_title("Cumulative Equity")
    axis.set_xlabel("Date")
    axis.set_ylabel("Equity")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_drawdown(performance: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5))
    for strategy, frame in performance.groupby("strategy", sort=False):
        drawdown = (frame["equity"] / frame["equity"].cummax()) - 1.0
        axis.plot(frame["date"], drawdown, label=strategy)

    axis.set_title("Drawdown")
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_turnover(performance: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5))
    for strategy, frame in performance.groupby("strategy", sort=False):
        axis.plot(frame["date"], frame["turnover"], label=strategy)

    axis.set_title("Turnover")
    axis.set_xlabel("Date")
    axis.set_ylabel("Turnover")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_calibration_curves(calibration_diagnostics: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = sorted(calibration_diagnostics["model_name"].drop_duplicates().tolist())
    figure, axes = _subplot_axes(model_names)

    for axis, model_name in zip(axes, model_names):
        model_rows = calibration_diagnostics.loc[
            (calibration_diagnostics["model_name"] == model_name)
            & (calibration_diagnostics["sample_count"] > 0)
        ].copy()
        if model_rows.empty:
            axis.text(0.5, 0.5, "No occupied score bins", ha="center", va="center")
        else:
            weighted = model_rows.assign(
                weighted_mean_score=model_rows["mean_score"] * model_rows["sample_count"],
                weighted_observed_positive_rate=(
                    model_rows["observed_positive_rate"] * model_rows["sample_count"]
                ),
            )
            aggregated = (
                weighted.groupby(["bin_id", "bin_left", "bin_right"], as_index=False)
                .agg(
                    sample_count=("sample_count", "sum"),
                    weighted_mean_score=("weighted_mean_score", "sum"),
                    weighted_observed_positive_rate=("weighted_observed_positive_rate", "sum"),
                )
                .sort_values("bin_id")
                .reset_index(drop=True)
            )
            aggregated["mean_score"] = (
                aggregated["weighted_mean_score"] / aggregated["sample_count"]
            )
            aggregated["observed_positive_rate"] = (
                aggregated["weighted_observed_positive_rate"] / aggregated["sample_count"]
            )
            axis.plot(
                aggregated["mean_score"],
                aggregated["observed_positive_rate"],
                marker="o",
                label="Observed",
            )
        axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Ideal")
        axis.set_title(f"Calibration Curve: {model_name}")
        axis.set_xlabel("Mean score")
        axis.set_ylabel("Observed positive rate")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.3)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_score_histograms(score_histograms: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = sorted(score_histograms["model_name"].drop_duplicates().tolist())
    figure, axes = _subplot_axes(model_names)

    for axis, model_name in zip(axes, model_names):
        model_rows = score_histograms.loc[score_histograms["model_name"] == model_name].copy()
        aggregated = (
            model_rows.groupby(["target", "bin_id", "bin_left", "bin_right"], as_index=False)
            .agg(sample_count=("sample_count", "sum"))
            .sort_values(["target", "bin_id"])
            .reset_index(drop=True)
        )
        centers = aggregated[["bin_left", "bin_right"]].mean(axis=1)
        for target, color, label in [(0, "tab:blue", "target=0"), (1, "tab:orange", "target=1")]:
            target_rows = aggregated.loc[aggregated["target"] == target].copy()
            target_centers = centers.loc[target_rows.index]
            total_count = int(target_rows["sample_count"].sum())
            heights = (
                target_rows["sample_count"] / total_count if total_count > 0 else target_rows["sample_count"] * 0.0
            )
            axis.bar(target_centers, heights, width=0.08, alpha=0.5, color=color, label=label)
        axis.set_title(f"Score Histogram: {model_name}")
        axis.set_xlabel("Score bin")
        axis.set_ylabel("Fraction within target")
        axis.set_xlim(0.0, 1.0)
        axis.grid(alpha=0.3)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_threshold_sweeps(threshold_diagnostics: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = sorted(threshold_diagnostics["model_name"].drop_duplicates().tolist())
    figure, axes = _subplot_axes(model_names, height=4.5)

    for axis, model_name in zip(axes, model_names):
        model_rows = threshold_diagnostics.loc[
            threshold_diagnostics["model_name"] == model_name
        ].copy()
        aggregated = (
            model_rows.groupby("threshold", as_index=False)
            .agg(
                precision=("precision", "mean"),
                recall=("recall", "mean"),
                f1=("f1", "mean"),
                balanced_accuracy=("balanced_accuracy", "mean"),
                predicted_positive_rate=("predicted_positive_rate", "mean"),
            )
            .sort_values("threshold")
            .reset_index(drop=True)
        )
        axis.plot(aggregated["threshold"], aggregated["precision"], label="precision")
        axis.plot(aggregated["threshold"], aggregated["recall"], label="recall")
        axis.plot(aggregated["threshold"], aggregated["f1"], label="f1")
        axis.plot(aggregated["threshold"], aggregated["balanced_accuracy"], label="balanced_accuracy")
        axis.plot(
            aggregated["threshold"],
            aggregated["predicted_positive_rate"],
            linestyle="--",
            label="predicted_positive_rate",
        )
        axis.set_title(f"Threshold Sweep: {model_name}")
        axis.set_xlabel("Threshold")
        axis.set_ylabel("Metric")
        axis.set_xlim(0.05, 0.95)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.3)
        axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path
