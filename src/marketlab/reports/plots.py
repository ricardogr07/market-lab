from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marketlab.strategies.chart_patterns import PATTERN_COLUMNS


def _subplot_axes(model_names: list[str], *, height: float = 4.0) -> tuple[plt.Figure, list[plt.Axes]]:
    figure, axes = plt.subplots(
        len(model_names),
        1,
        figsize=(9, max(height * len(model_names), 4.0)),
        squeeze=False,
    )
    return figure, list(axes.flatten())


def _set_time_ticks(axis: plt.Axes, timestamps: pd.Series, *, max_ticks: int = 12) -> None:
    if timestamps.empty:
        return
    step = max(1, len(timestamps) // max_ticks)
    tick_positions = list(range(0, len(timestamps), step))
    if tick_positions[-1] != len(timestamps) - 1:
        tick_positions.append(len(timestamps) - 1)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [pd.Timestamp(timestamps.iloc[position]).strftime("%m-%d %H:%M") for position in tick_positions],
        rotation=45,
        ha="right",
    )


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


def plot_signal_price_overlay(diagnostics: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = diagnostics["symbol"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(symbols),
        1,
        figsize=(10, max(4.5 * len(symbols), 4.5)),
        squeeze=False,
    )

    for axis, symbol in zip(axes.flatten(), symbols):
        frame = diagnostics.loc[diagnostics["symbol"] == symbol].sort_values("timestamp")
        axis.plot(frame["timestamp"], frame["close"], label="close", color="black", linewidth=1.5)
        axis.plot(frame["timestamp"], frame["ema_fast"], label="EMA fast", color="tab:blue")
        axis.plot(frame["timestamp"], frame["ema_slow"], label="EMA slow", color="tab:orange")
        axis.plot(
            frame["timestamp"],
            frame["bollinger_upper"],
            label="Bollinger upper",
            color="tab:purple",
            linestyle="--",
            linewidth=0.9,
        )
        axis.plot(
            frame["timestamp"],
            frame["bollinger_lower"],
            label="Bollinger lower",
            color="tab:purple",
            linestyle="--",
            linewidth=0.9,
        )
        if frame["vwap"].notna().any():
            axis.plot(frame["timestamp"], frame["vwap"], label="VWAP", color="tab:green")

        long_rows = frame.loc[frame["target_weight"] > 0.0]
        cash_rows = frame.loc[frame["target_weight"] <= 0.0]
        axis.scatter(
            long_rows["timestamp"],
            long_rows["close"],
            marker="^",
            color="tab:green",
            label="long next bar",
            zorder=3,
        )
        axis.scatter(
            cash_rows["timestamp"],
            cash_rows["close"],
            marker="v",
            color="tab:red",
            label="cash next bar",
            zorder=3,
        )
        axis.set_title(f"Signal Price Overlay: {symbol}")
        axis.set_xlabel("Signal timestamp")
        axis.set_ylabel("Price")
        axis.grid(alpha=0.3)
        axis.legend(loc="best")

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_signal_confirmations(diagnostics: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = diagnostics["symbol"].drop_duplicates().tolist()
    confirmation_columns = [
        "ema_confirmed",
        "rsi_confirmed",
        "macd_confirmed",
        "bollinger_confirmed",
        "volume_confirmed",
        "vwap_confirmed",
        "target_weight",
    ]
    figure, axes = plt.subplots(
        len(symbols),
        1,
        figsize=(10, max(3.5 * len(symbols), 3.5)),
        squeeze=False,
    )

    for axis, symbol in zip(axes.flatten(), symbols):
        frame = diagnostics.loc[diagnostics["symbol"] == symbol].sort_values("timestamp")
        matrix = frame.loc[:, confirmation_columns].copy()
        matrix["target_weight"] = matrix["target_weight"].gt(0.0)
        values = matrix.astype(bool).T.astype(int)
        axis.imshow(values, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
        axis.set_title(f"Signal Confirmations: {symbol}")
        axis.set_yticks(range(len(confirmation_columns)))
        axis.set_yticklabels(confirmation_columns)
        _set_time_ticks(axis, frame["timestamp"])

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_signal_performance_focus(
    performance: pd.DataFrame,
    path: str | Path,
    *,
    strategy_names: set[str] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5))
    selected_strategies = strategy_names or {"indicator_stack", "buy_hold"}
    for strategy, frame in performance.groupby("strategy", sort=False):
        if strategy not in selected_strategies:
            continue
        ordered = frame.sort_values("date").copy()
        ordered["focus_equity"] = (1.0 + ordered["net_return"]).cumprod()
        axis.plot(ordered["date"], ordered["focus_equity"], label=strategy)

    axis.set_title("Focused Window Equity")
    axis.set_xlabel("Date")
    axis.set_ylabel("Rebased equity")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_pattern_price_overlay(diagnostics: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = diagnostics["symbol"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(symbols),
        1,
        figsize=(10, max(4.5 * len(symbols), 4.5)),
        squeeze=False,
    )

    for axis, symbol in zip(axes.flatten(), symbols):
        frame = diagnostics.loc[diagnostics["symbol"] == symbol].sort_values("timestamp")
        axis.plot(frame["timestamp"], frame["close"], label="close", color="black", linewidth=1.3)
        axis.plot(
            frame["timestamp"],
            pd.to_numeric(frame["resistance_level"], errors="coerce"),
            label="resistance",
            color="tab:orange",
            linestyle="--",
            linewidth=0.9,
        )
        axis.plot(
            frame["timestamp"],
            pd.to_numeric(frame["support_level"], errors="coerce"),
            label="support",
            color="tab:blue",
            linestyle="--",
            linewidth=0.9,
        )
        long_rows = frame.loc[frame["target_weight"] > 0.0]
        cash_rows = frame.loc[frame["target_weight"] <= 0.0]
        axis.scatter(
            long_rows["timestamp"],
            long_rows["close"],
            marker="^",
            color="tab:green",
            label="pattern long next bar",
            zorder=3,
        )
        axis.scatter(
            cash_rows["timestamp"],
            cash_rows["close"],
            marker="v",
            color="tab:red",
            label="pattern cash next bar",
            zorder=3,
        )
        axis.set_title(f"Chart Pattern Overlay: {symbol}")
        axis.set_xlabel("Signal timestamp")
        axis.set_ylabel("Price")
        axis.grid(alpha=0.3)
        axis.legend(loc="best")

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_pattern_detections(diagnostics: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = diagnostics["symbol"].drop_duplicates().tolist()
    pattern_columns = [*PATTERN_COLUMNS, "target_weight"]
    figure, axes = plt.subplots(
        len(symbols),
        1,
        figsize=(11, max(7.0 * len(symbols), 7.0)),
        squeeze=False,
    )

    for axis, symbol in zip(axes.flatten(), symbols):
        frame = diagnostics.loc[diagnostics["symbol"] == symbol].sort_values("timestamp")
        matrix = frame.loc[:, pattern_columns].copy()
        matrix["target_weight"] = matrix["target_weight"].gt(0.0)
        values = matrix.astype(bool).T.astype(int)
        axis.imshow(values, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
        axis.set_title(f"Chart Pattern Detections: {symbol}")
        axis.set_yticks(range(len(pattern_columns)))
        axis.set_yticklabels(pattern_columns)
        _set_time_ticks(axis, frame["timestamp"])

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def _active_pattern_label(row: pd.Series) -> str:
    active = [column for column in PATTERN_COLUMNS if bool(row[column])]
    if not active:
        return "no pattern"
    return ", ".join(active)


def plot_pattern_detection_windows(
    diagnostics: pd.DataFrame,
    path: str | Path,
    *,
    bars_before: int = 8,
    bars_after: int = 2,
    max_windows: int = 12,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = diagnostics.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    detection_mask = ordered.loc[:, PATTERN_COLUMNS].astype(bool).any(axis=1)
    hits = ordered.loc[detection_mask].head(max_windows)

    window_count = max(len(hits), 1)
    column_count = min(3, window_count)
    row_count = int(np.ceil(window_count / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 3.4 * row_count),
        squeeze=False,
    )
    flat_axes = list(axes.flatten())

    if hits.empty:
        axis = flat_axes[0]
        axis.text(0.5, 0.5, "No pattern detections in focus window", ha="center", va="center")
        axis.axis("off")
    else:
        for axis, (hit_index, hit) in zip(flat_axes, hits.iterrows()):
            start = max(0, hit_index - bars_before)
            end = min(len(ordered), hit_index + bars_after + 1)
            frame = ordered.iloc[start:end].copy()
            x_values = range(len(frame))
            signal_position = int(hit_index - start)
            is_buy = float(hit["target_weight"]) > 0.0
            decision_color = "tab:green" if is_buy else "tab:red"
            decision_label = "BUY next bar" if is_buy else "SELL/CASH next bar"
            axis.plot(
                x_values,
                frame["close"],
                marker="o",
                color=decision_color,
                linewidth=1.5,
            )
            axis.axvline(signal_position, color=decision_color, linestyle="--", linewidth=1.0)
            axis.scatter(
                [signal_position],
                [float(hit["close"])],
                marker="^" if is_buy else "v",
                color=decision_color,
                zorder=3,
            )
            axis.set_title(
                f"{pd.Timestamp(hit['timestamp']).strftime('%H:%M')} {decision_label} - "
                f"{_active_pattern_label(hit)}",
                fontsize=9,
            )
            axis.set_xticks(list(x_values))
            axis.set_xticklabels(
                [pd.Timestamp(value).strftime("%H:%M") for value in frame["timestamp"]],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            axis.grid(alpha=0.3)

    for axis in flat_axes[window_count:]:
        axis.axis("off")

    figure.suptitle("Pattern Detection Windows", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
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
