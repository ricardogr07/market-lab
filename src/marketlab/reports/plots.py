from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


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
