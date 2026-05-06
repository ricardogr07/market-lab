from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_synthetic_pattern_gallery(gallery: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pattern_names = gallery["pattern"].drop_duplicates().tolist()
    figure, axes = plt.subplots(5, 4, figsize=(15, 13), squeeze=False)

    for axis, pattern_name in zip(axes.flatten(), pattern_names):
        frame = gallery.loc[gallery["pattern"] == pattern_name].sort_values("bar")
        color = "tab:green" if bool(frame["implemented"].iloc[0]) else "tab:gray"
        axis.plot(frame["bar"], frame["close"], marker="o", color=color, linewidth=1.8)
        axis.set_title(pattern_name.replace("_", " "), fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(alpha=0.2)
        for spine in axis.spines.values():
            spine.set_alpha(0.3)

    for axis in axes.flatten()[len(pattern_names):]:
        axis.axis("off")

    figure.suptitle("Synthetic Chart-Pattern Gallery", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path)
    plt.close(figure)
    return output_path
