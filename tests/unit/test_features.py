from __future__ import annotations

from pathlib import Path

from marketlab.data.panel import load_panel_csv
from marketlab.features.engineering import add_feature_set


def test_feature_history_is_not_changed_by_future_outlier() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "market_panel.csv"
    baseline = load_panel_csv(fixture_path)
    modified = baseline.copy()
    last_row = modified.index[-1]
    modified.loc[last_row, "adj_close"] = modified.loc[last_row, "adj_close"] * 10

    baseline_features = add_feature_set(baseline, [2], [2, 3], [2], 2)
    modified_features = add_feature_set(modified, [2], [2, 3], [2], 2)

    compare_columns = ["return_2", "ma_2", "ma_3", "vol_2", "momentum"]
    for column in compare_columns:
        assert baseline_features.iloc[:-1][column].equals(modified_features.iloc[:-1][column])
