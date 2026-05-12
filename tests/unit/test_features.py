from __future__ import annotations

from pathlib import Path

from marketlab.config import ExperimentConfig
from marketlab.data.panel import load_panel_csv
from marketlab.features.engineering import add_feature_set
from marketlab.targets import build_modeling_dataset


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


def test_crypto_time_series_features_are_trailing_only() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "crypto_hourly_panel.csv"
    baseline = load_panel_csv(fixture_path)
    modified = baseline.copy()
    modified.loc[modified.index[-1], "adj_close"] *= 3.0
    modified.loc[modified.index[-1], "volume"] *= 5.0

    kwargs = {
        "return_windows": [1, 3],
        "ma_windows": [3],
        "vol_windows": [3],
        "momentum_window": 3,
        "crypto_time_series_enabled": True,
        "crypto_return_windows": [1, 3],
        "crypto_vol_windows": [3],
        "crypto_ma_windows": [3],
        "crypto_rsi_window": 3,
        "crypto_macd_fast_window": 3,
        "crypto_macd_slow_window": 6,
        "crypto_macd_signal_window": 3,
        "crypto_bollinger_window": 3,
        "crypto_bollinger_std": 2.0,
        "crypto_volume_window": 3,
        "crypto_time_features": True,
    }
    baseline_features = add_feature_set(baseline, **kwargs)
    modified_features = add_feature_set(modified, **kwargs)

    compare_columns = [
        "crypto_return_1",
        "crypto_return_3",
        "crypto_vol_3",
        "crypto_price_to_ma_3",
        "crypto_ma_slope_3",
        "crypto_rsi_3",
        "crypto_macd",
        "crypto_macd_signal",
        "crypto_macd_hist",
        "crypto_bollinger_z_3",
        "crypto_volume_z_3",
        "crypto_hour_sin",
        "crypto_dayofweek_cos",
    ]
    for column in compare_columns:
        assert baseline_features.iloc[:-1][column].equals(modified_features.iloc[:-1][column])


def test_crypto_regime_features_are_trailing_only() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "crypto_hourly_panel.csv"
    baseline = load_panel_csv(fixture_path)
    modified = baseline.copy()
    modified.loc[modified.index[-1], "adj_close"] *= 3.0
    modified.loc[modified.index[-1], "volume"] *= 5.0

    kwargs = {
        "return_windows": [1, 3],
        "ma_windows": [3],
        "vol_windows": [3],
        "momentum_window": 3,
        "crypto_regime_features_enabled": True,
        "crypto_regime_trend_windows": [3, 6],
        "crypto_regime_volatility_window": 3,
        "crypto_regime_percentile_window": 6,
        "crypto_regime_drawdown_window": 6,
        "crypto_regime_volume_window": 3,
    }
    baseline_features = add_feature_set(baseline, **kwargs)
    modified_features = add_feature_set(modified, **kwargs)

    compare_columns = [
        "crypto_regime_return_3",
        "crypto_regime_price_to_ma_6",
        "crypto_regime_realized_vol_3",
        "crypto_regime_vol_percentile_3_6",
        "crypto_regime_drawdown_6",
        "crypto_regime_volume_shock_3",
        "crypto_regime_trend_state",
        "crypto_regime_risk_off",
    ]
    for column in compare_columns:
        assert column in baseline_features.columns
        assert baseline_features.iloc[:-1][column].equals(modified_features.iloc[:-1][column])


def test_indicator_stack_ml_features_are_trailing_only() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "crypto_hourly_panel.csv"
    baseline = load_panel_csv(fixture_path)
    modified = baseline.copy()
    modified.loc[modified.index[-1], "adj_close"] *= 3.0
    modified.loc[modified.index[-1], "volume"] *= 5.0

    config = ExperimentConfig()
    config.features.return_windows = [1, 3]
    config.features.ma_windows = [3]
    config.features.vol_windows = [3]
    config.features.momentum_window = 3
    config.features.indicator_stack_ml_features_enabled = True
    config.target.horizon_days = 1
    config.portfolio.ranking.rebalance_frequency = "bar"
    config.baselines.indicator_stack.ema_fast_window = 3
    config.baselines.indicator_stack.ema_slow_window = 6
    config.baselines.indicator_stack.rsi_window = 3
    config.baselines.indicator_stack.macd_fast_window = 3
    config.baselines.indicator_stack.macd_slow_window = 6
    config.baselines.indicator_stack.macd_signal_window = 3
    config.baselines.indicator_stack.bollinger_window = 3
    config.baselines.indicator_stack.volume_window = 3

    baseline_dataset = build_modeling_dataset(baseline, config)
    modified_dataset = build_modeling_dataset(modified, config)

    compare_columns = [
        "indicator_ema_spread",
        "indicator_rsi",
        "indicator_macd_hist",
        "indicator_bollinger_z",
        "indicator_bollinger_side",
        "indicator_volume_ratio",
        "indicator_volume_z",
        "indicator_ema_confirmed",
        "indicator_confirmation_count",
    ]
    for column in compare_columns:
        assert baseline_dataset.iloc[:-1][column].equals(modified_dataset.iloc[:-1][column])
