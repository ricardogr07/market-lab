from __future__ import annotations

import pandas as pd

from marketlab.rebalance import next_rebalance_effective_date

REQUIRED_PREDICTION_COLUMNS = {
    "model_name",
    "fold_id",
    "signal_date",
    "effective_date",
    "symbol",
    "score",
}
REQUIRED_PANEL_COLUMNS = {"symbol", "timestamp"}
WEIGHTS_COLUMNS = ["strategy", "effective_date", "symbol", "weight"]


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Prediction frame is missing required columns: {joined}")

    working = predictions.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce")
    working["effective_date"] = pd.to_datetime(working["effective_date"], errors="coerce")
    if working["signal_date"].isna().any() or working["effective_date"].isna().any():
        raise ValueError("Prediction rows contain invalid signal_date or effective_date values.")

    model_names = working["model_name"].drop_duplicates().tolist()
    if len(model_names) != 1:
        raise ValueError("Ranking weights require predictions for exactly one model.")

    duplicate_keys = working.duplicated(subset=["fold_id", "signal_date", "symbol"])
    if duplicate_keys.any():
        raise ValueError("Ranking predictions must contain one row per fold, signal_date, and symbol.")

    return working.sort_values(["signal_date", "symbol"]).reset_index(drop=True)


def _validate_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing = REQUIRED_PANEL_COLUMNS - set(panel.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Panel is missing required columns: {joined}")

    working = panel.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    if working["timestamp"].isna().any():
        raise ValueError("Panel contains invalid timestamp values.")

    return working, sorted(working["symbol"].drop_duplicates().tolist())


def _zero_weight_frame(
    strategy_name: str,
    effective_date: pd.Timestamp,
    symbols: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": strategy_name,
            "effective_date": pd.Timestamp(effective_date),
            "symbol": symbols,
            "weight": [0.0] * len(symbols),
        }
    )


def _rank_signal_rows(
    signal_rows: pd.DataFrame,
    strategy_name: str,
    symbols: list[str],
    long_n: int,
    short_n: int,
) -> pd.DataFrame:
    effective_dates = signal_rows["effective_date"].drop_duplicates().tolist()
    if len(effective_dates) != 1:
        raise ValueError("Ranking predictions must map each signal date to exactly one effective date.")
    effective_date = pd.Timestamp(effective_dates[0])

    long_candidates = signal_rows.sort_values(["score", "symbol"], ascending=[False, True])
    long_symbols = long_candidates["symbol"].head(long_n).tolist()

    short_candidates = signal_rows.loc[~signal_rows["symbol"].isin(long_symbols)].sort_values(
        ["score", "symbol"],
        ascending=[True, True],
    )
    short_symbols = short_candidates["symbol"].head(short_n).tolist()

    if len(long_symbols) != long_n or len(short_symbols) != short_n:
        return _zero_weight_frame(strategy_name, effective_date, symbols)

    weights = {symbol: 0.0 for symbol in symbols}
    long_weight = 0.5 / float(long_n)
    short_weight = -0.5 / float(short_n)
    for symbol in long_symbols:
        weights[symbol] = long_weight
    for symbol in short_symbols:
        weights[symbol] = short_weight

    return pd.DataFrame(
        {
            "strategy": strategy_name,
            "effective_date": effective_date,
            "symbol": symbols,
            "weight": [weights[symbol] for symbol in symbols],
        }
    )


def _flatten_boundary_rows(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    strategy_name: str,
    symbols: list[str],
    frequency: str,
) -> list[pd.DataFrame]:
    actual_effective_dates = set(pd.to_datetime(predictions["effective_date"]).tolist())
    added_boundaries: set[pd.Timestamp] = set()
    boundary_frames: list[pd.DataFrame] = []

    for _, fold_rows in predictions.groupby("fold_id", sort=True):
        boundary_effective_date = next_rebalance_effective_date(
            panel,
            signal_date=pd.Timestamp(fold_rows["signal_date"].max()),
            frequency=frequency,
        )
        if boundary_effective_date is None:
            continue
        boundary_effective_date = pd.Timestamp(boundary_effective_date)
        if boundary_effective_date in actual_effective_dates or boundary_effective_date in added_boundaries:
            continue
        boundary_frames.append(_zero_weight_frame(strategy_name, boundary_effective_date, symbols))
        added_boundaries.add(boundary_effective_date)

    return boundary_frames


def generate_weights(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    long_n: int,
    short_n: int,
    frequency: str = "W-FRI",
    weighting: str = "equal",
) -> pd.DataFrame:
    if long_n < 1 or short_n < 1:
        raise ValueError("long_n and short_n must both be at least 1.")
    if weighting != "equal":
        raise ValueError("Ranking weighting='equal' only in Phase 2.")
    if predictions.empty:
        return pd.DataFrame(columns=WEIGHTS_COLUMNS)

    working_predictions = _validate_predictions(predictions)
    working_panel, symbols = _validate_panel(panel)
    if not symbols:
        return pd.DataFrame(columns=WEIGHTS_COLUMNS)

    strategy_name = f"ml_{working_predictions['model_name'].iat[0]}"
    weight_frames = [
        _rank_signal_rows(
            signal_rows=signal_rows,
            strategy_name=strategy_name,
            symbols=symbols,
            long_n=long_n,
            short_n=short_n,
        )
        for (_, _), signal_rows in working_predictions.groupby(["signal_date", "effective_date"], sort=True)
    ]
    weight_frames.extend(
        _flatten_boundary_rows(
            predictions=working_predictions,
            panel=working_panel,
            strategy_name=strategy_name,
            symbols=symbols,
            frequency=frequency,
        )
    )

    return pd.concat(weight_frames, ignore_index=True).sort_values(
        ["effective_date", "symbol"]
    ).reset_index(drop=True)
