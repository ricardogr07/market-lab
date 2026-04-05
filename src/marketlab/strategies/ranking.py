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
VALID_MODES = {"long_only", "long_short"}


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


def _validate_mode(mode: str) -> str:
    if mode not in VALID_MODES:
        joined = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"Ranking mode must be one of: {joined}.")
    return mode


def _validate_threshold(min_score_threshold: float) -> float:
    if not 0.0 <= min_score_threshold <= 1.0:
        raise ValueError("Ranking min_score_threshold must be between 0.0 and 1.0.")
    return float(min_score_threshold)


def _validate_cap(label: str, value: float | None) -> float | None:
    if value is None:
        return None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be between 0.0 and 1.0.")
    return float(value)


def _strategy_value_token(value: float) -> str:
    value_text = f"{value:.6f}".rstrip("0")
    if value_text.endswith("."):
        value_text += "0"
    if "." in value_text and len(value_text.split(".", maxsplit=1)[1]) == 1:
        value_text += "0"
    if "." not in value_text:
        value_text += ".00"
    return value_text.replace(".", "p")


def _strategy_name(
    model_name: str,
    mode: str,
    min_score_threshold: float,
    cash_when_underfilled: bool,
    max_position_weight: float | None,
    max_group_weight: float | None,
    max_long_exposure: float | None,
    max_short_exposure: float | None,
) -> str:
    base_name = f"ml_{model_name}"
    if mode == "long_short" and min_score_threshold == 0.0 and not cash_when_underfilled:
        strategy_name = base_name
    else:
        parts = [base_name, mode]
        if min_score_threshold > 0.0:
            parts.append(f"thr{_strategy_value_token(min_score_threshold)}")
        if cash_when_underfilled:
            parts.append("cash")
        strategy_name = "__".join(parts)

    cap_parts: list[str] = []
    if max_position_weight is not None:
        cap_parts.append(f"poscap{_strategy_value_token(max_position_weight)}")
    if max_group_weight is not None:
        cap_parts.append(f"groupcap{_strategy_value_token(max_group_weight)}")
    if max_long_exposure is not None:
        cap_parts.append(f"longcap{_strategy_value_token(max_long_exposure)}")
    if max_short_exposure is not None:
        cap_parts.append(f"shortcap{_strategy_value_token(max_short_exposure)}")

    if not cap_parts:
        return strategy_name
    return "__".join([strategy_name, *cap_parts])


def _weights_to_frame(
    strategy_name: str,
    effective_date: pd.Timestamp,
    symbols: list[str],
    weights: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": strategy_name,
            "effective_date": effective_date,
            "symbol": symbols,
            "weight": [weights[symbol] for symbol in symbols],
        }
    )


def _zero_weight_frame(
    strategy_name: str,
    effective_date: pd.Timestamp,
    symbols: list[str],
) -> pd.DataFrame:
    return _weights_to_frame(
        strategy_name,
        pd.Timestamp(effective_date),
        symbols,
        {symbol: 0.0 for symbol in symbols},
    )


def _clip_position_weights(
    weights: dict[str, float],
    max_position_weight: float | None,
) -> dict[str, float]:
    if max_position_weight is None:
        return dict(weights)

    clipped: dict[str, float] = {}
    for symbol, weight in weights.items():
        if weight > 0.0:
            clipped[symbol] = min(weight, max_position_weight)
        elif weight < 0.0:
            clipped[symbol] = -min(abs(weight), max_position_weight)
        else:
            clipped[symbol] = 0.0
    return clipped


def _clip_group_weights(
    weights: dict[str, float],
    symbol_groups: dict[str, str],
    max_group_weight: float | None,
) -> dict[str, float]:
    if max_group_weight is None:
        return dict(weights)

    clipped = dict(weights)
    for side in (1.0, -1.0):
        group_members: dict[str, list[str]] = {}
        for symbol, weight in clipped.items():
            if side > 0.0 and weight <= 0.0:
                continue
            if side < 0.0 and weight >= 0.0:
                continue
            group_name = symbol_groups[symbol]
            group_members.setdefault(group_name, []).append(symbol)

        for symbols_in_group in group_members.values():
            group_total = sum(abs(clipped[symbol]) for symbol in symbols_in_group)
            if group_total <= max_group_weight or group_total == 0.0:
                continue
            scale = max_group_weight / group_total
            for symbol in symbols_in_group:
                clipped[symbol] *= scale
    return clipped


def _clip_side_exposure(
    weights: dict[str, float],
    *,
    positive_side: bool,
    cap: float | None,
) -> dict[str, float]:
    if cap is None:
        return dict(weights)

    clipped = dict(weights)
    if positive_side:
        side_total = sum(weight for weight in clipped.values() if weight > 0.0)
        if side_total <= cap or side_total == 0.0:
            return clipped
        scale = cap / side_total
        for symbol, weight in clipped.items():
            if weight > 0.0:
                clipped[symbol] = weight * scale
        return clipped

    side_total = sum(abs(weight) for weight in clipped.values() if weight < 0.0)
    if side_total <= cap or side_total == 0.0:
        return clipped
    scale = cap / side_total
    for symbol, weight in clipped.items():
        if weight < 0.0:
            clipped[symbol] = weight * scale
    return clipped


def _validated_symbol_groups(
    symbols: list[str],
    symbol_groups: dict[str, str] | None,
    max_group_weight: float | None,
) -> dict[str, str]:
    resolved_symbol_groups = symbol_groups or {}
    if max_group_weight is None:
        return resolved_symbol_groups

    missing_group_symbols = sorted(set(symbols) - set(resolved_symbol_groups))
    if missing_group_symbols:
        joined = ", ".join(missing_group_symbols)
        raise ValueError(
            "Ranking max_group_weight requires symbol_groups for all symbols: "
            f"{joined}"
        )
    return {symbol: resolved_symbol_groups[symbol] for symbol in symbols}


def _apply_risk_caps(
    weights: dict[str, float],
    *,
    symbols: list[str],
    mode: str,
    symbol_groups: dict[str, str] | None,
    max_position_weight: float | None,
    max_group_weight: float | None,
    max_long_exposure: float | None,
    max_short_exposure: float | None,
) -> dict[str, float]:
    if mode == "long_only" and max_short_exposure is not None:
        raise ValueError("Ranking max_short_exposure is not allowed in long_only mode.")

    clipped = _clip_position_weights(weights, max_position_weight)
    resolved_symbol_groups = _validated_symbol_groups(symbols, symbol_groups, max_group_weight)
    clipped = _clip_group_weights(clipped, resolved_symbol_groups, max_group_weight)
    clipped = _clip_side_exposure(clipped, positive_side=True, cap=max_long_exposure)
    clipped = _clip_side_exposure(clipped, positive_side=False, cap=max_short_exposure)
    return clipped


def _weight_frame(
    strategy_name: str,
    effective_date: pd.Timestamp,
    symbols: list[str],
    long_symbols: list[str],
    short_symbols: list[str],
    *,
    mode: str,
    long_n: int,
    short_n: int,
    cash_when_underfilled: bool,
    symbol_groups: dict[str, str] | None,
    max_position_weight: float | None,
    max_group_weight: float | None,
    max_long_exposure: float | None,
    max_short_exposure: float | None,
) -> pd.DataFrame:
    if mode == "long_only":
        if len(long_symbols) != long_n and not cash_when_underfilled:
            return _zero_weight_frame(strategy_name, effective_date, symbols)

        weights = {symbol: 0.0 for symbol in symbols}
        long_weight = 1.0 / float(long_n)
        for symbol in long_symbols:
            weights[symbol] = long_weight
        weights = _apply_risk_caps(
            weights,
            symbols=symbols,
            mode=mode,
            symbol_groups=symbol_groups,
            max_position_weight=max_position_weight,
            max_group_weight=max_group_weight,
            max_long_exposure=max_long_exposure,
            max_short_exposure=max_short_exposure,
        )
        return _weights_to_frame(strategy_name, effective_date, symbols, weights)

    if (len(long_symbols) != long_n or len(short_symbols) != short_n) and not cash_when_underfilled:
        return _zero_weight_frame(strategy_name, effective_date, symbols)

    weights = {symbol: 0.0 for symbol in symbols}
    long_weight = 0.5 / float(long_n)
    short_weight = -0.5 / float(short_n)
    for symbol in long_symbols:
        weights[symbol] = long_weight
    for symbol in short_symbols:
        weights[symbol] = short_weight

    weights = _apply_risk_caps(
        weights,
        symbols=symbols,
        mode=mode,
        symbol_groups=symbol_groups,
        max_position_weight=max_position_weight,
        max_group_weight=max_group_weight,
        max_long_exposure=max_long_exposure,
        max_short_exposure=max_short_exposure,
    )
    return _weights_to_frame(strategy_name, effective_date, symbols, weights)


def _rank_signal_rows(
    signal_rows: pd.DataFrame,
    strategy_name: str,
    symbols: list[str],
    long_n: int,
    short_n: int,
    *,
    mode: str,
    min_score_threshold: float,
    cash_when_underfilled: bool,
    symbol_groups: dict[str, str] | None,
    max_position_weight: float | None,
    max_group_weight: float | None,
    max_long_exposure: float | None,
    max_short_exposure: float | None,
) -> pd.DataFrame:
    effective_dates = signal_rows["effective_date"].drop_duplicates().tolist()
    if len(effective_dates) != 1:
        raise ValueError("Ranking predictions must map each signal date to exactly one effective date.")
    effective_date = pd.Timestamp(effective_dates[0])

    long_candidates = signal_rows.loc[signal_rows["score"] >= min_score_threshold].sort_values(
        ["score", "symbol"],
        ascending=[False, True],
    )
    long_symbols = long_candidates["symbol"].head(long_n).tolist()

    if mode == "long_only":
        return _weight_frame(
            strategy_name,
            effective_date,
            symbols,
            long_symbols,
            [],
            mode=mode,
            long_n=long_n,
            short_n=short_n,
            cash_when_underfilled=cash_when_underfilled,
            symbol_groups=symbol_groups,
            max_position_weight=max_position_weight,
            max_group_weight=max_group_weight,
            max_long_exposure=max_long_exposure,
            max_short_exposure=max_short_exposure,
        )

    short_threshold = 1.0 - min_score_threshold
    short_candidates = signal_rows.loc[
        (~signal_rows["symbol"].isin(long_symbols)) & (signal_rows["score"] <= short_threshold)
    ].sort_values(
        ["score", "symbol"],
        ascending=[True, True],
    )
    short_symbols = short_candidates["symbol"].head(short_n).tolist()

    return _weight_frame(
        strategy_name,
        effective_date,
        symbols,
        long_symbols,
        short_symbols,
        mode=mode,
        long_n=long_n,
        short_n=short_n,
        cash_when_underfilled=cash_when_underfilled,
        symbol_groups=symbol_groups,
        max_position_weight=max_position_weight,
        max_group_weight=max_group_weight,
        max_long_exposure=max_long_exposure,
        max_short_exposure=max_short_exposure,
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
    mode: str = "long_short",
    min_score_threshold: float = 0.0,
    cash_when_underfilled: bool = False,
    symbol_groups: dict[str, str] | None = None,
    max_position_weight: float | None = None,
    max_group_weight: float | None = None,
    max_long_exposure: float | None = None,
    max_short_exposure: float | None = None,
) -> pd.DataFrame:
    mode = _validate_mode(mode)
    min_score_threshold = _validate_threshold(min_score_threshold)
    max_position_weight = _validate_cap("Ranking max_position_weight", max_position_weight)
    max_group_weight = _validate_cap("Ranking max_group_weight", max_group_weight)
    max_long_exposure = _validate_cap("Ranking max_long_exposure", max_long_exposure)
    max_short_exposure = _validate_cap("Ranking max_short_exposure", max_short_exposure)
    if long_n < 1:
        raise ValueError("long_n must be at least 1.")
    if mode == "long_short" and short_n < 1:
        raise ValueError("short_n must be at least 1 for long_short mode.")
    if weighting != "equal":
        raise ValueError("Ranking weighting='equal' only in Phase 2.")
    if predictions.empty:
        return pd.DataFrame(columns=WEIGHTS_COLUMNS)

    working_predictions = _validate_predictions(predictions)
    working_panel, symbols = _validate_panel(panel)
    if not symbols:
        return pd.DataFrame(columns=WEIGHTS_COLUMNS)

    strategy_name = _strategy_name(
        model_name=str(working_predictions["model_name"].iat[0]),
        mode=mode,
        min_score_threshold=min_score_threshold,
        cash_when_underfilled=cash_when_underfilled,
        max_position_weight=max_position_weight,
        max_group_weight=max_group_weight,
        max_long_exposure=max_long_exposure,
        max_short_exposure=max_short_exposure,
    )
    weight_frames = [
        _rank_signal_rows(
            signal_rows=signal_rows,
            strategy_name=strategy_name,
            symbols=symbols,
            long_n=long_n,
            short_n=short_n,
            mode=mode,
            min_score_threshold=min_score_threshold,
            cash_when_underfilled=cash_when_underfilled,
            symbol_groups=symbol_groups,
            max_position_weight=max_position_weight,
            max_group_weight=max_group_weight,
            max_long_exposure=max_long_exposure,
            max_short_exposure=max_short_exposure,
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
