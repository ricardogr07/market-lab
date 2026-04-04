from __future__ import annotations

import pandas as pd

from marketlab.rebalance import signal_effective_dates

ALLOCATION_STRATEGY_NAMES = {
    "equal": "allocation_equal",
    "group_weights": "allocation_group_weights",
    "symbol_weights": "allocation_symbol_weights",
}


def _empty_weights_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["strategy", "effective_date", "symbol", "weight"]
    )


def strategy_name_for_mode(mode: str) -> str:
    try:
        return ALLOCATION_STRATEGY_NAMES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported allocation mode: {mode}") from exc


def _target_weights_for_mode(
    *,
    symbols: list[str],
    mode: str,
    symbol_weights: dict[str, float] | None,
    symbol_groups: dict[str, str] | None,
    group_weights: dict[str, float] | None,
) -> dict[str, float]:
    if mode == "equal":
        equal_weight = 1.0 / len(symbols)
        return {symbol: equal_weight for symbol in symbols}

    if mode == "symbol_weights":
        if symbol_weights is None:
            raise ValueError("symbol_weights are required for symbol_weights mode.")
        return {symbol: float(symbol_weights[symbol]) for symbol in symbols}

    if mode == "group_weights":
        if symbol_groups is None or group_weights is None:
            raise ValueError(
                "symbol_groups and group_weights are required for group_weights mode."
            )

        grouped_symbols: dict[str, list[str]] = {}
        for symbol in symbols:
            try:
                group_name = symbol_groups[symbol]
            except KeyError as exc:
                raise ValueError(
                    f"Missing symbol group for allocation symbol '{symbol}'."
                ) from exc
            grouped_symbols.setdefault(group_name, []).append(symbol)

        target_weights: dict[str, float] = {}
        for group_name, members in grouped_symbols.items():
            group_weight = float(group_weights[group_name])
            member_weight = group_weight / len(members)
            for symbol in sorted(members):
                target_weights[symbol] = member_weight
        return target_weights

    raise ValueError(f"Unsupported allocation mode: {mode}")


def generate_weights(
    panel: pd.DataFrame,
    *,
    frequency: str = "W-FRI",
    mode: str = "equal",
    symbol_weights: dict[str, float] | None = None,
    symbol_groups: dict[str, str] | None = None,
    group_weights: dict[str, float] | None = None,
    strategy_name: str | None = None,
) -> pd.DataFrame:
    if panel.empty:
        return _empty_weights_frame()

    symbols = sorted(panel["symbol"].unique())
    effective_dates = {
        pd.Timestamp(panel["timestamp"].min()),
        *[pd.Timestamp(date) for date in signal_effective_dates(panel, frequency).tolist()],
    }
    target_weights = _target_weights_for_mode(
        symbols=symbols,
        mode=mode,
        symbol_weights=symbol_weights,
        symbol_groups=symbol_groups,
        group_weights=group_weights,
    )
    resolved_strategy_name = strategy_name or strategy_name_for_mode(mode)

    rows: list[dict[str, object]] = []
    for effective_date in sorted(effective_dates):
        for symbol in symbols:
            rows.append(
                {
                    "strategy": resolved_strategy_name,
                    "effective_date": effective_date,
                    "symbol": symbol,
                    "weight": target_weights[symbol],
                }
            )

    return pd.DataFrame(rows)
