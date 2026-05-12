from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from marketlab.rebalance import next_rebalance_effective_date
from marketlab.strategies.ranking import WEIGHTS_COLUMNS

ALLOCATION_TIERS = (0.0, 0.25, 0.50, 1.0)
REQUIRED_PREDICTION_COLUMNS = {
    "model_name",
    "fold_id",
    "signal_date",
    "effective_date",
    "symbol",
    "score",
}
REQUIRED_PANEL_COLUMNS = {"symbol", "timestamp"}


@dataclass(frozen=True, slots=True)
class RegimeParticipationPolicy:
    name: str = "model_only"
    bull_floor: float = 0.0
    sideways_floor: float = 0.0
    bear_floor: float = 0.0
    risk_off_cap: float | None = 0.25


def _validate_thresholds(thresholds: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(thresholds) != 3:
        raise ValueError("Tiered allocation requires three score thresholds.")
    previous = -math.inf
    for threshold in thresholds:
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("Tiered allocation thresholds must be between 0.0 and 1.0.")
        if threshold < previous:
            raise ValueError("Tiered allocation thresholds must be sorted ascending.")
        previous = threshold
    return thresholds


def target_weight_for_score(
    score: float,
    thresholds: tuple[float, float, float],
    *,
    risk_off: bool = False,
) -> float:
    cash_threshold, half_threshold, full_threshold = _validate_thresholds(thresholds)
    if score < cash_threshold:
        target_weight = 0.0
    elif score < half_threshold:
        target_weight = 0.25
    elif score < full_threshold:
        target_weight = 0.50
    else:
        target_weight = 1.0

    if risk_off and target_weight > 0.0:
        return min(target_weight, 0.25)
    return target_weight


def _shift_thresholds(
    thresholds: tuple[float, float, float],
    margin: float,
) -> tuple[float, float, float]:
    return _validate_thresholds(
        (
            max(0.0, min(1.0, thresholds[0] + margin)),
            max(0.0, min(1.0, thresholds[1] + margin)),
            max(0.0, min(1.0, thresholds[2] + margin)),
        )
    )


def _hysteresis_target_weight(
    *,
    score: float,
    thresholds: tuple[float, float, float],
    current_weight: float | None,
    risk_off: bool,
    hysteresis_margin: float,
) -> float:
    base_target = target_weight_for_score(score, thresholds, risk_off=risk_off)
    if current_weight is None or hysteresis_margin == 0.0 or base_target == current_weight:
        return base_target

    if risk_off and base_target < current_weight:
        return base_target

    directional_margin = hysteresis_margin if base_target > current_weight else -hysteresis_margin
    shifted_target = target_weight_for_score(
        score,
        _shift_thresholds(thresholds, directional_margin),
        risk_off=risk_off,
    )
    if base_target > current_weight and shifted_target > current_weight:
        return shifted_target
    if base_target < current_weight and shifted_target < current_weight:
        return shifted_target
    return current_weight


def _direct_hysteresis_target_weight(
    *,
    score: float,
    current_weight: float | None,
    risk_off: bool,
    hysteresis_margin: float,
) -> float:
    base_target = nearest_tier(score)
    if risk_off and base_target > 0.0:
        base_target = min(base_target, 0.25)
    if current_weight is None or hysteresis_margin == 0.0 or base_target == current_weight:
        return base_target

    if risk_off and base_target < current_weight:
        return base_target

    boundary = (base_target + current_weight) / 2.0
    if base_target > current_weight and score >= boundary + hysteresis_margin:
        return base_target
    if base_target < current_weight and score <= boundary - hysteresis_margin:
        return base_target
    return current_weight


def nearest_tier(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(ALLOCATION_TIERS, key=lambda tier: (abs(tier - value), tier))


def _regime_label(row: pd.Series) -> str:
    risk_off = bool(int(row.get("crypto_regime_risk_off", 0) or 0))
    if risk_off:
        return "risk_off"
    trend_state = int(row.get("crypto_regime_trend_state", 0) or 0)
    if trend_state > 0:
        return "bull"
    if trend_state < 0:
        return "bear"
    return "sideways"


def _apply_regime_policy(
    target_weight: float,
    row: pd.Series,
    policy: RegimeParticipationPolicy | None,
) -> float:
    if policy is None:
        return target_weight

    regime = _regime_label(row)
    if regime == "risk_off":
        if policy.risk_off_cap is None:
            return target_weight
        return min(target_weight, float(policy.risk_off_cap))
    if regime == "bull":
        return max(target_weight, float(policy.bull_floor))
    if regime == "bear":
        return max(target_weight, float(policy.bear_floor))
    return max(target_weight, float(policy.sideways_floor))


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
        raise ValueError("Tiered allocation requires predictions for exactly one model.")

    duplicate_keys = working.duplicated(subset=["fold_id", "signal_date", "symbol"])
    if duplicate_keys.any():
        raise ValueError(
            "Tiered allocation predictions must contain one row per fold, signal_date, and symbol."
        )
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
            "effective_date": [pd.Timestamp(effective_date)] * len(symbols),
            "symbol": symbols,
            "weight": [0.0] * len(symbols),
        },
        columns=WEIGHTS_COLUMNS,
    )


def _signal_weight_frame(
    *,
    strategy_name: str,
    symbols: list[str],
    effective_date: pd.Timestamp,
    target_weight: float,
) -> pd.DataFrame:
    if len(symbols) != 1:
        raise ValueError("Tiered allocation is currently scoped to one BTC symbol.")

    return pd.DataFrame(
        {
            "strategy": [strategy_name],
            "effective_date": [pd.Timestamp(effective_date)],
            "symbol": [symbols[0]],
            "weight": [target_weight],
        },
        columns=WEIGHTS_COLUMNS,
    )


def _target_weight_for_signal_rows(
    signal_rows: pd.DataFrame,
    *,
    thresholds: tuple[float, float, float],
    max_long_exposure: float | None,
    current_weight: float | None,
    hysteresis_margin: float,
    direct_scores: bool,
    regime_policy: RegimeParticipationPolicy | None,
) -> tuple[float, bool]:
    effective_dates = signal_rows["effective_date"].drop_duplicates().tolist()
    if len(effective_dates) != 1:
        raise ValueError("Tiered allocation rows must map each signal date to one effective date.")

    row = signal_rows.sort_values(["score", "symbol"], ascending=[False, True]).iloc[0]
    risk_off = bool(int(row.get("crypto_regime_risk_off", 0) or 0))
    if direct_scores:
        target_weight = _direct_hysteresis_target_weight(
            score=float(row["score"]),
            current_weight=current_weight,
            risk_off=risk_off,
            hysteresis_margin=hysteresis_margin,
        )
    else:
        target_weight = _hysteresis_target_weight(
            score=float(row["score"]),
            thresholds=thresholds,
            current_weight=current_weight,
            risk_off=risk_off,
            hysteresis_margin=hysteresis_margin,
        )
    target_weight = _apply_regime_policy(target_weight, row, regime_policy)
    if max_long_exposure is not None:
        target_weight = min(target_weight, float(max_long_exposure))
    return target_weight, risk_off


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
    thresholds: tuple[float, float, float],
    *,
    frequency: str = "W-FRI",
    strategy_name: str | None = None,
    max_long_exposure: float | None = None,
    min_holding_period_bars: int = 0,
    hysteresis_margin: float = 0.0,
    direct_scores: bool = False,
    regime_policy: RegimeParticipationPolicy | None = None,
) -> pd.DataFrame:
    thresholds = _validate_thresholds(thresholds)
    if min_holding_period_bars < 0:
        raise ValueError("Tiered allocation min_holding_period_bars must be non-negative.")
    if (
        not math.isfinite(hysteresis_margin)
        or hysteresis_margin < 0.0
        or hysteresis_margin > 0.25
    ):
        raise ValueError("Tiered allocation hysteresis_margin must be between 0.0 and 0.25.")
    if predictions.empty:
        return pd.DataFrame(columns=WEIGHTS_COLUMNS)

    working_predictions = _validate_predictions(predictions)
    working_panel, symbols = _validate_panel(panel)
    if not symbols:
        return pd.DataFrame(columns=WEIGHTS_COLUMNS)

    resolved_strategy_name = strategy_name or (
        f"ml_{working_predictions['model_name'].iat[0]}__tiered_alloc"
    )
    current_weight: float | None = None
    last_change_bar = 0
    weight_frames: list[pd.DataFrame] = []
    signal_groups = working_predictions.groupby(
        ["signal_date", "effective_date"],
        sort=True,
    )
    for signal_bar, ((_, effective_date), signal_rows) in enumerate(signal_groups):
        desired_weight, risk_off = _target_weight_for_signal_rows(
            signal_rows,
            thresholds=thresholds,
            max_long_exposure=max_long_exposure,
            current_weight=current_weight,
            hysteresis_margin=hysteresis_margin,
            direct_scores=direct_scores,
            regime_policy=regime_policy,
        )
        if current_weight is None:
            target_weight = desired_weight
            last_change_bar = signal_bar
        elif desired_weight == current_weight:
            target_weight = current_weight
        else:
            bars_since_change = signal_bar - last_change_bar
            can_de_risk = risk_off and desired_weight < current_weight
            can_change = can_de_risk or bars_since_change >= min_holding_period_bars
            target_weight = desired_weight if can_change else current_weight
            if target_weight != current_weight:
                last_change_bar = signal_bar
        current_weight = target_weight
        weight_frames.append(
            _signal_weight_frame(
                strategy_name=resolved_strategy_name,
                symbols=symbols,
                effective_date=pd.Timestamp(effective_date),
                target_weight=target_weight,
            )
        )
    weight_frames.extend(
        _flatten_boundary_rows(
            predictions=working_predictions,
            panel=working_panel,
            strategy_name=resolved_strategy_name,
            symbols=symbols,
            frequency=frequency,
        )
    )
    return pd.concat(weight_frames, ignore_index=True).sort_values(
        ["effective_date", "symbol"]
    ).reset_index(drop=True)
