from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

ALLOCATION_TIERS = (0.0, 0.25, 0.50, 1.0)
STRATEGY_NAME = "ml_indicator_tuned__long_only__cash"
BENCHMARK_STRATEGY_NAME = "buy_hold"

PHASE8_BULL_PARTICIPATION_COLUMNS = [
    "section",
    "metric",
    "group",
    "subgroup",
    "value",
    "row_count",
    "detail",
]
PHASE8_BULL_PARTICIPATION_SUMMARY_COLUMNS = [
    "section",
    "metric",
    "value",
    "detail",
]


def _read_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _numeric(value: Any) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric_value if math.isfinite(numeric_value) else float("nan")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _append_detail(
    rows: list[dict[str, object]],
    *,
    section: str,
    metric: str,
    value: object,
    group: object = "",
    subgroup: object = "",
    row_count: object = pd.NA,
    detail: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "metric": metric,
            "group": group,
            "subgroup": subgroup,
            "value": value,
            "row_count": row_count,
            "detail": detail,
        }
    )


def _append_summary(
    rows: list[dict[str, object]],
    *,
    section: str,
    metric: str,
    value: object,
    detail: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "metric": metric,
            "value": value,
            "detail": detail,
        }
    )


def _tier_label(value: float) -> str:
    return f"{int(round(value * 100))}"


def _runtime_regime_for_values(risk_off: Any, trend_state: Any) -> str:
    risk_off_value = _numeric(risk_off)
    trend_state_value = _numeric(trend_state)
    if math.isfinite(risk_off_value) and int(risk_off_value) == 1:
        return "risk_off"
    if math.isfinite(trend_state_value) and int(trend_state_value) > 0:
        return "bull"
    if math.isfinite(trend_state_value) and int(trend_state_value) < 0:
        return "bear"
    return "sideways"


def _date_regime_from_probability(probability: pd.DataFrame) -> pd.DataFrame:
    required = {"runtime_regime", "crypto_regime_risk_off", "crypto_regime_trend_state"}
    if probability.empty or "runtime_regime" not in probability.columns:
        return pd.DataFrame()
    date_column = "effective_date" if "effective_date" in probability.columns else "signal_date"
    if date_column not in probability.columns:
        return pd.DataFrame()

    columns = [
        date_column,
        *[column for column in required if column in probability.columns],
        *[column for column in probability.columns if str(column).startswith("gate_")],
    ]
    date_regime = probability.loc[:, columns].copy()
    date_regime = date_regime.rename(columns={date_column: "date"})
    date_regime["date"] = pd.to_datetime(date_regime["date"], errors="coerce")
    date_regime = date_regime.dropna(subset=["date"])
    return date_regime.drop_duplicates("date", keep="last").reset_index(drop=True)


def _last_percentile(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.rank(pct=True).iloc[-1])


def _gate_regime_masks(panel: pd.DataFrame, config: Any) -> dict[str, pd.Index]:
    if panel.empty:
        return {}
    working = (
        panel.sort_values(["symbol", "timestamp"])
        .drop_duplicates(["timestamp"])
        .assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))
        .reset_index(drop=True)
    )
    close = working["adj_close"].astype(float)
    returns = close.pct_change()
    trend_window = max(config.features.crypto_regime_trend_windows or [180])
    vol_window = config.features.crypto_regime_volatility_window
    pct_window = config.features.crypto_regime_percentile_window
    drawdown_window = config.features.crypto_regime_drawdown_window

    moving_average = close.rolling(trend_window, min_periods=trend_window).mean()
    trend_return = close.pct_change(trend_window)
    rolling_high = close.rolling(drawdown_window, min_periods=drawdown_window).max()
    drawdown = (close / rolling_high.replace(0.0, pd.NA)) - 1.0
    realized_vol = returns.rolling(vol_window, min_periods=vol_window).std(ddof=0)
    vol_percentile = realized_vol.rolling(
        pct_window,
        min_periods=min(vol_window, pct_window),
    ).apply(_last_percentile, raw=False)

    dates = pd.Index(working["timestamp"])
    bull = close.ge(moving_average) & trend_return.gt(0.0)
    bear = close.lt(moving_average) & (trend_return.lt(0.0) | drawdown.le(-0.20))
    high_volatility = vol_percentile.ge(0.80)
    sideways = ~(bull.fillna(False) | bear.fillna(False))
    last_date = pd.Timestamp(dates.max())
    recent_start = last_date - pd.DateOffset(
        months=config.evaluation.strict_research_gate.recent_window_months
    )
    return {
        "gate_bull": dates[bull.fillna(False).to_numpy()],
        "gate_bear": dates[bear.fillna(False).to_numpy()],
        "gate_sideways": dates[sideways.fillna(False).to_numpy()],
        "gate_high_volatility": dates[high_volatility.fillna(False).to_numpy()],
        "gate_recent": dates[dates >= recent_start],
    }


def _date_regime_from_config(config_path: str | Path | None) -> pd.DataFrame:
    if config_path is None:
        return pd.DataFrame()

    from marketlab.config import load_config
    from marketlab.features.engineering import add_feature_set

    config = load_config(config_path)
    panel_path = config.prepared_panel_path
    if not panel_path.exists():
        return pd.DataFrame()

    panel = pd.read_csv(panel_path)
    if panel.empty or not {"symbol", "timestamp", "adj_close"}.issubset(panel.columns):
        return pd.DataFrame()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce")
    panel = panel.dropna(subset=["timestamp"]).copy()

    feature_options = asdict(config.features)
    feature_options.pop("indicator_stack_ml_features_enabled", None)
    featured = add_feature_set(panel=panel, **feature_options)
    required = {"timestamp", "crypto_regime_risk_off", "crypto_regime_trend_state"}
    if not required.issubset(featured.columns):
        return pd.DataFrame()

    date_regime = (
        featured.drop_duplicates("timestamp")
        .loc[:, ["timestamp", "crypto_regime_risk_off", "crypto_regime_trend_state"]]
        .rename(columns={"timestamp": "date"})
        .copy()
    )
    date_regime["runtime_regime"] = date_regime.apply(
        lambda row: _runtime_regime_for_values(
            row["crypto_regime_risk_off"],
            row["crypto_regime_trend_state"],
        ),
        axis=1,
    )
    for gate_name, dates in _gate_regime_masks(panel, config).items():
        date_regime[gate_name] = date_regime["date"].isin(pd.to_datetime(list(dates)))
    return date_regime.reset_index(drop=True)


def _resolve_date_regime(
    *,
    probability: pd.DataFrame,
    config_path: str | Path | None,
) -> tuple[pd.DataFrame, str]:
    config_regime = _date_regime_from_config(config_path)
    if not config_regime.empty:
        return config_regime, "config"
    probability_regime = _date_regime_from_probability(probability)
    if not probability_regime.empty:
        return probability_regime, "allocation_probability_diagnostics"
    return pd.DataFrame(), "missing"


def _target_prediction_rows(
    *,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    probability: pd.DataFrame,
) -> None:
    if probability.empty:
        _append_summary(
            summary_rows,
            section="target_prediction",
            metric="allocation_probability_present",
            value=False,
            detail="allocation_probability_diagnostics.csv was not found",
        )
        return

    _append_summary(
        summary_rows,
        section="target_prediction",
        metric="allocation_probability_present",
        value=True,
        detail=f"rows={len(probability)}",
    )

    total_rows = max(1, len(probability))
    if "target_weight" in probability.columns:
        target_weights = probability["target_weight"].map(_numeric)
        for tier in ALLOCATION_TIERS:
            count = int(target_weights.sub(tier).abs().le(1e-9).sum())
            fraction = count / total_rows
            _append_detail(
                detail_rows,
                section="target_prediction",
                metric="target_tier_fraction",
                group=_tier_label(tier),
                value=fraction,
                row_count=count,
            )
            _append_summary(
                summary_rows,
                section="target_prediction",
                metric=f"target_tier_{_tier_label(tier)}_fraction",
                value=fraction,
                detail="actual utility target support in selected OOS prediction rows",
            )

    if "predicted_tier_weight" in probability.columns:
        predicted_tiers = probability["predicted_tier_weight"].map(_numeric)
        for tier in ALLOCATION_TIERS:
            count = int(predicted_tiers.sub(tier).abs().le(1e-9).sum())
            fraction = count / total_rows
            _append_detail(
                detail_rows,
                section="target_prediction",
                metric="predicted_tier_fraction",
                group=_tier_label(tier),
                value=fraction,
                row_count=count,
            )
            _append_summary(
                summary_rows,
                section="target_prediction",
                metric=f"predicted_tier_{_tier_label(tier)}_fraction",
                value=fraction,
                detail="nearest traded tier support in selected OOS prediction rows",
            )

    if {"target_weight", "predicted_tier_weight"}.issubset(probability.columns):
        working = probability.copy()
        working["target_weight"] = working["target_weight"].map(_numeric)
        working["predicted_tier_weight"] = working["predicted_tier_weight"].map(_numeric)
        counts = (
            working.groupby(["target_weight", "predicted_tier_weight"], dropna=False)
            .size()
            .rename("row_count")
            .reset_index()
        )
        totals = working.groupby("target_weight", dropna=False).size().rename("total")
        counts = counts.merge(totals, on="target_weight", how="left")
        for _, row in counts.iterrows():
            target = _numeric(row["target_weight"])
            predicted = _numeric(row["predicted_tier_weight"])
            count = int(row["row_count"])
            total = int(row["total"])
            _append_detail(
                detail_rows,
                section="target_prediction",
                metric="target_vs_predicted_tier_fraction",
                group=_tier_label(target) if math.isfinite(target) else "missing",
                subgroup=(
                    _tier_label(predicted) if math.isfinite(predicted) else "missing"
                ),
                value=count / total if total else float("nan"),
                row_count=count,
                detail="fraction is normalized within actual target tier",
            )

    for column in ("score", "prob_tier_100"):
        if column not in probability.columns:
            continue
        values = probability[column].map(_numeric).dropna()
        if values.empty:
            continue
        for metric_name, value in {
            f"{column}_mean": float(values.mean()),
            f"{column}_median": float(values.median()),
            f"{column}_max": float(values.max()),
        }.items():
            _append_summary(
                summary_rows,
                section="target_prediction",
                metric=metric_name,
                value=value,
                detail="selected OOS prediction rows",
            )


def _join_date_regime(
    frame: pd.DataFrame,
    *,
    date_column: str,
    date_regime: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty or date_regime.empty or date_column not in frame.columns:
        return frame.copy()
    working = frame.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    regime = date_regime.copy()
    regime["date"] = pd.to_datetime(regime["date"], errors="coerce")
    return working.merge(regime, left_on=date_column, right_on="date", how="left")


def _runtime_participation_rows(
    *,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    daily_exposure: pd.DataFrame,
    date_regime: pd.DataFrame,
    date_regime_source: str,
) -> None:
    if daily_exposure.empty:
        _append_summary(
            summary_rows,
            section="runtime_participation",
            metric="daily_exposure_present",
            value=False,
            detail="daily_exposure.csv was not found",
        )
        return
    if date_regime.empty:
        _append_summary(
            summary_rows,
            section="runtime_participation",
            metric="runtime_regime_present",
            value=False,
            detail="runtime regime labels were not found; pass --config for legacy runs",
        )
        return

    strategy_rows = daily_exposure.loc[
        daily_exposure["strategy"].astype(str).eq(STRATEGY_NAME)
    ].copy()
    joined = _join_date_regime(strategy_rows, date_column="date", date_regime=date_regime)
    joined = joined.dropna(subset=["runtime_regime"])
    if joined.empty:
        _append_summary(
            summary_rows,
            section="runtime_participation",
            metric="runtime_regime_joined_rows",
            value=0,
            detail=f"date regime source={date_regime_source}",
        )
        return

    _append_summary(
        summary_rows,
        section="runtime_participation",
        metric="runtime_regime_source",
        value=date_regime_source,
        detail=f"joined_rows={len(joined)}",
    )
    for regime, group in joined.groupby("runtime_regime", sort=True):
        long_exposure = group["long_exposure"].map(_numeric).dropna()
        cash_weight = group["cash_weight"].map(_numeric).dropna()
        if not long_exposure.empty:
            _append_detail(
                detail_rows,
                section="runtime_participation",
                metric="average_long_exposure",
                group=regime,
                value=float(long_exposure.mean()),
                row_count=len(group),
            )
            _append_detail(
                detail_rows,
                section="runtime_participation",
                metric="median_long_exposure",
                group=regime,
                value=float(long_exposure.median()),
                row_count=len(group),
            )
            _append_summary(
                summary_rows,
                section="runtime_participation",
                metric=f"{regime}_average_long_exposure",
                value=float(long_exposure.mean()),
                detail=f"rows={len(group)}",
            )
        if not cash_weight.empty:
            _append_detail(
                detail_rows,
                section="runtime_participation",
                metric="average_cash_weight",
                group=regime,
                value=float(cash_weight.mean()),
                row_count=len(group),
            )

    if "gate_bull" in joined.columns:
        gate_bull = joined.loc[joined["gate_bull"].map(_truthy)].copy()
        if not gate_bull.empty:
            average_exposure = gate_bull["long_exposure"].map(_numeric).dropna().mean()
            _append_summary(
                summary_rows,
                section="runtime_participation",
                metric="gate_bull_average_long_exposure",
                value=float(average_exposure),
                detail=f"rows={len(gate_bull)}",
            )


def _gate_runtime_rows(
    *,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    date_regime: pd.DataFrame,
    run_dates: pd.Index,
) -> None:
    if date_regime.empty or "gate_bull" not in date_regime.columns:
        _append_summary(
            summary_rows,
            section="gate_runtime_overlap",
            metric="gate_bull_overlap_present",
            value=False,
            detail="gate bull labels require --config for legacy artifacts",
        )
        return

    working_regime = date_regime.copy()
    if len(run_dates) > 0:
        working_regime = working_regime.loc[
            pd.to_datetime(working_regime["date"]).isin(pd.to_datetime(run_dates))
        ].copy()

    gate_bull = working_regime.loc[working_regime["gate_bull"].map(_truthy)].copy()
    total = len(gate_bull)
    _append_summary(
        summary_rows,
        section="gate_runtime_overlap",
        metric="gate_bull_days",
        value=int(total),
        detail="strict-gate bull slice days in the run window with runtime regime labels",
    )
    if total == 0:
        return

    counts = gate_bull["runtime_regime"].fillna("missing").value_counts(sort=False)
    for regime, count in counts.sort_index().items():
        fraction = int(count) / total
        _append_detail(
            detail_rows,
            section="gate_runtime_overlap",
            metric="gate_bull_runtime_regime_fraction",
            group=regime,
            value=fraction,
            row_count=int(count),
        )
        _append_summary(
            summary_rows,
            section="gate_runtime_overlap",
            metric=f"gate_bull_runtime_{regime}_fraction",
            value=fraction,
            detail=f"days={int(count)}",
        )


def _bull_active_return_rows(
    *,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    benchmark_relative: pd.DataFrame,
    daily_exposure: pd.DataFrame,
    date_regime: pd.DataFrame,
) -> None:
    if benchmark_relative.empty:
        _append_summary(
            summary_rows,
            section="bull_active_return",
            metric="benchmark_relative_present",
            value=False,
            detail="benchmark_relative.csv was not found",
        )
        return
    if date_regime.empty or "gate_bull" not in date_regime.columns:
        _append_summary(
            summary_rows,
            section="bull_active_return",
            metric="gate_bull_active_return_present",
            value=False,
            detail="gate bull active return attribution requires --config for legacy artifacts",
        )
        return

    relative_rows = benchmark_relative.loc[
        benchmark_relative["strategy"].astype(str).eq(STRATEGY_NAME)
        & benchmark_relative["benchmark_strategy"].astype(str).eq(BENCHMARK_STRATEGY_NAME)
    ].copy()
    if relative_rows.empty:
        _append_summary(
            summary_rows,
            section="bull_active_return",
            metric="buy_hold_relative_rows",
            value=0,
            detail="strategy-vs-buy_hold rows were not found",
        )
        return
    exposure_rows = daily_exposure.loc[
        daily_exposure["strategy"].astype(str).eq(STRATEGY_NAME)
    ].copy()
    if not exposure_rows.empty:
        exposure_rows["date"] = pd.to_datetime(exposure_rows["date"], errors="coerce")
        exposure_rows = exposure_rows.loc[:, ["date", "long_exposure", "cash_weight"]]
        relative_rows["date"] = pd.to_datetime(relative_rows["date"], errors="coerce")
        relative_rows = relative_rows.merge(exposure_rows, on="date", how="left")

    joined = _join_date_regime(relative_rows, date_column="date", date_regime=date_regime)
    gate_bull = joined.loc[joined["gate_bull"].map(_truthy)].copy()
    if gate_bull.empty:
        _append_summary(
            summary_rows,
            section="bull_active_return",
            metric="gate_bull_relative_rows",
            value=0,
            detail="no benchmark_relative rows intersect strict-gate bull dates",
        )
        return

    excess = gate_bull["excess_return"].map(_numeric).dropna()
    _append_summary(
        summary_rows,
        section="bull_active_return",
        metric="gate_bull_active_return_sum",
        value=float(excess.sum()) if not excess.empty else float("nan"),
        detail=f"daily excess_return sum; rows={len(gate_bull)}",
    )
    _append_summary(
        summary_rows,
        section="bull_active_return",
        metric="gate_bull_active_return_mean",
        value=float(excess.mean()) if not excess.empty else float("nan"),
        detail="daily excess_return mean",
    )

    for regime, group in gate_bull.groupby("runtime_regime", dropna=False, sort=True):
        regime_name = str(regime)
        regime_excess = group["excess_return"].map(_numeric).dropna()
        if regime_excess.empty:
            continue
        _append_detail(
            detail_rows,
            section="bull_active_return",
            metric="gate_bull_excess_return_sum",
            group=regime_name,
            value=float(regime_excess.sum()),
            row_count=len(group),
        )
        _append_detail(
            detail_rows,
            section="bull_active_return",
            metric="gate_bull_excess_return_mean",
            group=regime_name,
            value=float(regime_excess.mean()),
            row_count=len(group),
        )
        _append_summary(
            summary_rows,
            section="bull_active_return",
            metric=f"gate_bull_runtime_{regime_name}_active_return_sum",
            value=float(regime_excess.sum()),
            detail=f"rows={len(group)}",
        )

    if {"benchmark_net_return", "long_exposure"}.issubset(gate_bull.columns):
        benchmark_return = gate_bull["benchmark_net_return"].map(_numeric)
        exposure = gate_bull["long_exposure"].map(_numeric)
        positive_benchmark = benchmark_return.gt(0.0)
        underexposed_positive = positive_benchmark & exposure.lt(0.999)
        positive_count = int(positive_benchmark.sum())
        underexposed_count = int(underexposed_positive.sum())
        fraction = underexposed_count / positive_count if positive_count else float("nan")
        _append_summary(
            summary_rows,
            section="bull_active_return",
            metric="gate_bull_underexposed_positive_benchmark_fraction",
            value=fraction,
            detail=f"underexposed_positive_days={underexposed_count}; positive_benchmark_days={positive_count}",
        )
        if underexposed_count:
            _append_summary(
                summary_rows,
                section="bull_active_return",
                metric="gate_bull_underexposed_positive_benchmark_return_sum",
                value=float(benchmark_return.loc[underexposed_positive].sum()),
                detail="buy-hold return on positive gate-bull days with long_exposure < 1.0",
            )


def _selection_context_rows(
    *,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    selections: pd.DataFrame,
) -> None:
    if selections.empty:
        _append_summary(
            summary_rows,
            section="selection_context",
            metric="selection_artifact_present",
            value=False,
            detail="ml_strategy_tuning_selections.csv was not found",
        )
        return

    total = len(selections)
    status = selections["selection_status"].astype(str)
    selected_count = int(status.eq("selected").sum())
    no_valid_count = int(status.eq("no_valid_candidate").sum())
    _append_summary(
        summary_rows,
        section="selection_context",
        metric="selected_fold_fraction",
        value=selected_count / total if total else float("nan"),
        detail=f"selected={selected_count}; total={total}",
    )
    _append_summary(
        summary_rows,
        section="selection_context",
        metric="no_valid_candidate_folds",
        value=no_valid_count,
        detail="runtime folds that stayed cash",
    )

    if "selection_source" in selections.columns:
        counts = selections["selection_source"].fillna("missing").value_counts(sort=False)
        for source, count in counts.sort_index().items():
            _append_detail(
                detail_rows,
                section="selection_context",
                metric="selection_source_count",
                group=source,
                value=int(count),
                row_count=int(count),
            )
            _append_summary(
                summary_rows,
                section="selection_context",
                metric=f"selection_source_{source}_folds",
                value=int(count),
                detail="counted from ml_strategy_tuning_selections.csv",
            )

    if "selected_regime_policy" in selections.columns:
        policy_counts = (
            selections.loc[status.eq("selected"), "selected_regime_policy"]
            .dropna()
            .astype(str)
            .value_counts(sort=False)
        )
        for policy_name, count in policy_counts.sort_index().items():
            _append_detail(
                detail_rows,
                section="selection_context",
                metric="selected_regime_policy_count",
                group=policy_name,
                value=int(count),
                row_count=int(count),
            )

    for column in (
        "selected_regime_bull_floor",
        "selected_regime_risk_off_cap",
        "selected_regime_gate_bull_floor",
    ):
        if column not in selections.columns:
            continue
        values = selections.loc[status.eq("selected"), column].map(_numeric).dropna()
        if not values.empty:
            _append_summary(
                summary_rows,
                section="selection_context",
                metric=f"{column}_mode",
                value=float(values.mode().iloc[0]),
                detail="mode among selected folds",
            )


def _strategy_run_dates(
    *,
    daily_exposure: pd.DataFrame,
    benchmark_relative: pd.DataFrame,
) -> pd.Index:
    if not daily_exposure.empty and {"date", "strategy"}.issubset(daily_exposure.columns):
        dates = pd.to_datetime(
            daily_exposure.loc[
                daily_exposure["strategy"].astype(str).eq(STRATEGY_NAME),
                "date",
            ],
            errors="coerce",
        ).dropna()
        if not dates.empty:
            return pd.Index(dates.drop_duplicates())

    if not benchmark_relative.empty and {"date", "strategy"}.issubset(
        benchmark_relative.columns
    ):
        dates = pd.to_datetime(
            benchmark_relative.loc[
                benchmark_relative["strategy"].astype(str).eq(STRATEGY_NAME),
                "date",
            ],
            errors="coerce",
        ).dropna()
        if not dates.empty:
            return pd.Index(dates.drop_duplicates())

    return pd.Index([])


def build_phase8_bull_participation(
    run_dir: str | Path,
    *,
    config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_run_dir = Path(run_dir)
    probability = _read_csv(resolved_run_dir, "allocation_probability_diagnostics.csv")
    daily_exposure = _read_csv(resolved_run_dir, "daily_exposure.csv")
    benchmark_relative = _read_csv(resolved_run_dir, "benchmark_relative.csv")
    date_regime, date_regime_source = _resolve_date_regime(
        probability=probability,
        config_path=config_path,
    )
    run_dates = _strategy_run_dates(
        daily_exposure=daily_exposure,
        benchmark_relative=benchmark_relative,
    )

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    _target_prediction_rows(
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        probability=probability,
    )
    _runtime_participation_rows(
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        daily_exposure=daily_exposure,
        date_regime=date_regime,
        date_regime_source=date_regime_source,
    )
    _gate_runtime_rows(
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        date_regime=date_regime,
        run_dates=run_dates,
    )
    _bull_active_return_rows(
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        benchmark_relative=benchmark_relative,
        daily_exposure=daily_exposure,
        date_regime=date_regime,
    )
    _selection_context_rows(
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        selections=_read_csv(resolved_run_dir, "ml_strategy_tuning_selections.csv"),
    )
    return (
        pd.DataFrame(detail_rows, columns=PHASE8_BULL_PARTICIPATION_COLUMNS),
        pd.DataFrame(summary_rows, columns=PHASE8_BULL_PARTICIPATION_SUMMARY_COLUMNS),
    )


def write_phase8_bull_participation(
    run_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    resolved_run_dir = Path(run_dir)
    resolved_output_dir = Path(output_dir) if output_dir is not None else resolved_run_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    detail, summary = build_phase8_bull_participation(
        resolved_run_dir,
        config_path=config_path,
    )
    detail_path = resolved_output_dir / "phase8_bull_participation.csv"
    summary_path = resolved_output_dir / "phase8_bull_participation_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return detail_path, summary_path
