from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from marketlab.backtest.metrics import compute_strategy_metrics
from marketlab.config import load_config
from marketlab.reports.phase8_bull_participation import (
    _join_date_regime,
    _numeric,
    _read_csv,
    _resolve_date_regime,
    _strategy_run_dates,
    _truthy,
)

STRATEGY_NAME = "ml_indicator_tuned__long_only__cash"
BENCHMARK_STRATEGY_NAME = "buy_hold"
SCENARIOS = (
    "actual_runtime",
    "force_runtime_bull_100",
    "gate_bull_overrides_risk_off",
    "buy_hold_gate_bull_model_elsewhere",
)

PHASE8_BULL_COUNTERFACTUAL_COLUMNS = [
    "scenario",
    "section",
    "metric",
    "group",
    "value",
    "row_count",
    "detail",
]
PHASE8_BULL_COUNTERFACTUAL_SUMMARY_COLUMNS = [
    "scenario",
    "metric",
    "value",
    "detail",
]
PHASE8_BULL_COUNTERFACTUAL_GATE_COLUMNS = [
    "scenario",
    "condition",
    "passed",
    "observed",
    "required",
    "diagnostic_only",
    "detail",
]


def _append_detail(
    rows: list[dict[str, object]],
    *,
    scenario: str,
    section: str,
    metric: str,
    value: object,
    group: object = "",
    row_count: object = pd.NA,
    detail: str = "",
) -> None:
    rows.append(
        {
            "scenario": scenario,
            "section": section,
            "metric": metric,
            "group": group,
            "value": value,
            "row_count": row_count,
            "detail": detail,
        }
    )


def _append_summary(
    rows: list[dict[str, object]],
    *,
    scenario: str,
    metric: str,
    value: object,
    detail: str = "",
) -> None:
    rows.append(
        {
            "scenario": scenario,
            "metric": metric,
            "value": value,
            "detail": detail,
        }
    )


def _append_gate(
    rows: list[dict[str, object]],
    *,
    scenario: str,
    condition: str,
    passed: bool,
    observed: object,
    required: object,
    detail: str = "",
) -> None:
    rows.append(
        {
            "scenario": scenario,
            "condition": condition,
            "passed": bool(passed),
            "observed": observed,
            "required": required,
            "diagnostic_only": True,
            "detail": detail,
        }
    )


def _condition_suffix(strategy_name: str) -> str:
    return (
        str(strategy_name)
        .strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _compound(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if values.empty:
        return float("nan")
    return float((1.0 + values).prod() - 1.0)


def _gate_benchmark_strategies(config: Any) -> list[str]:
    gate = config.evaluation.strict_research_gate
    names = [
        gate.benchmark_strategy,
        *[str(value) for value in gate.required_benchmark_strategies],
    ]
    return list(dict.fromkeys(name for name in names if name.strip()))


def _base_counterfactual_frame(
    *,
    run_dir: Path,
    config_path: str | Path,
) -> tuple[pd.DataFrame, Any, str]:
    config = load_config(config_path)
    probability = _read_csv(run_dir, "allocation_probability_diagnostics.csv")
    benchmark_relative = _read_csv(run_dir, "benchmark_relative.csv")
    daily_exposure = _read_csv(run_dir, "daily_exposure.csv")
    if benchmark_relative.empty:
        raise ValueError("phase8-bull-counterfactual requires benchmark_relative.csv")
    if daily_exposure.empty:
        raise ValueError("phase8-bull-counterfactual requires daily_exposure.csv")

    date_regime, date_regime_source = _resolve_date_regime(
        probability=probability,
        config_path=config_path,
    )
    if date_regime.empty:
        raise ValueError(
            "phase8-bull-counterfactual requires runtime/gate regime labels; pass --config"
        )
    run_dates = _strategy_run_dates(
        daily_exposure=daily_exposure,
        benchmark_relative=benchmark_relative,
    )
    if len(run_dates) > 0:
        date_regime = date_regime.loc[
            pd.to_datetime(date_regime["date"]).isin(pd.to_datetime(run_dates))
        ].copy()

    relative_rows = benchmark_relative.loc[
        benchmark_relative["strategy"].astype(str).eq(STRATEGY_NAME)
        & benchmark_relative["benchmark_strategy"].astype(str).eq(BENCHMARK_STRATEGY_NAME)
    ].copy()
    if relative_rows.empty:
        raise ValueError(
            "phase8-bull-counterfactual requires strategy-vs-buy_hold benchmark rows"
        )
    exposure_rows = daily_exposure.loc[
        daily_exposure["strategy"].astype(str).eq(STRATEGY_NAME)
    ].copy()
    if exposure_rows.empty:
        raise ValueError("phase8-bull-counterfactual requires strategy exposure rows")

    relative_rows["date"] = pd.to_datetime(relative_rows["date"], errors="coerce")
    exposure_rows["date"] = pd.to_datetime(exposure_rows["date"], errors="coerce")
    base = relative_rows.merge(
        exposure_rows.loc[:, ["date", "long_exposure"]],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    base = _join_date_regime(base, date_column="date", date_regime=date_regime)
    required_regime_columns = {"runtime_regime", "gate_bull"}
    if base.empty or not required_regime_columns.issubset(base.columns):
        raise ValueError(
            "phase8-bull-counterfactual could not join runtime/gate regime labels"
        )
    return base.sort_values("date").reset_index(drop=True), config, date_regime_source


def _scenario_exposure(base: pd.DataFrame, scenario: str) -> pd.Series:
    exposure = pd.to_numeric(base["long_exposure"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if scenario == "actual_runtime":
        return exposure
    if scenario == "force_runtime_bull_100":
        mask = base["runtime_regime"].astype(str).eq("bull")
        return exposure.mask(mask, 1.0)
    if scenario == "gate_bull_overrides_risk_off":
        mask = base["gate_bull"].map(_truthy) & base["runtime_regime"].astype(str).eq("risk_off")
        return exposure.mask(mask, 1.0)
    if scenario == "buy_hold_gate_bull_model_elsewhere":
        mask = base["gate_bull"].map(_truthy)
        return exposure.mask(mask, 1.0)
    raise ValueError(f"Unknown bull counterfactual scenario: {scenario}")


def _scenario_performance(
    base: pd.DataFrame,
    *,
    scenario: str,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.Series]:
    exposure = _scenario_exposure(base, scenario)
    turnover = exposure.diff().abs()
    if not turnover.empty:
        turnover.iloc[0] = abs(float(exposure.iloc[0]))
    benchmark_return = pd.to_numeric(base["benchmark_net_return"], errors="coerce").fillna(0.0)
    gross_return = exposure * benchmark_return
    net_return = gross_return - (turnover.fillna(0.0) * (float(cost_bps) / 10_000.0))
    equity = (1.0 + net_return).cumprod()
    performance = pd.DataFrame(
        {
            "date": pd.to_datetime(base["date"], errors="coerce"),
            "strategy": scenario,
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover.fillna(0.0),
            "equity": equity,
        }
    )
    return performance, exposure


def _benchmark_cumulative_return(strategy_summary: pd.DataFrame, name: str) -> float:
    rows = strategy_summary.loc[strategy_summary["strategy"].astype(str).eq(name)]
    if rows.empty:
        return float("nan")
    return _numeric(rows.iloc[0].get("cumulative_return"))


def _benchmark_cost_return(cost_sensitivity: pd.DataFrame, *, name: str, bps: float) -> float:
    if cost_sensitivity.empty:
        return float("nan")
    rows = cost_sensitivity.loc[
        cost_sensitivity["strategy"].astype(str).eq(name)
        & pd.to_numeric(cost_sensitivity["bps_per_trade"], errors="coerce").eq(float(bps))
    ]
    if rows.empty:
        return float("nan")
    return _numeric(rows.iloc[0].get("cumulative_return"))


def _scenario_metrics(
    performance: pd.DataFrame,
    exposure: pd.Series,
    *,
    periods_per_year: float,
) -> dict[str, float]:
    metrics = compute_strategy_metrics(performance, periods_per_year=periods_per_year)
    row = metrics.iloc[0] if not metrics.empty else pd.Series(dtype=object)
    return {
        "cumulative_return": _numeric(row.get("cumulative_return")),
        "annualized_return": _numeric(row.get("annualized_return")),
        "annualized_volatility": _numeric(row.get("annualized_volatility")),
        "sharpe_like": _numeric(row.get("sharpe_like")),
        "max_drawdown": _numeric(row.get("max_drawdown")),
        "avg_turnover": _numeric(row.get("avg_turnover")),
        "total_turnover": _numeric(row.get("total_turnover")),
        "avg_long_exposure": float(exposure.mean()) if not exposure.empty else float("nan"),
    }


def _mutated_day_count(base: pd.DataFrame, exposure: pd.Series) -> int:
    original = pd.to_numeric(base["long_exposure"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return int(exposure.sub(original).abs().gt(1e-9).sum())


def _add_scenario_rows(
    *,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    scenario: str,
    base: pd.DataFrame,
    performance: pd.DataFrame,
    exposure: pd.Series,
    metrics: dict[str, float],
    strategy_summary: pd.DataFrame,
    required_benchmarks: list[str],
) -> None:
    for metric in (
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_like",
        "max_drawdown",
        "avg_long_exposure",
        "avg_turnover",
        "total_turnover",
    ):
        _append_detail(
            detail_rows,
            scenario=scenario,
            section="scenario_metrics",
            metric=metric,
            value=metrics[metric],
            row_count=len(performance),
        )
    _append_summary(
        summary_rows,
        scenario=scenario,
        metric="cumulative_return",
        value=metrics["cumulative_return"],
        detail="diagnostic exposure proxy after configured costs",
    )
    _append_summary(
        summary_rows,
        scenario=scenario,
        metric="avg_long_exposure",
        value=metrics["avg_long_exposure"],
        detail="mean scenario exposure",
    )
    _append_summary(
        summary_rows,
        scenario=scenario,
        metric="mutated_days",
        value=_mutated_day_count(base, exposure),
        detail="days where scenario exposure differs from actual runtime exposure",
    )
    if scenario == "actual_runtime":
        artifact_return = _compound(base["strategy_net_return"])
        _append_summary(
            summary_rows,
            scenario=scenario,
            metric="artifact_cumulative_return",
            value=artifact_return,
            detail="current artifact strategy_net_return from benchmark_relative.csv",
        )

    for benchmark_name in required_benchmarks:
        benchmark_return = _benchmark_cumulative_return(strategy_summary, benchmark_name)
        delta = (
            metrics["cumulative_return"] - benchmark_return
            if math.isfinite(benchmark_return)
            else float("nan")
        )
        _append_detail(
            detail_rows,
            scenario=scenario,
            section="benchmark_deltas",
            metric=f"active_return_vs_{benchmark_name}",
            value=delta,
            row_count=len(performance),
            detail="scenario cumulative_return minus persisted benchmark cumulative_return",
        )


def _slice_return(
    performance: pd.DataFrame,
    benchmark_return: pd.Series,
    mask: pd.Series,
) -> tuple[float, float, float]:
    scenario_returns = performance.loc[mask, "net_return"]
    benchmark_returns = benchmark_return.loc[mask]
    strategy_value = _compound(scenario_returns)
    benchmark_value = _compound(benchmark_returns)
    return strategy_value, benchmark_value, strategy_value - benchmark_value


def _add_regime_rows(
    *,
    detail_rows: list[dict[str, object]],
    scenario: str,
    base: pd.DataFrame,
    performance: pd.DataFrame,
) -> int:
    positive_slices = 0
    benchmark_return = pd.to_numeric(base["benchmark_net_return"], errors="coerce").fillna(0.0)
    for slice_name in ("gate_bull", "gate_bear", "gate_sideways", "gate_high_volatility", "gate_recent"):
        if slice_name not in base.columns:
            continue
        mask = base[slice_name].map(_truthy)
        if not bool(mask.any()):
            continue
        strategy_value, benchmark_value, active_value = _slice_return(
            performance,
            benchmark_return,
            mask,
        )
        if active_value > 0.0:
            positive_slices += 1
        _append_detail(
            detail_rows,
            scenario=scenario,
            section="regime_slices",
            metric="active_return",
            group=slice_name,
            value=active_value,
            row_count=int(mask.sum()),
            detail=(
                f"scenario={strategy_value:g}; benchmark={benchmark_value:g}; "
                "diagnostic counterfactual slice"
            ),
        )
    return positive_slices


def _add_gate_rows(
    *,
    gate_rows: list[dict[str, object]],
    scenario: str,
    config: Any,
    metrics: dict[str, float],
    cost_metrics: dict[float, float],
    strategy_summary: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    required_benchmarks: list[str],
    positive_slices: int,
) -> None:
    gate = config.evaluation.strict_research_gate
    available_benchmarks = set(strategy_summary["strategy"].astype(str)) if not strategy_summary.empty else set()
    missing = [name for name in required_benchmarks if name not in available_benchmarks]
    _append_gate(
        gate_rows,
        scenario=scenario,
        condition="required_benchmark_strategies_present",
        passed=not missing,
        observed=", ".join(missing) if missing else "present",
        required=", ".join(required_benchmarks),
        detail="counterfactual diagnostic uses persisted benchmark artifacts",
    )

    benchmark_rows = strategy_summary.loc[
        strategy_summary["strategy"].astype(str).eq(gate.benchmark_strategy)
    ]
    if benchmark_rows.empty:
        _append_gate(
            gate_rows,
            scenario=scenario,
            condition=f"primary_benchmark_present_{_condition_suffix(gate.benchmark_strategy)}",
            passed=False,
            observed="missing",
            required=gate.benchmark_strategy,
        )
    else:
        benchmark_row = benchmark_rows.iloc[0]
        sharpe_delta = metrics["sharpe_like"] - _numeric(benchmark_row.get("sharpe_like"))
        drawdown_delta = metrics["max_drawdown"] - _numeric(benchmark_row.get("max_drawdown"))
        _append_gate(
            gate_rows,
            scenario=scenario,
            condition="sharpe_like_matches_or_improves",
            passed=sharpe_delta >= 0.0,
            observed=sharpe_delta,
            required=">= 0",
        )
        _append_gate(
            gate_rows,
            scenario=scenario,
            condition="max_drawdown_matches_or_improves",
            passed=drawdown_delta >= 0.0,
            observed=drawdown_delta,
            required=">= 0",
        )

    for benchmark_name in required_benchmarks:
        benchmark_return = _benchmark_cumulative_return(strategy_summary, benchmark_name)
        condition = f"net_cumulative_return_beats_{_condition_suffix(benchmark_name)}"
        if not math.isfinite(benchmark_return):
            _append_gate(
                gate_rows,
                scenario=scenario,
                condition=condition,
                passed=False,
                observed="missing",
                required=benchmark_name,
            )
            continue
        delta = metrics["cumulative_return"] - benchmark_return
        _append_gate(
            gate_rows,
            scenario=scenario,
            condition=condition,
            passed=delta > 0.0,
            observed=delta,
            required="> 0",
        )

    _append_gate(
        gate_rows,
        scenario=scenario,
        condition="average_exposure_in_range",
        passed=gate.min_average_exposure <= metrics["avg_long_exposure"] <= gate.max_average_exposure,
        observed=metrics["avg_long_exposure"],
        required=f"{gate.min_average_exposure:g} to {gate.max_average_exposure:g}",
    )
    turnover_budget = config.evaluation.ml_strategy_tuning.max_annualized_turnover
    annualized_turnover = metrics["avg_turnover"] * config.evaluation.periods_per_year
    _append_gate(
        gate_rows,
        scenario=scenario,
        condition="annualized_turnover_budget",
        passed=turnover_budget is None or annualized_turnover <= turnover_budget,
        observed=annualized_turnover,
        required=f"<= {turnover_budget:g}" if turnover_budget is not None else "not configured",
    )

    for label, bps in {
        "cost_gate_bps": float(gate.cost_gate_bps),
        "acceptable_cost_bps": float(gate.acceptable_cost_bps),
    }.items():
        scenario_return = cost_metrics.get(bps, float("nan"))
        benchmark_return = _benchmark_cost_return(
            cost_sensitivity,
            name=gate.benchmark_strategy,
            bps=bps,
        )
        delta = (
            scenario_return - benchmark_return
            if math.isfinite(scenario_return) and math.isfinite(benchmark_return)
            else float("nan")
        )
        _append_gate(
            gate_rows,
            scenario=scenario,
            condition=label,
            passed=math.isfinite(delta) and delta > 0.0,
            observed=delta if math.isfinite(delta) else "missing",
            required="> 0",
        )
        for benchmark_name in required_benchmarks:
            benchmark_cost_return = _benchmark_cost_return(
                cost_sensitivity,
                name=benchmark_name,
                bps=bps,
            )
            benchmark_delta = (
                scenario_return - benchmark_cost_return
                if math.isfinite(scenario_return) and math.isfinite(benchmark_cost_return)
                else float("nan")
            )
            _append_gate(
                gate_rows,
                scenario=scenario,
                condition=f"{label}_vs_{_condition_suffix(benchmark_name)}",
                passed=math.isfinite(benchmark_delta) and benchmark_delta > 0.0,
                observed=benchmark_delta if math.isfinite(benchmark_delta) else "missing",
                required="> 0",
            )

    _append_gate(
        gate_rows,
        scenario=scenario,
        condition="positive_active_return_regime_slices",
        passed=positive_slices >= gate.min_positive_regime_slices,
        observed=positive_slices,
        required=f">= {gate.min_positive_regime_slices:g}",
    )

    scenario_rows = [row for row in gate_rows if row["scenario"] == scenario]
    _append_gate(
        gate_rows,
        scenario=scenario,
        condition="overall",
        passed=all(bool(row["passed"]) for row in scenario_rows),
        observed="",
        required="all diagnostic counterfactual conditions pass",
        detail="does not approve or replace strict_research_gate.csv",
    )


def build_phase8_bull_counterfactual(
    run_dir: str | Path,
    *,
    config_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolved_run_dir = Path(run_dir)
    base, config, date_regime_source = _base_counterfactual_frame(
        run_dir=resolved_run_dir,
        config_path=config_path,
    )
    strategy_summary = _read_csv(resolved_run_dir, "strategy_summary.csv")
    cost_sensitivity = _read_csv(resolved_run_dir, "cost_sensitivity.csv")
    required_benchmarks = _gate_benchmark_strategies(config)
    base_cost_bps = float(config.portfolio.costs.bps_per_trade)
    gate_bps = {
        base_cost_bps,
        float(config.evaluation.strict_research_gate.cost_gate_bps),
        float(config.evaluation.strict_research_gate.acceptable_cost_bps),
    }

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        performance, exposure = _scenario_performance(
            base,
            scenario=scenario,
            cost_bps=base_cost_bps,
        )
        metrics = _scenario_metrics(
            performance,
            exposure,
            periods_per_year=float(config.evaluation.periods_per_year),
        )
        cost_metrics: dict[float, float] = {}
        for bps in gate_bps:
            cost_performance, _ = _scenario_performance(
                base,
                scenario=scenario,
                cost_bps=bps,
            )
            cost_metrics[bps] = _compound(cost_performance["net_return"])
        positive_slices = _add_regime_rows(
            detail_rows=detail_rows,
            scenario=scenario,
            base=base,
            performance=performance,
        )
        _add_scenario_rows(
            detail_rows=detail_rows,
            summary_rows=summary_rows,
            scenario=scenario,
            base=base,
            performance=performance,
            exposure=exposure,
            metrics=metrics,
            strategy_summary=strategy_summary,
            required_benchmarks=required_benchmarks,
        )
        _append_summary(
            summary_rows,
            scenario=scenario,
            metric="regime_source",
            value=date_regime_source,
            detail="runtime/gate regime labels used for scenario masks",
        )
        _add_gate_rows(
            gate_rows=gate_rows,
            scenario=scenario,
            config=config,
            metrics=metrics,
            cost_metrics=cost_metrics,
            strategy_summary=strategy_summary,
            cost_sensitivity=cost_sensitivity,
            required_benchmarks=required_benchmarks,
            positive_slices=positive_slices,
        )

    return (
        pd.DataFrame(detail_rows, columns=PHASE8_BULL_COUNTERFACTUAL_COLUMNS),
        pd.DataFrame(summary_rows, columns=PHASE8_BULL_COUNTERFACTUAL_SUMMARY_COLUMNS),
        pd.DataFrame(gate_rows, columns=PHASE8_BULL_COUNTERFACTUAL_GATE_COLUMNS),
    )


def write_phase8_bull_counterfactual(
    run_dir: str | Path,
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    resolved_run_dir = Path(run_dir)
    resolved_output_dir = Path(output_dir) if output_dir is not None else resolved_run_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    detail, summary, gate = build_phase8_bull_counterfactual(
        resolved_run_dir,
        config_path=config_path,
    )
    detail_path = resolved_output_dir / "phase8_bull_counterfactual.csv"
    summary_path = resolved_output_dir / "phase8_bull_counterfactual_summary.csv"
    gate_path = resolved_output_dir / "phase8_bull_counterfactual_gate.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    gate.to_csv(gate_path, index=False)
    return detail_path, summary_path, gate_path
