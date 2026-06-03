from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from marketlab.reports.phase8_bull_counterfactual import (
    _base_counterfactual_frame,
    _benchmark_cumulative_return,
    _compound,
    _gate_benchmark_strategies,
    _scenario_metrics,
)
from marketlab.reports.phase8_bull_participation import _read_csv, _truthy

BULL_EXPOSURES = (0.75, 1.0)
SIDEWAYS_EXPOSURES = (0.0, 0.25, 0.50)
BEAR_EXPOSURES = (0.0,)
RISK_OFF_CAPS = (0.0, 0.25)
GATE_BULL_OVERRIDES = (False, True)

PHASE8_REGIME_POLICY_SWEEP_COLUMNS = [
    "policy_name",
    "section",
    "metric",
    "group",
    "value",
    "row_count",
    "detail",
]
PHASE8_REGIME_POLICY_SWEEP_SUMMARY_COLUMNS = [
    "policy_name",
    "bull_exposure",
    "sideways_exposure",
    "bear_exposure",
    "risk_off_cap",
    "gate_bull_override",
    "cumulative_return",
    "active_return_vs_buy_hold",
    "min_active_return_vs_required_benchmarks",
    "sharpe_like",
    "max_drawdown",
    "avg_long_exposure",
    "avg_turnover",
    "annualized_turnover",
    "total_turnover",
    "mutated_days",
    "gate_bull_average_long_exposure",
    "gate_bull_active_return_sum",
    "gate_bull_compound_active_return",
    "gate_bull_underexposed_positive_benchmark_fraction",
    "gate_bull_underexposed_positive_benchmark_return_sum",
    "regime_source",
]


def _policy_name(
    *,
    bull_exposure: float,
    sideways_exposure: float,
    bear_exposure: float,
    risk_off_cap: float,
    gate_bull_override: bool,
) -> str:
    def tier(value: float) -> int:
        return int(round(float(value) * 100))

    suffix = "gate_bull_100" if gate_bull_override else "runtime_only"
    return (
        f"bull{tier(bull_exposure)}_sideways{tier(sideways_exposure)}_"
        f"bear{tier(bear_exposure)}_riskoff{tier(risk_off_cap)}_{suffix}"
    )


def _append_detail(
    rows: list[dict[str, object]],
    *,
    policy_name: str,
    section: str,
    metric: str,
    value: object,
    group: object = "",
    row_count: object = pd.NA,
    detail: str = "",
) -> None:
    rows.append(
        {
            "policy_name": policy_name,
            "section": section,
            "metric": metric,
            "group": group,
            "value": value,
            "row_count": row_count,
            "detail": detail,
        }
    )


def _policy_exposure(
    base: pd.DataFrame,
    *,
    bull_exposure: float,
    sideways_exposure: float,
    bear_exposure: float,
    risk_off_cap: float,
    gate_bull_override: bool,
) -> pd.Series:
    runtime_regime = base["runtime_regime"].astype(str)
    exposure = pd.Series(float(sideways_exposure), index=base.index, dtype=float)
    exposure.loc[runtime_regime.eq("bull")] = float(bull_exposure)
    exposure.loc[runtime_regime.eq("bear")] = float(bear_exposure)
    exposure.loc[runtime_regime.eq("risk_off")] = float(risk_off_cap)
    if gate_bull_override and "gate_bull" in base.columns:
        exposure.loc[base["gate_bull"].map(_truthy)] = 1.0
    return exposure.clip(0.0, 1.0)


def _policy_performance(
    base: pd.DataFrame,
    *,
    policy_name: str,
    exposure: pd.Series,
    cost_bps: float,
) -> pd.DataFrame:
    turnover = exposure.diff().abs()
    if not turnover.empty:
        turnover.iloc[0] = abs(float(exposure.iloc[0]))
    benchmark_return = pd.to_numeric(base["benchmark_net_return"], errors="coerce").fillna(0.0)
    net_return = exposure * benchmark_return - (
        turnover.fillna(0.0) * (float(cost_bps) / 10_000.0)
    )
    return pd.DataFrame(
        {
            "date": pd.to_datetime(base["date"], errors="coerce"),
            "strategy": policy_name,
            "gross_return": exposure * benchmark_return,
            "net_return": net_return,
            "turnover": turnover.fillna(0.0),
            "equity": (1.0 + net_return).cumprod(),
        }
    )


def _mutated_day_count(base: pd.DataFrame, exposure: pd.Series) -> int:
    actual = pd.to_numeric(base["long_exposure"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return int(exposure.sub(actual).abs().gt(1e-9).sum())


def _gate_bull_metrics(
    base: pd.DataFrame,
    performance: pd.DataFrame,
    exposure: pd.Series,
) -> dict[str, float]:
    if "gate_bull" not in base.columns:
        return {
            "gate_bull_average_long_exposure": float("nan"),
            "gate_bull_active_return_sum": float("nan"),
            "gate_bull_compound_active_return": float("nan"),
            "gate_bull_underexposed_positive_benchmark_fraction": float("nan"),
            "gate_bull_underexposed_positive_benchmark_return_sum": float("nan"),
        }
    mask = base["gate_bull"].map(_truthy)
    benchmark_return = pd.to_numeric(base["benchmark_net_return"], errors="coerce").fillna(0.0)
    if not bool(mask.any()):
        return {
            "gate_bull_average_long_exposure": float("nan"),
            "gate_bull_active_return_sum": float("nan"),
            "gate_bull_compound_active_return": float("nan"),
            "gate_bull_underexposed_positive_benchmark_fraction": float("nan"),
            "gate_bull_underexposed_positive_benchmark_return_sum": float("nan"),
        }
    scenario_gate_return = performance.loc[mask, "net_return"]
    benchmark_gate_return = benchmark_return.loc[mask]
    positive_gate_bull = mask & benchmark_return.gt(0.0)
    underexposed_positive = positive_gate_bull & exposure.lt(1.0 - 1e-9)
    positive_count = int(positive_gate_bull.sum())
    return {
        "gate_bull_average_long_exposure": float(exposure.loc[mask].mean()),
        "gate_bull_active_return_sum": float(
            performance.loc[mask, "net_return"].sub(benchmark_return.loc[mask]).sum()
        ),
        "gate_bull_compound_active_return": _compound(scenario_gate_return)
        - _compound(benchmark_gate_return),
        "gate_bull_underexposed_positive_benchmark_fraction": (
            float(underexposed_positive.sum() / positive_count)
            if positive_count > 0
            else float("nan")
        ),
        "gate_bull_underexposed_positive_benchmark_return_sum": float(
            benchmark_return.loc[underexposed_positive].sum()
        ),
    }


def _benchmark_deltas(
    *,
    cumulative_return: float,
    strategy_summary: pd.DataFrame,
    required_benchmarks: list[str],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for benchmark_name in required_benchmarks:
        benchmark_return = _benchmark_cumulative_return(strategy_summary, benchmark_name)
        deltas[benchmark_name] = (
            cumulative_return - benchmark_return
            if math.isfinite(benchmark_return)
            else float("nan")
        )
    return deltas


def _add_runtime_regime_details(
    *,
    rows: list[dict[str, object]],
    policy_name: str,
    base: pd.DataFrame,
    performance: pd.DataFrame,
    exposure: pd.Series,
) -> None:
    benchmark_return = pd.to_numeric(base["benchmark_net_return"], errors="coerce").fillna(0.0)
    for regime, regime_rows in base.groupby(base["runtime_regime"].astype(str), sort=True):
        mask = base.index.isin(regime_rows.index)
        _append_detail(
            rows,
            policy_name=policy_name,
            section="runtime_regime",
            metric="average_long_exposure",
            group=regime,
            value=float(exposure.loc[mask].mean()) if bool(mask.any()) else float("nan"),
            row_count=int(mask.sum()),
        )
        _append_detail(
            rows,
            policy_name=policy_name,
            section="runtime_regime",
            metric="active_return_sum",
            group=regime,
            value=float(performance.loc[mask, "net_return"].sub(benchmark_return.loc[mask]).sum()),
            row_count=int(mask.sum()),
        )


def build_phase8_regime_policy_sweep(
    run_dir: str | Path,
    *,
    config_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_run_dir = Path(run_dir)
    base, config, date_regime_source = _base_counterfactual_frame(
        run_dir=resolved_run_dir,
        config_path=config_path,
    )
    strategy_summary = _read_csv(resolved_run_dir, "strategy_summary.csv")
    required_benchmarks = _gate_benchmark_strategies(config)
    base_cost_bps = float(config.portfolio.costs.bps_per_trade)
    periods_per_year = float(config.evaluation.periods_per_year)

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for bull_exposure in BULL_EXPOSURES:
        for sideways_exposure in SIDEWAYS_EXPOSURES:
            for bear_exposure in BEAR_EXPOSURES:
                for risk_off_cap in RISK_OFF_CAPS:
                    for gate_bull_override in GATE_BULL_OVERRIDES:
                        policy_name = _policy_name(
                            bull_exposure=bull_exposure,
                            sideways_exposure=sideways_exposure,
                            bear_exposure=bear_exposure,
                            risk_off_cap=risk_off_cap,
                            gate_bull_override=gate_bull_override,
                        )
                        exposure = _policy_exposure(
                            base,
                            bull_exposure=bull_exposure,
                            sideways_exposure=sideways_exposure,
                            bear_exposure=bear_exposure,
                            risk_off_cap=risk_off_cap,
                            gate_bull_override=gate_bull_override,
                        )
                        performance = _policy_performance(
                            base,
                            policy_name=policy_name,
                            exposure=exposure,
                            cost_bps=base_cost_bps,
                        )
                        metrics = _scenario_metrics(
                            performance,
                            exposure,
                            periods_per_year=periods_per_year,
                        )
                        benchmark_deltas = _benchmark_deltas(
                            cumulative_return=metrics["cumulative_return"],
                            strategy_summary=strategy_summary,
                            required_benchmarks=required_benchmarks,
                        )
                        gate_metrics = _gate_bull_metrics(base, performance, exposure)
                        active_vs_buy_hold = benchmark_deltas.get("buy_hold", float("nan"))
                        finite_deltas = [
                            value for value in benchmark_deltas.values() if math.isfinite(value)
                        ]
                        min_active_required = min(finite_deltas) if finite_deltas else float("nan")
                        annualized_turnover = metrics["avg_turnover"] * periods_per_year
                        summary_rows.append(
                            {
                                "policy_name": policy_name,
                                "bull_exposure": bull_exposure,
                                "sideways_exposure": sideways_exposure,
                                "bear_exposure": bear_exposure,
                                "risk_off_cap": risk_off_cap,
                                "gate_bull_override": gate_bull_override,
                                "cumulative_return": metrics["cumulative_return"],
                                "active_return_vs_buy_hold": active_vs_buy_hold,
                                "min_active_return_vs_required_benchmarks": (
                                    min_active_required
                                ),
                                "sharpe_like": metrics["sharpe_like"],
                                "max_drawdown": metrics["max_drawdown"],
                                "avg_long_exposure": metrics["avg_long_exposure"],
                                "avg_turnover": metrics["avg_turnover"],
                                "annualized_turnover": annualized_turnover,
                                "total_turnover": metrics["total_turnover"],
                                "mutated_days": _mutated_day_count(base, exposure),
                                **gate_metrics,
                                "regime_source": date_regime_source,
                            }
                        )
                        for metric in (
                            "cumulative_return",
                            "sharpe_like",
                            "max_drawdown",
                            "avg_long_exposure",
                            "avg_turnover",
                            "total_turnover",
                        ):
                            _append_detail(
                                detail_rows,
                                policy_name=policy_name,
                                section="policy_metrics",
                                metric=metric,
                                value=metrics[metric],
                                row_count=len(performance),
                            )
                        for benchmark_name, delta in benchmark_deltas.items():
                            _append_detail(
                                detail_rows,
                                policy_name=policy_name,
                                section="benchmark_deltas",
                                metric=f"active_return_vs_{benchmark_name}",
                                group=benchmark_name,
                                value=delta,
                                row_count=len(performance),
                            )
                        for metric, value in gate_metrics.items():
                            _append_detail(
                                detail_rows,
                                policy_name=policy_name,
                                section="gate_bull",
                                metric=metric,
                                value=value,
                                row_count=int(base.get("gate_bull", pd.Series(False)).map(_truthy).sum()),
                            )
                        _add_runtime_regime_details(
                            rows=detail_rows,
                            policy_name=policy_name,
                            base=base,
                            performance=performance,
                            exposure=exposure,
                        )

    detail = pd.DataFrame(detail_rows, columns=PHASE8_REGIME_POLICY_SWEEP_COLUMNS)
    summary = pd.DataFrame(
        summary_rows,
        columns=PHASE8_REGIME_POLICY_SWEEP_SUMMARY_COLUMNS,
    )
    if not summary.empty:
        summary = summary.sort_values(
            [
                "active_return_vs_buy_hold",
                "gate_bull_active_return_sum",
                "cumulative_return",
            ],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return detail, summary


def write_phase8_regime_policy_sweep(
    run_dir: str | Path,
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    resolved_run_dir = Path(run_dir)
    resolved_output_dir = Path(output_dir) if output_dir is not None else resolved_run_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    detail, summary = build_phase8_regime_policy_sweep(
        resolved_run_dir,
        config_path=config_path,
    )
    detail_path = resolved_output_dir / "phase8_regime_policy_sweep.csv"
    summary_path = resolved_output_dir / "phase8_regime_policy_sweep_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return detail_path, summary_path
