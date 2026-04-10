# Phase 5 Scenario Pack

The canonical Phase 5 scenario pack is a set of one-config-per-run comparisons built on the same universe, walk-forward frame, benchmark anchor, and implementation-cost assumptions.

Every scenario in this pack runs through `run-experiment`. That keeps the output on the shared out-of-sample frame already used for baseline-plus-ML comparisons. There is still no multi-scenario runner in this phase; you run one config at a time and compare the resulting artifacts yourself.

## Shared Comparison Frame

All seven scenarios use:

- `data.symbols: [VOO, QQQ, SMH, XLV, IEMG]`
- `data.start_date: "2018-01-01"`
- `data.end_date: "2025-12-31"`
- `data.symbol_groups` with `broad_market`, `growth`, and `defensive` sleeves
- the default six-model weekly comparison set
- the default weekly walk-forward settings from `configs/experiment.weekly_rank.yaml`
- `evaluation.benchmark_strategy: "buy_hold"`
- `evaluation.cost_sensitivity_bps: [5.0, 25.0]`
- `portfolio.costs.bps_per_trade: 10`

## Scenario Table

| Repo config | Packaged template | Primary strategy rows to inspect | What changes |
| --- | --- | --- | --- |
| `configs/experiment.phase5.allocation_equal.yaml` | `phase5_allocation_equal` | `allocation_equal` | Adds the periodic equal-weight allocation baseline. |
| `configs/experiment.phase5.allocation_group.yaml` | `phase5_allocation_group` | `allocation_group_weights` | Adds a grouped allocation baseline with `broad_market: 0.50`, `growth: 0.30`, `defensive: 0.20`. |
| `configs/experiment.phase5.ranking_default.yaml` | `phase5_ranking_default` | `buy_hold`, `sma`, `ml_*` | Uses the default weekly ranking comparison under the shared Phase 5 frame. |
| `configs/experiment.phase5.ranking_capped.yaml` | `phase5_ranking_capped` | capped `ml_*` rows | Adds ranking caps: `max_position_weight=0.30`, `max_group_weight=0.35`, `max_long_exposure=0.60`, `max_short_exposure=0.60`. |
| `configs/experiment.phase5.mean_variance.yaml` | `phase5_mean_variance` | `mean_variance` | Enables the long-only mean-variance optimized baseline. |
| `configs/experiment.phase5.risk_parity.yaml` | `phase5_risk_parity` | `risk_parity` | Enables the long-only risk-parity optimized baseline. |
| `configs/experiment.phase5.black_litterman.yaml` | `phase5_black_litterman` | `black_litterman` | Enables the Black-Litterman baseline with neutral equal equilibrium weights and two mild signed-basket views. |

## What To Compare

Start with `strategy_summary.csv`. Across scenarios, compare:

- cumulative return, volatility, Sharpe, and max drawdown
- exposure fields such as average gross exposure, cash weight, and max group weight
- benchmark-relative fields when the strategy is measured against `buy_hold`
- cost-drag differences in `cost_sensitivity.csv`

When a scenario enables an optimized baseline, also inspect:

- `covariance_diagnostics.csv` and the `Covariance Diagnostics` report section
- `black_litterman_assumptions.csv` and the `Black-Litterman Assumptions` report section for the Black-Litterman scenario

Interpretation rules:

- lower risk can come from lower exposure or more cash, not better signal quality
- benchmark-relative performance and absolute return answer different questions
- factor and covariance diagnostics remain review artifacts, not scenario-selection inputs

## Commands

Repo checkout:

```bash
python scripts/run_marketlab.py run-experiment --config configs/experiment.phase5.mean_variance.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.phase5.ranking_capped.yaml
```

Installed package:

```bash
marketlab list-configs
marketlab write-config --name phase5_risk_parity --output phase5_risk_parity.yaml
marketlab run-experiment --config phase5_risk_parity.yaml
```
