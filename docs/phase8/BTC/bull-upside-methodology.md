# BTC Phase 8 Bull-Upside Methodology

This document captures the current BTC Phase 8 research state and the next
methodological step for improving bull-market upside capture without weakening
the strict research gate.

## Current State

The latest completed local methodology run is:

```text
artifacts/runs/btc_phase8_methodology_review/20260522T162012Z
```

It confirms that the current strict methodology is not deployment-ready:

| Area | Current evidence |
| --- | --- |
| Strict gate | Failed. Only 1 of 15 folds selected a candidate. |
| Net return | Strategy returned 0.0951 versus BTC buy-and-hold at 12.8086. |
| Exposure | Average gross exposure was 0.0167, below the 0.20 strict minimum. |
| Bull participation | Gate-bull average long exposure was 0.0100. |
| Missed upside | Positive BTC gate-bull days with underexposure contributed 17.0353 buy-hold return. |
| Score validity | Score-to-target, score-to-forward-return, and score-to-realized-utility correlations were negative. |
| Predicted tiers | Selected OOS predictions collapsed to 25% exposure and produced no 100% tier support. |

The diagnostic counterfactuals are useful but not approval evidence. Forcing
runtime bull days to 100% exposure returned 75.3241 with 0.4502 average exposure,
and treating strict-gate bull days as buy-hold returned 58.9625 with 0.5164
average exposure. Those rows show that upside exists in the run window; they do
not show that the validation-selected model can identify it without look-ahead.

## What Has Been Tested

The completed BTC Phase 8 experiments currently show three distinct failure
modes:

| Experiment family | Best observed role | Main limitation |
| --- | --- | --- |
| Strict methodology review | Preserves the unchanged gate and exposes target/score failure clearly. | Too little exposure; almost every fold stays cash. |
| Bull-floor immediate diagnostic | Improves absolute return and risk profile versus strict cash-heavy runs. | Still lags buy-and-hold and static BTC/cash benchmarks. |
| Full-tier score diagnostic | Tests 100% tier score support. | Still compresses selected OOS predictions and misses bull upside. |
| Fallback diagnostics | Improves selection coverage. | Does not solve buy-and-hold underperformance. |
| Allocation and regime-state challengers | Explores alternate labels. | No evidence yet that they produce deployable BTC participation. |

The financial interpretation is straightforward: the current system behaves
more like a low-exposure risk filter than a BTC bull-capture strategy. That can
be valuable only if it preserves a large enough portion of upside while
materially reducing drawdown. Current artifacts do not satisfy that tradeoff.

## Next Run Matrix

Run the next grid before adding new model families. The current failure is not
yet proven to be a scikit-learn model limitation; the artifacts first need to
separate selection objective, score mapping, and regime-floor effects.

| Config | Purpose | Expected question |
| --- | --- | --- |
| `configs/experiment.btc_phase8_bull_capture_rebalanced_gate.yaml` | Rebalanced benchmark selection with bull floors in the validation grid. | Can the strategy beat smoother BTC/cash benchmarks while keeping enough bull exposure? |
| `configs/experiment.btc_phase8_bull_capture_prob100_grid.yaml` | Lower `prob_tier_100` trigger to 0.16 and remove probability calibration. | Is 100% exposure being suppressed by score mapping/calibration rather than by the labels? |
| `configs/experiment.btc_phase8_bull_capture_static_audit.yaml` | Include static and rebalanced benchmarks in selection. | Does any candidate survive when the validation objective matches the full strict benchmark family? |

Run one experiment at a time because each BTC long-history run is expensive:

```bash
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_bull_capture_rebalanced_gate.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_bull_capture_prob100_grid.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_bull_capture_static_audit.yaml
```

After each run, regenerate diagnostics:

```bash
python scripts/run_marketlab.py phase8-summary --run-dir artifacts/runs/<experiment>/<run-id>
python scripts/run_marketlab.py phase8-bull-participation --run-dir artifacts/runs/<experiment>/<run-id> --config configs/<experiment>.yaml
python scripts/run_marketlab.py phase8-score-diagnostic --run-dir artifacts/runs/<experiment>/<run-id>
python scripts/run_marketlab.py phase8-bull-counterfactual --run-dir artifacts/runs/<experiment>/<run-id> --config configs/<experiment>.yaml
python scripts/run_marketlab.py phase8-methodology-review --run-dir artifacts/runs/<experiment>/<run-id>
```

Then compare all completed BTC Phase 8 runs:

```bash
python scripts/run_marketlab.py phase8-grid-compare --runs-root artifacts/runs --output artifacts/runs/phase8_btc_grid_comparison.csv
```

## Decision Criteria

Use `phase8_btc_grid_comparison.csv` to rank the next BTC method by these
criteria, in this order:

1. `strict_gate_passed` remains the only deployment pass condition.
2. `active_return_vs_buy_hold` must improve materially, not only versus
   rebalanced benchmarks.
3. `bull_upside_capture_ratio` and `gate_bull_average_long_exposure` must show
   actual BTC bull participation.
4. `gate_bull_underexposed_positive_benchmark_return_sum` must shrink; a high
   value means the strategy still misses positive BTC days while underexposed.
5. `downside_capture_ratio`, `strategy_max_drawdown`, and `strategy_sharpe_like`
   must preserve the reason to use an active strategy instead of buy-and-hold.
6. `selected_fold_fraction` must reach the strict coverage threshold rather
   than relying on one lucky fold.
7. Score diagnostics must stop showing negative score-to-outcome relationships
   before adding heavier ML models.

If none of the three configs improves bull capture and score validity, then the
next logical pivot is target/model research:

- add target diagnostics that distinguish bull-continuation labels from
  drawdown-defense labels
- test a binary bull-participation classifier as a separate candidate feature,
  not as an override
- compare gradient boosting variants only after the current sklearn trio is
  shown to fail under a better-defined target
- keep counterfactual exposure rules diagnostic until they are selected by
  validation and survive OOS strict-gate review

## Artifact Pruning Policy

Do not delete artifacts while methodology is still moving. First write a
comparison manifest:

```bash
python scripts/run_marketlab.py phase8-grid-compare --runs-root artifacts/runs --output artifacts/runs/phase8_btc_grid_comparison.csv
```

The `recommended_artifact_action` column is intentionally conservative:

| Action | Meaning |
| --- | --- |
| `keep_latest_complete` | Latest complete run for that experiment. Keep. |
| `keep_for_grid_comparison` | Complete older run. Keep until the comparison is reviewed. |
| `review_before_pruning` | Diagnostic or summary-only directory. Inspect manually. |
| `archive_or_prune_after_manifest` | Incomplete one/two-file run candidate. Archive or delete only after explicit approval. |

Pruning should happen only after the manifest is reviewed and copied into the
research notes or PR summary. The CLI never removes files.
