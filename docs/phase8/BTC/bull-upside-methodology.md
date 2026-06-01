# BTC Phase 8 Bull-Upside Methodology

This document records the completed BTC Phase 8 bull-capture allocation grid
and the resulting research pivot. The strict research gate remains the only
deployment pass condition.

## Current State

The latest completed baseline methodology run is:

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

The diagnostic counterfactuals remain hypothesis evidence only. Forcing runtime
bull days to 100% exposure returned 75.3241 with 0.4502 average exposure, and
treating strict-gate bull days as buy-hold returned 58.9625 with 0.5164 average
exposure. Those rows show that upside exists in the run window; they do not show
that the validation-selected model can identify it without look-ahead.

## Completed Bull-Capture Matrix

The previous next-run matrix is now complete in
`artifacts/runs/phase8_btc_grid_comparison.csv`.

| Config | Run ID | Strict gate | Active return vs buy-hold | Selected fold fraction | Predicted 100% tier | Score validity |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `configs/experiment.btc_phase8_bull_capture_rebalanced_gate.yaml` | `20260524T194740Z` | Failed | -10.1846 | 1.0000 | 0.0000 | Negative score/outcome correlations. |
| `configs/experiment.btc_phase8_bull_capture_prob100_grid.yaml` | `20260524T221456Z` | Failed | -10.4788 | 0.9333 | 0.1927 | 100% predictions appeared, but score/outcome correlations stayed negative. |
| `configs/experiment.btc_phase8_bull_capture_static_audit.yaml` | `20260524T232741Z` | Failed | -10.1846 | 1.0000 | 0.0000 | Negative score/outcome correlations. |

The matrix improved selection coverage relative to the cash-heavy methodology
review, but it did not solve the research problem:

- every completed run failed the unchanged strict gate
- active return versus buy-and-hold stayed deeply negative
- gate-bull active return stayed negative
- positive gate-bull days were still materially underexposed
- target support stayed concentrated in 25% and 50% labels
- selected OOS 100% prediction support was absent or insufficient
- score-to-target, score-to-forward-return, and score-to-realized-utility
  correlations stayed negative

Allocation-grid tuning is therefore no longer the next best path. The current
evidence points to a target/score definition problem: the labels and scores do
not reliably separate BTC bull-continuation participation from drawdown-defense
behavior.

## Pivot

The next methodology work is tracked in
[BTC Phase 8 Target/Score Pivot](target-score-pivot.md).

The pivot adds one artifact-only diagnostic and three runnable configs:

- `phase8-target-diagnostic` to inspect target and prediction behavior by
  runtime regime, gate-bull rows, forward-return sign, target tier, predicted
  tier, drawdown, and realized utility
- `phase8-regime-policy-sweep` to test completed-bar runtime and gate-bull
  exposure policies against persisted artifacts before launching another
  expensive BTC training batch
- `configs/experiment.btc_phase8_target_return_capture_utility.yaml`
- `configs/experiment.btc_phase8_bull_signal_feature_utility.yaml`
- `configs/experiment.btc_phase8_regime_state_bull_signal.yaml`

The first target/score batch found one useful branch:
`btc_phase8_bull_floor_signal_return_capture_fallback` beat buy-and-hold by
`+0.1227`, but still failed the strict gate and produced no selected OOS 100%
predicted tiers. The next repair batch adds research-only score transforms to
that branch:

- `configs/experiment.btc_phase8_bull_floor_score_boost_fallback.yaml`
- `configs/experiment.btc_phase8_bull_floor_score_boost_uncalibrated.yaml`
- `configs/experiment.btc_phase8_bull_floor_score_boost_long_train.yaml`

Run the pivot batch one experiment at a time because each BTC long-history run
is expensive:

```bash
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_target_return_capture_utility.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_bull_signal_feature_utility.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_regime_state_bull_signal.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_bull_floor_score_boost_fallback.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_bull_floor_score_boost_uncalibrated.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_bull_floor_score_boost_long_train.yaml
```

After each run, regenerate diagnostics:

```bash
python scripts/run_marketlab.py phase8-summary --run-dir artifacts/runs/<experiment>/<run-id>
python scripts/run_marketlab.py phase8-target-diagnostic --run-dir artifacts/runs/<experiment>/<run-id> --config configs/<experiment>.yaml
python scripts/run_marketlab.py phase8-bull-participation --run-dir artifacts/runs/<experiment>/<run-id> --config configs/<experiment>.yaml
python scripts/run_marketlab.py phase8-score-diagnostic --run-dir artifacts/runs/<experiment>/<run-id>
python scripts/run_marketlab.py phase8-bull-counterfactual --run-dir artifacts/runs/<experiment>/<run-id> --config configs/<experiment>.yaml
python scripts/run_marketlab.py phase8-regime-policy-sweep --run-dir artifacts/runs/<experiment>/<run-id> --config configs/<experiment>.yaml
python scripts/run_marketlab.py phase8-methodology-review --run-dir artifacts/runs/<experiment>/<run-id>
```

Then compare all completed BTC Phase 8 runs:

```bash
python scripts/run_marketlab.py phase8-grid-compare --runs-root artifacts/runs --output artifacts/runs/phase8_btc_grid_comparison.csv
```

## Decision Criteria

Use `phase8_btc_grid_comparison.csv` plus the target diagnostic to rank the
next BTC method by these criteria, in this order:

1. `strict_gate_passed` remains the only deployment pass condition.
2. `active_return_vs_buy_hold` must improve materially, not only versus
   rebalanced benchmarks.
3. `bull_upside_capture_ratio` and `gate_bull_average_long_exposure` must show
   actual BTC bull participation.
4. `gate_bull_underexposed_positive_benchmark_return_sum` must shrink.
5. Score diagnostics must stop showing negative score-to-outcome relationships.
6. `phase8-target-diagnostic` should show higher full-target and
   full-prediction fractions on bull-continuation rows without turning
   drawdown-defense rows into full-exposure labels.
7. `selected_fold_fraction` must reach the strict coverage threshold without
   relying on fallback-only success.

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
