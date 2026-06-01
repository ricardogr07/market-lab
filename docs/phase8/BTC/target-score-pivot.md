# BTC Phase 8 Target/Score Pivot

This note is the Phase 8 handoff after the completed bull-capture allocation
matrix. It keeps the strict research gate unchanged and pivots the next batch
away from allocation-grid tuning toward target and score methodology.

## Completed Evidence

The completed matrix in `artifacts/runs/phase8_btc_grid_comparison.csv` shows
that selection coverage improved, but the strategy still did not produce
deployable BTC bull participation.

| Experiment | Run ID | Strict gate | Active return vs buy-hold | Gate-bull exposure | Gate-bull active return | Score/target correlation |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `btc_phase8_bull_capture_rebalanced_gate` | `20260524T194740Z` | Failed | -10.1846 | 0.5322 | -2.8026 | -0.1084 |
| `btc_phase8_bull_capture_prob100_grid` | `20260524T221456Z` | Failed | -10.4788 | 0.4945 | -2.9749 | -0.0365 |
| `btc_phase8_bull_capture_static_audit` | `20260524T232741Z` | Failed | -10.1846 | 0.5322 | -2.8026 | -0.1084 |

The matrix answered the allocation-grid question:

- selection coverage reached 0.9333 to 1.0000
- target support for 25% and 50% labels remained near 0.0433 and 0.0494
- 100% predicted tier support was either absent or only 0.1927
- every run stayed negative versus buy-and-hold by more than 10 cumulative
  return points
- gate-bull active return stayed negative
- positive gate-bull days were still underexposed
- score-to-target, score-to-forward-return, and score-to-realized-utility
  correlations stayed negative

The evidence does not justify more bull-floor or threshold-grid tuning as the
next default step. The failure mode is now target/score validity: selected
models are not learning a score that rewards BTC bull continuation while still
defending drawdowns.

## New Diagnostic

Run the artifact-only target diagnostic after each new experiment:

```bash
python scripts/run_marketlab.py phase8-target-diagnostic --run-dir artifacts/runs/<experiment>/<run-id> --config configs/<experiment>.yaml
```

It writes:

- `phase8_target_diagnostic.csv`
- `phase8_target_diagnostic_summary.csv`

The report reads existing run artifacts and, when `--config` is supplied, joins
strict-gate bull labels from the prepared panel. It does not retrain models,
change exposure, change paper behavior, or redefine the strict research gate.

Review these summary rows first:

- `bull_continuation_full_target_fraction`
- `bull_continuation_full_prediction_fraction`
- `positive_return_rows_assigned_below_100_fraction`
- `drawdown_defense_rows_assigned_below_100_fraction`
- `score_target_weight_correlation`
- `score_forward_return_correlation`
- `score_realized_utility_correlation`
- `gate_bull_full_target_fraction`
- `gate_bull_full_prediction_fraction`

Run the regime-policy sweep before the next expensive batch when a completed
run shows recoverable bull upside:

```bash
python scripts/run_marketlab.py phase8-regime-policy-sweep --run-dir artifacts/runs/<experiment>/<run-id> --config configs/<experiment>.yaml
```

It writes:

- `phase8_regime_policy_sweep.csv`
- `phase8_regime_policy_sweep_summary.csv`

The sweep tests completed-bar runtime and gate-bull exposure policies against
persisted artifacts. It is diagnostic only and does not approve deployment.

## Next Experiment Batch

The target/score pivot batch completed after this note was first written. The
only branch that beat buy-and-hold was
`btc_phase8_bull_floor_signal_return_capture_fallback`, but it still failed the
strict gate and predicted no 100% allocation tiers. The next batch therefore
keeps that bull-floor fallback structure and repairs score mapping first.

## Gate-Bull Score-Validity Repair

The score-validity control is
`btc_phase8_bull_floor_score_validity_selection/20260529T233127Z`. The next
research-only batch evaluates a narrower `gate_bull_prob100_threshold` repair:

- `configs/experiment.btc_phase8_bull_floor_gate_bull_prob100_score_validity.yaml`
- `configs/experiment.btc_phase8_bull_floor_gate_bull_prob100_score_validity_uncalibrated.yaml`
- `configs/experiment.btc_phase8_bull_floor_gate_bull_prob100_score_validity_low_turnover.yaml`

The repair may promote a row to 100% BTC only when the completed-bar
`gate_bull` label is true, `prob_tier_100` passes its configured threshold,
`prob_tier_100 >= prob_tier_0`, and raw validation expected-allocation scores
have finite non-negative correlation with forward returns. Authorization is
computed before promotion, so repaired scores cannot validate their own use.
Negative-correlation fallback folds retain expected-allocation scores and
persist the denial reason.

All three challengers keep `paper.enabled: false`, do not use
`gate_bull_floor`, and leave the strict research gate unchanged. Run the
foreground batch with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_phase8_gate_bull_score_validity_repair_batch.ps1
```

Use the regenerated `phase8_btc_grid_comparison.csv` to compare each challenger
against the control. A challenger is research-worthy only when selected OOS
100% support appears with positive score/forward-return correlation and bull
participation improves without regressing buy-hold or 50 bps cost evidence.
The unchanged strict gate remains the only paper or live deployment approval.

### Completed Repair Batch

The batch completed on June 1, 2026. The calibrated repair produced selected
OOS 100% tier support with positive score/forward-return correlation, proving
that the non-circular promotion path is active. It did not improve portfolio
results versus the control because the existing validation-selected runtime
policies already produced the same effective allocations.

| Experiment | Run ID | Active return vs buy-hold | 50 bps active return vs buy-hold | Gate-bull exposure | OOS 100% tier fraction | Score/forward-return correlation | Strict gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `btc_phase8_bull_floor_score_validity_selection` | `20260529T233127Z` | +0.1227 | -1.2909 | 0.8327 | 0.0000 | 0.0805 | Failed |
| `btc_phase8_bull_floor_gate_bull_prob100_score_validity` | `20260531T182636Z` | +0.1227 | -1.2909 | 0.8327 | 0.1380 | 0.0620 | Failed |
| `btc_phase8_bull_floor_gate_bull_prob100_score_validity_uncalibrated` | `20260601T023458Z` | -4.1469 | -5.3712 | 0.7929 | 0.2218 | -0.1083 | Failed |
| `btc_phase8_bull_floor_gate_bull_prob100_score_validity_low_turnover` | `20260601T064223Z` | -5.4814 | -6.1353 | 0.7483 | 0.1067 | 0.0883 | Failed |

No challenger is promoted for further research from this batch:

- every negative-raw-correlation candidate was denied repair authorization
- invalid selected fallback folds produced zero repaired 100% trigger rows
- no challenger improved bull participation without regressing buy-hold or
  50 bps evidence
- all strict gates remained failed, so paper and live deployment remain blocked

Run one experiment at a time:

```bash
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_target_return_capture_utility.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_bull_signal_feature_utility.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_regime_state_bull_signal.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_bull_floor_score_boost_fallback.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_bull_floor_score_boost_uncalibrated.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.btc_phase8_bull_floor_score_boost_long_train.yaml
```

After each run:

```bash
python scripts/run_marketlab.py phase8-summary --run-dir artifacts/runs/<experiment>/<run-id>
python scripts/run_marketlab.py phase8-target-diagnostic --run-dir artifacts/runs/<experiment>/<run-id> --config configs/<experiment>.yaml
python scripts/run_marketlab.py phase8-bull-participation --run-dir artifacts/runs/<experiment>/<run-id> --config configs/<experiment>.yaml
python scripts/run_marketlab.py phase8-score-diagnostic --run-dir artifacts/runs/<experiment>/<run-id>
python scripts/run_marketlab.py phase8-bull-counterfactual --run-dir artifacts/runs/<experiment>/<run-id> --config configs/<experiment>.yaml
python scripts/run_marketlab.py phase8-regime-policy-sweep --run-dir artifacts/runs/<experiment>/<run-id> --config configs/<experiment>.yaml
python scripts/run_marketlab.py phase8-methodology-review --run-dir artifacts/runs/<experiment>/<run-id>
```

Then regenerate the comparison manifest:

```bash
python scripts/run_marketlab.py phase8-grid-compare --runs-root artifacts/runs --output artifacts/runs/phase8_btc_grid_comparison.csv
```

## Config Intent

| Config | Purpose | Isolation boundary |
| --- | --- | --- |
| `configs/experiment.btc_phase8_target_return_capture_utility.yaml` | Tests lower drawdown/volatility penalties so positive BTC continuation is less likely to be demoted from 100% exposure. | Uses only `model_only` regime participation and keeps signal features disabled. |
| `configs/experiment.btc_phase8_bull_signal_feature_utility.yaml` | Tests explicit binary bull-participation and drawdown-defense signals as model inputs. | Signals do not override exposure, strict gate, or paper behavior. |
| `configs/experiment.btc_phase8_regime_state_bull_signal.yaml` | Tests whether the `regime_state` target separates risk-off/reduced/risk-on behavior better when the model receives explicit regime signals. | Still uses the unchanged strict benchmark family and `model_only` participation policy. |
| `configs/experiment.btc_phase8_bull_floor_signal_return_capture_fallback.yaml` | Tests explicit research-only bull floors plus regime signal features when no ML candidate is valid. | Uses `no_valid_candidate_regime_fallback`; strict gate remains unchanged. |
| `configs/experiment.btc_phase8_prob100_signal_threshold_fallback.yaml` | Tests a prob100 threshold grid with regime signal features and fallback coverage. | Threshold grid changes model scoring only inside research runs. |
| `configs/experiment.btc_phase8_regime_floor_state_fallback.yaml` | Tests `regime_state` target behavior with explicit bull-floor regime policies. | Research-only exposure policies do not change paper behavior. |
| `configs/experiment.btc_phase8_bull_floor_score_boost_fallback.yaml` | Tests runtime-bull score boosts on the current champion branch. | Score transforms run after model scoring and before tier mapping; strict gate remains unchanged. |
| `configs/experiment.btc_phase8_bull_floor_score_boost_uncalibrated.yaml` | Tests the same score-transform grid without sigmoid calibration. | Isolates whether calibration is compressing 100% participation. |
| `configs/experiment.btc_phase8_bull_floor_score_boost_long_train.yaml` | Tests the same score-transform grid with longer rolling train windows. | Keeps the bull-floor fallback and benchmark family unchanged. |
| `configs/experiment.btc_phase8_bull_floor_gate_bull_prob100_score_validity.yaml` | Tests validation-authorized completed-bar gate-bull 100% tier repair. | Uses raw pre-promotion score validity and leaves the strict gate unchanged. |
| `configs/experiment.btc_phase8_bull_floor_gate_bull_prob100_score_validity_uncalibrated.yaml` | Tests the same repair without sigmoid calibration. | Isolates probability calibration without using `gate_bull_floor`. |
| `configs/experiment.btc_phase8_bull_floor_gate_bull_prob100_score_validity_low_turnover.yaml` | Tests the same repair with longer holding periods and lower turnover. | Keeps the score-validity objective and strict gate unchanged. |

## Decision Rule

A candidate becomes interesting only if it improves target/score validity and
BTC participation together:

- full-target and full-prediction support should rise on bull-continuation rows
- drawdown-defense rows should still be assigned below 100% exposure
- score correlations should stop being negative
- active return versus buy-and-hold should improve beyond the current champion
  margin of `+0.1227`
- strict-gate status remains the deployment decision

Counterfactual reports and target diagnostics can explain failure modes, but
they cannot approve deployment or rewrite strategy weights.
