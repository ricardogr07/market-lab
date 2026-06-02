# BTC Phase 8 Shadow-Confirmation Plan

Yes, challenger 3 beat BTC buy-and-hold on the inspected historical
out-of-sample window. Active return versus buy-and-hold was `+2.2940` at
`35 bps` and remained positive at `+0.8094` at `50 bps`. The unchanged strict
gate passed.

This is not paper or live approval. The rule was designed against inspected
history, methodology diagnostics still fail, and every selected fold depends
on a fallback path. The next step is an unchanged, signals-only forward shadow
lane.

## Historical Decision Snapshot

Lock this candidate without parameter changes:

```text
config: configs/experiment.btc_phase8_guarded_gate_bull_risk_off_override_partial_support.yaml
run: artifacts/runs/btc_phase8_guarded_gate_bull_risk_off_override_partial_support/20260602T081225Z
code lock: ce01124
```

The completed control-to-champion sequence is:

| Run | Active vs buy-hold | 50 bps active | OOS 100% tier | Score/return correlation | Strict gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Score-validity control | +0.1227 | -1.2909 | 0.0000 | 0.0805 | Failed |
| Cost-robust selector | +0.1227 | -1.2909 | 0.1380 | 0.0620 | Failed |
| Guarded risk-off override | +1.5032 | +0.1636 | 0.0629 | 0.0721 | Failed |
| Partial-support challenger | +2.2940 | +0.8094 | 0.0569 | 0.0430 | Passed |

The passed historical row still has material weaknesses:

- score-to-realized-utility correlation: `-0.0614`
- gate-bull active-return sum: `-0.6571`
- missed positive gate-bull benchmark return: `2.0599`
- selected folds: `10` `best_active_fallback`, `5` deterministic
  `regime_policy_fallback`, `0` strict selections
- validation candidates: `2160` rows, `0` strict passes
- every candidate failed required-benchmark excess and 35/50 bps robust-cost
  excess

## Locked Shadow Lane

Use the signals-only shadow lane as the approval lane:

- Freeze challenger 3 without parameter changes.
- Run from `June 3, 2026` through `June 2, 2027`.
- Perform the final labeled evaluation no earlier than `June 16, 2027`
  because the target horizon is `14` daily bars.
- Recompute daily after the completed BTC bar using only matured labels
  available at that time.
- Emit the next-effective `0%`, `25%`, `50%`, or `100%` allocation into an
  append-only journal.
- Publish monthly non-promoting progress snapshots and one final report.
- Treat any tuning change as a new candidate that restarts its confirmation
  clock.

## Separate Paper Observation Lane

The current `configs/experiment.btc_paper_daily.yaml` stack is not a mirror of
challenger 3. It is a separate `4h` direction-target consensus strategy.

Plan a later implementation packet for a true challenger-3 mirror:

- Keep Alpaca paper execution isolated from the signals-only approval lane.
- Use the same daily rolling allocation-utility procedure and completed-bar
  timing.
- Preserve next-effective allocation semantics.
- Persist paper fills, slippage, and broker-state evidence separately.
- Treat paper results as operational evidence only, never as approval
  evidence.

## Graduation Criteria

Require all of the following before paper-deployment review:

- Complete the full 365-day shadow window and 14-day maturity lag.
- Pass the unchanged strict research gate.
- Keep active return versus buy-and-hold positive at `35` and `50 bps`.
- Pass the methodology `signal_validity_gate`, including positive
  score-to-utility correlation.
- Pass the methodology `bull_participation_gate`, including positive
  gate-bull active return and no missed positive underexposed benchmark return.
- Record zero `best_active_fallback` selections.
- Record zero deterministic regime-policy fallback selections.
- Allow no paper or live promotion from monthly snapshots.

## Parallel Research Fork

Keep this ordered experiment backlog separate while the locked lane accumulates
evidence:

1. Run an artifact-only rejection audit by fold and benchmark cost.
2. Add a strict-only control cloned from challenger 3 with both fallback paths
   disabled.
3. Test one-variable strict-candidate viability challengers without relaxing
   benchmark families or 35/50 bps costs.
4. Prioritize validation robustness and score-validity repair before heavier
   models.
5. Keep heavier model families deferred until strict candidate coverage
   materially improves.
6. Compare every fork against the locked challenger without modifying the
   locked shadow lane.
