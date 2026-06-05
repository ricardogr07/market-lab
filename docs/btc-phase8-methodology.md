# BTC Phase 8 Methodology

Phase 8 is retained as historical BTC research evidence for the Phase 9 BTC
paper handoff. It is no longer maintained as a broad checked-in experiment
library.

The retained config is:

- `configs/experiment.btc_phase8_guarded_gate_bull_risk_off_override_partial_support.yaml`

The retained local evidence run is:

- `artifacts/runs/btc_phase8_guarded_gate_bull_risk_off_override_partial_support/20260602T081225Z`

## Methodology Summary

Phase 8 tested BTC allocation rules that map completed-bar model signals into
explicit BTC exposure tiers: `0%`, `25%`, `50%`, and `100%`. The research gate
compared selected strategies against BTC buy-and-hold plus static and
rebalanced BTC/cash partial-exposure benchmarks so a mostly-cash strategy could
not pass only by avoiding BTC drawdowns.

The retained partial-support config uses a daily BTC panel, allocation-utility
targets, strict cost-aware selection, and a guarded completed-bar bull/risk-off
override. It keeps `paper.enabled: false`; it is evidence for review, not a
paper-trading entry point.

## Phase 9 Boundary

Phase 9 uses `configs/experiment.btc_paper_daily.yaml` and the isolated BTC
paper Docker shape. It must stay separate from the QQQ/VOO paper inbox and
state directories.

The Phase 8 evidence can inform Phase 9 review, but it does not automatically
approve deployment, invent trades, or resize target exposure. Paper proposals
still flow through the deterministic BTC paper config, persisted evidence, and
approve/reject controls.

## Review Commands

```bash
python scripts/run_marketlab.py phase8-summary --run-dir artifacts/runs/btc_phase8_guarded_gate_bull_risk_off_override_partial_support/20260602T081225Z
python scripts/run_marketlab.py phase8-methodology-review --run-dir artifacts/runs/btc_phase8_guarded_gate_bull_risk_off_override_partial_support/20260602T081225Z
```
