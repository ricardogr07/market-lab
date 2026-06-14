# P9-05 Worker Plan: BTC Shadow Operations And Evidence Reporting

## Packet Identity

- Branch: `feature/phase-9-btc-shadow-operations`
- Pull request: `feat(phase9): add BTC shadow operations and evidence reports`
- Dependency: merged P9-04 BTC shadow decision journal

P9-05 operationalizes the frozen BTC candidate without changing its behavior.
It adds the native evaluator, one-shot daily scheduler, append-only operational
evidence, status aggregation, and deterministic monthly and final reports.

The packet calculates every graduation metric, but it never approves trading
and never chooses `continue`, `restart`, or `stop`. P9-15 owns that formal
decision.

## Runtime Contract

The native `ShadowDecisionEvaluator` reuses the Phase 8 training, selection,
scoring, and allocation path. It only receives bars completed before the UTC
runtime cutoff. Training labels must have completed their full 14-bar horizon
before the signal date.

```text
phase9-shadow-scheduler \
  --config configs/experiment.btc_phase9_shadow_daily.yaml \
  --once \
  [--as-of 2026-06-11T01:15:00Z]
```

Each invocation:

1. re-runs `verify_shadow_contract`
2. refreshes the configured BTC panel
3. records every earlier unaccounted protocol date as missed
4. evaluates only the current effective date
5. delegates the immutable decision write to P9-04
6. writes linked decision evidence and any newly matured label evidence

The scheduler cannot reconstruct an earlier decision. Repeated current-date
invocations may create multiple attempt records, while the P9-04 journal
retains decision idempotency and conflict rejection.

## Append-Only Records

Canonical records live under:

```text
artifacts/phase9-shadow/
  decisions/<effective-date>.json
  attempts/<effective-date>/<attempt-id>.json
  evidence/decisions/<effective-date>.json
  evidence/labels/<effective-date>.json
  state/status.json
  reports/monthly/<year-month>/
  reports/final/<as-of>/
```

Attempt records include execution identity, scheduled and effective dates,
timestamps, outcome, decision path and fingerprint when available, sanitized
failure information, and all frozen contract hashes.

Decision evidence links the P9-04 output fingerprint to raw score, selected
tier, selection source, fallback mode, regime classification, input cutoff,
and a deterministic diagnostic fingerprint.

Label evidence is written only after 14 completed daily bars. It preserves the
adjusted-price inputs, realized strategy and benchmark returns, utility,
exposure, turnover, and linked decision and evidence fingerprints.

All append-only stores use exclusive creation, identical-write idempotency,
fingerprint verification on reads, and conflict preservation.
`state/status.json` is a replaceable derived cache and is never canonical
report evidence.

## Status And Reports

```text
phase9-shadow-status \
  --config configs/experiment.btc_phase9_shadow_daily.yaml \
  [--as-of 2026-06-30]

phase9-shadow-report \
  --config configs/experiment.btc_phase9_shadow_daily.yaml \
  [--as-of 2026-06-30]
```

Status accounts for every expected calendar date as successful, skipped,
failed, missed, pending, or label-pending and reports integrity failures
separately.

Monthly JSON and Markdown reports are provisional. Final snapshots cannot be
generated before `2027-06-16`. Reports rebuild from append-only records and
cannot modify evidence, enable paper execution, invoke approval services, or
call a broker.

Reports include completeness and integrity, frozen contract invariants, the
unchanged strict research gate, active return versus BTC buy-and-hold at
`35 bps` and `50 bps`, `signal_validity_gate`,
`bull_participation_gate`, fallback counts, maturity coverage, allocations,
exposure, turnover, failures, skips, and missing dates.

Graduation requires zero `best_active_fallback` and zero
`regime_policy_fallback` selections. Report output is informational and does
not record a promotion decision.

## Validation

```text
python -m pytest -q tests/unit/test_phase9_shadow_evaluator.py tests/unit/test_phase9_shadow_scheduler.py tests/unit/test_phase9_shadow_evidence.py tests/unit/test_phase9_shadow_report.py tests/unit/test_phase9_shadow_decision.py tests/unit/test_phase9_shadow_contract.py tests/unit/test_phase9_plan_docs.py
python -m ruff check src/marketlab/shadow tests/unit/test_phase9_shadow_*.py
python -m mypy src/marketlab/shadow
python -m mkdocs build --strict
py -3.14 -m tox -e preflight
git diff --check
```
