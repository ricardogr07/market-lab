# P9-04 Worker Plan: BTC Shadow Decision Journal

## Packet Identity

- Branch: `feature/phase-9-btc-shadow-decision`
- Pull request: `feat(phase9): add BTC shadow decision journal`
- Dependency: P9-03 frozen BTC shadow contract

P9-04 turns the verified P9-03 contract into a deterministic decision-record
service. It validates completed-bar timing, derives the effective date and
matured-label cutoff, fingerprints the inputs and output, and writes exactly
one append-only JSON record per effective date.

The packet does not schedule runs, generate monthly or final reports, enable
paper execution, call a broker, change strategy settings, or define a second
behavior hash. P9-05 owns scheduling, status aggregation, and reports.

## Decision Boundary

The service accepts a `VerifiedShadowContract`, a timezone-aware runtime as-of
cutoff, BTC daily bars, and a typed evaluator. It derives a label-safe
`ShadowDecisionContext` before invoking the evaluator. The returned
`ShadowDecisionEvaluation` is the output of the frozen Phase 8 strategy path.
P9-04 validates and journals that output; it does not duplicate the Phase 8
model-selection implementation.

Every invocation re-runs `verify_shadow_contract` before inspecting market
data or writing a journal entry. The supplied contract metadata must match the
fresh verification result.

For a run at `2026-06-11T01:15:00Z`:

- the `2026-06-11` in-progress daily bar is ignored
- the latest completed signal bar must be `2026-06-10`
- the effective date is `2026-06-11`
- the 14-bar matured-label cutoff is `2026-05-27`

BTC bars must be midnight UTC, unique, and continuous. The service rejects
future bars, stale latest bars, missing daily bars, insufficient maturity
history, non-BTC symbols, timezone-free cutoffs, and effective dates outside
the frozen protocol. A stale run cannot reconstruct an earlier decision.

## Record Contract

Records are written under:

```text
artifacts/phase9-shadow/decisions/<effective-date>.json
```

Each record contains:

- schema version and explicit `success`, `skipped`, or `failed` status
- candidate ID, behavior version, config hash, behavior hash, and code lock
- decision timestamp, signal date, effective date, and matured-label cutoff
- selection source, fallback mode, and target allocation
- deterministic input and output SHA-256 fingerprints
- an explicit reason for skipped or failed evaluations

Successful allocations are restricted to the frozen BTC tier set: `0.0`,
`0.25`, `0.50`, or `1.0`. Selection and fallback metadata must agree. Skipped
or failed evaluations use `selection_source: none`, have no target allocation,
and require a reason.

## Append-Only Rules

The journal creates records with exclusive file creation. An identical repeat
returns the existing record without rewriting it. Any changed metadata,
fingerprint, status, or allocation for the same effective date raises
`ShadowJournalConflictError` and preserves the original bytes.

Read and list helpers verify the stored output fingerprint before returning a
record. Malformed or modified evidence fails closed.

## CLI

The standalone wrapper directly delegates to the service:

```text
phase9-shadow-decision \
  --config configs/experiment.btc_phase9_shadow_daily.yaml \
  --evaluation runtime-shadow-evaluation.json \
  [--panel artifacts/data-btc-phase9-shadow-1d/btc_usd_phase9_shadow_1d_panel.csv] \
  [--as-of 2026-06-11T01:15:00Z]
```

The evaluation JSON contains `status`, `selection_source`, `fallback_mode`,
`target_allocation`, optional `reason`, and an `input_payload` object used in
the input fingerprint. The CLI defaults the panel to the verified config's
prepared panel path and the cutoff to the current UTC time.

## Validation And Acceptance

Run:

```text
python -m pytest -q tests/unit/test_phase9_shadow_decision.py tests/unit/test_phase9_shadow_cli.py tests/unit/test_phase9_shadow_contract.py tests/unit/test_phase9_plan_docs.py
python -m ruff check src/marketlab/shadow tests/unit/test_phase9_shadow_decision.py tests/unit/test_phase9_shadow_cli.py tests/unit/test_phase9_plan_docs.py
python -m mypy src/marketlab/shadow/contract.py src/marketlab/shadow/decision.py src/marketlab/shadow/journal.py src/marketlab/shadow/cli.py
python -m mkdocs build --strict
py -3.14 -m tox -e preflight
git diff --check
```

P9-04 is complete when valid decisions produce stable records, identical
replays are idempotent, conflicting replays preserve the original evidence,
and contract or timing drift fails before any journal write.
