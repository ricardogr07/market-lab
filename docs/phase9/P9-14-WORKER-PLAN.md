# Phase 9 P9-14 Worker Plan: QQQ Post-Cutover Observation And Closeout

## Summary

P9-14 adds the source-side gates and operator documentation for QQQ
post-cutover closeout after P9-13 paper-prod cutover has completed and the
first production-paper cycle has been accepted. It records the required
ten-trading-day observation window, rollback rehearsal evidence, and final
local-state archival boundary.

P9-14 does not run the live observation window, apply Terraform, change
secrets, enable Azure jobs, stop or restart schedulers, submit broker orders,
or make the BTC final evidence decision. P9-15 remains the separate BTC
decision packet.

- Branch: `feature/phase-9-qqq-post-cutover-closeout`
- PR title: `feat(phase9): add QQQ post-cutover closeout gates`
- Dependency: P9-13 paper-prod cutover completed and first paper-prod cycle
  accepted
- Canonical config remains `configs/experiment.qqq_paper_daily.yaml`

## Key Changes

- Add `paper-closeout-report`, a read-only command that consumes exported
  paper-prod evidence roots and produces deterministic JSON and Markdown.
- Require explicit operator-exported roots and an evidence window:
  - `--config configs/experiment.qqq_paper_daily.yaml`
  - `--paper-prod-state-dir <export>/state`
  - `--paper-prod-artifact-dir <export>/artifacts`
  - `--start <yyyy-mm-dd>`
  - `--end <yyyy-mm-dd>`
  - optional `--min-trading-days`, default `10`
  - optional `--rollback-evidence <json>`
  - optional `--report-path <json>`
  - optional `--markdown-path <md>`
- Check proposal, evidence, approval, submission, order-status reconciliation,
  notification inventory, report inventory, duplicate broker-submission
  identifiers, alert evidence, dead-letter evidence, failed-job evidence,
  non-terminal order evidence, and rollback rehearsal acceptance.
- Link this worker plan from the Phase 9 roadmap, docs index, and MkDocs nav.

The report is accepted only when:

- at least `10` observed trading days have complete evidence
- required decision, approval, submission, reconciliation, notification, and
  report evidence is present for every observed day
- duplicate broker-submission identifiers are absent
- unresolved alerts, dead letters, failed jobs, and non-terminal orders are
  absent or explicitly marked `accepted`, `expected`, or `resolved`
- rollback rehearsal evidence is present and accepted

## Closeout Runbook Contract

Extend `docs/paper-trading.md` with P9-14 steps:

- confirm the first paper-prod cycle and P9-13 cutover evidence were accepted
- observe QQQ Azure paper-prod for `10` additional NYSE trading days
- review PostgreSQL, Blob, Service Bus, notifications, alerts, and Alpaca
  paper broker state each day
- keep local production scheduling disabled while preserving a reviewed local
  rollback runner that can use the same PostgreSQL and Blob adapters
- generate `paper-closeout-report` from exported evidence roots
- archive old local QQQ state only after the closeout report is accepted
- record operator, commit SHA, evidence URIs, archive location, and final
  closeout decision outside tracked files

Tracked documentation must not contain live Azure names, DSNs, tfvars, backend
files, Terraform state, Terraform plans, Key Vault secret IDs, broker account
identifiers, or secret values.

## Tests And Validation

- Add unit tests for accepted windows, short windows, missing evidence,
  duplicate submissions, unresolved alerts, unresolved dead letters, unresolved
  failed jobs, unresolved non-terminal orders, rollback evidence handling,
  malformed JSON, deterministic checksums, and QQQ config enforcement.
- Add CLI tests for argument forwarding, default `--min-trading-days 10`,
  custom minimum windows, JSON stdout, output paths, and invalid config
  handling.
- Add docs tests for P9-14 links, boundaries, command contract, observation
  gates, rollback language, closeout archive language, and P9-15 separation.

Validation commands:

```bash
py -3.14 -m pytest -q tests/unit/test_paper_closeout.py tests/unit/test_cli.py tests/unit/test_phase9_plan_docs.py
python -m ruff check .
python -m mkdocs build --strict
python -m mypy src/marketlab/paper/closeout.py src/marketlab/cli.py
git diff --check
```

Run `py -3.14 -m tox -e preflight` before publication when the local Windows
workspace is free of generated artifact file locks.

## Assumptions And Boundaries

- P9-14 is source tooling and documentation for closeout evidence.
- Human operators still own the live ten-trading-day observation, evidence
  export, rollback rehearsal, local-state archival, and final acceptance.
- QQQ strategy, approval semantics, broker endpoint behavior, Terraform gates,
  VOO, BTC, live-money support, and P9-15 BTC final evidence decision remain
  out of scope.
