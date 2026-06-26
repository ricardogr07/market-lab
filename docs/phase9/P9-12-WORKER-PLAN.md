# Phase 9 P9-12 Worker Plan: QQQ Dev And UAT Shadow Validation

## Summary

Implement P9-12 as a source-only UAT evidence packet after P9-10 and P9-11.
It adds a bounded `paper-parity-report` command and operator runbooks for
reviewing QQQ local-versus-shadow artifacts over a ten-trading-day window.

P9-12 does not run live UAT by itself, enable Azure schedules, consume Service
Bus messages, call providers, send Telegram notifications, submit broker
orders, change QQQ strategy behavior, or perform Terraform plan/apply/import.

- Branch: `feature/phase-9-qqq-uat-validation`
- PR title: `feat(phase9): add QQQ UAT parity evidence tooling`
- Dependencies: P9-10 QQQ Azure dev infrastructure and P9-11 state import
  runbooks
- Canonical config remains `configs/experiment.qqq_paper_daily.yaml`

## Key Changes

- Add `paper-parity-report`, a read-only CLI command that compares local QQQ
  paper artifacts with a shadow or UAT artifact export.
- Require explicit artifact roots and an evidence window:
  - `--config configs/experiment.qqq_paper_daily.yaml`
  - `--local-state-dir artifacts/paper/state`
  - `--shadow-state-dir <shadow-export-root>/state`
  - `--start <yyyy-mm-dd>`
  - `--end <yyyy-mm-dd>`
  - optional `--min-trading-days`, default `10`
  - optional `--explanations <json>`
  - optional `--report-path <json>`
  - optional `--markdown-path <md>`
- Compare proposal, evidence, approval, submission, order preview, account
  snapshot, order status, status, notification inventory, and report inventory
  surfaces without mutating either source.
- Produce deterministic JSON and Markdown evidence with:
  - observed trade dates
  - maximum consecutive weekday evidence window
  - matched and different counts by surface
  - unresolved differences
  - per-surface manifest checksums
  - aggregate manifest checksum
  - operator explanation status for each difference
- Treat a report as accepted only when the evidence window passes and all
  differences are either absent or explicitly marked `accepted` or `expected`.

## Operator Runbooks

Extend `docs/paper-trading.md` with P9-12 steps:

- pre-UAT checklist
- export or mount the reviewed shadow artifact root
- run `paper-parity-report`
- review unresolved differences and checksums
- maintain an explanations file for accepted or expected differences
- rerun the report until it is accepted
- record failure-drill evidence for duplicate delivery, provider timeout,
  broker timeout, rejected order, partial fill, stale data, missing bar, queue
  retry, dead-letter recovery, PostgreSQL restore, and Blob restore

Runbooks must use placeholders for secrets and live Azure names. They must not
include tracked DSNs, tfvars, backend files, Terraform state, Terraform plans,
or live identifiers.

State clearly that P9-12 owns ten-trading-day UAT parity and failure drills,
while P9-13 owns production cutover, local scheduler stop, final state delta
import, and Azure job enablement.

## Tests And Validation

- Add unit tests for parity report generation, deterministic checksums,
  ten-day evidence acceptance, unresolved difference detection, explanation
  handling, malformed JSON, short windows, and QQQ config enforcement.
- Add CLI tests for `paper-parity-report` argument forwarding, default
  `--min-trading-days 10`, custom minimum windows, JSON stdout, report output,
  and invalid input handling.
- Add docs tests requiring P9-12 links and boundary language.

Validation commands:

```bash
py -3.14 -m pytest -q tests/unit/test_paper_parity.py tests/unit/test_cli.py tests/unit/test_phase9_plan_docs.py
python -m ruff check .
python -m mkdocs build --strict
python -m mypy src/marketlab/paper/parity.py src/marketlab/cli.py
git diff --check
```

Run `py -3.14 -m tox -e preflight` before publication when the local Windows
workspace is free of generated artifact file locks.

## Assumptions And Boundaries

- P9-12 is evidence tooling and documentation work only.
- The command compares exported artifacts; it does not fetch live market data,
  read Azure resources directly, or certify an NYSE calendar.
- Human operators still own the supervised UAT run, failure-drill execution,
  and final acceptance decision.
- P9-13 owns paper-prod cutover and the final state delta import.
