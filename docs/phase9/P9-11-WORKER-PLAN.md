# Phase 9 P9-11 Worker Plan

- Branch: `feature/phase-9-qqq-state-import-runbooks`
- Pull request: `feat(phase9): add QQQ state import and operations runbooks`
- Dependencies: P9-08 PostgreSQL persistence, P9-09 paper Azure runtime seams,
  and P9-10 QQQ Azure dev deployment

## Objective

Build the QQQ state-import and operator-readiness packet without activating
Azure runtime behavior. P9-11 adds `paper-state-import`, deterministic
checksum reporting, and runbooks for backup, restore, rollback, and supervised
dev import review.

This packet does not run a live import, apply Terraform, enable Container Apps
Jobs, consume Service Bus messages, call data providers, send Telegram
notifications, or submit broker orders. `configs/experiment.qqq_paper_daily.yaml`
remains the canonical QQQ strategy config.

## Scope

- add a bounded `paper-state-import` CLI command
- require `--config`, `--source-state-dir`, and `--source-inbox-dir`
- default to `--dry-run`; require `--apply` for PostgreSQL mutation
- support optional `--report-path` for the deterministic import report
- require `paper.persistence_backend: "postgres"` and
  `MARKETLAB_PAPER_POSTGRES_DSN`
- import existing proposal, evidence, approval, submission, order-status, and
  latest-status JSON into the P9-08 PostgreSQL tables
- include notification audits and local paper reports as checksum-only review
  artifacts for the later `paper-blob-sync` operator step
- preserve source JSON payload meanings exactly
- fail closed on malformed JSON, missing identity fields, duplicate conflicting
  proposal IDs, and existing PostgreSQL payload conflicts
- keep repeated imports idempotent when source payloads match target payloads

## Out Of Scope

- new QQQ strategy, model, timing, approval, broker, or notification semantics
- automatic Blob synchronization during import
- Terraform plan, apply, destroy, import, refresh-only planning, Azure login, or
  provider registration
- Azure job activation, schedule enablement, Service Bus trigger enablement, or
  broker secret references
- VOO migration, BTC migration, P9-12 parity acceptance, P9-13 paper-prod
  cutover, or final delta import

## Operator Contract

The supervised import flow is:

1. create and verify a PostgreSQL backup using approved operator tooling
2. run `paper-db-migrate` against the reviewed PostgreSQL config
3. run `paper-state-import --dry-run` against the current QQQ filesystem state
4. review the checksum report and confirm no conflicts
5. run `paper-state-import --apply` only after explicit approval
6. run `paper-blob-sync` after the import report is accepted
7. rehearse PostgreSQL restore and Blob restore before P9-12 parity work

Rollback is database restore plus a disabled local QQQ scheduler path. P9-11
documents rollback and restore procedures, but production cutover remains P9-13.

## Validation

Repository validation for this packet:

```text
py -3.14 -m pytest -q tests/unit/test_paper_state_import.py tests/unit/test_cli.py tests/unit/test_phase9_plan_docs.py
py -3.14 -m pytest -q tests/unit/test_paper_persistence.py tests/unit/test_paper_blob_artifacts.py
python -m ruff check .
python -m mkdocs build --strict
py -3.14 -m tox -e preflight
git diff --check
```

P9-11 is complete when dry-run and apply paths produce deterministic reports,
idempotent replays skip identical records, conflicts fail before mutation, docs
contain the backup/restore/rollback runbooks, and P9-12 remains explicitly
responsible for ten-trading-day UAT parity and failure drills.
