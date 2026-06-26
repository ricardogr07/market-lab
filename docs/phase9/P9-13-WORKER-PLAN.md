# Phase 9 P9-13 Worker Plan: QQQ Paper-Prod Cutover

## Summary

P9-13 prepares the guarded QQQ paper-prod cutover after P9-12 UAT parity and
failure-drill evidence has been accepted. It converts the QQQ Azure root from a
dev-only source contract into an explicit `dev` and `paper-prod` contract with
production launch gates.

P9-13 does not stop the local scheduler, run a final import, apply Terraform,
change secrets, enable Azure jobs, submit broker orders, or perform the live
cutover by itself. Those actions require a separate supervised operator
session after this source packet is reviewed and merged.

- Branch: `feature/phase-9-qqq-paper-prod-cutover`
- PR title: `feat(phase9): add QQQ paper-prod cutover gates`
- Dependencies: accepted P9-12 QQQ UAT parity and failure-drill evidence
- Canonical config remains `configs/experiment.qqq_paper_daily.yaml`

## Key Changes

- Extend `infra/azure/qqq-paper` to support only `dev` and `paper-prod`.
- Keep dev defaults disabled while allowing paper-prod trigger enablement only
  when production evidence URIs, non-placeholder image digest, secret refs,
  broker secret refs, and job creation are all explicitly configured.
- Add the paper-prod backend example using `qqq-paper-prod.tfstate`; keep real
  backend files, tfvars, plans, state, DSNs, and Key Vault secret IDs untracked.
- Document the paper-prod cutover runbook in `docs/paper-trading.md`.
- Link this worker plan from the Phase 9 roadmap, docs index, and MkDocs nav.

Production launch gates require:

- `environment = "paper-prod"`
- `create_jobs = true`
- a non-placeholder immutable `marketlab_image_digest`
- `enable_broker_secret_refs = true`
- populated Key Vault secret IDs for Alpaca, Anthropic, and Telegram runtime
  secrets
- reviewed HTTPS evidence URIs for P9-12 parity, final import, backup/restore,
  rollback, and alert checks

## Cutover Runbook Contract

The P9-13 runbook must require operators to:

- confirm P9-12 parity and failure-drill acceptance
- stop the local QQQ scheduler and agent before paper-prod trigger enablement
- verify there is no unresolved proposal or non-terminal order
- run final `paper-state-import --dry-run`, then approved `--apply`
- run `paper-blob-sync`
- manually smoke-test Azure jobs while schedules remain disabled
- enable the paper-prod scheduler and Service Bus approval trigger only after
  the smoke test passes
- verify the first production-paper cycle from PostgreSQL, Blob, Service Bus,
  alerts, and paper broker evidence
- roll back by disabling Azure jobs and restoring PostgreSQL/Blob state if a
  gate fails

P9-14 owns the ten-trading-day post-cutover observation window and closeout.

## Tests And Validation

- Add docs tests for P9-13 links, boundaries, branch metadata, cutover gates,
  rollback language, and the P9-14 handoff.
- Add Terraform static tests for the `dev`/`paper-prod` environment set,
  production-only trigger gates, evidence URI requirements, non-placeholder
  image digest requirement, secret-reference requirements, and paper-prod
  backend key.

Validation commands:

```bash
py -3.14 -m pytest -q tests/unit/test_phase9_plan_docs.py tests/unit/test_phase9_azure_bootstrap.py
python -m ruff check .
python -m mkdocs build --strict
git diff --check
```

Run `py -3.14 -m tox -e preflight` before publication when the local Windows
workspace is free of generated artifact file locks.

## Assumptions And Boundaries

- P9-13 prepares source gates and operator documentation only.
- Live Azure apply, local scheduler shutdown, final import, secret changes, and
  trigger enablement require explicit supervised approval after merge.
- QQQ strategy, approval semantics, Alpaca paper-only endpoint checks, VOO, BTC,
  and live-money support remain out of scope.
