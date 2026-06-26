# Phase 9 P9-10 Worker Plan

- Branch: `feature/phase-9-qqq-paper-azure`
- Pull request: `feat(phase9): add QQQ paper Azure dev deployment`
- Dependencies: P9-02 Terraform bootstrap and P9-09 paper Azure runtime seams

## Objective

Build the QQQ paper Azure deployment foundation in a small source-only packet.
The completed work adds the `infra/azure/qqq-paper/` Terraform root, a
validation-only CI path, and disabled-by-default Container Apps Jobs wired to
the existing P9-07, P9-08, and P9-09 paper runtime seams.

This packet does not apply Terraform, enable schedules, import local QQQ
state, run UAT shadow validation, or submit broker orders. All Azure execution
remains blocked until a separate supervised deployment session reviews exact
costs, names, secrets, image digest, plan output, and activation evidence.

## Scope

- add the `infra/azure/qqq-paper/` Terraform root
- reserve the dev remote-state key `qqq-paper-dev.tfstate`
- create only QQQ paper dev infrastructure examples and validation contracts
- add a resource group, managed identity, ACR, Log Analytics, Application
  Insights, Container Apps Environment, PostgreSQL Flexible Server and
  database, StorageV2 Blob artifact container, Service Bus namespace and queue,
  Key Vault, role assignments, and disabled/manual Container Apps Jobs
- wire jobs for `paper-db-migrate`, `paper-scheduler --once`,
  `paper-agent-approve --once`, `paper-outbox-deliver`,
  `paper-notifications-deliver`, `paper-blob-sync`, and
  `paper-service-bus-receive`
- keep `create_jobs = false`, `enable_scheduler_schedule = false`,
  `enable_service_bus_approval_trigger = false`, and
  `enable_broker_secret_refs = false` by default
- pass hosted runtime endpoints through opt-in environment overrides using
  `MARKETLAB_PAPER_RUNTIME_ENV_OVERRIDES=1`
- keep `configs/experiment.qqq_paper_daily.yaml` as the canonical QQQ strategy
  config
- continue using `MARKETLAB_PAPER_POSTGRES_DSN` from Key Vault for dev
  PostgreSQL access
- extend Terraform CI to run format, backend-disabled init, and validate for
  the QQQ paper root

## Out Of Scope

- Terraform plan, apply, destroy, import, provider registration, or Azure login
  in repository CI
- tracked `.tfvars`, backend files, Terraform state, plans, secret values, or
  live Azure identifiers
- passwordless PostgreSQL managed-identity authentication
- broker submission enablement or any live-money endpoint
- QQQ strategy, model, schedule, approval, proposal, submission, report, or
  notification semantic changes
- VOO migration, BTC broker migration, state import, backup/restore runbooks,
  UAT parity, failure drills, paper-prod cutover, or post-cutover closeout

## Activation Gate

The first supervised dev deployment must keep every job manual or disabled.
`create_jobs` may only be set after an immutable image digest exists in the
approved ACR. `enable_scheduler_schedule` and
`enable_service_bus_approval_trigger` remain false until migration, manual job
smoke evidence, and broker-disabled dev checks are reviewed.

Broker-facing secrets are not referenced unless `enable_broker_secret_refs`
is true in an explicitly reviewed Terraform change. Dev proof should first use
database migration, Blob synchronization, and outbox commands that do not place
broker orders.

## Validation

Repository validation for this packet:

```text
python -m pytest -q tests/unit/test_phase9_plan_docs.py tests/unit/test_phase9_azure_bootstrap.py tests/unit/test_config.py tests/unit/test_cli.py
python -m ruff check .
python -m mkdocs build --strict
python -m tox -e terraform
py -3.14 -m tox -e preflight
git diff --check
```

Terraform validation is format, backend-disabled init, and validate only. It
must not run plan, apply, destroy, import, refresh-only planning, Azure login,
or provider registration.
