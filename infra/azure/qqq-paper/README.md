# Phase 9 QQQ Paper Azure Root

This Terraform root defines the source contract for P9-10: a QQQ paper dev
control plane with disabled-by-default Container Apps Jobs. It validates the
shape of the Azure resources and deployment wiring, but it does not authorize
Azure apply, job activation, state import, UAT shadow validation, or broker
submission.

## Local Validation

Run validation without configuring the backend:

```powershell
terraform -chdir=infra/azure/qqq-paper init -backend=false -lockfile=readonly -input=false
terraform -chdir=infra/azure/qqq-paper validate -no-color
```

The repository-level entrypoint is:

```powershell
python -m tox -e terraform
```

CI must remain validation-only. It must not run `az login`, provider
registration, `terraform plan`, `terraform apply`, `terraform destroy`,
`terraform import`, or refresh-only planning.

## Supervised Dev Deployment Gate

A separate supervised deployment session must review exact costs, names, image
digest, secret IDs, backend settings, and Terraform plan output before any
apply. Keep these defaults for the first reviewed apply:

```hcl
create_jobs                         = false
enable_scheduler_schedule           = false
enable_service_bus_approval_trigger = false
enable_broker_secret_refs           = false
```

After the ACR exists and an immutable MarketLab image digest is published, a
second reviewed apply may set `create_jobs = true` while still keeping the
scheduler and Service Bus triggers disabled.

## Runtime Configuration

Jobs use the canonical checked-in config:

```text
/app/configs/experiment.qqq_paper_daily.yaml
```

Azure-created endpoints are passed through explicit environment overrides:

```text
MARKETLAB_PAPER_RUNTIME_ENV_OVERRIDES=1
MARKETLAB_PAPER_PERSISTENCE_BACKEND=postgres
MARKETLAB_PAPER_AZURE_ARTIFACT_BACKEND=azure_blob
MARKETLAB_PAPER_AZURE_SECRET_BACKEND=key_vault
MARKETLAB_PAPER_AZURE_SERVICE_BUS_BACKEND=azure_service_bus
```

`MARKETLAB_PAPER_POSTGRES_DSN` is injected from a Key Vault secret reference.
P9-10 does not implement passwordless PostgreSQL managed-identity
authentication.

## Job Activation Order

Initial dev proof should run only manual jobs that cannot place broker orders:

1. `paper-db-migrate`
2. `paper-blob-sync`
3. `paper-outbox-deliver` against no pending broker-facing records
4. `paper-notifications-deliver` with Telegram forced off
5. `paper-service-bus-receive` against a controlled queue

Do not enable broker-facing secret references until a reviewed change sets
`enable_broker_secret_refs = true`. Do not enable the scheduler trigger until
P9-11 import and P9-12 dev/UAT parity gates are accepted.

## P9-10 Boundary

This root must not import QQQ state, define VOO or BTC resources, enable live
money, or change the QQQ strategy. P9-11 owns import, backup, restore, and
rollback runbooks. P9-12 owns parity and failure drills.
