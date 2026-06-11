# P9-02 Worker Plan: Azure Terraform Bootstrap

## Status And Approval Boundary

This is the implementation plan for roadmap packet `P9-02`. The current branch
is `feature/phase-9-azure-bootstrap`.

As of June 7, 2026, the approved Azure resources and remote-state migration are
complete, and the final remote-backed plan reports no changes. The redacted
execution record is in [P9-02 Azure Bootstrap Evidence](P9-02-BOOTSTRAP-EVIDENCE.md).
The Cost Management budget, Activity Log review, commit, and push remain
pending approval, so the packet is not yet closed.

No Azure subscription, portal, tenant, Microsoft Entra, billing, or resource
operation is authorized by this document. P9-02 is split into a repository-only
implementation and a separately approved bootstrap session.

The worker must stop and obtain explicit approval before:

- running `az login` or reading the user's Azure subscription or tenant details
- opening or changing Azure Portal settings
- registering Azure resource providers
- creating an app registration, service principal, federated credential, role
  assignment, budget, resource group, storage account, container, or lock
- running a Terraform command that refreshes, plans against, imports, changes,
  or destroys real Azure resources
- committing the implementation
- pushing the branch or opening a pull request

Repository-only formatting, static tests, `terraform init -backend=false`, and
`terraform validate` are permitted because they do not contact an Azure
subscription or create resources.

## Goal

P9-02 creates the smallest secure foundation needed for later Azure work:

- a reproducible Terraform bootstrap root for remote state
- a local and CI validation contract that never applies infrastructure
- a supervised runbook for the one-time Azure bootstrap
- a cost and ownership model the user can inspect before spending begins

P9-02 does not deploy BTC shadow execution, QQQ paper execution, Container Apps,
PostgreSQL, Service Bus, Key Vault, ACR, Log Analytics, private networking, or
application credentials. Those belong to later packets and require separate
cost reviews and approvals.

## Locked Technical Decisions

### Toolchain

- Terraform CLI: `1.15.5`
- AzureRM provider: `~> 4.74.0`, with the generated lock file committed
- Azure region: `eastus2` by default, subject to user confirmation before plan
- Infrastructure directory: `infra/azure/bootstrap`
- State backend: private Azure Blob container using the `azurerm` backend
- Authentication: Microsoft Entra authentication, not storage access keys
- Local operator authentication: Azure CLI under the user's interactive login
- CI in P9-02: format and validate only; no Azure login and no apply job
- Future deployment authentication: GitHub OIDC, added only in a later approved
  packet

Terraform state can contain sensitive values. The backend therefore uses Blob
encryption at rest, lease-based state locking, versioning, soft delete, RBAC,
and a delete lock. Backend configuration, local state, `.tfvars`, and
`.terraform/` content remain untracked.

### Bootstrap Resources

The supervised apply creates only:

| Resource | Planned configuration | Purpose |
| --- | --- | --- |
| Resource group | `rg-marketlab-terraform-<suffix>` | Isolate state and cost ownership |
| Storage account | `mltf<suffix>`, StorageV2, Standard ZRS | Durable Terraform state |
| Blob container | `tfstate`, private | Store one state key per future Terraform root |
| RBAC assignment | Current user as Storage Blob Data Contributor | Entra-based state access |
| Resource lock | `CanNotDelete` on the storage account | Prevent accidental state-backend deletion |

Storage settings are TLS 1.2 minimum, HTTPS-only, anonymous Blob access
disabled, shared-key access disabled, Blob versioning enabled, and Blob and
container soft-delete retention set to `30` days.

The storage public endpoint remains reachable in P9-02 because the operator is
using a local machine and future GitHub-hosted runners do not have fixed private
network access. Data is not public: authentication remains Entra-only and the
container remains private. A private endpoint or self-hosted runner is a later
decision and must not be added implicitly because it changes cost and
operations.

### State Layout

Use one container with separate state keys:

```text
tfstate/
  phase9-shadow.tfstate
  qqq-paper-dev.tfstate
  qqq-paper-uat.tfstate
  qqq-paper-prod.tfstate
```

P9-02 creates the backend and migrates the bootstrap root's own local state to:

```text
tfstate/bootstrap.tfstate
```

No application state or artifacts belong in this account.

## Repository Deliverables

### Terraform Root

Add `infra/azure/bootstrap` with:

- `versions.tf`: exact Terraform and provider constraints for the initial
  local-state bootstrap
- `backend.tf.example`: the post-apply `azurerm` backend declaration requiring
  Entra and Azure CLI authentication; copied to ignored `backend.tf` only when
  the backend exists and local state is ready to migrate
- `main.tf`: resource group, storage account, container, current-principal
  RBAC, and delete lock
- `variables.tf`: validated resource suffix, location, and required tags
- `outputs.tf`: backend values for later roots without credentials
- `terraform.tfvars.example`: placeholder-only operator inputs
- `.terraform.lock.hcl`: provider checksum and exact resolved version
- `README.md`: local validation, supervised apply, state migration, recovery,
  and destroy restrictions

Required tags are:

```text
project=marketlab
managed-by=terraform
phase=9
purpose=terraform-state
owner=<user-approved value>
cost-center=<user-approved value>
```

The owner and cost-center values are decided before the real plan. They must
not be guessed or hardcoded.

### Validation Contract

Add a `terraform` tox environment that runs:

```text
terraform fmt -check -recursive infra/azure
terraform -chdir=infra/azure/bootstrap init -backend=false -lockfile=readonly -input=false
terraform -chdir=infra/azure/bootstrap validate
```

Add `.github/workflows/terraform.yml` with:

- exact Terraform `1.15.5`
- `contents: read` permissions only
- pull-request and `master` path filters for Terraform and workflow files
- format, backend-disabled initialization, and validation
- no `id-token: write`
- no Azure login
- no `terraform plan`, `apply`, `destroy`, or import
- no subscription, tenant, client, billing, or secret variables

Add deterministic repository tests that verify:

- the bootstrap resource set stays limited to the five approved resource types
- shared-key and anonymous Blob access stay disabled
- versioning, soft delete, Entra backend authentication, and delete lock remain
  configured
- automatic Azure resource-provider registration remains disabled
- the workflow cannot authenticate to Azure or apply changes
- state, backend, and variable files with operator values remain ignored

## Step-By-Step Implementation

### Stage 1: Repository-Only Work

The worker will:

1. Add the Terraform root and documentation using the locked resource set.
2. Generate and review the provider lock file without an Azure login.
3. Add the tox environment and non-applying GitHub Actions validation.
4. Add repository tests for security and scope boundaries.
5. Run Terraform format and validation locally with `-backend=false`.
6. Run targeted tests, docs validation, lint, and the canonical preflight gate.
7. Present the complete diff and validation results to the user.

The worker then stops. No commit or push occurs until the user approves the
repository diff.

### Stage 2: User Preflight

Before any Azure command, the user confirms:

- Azure subscription name and subscription ID
- Microsoft Entra tenant ID
- billing account and payment method are understood
- `eastus2` is acceptable
- globally unique lowercase suffix of `4-8` characters
- tag values for `owner` and `cost-center`
- email address for cost notifications
- a `USD 5` monthly bootstrap budget
- budget notifications at `50%`, `80%`, and `100%`
- whether the operator account has Owner, or Contributor plus permission to
  create role assignments and resource locks

The worker prepares the exact commands with the selected values but does not
run them. The user reviews the command list and approves the session.

### Stage 3: Supervised Azure Discovery

After explicit approval, run read-only commands first:

```powershell
az login
az account list --output table
az account set --subscription <approved-subscription-id>
az account show --output json
az provider show --namespace Microsoft.Storage --query registrationState
az provider show --namespace Microsoft.Authorization --query registrationState
az provider show --namespace Microsoft.Consumption --query registrationState
```

Stop and report the selected tenant, subscription, region, role capabilities,
and provider status. Provider registration is a separate write approval if any
required provider is not registered.

### Stage 4: Supervised Terraform Plan

Create an untracked `terraform.tfvars` with the approved suffix, location, and
tags, then run:

```powershell
terraform -chdir=infra/azure/bootstrap init -backend=false
terraform -chdir=infra/azure/bootstrap plan -out bootstrap.tfplan
terraform -chdir=infra/azure/bootstrap show bootstrap.tfplan
```

The plan must show only the approved bootstrap resources. The worker provides:

- exact add/change/destroy counts
- exact Azure names and region
- role assignment scope and principal ID
- storage redundancy and protection settings
- estimated monthly cost and budget threshold
- any warnings or provider registrations

Stop for explicit apply approval. Any unexpected resource or replacement
invalidates the approval and requires a new plan.

### Stage 5: Supervised Bootstrap Apply

After explicit approval:

```powershell
terraform -chdir=infra/azure/bootstrap apply bootstrap.tfplan
```

Verify the outputs and Azure resources with read-only CLI commands. Wait for
RBAC propagation, then copy `backend.tf.example` to ignored `backend.tf`,
create an untracked `backend.hcl` using the output, and migrate the bootstrap
state:

```powershell
Copy-Item infra/azure/bootstrap/backend.tf.example infra/azure/bootstrap/backend.tf
terraform -chdir=infra/azure/bootstrap init `
  -migrate-state `
  -backend-config=backend.hcl
terraform -chdir=infra/azure/bootstrap state list
terraform -chdir=infra/azure/bootstrap plan -detailed-exitcode
```

The final plan must report no changes. Confirm that no local `terraform.tfstate`
remains and that the remote Blob holds `bootstrap.tfstate` with versioning and
a lease during state writes.

### Stage 6: Cost And Ownership Controls

In the same supervised session:

1. Create the `USD 5` monthly resource-group budget with email alerts at
   `50%`, `80%`, and `100%`.
2. Confirm the storage-account `CanNotDelete` lock.
3. Review Azure Activity Log entries for every created resource and role
   assignment.
4. Save no credentials locally or in GitHub; only IDs and non-secret backend
   coordinates may be recorded.
5. Capture a redacted bootstrap evidence report containing resource IDs,
   configuration, plan summary, cost controls, and validation results.

Budget alerts notify; they do not automatically stop resources. Any automatic
shutdown policy is deferred because the state backend must remain available.

### Stage 7: Commit And Push Gate

The worker presents:

- final repository diff
- local and CI-equivalent validation results
- redacted Terraform plan and no-change verification
- actual Azure resources created
- current Cost Analysis view and budget configuration
- remaining manual actions and operational risks

The user decides whether to commit. After the local commit is reviewed, the
user separately decides whether to push and open a pull request. No push is
implied by apply approval.

## Cost Model

### P9-02 Before Bootstrap

Repository implementation and local validation create no Azure resources and
incur `USD 0` of Azure spend.

### P9-02 After Bootstrap

| Item | Expected monthly cost | Management limit |
| --- | ---: | --- |
| Resource group | `USD 0` | None |
| Entra/RBAC assignment and delete lock | `USD 0` | None |
| Standard ZRS Blob storage for small Terraform state and versions | Typically below `USD 1` | Keep state and versions below `1 GiB` |
| Low-volume Blob transactions | Typically cents | Investigate unexpected transaction growth |
| Azure Activity Log | `USD 0` for default `90` day retention | Do not export in P9-02 |
| Cost Management budget and alerts | `USD 0` | `USD 5/month` budget |

The expected P9-02 Azure total is below `USD 1/month`; `USD 5/month` is the
investigation threshold, not a spending target. Prices vary by agreement,
currency, region, and date, so the Azure pricing calculator and Cost Analysis
must be checked during the supervised plan session.

P9-02 intentionally avoids the material later costs: PostgreSQL compute and
storage, private endpoints, Service Bus, Log Analytics ingestion, ACR storage,
and Container Apps executions. P9-06 and P9-10 each require a separate approved
cost sheet before their first plan.

## User Management Runbook

### After P9-02

The user's normal responsibility is light:

- review Cost Management once per month
- respond to any `50%`, `80%`, or `100%` budget alert
- review Activity Log after each approved infrastructure change
- keep the state storage account and resource group delete lock in place
- never edit Terraform-managed resources manually without recording and
  reconciling the change
- run `terraform plan` before every apply and require a no-surprise review
- retain Owner or RBAC-administration rights personally; automation receives
  only the minimum scoped role required by its later packet

### Recovery

- A stuck Terraform lease is investigated before it is broken; never use
  `-lock=false` as routine recovery.
- A deleted or overwritten state version is recovered from Blob versioning or
  soft delete under user supervision.
- A backend outage pauses infrastructure changes; it does not affect local
  MarketLab research or current paper execution.
- Destroying the backend requires an explicit recovery export, lock removal,
  and a separately reviewed destroy plan.

## Acceptance Criteria

P9-02 is complete only when:

- the bootstrap root is formatted, validated, tested, and documented
- CI validates Terraform without Azure authentication or apply capability
- the provider lock file and exact Terraform version are pinned
- all local state, backend configuration, and operator variables are ignored
- the supervised plan contains only the approved bootstrap resources
- remote state migration succeeds and the post-migration plan is empty
- state access uses Entra RBAC rather than shared keys
- versioning, soft delete, and the delete lock are verified
- the `USD 5/month` budget and notification recipients are verified
- the user approves the final evidence before commit and separately before push

## Official References

- [Store Terraform state in Azure Storage](https://learn.microsoft.com/en-us/azure/developer/terraform/store-state-in-azure-storage)
- [Terraform `azurerm` backend](https://developer.hashicorp.com/terraform/language/settings/backends/azurerm)
- [Terraform releases](https://releases.hashicorp.com/terraform/)
- [AzureRM provider](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)
- [Use GitHub Actions to connect to Azure](https://learn.microsoft.com/en-us/azure/developer/github/connect-from-azure)
- [Azure Cost Management overview](https://learn.microsoft.com/en-us/azure/cost-management-billing/costs/overview-cost-management)
- [Azure Activity Log](https://learn.microsoft.com/en-us/azure/azure-monitor/platform/activity-log)
- [Azure resource locks](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/lock-resources)
- [Azure Blob Storage pricing](https://azure.microsoft.com/en-us/pricing/details/storage/)
