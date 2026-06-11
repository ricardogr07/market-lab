# Azure Terraform State Bootstrap

This Terraform root implements the repository-local portion of Phase 9 packet
`P9-02`. It is intentionally limited to the remote-state foundation. It does
not deploy MarketLab workloads or create application credentials.

## Approval Boundary

Local formatting, backend-disabled initialization, validation, and tests are
safe before Azure approval. Stop and obtain explicit user approval before:

- `az login`, subscription or tenant discovery, or Azure Portal access
- resource-provider registration
- any Terraform command that reads, plans against, imports, changes, or
  destroys Azure resources
- creating a Cost Management budget or any other Azure object
- committing these files
- pushing the branch or opening a pull request

The provider sets `resource_provider_registrations = "none"`. Terraform must
not register Azure providers implicitly during the supervised plan. The
provider and eventual remote backend use Microsoft Entra authentication;
shared storage keys are disabled. The committed root initially uses local
state because it must create its own remote backend. After apply, copy
`backend.tf.example` to ignored `backend.tf` and migrate immediately.

## Resource Scope

The supervised bootstrap may create only:

1. `azurerm_resource_group`
2. `azurerm_storage_account`
3. `azurerm_role_assignment`
4. `azurerm_storage_container`
5. `azurerm_management_lock`

The storage account is Standard ZRS with TLS 1.2, HTTPS-only traffic, Blob
versioning, 30-day Blob and container soft delete, private container access,
and a `CanNotDelete` lock. Its public endpoint remains enabled for the local
operator and future GitHub-hosted runners, but data access requires Entra RBAC.

## Local Validation Only

Terraform `1.15.5` is required. These commands download the pinned provider but
do not authenticate to or inspect an Azure subscription:

```powershell
terraform fmt -check -recursive infra/azure
terraform -chdir=infra/azure/bootstrap init -backend=false -lockfile=readonly -input=false
terraform -chdir=infra/azure/bootstrap validate -no-color
python -m tox -e terraform
```

The committed provider lock must cover Linux AMD64 and Windows AMD64 so local
validation and GitHub Actions resolve the same AzureRM release.

## Supervised Bootstrap Session

Do not run this section until the user approves Stage 3 in
`docs/phase9/P9-02-WORKER-PLAN.md`.

### 1. Confirm Inputs

Confirm the subscription ID, tenant ID, `eastus2` region, unique suffix,
`owner`, `cost-center`, budget recipient, and operator permissions. Set the
AzureRM v4 subscription variable only for the supervised shell:

```powershell
$env:ARM_SUBSCRIPTION_ID = "<approved-subscription-id>"
```

Create an ignored `infra/azure/bootstrap/terraform.tfvars` from the example.
Never place credentials in that file.

### 2. Discover Azure State

Run the exact read-only Azure CLI discovery commands from the worker plan. If
`Microsoft.Storage`, `Microsoft.Authorization`, or `Microsoft.Consumption` is
not registered, stop. Registration is a separate write approval.

### 3. Review A Saved Plan

```powershell
terraform -chdir=infra/azure/bootstrap init -backend=false -lockfile=readonly -input=false
terraform -chdir=infra/azure/bootstrap plan -out bootstrap.tfplan
terraform -chdir=infra/azure/bootstrap show bootstrap.tfplan
```

The plan is invalid if it contains anything outside the five approved resource
types, any replacement, or any deletion. Report resource names, location,
principal ID, add/change/destroy counts, provider warnings, and updated cost
estimates. Stop for explicit apply approval.

### 4. Apply And Migrate State

After approval, apply only the reviewed `bootstrap.tfplan`. The role assignment
must exist before the private container because shared-key access is disabled.
Azure RBAC propagation can take several minutes; if container creation returns
HTTP 403, wait for propagation and re-run the reviewed configuration instead of
enabling shared keys.

After verification, copy `backend.tf.example` to ignored `backend.tf`, create
ignored `backend.hcl` from the Terraform `backend_config` output, and migrate
local state:

```powershell
Copy-Item infra/azure/bootstrap/backend.tf.example infra/azure/bootstrap/backend.tf
terraform -chdir=infra/azure/bootstrap init -migrate-state -backend-config=backend.hcl
terraform -chdir=infra/azure/bootstrap state list
terraform -chdir=infra/azure/bootstrap plan -detailed-exitcode
```

Exit code `0` is required from the final plan. Confirm that no local state file
remains and that Blob versioning, soft delete, Entra access, and the lease-based
state lock work as expected.

### 5. Cost And Evidence

Under user supervision, create the separate USD 5 monthly resource-group
budget with email notifications at 50%, 80%, and 100%. Budget alerts notify;
they do not stop resources. Capture a redacted evidence report with resource
IDs, plan summary, budget configuration, Activity Log entries, and the final
no-change result. Do not record credentials, tokens, or storage keys.

## Recovery And Destruction

- Investigate a state lease before using `terraform force-unlock`; never make
  `-lock=false` routine.
- Recover deleted or overwritten state through Blob versioning or soft delete.
- Keep the storage-account delete lock in place during normal operation.
- Backend destruction requires a state export, explicit lock removal, and a
  separately reviewed destroy plan.
