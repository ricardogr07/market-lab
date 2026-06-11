# P9-02 Azure Bootstrap Evidence

## Status

The P9-02 Azure state foundation was created and converged on June 7, 2026.
The Terraform bootstrap state now uses the Azure Blob backend and the final
remote-backed plan reported no changes.

The separately supervised Cost Management budget and Azure Activity Log review
are complete as of June 11, 2026. Commit and pull-request publication are the
remaining repository publication steps.

## Approved Scope

| Item | Approved value |
| --- | --- |
| Subscription | Approved Phase 9 subscription; name and ID retained locally |
| Tenant | Approved tenant; ID retained locally |
| Operator | Approved interactive Azure CLI principal; UPN and object ID retained locally |
| Region | `eastus2` |
| Resource suffix | Retained locally |
| Cost center | Approved value retained locally |

`Microsoft.Storage` registration was explicitly approved and completed before
resource creation. The required `Microsoft.Resources`,
`Microsoft.Authorization`, and `Microsoft.Consumption` providers were already
registered.

## Created Resources

| Terraform address | Azure resource | Verified state |
| --- | --- | --- |
| `azurerm_resource_group.terraform_state` | Name retained locally | Provisioning succeeded in `eastus2`; required tags present |
| `azurerm_storage_account.terraform_state` | Name retained locally | StorageV2, Standard ZRS, HTTPS only, TLS 1.2, public Blob access disabled, shared keys disabled, OAuth default enabled, local users disabled |
| `azurerm_role_assignment.current_principal_state_access` | Storage Blob Data Contributor | Scoped to the state account and assigned to the approved operator; assignment ID retained locally |
| `azurerm_storage_container.terraform_state` | `tfstate` | Private container; anonymous access disabled |
| `azurerm_management_lock.terraform_state` | `lock-marketlab-terraform-state` | `CanNotDelete` on the storage account |

Blob versioning is enabled. Blob and container soft-delete retention are both
30 days, with permanent delete disabled.

## Terraform Convergence

The reviewed configuration contained five resource additions and no changes
or destroys. Direct Terraform apply calls encountered repeatable local IPv6
connection resets before Azure returned complete responses. Under the user's
one-at-a-time approval, each approved resource was therefore created through
authenticated Azure ARM or Blob data-plane requests and immediately imported
into its matching Terraform address.

After all imports:

- the local-backed Terraform plan reported `No changes`
- the bootstrap state was migrated to `tfstate/bootstrap.tfstate`
- the remote state reported serial `12` and five managed resources
- Terraform acquired and released the Azure Blob lease during backend access
- the final remote-backed Terraform plan exited `0` and reported `No changes`

This recovery changed the creation transport, not the declared resource set or
the final Terraform ownership model.

## State And Credential Controls

- Backend access uses Microsoft Entra and Azure CLI authentication.
- Storage account shared-key access is disabled.
- `backend.tf`, `backend.hcl`, `terraform.tfvars`, saved plans, local state,
  and `.terraform/` content are ignored.
- No access token, storage key, client secret, or other credential was written
  to tracked files.
- The obsolete root-level local state copies were removed after the successful
  backend migration and no-change plan.

## Cost And Remaining Gates

The expected bootstrap cost remains below `USD 1/month` for a small amount of
Standard ZRS Blob state and low transaction volume. The `USD 5/month`
resource-group budget is an alert threshold, not a spending target.

| Gate | Status |
| --- | --- |
| Azure state resources | Complete |
| Remote state migration | Complete |
| Final no-change plan | Complete |
| `USD 5/month` budget with `50%`, `80%`, and `100%` alerts | Complete |
| Azure Activity Log review | Complete |
| Repository validation | Complete |
| Commit | Pending publication |
| Push or pull request | Pending publication |

## Repository Validation

The final local closeout produced these results:

- Terraform formatting check passed.
- Terraform configuration validation passed.
- Phase 9 bootstrap and roadmap tests: `15 passed`.
- Full unit suite: `586 passed`.
- Offline integration suite: `50 passed`, `2 deselected`.
- Ruff repository lint: passed.
- MkDocs strict build: passed.
- Packaging and type-check tox lanes: passed.
- Full preflight gate: passed.
- `git diff --check`: passed.

The integration suite emitted existing pandas frequency deprecation warnings;
it reported no test failures.

## References

- [Terraform `azurerm` backend](https://developer.hashicorp.com/terraform/language/settings/backends/azurerm)
- [Store Terraform state in Azure Storage](https://learn.microsoft.com/en-us/azure/developer/terraform/store-state-in-azure-storage)
- [Azure Blob versioning](https://learn.microsoft.com/en-us/azure/storage/blobs/versioning-overview)
- [Azure resource locks](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/lock-resources)
