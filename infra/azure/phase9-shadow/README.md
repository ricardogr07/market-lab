# Phase 9 BTC Shadow Azure Root

This Terraform root is the repository-only implementation of P9-06. It creates
the BTC shadow evidence lane infrastructure, but the scheduled Container Apps
Job is launch-blocked by default.

## Approval Boundary

Formatting, backend-disabled initialization, validation, and tests are allowed
without Azure approval. Stop and obtain explicit user approval before:

- `az login`, subscription or tenant discovery, Azure Portal access, or
  resource-provider registration
- any Terraform command that refreshes, plans against, imports, changes, or
  destroys real Azure resources
- setting `enable_shadow_schedule = true`
- applying, committing, pushing, or opening the pull request if the user has
  not approved that publication step

This root must not run a broker phase, request approval, submit an order,
create QQQ resources, or introduce the P9-09 generic Blob or Service Bus
adapter layer. The archive bridge must not become the P9-09 generic Blob or
Service Bus adapter layer.

## Local Validation Only

Terraform `1.15.5` is required. These commands must not authenticate to Azure:

```powershell
terraform fmt -check -recursive infra/azure
terraform -chdir=infra/azure/phase9-shadow init -backend=false -lockfile=readonly -input=false
terraform -chdir=infra/azure/phase9-shadow validate -no-color
python -m tox -e terraform
```

The committed root does not include `backend.tf`. After a supervised bootstrap,
copy `backend.tf.example` to ignored `backend.tf` and use `backend.hcl` with
the reserved `phase9-shadow.tfstate` key.

## Resource Scope

P9-06 provisions only:

- resource group
- user-assigned managed identity
- Azure Container Registry with admin access disabled
- Log Analytics and Application Insights
- Container Apps environment
- Container Apps Job using the existing Docker entrypoint, created only after
  the reviewed immutable image digest exists in ACR:
  `marketlab phase9-shadow-scheduler --config /app/configs/experiment.btc_phase9_shadow_daily.yaml --once`
- StorageV2 account with Azure Files live artifacts and a private, versioned
  Blob archive container
- Azure Monitor action group and scheduled-query alerts for job failures and
  missing shadow evidence

The Container Apps Job has no ingress block. Its registry pull and storage
permissions use the managed identity. The Azure Files mount uses the storage
account key because Container Apps environment storage requires it; the key
must remain protected in Terraform state and is not a broker, provider, or
approval secret.

The first supervised apply keeps `create_shadow_job = false` so Azure can
create the ACR before an image exists. After the immutable image digest is
published to that ACR and reviewed, a second supervised apply may set
`create_shadow_job = true` while keeping `enable_shadow_schedule = false`.

## Launch Gate

The default variable state is:

```hcl
enable_shadow_schedule = false
enable_shadow_alerts   = false
create_shadow_job      = false
```

With that default, Terraform creates no job resource, avoiding an apply-time
image pull failure against an empty new ACR. After image publication,
`create_shadow_job = true` creates a manual-trigger job only. A scheduled
trigger is emitted only when `enable_shadow_schedule = true`. That change is
invalid unless `launch_gate_evidence_uri` points to a reviewed evidence
package proving:

- immutable image digest was reviewed
- `verify_shadow_contract` still passes
- append-only decision and evidence conflict handling is proven
- missed-run accounting is proven
- archive copy succeeds
- restore into a scratch path succeeds
- job failure and missing-evidence alerts fire in tests

Azure enablement is not automatic. P9-15 owns any final BTC evidence decision.

## Archive Snapshot

Copy the live Azure Files tree into a dated Blob archive prefix:

```powershell
$snapshotDate = "2026-06-17"
$storage = "<approved-storage-account>"
$share = "phase9-shadow-live"
$container = "phase9-shadow-archive"

az storage copy `
  --source-share $share `
  --source-path "phase9-shadow" `
  --destination-container $container `
  --destination-path "snapshots/$snapshotDate" `
  --account-name $storage `
  --auth-mode login `
  --recursive
```

The archive container has Blob versioning enabled at the storage account and a
`450` day lifecycle retention policy for `snapshots/`.

## Restore Check

Restore only into a scratch path. Never overwrite the live artifact tree:

```powershell
$snapshotDate = "2026-06-17"
$storage = "<approved-storage-account>"
$container = "phase9-shadow-archive"
$scratch = "restore-check/$snapshotDate"

az storage copy `
  --source-container $container `
  --source-path "snapshots/$snapshotDate" `
  --destination-share "phase9-shadow-live" `
  --destination-path $scratch `
  --account-name $storage `
  --auth-mode login `
  --recursive
```

Compare the restored `decisions/`, `attempts/`, `evidence/`, `reports/`, and
`state/status.json` paths with the source snapshot before schedule enablement.
