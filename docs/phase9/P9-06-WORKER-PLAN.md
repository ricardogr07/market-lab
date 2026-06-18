# P9-06 Worker Plan: BTC Shadow Azure Infrastructure And Launch Gate

## Packet Identity

- Branch: `feature/phase-9-btc-shadow-azure`
- Pull request: `feat(phase9): add BTC shadow Azure launch gate`
- Dependencies: P9-02 Azure Terraform bootstrap and P9-05 BTC shadow scheduler

P9-06 adds the disabled-by-default Azure infrastructure for the frozen BTC
shadow lane. It proves that the P9-05 one-shot scheduler can be hosted as a
Container Apps scheduled job without enabling it, changing strategy behavior,
or introducing any broker, approval, or generic artifact-store path.

No Azure apply, plan against live resources, provider registration, or job
enablement is authorized by this repository packet. Those actions require a
separate supervised session with reviewed cost, names, image digest, alert
recipient, provider status, and exact Terraform plan output.

## Scope

The Terraform root is:

```text
infra/azure/phase9-shadow/
```

The root uses the existing bootstrap state convention and reserves the remote
state key:

```text
phase9-shadow.tfstate
```

It may create only the BTC shadow evidence lane resources:

1. resource group
2. managed identity
3. Azure Container Registry
4. Log Analytics workspace
5. Application Insights component
6. Container Apps environment
7. disabled scheduled Container Apps Job
8. StorageV2 account with Azure Files for live path-oriented artifacts
9. versioned private Blob archive container
10. Azure Monitor action group and query alerts for job failures and missing
    evidence

The scheduled job runs the existing container image entrypoint through the
`marketlab` CLI alias:

```text
marketlab phase9-shadow-scheduler --config /app/configs/experiment.btc_phase9_shadow_daily.yaml --once
```

The job remains launch-blocked until `enable_shadow_schedule = true` is set in
an explicitly reviewed Terraform change. The default is `false`.

## Launch Gate

The launch gate is intentionally manual and evidence-driven. The schedule may
not be enabled until all evidence below is attached to the supervised change:

- P9-05 scheduler command succeeds from the immutable image digest
- frozen config and behavior hashes still pass `verify_shadow_contract`
- append-only journal conflict handling is proven
- missed-run accounting is proven
- archive copy from Azure Files to Blob succeeds
- restore from a dated Blob snapshot into a scratch path succeeds
- Container Apps job failure alert fires in a test
- missing shadow evidence alert fires in a test
- no broker, approval, Alpaca, Telegram, Service Bus, or Key Vault secret is
  present in the BTC shadow root

The variable validation requires `launch_gate_evidence_uri` when the schedule
is enabled. That URI points to the reviewed evidence package; it is not a
secret.

## Archive And Restore Runbook

P9-06 keeps the live artifact contract filesystem-oriented by mounting Azure
Files at `/app/artifacts`, preserving the checked-in
`artifacts/phase9-shadow` relative path inside the container. The durable
graduation copy is the Blob archive. This is a BTC-shadow-specific bridge and
must not become the P9-09 generic artifact store.

Archive a live tree into a dated prefix:

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

Restore a dated archive snapshot into a scratch path only:

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

The restore check must compare decision, attempt, evidence, report, and
`state/status.json` paths before any schedule enablement. Restores never write
over the live artifact root.

## Validation

Repository validation for this packet:

```text
python -m pytest -q tests/unit/test_phase9_plan_docs.py tests/unit/test_phase9_azure_bootstrap.py tests/unit/test_phase9_shadow_cli.py
python -m pytest -q tests/unit
python -m ruff check .
python -m mypy src/marketlab/shadow src/marketlab/cli.py
python -m mkdocs build --strict
python -m tox -e terraform
py -3.14 -m tox -e preflight
git diff --check
```

Terraform validation is format, backend-disabled init, and validate only. It
must not run plan, apply, destroy, import, refresh-only planning, Azure login,
or provider registration.

## Non-Goals

- BTC broker path
- BTC approval flow
- QQQ Azure migration
- generic Blob or Service Bus adapter layer
- automatic promotion decision
- immutable Blob policy lock
- Azure apply without separate explicit approval
