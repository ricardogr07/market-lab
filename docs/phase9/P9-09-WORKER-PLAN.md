# Phase 9 P9-09 Worker Plan

- Branch: `feature/phase-9-paper-azure-seams`
- Pull request: `feat(phase9): add paper Azure config and secret seams`

## Objective

Build the P9-09 adapter layer in small, reviewable steps. The completed work
introduces the shared Azure runtime seams, a Blob artifact adapter, and a
durable outbox with notification delivery plus Azure Service Bus publisher,
receiver, and approval-consumer boundaries.

This packet keeps the filesystem backend as the default and does not deploy or
provision Azure resources. Enabling `azure_blob` is an explicit config choice;
the adapter uses workload identity at runtime and is covered by deterministic
fake-client tests only.

## Scope

- add a typed paper.azure config section with explicit backend selection
- add a reusable paper secret-provider port with environment fallback
- add `EnvironmentPaperSecretProvider` as the local secret-provider implementation
- let Alpaca and Telegram resolve secrets through the new port
- route paper artifact-store construction through a backend selector
- add `AzureBlobPaperArtifactStore` for account snapshots and order previews
- preserve the filesystem JSON serialization exactly in Blob payloads
- scope Blob addresses by environment, deployment, and ISO trade date
- synchronize the full local review surface: proposal, evidence, approval,
  submission, order status, account snapshot, order preview, notification
  audit, paper status, and reports
- require an HTTPS Blob account URL, container name, environment, and deployment ID
- keep the filesystem backend as the default artifact implementation
- add tests for the new config shape, secret resolution, and notification path
- add artifact-parity tests with a deterministic Blob client fake
- add idempotent outbox records to filesystem, SQLite, and PostgreSQL adapters
- add the forward-only PostgreSQL `003_paper_outbox.sql` migration
- enqueue a hosted agent-approval request atomically with its proposal
- add an outbox dispatcher that closes its unit of work before publication
- add an Azure Service Bus publisher that preserves the outbox message ID
- prove failed delivery retries and delivered records are not republished
- persist Telegram notification intent atomically with decision, approval, and submission state
- dispatch notification-only outbox records after the phase transaction commits
- retain failed Telegram deliveries for retry without changing phase outcomes
- validate and consume one `paper.approval.requested` envelope at a time
- suppress a duplicate approval message after its proposal is no longer pending
- add an Azure Service Bus receiver loop and message settlement: complete only
  after domain processing succeeds, otherwise abandon for queue-policy retry
- provide bounded local worker commands for outbox delivery, notification
  delivery, Blob synchronization, and Service Bus approval consumption

## Out Of Scope

- scheduling those bounded commands as Container Apps Jobs or another managed
  runtime
- Azure role assignment, queue subscription policy, retry policy, or
  dead-letter configuration
- Terraform or GitHub Actions changes
- any Azure deployment or resource provisioning

## Validation

Run the targeted unit tests that cover the runtime seams and Blob parity:

```bash
py -3.14 -m pytest -q tests/unit/test_config.py tests/unit/test_paper_secrets.py tests/unit/test_paper_notifications.py tests/unit/test_paper_alpaca.py tests/unit/test_paper_blob_artifacts.py tests/unit/test_paper_outbox.py tests/unit/test_paper_outbox_delivery.py tests/unit/test_paper_service_bus.py
```

Then run the paper contract and persistence suites that exercise the unchanged
local paths:

```bash
py -3.14 -m pytest -q tests/unit/test_paper_contracts.py tests/unit/test_paper_persistence.py tests/unit/test_paper_service.py
```

P9-09 source work is complete when the four bounded commands preserve the same
duplicate-safe phase idempotency and artifact meanings as the local adapters.
Running them on a schedule or in Azure remains a deployment decision and must
not run without explicit approval.
