# Cloud Migration Plan

## Purpose

This document records the agreed long-term migration plan for moving MarketLab paper trading from the current local Docker plus file-backed runtime into a production-capable Google Cloud architecture.

It is a planning and decision document, not an implementation record. The current local laptop setup remains the active runtime until the cloud migration work begins.

Implementation is deferred until **April 27, 2026** or later.

## Current State

### Runtime Today

The current paper-trading runtime is local and intentionally narrow:

- local Docker Compose
- `marketlab-paper-scheduler`
- `marketlab-paper-agent`
- `marketlab-paper-mcp`
- file-backed shared state under `artifacts/paper/`
- Telegram notifications for operational events
- `30` second poll loops for scheduler and agent worker

### Current Strengths

- simple local recovery and inspection
- explicit JSON artifacts for proposal, approval, submission, and order state
- deterministic fallback path for approvals
- working end-to-end paper-trading flow
- clear local debugging path through artifacts, Telegram, and Docker logs

### Current Limits

- always-on polling is acceptable locally but is not the desired production control-plane shape
- local filesystem state does not provide production-grade concurrency, durability, or multi-deployment isolation
- no environment separation between dev, UAT, and prod
- no cloud-native scheduling or queueing model
- no centralized structured observability
- no canonical deployment registry for multiple paper accounts or strategy variants

## Locked Decisions

### Runtime Model

The first production-grade target is **managed-first GCP**, not Kubernetes-first:

- **Cloud Run private service** for control-plane execution
- **Cloud Scheduler** for time-based phase triggers
- **Cloud Tasks** for async follow-up work and retries

### State And Artifacts

- **Cloud SQL Postgres** for canonical workflow state
- **GCS** for artifacts and reports

### Secrets

- **Secret Manager**

### Observability

- **Cloud Logging**
- **Cloud Monitoring**
- **Error Reporting**
- **Cloud Trace**
- **OpenTelemetry instrumentation** for portability

### Environment Model

- separate **dev**, **uat**, and **prod** environments
- separate GCP projects per environment

### MCP Role

MCP remains:

- inspection
- approval
- troubleshooting

MCP is explicitly **not** part of runtime orchestration.

### Scale Target

The first cloud architecture should optimize for:

- **1-10 paper deployments/accounts**

### GKE Posture

GKE is explicitly deferred. Revisit it only if the managed-first design reaches concrete scale or control-plane limits.

## Constraints

- No production implementation before **April 27, 2026**.
- The current local paper flow must keep working until the cloud control plane is ready.
- Alpaca, OpenAI, Anthropic, and Telegram credentials must never be committed to source control.
- The migrated design must preserve the current artifact semantics:
  - proposal
  - evidence
  - approval
  - order preview
  - submission
  - order status
  - reports
  - notification audit records
- The production design must support multiple paper deployments without duplicating code per account.
- The first cloud architecture should minimize ops burden and cost.
- The first cloud architecture should avoid introducing Kubernetes unless there is a concrete need.
- Existing local CLI and Docker-based development workflows should remain usable during the migration period.

## Pre-Cloud Hardening Decisions

Before hosted runtime work begins, the codebase should be hardened into a modular monolith.

Locked pre-cloud decisions:

- do not start with deployed microservices
- the first production-oriented refactor target is the paper control plane
- use one transactional control store as the future system of record
- keep the application persistence boundary DB-agnostic
- keep artifacts separate from canonical transactional state
- keep MCP out of the execution path
- preserve existing research timing and artifact contracts by default during the hardening track

The companion audit document is:

- [`docs/solid-architecture-audit.md`](docs/solid-architecture-audit.md)

## Target Architecture

### Phase Execution Contract

The control plane should converge on one internal execution contract for workflow phases:

- `decision`
- `agent_approve`
- `submit`
- `reconcile`

Each execution request should carry enough context to make the phase observable, idempotent, and environment-aware:

- `deployment_id`
- `environment`
- `execution_id`
- `correlation_id`
- `trigger_source`
- `requested_at`
- `phase`

### Internal Control-Plane API

The target runtime should expose one internal control-plane endpoint:

- `POST /internal/paper/run-phase`

This endpoint should be:

- private
- authenticated only by internal service identities
- idempotent for duplicate scheduler or task delivery
- unsuitable for public internet access

### Trigger Model

- Cloud Scheduler triggers `decision`
- Cloud Scheduler triggers `submit`
- `decision` can enqueue `agent_approve` via Cloud Tasks
- `reconcile` runs on a lower-frequency schedule or targeted retry path

The current permanent poll loop should not be the production scheduling model.

### Deployment Registry

The target design should include a first-class deployment registry with fields such as:

- `deployment_id`
- `environment`
- `enabled`
- `account_slug`
- `strategy_slug`
- `symbol`
- `execution_mode`
- `agent_backend`
- `agent_model`
- `schedule_timezone`
- `decision_time`
- `submission_time`
- `notification_profile`
- `config_version`

### Persistence Split

Persistence should be split into two concerns:

- canonical workflow state in Postgres
- immutable or reviewable artifacts in GCS

The application layer should reach that target through DB-agnostic ports first:

- `TradeRepository`
- `PhaseRunRepository`
- `PositionRepository`
- `DeploymentRepository`
- `OutboxRepository`
- `UnitOfWork`
- `ArtifactStore`

Workflow state should capture:

- proposals
- approvals
- submissions
- reconciliations
- execution history
- retry state
- deployment metadata

Artifacts should capture:

- JSON snapshots that mirror the current paper artifacts
- monthly reports
- notification audit records
- debug exports when needed

### Artifact Addressing

Artifacts should be environment-scoped, deployment-scoped, and trade-date-scoped. The exact naming can be finalized during implementation, but it should support:

- environment isolation
- deployment isolation
- easy retrieval of one trade date or one month-run

## Environment Model

### Dev

- lowest cost
- lowest retention
- experimentation allowed
- safe for schema changes and control-plane iteration

### UAT

- mirrors production topology
- paper-only validation environment
- used to validate workflow behavior before prod promotion

### Prod

- production-grade paper control plane
- strictest IAM and retention posture
- no experimentation

### Environment Isolation

Each environment should have separate:

- GCP project
- service accounts
- scheduler jobs
- task queues
- secrets
- database
- artifact bucket
- dashboards
- alerting policies

## Observability And Debugging

### Logging

Runtime services should emit structured JSON logs to stdout. At minimum, logs should carry:

- `environment`
- `deployment_id`
- `phase`
- `execution_id`
- `correlation_id`
- `proposal_id`
- `trade_date`
- `order_id`
- `provider`
- `model`
- `outcome`
- `duration_ms`

### Metrics

The observability contract should include:

- phase success and failure counts
- phase duration
- duplicate suppression counts
- queue depth and retry counts
- submission outcomes
- broker rejection counts

### Errors

- uncaught exceptions should surface in Error Reporting
- high-signal operational failures should be alertable through Cloud Monitoring

### Tracing

- request and phase traces should be instrumented with OpenTelemetry
- the default backend should be Google Cloud observability
- the instrumentation should remain portable to future vendor or OSS sinks

### Durable Debugging Surfaces

Debugging should rely on durable system records, not only container stdout:

- workflow rows in Postgres
- GCS artifact snapshots
- Cloud Logging queryable runtime logs
- execution metadata with correlation IDs

### Deferred Observability Tooling

The initial cloud design explicitly defers:

- Splunk
- Datadog
- self-hosted Loki
- self-hosted Tempo
- self-hosted Grafana

Those can be reconsidered later if scale, compliance, or cost signals justify them.

## Security And Secret Handling

Use Secret Manager for:

- Alpaca credentials
- OpenAI API key
- Anthropic API key
- Telegram bot token and chat ID

Security expectations:

- dedicated service accounts per environment
- least-privilege IAM
- no long-lived service account keys in repo or containers
- no credentials in YAML or committed config files

If GKE is adopted later, use **Workload Identity Federation for GKE** instead of key files.

## Migration Phases

### Phase 0: Planning Freeze

- complete `CLOUD_MIGRATION_PLAN.md`
- complete the SOLID-first architecture audit and hardening roadmap
- confirm architecture decisions
- confirm environment model
- confirm observability posture
- confirm what stays local until implementation starts

### Phase 1: Pre-Cloud Hardening

- add enforced static type-checking and typed service contracts
- split paper orchestration into phase-oriented application services
- introduce DB-agnostic repository and unit-of-work interfaces
- isolate broker, notifier, approval-provider, and artifact-store adapters
- add structured logging and execution context fields
- prove artifact and behavior parity before changing any persistence source of truth

### Phase 2: Runtime Foundation

- extract deployment-aware phase execution contracts
- separate local loop orchestration from one-shot phase handlers
- preserve current local CLI behavior

### Phase 3: State And Artifact Abstractions

- introduce workflow-state repository abstraction
- introduce artifact-store abstraction
- keep filesystem adapters for local compatibility

### Phase 4: Control-Plane API

- add internal authenticated phase endpoint
- add execution IDs and correlation IDs
- add idempotency guarantees for duplicate scheduler or task delivery

### Phase 5: Structured Observability

- add structured logs
- add OpenTelemetry tracing hooks
- add monitoring and error-reporting-ready event shapes

### Phase 6: GCP Deployment Scaffolding

- add infrastructure definitions for:
  - Cloud Run
  - Cloud Scheduler
  - Cloud Tasks
  - Cloud SQL
  - GCS
  - Secret Manager
  - service accounts and IAM
- start with `dev`
- then `uat`
- then `prod`

### Phase 7: Cutover Validation

- run side-by-side validation against the local design
- verify artifact parity
- verify scheduler timing and idempotency
- verify approval and submit outcomes
- verify alerting and debugging workflows

## Action Items

### Planning Deliverables

- write and approve `CLOUD_MIGRATION_PLAN.md`
- write and approve the SOLID-first architecture audit
- create an ADR for the managed-first GCP architecture
- define the `PaperDeployment` contract
- define the phase execution request and response schema
- define the canonical workflow state model
- define the GCS artifact layout
- define the structured log schema
- define the OpenTelemetry instrumentation baseline
- define the environment and project naming convention
- define the IAM and secret model
- define rollback and retry rules
- define the migration acceptance checklist
- define explicit criteria for moving to GKE

### Suggested Sequence After April 27, 2026

1. Land the pre-cloud hardening work: types, application services, persistence ports, and adapter seams.
2. Land the runtime foundation and execution contracts.
3. Land the state and artifact abstraction layer while keeping filesystem adapters.
4. Land the internal control-plane API and idempotency model.
5. Land structured observability and tracing.
6. Land dev-only GCP scaffolding.
7. Validate in UAT.
8. Promote to prod only after UAT parity is confirmed.

## GKE Decision Gate

GKE should only be reconsidered if one or more of the following become true:

- scale grows beyond the small-fleet assumption
- custom controllers or complex queue consumers become necessary
- per-deployment runtime customization exceeds Cloud Run simplicity
- multi-tenant workload isolation requires cluster-level scheduling control

Until then, prefer managed GCP services.

## Acceptance Criteria For Migration Readiness

- architecture decisions are explicit and documented
- no unresolved platform choices remain for the first implementation slice
- dev, UAT, and prod boundaries are defined
- observability and artifact strategy are explicit
- the cloud target no longer depends on permanent poll loops
- the next implementation PR can start without requiring new architecture decisions

## Appendix

### Glossary Of Runtime Phases

- `decision`: generate or refresh the proposal for the next trading date
- `agent_approve`: approve or reject a persisted proposal under agent-driven execution modes
- `submit`: reconcile the proposal against the broker account and submit or no-op
- `reconcile`: refresh broker-side terminal state after submission

### Mapping From Current Local Artifacts To Future Cloud Surfaces

| Current local surface | Future canonical home | Notes |
| --- | --- | --- |
| `artifacts/paper/inbox/*.json` | Postgres + optional debug artifact export | Inbox becomes logical workflow state, not a shared filesystem queue |
| `artifacts/paper/state/trades/<trade-date>/proposal.json` | Postgres row + GCS snapshot | Keep JSON snapshot semantics for debugging |
| `artifacts/paper/state/trades/<trade-date>/evidence.json` | Postgres row + GCS snapshot | Preserve reviewability |
| `artifacts/paper/state/trades/<trade-date>/approval.json` | Postgres row + GCS snapshot | Preserve actor and rationale history |
| `artifacts/paper/state/trades/<trade-date>/order_preview.json` | GCS artifact + execution record | Operational debug surface |
| `artifacts/paper/state/trades/<trade-date>/account_snapshot.json` | GCS artifact + execution record | Operational debug surface |
| `artifacts/paper/state/trades/<trade-date>/submission.json` | Postgres row + GCS snapshot | Canonical submission state lives in DB |
| `artifacts/paper/state/trades/<trade-date>/order_status.json` | Postgres row + GCS snapshot | Preserve reconciliation audit trail |
| `artifacts/paper/state/status.json` | control-plane status query + optional cached summary artifact | No longer the primary source of truth |
| `artifacts/paper/state/notifications/*.json` | GCS artifact + optional relational audit row | Keep debug and compliance value |
| `artifacts/paper/reports/<start>_<end>/` | GCS | Long-lived reporting artifacts |

### Known Risks And Deferred Items

- local-to-cloud parity can drift if artifact semantics are not preserved during refactoring
- idempotency must be designed before cloud scheduling is introduced
- queue retries can create duplicate submission risk if correlation and dedupe rules are weak
- environment sprawl can become costly if dev, UAT, and prod are not right-sized
- GKE remains intentionally deferred
- vendor or OSS observability expansion remains intentionally deferred

### Google Cloud References To Revisit During Implementation

- Cloud Run private services
- Cloud Scheduler for time-based triggers
- Cloud Tasks for async service-to-service execution
- Cloud SQL for workflow state
- GCS for durable artifacts
- Secret Manager for runtime secrets
- Cloud Logging, Monitoring, Error Reporting, and Trace
- Workload Identity Federation for GKE if Kubernetes is later introduced

## Assumptions

- This document lives at the repo root as `CLOUD_MIGRATION_PLAN.md`.
- This is a planning artifact only for now.
- Implementation begins no earlier than **April 27, 2026**.
- The current local paper-trading stack remains the operational system until a later migration slice is approved.
