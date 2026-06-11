# Phase 9 Roadmap

## Outcome

Phase 9 is a three-track program:

1. Run a locked, signals-only BTC shadow confirmation and produce a formal
   evidence decision.
2. Establish the reusable Azure foundation for scheduled MarketLab workloads.
3. Move the existing QQQ paper operation from local Docker scheduling to
   Azure without changing its strategy or approval semantics.

Phase 9 ends only when the BTC evidence lane is operating under its frozen
contract, QQQ paper operations are running on Azure with a proven rollback
path, and the shared Azure foundation is documented and reproducible.

This remains a paper-only phase. Live-money endpoints, automatic promotion of
research winners, VOO migration, and broker-facing BTC migration are out of
scope.

## Current State

Phases 1 through 7 are complete. Phase 8 retained one historical BTC winner as
research evidence for Phase 9:

```text
config: configs/experiment.btc_phase8_guarded_gate_bull_risk_off_override_partial_support.yaml
run: artifacts/runs/btc_phase8_guarded_gate_bull_risk_off_override_partial_support/20260602T081225Z
historical code lock: ce01124
```

The inspected run beat BTC buy-and-hold by `+2.2940` at `35 bps` and `+0.8094`
at `50 bps`, but it still relied on fallback selections and failed important
signal-validity and bull-participation diagnostics. It is eligible only for
forward shadow confirmation, not paper or live promotion.

The active operational surface already includes:

- `marketlab phase8-summary --run-dir ...`
- `marketlab phase8-methodology-review --run-dir ...`
- `marketlab paper-decision --config ...`
- `marketlab paper-status --config ...`
- `marketlab paper-approve --config ...`
- `marketlab paper-agent-approve --config ...`
- `marketlab paper-submit --config ...`
- `marketlab paper-scheduler --config ...`
- `marketlab paper-report --config ...`
- phase-oriented decision, approval, submission, and reconciliation services
- filesystem and SQLite persistence implementations behind repository and
  unit-of-work contracts
- structured paper logs and isolated broker, notification, approval-provider,
  and artifact-store adapters
- local QQQ/VOO paper operations through `docker/compose.paper.yml`
- an isolated BTC paper stack through
  `configs/experiment.btc_paper_daily.yaml`,
  `docker/compose.btc-paper.yml`, and `marketlab-btc-paper-mcp`

The existing BTC paper stack is a separate `4h` direction-target consensus
strategy. It is not the Phase 8 winner mirror and cannot supply Phase 9
graduation evidence.

## Locked Phase 9 Decisions

- Azure is the only hosted platform planned for Phase 9.
- Terraform is the infrastructure-as-code standard.
- Azure Container Apps Jobs run finite, one-shot workloads; no always-on paper
  scheduler container is required in Azure.
- Azure Container Registry stores images tagged by immutable Git commit SHA.
- Azure Key Vault and managed identities provide runtime secret access.
- Azure Database for PostgreSQL Flexible Server is canonical QQQ workflow
  state after cutover.
- Azure Blob Storage holds reviewable artifacts and reports; versioning and
  retention protect evidence.
- Azure Service Bus carries asynchronous phase and notification work.
- Log Analytics, Application Insights, and Azure Monitor provide logs,
  traces, dashboards, and alerts.
- GitHub Actions uses OpenID Connect for Azure access; no long-lived Azure
  credentials are stored in GitHub.
- MCP remains an inspection and approval surface and is not part of runtime
  orchestration.
- One Azure subscription is sufficient for Phase 9, with separate resource
  groups, identities, secrets, storage namespaces, and databases for `shadow`,
  `dev`, `uat`, and `paper-prod`.
- The default region is `eastus2`; changing it is an infrastructure decision,
  not a runtime behavior change.
- Public ingress is disabled. Private networking is required before QQQ UAT
  can use broker credentials or production-paper state.

Azure Container Apps scheduled jobs evaluate cron in UTC. QQQ therefore uses a
frequent UTC scheduler tick and keeps `America/New_York` due-time enforcement
inside MarketLab. This preserves daylight-saving behavior and the existing
`16:10` decision and `19:05` submission contract.

## Track A: BTC Evidence Gate

### Purpose

Confirm the frozen Phase 8 winner on future, uninspected data. The lane records
signals and evidence only. It cannot request approval, call a broker, submit an
order, reconcile an account, or promote the candidate.

### Confirmation Window

- Protocol start: `June 3, 2026`
- Protocol end: `June 2, 2027`
- Earliest final labeled evaluation: `June 16, 2027`
- Target maturity lag: `14` daily bars

Any run missed before launch or during an outage must be recorded explicitly.
It must not be silently backfilled with data unavailable at the original
decision time. Any change to features, targets, models, parameters, selection,
or fallback behavior creates a new candidate and restarts its confirmation
clock.

### Planned Command Contract

```text
phase9-shadow-decision --config configs/experiment.btc_phase9_shadow_daily.yaml
phase9-shadow-scheduler --config configs/experiment.btc_phase9_shadow_daily.yaml --once
phase9-shadow-status --config configs/experiment.btc_phase9_shadow_daily.yaml
phase9-shadow-report --config configs/experiment.btc_phase9_shadow_daily.yaml [--as-of YYYY-MM-DD]
```

The new config must be a paper-disabled mirror of the locked winner and must
fail closed when its deterministic behavior hash differs from the approved
contract.

### Artifact Contract

```text
artifacts/phase9-shadow/
  decisions/<effective-date>.json
  reports/monthly/<year-month>/
  reports/final/
  state/status.json
```

Each decision record includes the candidate and config hashes, code lock,
decision timestamp, signal date, next-effective date, matured-label cutoff,
selection source, fallback mode, target allocation, input fingerprint, output
fingerprint, and explicit success, skipped, or failed status.

Identical repeated writes are idempotent. A conflicting write for an existing
effective date fails without modifying the original record. Reports cannot
rewrite decision evidence.

### Azure Runtime

The shadow lane runs once daily at `01:15 UTC` through a disabled-by-default
Container Apps scheduled job under `infra/azure/phase9-shadow`. Azure Files may
preserve the live path-oriented contract during the first implementation, but
the durable graduation archive is versioned Blob Storage with a `450` day
retention policy. Immutability remains unlocked until archive and restore
behavior is proven because locking the policy is irreversible.

The job cannot be enabled until the command, frozen config, behavior lock,
append-only conflict handling, missing-run recording, immutable image tagging,
archive copy, restore test, and failed-run alert all pass.

### Graduation Criteria

All criteria are required before any winner-mirror paper review:

- complete the full window and maturity lag with an immutable journal
- account explicitly for every scheduled day, including missed or skipped runs
- preserve the locked behavior hash for the entire lane
- pass the unchanged strict research gate
- keep active return versus buy-and-hold positive at `35 bps` and `50 bps`
- pass `signal_validity_gate`, including positive score-to-utility correlation
- pass `bull_participation_gate`, including positive gate-bull active return
  and no missed positive underexposed benchmark return
- record zero `best_active_fallback` selections
- record zero deterministic regime-policy fallback selections

Monthly reports are progress evidence only. The final report records one of
`continue-to-paper-review`, `restart-with-new-candidate`, or `stop` and cannot
automatically enable trading.

## Track B: Azure Foundation

### Infrastructure Roots

Phase 9 introduces these Terraform roots:

```text
infra/azure/bootstrap/
infra/azure/phase9-shadow/
infra/azure/qqq-paper/
```

`bootstrap` creates the versioned Blob backend for Terraform state.
`phase9-shadow` creates the isolated BTC evidence workload. `qqq-paper` creates
the shared paper-control-plane resources and per-environment QQQ deployments.
All executable jobs remain disabled by default until their activation gates
pass.

### Shared Runtime Contract

Add typed execution metadata to every hosted phase:

```text
deployment_id
environment
phase
execution_id
correlation_id
idempotency_key
trigger_source
requested_at
config_version
image_digest
```

The supported phases remain `decision`, `agent_approve`, `submit`, and
`reconcile`. CLI, scheduler, agent, and future hosted jobs call the same
application services. Business rules do not depend on Azure SDKs, Terraform,
Container Apps, Service Bus, or MCP.

### Canonical State And Artifacts

PostgreSQL stores deployments, phase runs, proposals, approvals, submission
attempts, broker order state, account snapshots, positions, idempotency keys,
notification events, and outbox records. Schema changes use reviewed,
forward-only migrations with a tested backup and restore path.

Blob Storage preserves the existing JSON meanings for proposal, evidence,
approval, order preview, account snapshot, submission, order status,
notification audit, and reports. Artifact addresses are environment,
deployment, and trade-date scoped. PostgreSQL is canonical state; Blob
artifacts remain the review and debugging contract.

No database transaction remains open across Alpaca, OpenAI, Anthropic,
Telegram, Service Bus, or Blob calls. Persist intent first, perform the side
effect, then persist the result. Outbox delivery and phase execution are
idempotent and safe under duplicate messages.

### Security And Delivery

- use one managed identity per workload and environment
- use Key Vault references for Alpaca, provider, and Telegram secrets
- reject non-paper Alpaca endpoints before any broker-facing phase starts
- use private endpoints and VNet-integrated Container Apps for QQQ UAT and
  `paper-prod`
- build once, scan once, and promote the same immutable image digest
- run Terraform format, initialization, validation, and plan in pull requests
- require an environment approval for Terraform apply and application rollout
- keep Terraform state, `.tfvars`, backend files, and secrets untracked

## Track C: QQQ Operations On Azure

### Preserved Strategy Contract

The migration uses `configs/experiment.qqq_paper_daily.yaml` unchanged unless a
separate research decision approves a strategy change. Preserve:

- symbol `QQQ`
- daily bars and one-day direction target
- six-model deterministic `4-of-6` consensus
- long-or-cash behavior with no shorts
- `agent_approval` execution mode and deterministic fallback
- `America/New_York` scheduling
- `16:10` decision time and `19:05` submission time
- Alpaca paper endpoints only
- existing proposal, approval, submission, reconciliation, report, and
  notification meanings

VOO and both BTC operational stacks stay local during Phase 9.

### Azure Job Shape

- A scheduled dispatcher runs every five minutes in UTC and invokes the
  one-shot scheduler contract. MarketLab decides whether a New York phase is
  due and suppresses duplicate execution through the idempotency key.
- A Service Bus event triggers the one-shot approval job after a proposal is
  committed.
- Submission remains time-gated and cannot run without the required approval.
- Reconciliation runs on the scheduler tick while an order is non-terminal and
  on a lower-frequency recovery schedule afterward.
- Notifications are delivered from persisted outbox records and cannot change
  phase outcomes.
- No public HTTP control-plane endpoint is required for the Phase 9 cutover.

### Migration And Cutover

1. **Dev proof:** deploy with broker submission disabled; prove database,
   artifacts, queues, identities, alerts, and manual job execution.
2. **State import:** add an idempotent importer for current QQQ file artifacts,
   preserving identifiers, timestamps, payloads, and checksums.
3. **UAT shadow:** run Azure against live market data for at least `10`
   consecutive NYSE trading days with submission disabled while local QQQ
   remains authoritative.
4. **Parity review:** compare decision inputs, votes, proposal, approval,
   target position, order preview, reports, notifications, and reconciliation
   expectations. Every difference must be explained and classified.
5. **Failure drills:** prove duplicate delivery, provider timeout, broker
   timeout, rejected order, partial fill, stale data, missing bar, queue retry,
   dead-letter recovery, database restore, and Blob restore behavior.
6. **Cutover:** stop the local scheduler and agent, verify no in-flight
   non-terminal order or unresolved proposal, import the final state delta,
   then enable Azure `paper-prod` jobs.
7. **Observation:** operate Azure for `10` additional NYSE trading days while
   retaining a disabled local rollback runner that can use the same PostgreSQL
   and Blob adapters.
8. **Closeout:** archive the old local state, document the cutover, and keep
   local Docker available for development rather than scheduling production
   paper phases.

Cutover requires zero unexplained proposal differences, zero unsafe-order
differences, zero duplicate broker submissions, successful backup and restore,
working alerts, and a rehearsed rollback.

## Ordered Delivery Roadmap

| Packet | Deliverable | Depends on | Acceptance gate |
| --- | --- | --- | --- |
| P9-01 | Canonical roadmap, Azure ADR, and protected docs tests | None | Legacy cloud plan removed; BTC, Azure, and QQQ boundaries explicit |
| [P9-02](phase9/P9-02-WORKER-PLAN.md) | Terraform bootstrap and validation workflow | P9-01 | Remote state backend reproducible; CI validates all roots without applying |
| [P9-03](phase9/P9-03-WORKER-PLAN.md) | Frozen BTC shadow config and behavior lock | P9-01 | Any drift fails closed; paper remains disabled |
| P9-04 | Shadow decision service and append-only journal | P9-03 | Label-safe, idempotent, conflict-preserving decisions |
| P9-05 | Shadow scheduler, status, monthly, and final reports | P9-04 | Missing and failed runs explicit; reports cannot promote |
| P9-06 | Azure shadow infrastructure and launch gate | P9-02, P9-05 | Disabled job validates; archive, restore, and alert tests pass |
| P9-07 | Hosted execution context and deployment registry | P9-01 | All four paper phases share typed, idempotent metadata |
| P9-08 | PostgreSQL repositories and migrations | P9-07 | Filesystem, SQLite, and PostgreSQL contract suites pass |
| P9-09 | Blob artifact store, outbox, and Service Bus adapters | P9-08 | Artifact parity and duplicate-delivery tests pass |
| P9-10 | QQQ Azure Terraform and deployment pipeline | P9-02, P9-09 | Dev environment deploys disabled-by-default jobs using managed identity |
| P9-11 | QQQ state importer and operational runbooks | P9-08, P9-09 | Repeatable import, checksum report, backup, restore, and rollback documented |
| P9-12 | QQQ dev and UAT shadow validation | P9-10, P9-11 | Ten-trading-day parity report and failure drills accepted |
| P9-13 | QQQ paper-prod cutover | P9-12 | Local scheduler stopped; final delta imported; Azure jobs enabled safely |
| P9-14 | QQQ post-cutover closeout | P9-13 | Ten-trading-day observation passes; rollback remains rehearsed |
| P9-15 | BTC final evidence decision | P9-06 and maturity window | Final report records continue, restart, or stop |

Each packet should be a small reviewable pull request. Infrastructure and
runtime changes must not be mixed with strategy changes.

## Validation Requirements

Every implementation packet runs the relevant subset of the canonical tox
lanes, with full `preflight` before publication. Phase 9 additionally requires:

- contract tests across filesystem, SQLite, and PostgreSQL repositories
- deterministic fake adapters for Blob, Service Bus, Key Vault, and broker IO
- artifact parity tests for all existing QQQ JSON meanings
- execution-id and idempotency tests under duplicate and reordered delivery
- migration tests from real-shaped local paper fixtures
- Terraform format and validation checks for every root
- container smoke tests using immutable image tags
- opt-in Azure integration tests in a disposable dev resource group
- security tests that reject live broker endpoints and missing deployment
  isolation
- UAT parity and failure-drill reports stored as release evidence

## Definition Of Done

Phase 9 is complete when:

- the BTC shadow lane is running under the frozen contract and its final
  evidence decision is recorded after `June 16, 2027`
- QQQ paper decision, approval, submission, reconciliation, notifications, and
  reporting run from Azure
- PostgreSQL is canonical QQQ workflow state and Blob Storage contains the
  reviewable artifact mirror
- no local always-on scheduler is required for QQQ operations
- infrastructure can be recreated from Terraform and deployed through an
  approved GitHub Actions path
- backup, restore, incident response, dead-letter recovery, credential
  rotation, rollback, and manual phase execution are documented and tested
- local Docker, CLI, and MCP development workflows still work

## Deferred

- live-money broker support
- VOO Azure migration
- broker-facing BTC Azure migration
- BTC paper promotion before the final shadow decision
- AKS or Kubernetes
- multi-tenant public control-plane APIs
- automatic research-winner promotion
- strategy or model changes hidden inside infrastructure work

## Azure References

- [Azure Container Apps Jobs](https://learn.microsoft.com/en-us/azure/container-apps/jobs)
- [Azure Container Apps scaling](https://learn.microsoft.com/en-us/azure/container-apps/scale-app)
- [Azure Database for PostgreSQL Flexible Server](https://learn.microsoft.com/en-us/azure/postgresql/flexible-server/quickstart-create-server)
- [Microsoft Entra authentication for PostgreSQL](https://learn.microsoft.com/en-us/azure/postgresql/flexible-server/security-entra-concepts)
- [Azure Blob versioning](https://learn.microsoft.com/en-us/azure/storage/blobs/versioning-overview)
- [Azure Blob immutability policies](https://learn.microsoft.com/en-us/azure/storage/blobs/immutable-policy-configure-version-scope)
