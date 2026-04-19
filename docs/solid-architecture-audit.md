# SOLID-First Architecture Audit

## Purpose

This document is the pre-cloud architecture and technical-debt audit for MarketLab.

It is intentionally critical. The goal is not to defend the current implementation, but to identify what must stay simple, what must be modularized, and what must be separated behind stable interfaces before any hosted control plane or transactional persistence work begins.

This audit complements:

- [Architecture](architecture.md) for the current system map
- [Phase 7 Paper Trading](paper-trading.md) for the local paper runtime
- `CLOUD_MIGRATION_PLAN.md` at the repo root for the later hosted deployment path

## Locked Design Posture

- Use OOP with composition-first boundaries.
- Apply SOLID as a review gate, especially SRP and dependency inversion at IO boundaries.
- Bias toward KISS. Keep pure local logic simple and explicit.
- Apply DRY only when duplication is structural, not merely similar-looking.
- Keep the application layer open to new implementations through ports and adapters.
- Do not couple business logic to cloud vendors, database clients, container runtimes, or MCP transports.
- Do not jump directly to microservices. First build a modular monolith with explicit boundaries, typed contracts, and transactional seams.
- Preserve the frozen research contracts unless a later PR proves a concrete need to change them:
  - panel semantics
  - feature timing
  - Friday-close to next-open execution assumptions
  - backtest performance outputs
  - current reviewable artifact meanings

## Repo-Wide Review Rubric

Every module group should be reviewed with the same questions:

- What is its actual responsibility today?
- Does it have more than one reason to change?
- Which external dependencies or side effects does it own?
- Which inputs and outputs should be explicit typed contracts?
- Where is it coupled to filesystem paths, SDKs, CLI behavior, environment variables, or container assumptions?
- Which duplication is worth unifying, and which duplication is clearer left alone?
- Which abstractions are missing at external boundaries?
- Which abstractions would be premature or harmful?

## Audit Findings By Module Group

### Shared Foundation

Targets:

- `src/marketlab/config.py`
- `src/marketlab/cli.py`
- `src/marketlab/log.py`
- shared utility surfaces

Current assessment:

- `cli.py` is already close to the right shape. It should remain a transport layer, not an orchestration layer.
- `config.py` is explicit and readable, but it is becoming a central dependency for both research and runtime concerns.
- `log.py` is too thin for the next stage. Basic stdlib logging is fine locally, but it does not provide execution context, structured fields, or future observability seams.

Keep:

- dataclass-based config objects
- thin CLI command handlers
- simple local logging bootstrap for dev ergonomics

Harden:

- add a structured logging envelope without pushing logging concerns into domain logic
- separate config loading from higher-level runtime validation rules where cross-field checks become non-trivial
- keep shared helpers boring and explicit; do not hide side effects behind convenience wrappers

### Research Runtime

Targets:

- `pipeline`
- `data`
- `features`
- `targets`
- `evaluation`
- `models`
- `strategies`
- `backtest`
- `reports`

Current assessment:

- the research stack is mostly package-first and should stay that way
- `data`, `features`, `targets`, `strategies`, and most of `backtest` are good candidates to remain pure or near-pure modules
- `pipeline.py` is the main orchestration hotspot in the research path; it mixes workflow coordination with artifact persistence and reporting assembly
- reporting is valuable, but artifact-writing concerns should not keep spreading into orchestration surfaces

Keep:

- package-first research execution
- pure function style for feature, target, strategy, and backtest logic
- explicit artifact generation for reviewable outputs
- current market-data and performance contracts unless a dedicated PR proves a required contract change

Harden:

- split workflow coordination from artifact persistence in the research path
- introduce typed input/output contracts at stage boundaries where DataFrame-based handoffs are currently implicit
- add type-checking coverage to catch hidden coupling between modeling, reporting, and artifact assembly

Do not do:

- do not force the research stack into service boundaries early
- do not add abstraction layers around pure math, feature engineering, or deterministic reporting logic without a concrete substitution need

### Paper Control Plane

Targets:

- `src/marketlab/paper/service.py`
- `src/marketlab/paper/agent.py`
- `src/marketlab/paper/scheduler.py`
- `src/marketlab/paper/alpaca.py`
- `src/marketlab/paper/notifications.py`
- `src/marketlab/paper/report.py`

Current assessment:

- this is the highest-risk part of the codebase for the next phase
- `paper/service.py` is the main orchestration hotspot and currently carries too many responsibilities
- `paper/agent.py` mixes worker control, provider policy, prompting, fallback behavior, and approval orchestration
- scheduler and agent loops are acceptable as local adapters, but they should not own business rules
- the current filesystem-first state model is workable for the laptop flow, but it is too tightly coupled to orchestration logic for a future transactional runtime

Keep:

- explicit phase language: `decision`, `agent_approve`, `submit`, `reconcile`
- reviewable artifacts and deterministic fallback semantics
- adapters for local scheduler and local agent loops

Harden:

- split phase orchestration into application services:
  - `DecisionService`
  - `ApprovalService`
  - `SubmissionService`
  - `ReconciliationService`
- move Alpaca, Telegram, artifact writing, and provider calls behind explicit interfaces
- make scheduler, CLI, MCP, and agent worker thin entry adapters over the same one-shot service contracts
- define typed request/response objects for each phase instead of relying on raw dicts and ad hoc file payloads
- preserve current paper-state artifact semantics while transactional persistence is being introduced, so parity can be proven before any source-of-truth transition

### MCP And Ops Tooling

Targets:

- `src/marketlab/mcp/server.py`
- `src/marketlab/mcp/jobs.py`
- MCP tool modules
- workspace sandbox

Current assessment:

- MCP is correctly positioned as an ops and review surface, not the execution backend
- `mcp/jobs.py` is a local in-process queue for repo workflows; it should not become the production control-plane model
- MCP tools should stay thin and call application services rather than growing their own runtime logic

Keep:

- sandboxed MCP tool surface
- queued local workflow control for research and local ops
- artifact inspection and paper review tools

Harden:

- ensure MCP approval and status tools depend on the same application services as CLI and local workers
- keep MCP-side abstractions separate from future cloud scheduling abstractions

Do not do:

- do not let MCP become a hidden dependency of the future runtime
- do not move business rules into tool handlers

### Tests And Tooling

Targets:

- `tox.ini`
- `pyproject.toml`
- unit and integration suites

Current assessment:

- the repo has meaningful behavioral coverage, especially around the paper path
- static typing is not enforced today; that is a real gap for the next phase
- many tests still prove behavior through filesystem artifacts, which is useful, but not enough for a future DB-backed control plane

Keep:

- behavioral tests that validate current paper artifacts and local workflow outcomes
- docs, packaging, and integration gates in tox

Harden:

- add a real type-check gate to tox and CI
- add contract tests for phase services, repositories, and adapters
- keep artifact parity tests while adding persistence-agnostic service tests
- prepare for multi-adapter testing of the same repository contracts

Required QA posture:

- treat silent changes to artifact paths, report shapes, or paper-state semantics as regressions
- keep deterministic fixture coverage for pure logic and transition rules
- add parity checks before any persistence source-of-truth change
- require one clear acceptance checklist per hardening packet, not only broad architectural intent

## Target Modular Monolith Shape

The next architecture step is a modular monolith, not deployed microservices.

Application layer:

- `DecisionService`
- `ApprovalService`
- `SubmissionService`
- `ReconciliationService`

Outbound ports:

- `TradeRepository`
- `PhaseRunRepository`
- `PositionRepository`
- `DeploymentRepository`
- `OutboxRepository`
- `UnitOfWork`
- `ArtifactStore`
- `BrokerClient`
- `NotificationSink`
- `ApprovalClient`

Inbound adapters:

- CLI commands
- scheduler loop
- agent worker loop
- MCP paper tools

Pure domain logic should stay independent of IO:

- consensus evaluation
- signal validation
- position sizing policy
- state-transition rules
- idempotency and retry rules

## Transactional Persistence Target

The future source of truth is one transactional control store.

Application-layer persistence must remain DB-agnostic:

- local development can start with a simple adapter such as SQLite
- production can later use a Postgres-compatible adapter
- business logic must not depend on ORM sessions, SQL dialects, or cloud-managed database types

Canonical records to preserve later:

- deployments
- proposals
- evidence
- approvals
- submission attempts
- broker order state
- positions and account snapshots
- phase executions and idempotency keys
- notification events and outbox items

Artifact snapshots remain separate from transactional state:

- reviewable JSON artifacts still matter
- artifacts are a debugging and audit surface
- artifacts do not replace the canonical transactional record

### Transaction Rules

- Persist internal state changes atomically inside one application-level transaction.
- Do not hold a transaction open across:
  - broker calls
  - LLM/provider calls
  - Telegram delivery
- Use persisted phase-run state plus an outbox or event handoff for side effects.
- Treat idempotency as a first-class persistence concern, not as a best-effort filesystem convention.

## Ordered Hardening Packets

1. `feat(types): add enforced type-checking and typed service contracts`
2. `refactor(paper): split orchestration into phase-oriented application services`
3. `refactor(persistence): introduce DB-agnostic repositories and unit-of-work`
4. `refactor(adapters): isolate broker, notifier, LLM, and artifact store behind ports`
5. `refactor(obs): add structured logging and execution context`
6. `feat(readiness): define service-extraction readiness rules`

### Packet-Level Validation Expectations

`feat(types)` must prove:

- type-check gate is wired into tox and CI
- typed contracts cover config, phase requests/responses, and adapter interfaces
- no behavioral drift in existing paper and research flows

`refactor(paper)` must prove:

- scheduler, CLI, MCP, and agent paths still call the same business rules
- current paper proposal, approval, submission, and reconciliation artifacts remain reviewable
- no change to current trade semantics or timing assumptions

`refactor(persistence)` must prove:

- repository contracts pass the same behavioral tests across at least one local adapter and one future-ready adapter shape
- transaction boundaries are explicit and tested
- side effects remain outside DB transactions

`refactor(adapters)` must prove:

- business logic no longer imports concrete broker, notifier, or provider SDKs directly
- adapter substitutions can be tested with deterministic fixtures
- no new abstraction erases domain meaning

`refactor(obs)` must prove:

- structured logs carry execution context without polluting domain logic
- existing local debugging remains usable
- log changes do not become hidden control flow

`feat(readiness)` must prove:

- extraction criteria are explicit
- the paper control plane can be split later without reopening core contracts
- research/runtime modules still remain package-first unless a later decision changes that

## Module Review Sequence

Use this order so the highest-coupling seams are reviewed first:

1. shared foundation
2. research orchestration hotspots
3. paper control plane
4. MCP and ops tooling
5. tests, typing, and validation tooling

For each review packet, produce:

- current responsibility summary
- SRP and coupling findings
- missing contracts and adapter seams
- concrete keep/split/defer recommendations
- required tests and regression coverage

## Acceptance Criteria

- every major module group has a completed audit entry
- the paper control plane has a documented modular-monolith target shape
- the persistence boundary is explicitly DB-agnostic
- transaction rules are documented
- MCP is explicitly preserved as ops-only
- frozen research and artifact contracts are explicitly called out as protected by default
- future hardening packets have concrete validation obligations, not only architectural goals
- the next implementation PR can start without reopening architecture decisions
