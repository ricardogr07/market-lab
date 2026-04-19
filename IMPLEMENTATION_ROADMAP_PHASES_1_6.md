# Implementation Roadmap: Hardening Phases 1-6

## Purpose And Status

This document is the tracked execution plan for hardening phases 1-6 from the SOLID-first architecture audit.

It exists to turn the architecture and cloud-readiness planning into a sequence of small, reviewable PRs that keep the current MarketLab repo functional while tightening contracts, tests, and boundaries.

This roadmap is intentionally implementation-oriented:

- it names the phase goals
- it defines the PR sequence
- it records expected inputs and outputs
- it shows how to use worker roles and `/parallelize`
- it keeps the current working state of the repo as the protected baseline

Status legend:

- `not started`
- `in progress`
- `blocked`
- `done`

Current status:

- Phase 1: `not started`
- Phase 2: `not started`
- Phase 3: `not started`
- Phase 4: `not started`
- Phase 5: `not started`
- Phase 6: `not started`

Companion planning documents:

- [docs/solid-architecture-audit.md](docs/solid-architecture-audit.md)
- [CLOUD_MIGRATION_PLAN.md](CLOUD_MIGRATION_PLAN.md)
- [docs/architecture.md](docs/architecture.md)

## Global Guardrails

These rules apply to every implementation PR in phases 1-6.

- Preserve the current working paper runtime by default.
- Do not change scheduler times, CLI commands, MCP paper tools, or tracked artifact meanings unless a PR explicitly says so.
- Do not switch the default source of truth away from the current working path until parity is proven.
- Keep all required CI checks green at every phase.
- Keep PRs offline-safe by default; live Alpaca and Telegram checks remain manual only.
- Keep CLI, scheduler, agent worker, and MCP as adapters over shared application services.
- Preserve frozen research contracts by default:
  - panel semantics
  - feature timing
  - Friday-close to next-open execution assumptions
  - backtest performance outputs
  - current reviewable artifact meanings
- Treat silent changes to artifact paths, report shapes, or paper-state semantics as regressions.
- Do not introduce cloud-specific types, SDKs, or deployment assumptions into domain or application services.
- Prefer composition-first OOP, explicit interfaces, and KISS over broad abstraction layers.
- Add parity tests before changing any persistence source-of-truth behavior.
- Keep the repo mergeable after each PR. No phase may leave the default branch in a partially migrated state.

## Role Map

Use these roles consistently when planning or executing the phase PRs.

### `/orchestrator`

Primary planning role before each PR.

Responsibilities:

- slice the phase into small ordered packets
- identify the critical path
- define PR boundaries
- assign worker ownership
- set acceptance criteria
- keep scope aligned with the hardening roadmap

### `/architect`

There is no dedicated installed `architect` skill in this session.

Use the main agent as the architecture owner, grounded in:

- `docs/architecture.md`
- `docs/solid-architecture-audit.md`
- `CLOUD_MIGRATION_PLAN.md`

Every architecture decision should still go through a mandatory `/critic` pass.

### `/worker`

Implementation role for bounded change packets.

Rules:

- one worker per disjoint write scope
- no overlapping edits unless the interface owner has landed first
- keep changes small and behavior-preserving unless the PR explicitly changes behavior

### `/qa`

Validation and regression ownership.

Responsibilities:

- protect artifact and behavior parity
- expand deterministic tests when contracts are not yet provable
- verify current paper and research flows remain intact
- reject silent regressions

### `/critic`

Adversarial design review.

Responsibilities:

- challenge SRP violations
- challenge leaky abstractions
- block cloud coupling too early
- block changes that mutate frozen contracts without explicit justification
- block CLI or MCP logic drift into orchestration

### `/ci-engineer`

CI and tox ownership.

Responsibilities:

- own `tox.ini`, `pyproject.toml`, and `.github/workflows/ci.yml` changes
- keep required-check names stable
- make new gates tox-first, not YAML-first
- keep PR checks offline-safe by default

### `/parallelize`

Use only when write scopes are disjoint and the shared interface is already fixed.

Do not parallelize:

- highly coupled service extraction before the common interfaces are locked
- phases where one packet depends on unfinished public contracts from another packet
- changes that would create merge conflicts inside the same module

## Phase Overview Table

| Phase | Goal | Inputs | Outputs | PRs | Can Parallelize | Exit Gate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Add type checking and typed paper contracts | `pyproject.toml`, `tox.ini`, CI workflow, `config.py`, paper modules/tests | `mypy` gate, typed DTOs, typed ports | 2 | Yes, after `mypy` scope is fixed | `typecheck` passes and behavior is unchanged |
| 2 | Extract phase-oriented paper services | typed contracts, current paper modules, artifact semantics | shared application service layer, thin adapters | 2 | Mostly no | all adapters call the same services and artifacts are unchanged |
| 3 | Add DB-agnostic persistence ports | extracted services, current `PaperStateStore`, current paper artifacts | repositories, `UnitOfWork`, filesystem adapter, optional SQLite adapter, contract tests | 2 | Yes, after port contracts are fixed | default local behavior still works and adapter contract tests pass |
| 4 | Isolate external IO behind ports | services, persistence ports, Alpaca/Telegram/provider/artifact modules | `BrokerClient`, `NotificationSink`, `ApprovalClient`, `ArtifactStore` | 2 | Yes, after outbound interfaces are fixed | application services stop importing concrete IO adapters |
| 5 | Add structured observability | `log.py`, paper services, entry adapters | structured log envelope, execution context propagation | 1 | Usually no | structured logs exist and behavior remains unchanged |
| 6 | Define extraction readiness and add boundary guardrails | completed seams from phases 1-5, architecture docs | readiness criteria, boundary tests, explicit keep-package-first rules | 2 | Yes, after criteria are fixed | boundary rules are automated and the repo is still a modular monolith |

## Detailed Phase Plans

### Phase 1: Type System Foundation

Status: `not started`

Objective:

- add static type checking and typed paper contracts without changing runtime behavior

Inputs:

- `pyproject.toml`
- `tox.ini`
- `.github/workflows/ci.yml`
- `src/marketlab/config.py`
- current paper modules and tests

Expected outputs:

- `mypy`-based type-check env and CI job
- typed paper request and response DTOs
- typed protocol interfaces for key paper seams
- narrow initial type-check scope limited to shared config and paper-control modules

PR sequence:

1. Branch: `feat/typecheck-bootstrap`
   PR title: `feat: add mypy gate for core paper modules`
   Goal: wire type checking into tox and CI with a narrow initial scope

2. Branch: `feat/paper-typed-contracts`
   PR title: `feat: add typed paper phase contracts and protocol interfaces`
   Goal: introduce DTOs and protocols for the paper application boundary

Worker packet plan:

- main/orchestrator critical path:
  - lock the initial `mypy` scope
  - lock the failure policy
  - lock the required CI job name before parallel work starts
- worker A:
  - own typed paper DTOs and config annotations
  - write scope: config and paper contract modules
- worker B:
  - own tox and CI wiring
  - write scope: `tox.ini`, `pyproject.toml`, `.github/workflows/ci.yml`
- `/qa`:
  - review for behavior-preserving changes
  - add missing contract tests if DTOs or protocols are unproven
- `/critic`:
  - block whole-repo gating too early
  - block broad annotation churn outside the agreed narrow scope
- `/ci-engineer`:
  - verify stable required-check naming
  - keep tox as the source of truth

Validation:

- `py -3.14 -m tox -e preflight`
- targeted gates:
  - `tests/unit/test_config.py`
  - `tests/unit/test_cli.py`
  - `tests/unit/test_paper_service.py`
  - `tests/unit/test_paper_agent.py`
  - `tests/unit/test_paper_scheduler.py`
  - `tests/unit/test_paper_alpaca.py`
  - `tests/integration/test_mcp_server.py`

Exit criteria:

- the new `typecheck` gate passes
- typed contracts exist for paper phase boundaries
- no runtime behavior drift
- existing paper and research tests remain green

### Phase 2: Phase-Oriented Paper Services

Status: `not started`

Objective:

- extract paper application services while keeping CLI, scheduler, MCP, and agent behavior unchanged

Inputs:

- phase 1 typed contracts
- current `paper/service.py`, `paper/agent.py`, `paper/scheduler.py`
- current paper artifact semantics

Expected outputs:

- shared paper application service layer
- thin adapters for CLI, scheduler, agent, and MCP
- unchanged proposal, approval, submission, and reconciliation artifacts

PR sequence:

1. Branch: `refactor/decision-approval-services`
   PR title: `refactor: extract paper decision and approval services`
   Goal: move decision and approval orchestration into shared application services

2. Branch: `refactor/submit-reconcile-services`
   PR title: `refactor: extract paper submission and reconciliation services`
   Goal: move submission and reconciliation orchestration into shared application services

Worker packet plan:

- default mode:
  - single implementation owner because write scope is highly coupled
- allowed parallelization mode:
  - only after one seed commit creates the application package and fixes constructor contracts
  - worker A: decision and approval extraction
  - worker B: submit and reconcile extraction
  - write scopes must stay disjoint after the seed interface commit
- `/qa`:
  - own regression proof for artifact parity
  - verify CLI, scheduler, MCP, and agent all reuse the same services
- `/critic`:
  - block changes to trade timing
  - block orchestration drift into CLI or MCP handlers

Validation:

- `py -3.14 -m tox -e preflight`
- targeted paper regressions:
  - `tests/unit/test_paper_service.py`
  - `tests/unit/test_paper_agent.py`
  - `tests/unit/test_paper_scheduler.py`
  - `tests/unit/test_paper_notifications.py`
  - `tests/integration/test_mcp_server.py`
  - `tests/integration/test_paper_notification_flow.py`

Exit criteria:

- scheduler, CLI, MCP, and agent all call the same extracted services
- current paper artifact meanings remain unchanged
- no timing or trading behavior drift

### Phase 3: DB-Agnostic Persistence Ports

Status: `not started`

Objective:

- introduce transactional persistence seams without breaking the current working path

Inputs:

- extracted paper services
- current `PaperStateStore` behavior
- current artifact layout under `artifacts/paper/`

Expected outputs:

- repository interfaces
- `UnitOfWork`
- filesystem-backed default adapter
- optional SQLite adapter behind the same contracts
- contract tests for multiple adapters

PR sequence:

1. Branch: `refactor/paper-persistence-ports`
   PR title: `refactor: add paper repositories and unit-of-work ports`
   Goal: introduce the transactional persistence interfaces and keep the filesystem path as the default adapter

2. Branch: `feat/sqlite-paper-store-adapter`
   PR title: `feat: add sqlite paper control-store adapter behind persistence ports`
   Goal: prove the interfaces are DB-agnostic with a simple local transactional backend

Worker packet plan:

- main/orchestrator critical path:
  - lock repository interfaces
  - lock transaction rules
  - lock adapter contract tests before parallel work
- worker A:
  - filesystem-backed adapter migration
  - write scope: persistence ports plus filesystem adapter
- worker B:
  - SQLite adapter and contract-test harness
  - write scope: adapter module and parity tests
- `/qa`:
  - own parity testing against current artifact semantics
  - require transaction-boundary tests
- `/critic`:
  - block ORM leakage
  - block distributed-transaction thinking
  - block premature service-owned DB splits

Validation:

- `py -3.14 -m tox -e preflight`
- targeted paper regression set
- new repository contract tests across at least two adapter shapes
- parity tests for proposal, approval, submission, and order-status artifact behavior

Exit criteria:

- default local behavior still works
- repository contract tests pass against more than one adapter shape
- side effects remain outside DB transactions

### Phase 4: Adapter Isolation

Status: `not started`

Objective:

- move all external IO behind ports so application services are implementation-agnostic

Inputs:

- paper application services
- persistence ports
- current Alpaca, Telegram, LLM, and artifact-writing modules

Expected outputs:

- `BrokerClient`
- `NotificationSink`
- `ApprovalClient`
- `ArtifactStore`
- application services with no direct imports of concrete SDK adapters

PR sequence:

1. Branch: `refactor/broker-artifact-ports`
   PR title: `refactor: isolate paper broker and artifact store adapters`
   Goal: move broker and artifact-writing IO behind outbound ports

2. Branch: `refactor/notifier-approval-ports`
   PR title: `refactor: isolate paper notification and approval-provider adapters`
   Goal: move notification and LLM approval-provider IO behind outbound ports

Worker packet plan:

- main/orchestrator critical path:
  - lock outbound port interfaces
  - lock ownership boundaries before parallel work
- worker A:
  - broker and artifact store write scope
- worker B:
  - notifier and approval-provider write scope
- `/qa`:
  - own deterministic adapter substitution tests
  - verify no behavior drift in current paper flows
- `/critic`:
  - reject generic abstractions that erase domain meaning
  - reject adapters that still leak SDK types into application code

Validation:

- `py -3.14 -m tox -e preflight`
- targeted paper regressions
- relevant MCP unit tests
- deterministic adapter substitution tests

Exit criteria:

- business logic no longer imports concrete broker, notifier, or provider implementations
- substitutions are testable with fixtures
- current behavior remains unchanged

### Phase 5: Structured Observability

Status: `not started`

Objective:

- add structured execution context without changing business logic or cloud-coupling the repo

Inputs:

- `src/marketlab/log.py`
- paper services and entry adapters
- current local debugging surfaces

Expected outputs:

- structured log envelope
- execution ID propagation
- correlation ID propagation
- phase, deployment, trade, proposal, order, provider, outcome, and duration context fields

PR sequence:

1. Branch: `refactor/structured-paper-logging`
   PR title: `refactor: add structured execution logging for paper control plane`
   Goal: add transport-agnostic structured execution logging

Worker packet plan:

- recommended single-owner change because logging touches shared execution paths
- `/qa`:
  - verify logs do not replace artifact-based debugging
  - verify current debugging surfaces remain usable
- `/critic`:
  - block logging changes that become hidden control flow
  - block cloud-vendor assumptions in the log shape
- `/ci-engineer`:
  - only involved if CI log capture or required checks change

Validation:

- `py -3.14 -m tox -e preflight`
- targeted paper regressions
- any new log-shape tests

Exit criteria:

- structured logs exist and remain transport-agnostic
- current local debugging via artifacts and tests still works
- no behavior change

### Phase 6: Extraction Readiness Rules

Status: `not started`

Objective:

- define when the paper control plane is safe to split later and add light boundary enforcement

Inputs:

- services, ports, logging, and persistence seams from phases 1-5
- current architecture and audit docs

Expected outputs:

- extraction-readiness criteria
- architecture boundary guardrail tests
- explicit keep-package-first decision for research modules

PR sequence:

1. Branch: `feat/paper-extraction-readiness`
   PR title: `feat: define paper service-extraction readiness rules`
   Goal: define measurable criteria for future service extraction

2. Branch: `test/architecture-boundary-guardrails`
   PR title: `test: add lightweight architecture boundary guardrails`
   Goal: enforce the key modular-monolith boundaries in tests

Worker packet plan:

- main/orchestrator critical path:
  - lock the readiness checklist
  - lock prohibited dependency rules
- worker A:
  - docs and readiness criteria
- worker B:
  - `pytest`-based boundary guardrails
- `/qa`:
  - confirm the new tests are stable and useful
  - reject flaky or overly brittle boundary checks
- `/critic`:
  - challenge any readiness claim not backed by actual implemented boundaries

Validation:

- `py -3.14 -m tox -e preflight`
- targeted paper regressions
- new architecture-boundary tests

Exit criteria:

- extraction criteria are explicit
- boundary guardrails are automated
- the repo is still a modular monolith, not prematurely split services

## Merge Rules

- Branch every PR from clean `master`.
- Do not stack unrelated scope into a phase PR.
- Do not merge a persistence PR unless parity tests are in place.
- Add new required CI jobs only after the job is stable in PR CI.
- Keep job names stable once branch protection depends on them.
- Prefer 1 to 3 focused commits per PR:
  - contracts or interfaces
  - integration into current entry adapters
  - tests and docs
- If a phase requires more than two PRs, split by stable interface boundary, not by arbitrary file count.
- If the working tree already contains unmerged docs changes, land the docs baseline first or explicitly include it in the roadmap PR before phase implementation starts.

## Tracking Checklist

### Docs Baseline

- [ ] Merge the SOLID audit and cloud-migration docs baseline first, or fold that baseline into the roadmap doc PR.
- [ ] Merge this roadmap doc so phases 1-6 have a tracked execution source.

### Phase 1

- [ ] PR: `feat: add mypy gate for core paper modules`
- [ ] PR: `feat: add typed paper phase contracts and protocol interfaces`
- [ ] Confirm `typecheck` is stable in PR CI

### Phase 2

- [ ] PR: `refactor: extract paper decision and approval services`
- [ ] PR: `refactor: extract paper submission and reconciliation services`
- [ ] Confirm CLI, scheduler, MCP, and agent all reuse the same services

### Phase 3

- [ ] PR: `refactor: add paper repositories and unit-of-work ports`
- [ ] PR: `feat: add sqlite paper control-store adapter behind persistence ports`
- [ ] Confirm adapter contract tests pass and parity is proven

### Phase 4

- [ ] PR: `refactor: isolate paper broker and artifact store adapters`
- [ ] PR: `refactor: isolate paper notification and approval-provider adapters`
- [ ] Confirm business logic no longer imports concrete IO adapters

### Phase 5

- [ ] PR: `refactor: add structured execution logging for paper control plane`
- [ ] Confirm current local debugging still works

### Phase 6

- [ ] PR: `feat: define paper service-extraction readiness rules`
- [ ] PR: `test: add lightweight architecture boundary guardrails`
- [ ] Confirm the repo remains a modular monolith with automated boundary checks

### Safe To Begin Cloud Implementation

- [ ] typed contracts exist for paper phase boundaries
- [ ] phase services are shared across CLI, scheduler, agent, and MCP
- [ ] transactional persistence ports are in place
- [ ] more than one persistence adapter shape is proven by contract tests
- [ ] external IO is isolated behind ports
- [ ] structured execution context exists
- [ ] extraction readiness rules are explicit
- [ ] current paper runtime and frozen research contracts remain intact
