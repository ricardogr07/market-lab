# Service-Extraction Readiness Rules

This checklist is the gate for future MarketLab service extraction work. Use it
before moving code into an application service, introducing a port or adapter, or
splitting a package-first module behind a new boundary.

The default posture is conservative: keep simple code simple, preserve existing
contracts, and require evidence before adding a boundary.

Readiness-only PRs must not change runtime behavior. They should document or
test the decision gate without moving code, changing command behavior, or
redefining artifact contracts.

## Keep Package-First

Keep a module package-first when it is mostly pure or near-pure logic, has stable
function inputs and outputs, and does not own external side effects.

This is the default for research/runtime modules such as feature engineering,
target construction, backtest math, strategy weight generation, metrics, and
deterministic report assembly.

Package-first code should still improve through typed stage contracts, clearer
artifact boundaries, and focused tests. It should not be wrapped in a service
only because the module is large or likely to change.

## Extract Service

Extract an application service only when orchestration has a stable business
phase, multiple inbound callers, and enough side-effect coordination that a plain
function boundary no longer makes the behavior clear.

The paper control plane is the primary extraction candidate. Future paper
services should keep the phase language explicit:

- decision
- agent approval
- submission
- reconciliation

Before extracting, the PR must show these are true:

- CLI, scheduler, agent, MCP, or other inbound callers can share the same
  one-shot contract.
- Typed request and response objects are defined or preserved.
- Current paper proposal, approval, submission, reconciliation, and status
  semantics remain unchanged.
- Existing artifact paths and reviewable payload meanings are protected by
  parity tests.
- Side effects are named and kept out of pure domain logic.

## Introduce Port Or Adapter

Introduce a port or adapter when business logic depends on an external system,
provider SDK, persistence backend, artifact store, notification channel, or
runtime transport that must be replaceable in tests or future deployments.

Good port candidates include broker access, approval providers, notification
sinks, artifact stores, repositories, units of work, and runtime-facing side
effect boundaries.

Before adding a port, the PR must show:

- the concrete dependency is outside the domain or application core;
- at least one deterministic fake, fixture, or alternate adapter can prove the
  substitution;
- the port preserves domain vocabulary instead of erasing it behind generic IO;
- transactions do not stay open across broker calls, LLM/provider calls, or
  notification delivery.

## Preserve Artifact Parity

Artifact compatibility is part of the public development contract for the current
local workflow. A service extraction or adapter split must not silently change
artifact paths, payload meanings, report shapes, paper-state semantics, or the
review surface used by CLI, scheduler, agent, and MCP workflows.

Before changing any source-of-truth boundary, the PR must keep artifact parity
tests in place and add persistence-agnostic tests for the new contract. JSON
artifacts can remain the debugging and audit surface even when transactional
state moves behind repositories.

## Defer Extraction

Defer extraction when the proposed boundary is only organizational, when the
substitution point is hypothetical, or when tests would only assert wiring instead
of behavior.

Do not:

- force research math, feature engineering, target creation, strategy logic, or
  deterministic reporting into service boundaries without a concrete
  orchestration problem;
- add a port around pure functions or stable in-process helpers;
- move business rules into MCP tools, CLI handlers, scheduler loops, or agent
  loops;
- make MCP the execution backend for a future production runtime;
- change frozen research contracts without a dedicated PR that proves the need.

If a module has unclear responsibilities but no stable service contract yet,
first document the current responsibility split, add typed contracts or tests at
the existing boundary, and defer the extraction decision.
