# P9-07 Worker Plan: Hosted Execution Context And Deployment Registry

- Branch: `feature/phase-9-hosted-execution-registry`
- Pull request: `feat(phase9): add hosted paper execution registry`
- Dependency: P9-01 canonical Phase 9 roadmap

P9-07 adds typed hosted execution metadata and local deployment registry
adapters for the existing QQQ paper control plane. It prepares the paper
phases for later hosted jobs while preserving the current local CLI, Docker,
MCP, scheduler, approval worker, broker, proposal, approval, submission,
reconciliation, notification, and report semantics.

P9-07 is a local contract packet only. It does not add PostgreSQL, Blob
Storage, Service Bus, QQQ Terraform, Azure SDK calls, managed identity wiring,
broker behavior changes, strategy changes, config changes, or deployment
automation. P9-08 owns PostgreSQL migrations. P9-09 owns Blob, outbox, and
Service Bus adapters. P9-10 owns QQQ Azure Terraform and rollout wiring.

## Checkpoint: 2026-06-20

Current branch: `feature/phase-9-hosted-execution-registry`.

Implemented so far:

- added this P9-07 worker plan and linked it from `docs/PLAN.md`,
  `docs/index.md`, `mkdocs.yml`, and `tests/unit/test_phase9_plan_docs.py`
- added hosted execution contracts in `marketlab.paper.contracts`:
  `PaperHostedExecutionContext`, `PaperDeploymentRecord`,
  `PaperPhaseRunRecord`, `PaperDeploymentRegistry`, hosted environment and
  phase literals, exact metadata field validation, and conflict error type
- added filesystem and SQLite local registry adapters beside the existing paper
  persistence adapters
- filesystem registry writes under `paper_state_dir/deployments/` and
  `paper_state_dir/phase-runs/`
- SQLite registry adds local P9-07 registry tables only:
  `paper_deployment_records` and `paper_phase_run_records`
- duplicate hosted `idempotency_key` values with identical metadata are
  accepted; conflicting metadata raises before paper phase side effects
- threaded optional hosted metadata through `run_paper_decision`,
  `decide_paper_proposal`, `run_paper_submit`, and
  `reconcile_latest_submission_status`
- threaded hosted metadata through scheduler and agent paths, deriving child
  phase contexts from the supplied hosted job context
- added CLI flags and `MARKETLAB_*` environment defaults for hosted metadata,
  with explicit flags taking precedence
- expanded unit coverage for contracts, adapter behavior, service guard
  ordering, scheduler derivation, agent derivation, CLI precedence, partial
  metadata rejection, and duplicate idempotency conflicts

Validation completed before stopping:

```text
python -m pytest -q tests/unit/test_phase9_plan_docs.py tests/unit/test_paper_contracts.py tests/unit/test_paper_persistence.py tests/unit/test_paper_service.py tests/unit/test_paper_scheduler.py tests/unit/test_paper_agent.py tests/unit/test_cli.py
# 111 passed

python -m pytest -q tests/unit
# 685 passed

python -m ruff check .
# All checks passed

python -m mypy src/marketlab/log.py src/marketlab/paper/observability.py src/marketlab/paper/contracts.py src/marketlab/paper/persistence/filesystem.py src/marketlab/paper/persistence/sqlite.py src/marketlab/paper/service.py src/marketlab/paper/agent.py src/marketlab/paper/scheduler.py src/marketlab/cli.py
# Success: no issues found in 9 source files

python -m mkdocs build --strict
# passed

git diff --check
# passed, with only Git line-ending warnings
```

Preflight status:

- `py -3.14 -m tox -e preflight` passed `lint`, `docs`, `typecheck`,
  `package`, and `py312`
- the first `integration` lane failed because `.tox/integration` was missing
  `pyvenv.cfg` after a Windows virtualenv/cache issue
- the tox integration environment was repaired with repo-local temp and
  `VIRTUALENV_OVERRIDE_APP_DATA`
- a follow-up integration run then failed during pytest basetemp cleanup with
  `WinError 5` against `artifacts/tmp/pytest-integration`
- the stale pytest temp directory was removed after path verification
- the final integration rerun was intentionally interrupted by the user before
  completion

Resume checklist:

- verify no stray Python/tox validation process is running
- decide whether to remove the generated `.virtualenv-app-data/` directory
  before publication
- rerun the integration lane with repo-local temp settings:

```text
$env:TMP='C:\git\market-lab\artifacts\tmp'
$env:TEMP='C:\git\market-lab\artifacts\tmp'
$env:TMPDIR='C:\git\market-lab\artifacts\tmp'
$env:VIRTUALENV_OVERRIDE_APP_DATA='C:\git\market-lab\.virtualenv-app-data'
py -3.14 -m tox -e integration
```

- if integration passes, rerun full `py -3.14 -m tox -e preflight`
- inspect `git status --short` and remove generated validation artifacts from
  the working tree before committing
- then review the diff for scope boundaries and, if still clean, continue with
  the planned small-commit publication path

## Hosted Metadata Contract

Every hosted phase uses exactly these metadata fields:

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

Supported environments are `dev`, `uat`, and `paper-prod`. Supported phases
are `decision`, `agent_approve`, `submit`, and `reconcile`.

The public typed contracts are:

- `PaperHostedExecutionContext`
- `PaperDeploymentRecord`
- `PaperPhaseRunRecord`
- `PaperDeploymentRegistry`

The existing structured log phases remain `paper-decision`, `paper-approve`,
`paper-submit`, and `paper-submit-reconcile`; the hosted `phase` field is the
registry contract and not a replacement for existing log names.

## Local Registry Adapters

The filesystem adapter writes under the existing paper state root:

```text
artifacts/paper/state/deployments/
artifacts/paper/state/phase-runs/
```

The SQLite adapter adds local P9-07 registry tables beside the existing paper
state tables. It does not mirror registry state to JSON and does not introduce
PostgreSQL compatibility code.

Duplicate `idempotency_key` values with identical metadata are accepted as
idempotent repeats. Duplicate `idempotency_key` values with conflicting
metadata raise before a paper phase creates providers, calls brokers, sends
notifications, or mutates proposal, approval, submission, or reconciliation
state.

## Service Wiring

The four phase entrypoints accept optional hosted metadata:

```text
run_paper_decision
decide_paper_proposal
run_paper_submit
reconcile_latest_submission_status
```

When hosted metadata is missing, local behavior and existing log context
remain unchanged. When hosted metadata is present, the service records the
phase run before phase side effects and uses the hosted `execution_id` and
`correlation_id` in structured logs.

The scheduler derives child phase contexts for `decision`, `submit`, and
`reconcile` from the scheduler job metadata. The agent approval worker derives
one `agent_approve` context per proposal from the worker job metadata. Callers
derive from the supplied hosted metadata rather than inventing unrelated phase
identifiers.

## CLI Contract

Paper phase-capable CLI commands accept explicit hosted metadata flags and
`MARKETLAB_*` environment defaults. Explicit flags win over environment
values.

```text
--deployment-id        MARKETLAB_DEPLOYMENT_ID
--environment          MARKETLAB_ENVIRONMENT
--execution-id         MARKETLAB_EXECUTION_ID
--correlation-id       MARKETLAB_CORRELATION_ID
--idempotency-key      MARKETLAB_IDEMPOTENCY_KEY
--trigger-source       MARKETLAB_TRIGGER_SOURCE
--requested-at         MARKETLAB_REQUESTED_AT
--config-version       MARKETLAB_CONFIG_VERSION
--image-digest         MARKETLAB_IMAGE_DIGEST
                       MARKETLAB_PHASE
```

`MARKETLAB_PHASE` is validated for direct phase commands. The scheduler may
accept it as seed metadata because scheduler iterations can derive multiple
phase contexts from one hosted job.

Partial hosted metadata is rejected. If no hosted metadata is supplied, the CLI
uses the existing local execution path.

## Acceptance

P9-07 is complete when:

- docs and navigation link this worker plan
- hosted contracts reject unsupported phases, environments, missing fields,
  extra fields, and malformed `requested_at` values
- filesystem and SQLite registry adapters share duplicate and conflict tests
- service wrappers reject conflicting hosted metadata before phase side effects
- scheduler and agent worker derive phase contexts from hosted job metadata
- CLI flags override `MARKETLAB_*` values, partial metadata fails, and duplicate
  idempotency conflicts surface through CLI execution
- no trading semantics, strategy parameters, broker behavior, or approval
  decisions change

## Validation

```text
python -m pytest -q tests/unit/test_phase9_plan_docs.py tests/unit/test_paper_contracts.py tests/unit/test_paper_persistence.py tests/unit/test_paper_service.py tests/unit/test_paper_scheduler.py tests/unit/test_paper_agent.py tests/unit/test_cli.py
python -m pytest -q tests/unit
python -m ruff check .
python -m mypy src/marketlab/log.py src/marketlab/paper/observability.py src/marketlab/paper/contracts.py src/marketlab/paper/persistence/filesystem.py src/marketlab/paper/persistence/sqlite.py src/marketlab/paper/service.py src/marketlab/paper/agent.py src/marketlab/paper/scheduler.py src/marketlab/cli.py
python -m mkdocs build --strict
py -3.14 -m tox -e preflight
git diff --check
```
