# MarketLab Phase 2 Execution Plan

## Objective

Phase 2 delivers the first ML MVP on top of the frozen Phase 1 scaffold.

Success means MarketLab can:

- build weekly supervised modeling rows from the canonical market panel
- generate walk-forward train/test folds
- train configured models through `train-models`
- convert model scores into ranked portfolio weights
- run baseline and ML strategies together through `run-experiment`
- produce artifacts and reports that are reviewable by humans

Phase 2 starts from the current Phase 1 baseline already on `master`. Do not create retroactive PRs for Phase 1. All new work lands through small feature branches and small PRs.

## Frozen Assumptions

- Preserve the Phase 1 `MarketPanel`, `WeightsFrame`, and `PerformanceFrame` contracts.
- Keep local execution on `python scripts/run_marketlab.py ...`.
- Keep orchestration in `pipeline.py`; do not move workflow logic into `cli.py`.
- Weekly rebalance-date snapshots are the only ML sample cadence in Phase 2.
- `backtest` remains the simpler baseline path.
- `train-models` and `run-experiment` become real Phase 2 commands.
- The current fixed ETF universe remains in force unless a later phase expands it explicitly.
- Phase 3 still owns CI, Docker, and broader productization.

## PR-First Workflow

### Working Rules

1. Start every new scope from updated `master`.
2. Create a feature branch named `feature/phase-2-<scope>`.
3. Make small intentional commits on that branch.
4. Run local validation before any push.
5. Ask the user before pushing the branch.
6. After push, create exactly one PR for that branch.
7. Use the GitHub connector to inspect commit and PR state before and after PR creation.
8. Wait for user review and approval.
9. After the user confirms the PR was merged:
   - `git checkout master`
   - `git pull --ff-only origin master`
   - create the next feature branch from refreshed `master`

### Constraints

- Do not push without user approval.
- Do not stack unrelated scopes into one PR.
- Do not start the next PR branch from a stale feature branch.
- Do not open synthetic PRs for work that is already on `master`.

### Tool Split

- GitHub connector:
  - inspect repo state
  - inspect commits
  - inspect PR status, comments, and metadata
  - create PRs after a branch is pushed
- Local `git`:
  - create branches
  - stage files
  - create commits
  - push only after user approval
  - switch back to `master` after merge
- Local validation:
  - use `marketlab-pre-commit-checks` before asking to push
  - use tox, uv, or E2E only when the PR scope justifies it

### Commit Policy Inside Every PR

- Every PR should contain 2 to 4 commits.
- Every commit should represent one logical step.
- Commit subjects should use short imperative form.
- Test commits are allowed when they prove the immediately preceding behavior.
- Do not bundle docs into implementation commits unless the docs are required to explain a new interface.
- If a PR grows beyond one feature sentence, split it into a new PR.

## Phase 2 PR Roadmap

### PR 1: Weekly targets and modeling dataset

- Branch:
  - `feature/phase-2-weekly-targets`
- Goal:
  - turn the Phase 1 panel and features into weekly supervised modeling rows aligned to rebalance dates
- Scope:
  - weekly rebalance snapshot builder
  - target generation for next-horizon outcome
  - dataset contract keyed by symbol and signal date
  - tests for date alignment and no-lookahead behavior
- Suggested commit sequence:
  - `feat: add weekly sample builder for rebalance dates`
  - `feat: add target generation for weekly modeling rows`
  - `test: cover weekly samples and target alignment`
- Subagents:
  - `marketlab-worker`: implement weekly sample builder and target generation
  - `marketlab-qa`: write deterministic fixture tests for sample and target alignment
  - `marketlab-critic`: review leakage and contract boundaries before merge
  - `marketlab-financial-expert`: verify Friday-close to next-open semantics remain coherent
- Acceptance criteria:
  - modeling rows are generated only on weekly rebalance dates
  - targets use only future returns after the signal date
  - tests prove no overlap between feature timestamp and target horizon

### PR 2: Walk-forward fold engine

- Branch:
  - `feature/phase-2-walk-forward`
- Goal:
  - activate `evaluation.walk_forward` and produce reusable fold definitions
- Scope:
  - fold generator based on train-years, test-months, and step-months
  - fold metadata structure or artifact
  - tests for boundaries, ordering, and non-overlap
- Suggested commit sequence:
  - `feat: add walk-forward fold generator`
  - `feat: add fold metadata for experiment runs`
  - `test: cover walk-forward boundary logic`
- Subagents:
  - `marketlab-worker`: implement fold generator
  - `marketlab-qa`: build fold fixtures and boundary tests
  - `marketlab-critic`: review temporal leakage and boundary correctness
- Acceptance criteria:
  - folds are ordered and reproducible
  - test windows do not overlap
  - train/test windows respect configured durations
  - fold generation is independent from model-specific logic

### PR 3: Model registry and `train-models`

- Branch:
  - `feature/phase-2-train-models`
- Goal:
  - turn `train-models` from a stub into a real command
- Scope:
  - lightweight model wrapper registry for configured model names
  - normalized training and prediction interface
  - `train-models` pipeline integration
  - model artifact manifest and fold prediction outputs
- Suggested commit sequence:
  - `feat: add model registry for configured estimators`
  - `feat: implement walk-forward training pipeline`
  - `feat: activate train-models command`
  - `test: cover model registry and training artifacts`
- Subagents:
  - `marketlab-worker` worker-1: model wrapper registry
  - `marketlab-worker` worker-2: training pipeline and CLI integration
  - `marketlab-qa`: artifact and command-level tests
  - `marketlab-critic`: review abstraction size and sprint fit
- Acceptance criteria:
  - `train-models --config ...` exits successfully
  - each configured model produces fold-level predictions
  - outputs are normalized enough for downstream ranking
  - tests cover successful and failing paths

### PR 4: Ranking strategy and ML backtest integration

- Branch:
  - `feature/phase-2-ranking-integration`
- Goal:
  - turn model scores into weights and run ML strategies alongside baselines
- Scope:
  - ranking strategy using configured `long_n`, `short_n`, and equal weighting
  - score-to-`WeightsFrame` conversion
  - ML strategy backtest integration
  - `run-experiment` orchestration upgrade from baseline-only to baseline-plus-ML
- Suggested commit sequence:
  - `feat: add ranking strategy from model scores`
  - `feat: backtest ranked model portfolios`
  - `feat: integrate ml path into run-experiment`
  - `test: cover ranking weights and ml experiment flow`
- Subagents:
  - `marketlab-worker` worker-1: ranking strategy and weights output
  - `marketlab-worker` worker-2: pipeline integration and experiment orchestration
  - `marketlab-qa`: integration tests for ML plus baseline outputs
  - `marketlab-financial-expert`: verify long/short semantics, turnover, and cost behavior
  - `marketlab-critic`: verify no contract breakage in `WeightsFrame` and `PerformanceFrame`
- Acceptance criteria:
  - model scores produce valid long/short weights at weekly rebalance dates
  - ML strategies appear in the same outputs as baselines
  - `run-experiment` remains the top-level orchestration path
  - Phase 1 baseline behavior still passes unchanged

### PR 5: Phase 2 reporting and experiment artifacts

- Branch:
  - `feature/phase-2-reporting`
- Goal:
  - make Phase 2 outputs reviewable without digging through raw prediction files
- Scope:
  - fold summary metrics
  - model comparison outputs
  - report updates for baseline vs ML comparison
  - artifact documentation refresh
- Suggested commit sequence:
  - `feat: add fold and model summary outputs`
  - `feat: extend experiment report for baseline and ml comparison`
  - `docs: refresh phase 2 usage and artifact documentation`
- Subagents:
  - `marketlab-worker`: reporting outputs and report composition
  - `marketlab-qa`: verify artifact presence and report completeness
  - `marketlab-financial-expert`: review metric naming and interpretation
  - `marketlab-critic`: verify reporting changes do not hide weak assumptions
- Acceptance criteria:
  - reports summarize baseline and ML strategies together
  - fold and model summaries are persisted and understandable
  - documentation matches actual Phase 2 command behavior

## Validation And Merge Protocol

### Before Asking To Push

- Run `python -m pytest -q --basetemp .pytest_tmp`.
- Run PR-specific tests for the new scope.
- Run `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1 -SkipPytest` only when the PR touches runtime orchestration or real-data behavior.
- Summarize:
  - branch name
  - commits on branch
  - tests run
  - PR title and PR body draft

### After User Approves Push

- Push the branch.
- Create the PR.
- Use the GitHub connector to inspect the PR state.
- Wait for user review and approval.

### After User Confirms Merge

- `git checkout master`
- `git pull --ff-only origin master`
- verify local `master` matches remote
- create the next feature branch from refreshed `master`

## Test Plan

### PR 1

- unit tests for weekly sample rows and target alignment
- explicit no-lookahead cases

### PR 2

- unit tests for fold generation and date boundaries
- edge case coverage for insufficient history

### PR 3

- unit tests for model registry selection
- integration test for `train-models` artifact creation

### PR 4

- unit tests for ranking weight generation
- integration test for `run-experiment` including ML strategies
- regression test that Phase 1 baselines still pass

### PR 5

- integration test for final artifact bundle and report contents
- metric and report sanity checks for baseline-vs-ML comparison

### Workflow Checks

- before each push, run the minimum relevant `marketlab-pre-commit-checks` path
- after each merge, verify local `master` fast-forwards cleanly from remote

## Deferred Beyond Phase 2

- CI
- Docker
- broader productization
- larger universe or multi-config expansion
- daily-sample ML workflows
- deeper evaluation/reporting layers beyond the first ML MVP
