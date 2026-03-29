# Project Status

## Current Outcome

Phases 1 through 3 are complete.

MarketLab is now a public research MVP with:

- an installable Python package on PyPI
- a packaged CLI with bundled example configs
- reproducible local and CI validation
- a Dockerized manual runner workflow
- release automation and a monthly-batched release path

The repository now supports a small, reviewable end-to-end research workflow rather than a local-only scaffold.

## Delivered By Phase

### Phase 1

- canonical market panel preparation from raw market data
- baseline feature engineering and weekly target scaffolding
- baseline strategies and backtest engine
- initial metrics, plots, and markdown reporting

### Phase 2

- weekly supervised modeling dataset
- walk-forward fold generation and train/test slicing
- configurable model registry and `train-models`
- rank-based ML portfolio weights from model scores
- shared out-of-sample comparison between baseline and ML strategies
- reviewable model, fold, and strategy artifacts

### Phase 3

- GitHub Actions validation for lint, docs, package, tests, and integration
- local tox preflight gate
- offline CLI integration coverage
- Dockerized CLI image and manual GitHub Actions runner
- packaging hardening and public repository readiness
- deeper analytics artifacts and reporting
- release automation with Release PR batching and PyPI publish path
- installed-package CLI bootstrap with bundled config templates

## Current MVP Capabilities

- `marketlab prepare-data --config ...`
- `marketlab backtest --config ...`
- `marketlab train-models --config ...`
- `marketlab run-experiment --config ...`
- `marketlab list-configs`
- `marketlab write-config --name ... --output ...`

Current workflow surface:

- repo-local source execution through `python scripts/run_marketlab.py ...`
- installed-package execution through the packaged `marketlab` CLI
- baseline and ML strategy comparison on the same shared out-of-sample window
- CSV, plot, and markdown reporting artifacts
- packaged release and publish automation

## Current Boundaries

MarketLab is a research-oriented MVP, not production trading infrastructure.

Current constraints:

- classifier-oriented modeling only, with `target.type="direction"`
- small weekly ETF-focused research surface by default
- simple walk-forward fold policy
- one rank-based ML portfolio construction path
- lightweight analytics suitable for research review, not institutional reporting
- `yfinance` remains an unstable external dependency for raw market data acquisition

## Phase 4 Direction

Phase 4 should focus on evaluation rigor first, then strategy research, then lightweight model expansion.

Priority areas:

1. stronger walk-forward guardrails and skipped-fold diagnostics
2. ranking-aware model evaluation beyond ROC AUC and accuracy
3. calibration and score-quality diagnostics
4. improved strategy construction such as long-only and confidence-gated ranking variants
5. broader sklearn-only model comparison baselines

This next phase should stay lightweight on dependencies and preserve the current reproducible research posture.

## Deferred / Optional

- stricter branch/ruleset hardening can wait
- wiki polish is optional and deferred
- sponsors are already in place
- cron-based Docker automation is optional follow-on work, not an MVP requirement

## Phase 4 Issue Seeds

The next issue/project track should be created around:

1. Refresh repo plan through Phase 3 and seed the Phase 4 roadmap
2. Add walk-forward guardrails and skipped-fold diagnostics
3. Add ranking-aware model evaluation metrics and artifacts
4. Add calibration and threshold diagnostics to model reporting
5. Add long-only and confidence-gated ranking strategy modes
6. Add lightweight model expansion for comparison baselines
7. Publish a Phase 4 results review and refresh the roadmap
8. Epic issue: Phase 4: evaluation rigor and strategy research
