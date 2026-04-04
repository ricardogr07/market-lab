# Plan

## Current State

Phases 1 through 4 are complete.

MarketLab is a public research MVP with:

- an installable Python package and packaged CLI
- reproducible local and CI validation
- a Dockerized manual runner workflow
- release automation and a batched release path
- Phase 4 evaluation, diagnostics, and strategy-surface improvements

The project now supports a small, reviewable end-to-end research workflow rather than a local-only scaffold.

## Current MVP Capabilities

- `marketlab prepare-data --config ...`
- `marketlab backtest --config ...`
- `marketlab train-models --config ...`
- `marketlab run-experiment --config ...`
- `marketlab list-configs`
- `marketlab write-config --name ... --output ...`

Current workflow surface:

- repo-local execution through `python scripts/run_marketlab.py ...`
- installed-package execution through the packaged `marketlab` CLI
- baseline and ML strategy comparison on the same shared out-of-sample window
- walk-forward fold diagnostics for accepted and skipped candidates
- ranking-aware, downside-aware, calibration, and threshold review artifacts
- long-short, long-only, and confidence-gated cash-underfilled execution modes
- lightweight sklearn comparison baseline
- CSV, plot, and Markdown reporting artifacts

## Phase 4 Outcomes Worth Keeping

The most important Phase 4 outcomes were about research quality rather than headline performance:

- walk-forward guardrails, embargo handling, and explicit skipped-fold diagnostics
- ranking-aware and downside-aware evaluation beyond plain classifier metrics
- calibration and threshold diagnostics so model scores can be reviewed as confidence signals
- long-only and confidence-gated strategy modes for exposure experiments
- broader lightweight sklearn model comparisons without adding heavier dependencies

The key lesson from Phase 4 is that score quality, realized strategy outcomes, and overall research robustness are separate questions. Phase 4 improved how clearly the repo answers those questions; it did not, by itself, establish durable trading edge.

## Phase 5 Direction

Phase 5 should focus on richer portfolio risk controls and exposure-aware comparisons.

Priority direction:

- add risk-aware portfolio controls on top of the current strategy modes
- compare strategies with explicit attention to exposure differences
- keep the current lightweight dependency posture
- avoid starting with another round of model expansion

## Phase 5 Candidate Workstreams

- risk controls in portfolio construction rather than only score gating
- exposure-aware reporting and comparisons across long-short, long-only, and cash-heavy variants
- turnover and cost sensitivity as first-class parts of strategy interpretation
- scenario configs that compare the same strategy ideas under a common risk framing

## Deferred

- stricter branch and ruleset hardening can wait
- wiki polish remains optional
- cron-based Docker automation is still optional and not part of the MVP
