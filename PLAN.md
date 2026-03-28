# Phase 2 Exit Review

## Outcome

Phase 2 is complete as a working ML MVP on top of the frozen Sprint 1 scaffold.

Delivered capabilities:

- weekly supervised modeling rows from the canonical market panel
- walk-forward train/test folds
- `train-models` with per-fold estimators, predictions, and summary artifacts
- rank-based ML weights derived from model scores
- `run-experiment` with baseline and ML strategies on the same shared out-of-sample window
- reviewable reporting with strategy, model, and fold summaries

The repository now supports a small, reproducible research workflow instead of a Sprint 1-only baseline scaffold.

## Delivered PRs

- PR 1: weekly targets and modeling dataset
- PR 2: walk-forward fold engine
- PR 3: model registry and `train-models`
- PR 4: ranking strategy and ML backtest integration
- PR 5: reporting, artifact summaries, and phase wrap-up

## Validation Evidence

Phase 2 was validated with:

- unit tests for panel, features, targets, walk-forward folds, ranking, and summary logic
- integration tests for `train-models`, `run-experiment`, and baseline-only `backtest`
- an opt-in real-data E2E smoke flow using the repository launcher
- packaging and local pre-commit checks where the scope required them

The important invariant is that the baseline behavior still works and the ML path stays on the same shared OOS comparison window.

## Residual Risks

- `yfinance` remains an unstable external dependency.
- The model registry is still classifier-oriented and assumes `target.type="direction"`.
- `train-models` and `run-experiment` may leave per-fold model pickles in run directories as a side effect of reusing the training layer.
- The reporting layer is suitable for the first ML MVP, not for a full institutional analytics stack.

## Phase 3 Handoff

Phase 3 should focus on:

- CI
- Docker
- broader packaging hardening
- larger universe or multi-config expansion
- deeper analytics and reporting beyond the current CSV summaries and Markdown report
- any deployment or productization work outside the current research scaffold

Phase 2 should not be reopened unless a regression breaks one of the frozen contracts or the shared OOS comparison behavior.
