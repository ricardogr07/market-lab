# Plan

## Current State

Phases 1 through 5 are complete.

MarketLab is a public research MVP with:

- an installable Python package and packaged CLI
- a Docker-deployable MCP server surface for LLM-driven workflow access
- reproducible local and CI validation
- a Dockerized manual runner workflow
- release automation and a batched release path
- richer Phase 5 baseline, diagnostics, and scenario-pack coverage

The project now supports a small, reviewable end-to-end research workflow rather than a local-only scaffold.

## Current MVP Capabilities

- `marketlab prepare-data --config ...`
- `marketlab backtest --config ...`
- `marketlab train-models --config ...`
- `marketlab run-experiment --config ...`
- `marketlab list-configs`
- `marketlab write-config --name ... --output ...`
- `marketlab-mcp --workspace-root ... --artifact-root ...`

Current workflow surface:

- repo-local execution through `python scripts/run_marketlab.py ...`
- installed-package execution through the packaged `marketlab` CLI
- generic stdio MCP execution through the packaged `marketlab-mcp` server
- baseline and ML strategy comparison on the same shared out-of-sample window
- walk-forward fold diagnostics for accepted and skipped candidates
- ranking-aware, downside-aware, calibration, and threshold review artifacts
- long-short, long-only, and confidence-gated cash-underfilled execution modes
- lightweight sklearn comparison baseline
- CSV, plot, and Markdown reporting artifacts
- sandboxed config authoring, async job control, and run-artifact inspection for MCP clients

## Phase 4 Outcomes Worth Keeping

The most important Phase 4 outcomes were about research quality rather than headline performance:

- walk-forward guardrails, embargo handling, and explicit skipped-fold diagnostics
- ranking-aware and downside-aware evaluation beyond plain classifier metrics
- calibration and threshold diagnostics so model scores can be reviewed as confidence signals
- long-only and confidence-gated strategy modes for exposure experiments
- broader lightweight sklearn model comparisons without adding heavier dependencies

The key lesson from Phase 4 is that score quality, realized strategy outcomes, and overall research robustness are separate questions. Phase 4 improved how clearly the repo answers those questions; it did not, by itself, establish durable trading edge.

## Phase 6 Direction

Phase 6 should productize the current workflow for LLM-driven use through a Docker-friendly MCP server.

Priority direction:

- keep the protocol surface tools-first and generic-client-friendly
- make config authoring and execution safe through workspace sandboxing and confirmation gates
- preserve the installed CLI as the execution backend so MCP behavior matches the packaged product
- keep Docker as the deployment wrapper instead of introducing a second runtime architecture

## Phase 6 Candidate Workstreams

- MCP server scaffold and workspace sandboxing
- template-driven config authoring and validation tools
- async job planning, queued execution, and log tails
- artifact inspection, plot retrieval, and compact run comparison
- Docker sidecar docs and required MCP CI

## Deferred

- stricter branch and ruleset hardening can wait
- wiki polish remains optional
- cron-based Docker automation is still optional and not part of the MVP
