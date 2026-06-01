# Plan

## Current State

Phases 1 through 7 are complete, and Phase 8 is active as a research-first crypto/BTC lane.

MarketLab is a public research MVP with:

- an installable Python package and packaged CLI
- a Docker-deployable MCP server surface for LLM-driven workflow access
- reproducible local and CI validation
- a Dockerized manual runner workflow
- release automation and a batched release path
- richer Phase 5 baseline, diagnostics, and scenario-pack coverage
- a Phase 7 local paper-trading MVP for a configurable daily single-ETF timing loop
- Phase 8 crypto/BTC research configs, diagnostics, and strict paper-gating methodology
- an isolated BTC paper stack that stays separated from the QQQ/VOO paper state

The project now supports a small, reviewable end-to-end research workflow rather than a local-only scaffold.

## Current MVP Capabilities

- `marketlab prepare-data --config ...`
- `marketlab backtest --config ...`
- `marketlab train-models --config ...`
- `marketlab run-experiment --config ...`
- `marketlab list-configs`
- `marketlab write-config --name ... --output ...`
- `marketlab-mcp --workspace-root ... --artifact-root ...`
- `marketlab paper-decision --config ...`
- `marketlab paper-status --config ...`
- `marketlab paper-approve --config ... --proposal-id ... --decision ... --actor ...`
- `marketlab paper-agent-approve --config ...`
- `marketlab paper-submit --config ...`
- `marketlab paper-scheduler --config ...`
- `marketlab paper-report --config ... --start ... --end ...`
- `marketlab phase8-summary --run-dir ...`
- `marketlab phase8-selection-probe --run-dir ...`
- `marketlab phase8-bull-participation --run-dir ... --config ...`
- `marketlab phase8-score-diagnostic --run-dir ...`
- `marketlab phase8-target-diagnostic --run-dir ... --config ...`
- `marketlab phase8-bull-counterfactual --run-dir ... --config ...`
- `marketlab phase8-regime-policy-sweep --run-dir ... --config ...`
- `marketlab phase8-methodology-review --run-dir ...`
- `marketlab phase8-grid-compare --runs-root ... --output ...`

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
- local file-backed paper-trading proposals, approvals, and submission state
- local Docker Compose scheduling for the daily single-ETF paper loop
- isolated BTC paper proposal, approval, and submission state under `artifacts/btc-paper` / `artifacts-btc-paper`
- artifact-only Phase 8 diagnostic commands for post-run review, including a consolidated methodology review
- artifact-only Phase 8 target diagnostics for separating bull-continuation and drawdown-defense labels
- artifact-only BTC regime-policy sweeps for testing completed-bar exposure policies before expensive retraining
- research-only BTC score-transform grids for repairing compressed bull-floor fallback scores before tier mapping
- cross-run BTC Phase 8 grid comparison for bull-upside capture, downside capture, score validity, counterfactual hypotheses, and artifact-pruning review

## Phase 4 Outcomes Worth Keeping

The most important Phase 4 outcomes were about research quality rather than headline performance:

- walk-forward guardrails, embargo handling, and explicit skipped-fold diagnostics
- ranking-aware and downside-aware evaluation beyond plain classifier metrics
- calibration and threshold diagnostics so model scores can be reviewed as confidence signals
- long-only and confidence-gated strategy modes for exposure experiments
- broader lightweight sklearn model comparisons without adding heavier dependencies

The key lesson from Phase 4 is that score quality, realized strategy outcomes, and overall research robustness are separate questions. Phase 4 improved how clearly the repo answers those questions; it did not, by itself, establish durable trading edge.

## Phase 7 Direction

Phase 7 extends the research workflow into a narrow paper-only execution loop for a configurable daily single-ETF timing strategy.

Priority direction:

- keep the live-ish path intentionally narrow and paper-only
- keep model selection explicit through tracked configs instead of auto-promoting research winners
- preserve the CLI as the execution backend and use MCP only for proposal review and approval
- keep the scheduler local and Dockerized instead of introducing a hosted control plane
- keep the autonomous agent limited to approve-or-reject of the deterministic consensus proposal

## Phase 7 Workstreams

- daily one-day timing support for the single-ETF paper path
- six-model consensus paper proposals with persisted evidence artifacts
- autonomous approval worker with `openai`, `claude`, and deterministic fallback
- Alpaca paper order submission, account snapshots, and order polling
- tracked `QQQ` plus alternate `VOO` paper configs
- month-run paper reporting against consensus, per-model paths, `buy_hold`, and `sma`
- local Docker Compose scheduler and agent worker plus the Phase 7 operations runbook

## Phase 8 Direction

Phase 8 is a research lane for crypto and BTC timing, not a production trading lane. It extends the package-first research workflow with intraday bars, indicator-stack baselines, deterministic chart-pattern detectors, direct time-series ML, allocation-utility targets, BTC tiered exposure, and strict benchmark-relative diagnostics.

Current Phase 8 rules:

- use checked-in configs for repeatable BTC and crypto experiments
- keep BTC exposure decisions limited to `0%`, `25%`, `50%`, or `100%`
- compare selected strategies against BTC buy-and-hold plus static and rebalanced BTC/cash benchmarks
- treat Phase 8 diagnostic commands as artifact review only
- use `phase8-methodology-review` to separate deployment readiness from risk allocation, score validity, bull participation, and diagnostic-only counterfactual hypotheses
- use `phase8-grid-compare`, `phase8-target-diagnostic`, `phase8-regime-policy-sweep`, and the BTC target/score pivot notes before pivoting to heavier ML models
- keep crypto/BTC paper work blocked unless the strict research gate passes

## Phase 9 Boundary

Phase 9 is an isolated BTC paper experiment, not an extension of the QQQ/VOO paper inbox. It uses `configs/experiment.btc_paper_daily.yaml`, `.env.btc-paper.example`, `docker/compose.btc-paper.yml`, `marketlab-btc-paper-mcp`, separate scheduler and agent container names, and separate artifact roots. The LLM approval role remains approve/reject only; it cannot invent trades or resize target exposure.

## Deferred

- stricter branch and ruleset hardening can wait
- wiki polish remains optional
- live-money broker support remains out of scope
