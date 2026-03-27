# MarketLab Sprint 1 Exit Review

## Status

Sprint 1 is ready to freeze as a working research scaffold.

## Outcome

The repo now supports the full baseline workflow for a fixed ETF universe:

- prepare or reuse a canonical market panel
- engineer trailing features
- run `buy_hold` and `sma`
- backtest both strategies with turnover-based costs
- emit metrics, plots, and a Markdown report
- validate the path with fixture tests and an opt-in real-data E2E runner

The canonical local command is now:

```bash
python scripts/run_marketlab.py run-experiment --config configs/experiment.weekly_rank.yaml
```

This avoids accidentally executing an older installed copy of `marketlab` in a `src/` layout.

## Evidence

- Local test suite: `9 passed, 1 skipped`
- Real-data E2E: `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1`
- Real-data ingestion hardened against current `yfinance` MultiIndex column output and cached header-row artifacts

## Completed Scope

- package-first repo scaffold under `src/marketlab`
- YAML-driven `ExperimentConfig` dataclass tree
- canonical `MarketPanel`, `WeightsFrame`, and `PerformanceFrame` contracts
- raw data cache plus prepared panel cache
- baseline feature engineering
- baseline strategy generation for `buy_hold` and `sma`
- daily backtest engine and summary metrics
- Markdown and plot reporting
- repo-local skills for worker, QA, critic, orchestrator, financial review, shared context, and pre-commit validation
- architecture and execution documentation in [ARCHITECTURE.md](ARCHITECTURE.md)

## Deferred Work

- `train-models` implementation
- ranking strategy
- model wrappers
- walk-forward evaluation
- CI and Docker
- broader experiment/config variants

## Residual Risks

- `yfinance` remains an external dependency with unstable response shapes
- `run-experiment` is still equivalent to the Sprint 1 baseline `backtest` path
- metrics are suitable for baseline research, not yet a full institutional evaluation layer

## Sprint 2 Entry Criteria

- preserve current data and backtest contracts
- keep local execution on `python scripts/run_marketlab.py ...`
- add model and ranking logic without moving orchestration out of `pipeline.py`
- introduce walk-forward evaluation without leaking future information into features or signal generation
