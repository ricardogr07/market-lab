# MarketLab Architecture

## Purpose

MarketLab is a research scaffold for reproducible market experiments over a fixed ETF universe. The current implementation covers the frozen Sprint 1 runtime path plus the first Phase 2 ML foundations: canonical market data, trailing features, weekly modeling datasets, walk-forward fold generation, a lightweight model registry, the `train-models` command, two baseline strategies, backtests, and reviewable artifacts.

This document ties the current pieces together and freezes the working rules that should guide later iterations.

## Scope

- In scope now:
  - canonical market panel preparation
  - trailing feature engineering
  - weekly modeling dataset generation
  - walk-forward fold generation
  - model registry for configured estimators
  - walk-forward `train-models` execution and artifact generation
  - `buy_hold` and `sma` baselines
  - daily backtest with turnover-based costs
  - metrics, plots, and Markdown reporting
  - fixture-backed tests and an opt-in real-data E2E runner
- Deferred to later sprints:
  - ranking strategy
  - ML integration into `run-experiment`
  - CI, Docker, and broader packaging hardening

## Canonical Local Entry Points

- Local repo execution:
  - `python scripts/run_marketlab.py run-experiment --config configs/experiment.weekly_rank.yaml`
  - `python scripts/run_marketlab.py train-models --config configs/experiment.weekly_rank.yaml`
- Real-data E2E:
  - `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1`
- Fast validation:
  - `python -m pytest -q --basetemp .pytest_tmp`

The repo uses a `src/` layout. That means `python -m marketlab.cli ...` is not a safe default for local source execution unless the environment is known to point at the current editable install. The launcher script exists to remove that ambiguity.

## System Map

```mermaid
flowchart TD
    User[User or automation] --> Launcher[scripts/run_marketlab.py]
    Launcher --> CLI[src/marketlab/cli.py]
    CLI --> Config[src/marketlab/config.py]
    CLI --> Pipeline[src/marketlab/pipeline.py]

    Pipeline --> Market[src/marketlab/data/market.py]
    Pipeline --> Panel[src/marketlab/data/panel.py]
    Pipeline --> Features[src/marketlab/features/engineering.py]
    Features --> Targets[src/marketlab/targets/weekly.py]
    Targets --> ModelingDataset[Weekly modeling dataset]
    ModelingDataset --> Evaluation[src/marketlab/evaluation/walk_forward.py]
    Pipeline --> Models[src/marketlab/models/registry.py]
    Pipeline --> BuyHold[src/marketlab/strategies/buy_hold.py]
    Pipeline --> SMA[src/marketlab/strategies/sma.py]
    Pipeline --> Engine[src/marketlab/backtest/engine.py]
    Pipeline --> Metrics[src/marketlab/backtest/metrics.py]
    Pipeline --> Markdown[src/marketlab/reports/markdown.py]
    Pipeline --> Plots[src/marketlab/reports/plots.py]

    Market --> RawCache[Raw symbol CSV cache]
    Panel --> PreparedPanel[Prepared panel CSV]
    Evaluation --> FoldDefs[folds.csv]
    Models --> ModelArtifacts[Per-fold model pickles]
    Models --> PredictionArtifacts[predictions.csv and model_metrics.csv]
    Engine --> Performance[PerformanceFrame]
    Metrics --> MetricsCsv[metrics.csv]
    Markdown --> ReportMd[report.md]
    Plots --> PlotFiles[cumulative_returns.png and drawdown.png]
```

## Baseline Runtime Flow

```mermaid
sequenceDiagram
    participant U as User
    participant L as run_marketlab.py
    participant C as cli.py
    participant CFG as config.py
    participant P as pipeline.py
    participant M as data/market.py
    participant N as data/panel.py
    participant S as strategies/*
    participant B as backtest/*
    participant R as reports/*

    U->>L: python scripts/run_marketlab.py run-experiment --config ...
    L->>C: main(argv)
    C->>CFG: load_config(path)
    CFG-->>C: ExperimentConfig
    C->>P: run_experiment(config)
    P->>P: prepare_data(config)

    alt prepared panel exists
        P->>N: load_panel_csv(path)
        N-->>P: MarketPanel
    else panel missing
        P->>M: load_symbol_frames(config)
        M-->>P: raw symbol frames
        P->>N: build_market_panel(frames)
        N-->>P: MarketPanel
    end

    P->>P: add_feature_set(panel)
    P->>S: buy_hold.generate_weights()
    P->>S: sma.generate_weights()
    S-->>P: WeightsFrame(s)
    P->>B: run_backtest(panel, weights, cost_bps)
    B-->>P: PerformanceFrame
    P->>B: compute_strategy_metrics(performance)
    B-->>P: metrics table
    P->>R: write_markdown_report(...)
    P->>R: plot_cumulative_returns(...)
    P->>R: plot_drawdown(...)
    P-->>C: ExperimentArtifacts
    C-->>U: run directory path
```

## Train-Models Flow

```mermaid
sequenceDiagram
    participant U as User
    participant L as run_marketlab.py
    participant C as cli.py
    participant CFG as config.py
    participant P as pipeline.py
    participant T as targets/weekly.py
    participant E as evaluation/walk_forward.py
    participant M as models/registry.py

    U->>L: python scripts/run_marketlab.py train-models --config ...
    L->>C: main(argv)
    C->>CFG: load_config(path)
    CFG-->>C: ExperimentConfig
    C->>P: train_models(config)
    P->>P: prepare_data(config)
    P->>T: build_weekly_modeling_dataset(panel, config)
    T-->>P: modeling dataset
    P->>E: build_walk_forward_folds(dataset, walk_forward)
    E-->>P: WalkForwardFold list

    loop configured model x fold
        P->>E: slice_fold_rows(dataset, fold)
        E-->>P: train rows, test rows
        P->>M: build_model_estimator(model_name, target_type)
        M-->>P: estimator
        P->>M: predict_direction_scores(estimator, test features)
        M-->>P: normalized scores
    end

    P-->>C: TrainModelsArtifacts
    C-->>U: run directory path
```

## Configuration Model

```mermaid
classDiagram
    class ExperimentConfig {
      +experiment_name: str
      +data: DataConfig
      +features: FeaturesConfig
      +target: TargetConfig
      +portfolio: PortfolioConfig
      +baselines: BaselinesConfig
      +models: list[ModelSpec]
      +evaluation: EvaluationConfig
      +artifacts: ArtifactsConfig
      +cache_dir: Path
      +prepared_panel_path: Path
      +output_dir: Path
      +resolve_path(value) Path
    }

    class DataConfig {
      +symbols: list[str]
      +start_date: str
      +end_date: str
      +interval: str
      +cache_dir: str
      +prepared_panel_filename: str
    }

    class FeaturesConfig {
      +return_windows: list[int]
      +ma_windows: list[int]
      +vol_windows: list[int]
      +momentum_window: int
    }

    class TargetConfig {
      +horizon_days: int
      +type: str
    }

    class PortfolioConfig {
      +ranking: RankingConfig
      +costs: CostsConfig
    }

    class RankingConfig {
      +long_n: int
      +short_n: int
      +rebalance_frequency: str
      +weighting: str
    }

    class CostsConfig {
      +bps_per_trade: float
    }

    class BaselinesConfig {
      +buy_hold: bool
      +sma: SMAConfig
    }

    class SMAConfig {
      +enabled: bool
      +fast_window: int
      +slow_window: int
    }

    class EvaluationConfig {
      +walk_forward: WalkForwardConfig
    }

    class WalkForwardConfig {
      +train_years: int
      +test_months: int
      +step_months: int
    }

    class ArtifactsConfig {
      +output_dir: str
      +save_predictions: bool
      +save_metrics_csv: bool
      +save_report_md: bool
      +save_plots: bool
    }

    class ModelSpec {
      +name: str
    }

    class ExperimentArtifacts {
      +run_dir: Path
      +panel_path: Path
      +metrics_path: Path
      +performance_path: Path
      +report_path: Path
      +cumulative_plot_path: Path
      +drawdown_plot_path: Path
    }

    class TrainModelsArtifacts {
      +run_dir: Path
      +panel_path: Path
      +folds_path: Path
      +model_manifest_path: Path
      +metrics_path: Path
      +predictions_path: Path
    }

    ExperimentConfig *-- DataConfig
    ExperimentConfig *-- FeaturesConfig
    ExperimentConfig *-- TargetConfig
    ExperimentConfig *-- PortfolioConfig
    ExperimentConfig *-- BaselinesConfig
    ExperimentConfig *-- EvaluationConfig
    ExperimentConfig *-- ArtifactsConfig
    ExperimentConfig *-- ModelSpec
    PortfolioConfig *-- RankingConfig
    PortfolioConfig *-- CostsConfig
    BaselinesConfig *-- SMAConfig
    EvaluationConfig *-- WalkForwardConfig
```

## Frozen Data Contracts

### MarketPanel

Long-format pandas frame sorted by `symbol`, then `timestamp`.

Required columns:

- `symbol`
- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `adj_close`
- `adj_factor`
- `adj_open`
- `adj_high`
- `adj_low`

### Weekly Modeling Dataset

Weekly modeling rows keyed by `symbol` and `signal_date`.

Required columns:

- `symbol`
- `signal_date`
- `effective_date`
- `target_end_date`
- feature columns derived only from signal-date information
- `forward_return`
- `target`

`target_end_date` is part of the stable contract because walk-forward training depends on it to keep label availability explicit.

### WeightsFrame

Columns:

- `strategy`
- `effective_date`
- `symbol`
- `weight`

The effective date means the next market open when the rebalance becomes active. Strategies that rebalance must emit the full symbol set so zero weights are explicit.

### PerformanceFrame

Columns:

- `date`
- `strategy`
- `gross_return`
- `net_return`
- `turnover`
- `equity`

## Module Responsibilities

### `scripts/run_marketlab.py`

- Canonical local launcher.
- Prepends `src/` to `sys.path`.
- Delegates immediately to `marketlab.cli.main`.
- Exists only to remove ambiguity from editable installs, stale installs, or PATH differences.

### `src/marketlab/cli.py`

- Parses subcommands.
- Loads the experiment config.
- Dispatches to pipeline functions.
- Prints either the prepared panel path or the run directory path.

Best practice:
- Keep this file thin.
- Do not move orchestration logic into CLI handlers.

### `src/marketlab/config.py`

- Defines the dataclass tree for all current config sections.
- Loads YAML and filters unknown keys out at section boundaries.
- Resolves relative paths from repo root when the config lives under `configs/`.

Best practice:
- Keep config loading permissive enough for staged future sections, but keep runtime behavior strict at the domain layer.

### `src/marketlab/pipeline.py`

- Orchestrates the end-to-end Sprint 1 workflow plus the Phase 2 `train-models` path.
- Decides whether to reuse the prepared panel or rebuild it.
- Runs enabled baselines for backtests and reports.
- Builds modeling datasets, walk-forward folds, trained estimators, and training artifacts for `train-models`.

Best practice:
- Put workflow coordination here, not in strategies, reports, model registry, or CLI code.

### `src/marketlab/data/market.py`

- Loads raw symbol data from cache when available.
- Downloads missing raw histories through `yfinance`.
- Flattens provider column shapes before caching them.

Best practice:
- Keep the provider seam thin and isolated here.
- Treat provider quirks as an ingestion concern, not a backtest concern.

### `src/marketlab/data/panel.py`

- Normalizes raw OHLCV frames into the canonical `MarketPanel`.
- Computes adjusted OHLC columns from `adj_close / close`.
- Validates uniqueness and sorted order.
- Cleans the cached `yfinance` header-row artifact introduced by MultiIndex downloads.

Best practice:
- Protect the panel contract aggressively.
- Make ingestion tolerant to provider shape issues, then make the normalized panel strict.

### `src/marketlab/features/engineering.py`

- Adds trailing returns, moving averages, price-to-MA ratios, MA spreads, rolling volatility, and momentum.
- Operates symbol-by-symbol on adjusted close data.

Best practice:
- Only add trailing features in Sprint 1 and early Phase 2.
- Do not introduce forward-looking features or label leakage.

### `src/marketlab/rebalance.py`

- Centralizes the shared weekly rebalance calendar.
- Resolves the last available signal date in each `W-FRI` period.
- Resolves the next effective trading date after each signal date.

Best practice:
- Keep weekly signal timing in one shared module so targets, evaluation, and strategies cannot drift.

### `src/marketlab/targets/weekly.py`

- Builds weekly modeling rows from the featured daily panel.
- Copies only signal-date feature values into the weekly sample set.
- Builds forward targets and drops rows whose label horizon is incomplete.

Best practice:
- Keep target generation label-safe and aligned to the existing Friday-close, next-open convention.
- Treat `target_end_date` as part of the modeling dataset contract, not as derived throwaway metadata.

### `src/marketlab/evaluation/walk_forward.py`

- Builds reusable walk-forward folds from the weekly modeling dataset.
- Enforces label-aware training windows by requiring `target_end_date <= test_start`.
- Produces stable fold metadata and row slices for model-training work.

Best practice:
- Keep evaluation logic independent from model wrappers and CLI orchestration.
- Slice training rows by label availability, not just by signal date.

### `src/marketlab/models/registry.py`

- Maps configured model names to concrete scikit-learn estimators.
- Keeps target-type validation at the model-entry seam.
- Produces normalized direction scores from `predict_proba` for downstream ranking work.

Best practice:
- Keep the registry lightweight and explicit.
- Avoid premature model-abstraction layers beyond construction and score extraction.

### `src/marketlab/strategies/buy_hold.py`

- Emits one equal-weight allocation on the first available date.

### `src/marketlab/strategies/sma.py`

- Builds weekly signal dates from the last close in each `W-FRI` period.
- Applies the rebalance on the next market open.
- Emits explicit zero weights when no symbol passes the rule.

Best practice:
- Strategy modules should produce weights, not portfolio returns.
- Keep strategy semantics isolated from execution semantics.

### `src/marketlab/backtest/engine.py`

- Joins weights to adjusted open and close data.
- Splits return computation into overnight and intraday components.
- Applies turnover-based trading costs.
- Produces the canonical `PerformanceFrame`.

Best practice:
- Keep execution timing explicit.
- Use adjusted open and close consistently so splits and dividends do not distort returns.

### `src/marketlab/backtest/metrics.py`

- Summarizes performance by strategy.
- Computes cumulative return, annualized return, annualized volatility, sharpe-like ratio, max drawdown, hit rate, and turnover metrics.

Best practice:
- Treat this as a reporting summary layer, not a source of trading logic.

### `src/marketlab/reports/markdown.py`

- Produces a compact Markdown report for each run.

### `src/marketlab/reports/plots.py`

- Produces cumulative equity and drawdown charts.

Best practice:
- Report modules should only render artifacts from already-computed outputs.

### `tests/`

- Unit tests protect contracts and math.
- Integration tests validate fixture-backed pipeline behavior.
- Real-data smoke tests stay opt-in because provider behavior and network access are unstable by nature.

### `.codex/skills/`

- Stores repo-local MarketLab roles and internal tooling guidance.
- Encodes conventions for planning, coding, QA, critical review, and market-domain review.

## Best Practices

- Use `python scripts/run_marketlab.py ...` for repo-local execution.
- Use `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1` for full local smoke validation.
- Keep `cli.py` thin and `pipeline.py` orchestration-focused.
- Preserve the `MarketPanel`, weekly modeling dataset, `WeightsFrame`, and `PerformanceFrame` contracts.
- Build features from trailing information only.
- Keep provider normalization inside the data layer.
- Keep strategies responsible for weights, not return calculation.
- Keep backtest timing explicit: Friday-close signal, next-open execution.
- Keep walk-forward training windows label-aware: only train on rows whose `target_end_date` is known by `test_start`.
- Treat no allocation as cash with zero return.
- Keep `train-models` focused on model artifacts only until ranking integration lands.

## Current Risks

- `yfinance` remains an unstable external dependency despite the new column-flattening and cached-header cleanup.
- `run-experiment` is still equivalent to the baseline `backtest` path.
- ranking strategy and ML portfolio integration are still not implemented.
- the model registry currently assumes classifier-style `predict_proba` outputs and `target.type="direction"`.
- metric definitions are suitable for a baseline research scaffold, not yet a full institutional evaluation stack.

## Extension Rules For Phase 2

- Add ranking behavior without breaking the existing panel, weekly modeling dataset, weights, or performance contracts.
- Keep walk-forward evaluation in the evaluation layer, not inside current baseline strategy modules.
- Reuse the fold engine and its `target_end_date <= test_start` rule rather than rebuilding train/test masks in model code.
- Keep model construction in the registry and workflow orchestration in `pipeline.py`.
- Do not redesign the current data layer just to support later model abstractions.
- Preserve the local launcher and E2E runner as the default developer entrypoints.
