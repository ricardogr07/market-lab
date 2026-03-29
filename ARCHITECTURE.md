# MarketLab Architecture

## Purpose

MarketLab is a package-first research toolkit for reproducible market experiments over a fixed ETF universe. The current implementation includes canonical market data, trailing features, weekly modeling datasets, walk-forward fold generation, a lightweight model registry, the `train-models` command, a ranking strategy, two baseline strategies, unified `run-experiment` baseline-plus-ML comparison, fold and model summaries, strategy analytics artifacts, backtests, and reviewable artifacts.

This document ties the current pieces together and records the working rules that should guide later iterations.

## Scope

- In scope now:
  - canonical market panel preparation
  - trailing feature engineering
  - weekly modeling dataset generation
  - walk-forward fold generation
  - model registry for configured estimators
  - walk-forward `train-models` execution and artifact generation
  - score-to-weight ranking strategy for ML portfolios
  - fold and model summary artifacts
  - `buy_hold` and `sma` baselines
  - unified `run-experiment` comparison across baselines and ML strategies on a shared OOS window
  - daily backtest with turnover-based costs
  - metrics, strategy analytics CSVs, plots, and Markdown reporting
  - required PR CI for lint, docs, packaging, unit tests, and offline integration tests
  - Docker packaging for the installed CLI plus a manual GitHub Actions Docker runner
  - fixture-backed tests and a real-data E2E runner that validates baseline, training, experiment, and analytics artifact sets
- Deferred to later sprints:
  - scheduled Docker automation for recurring runs
  - release automation and broader packaging hardening

## Canonical Local Entry Points

- Local repo execution:
  - `python scripts/run_marketlab.py run-experiment --config configs/experiment.weekly_rank.yaml`
  - `python scripts/run_marketlab.py train-models --config configs/experiment.weekly_rank.yaml`
- Local Docker validation:
  - `docker build -t marketlab-cli .`
  - `docker run --rm marketlab-cli --help`
- Manual GitHub Actions Docker automation:
  - `.github/workflows/docker-runner.yml`
  - `workflow_dispatch` only
  - defaults to `backtest` on `configs/experiment.weekly_rank.smoke.yaml`
- Real-data E2E:
  - `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1`
  - covers `prepare-data`, `backtest`, `train-models`, and `run-experiment` on the smoke config
- Fast validation:
  - `python -m pytest -q --basetemp .pytest_tmp`

The repo uses a `src/` layout. That means `python -m marketlab.cli ...` is not a safe default for local source execution unless the environment is known to point at the current editable install. The launcher script exists to remove that ambiguity.

The Docker image deliberately uses the installed `marketlab` console script instead of the repo-local launcher. That split keeps local development pointed at the source tree while the container validates the packaged CLI path.

## Validation Flow

```mermaid
flowchart TD
    Runner[scripts/run-e2e.ps1] --> Pytest[optional pytest gate]
    Runner --> Launcher[scripts/run_marketlab.py]
    Launcher --> Prepare[prepare-data]
    Launcher --> Backtest[backtest]
    Launcher --> Train[train-models]
    Launcher --> Experiment[run-experiment]

    Prepare --> Panel[prepared panel cache]
    Backtest --> BaselineArtifacts[metrics.csv performance.csv analytics report.md plots]
    Train --> TrainingArtifacts[folds.csv manifest metrics predictions summaries models/]
    Experiment --> ExperimentArtifacts[metrics.csv performance.csv analytics report.md plots summaries optional models/]
```

## Automation Split

```mermaid
flowchart TD
    PRCI[Required PR CI] --> Tox[lint docs package py312 integration]
    Manual[Docker Runner workflow_dispatch] --> Build[Dockerfile build]
    Build --> InstalledCli[installed marketlab entrypoint]
    InstalledCli --> SmokeConfig[historical smoke config]
    InstalledCli --> DockerArtifacts[/app/artifacts upload]
```

The required CI path stays offline and deterministic through tox. The Docker runner is separate, manual, and allowed to exercise the historical real-data smoke config without becoming a required PR gate.

## System Map

```mermaid
flowchart TD
    User[User or automation] --> Launcher[scripts/run_marketlab.py]
    User --> DockerWorkflow[.github/workflows/docker-runner.yml]
    Launcher --> CLI[src/marketlab/cli.py]
    DockerWorkflow --> DockerImage[Dockerfile]
    DockerImage --> InstalledCLI[marketlab console script]
    InstalledCLI --> CLI
    CLI --> Config[src/marketlab/config.py]
    CLI --> Pipeline[src/marketlab/pipeline.py]

    Pipeline --> Market[src/marketlab/data/market.py]
    Pipeline --> Panel[src/marketlab/data/panel.py]
    Pipeline --> Features[src/marketlab/features/engineering.py]
    Features --> Targets[src/marketlab/targets/weekly.py]
    Targets --> ModelingDataset[Weekly modeling dataset]
    ModelingDataset --> Evaluation[src/marketlab/evaluation/walk_forward.py]
    Pipeline --> Models[src/marketlab/models/training.py]
    Models --> Predictions[Fold predictions]
    Predictions --> Ranking[src/marketlab/strategies/ranking.py]
    Pipeline --> Summary[src/marketlab/reports/summary.py]
    Pipeline --> Analytics[src/marketlab/reports/analytics.py]
    Pipeline --> BuyHold[src/marketlab/strategies/buy_hold.py]
    Pipeline --> SMA[src/marketlab/strategies/sma.py]
    BuyHold --> Engine[src/marketlab/backtest/engine.py]
    SMA --> Engine
    Ranking --> Engine
    Engine --> Performance[PerformanceFrame]
    Pipeline --> Metrics[src/marketlab/backtest/metrics.py]
    Summary --> ModelSummaryCsv[model_summary.csv]
    Summary --> FoldSummaryCsv[fold_summary.csv]
    Analytics --> StrategySummaryCsv[strategy_summary.csv]
    Analytics --> MonthlyReturnsCsv[monthly_returns.csv]
    Analytics --> TurnoverCostsCsv[turnover_costs.csv]
    Pipeline --> Markdown[src/marketlab/reports/markdown.py]
    Pipeline --> Plots[src/marketlab/reports/plots.py]

    Market --> RawCache[Raw symbol CSV cache]
    Panel --> PreparedPanel[Prepared panel CSV]
    Evaluation --> FoldDefs[folds.csv]
    Models --> ModelArtifacts[Per-fold model pickles]
    Models --> TrainArtifacts[model_manifest.csv, model_metrics.csv, predictions.csv]
    Metrics --> MetricsCsv[metrics.csv]
    Markdown --> ReportMd[report.md]
    Plots --> PlotFiles[cumulative_returns.png drawdown.png and turnover.png]
```

## Run-Experiment Flow

```mermaid
sequenceDiagram
    participant U as User
    participant L as run_marketlab.py
    participant C as cli.py
    participant CFG as config.py
    participant P as pipeline.py
    participant M as data/market.py
    participant N as data/panel.py
    participant T as targets/weekly.py
    participant E as evaluation/walk_forward.py
    participant RANK as strategies/ranking.py
    participant SUM as reports/summary.py
    participant AN as reports/analytics.py
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

    P->>P: run baseline strategies
    P->>B: run_backtest(...) per baseline strategy
    B-->>P: baseline PerformanceFrame rows
    P->>T: build_weekly_modeling_dataset(panel, config)
    T-->>P: modeling dataset
    P->>E: build_walk_forward_folds(dataset, walk_forward)
    E-->>P: WalkForwardFold list
    P->>P: train_direction_models_on_folds(...)
    P->>RANK: generate_weights(...) per model
    RANK-->>P: WeightsFrame per ml_* strategy
    P->>B: run_backtest(...) per ml_* strategy
    B-->>P: ml PerformanceFrame rows
    P->>P: slice to shared OOS daily window
    P->>P: rebase equity per strategy
    P->>B: compute_strategy_metrics(sliced performance)
    B-->>P: metrics table
    P->>SUM: build_model_summary(...) and build_fold_summary(...)
    SUM-->>P: summary tables
    P->>AN: build_strategy_summary(...) monthly returns turnover costs
    AN-->>P: analytics tables
    P->>R: write_markdown_report(...)
    P->>R: plot_cumulative_returns(...)
    P->>R: plot_drawdown(...)
    P->>R: plot_turnover(...)
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

    P->>P: build_model_summary(...) and build_fold_summary(...)
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
      +strategy_summary_path: Path
      +monthly_returns_path: Path
      +turnover_costs_path: Path
      +model_summary_path: Path | None
      +fold_summary_path: Path | None
      +report_path: Path | None
      +cumulative_plot_path: Path | None
      +drawdown_plot_path: Path | None
      +turnover_plot_path: Path | None
    }

    class TrainModelsArtifacts {
      +run_dir: Path
      +panel_path: Path
      +folds_path: Path
      +model_manifest_path: Path
      +metrics_path: Path | None
      +predictions_path: Path | None
      +model_summary_path: Path
      +fold_summary_path: Path
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

### `scripts/run-e2e.ps1`

- Runs the current real-data smoke validation path against the smoke config.
- Optionally gates on fixture-backed pytest first.
- Verifies artifact sets for `prepare-data`, `backtest`, `train-models`, and `run-experiment`, including analytics outputs where applicable.
- Prints the resolved run directories used for the smoke review.

Best practice:
- Keep the smoke assertions aligned with the current command artifact surface.
- Treat smoke results as validation evidence, not robust model-selection proof.

### `Dockerfile`

- Builds a packaged MarketLab CLI image from the repository source.
- Installs the package into a Python 3.12 runtime image.
- Copies checked-in `configs/` into `/app/configs`.
- Runs as a non-root user with writable `/app/artifacts`.

Best practice:
- Keep Docker as an execution wrapper around the installed package, not as a second source-tree launcher.
- Preserve the repo-local launcher for development and the installed CLI for container automation.

### `.github/workflows/docker-runner.yml`

- Manually dispatches a Dockerized MarketLab run for `backtest`, `train-models`, or `run-experiment`.
- Builds the image, runs the selected command, captures the resolved run directory, and uploads copied artifacts.
- Keeps the workflow outside the required PR CI check set.

Best practice:
- Treat this workflow as manual historical smoke automation, not as a rolling live-market schedule.
- Keep required PR CI deterministic and offline; use the Docker runner when a packaged execution path or real-data smoke replay is the goal.

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

- Orchestrates the baseline backtest workflow, the `train-models` artifact path, and the unified `run-experiment` comparison path.
- Decides whether to reuse the prepared panel or rebuild it.
- Runs enabled baselines for backtests and reports.
- Builds modeling datasets, walk-forward folds, trained estimators, ML strategy weights, shared OOS slices, summary tables, analytics tables, and experiment artifacts.

Best practice:
- Put workflow coordination here, not in strategies, reports, model registry, or CLI code.
- Recompute metrics from the sliced OOS `PerformanceFrame`, not from the full-history backtest output.

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
- Only add trailing features unless a future phase deliberately expands the feature contract.
- Do not introduce forward-looking features or label leakage.

### `src/marketlab/rebalance.py`

- Centralizes the shared weekly rebalance calendar.
- Resolves the last available signal date in each `W-FRI` period.
- Resolves the next effective trading date after each signal date.
- Resolves the first future rebalance effective date after an existing signal date for fold-boundary flattening.

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

### `src/marketlab/strategies/ranking.py`

- Turns one model's fold predictions into a canonical `WeightsFrame`.
- Ranks longs from the highest scores and shorts from the lowest scores with `symbol` as the deterministic tie-breaker.
- Emits full-symbol weight rows with explicit zeros for non-selected names.
- Adds zero-weight boundary rows at the next rebalance `effective_date` when a later fold does not already begin there.

Best practice:
- Keep ranking prediction-only; do not derive scores from the panel in this layer.
- Keep the exposure policy explicit: `+0.5` total long, `-0.5` total short, gross exposure `1.0`.

### `src/marketlab/backtest/engine.py`

- Joins weights to adjusted open and close data.
- Splits return computation into overnight and intraday components.
- Applies turnover-based trading costs.
- Produces the canonical `PerformanceFrame`.

Best practice:
- Keep execution timing explicit.
- Use adjusted open and close consistently so splits and dividends do not distort returns.
- Backtest one strategy at a time, then concatenate performance frames at the pipeline layer.

### `src/marketlab/backtest/metrics.py`

- Summarizes performance by strategy.
- Computes cumulative return, annualized return, annualized volatility, sharpe-like ratio, max drawdown, hit rate, and turnover metrics.

Best practice:
- Treat this as a reporting summary layer, not a source of trading logic.

### `src/marketlab/reports/analytics.py`

- Builds strategy-level analytics tables from the canonical `PerformanceFrame`.
- Produces `strategy_summary.csv`, `monthly_returns.csv`, and `turnover_costs.csv`.
- Keeps analytics derived and deterministic rather than introducing new backtest state.

Best practice:
- Keep analytics builders pure and schema-stable.
- Derive analytics from the existing `PerformanceFrame`, not from alternate portfolio state.

### `src/marketlab/reports/markdown.py`

- Produces a compact Markdown report for each run.
- Derives the strategy list from the actual `PerformanceFrame`.
- Adds strategy summary, monthly net return, and turnover-and-cost sections when those analytics are available.
- Switches scope text when ML strategies are present.

### `src/marketlab/reports/summary.py`

- Builds fold-level and model-level summary tables from existing training metrics and manifests.
- Keeps the reporting summaries additive and derived from raw training outputs.

Best practice:
- Keep summary generation pure and deterministic.
- Do not invent new metrics in the summary layer.

### `src/marketlab/reports/plots.py`

- Produces cumulative equity, drawdown, and turnover charts.

Best practice:
- Report modules should only render artifacts from already-computed outputs.

### `tests/`

- Unit tests protect contracts and math.
- Integration tests validate fixture-backed pipeline behavior.
- The real-data smoke runner validates baseline backtest, training artifacts, experiment artifacts, analytics outputs, and summary outputs on real data.
- Real-data smoke tests stay opt-in because provider behavior and network access are unstable by nature.

### User-Local Codex Skills

- MarketLab role skills are expected to live in the developer's user-local Codex home, not in the repository.
- Keep repo automation and public packaging independent from private Codex workflow assets.

## Best Practices

- Use `python scripts/run_marketlab.py ...` for repo-local execution.
- Use the Docker image to validate the installed `marketlab` CLI path, not to replace the repo-local launcher during development.
- Use `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1` for full local smoke validation of the current artifact surface.
- Treat `.github/workflows/docker-runner.yml` as manual historical smoke automation, not as a required CI or rolling production schedule.
- Keep `cli.py` thin and `pipeline.py` orchestration-focused.
- Preserve the `MarketPanel`, weekly modeling dataset, `WeightsFrame`, and `PerformanceFrame` contracts.
- Build features from trailing information only.
- Keep provider normalization inside the data layer.
- Keep strategies responsible for weights, not return calculation.
- Keep backtest timing explicit: Friday-close signal, next-open execution.
- Keep walk-forward training windows label-aware: only train on rows whose `target_end_date` is known by `test_start`.
- Compare baseline and ML strategies on the same shared OOS daily window inside `run-experiment`.
- Derive `model_summary.csv` and `fold_summary.csv` from existing model metrics and manifests, not from new training state.
- Treat no allocation as cash with zero return.
- Keep `train-models` artifact-focused; use `run-experiment` for unified baseline-plus-ML comparison.
- Use the smoke runner to validate the full current artifact surface after runtime changes.

## Current Risks

- `yfinance` remains an unstable external dependency despite the new column-flattening and cached-header cleanup.
- `run-experiment` now trains models in-process and may leave per-fold model pickles in experiment run directories as a side effect of reusing the training layer.
- summary and analytics outputs are derived from the existing training/performance artifacts, so metric changes propagate through both the raw CSVs and the report tables.
- the model registry currently assumes classifier-style `predict_proba` outputs and `target.type="direction"`.
- metric definitions are suitable for a research scaffold, not yet a full institutional evaluation stack.

## Extension Rules

- Add richer reporting without breaking the existing panel, weekly modeling dataset, weights, performance, or shared OOS comparison contracts.
- Keep walk-forward evaluation in the evaluation layer, not inside current strategy modules.
- Reuse the fold engine and its `target_end_date <= test_start` rule rather than rebuilding train/test masks in model code.
- Keep model construction in the registry and workflow orchestration in `pipeline.py`.
- Do not batch multiple strategies into a single `run_backtest(...)` call.
- Do not redesign the current data layer just to support later model abstractions.
- Preserve the local launcher and E2E runner as the default developer entrypoints.


