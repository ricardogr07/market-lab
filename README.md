# MarketLab

MarketLab is a package-first research toolkit for reproducible market experiments over a fixed ETF universe. The current implementation includes a working baseline-plus-ML workflow: weekly supervised modeling rows, walk-forward folds, trained models, rank-based ML strategies, periodic allocation baselines, executable mean-variance and risk-parity baselines, shared out-of-sample experiments, and reviewable artifact summaries.

See [docs/architecture.md](docs/architecture.md) for the system map, data contracts, execution flow, and extension rules.
See [docs/how-it-works.md](docs/how-it-works.md) for a narrative walkthrough of the library and the `voo_long_only_ytd` timing example.
See [docs/PLAN.md](docs/PLAN.md) for the current project status and Phase 5 direction.

## Current Commands

```bash
python scripts/run_marketlab.py prepare-data --config configs/experiment.weekly_rank.yaml
python scripts/run_marketlab.py backtest --config configs/experiment.weekly_rank.yaml
python scripts/run_marketlab.py train-models --config configs/experiment.weekly_rank.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.weekly_rank.yaml
```

`python scripts/run_marketlab.py ...` is the canonical local invocation path because it always resolves to the source tree under `src/`.

## What Each Command Does

- `prepare-data`: build or reuse the cached prepared panel.
- `backtest`: run the enabled baselines (`buy_hold`, `sma`, optional config-defined allocation baselines, and the optional executable optimized baseline) and write performance, analytics summaries, report, and plots.
- `train-models`: fit the configured models across walk-forward folds and write raw training artifacts plus fold/model summaries, ranking diagnostics, calibration diagnostics, threshold diagnostics, and review plots.
- `run-experiment`: run baselines and ML strategies together on the shared out-of-sample window and write the experiment outputs, analytics summaries, ranking-aware ML summary CSVs, calibration/threshold diagnostics, and review plots.

## Artifact Outputs

### `train-models`

Writes a timestamped folder under `artifacts/runs/<experiment_name>/` containing:

- `folds.csv`
- `fold_diagnostics.csv`
- `model_manifest.csv`
- `model_metrics.csv`
- `predictions.csv`
- `ranking_diagnostics.csv`
- `calibration_diagnostics.csv`
- `score_histograms.csv`
- `threshold_diagnostics.csv`
- `model_summary.csv`
- `fold_summary.csv`
- `calibration_curves.png`
- `score_histograms.png`
- `threshold_sweeps.png`
- per-fold model pickles under `models/`

### `backtest`

Writes a timestamped folder under `artifacts/runs/<experiment_name>/` containing:

- `metrics.csv`
- `performance.csv`
- `strategy_summary.csv`
- `monthly_returns.csv`
- `turnover_costs.csv`
- `cost_sensitivity.csv`
- `daily_exposure.csv`
- optional `group_exposure.csv`
- optional `benchmark_relative.csv`
- `report.md`
- `cumulative_returns.png`
- `drawdown.png`
- `turnover.png`

### `run-experiment`

Writes a timestamped folder under `artifacts/runs/<experiment_name>/` containing:

- `metrics.csv`
- `performance.csv`
- `strategy_summary.csv`
- `monthly_returns.csv`
- `turnover_costs.csv`
- `cost_sensitivity.csv`
- `daily_exposure.csv`
- optional `group_exposure.csv`
- optional `benchmark_relative.csv`
- `report.md`
- `cumulative_returns.png`
- `drawdown.png`
- `turnover.png`
- `fold_diagnostics.csv`
- `ranking_diagnostics.csv`
- `calibration_diagnostics.csv`
- `score_histograms.csv`
- `threshold_diagnostics.csv`
- `model_summary.csv`
- `fold_summary.csv`
- `calibration_curves.png`
- `score_histograms.png`
- `threshold_sweeps.png`
- optional per-fold model pickles under `models/`

## Walk-Forward Guardrails

`evaluation.walk_forward` now supports these additive guardrail keys:

- `min_train_rows`
- `min_test_rows`
- `min_train_positive_rate`
- `min_test_positive_rate`
- `embargo_periods`

The shipped `weekly_rank` templates opt into a conservative preset, while code defaults remain backward-compatible for older configs.

When `train-models` or `run-experiment` end up with zero usable folds, they still create the run directory, write `fold_diagnostics.csv`, and fail with an error that includes the diagnostics path. On successful ML experiment runs, `run-experiment` also includes a `Walk-Forward Diagnostics` section in `report.md`.

## Ranking-Aware Evaluation

Model evaluation now stays additive to the existing ROC AUC surface while reflecting the configured ranking mode. `ranking_diagnostics.csv` stores one row per model, fold, and signal date, now including `evaluation_mode` so the persisted diagnostics distinguish the default `long_short` path from `long_only` timing runs. `model_metrics.csv` keeps the existing classification fields and now also includes balanced classification metrics, Brier score, per-fold mean rank correlation, top/bottom bucket returns, top-bottom spread, spread hit rate, top-bucket hit rate, worst observed top-bucket return, worst observed spread, and the counts of usable ranking dates.

`model_summary.csv` and `fold_summary.csv` still preserve the ROC AUC winner fields for continuity, and they now add both spread-based and long-only-friendly winner fields. The report headline mirrors that split by showing the best model by mean ROC AUC, mean top-bucket return, and mean top-bottom spread. This remains evaluation-focused: it does not change how weights are generated or how ML strategies trade.

## Calibration And Threshold Diagnostics

Calibration review is also additive to the existing evaluation surface. `calibration_diagnostics.csv` stores fixed 10-bin score calibration rows per model and fold, including observed positive rate, calibration gap, and return context inside each score band. `score_histograms.csv` stores target-class score distributions over the same fixed bins, and `threshold_diagnostics.csv` stores threshold sweeps from `0.05` to `0.95` with deterministic classification metrics plus downside-oriented forward-return columns for predicted positives.

`model_metrics.csv` now adds fold-level `ece` and `max_calibration_gap`, while `model_summary.csv` and `fold_summary.csv` add `mean_ece` and `mean_max_calibration_gap`. When plots are enabled, `train-models` and `run-experiment` also write `calibration_curves.png`, `score_histograms.png`, and `threshold_sweeps.png`, and the experiment report includes a `Calibration And Threshold Diagnostics` section with compact calibration and threshold highlight tables. This PR remains evaluation-only: it does not recalibrate probabilities or change any strategy controls.

## Ranking Strategy Modes

`run-experiment` now supports additive execution controls under `portfolio.ranking`:

- `mode`: `long_short` or `long_only`
- `min_score_threshold`: minimum score required for long selection; in `long_short`, shorts require `score <= 1 - min_score_threshold`
- `cash_when_underfilled`: when `true`, keep fixed per-slot weights for the names that pass and leave the missing exposure in cash instead of zeroing the whole basket

Defaults remain backward-compatible:

- `mode: long_short`
- `min_score_threshold: 0.0`
- `cash_when_underfilled: false`

Execution semantics:

- `long_short` keeps the existing equal-weight market-neutral construction with `+0.5` total long exposure and `-0.5` total short exposure when the basket is fully populated.
- `long_only` allocates `+1.0 / long_n` per selected long and leaves all other names at `0.0`.
- With `cash_when_underfilled: false`, any underfilled basket still falls back to an all-zero allocation for that rebalance.
- With `cash_when_underfilled: true`, missing slots stay in cash and the selected names keep their fixed per-slot weights instead of being renormalized.

Non-default ML strategy variants are named explicitly in experiment outputs, for example `ml_logistic_regression__long_only` or `ml_random_forest__long_short__thr0p60__cash`.

`train-models` now keeps the existing issue #19 and #20 score-review artifacts while making the ranking diagnostics mode-aware for `long_short` and `long_only`. Threshold gating and cash-underfilled behavior still remain execution-only controls; the offline evaluation layer does not replay those execution variants.

## Ranking Exposure Caps

`run-experiment` now also supports structural risk caps for ML ranking strategies under `portfolio.risk`:

- `max_position_weight`
- `max_group_weight`
- `max_long_exposure`
- `max_short_exposure`

These caps apply only after the current ranking strategy has already selected longs and shorts. MarketLab clips single-name exposure first, then clips group exposure separately for the long and short sleeves, then caps total long exposure, and finally caps total short exposure. Any removed exposure stays in cash; it is never renormalized back into the book.

This remains a narrow structural-control step. The new caps do not change `buy_hold`, `sma`, or allocation baselines, and they do not add optimizer methods, factor-model attribution, or broader scenario-pack work yet. Lower realized volatility or drawdown under a capped ranking strategy may simply reflect lower invested exposure or more cash, not better signal quality.

## Exposure-Aware Analytics

`backtest` and `run-experiment` now also persist additive exposure analytics alongside the existing return and turnover artifacts.

- `daily_exposure.csv` stores end-of-day drifted long, short, gross, and net exposure for every strategy date.
- `cash_weight` is exposure-style slack: `max(0, 1 - gross_exposure)`.
- `engine_cash_weight` is the engine's carried cash or collateral weight, which matters for long-short books where gross exposure can be fully deployed while the engine still carries collateral cash.
- `group_exposure.csv` is written when `data.symbol_groups` covers the run universe, and it keeps long and short sleeves separate instead of netting them together.
- `strategy_summary.csv` now appends average exposure, cash, active-position, and concentration fields, and `report.md` includes an `Exposure Summary` section.

These analytics are interpretive, not predictive. Lower drawdown or volatility can simply reflect lower gross exposure or more cash, not better signal quality.


## Benchmark-Relative Reporting

`backtest` and `run-experiment` now also support optional benchmark-relative analytics under `evaluation.benchmark_strategy`.

- The benchmark is an existing strategy name already present in the run, not a raw symbol.
- When configured, MarketLab writes `benchmark_relative.csv` with daily strategy return, benchmark return, active return, and relative-equity paths on shared dates.
- `strategy_summary.csv` now also appends benchmark-relative fields such as excess cumulative return, annualized excess return, tracking error, information ratio, correlation to benchmark, and up/down capture.
- `report.md` includes a `Benchmark-Relative Summary` section when a benchmark is configured.

These metrics are comparative, not causal. Higher absolute return and better benchmark-relative performance are separate questions, and lower active risk does not imply outperformance.

## Turnover And Cost Sensitivity

`backtest` and `run-experiment` now also support additive turnover-cost sensitivity diagnostics under `evaluation.cost_sensitivity_bps`.

- `cost_sensitivity.csv` reprices each strategy path at `0.0` bps, the configured `portfolio.costs.bps_per_trade`, and any extra configured bps assumptions without rerunning the strategies.
- The `0.0` bps rows are theoretical gross-return baselines, not executable outcomes.
- The row at the configured trading-cost assumption matches the current net-return and cost-drag path already shown in `strategy_summary.csv`.
- `report.md` includes a `Cost Sensitivity` section so implementation-cost assumptions can be reviewed alongside turnover.

This remains a reporting-only diagnostic. It does not change weights, execution timing, or the backtest engine.

## Allocation Baselines And Symbol Groups

`backtest` and `run-experiment` now also support optional config-defined allocation baselines under `baselines.allocation`.

Add to `data`:

- `symbol_groups`: optional mapping from symbol to group name

Add to `baselines`:

- `allocation.enabled`
- `allocation.mode`: `equal`, `symbol_weights`, or `group_weights`
- `allocation.symbol_weights`
- `allocation.group_weights`

Allocation semantics:

- `buy_hold` emits one initial equal-weight allocation and then lets positions drift naturally.
- `allocation_equal` rebalances back to equal target weights on the existing rebalance cadence.
- `allocation_symbol_weights` rebalances back to exact configured symbol weights.
- `allocation_group_weights` rebalances back to configured group sleeves and splits each sleeve equally across the symbols in that group.

This first Phase 5 step stays narrow: allocation baselines are long-only, fully invested target-weight portfolios. Optimizer methods, factor diagnostics, and broader scenario comparisons remain later work.

## Optimized Baselines

`backtest` and `run-experiment` now also support executable long-only optimized baselines under `baselines.optimized`.

Add to `baselines`:

- `optimized.enabled`
- `optimized.method`: `mean_variance`, `risk_parity`, or `black_litterman`
- `optimized.lookback_days`
- `optimized.rebalance_frequency`
- `optimized.covariance_estimator`: `sample`, `ewma`, `diagonal_shrinkage`, or `external_csv`
- `optimized.external_covariance_path`
- `optimized.expected_return_source`: `historical_mean` or `external_csv`
- `optimized.external_expected_returns_path`
- `optimized.long_only`
- `optimized.target_gross_exposure`
- `optimized.risk_aversion`

Current Phase 5 behavior is intentionally narrow:

- `mean_variance` and `risk_parity` are executable optimized methods
- `black_litterman` still fails fast with an explicit not-implemented error
- the optimizer uses trailing daily adjusted-close returns ending on the `signal_date` and applies the weights on the next market open
- no allocation is emitted before the first rebalance window with a full optimizer lookback
- `target_gross_exposure < 1.0` leaves the undeployed exposure in cash
- `portfolio.risk.max_position_weight` and `portfolio.risk.max_group_weight` are enforced as hard long-only optimizer constraints
- `risk_aversion` only applies to `mean_variance`
- `risk_parity` uses only the configured covariance estimator and does not consume expected-return inputs
- capped `risk_parity` portfolios are the best feasible approximation to equal risk contributions, not exact parity under binding caps

External input rules:

- covariance CSVs must be square daily-return covariance matrices keyed by the configured symbols
- expected-return CSVs must contain exactly `symbol,expected_return`, where `expected_return` is a daily decimal return
- both loaders reorder to `data.symbols` and reject missing, extra, or non-numeric values

## Single-Symbol VOO Timing Example

`configs/experiment.voo_long_only.ytd.yaml` is a tracked one-symbol directional timing example built around `VOO` from `2018-01-01` through `2026-04-03`. It currently compares five sklearn models, runs in `long_only` mode with `long_n: 1`, and lowers `min_test_rows` to `10` so quarterly test folds stay viable on a one-symbol weekly dataset.

Treat this config as a timing study, not as a cross-sectional ranking experiment. Compare its ML outputs primarily against `buy_hold` and `sma`, and do not read the results as evidence about cross-sectional ranking skill.

## Lightweight Model Comparison Set

The default weekly configs now compare six sklearn-only direction classifiers:

- `logistic_regression`
- `logistic_l1`
- `random_forest`
- `extra_trees`
- `gradient_boosting`
- `hist_gradient_boosting`

This wave deliberately stays lightweight. It broadens the comparison baseline without adding external booster dependencies, model-specific pipeline branching, or tuning knobs.
## Environment

- Python 3.12+
- Installed packages:
  - `pandas`
  - `PyYAML`
  - `matplotlib`
  - `yfinance`
  - `scikit-learn`
  - `scipy`

## Quickstart

```bash
python -m pip install -e .[dev]
python scripts/run_marketlab.py run-experiment --config configs/experiment.weekly_rank.yaml
```

If `artifacts/data/panel.csv` already exists, the pipeline uses it and does not attempt a network download.

## Installed Package Quickstart

If you install MarketLab from PyPI or a built wheel, use the packaged CLI bootstrap flow instead of the repo launcher:

```bash
marketlab --version
marketlab list-configs
marketlab write-config --name weekly_rank --output weekly_rank.yaml
marketlab run-experiment --config weekly_rank.yaml
```

`list-configs` shows the bundled example templates, and `write-config` exports one of those templates into your working directory. That keeps the installed package self-contained without requiring a checkout of this repository.

## Local Validation

```bash
python -m pytest -q --basetemp .pytest_tmp
powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1
```

## Local CI Entry Points

```bash
python -m uv sync --group dev
py -3.12 -m tox -e lint
py -3.12 -m tox -e docs
py -3.12 -m tox -e py312
py -3.12 -m tox -e package
py -3.12 -m tox -e integration
py -3.12 -m tox -e preflight-fast
py -3.12 -m tox -e preflight-slow
py -3.12 -m tox -e preflight
py -3.12 scripts/profile_validation.py --env package --env integration
```

Use `py -3.12 -m tox -e preflight` as the canonical local pre-push gate so local validation matches the Python version used in GitHub Actions. For normal iteration, prefer the specific lane you touched or `preflight-fast`; leave `preflight-slow` and the full `preflight` run for packaging, artifact, or final push checks.

Current measured local Windows budgets are roughly:

- `lint`: under `30s`
- `docs`: under `30s`
- `py312`: under `60s`
- `package`: about `4-6m`
- `integration`: about `8-10m`
- `preflight`: about `14-16m`

That makes `preflight` intentionally long because it is the sum of the expensive packaging and integration lanes, not because the fast lanes are slow. Use `scripts/profile_validation.py` when you need per-lane timings or need to confirm which component is driving a slow local run.

## Investigating Slow Local Validation

Use this sequence when `preflight` feels slow or unstable:

1. `py -3.12 -m tox -e lint`
2. `py -3.12 -m tox -e docs`
3. `py -3.12 -m tox -e py312`
4. `py -3.12 -m tox -e package`
5. `py -3.12 -m tox -e integration`
6. `py -3.12 scripts/profile_validation.py --env package --env integration`
7. `py -3.12 -m tox -e preflight`

Interpret the result this way:

- if only `package` is unstable or much slower than expected, inspect `scripts/check_package.py` and its scratch or virtualenv path handling next
- if only `integration` dominates, profile that suite next, starting with pytest duration reporting inside the integration lane
- if both are stable individually but `preflight` still feels killed, treat that as a tooling-timeout or UX problem rather than a MarketLab runtime failure

The MkDocs site now builds directly from the `docs/` directory, which is the canonical home for the public documentation set.

## Contribution Workflow

- Branch from a refreshed `master` instead of working directly on the default branch.
- Keep changes in small intentional commits so review scope stays clear.
- Run `py -3.12 -m tox -e preflight-fast` during normal iteration.
- Run `py -3.12 -m tox -e package` for packaging or installed-CLI work.
- Run `py -3.12 -m tox -e integration` for pipeline, config, artifact, or report changes.
- Run `py -3.12 -m tox -e preflight` before pushing.
- Open a pull request for review instead of pushing directly to `master`.
- Treat the `Docker Runner` workflow as an optional manual smoke path, not as a required pre-push step.
- Keep Codex skills and other personal automation assets in the user-local Codex home rather than in the public repository or package surface.
- Expect `master` to move ahead of the last public release between monthly release batches.

## Dockerized CLI

```bash
docker build -t marketlab-cli .
docker run --rm marketlab-cli --help
docker run --rm marketlab-cli backtest --config configs/experiment.weekly_rank.smoke.yaml
```

The container uses the installed `marketlab` console script as its entrypoint. Keep using `python scripts/run_marketlab.py ...` for local source-tree development; the Docker image exists to validate the installed package path and to support manual GitHub Actions runs.

## Manual Docker Runner Workflow

GitHub Actions now includes a manual workflow named `Docker Runner` with these inputs:

- `command`: `backtest`, `train-models`, or `run-experiment`
- `config_path`: repo-relative config path inside the image, defaulting to `configs/experiment.weekly_rank.smoke.yaml`

The workflow defaults to `backtest`, builds the Docker image, runs the selected command inside the container, writes the resolved run directory into the job summary, and uploads the copied `artifacts/` tree as an Actions artifact.

This workflow is not part of the required PR CI checks. It is a manual historical real-data smoke runner around the checked-in smoke config, not a rolling weekly market automation job.

## Release Automation

GitHub Actions now includes a release workflow at `.github/workflows/release.yml`.

- Normal feature PRs still merge to `master` in sequence.
- Each merge to `master` updates the open Release PR managed by release-please.
- The Release PR accumulates the unreleased monthly or feature batch over time.
- Nothing is tagged or published when a normal feature PR lands on `master`.
- The actual Git tag, GitHub Release, and PyPI publish path run only when you merge the Release PR.

This means `master` can intentionally contain unreleased work while you continue merging feature PRs. The Release PR is the public-release gate.

### Manual Prerequisites

Before the first automated public release:

- verify that the `marketlab` package name is available on PyPI
- configure PyPI Trusted Publishing for this repository and the `pypi` environment
- create the GitHub Actions environment named `pypi`

The first automated public release target remains `v0.1.0`.






