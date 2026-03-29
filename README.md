# MarketLab

MarketLab is a package-first research toolkit for reproducible market experiments over a fixed ETF universe. The current implementation includes a working baseline-plus-ML workflow: weekly supervised modeling rows, walk-forward folds, trained models, rank-based ML strategies, shared out-of-sample experiments, and reviewable artifact summaries.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the system map, data contracts, execution flow, and extension rules.

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
- `backtest`: run the rule baselines only (`buy_hold` and `sma`) and write performance, analytics summaries, report, and plots.
- `train-models`: fit the configured models across walk-forward folds and write raw training artifacts plus fold/model summary and diagnostics CSVs.
- `run-experiment`: run baselines and ML strategies together on the shared out-of-sample window and write the experiment outputs, analytics summaries, ML summary CSVs, and fold diagnostics.

## Artifact Outputs

### `train-models`

Writes a timestamped folder under `artifacts/runs/<experiment_name>/` containing:

- `folds.csv`
- `fold_diagnostics.csv`
- `model_manifest.csv`
- `model_metrics.csv`
- `predictions.csv`
- `model_summary.csv`
- `fold_summary.csv`
- per-fold model pickles under `models/`

### `backtest`

Writes a timestamped folder under `artifacts/runs/<experiment_name>/` containing:

- `metrics.csv`
- `performance.csv`
- `strategy_summary.csv`
- `monthly_returns.csv`
- `turnover_costs.csv`
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
- `report.md`
- `cumulative_returns.png`
- `drawdown.png`
- `turnover.png`
- `fold_diagnostics.csv`
- `model_summary.csv`
- `fold_summary.csv`
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

## Environment

- Python 3.12+
- Installed packages:
  - `pandas`
  - `PyYAML`
  - `matplotlib`
  - `yfinance`
  - `scikit-learn`

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
python -m uv run tox -e lint
python -m uv run tox -e docs
python -m uv run tox -e package
python -m uv run tox -e py312
python -m tox -e integration
python -m tox -e preflight
```

Use `python -m tox -e preflight` as the canonical local pre-push gate. It runs the same lint, docs, packaging, unit-test, and offline integration checks that Phase 3 CI expects through one local entrypoint after the dev dependencies are installed.

The MkDocs site renders the current root Markdown docs through `mkdocs-include-markdown-plugin`, so the documentation build stays aligned with `README.md`, `ARCHITECTURE.md`, `Phase2-results.md`, and `PLAN.md`.

## Contribution Workflow

- Branch from a refreshed `master` instead of working directly on the default branch.
- Keep changes in small intentional commits so review scope stays clear.
- Run `python -m tox -e preflight` before pushing.
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
