# MarketLab

MarketLab is a package-first research lab for reproducible market experiments. The current implementation now includes a working Phase 2 ML MVP on top of the frozen Sprint 1 scaffold: weekly supervised modeling rows, walk-forward folds, trained models, rank-based ML strategies, baseline-plus-ML experiments, and reviewable artifact summaries.

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
- `backtest`: run the rule baselines only (`buy_hold` and `sma`) and write performance, metrics, report, and plots.
- `train-models`: fit the configured models across walk-forward folds and write raw training artifacts plus fold/model summary CSVs.
- `run-experiment`: run baselines and ML strategies together on the shared out-of-sample window and write the experiment outputs plus summary CSVs.

## Artifact Outputs

### `train-models`

Writes a timestamped folder under `artifacts/runs/<experiment_name>/` containing:

- `folds.csv`
- `model_manifest.csv`
- `model_metrics.csv`
- `predictions.csv`
- `model_summary.csv`
- `fold_summary.csv`
- per-fold model pickles under `models/`

### `run-experiment`

Writes a timestamped folder under `artifacts/runs/<experiment_name>/` containing:

- `metrics.csv`
- `performance.csv`
- `report.md`
- `cumulative_returns.png`
- `drawdown.png`
- `model_summary.csv`
- `fold_summary.csv`

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
