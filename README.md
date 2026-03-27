# MarketLab

MarketLab is a package-first research lab for reproducible market experiments. The current scaffold implements Sprint 1 only: cached panel preparation, trailing feature engineering, two rule-based baselines, a daily backtest engine, metrics, plots, and Markdown reporting.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the system map, data contracts, execution flow, and extension rules.

## Current Commands

```bash
python scripts/run_marketlab.py prepare-data --config configs/experiment.weekly_rank.yaml
python scripts/run_marketlab.py backtest --config configs/experiment.weekly_rank.yaml
python scripts/run_marketlab.py run-experiment --config configs/experiment.weekly_rank.yaml
python scripts/run_marketlab.py train-models --config configs/experiment.weekly_rank.yaml
```

`train-models` is reserved for Sprint 2 and currently exits with a clear stub message.
For local repo usage, `python scripts/run_marketlab.py ...` is the canonical path because it always resolves to the source tree under `src/`.

## Environment

- Python 3.12+
- Installed packages:
  - `pandas`
  - `PyYAML`
  - `matplotlib`
  - `yfinance`
  - `scikit-learn` for later ML work

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

## Output

Each experiment run writes a timestamped folder under `artifacts/runs/<experiment_name>/` containing:

- `metrics.csv`
- `performance.csv`
- `report.md`
- `cumulative_returns.png`
- `drawdown.png`
