# Tooling Notes

Use this note for internal validation workflows, not user-facing docs.

## Canonical Commands

- Preferred tox entrypoint:
  - `py -3.14 -m tox -e py314`
  - `py -3.14 -m tox -e package`
- Preferred uv entrypoint:
  - `python -m uv sync --dev`
  - `python -m uv run pytest -q --basetemp .pytest_tmp`
  - `python -m uv build`
- Preferred MarketLab CLI entrypoint:
  - `python scripts/run_marketlab.py prepare-data --config configs/experiment.weekly_rank.yaml`
  - `python scripts/run_marketlab.py backtest --config configs/experiment.weekly_rank.yaml`
  - `python scripts/run_marketlab.py run-experiment --config configs/experiment.weekly_rank.yaml`
- Preferred real-data E2E runner:
  - `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1`
  - `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1 -SkipPytest`

## Why These Forms

- Prefer `py -3.14 -m tox` over bare `tox`.
- On this machine, `tox.exe` on `PATH` may resolve to an older Python install.
- Prefer `python -m uv` over bare `uv`.
- `uv` may be installed as a module even when the executable is not on `PATH`.
- Prefer `python scripts/run_marketlab.py` for repo-local execution.
- Do not rely on the bare `marketlab` console script being on `PATH`.
- Do not assume `python -m marketlab.cli` will resolve to the repo source in a `src/` layout.

## Pre-Commit Validation Intent

- Use local pytest for the fast check.
- Use tox for managed-environment verification before commit when requested.
- Use uv when the user explicitly wants the uv workflow or lock/build behavior checked.
- Use `python -m marketlab.cli` only when explicitly validating the installed package entrypoint.
- Use `scripts/run-e2e.ps1` only for the opt-in real-data smoke path.
- The real-data runner covers Sprint 1 commands only: `prepare-data`, `backtest`, and `run-experiment`.
- Do not treat `train-models` as part of the passing E2E path until Sprint 2 exists.
- If tox hits sandbox temp or app-data restrictions, rerun with escalation instead of repeatedly retrying inside the sandbox.
