---
name: marketlab-pre-commit-checks
description: Run MarketLab validation commands before commits using the repo's canonical tox and uv entrypoints. Use when Codex is asked to sanity-check the repo, run tests before a commit, verify packaging, or execute the internal tox and uv workflows without guessing which executable or Python environment to use.
---

# MarketLab Pre-Commit Checks

Read `../marketlab-shared-context/references/tooling.md` first. Read `references/workflow.md` when you need the exact validation order.

## Use This Skill To

- run the repo test suite before making a commit
- verify tox-managed environments
- verify uv-managed workflows
- confirm the package still builds cleanly
- run the opt-in real-data Sprint 1 smoke flow when explicitly requested

## Run Order

- For the fastest confidence check, run:
  - `python -m pytest -q --basetemp .pytest_tmp`
- For local repo CLI execution, run:
  - `python scripts/run_marketlab.py run-experiment --config configs/experiment.weekly_rank.yaml`
- When the user asks for tox validation, run:
  - `py -3.14 -m tox -e py314`
  - `py -3.14 -m tox -e package`
- When the user asks for uv validation, run:
  - `python -m uv sync --dev`
  - `python -m uv run pytest -q --basetemp .pytest_tmp`
  - `python -m uv build`
- When the user asks for the real-data E2E smoke path, run:
  - `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1`
  - or `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1 -SkipPytest` if the fast pytest gate already passed in the same session

## Command Rules

- Prefer module invocation over bare executables.
- Use `py -3.14 -m tox` instead of `tox`.
- Use `python -m uv` instead of `uv`.
- Use `python scripts/run_marketlab.py` for repo-local MarketLab commands.
- Use `python -m marketlab.cli` only when validating the installed package entrypoint on purpose.
- Keep tox and uv checks separate so failures are easy to attribute.
- Treat the real-data path as opt-in because it depends on `yfinance`, network access, and provider stability.
- Keep `train-models` out of the passing E2E flow until Sprint 2 is implemented.
- If sandbox restrictions block tox or uv environment creation, request escalation instead of switching tools silently.

## Report Back

- Say which commands ran.
- Say which path passed or failed: pytest, tox, uv, or packaging.
- If a command was skipped because the user did not ask for that workflow, say so explicitly.
