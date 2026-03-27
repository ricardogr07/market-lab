# Workflow

## Default Validation

1. Run `python -m pytest -q --basetemp .pytest_tmp`.
2. If the user wants managed-environment coverage, run `py -3.14 -m tox -e py314`.
3. If the user wants packaging coverage, run `py -3.14 -m tox -e package`.

## uv Path

1. Run `python -m uv sync --dev`.
2. Run `python -m uv run pytest -q --basetemp .pytest_tmp`.
3. Run `python -m uv build`.

## Real-Data E2E Path

1. Run `python -m pytest -q --basetemp .pytest_tmp` unless it already passed and the user explicitly wants to skip it.
2. Run `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1`.
3. Use `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1 -SkipPytest` only when the same session already has a passing local pytest run.
4. Treat `train-models` as excluded from the passing E2E flow until Sprint 2 is implemented.

## Notes

- Prefer the Python-module forms because they are more reliable on this machine than bare `tox` or `uv`.
- Prefer `python scripts/run_marketlab.py` for repo-local CLI calls.
- Use `python -m marketlab.cli` only when intentionally validating the installed package entrypoint.
- Use escalation if sandbox restrictions block temp, app-data, or dependency installation paths.
