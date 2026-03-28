# Contributing to MarketLab

MarketLab is intentionally small and reviewable. Keep changes focused, documented, and easy to validate.

## Workflow

1. Start from a fresh branch off `master`.
2. Keep commits small and intentional.
3. Run the local pre-push gate before opening or updating a PR:
   - `python -m tox -e preflight`
4. Add tests for behavior changes.
5. Ask for approval before pushing if the change is part of an agent-led workflow.

## Scope

- Prefer narrow PRs that change one feature or one support surface.
- Avoid mixing runtime behavior changes with packaging or docs cleanup unless the docs are required to describe the new interface.
- Keep public wording factual and avoid overstating maturity.

## Local Tools

- `python scripts/run_marketlab.py ...` for repo-local execution
- `python -m tox -e preflight` for the standard local validation gate
- `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1` for real-data smoke validation when needed

## Pull Requests

- Describe the user-facing effect.
- List the validation commands you ran.
- Call out any follow-up work that is intentionally deferred.
