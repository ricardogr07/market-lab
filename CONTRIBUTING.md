# Contributing to MarketLab

MarketLab is intentionally small and reviewable. Keep changes focused, documented, and easy to validate.

## Workflow

1. Start from a fresh branch off `master`.
2. Keep commits small and intentional.
3. Run the local validation tiers before opening or updating a PR:
   - normal iteration: the specific lane you touched
   - normal pre-PR check: `py -3.12 -m tox -e preflight-fast`
   - final local push gate: `py -3.12 -m tox -e preflight`
4. Add tests for behavior changes.
5. Ask for approval before pushing if the change is part of an agent-led workflow.

## Scope

- Prefer narrow PRs that change one feature or one support surface.
- Avoid mixing runtime behavior changes with packaging or docs cleanup unless the docs are required to describe the new interface.
- Keep public wording factual and avoid overstating maturity.

## Local Tools

- `python scripts/run_marketlab.py ...` for repo-local execution
- `py -3.12 -m tox -e lint` for lint-only changes
- `py -3.12 -m tox -e docs` for docs-only changes
- `py -3.12 -m tox -e py312` for the unit suite
- `py -3.12 -m tox -e package` for packaging and installed-CLI checks
- `py -3.12 -m tox -e integration` for pipeline, config, artifact, and report changes
- `py -3.12 -m tox -e preflight-fast` for the standard fast local gate
- `py -3.12 -m tox -e preflight-slow` for the slow packaging plus integration lanes
- `py -3.12 -m tox -e preflight` for the full local pre-push gate
- `py -3.12 scripts/profile_validation.py --env package --env integration` to profile the slow lanes directly
- `powershell -ExecutionPolicy Bypass -File scripts/run-e2e.ps1` for real-data smoke validation when needed

Observed local Windows budgets are currently:

- `lint`: under `30s`
- `docs`: under `30s`
- `py312`: under `60s`
- `package`: about `4-6m`
- `integration`: about `8-10m`
- `preflight`: about `14-16m`

When `preflight` feels slow or unstable, diagnose it in this order:

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

## Pull Requests

- Describe the user-facing effect.
- List the validation commands you ran.
- Call out any follow-up work that is intentionally deferred.


