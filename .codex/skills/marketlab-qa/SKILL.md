---
name: marketlab-qa
description: MarketLab quality-assurance role for validating behavior, tests, fixtures, artifacts, and acceptance criteria across sprints. Use when Codex needs to review MarketLab changes for correctness, add missing tests required to prove behavior, or verify that outputs match the repo's frozen contracts and domain rules.
---

# MarketLab QA

Read `../marketlab-shared-context/references/data-contracts.md` and `../marketlab-shared-context/references/guardrails.md` first. Read `../marketlab-shared-context/references/domain-rules.md` when validating backtests or metrics.

## Validate

- Check behavior before polish.
- Add missing tests when acceptance criteria are not yet provable.
- Prefer small deterministic fixtures over large datasets.
- Verify artifact paths and file creation, not only in-memory outputs.
- Treat leakage, broken contracts, and silent artifact regressions as failures.

## Review Focus

- Unit coverage for pure logic
- Integration coverage for CLI-to-artifact flow
- Fixture realism for panel, strategy, and turnover math
- Reproducibility of outputs from the same cached input
