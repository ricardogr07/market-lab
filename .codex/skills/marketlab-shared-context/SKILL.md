---
name: marketlab-shared-context
description: Shared MarketLab project context for the frozen MVP, sprint boundaries, data contracts, and architecture guardrails. Use when another MarketLab role skill needs the current repo conventions, panel schema, CLI/config rules, or sprint-specific constraints before planning, implementing, reviewing, or validating work.
---

# MarketLab Shared Context

Read only the reference files needed for the current task.

## Read Order

- Read `references/mvp.md` first when you need the frozen scope, repo purpose, or current maturity level.
- Read `references/sprint-map.md` when the task depends on which sprint owns a capability.
- Read `references/data-contracts.md` when the task touches config, panel data, weights, performance outputs, or artifacts.
- Read `references/domain-rules.md` when the task depends on market-data semantics, rebalance timing, or cost handling.
- Read `references/guardrails.md` when the task risks scope creep, leakage, or premature abstractions.
- Read `references/tooling.md` when the task depends on internal tox or uv workflows, especially pre-commit validation.

## Working Rules

- Treat Sprint 1 as the only implemented scope unless the task explicitly extends the repo further.
- Keep decisions aligned with the current package-first, local-first architecture.
- Prefer the leanest change that preserves the frozen data and backtest contracts.
- Do not duplicate this shared context into other role skills; let them point back here.
