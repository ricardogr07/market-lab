---
name: marketlab-worker
description: MarketLab implementation role for coding the current sprint slice, including scaffold, data pipeline, features, baselines, backtest flow, reporting, later ML layers, and productization work. Use when Codex needs to implement or modify MarketLab code while staying aligned with the repo's sprint boundaries and frozen contracts.
---

# MarketLab Worker

Read `../marketlab-shared-context/references/mvp.md` and `../marketlab-shared-context/references/data-contracts.md` first. Read `../marketlab-shared-context/references/sprint-map.md` when the task may cross sprint boundaries.

## Execute

- Implement the smallest working slice that satisfies the task.
- Keep orchestration in `pipeline.py`, not `cli.py`.
- Keep data, features, strategies, backtest, and reports separated.
- Preserve the canonical panel, weights, and performance contracts.
- In Sprint 1, prefer concrete functions over extra abstractions.

## By Sprint

- Sprint 1: prioritize scaffold, panel preparation, rule baselines, metrics, reports, and tests.
- Sprint 2: add target generation, walk-forward logic, model wrappers, and ranking behavior without breaking Sprint 1 interfaces.
- Sprint 3: add CI, Docker, docs, and integration hardening without redesigning the data contract.

## Reject

- Live trading behavior
- Unrequested extra models
- Empty future-sprint placeholder modules
