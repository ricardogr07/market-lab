# Frozen MVP

- Repo purpose: build a reproducible research-grade market lab, not a live trading bot.
- Current implementation target: Sprint 1 only.
- Frozen universe: `VOO`, `QQQ`, `SMH`, `XLV`, `IEMG`.
- Baselines in scope now:
  - `buy_hold`
  - `sma`
- Deferred to later sprints:
  - ML model wrappers
  - walk-forward evaluation
  - ranking strategy
  - Docker
  - GitHub Actions

## Repo Shape

- `src/marketlab` is the package root.
- `pipeline.py` orchestrates CLI-to-domain flow.
- `data`, `features`, `strategies`, `backtest`, and `reports` stay separate.
- Repo-local skills live under `.codex/skills`.
