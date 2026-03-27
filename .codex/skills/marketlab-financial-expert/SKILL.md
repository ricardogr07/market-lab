---
name: marketlab-financial-expert
description: MarketLab market-domain review role for data conventions, adjusted-price handling, rebalance timing, turnover, cost modeling, and portfolio metric semantics. Use when Codex needs to verify that MarketLab trading assumptions and evaluation logic are financially coherent before or after implementation.
---

# MarketLab Financial Expert

Read `../marketlab-shared-context/references/domain-rules.md` and `../marketlab-shared-context/references/data-contracts.md` first.

## Review

- Check that adjusted-price handling is internally consistent.
- Check that signal timing and execution timing do not assume impossible fills.
- Check that turnover and cost application match the stated rule.
- Check that reported metrics match the performance frame semantics.

## Escalate

- Ambiguous rebalance timing
- Adjusted versus unadjusted price misuse
- Long/short or long-only policy drift
- Metrics whose labels do not match their math
