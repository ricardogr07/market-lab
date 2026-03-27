# Guardrails

- Do not add live trading logic.
- Do not add Sprint 2 or 3 modules as empty placeholders.
- Do not introduce model abstractions into Sprint 1 code paths.
- Do not use future information when generating features or weights.
- Keep `cli.py` thin; orchestration belongs in `pipeline.py`.
- Preserve the canonical panel and performance contracts when refactoring.
