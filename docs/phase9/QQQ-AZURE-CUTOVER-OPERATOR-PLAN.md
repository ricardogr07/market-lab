# QQQ Azure Paper Trading Cutover Operator Plan

## Summary

This plan is the supervised sequence to move QQQ paper trading authority from the local
`marketlab-paper-scheduler` and `marketlab-paper-agent` containers to Azure
Container Apps Jobs. It is an operator runbook, not an automation script. It
does not apply Terraform, create secrets, stop containers, enable triggers, or
submit broker orders by itself.

The release source must be reviewed before any Azure image is built. P9-14
closeout work should be merged, or a different release commit SHA must be
explicitly chosen and recorded outside tracked files.

## Source And Secret Readiness

- keep `configs/experiment.qqq_paper_daily.yaml` as the canonical QQQ config
- use a separate Alpaca paper account and paper-only API key pair for Azure QQQ
- keep Claude as the approval backend unless a separate source change updates
  the QQQ approval semantics
- store `ANTHROPIC_API_KEY`, Alpaca credentials, PostgreSQL DSN, Telegram bot
  token, and Telegram chat ID in Key Vault; tracked files may contain only
  placeholders
- use a separate Telegram bot/chat for Azure QQQ, and keep
  `enable_telegram_notifications = false` until that bot/chat is verified
- keep all tfvars, backend files, Terraform plans, Terraform state, DSNs, Key
  Vault secret IDs, broker account IDs, Telegram chat IDs, and evidence exports
  outside tracked files

## Sequential Gates

1. Source readiness: merge P9-14 and any Terraform notification-gate change,
   validate the repo, and record the release commit SHA.
2. Operator secret setup: create the separate Alpaca paper account and Telegram
   chat, verify paper-only endpoints, and prepare Key Vault secret IDs outside
   git.
3. Azure bootstrap: apply the QQQ Terraform root only in a supervised session
   with `environment = "dev"`, `create_jobs = false`, triggers disabled,
   broker secret refs disabled, and Telegram delivery disabled.
4. Immutable image: build from the accepted commit, push to the environment ACR,
   and record the non-placeholder `sha256:` digest.
5. Dev job smoke: set `create_jobs = true` while broker refs, triggers, and
   Telegram delivery remain disabled; manually run only non-authoritative jobs.
6. Shadow/parity: keep local QQQ scheduler and agent as authority while Azure is
   non-authoritative; accept parity, backup/restore, rollback, alert, and
   dead-letter evidence.
7. Paper-prod infrastructure: use `qqq-paper-prod.tfstate`, create paper-prod
   resources with jobs and triggers disabled, push the reviewed image digest,
   then create jobs without broker secret refs.
8. Broker secret activation: set `enable_broker_secret_refs = true` with
   triggers disabled; verify separate Alpaca, Anthropic, and Telegram secret
   resolution before any broker-facing scheduled execution.
9. Cutover day: run local status, stop only `marketlab-paper-scheduler` and
   `marketlab-paper-agent`, perform final import and Blob sync, then manually
   smoke-test paper-prod jobs in the reviewed order.
10. Trigger activation: set paper-prod jobs, broker refs, scheduler trigger, and
    Service Bus approval trigger on only after manual smoke acceptance.
11. First cycle and closeout: accept the first Azure-owned QQQ cycle, observe 10
    additional NYSE trading days, then run `paper-closeout-report` against
    operator-exported paper-prod evidence.

## Cutover Commands

Run local status before stopping local authority:

```bash
python scripts/run_marketlab.py paper-status --config configs/experiment.qqq_paper_daily.yaml
```

Stop only QQQ local authority:

```bash
docker compose --env-file .env -f docker/compose.paper.yml stop marketlab-paper-scheduler marketlab-paper-agent
```

Do not stop all `marketlab-*` containers as part of authority transfer.
`marketlab-paper-mcp` may remain available for inspection unless the operator
explicitly chooses to stop it.

Manual paper-prod smoke order after final import:

1. `paper-db-migrate`
2. `paper-blob-sync`
3. `paper-outbox-deliver`
4. `paper-service-bus-receive`
5. `paper-notifications-deliver`
6. `paper-scheduler --once`
7. `paper-agent-approve --once`

## Rollback Rules

- before trigger activation, leave Azure triggers disabled and keep local QQQ
  authority active until the cutover-day stop
- after authority transfer, first disable Azure scheduler and Service Bus
  triggers, then restore PostgreSQL and Blob from accepted backup evidence
- restart local `marketlab-paper-scheduler` and `marketlab-paper-agent` only
  after restored state is reviewed and the operator chooses local authority
- never run local and Azure QQQ scheduler/agent as simultaneous authorities

## Validation

Source validation before publication:

```bash
py -3.14 -m pytest -q tests/unit/test_phase9_azure_bootstrap.py tests/unit/test_phase9_plan_docs.py
python -m tox -e terraform
python -m ruff check .
python -m mkdocs build --strict
git diff --check
```

Full preflight remains the publication gate when the local Windows workspace is
free of generated artifact file locks.

P9-15 BTC evidence work remains separate and must not be mixed into this QQQ
Azure cutover.
