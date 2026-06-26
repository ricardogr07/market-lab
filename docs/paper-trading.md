# Phase 7 Paper Trading

Phase 7 adds a local, paper-only execution loop around a configurable single-ETF timing strategy. It stays deliberately narrow:

- one configured ETF ticker
- one target: `direction`
- one horizon: `1` trading day
- one execution style: `long` or `cash`
- one broker family: Alpaca paper only
- one deployment shape: local Docker Compose plus a file-backed approval inbox

This is still a paper-trading MVP. It is not a live-money workflow. The tracked unattended-month config uses `QQQ`, and `VOO` ships as the first alternate comparison config.

The tracked `QQQ` config is launch-ready for the unattended local month run, including Telegram notifications. The alternate `VOO` config keeps the same paper shape but leaves Telegram explicit and disabled.

Phase 8/9 adds an isolated BTC paper path with its own contract:

- BTC research must pass the strict Phase 8 regime gate before Phase 9 submits paper orders.
- The BTC model emits only `0%`, `25%`, `50%`, or `100%` target BTC exposure.
- BTC paper state uses `artifacts/btc-paper/...` and must not share QQQ state or approval inbox paths.
- Alpaca crypto paper orders use the configured crypto market order type, with `gtc` or `ioc` time-in-force.
- BTC LLM approval remains approve/reject only; the model output and target weight are persisted before the agent sees the proposal.

## Tracked Config

The tracked Phase 7 config is:

- `configs/experiment.qqq_paper_daily.yaml`
- `configs/experiment.voo_paper_daily.yaml` is the first alternate config with the same paper loop shape

It pins the current paper path to:

- `data.symbols: [QQQ]`
- `rebalance_frequency: "D"`
- `target.horizon_days: 1`
- `portfolio.ranking.mode: "long_only"`
- `long_n: 1`
- `short_n: 1`
- `min_score_threshold: 0.55`
- six configured models:
  - `logistic_regression`
  - `logistic_l1`
  - `random_forest`
  - `extra_trees`
  - `gradient_boosting`
  - `hist_gradient_boosting`
- consensus rule: `4` or more long votes out of `6`
- default execution mode: `agent_approval`
- default provider backend in the tracked config: `openai`
- required fallback backend: `deterministic_consensus`
- tracked `QQQ` config default: `paper.notifications.telegram.enabled: true`
- alternate `VOO` config default: `paper.notifications.telegram.enabled: false`

The paper path intentionally does not auto-pick the latest research winner at runtime. If the model set, threshold, or provider backend changes, do that by changing the tracked config and reviewing the research outcome first.

## Command Surface

New CLI commands:

```bash
python scripts/run_marketlab.py paper-decision --config configs/experiment.qqq_paper_daily.yaml
python scripts/run_marketlab.py paper-status --config configs/experiment.qqq_paper_daily.yaml
python scripts/run_marketlab.py paper-approve --config configs/experiment.qqq_paper_daily.yaml --proposal-id <id> --decision approve --actor agent
python scripts/run_marketlab.py paper-agent-approve --config configs/experiment.qqq_paper_daily.yaml --once
python scripts/run_marketlab.py paper-submit --config configs/experiment.qqq_paper_daily.yaml
python scripts/run_marketlab.py paper-scheduler --config configs/experiment.qqq_paper_daily.yaml --once
python scripts/run_marketlab.py paper-report --config configs/experiment.qqq_paper_daily.yaml --start 2026-04-13 --end 2026-05-15
```

BTC paper uses the same commands with the isolated config:

```bash
python scripts/run_marketlab.py paper-decision --config configs/experiment.btc_paper_daily.yaml
python scripts/run_marketlab.py paper-agent-approve --config configs/experiment.btc_paper_daily.yaml --once
python scripts/run_marketlab.py paper-submit --config configs/experiment.btc_paper_daily.yaml
```

Behavior:

- `paper-decision` refreshes Alpaca daily bars, rebuilds the latest feature snapshot, retrains all six configured models on the rolling historical window, and writes one consensus proposal plus one evidence artifact.
- `paper-status` reads the latest persisted status plus the latest proposal summary.
- `paper-approve` records an `approve` or `reject` decision by actor `agent` or `manual`.
- `paper-agent-approve` runs the autonomous agent worker once or in a loop. It may use `openai`, `claude`, or `deterministic_consensus`, but it may only approve or reject the existing proposal as written.
- `paper-submit` enforces the approval mode, refreshes the latest broker order status when a submission already exists, reconciles against the current paper position, and either submits one buy-side notional `DAY` market order, one sell-side fractional `DAY` market order, or records a skipped or no-op submission.
- `paper-scheduler` is the long-running local loop used by Docker Compose.
- `paper-report` reconstructs the paper-run outcome over a chosen date range and compares the realized paper path, the consensus path, each model path, `buy_hold`, and `sma`.

## Approval Modes

`paper.execution_mode` supports:

- `autonomous`: submit without approval
- `agent_approval`: require `paper-approve ... --actor agent`
- `manual_approval`: require `paper-approve ... --actor manual`

If approval is still missing when the submission phase runs, the trade is skipped and the submission state records the reason.

## Agent Backends

`paper.agent_backend` supports:

- `openai`
- `claude`
- `deterministic_consensus`

The tracked `QQQ` config currently uses `claude`, but the worker always falls back to `deterministic_consensus` when:

- the provider key is missing
- the provider times out
- the provider call fails
- the provider returns invalid structured output

The LLM is not allowed to invent a different trade. It only approves or rejects the consensus proposal and records a short rationale.

## Persisted State

Phase 7 uses a file-backed approval inbox and per-trade state under `artifacts/paper/`.

The main persisted surfaces are:

- inbox proposals: `artifacts/paper/inbox/*.json`
- trade proposal: `artifacts/paper/state/trades/<trade-date>/proposal.json`
- trade evidence: `artifacts/paper/state/trades/<trade-date>/evidence.json`
- trade approval: `artifacts/paper/state/trades/<trade-date>/approval.json`
- order preview: `artifacts/paper/state/trades/<trade-date>/order_preview.json`
- account snapshot: `artifacts/paper/state/trades/<trade-date>/account_snapshot.json`
- submission state: `artifacts/paper/state/trades/<trade-date>/submission.json`
- order polling result: `artifacts/paper/state/trades/<trade-date>/order_status.json`
- month-run reports: `artifacts/paper/reports/<start>_<end>/`
- latest status summary: `artifacts/paper/state/status.json`

This is the shared contract between the CLI, the scheduler, and the MCP paper tools.

## PostgreSQL Control State

Filesystem remains the default paper persistence backend. PostgreSQL is an
opt-in P9-08 control-state backend for QQQ preparation; it preserves the
existing JSON artifacts as a temporary local review mirror. It does not add
Blob Storage, an outbox, Service Bus, Azure resources, or any change to broker
or approval behavior.

To use it locally, make an untracked copy of the paper configuration and set
only the backend in that copy:

```yaml
paper:
  persistence_backend: "postgres"
```

The connection is configured exclusively with
`MARKETLAB_PAPER_POSTGRES_DSN`. Do not put a DSN, password, or other database
credential in YAML, checked-in configuration, logs, or command output. The
included `docker/compose.postgres.yml` service is only a disposable
development database for the PostgreSQL test suite; it is not a QQQ runtime
service.

Initialize or upgrade the schema explicitly before running any paper command:

```bash
export MARKETLAB_PAPER_POSTGRES_DSN='<local DSN supplied outside the repository>'
python scripts/run_marketlab.py paper-db-migrate --config configs/local.qqq-postgres.yaml
```

`paper-db-migrate` requires `paper.persistence_backend: "postgres"`, takes a
PostgreSQL advisory lock, applies only new numbered SQL migrations, validates
the checksum of every recorded migration, and reports the resulting schema
version. It does not create providers, call a broker, send notifications, or
run scheduler work. Normal paper commands never apply migrations implicitly.

Migrations are forward-only and append-only. Never edit, remove, reorder, or
reuse a numbered migration after it has been applied. There are no destructive
down migrations. Before an upgrade, take and verify a PostgreSQL backup using
your approved database tooling. If an upgrade must be rolled back, restore the
database backup and then add a forward corrective migration; do not manually
drop tables or attempt a down migration. P9-11 owns the production import and
restore rehearsal.

## P9-11 QQQ Import Runbooks

P9-11 prepares QQQ state import and restore operations; it does not run a live
import by itself, enable Azure schedules, consume Service Bus messages, call
providers, send Telegram notifications, or submit broker orders. P9-12 owns
the ten-trading-day UAT parity report and failure drills. P9-13 owns
paper-prod cutover and the final state delta import.

Use an untracked PostgreSQL config for import review:

```yaml
paper:
  persistence_backend: "postgres"
```

Set the DSN only in the operator environment:

```bash
export MARKETLAB_PAPER_POSTGRES_DSN="<postgres-dsn-from-approved-secret-store>"
```

### Pre-Import Checklist

- confirm the local QQQ scheduler is the authoritative runtime
- confirm `configs/experiment.qqq_paper_daily.yaml` has not changed strategy,
  model, schedule, approval, or broker semantics
- confirm `MARKETLAB_PAPER_POSTGRES_DSN` points at the approved dev database
- confirm `paper-db-migrate` has converged successfully
- confirm no Terraform plan, apply, import, destroy, Azure login, schedule
  enablement, or broker secret change is part of the import session
- record the operator, source artifact root, image or commit SHA, and reviewed
  config path outside tracked files

### Backup Before Import

Take a PostgreSQL backup with approved operator tooling before any apply. Use
placeholder names in runbooks and tickets:

```bash
pg_dump "<postgres-dsn-from-approved-secret-store>" \
  --format=custom \
  --file "<secure-backup-path>/qqq-paper-dev-before-import.dump"
```

Verify the backup into a scratch database before continuing:

```bash
pg_restore \
  --dbname "<scratch-restore-dsn>" \
  --clean \
  --if-exists \
  "<secure-backup-path>/qqq-paper-dev-before-import.dump"
```

### Dry-Run Import

Dry-run is the default and is safe to repeat:

```bash
python scripts/run_marketlab.py paper-state-import \
  --config configs/local.qqq-postgres.yaml \
  --source-state-dir artifacts/paper/state \
  --source-inbox-dir artifacts/paper/inbox \
  --dry-run \
  --report-path artifacts/paper/state/imports/qqq-dev-dry-run.json
```

Review `counts`, every `conflicting` entry, and the aggregate checksum. Do not
continue if any conflict is present or any source file is malformed.

### Apply Import

Apply only after the backup and dry-run report are accepted:

```bash
python scripts/run_marketlab.py paper-state-import \
  --config configs/local.qqq-postgres.yaml \
  --source-state-dir artifacts/paper/state \
  --source-inbox-dir artifacts/paper/inbox \
  --apply \
  --report-path artifacts/paper/state/imports/qqq-dev-apply.json
```

Repeat the same command once after apply. The repeat should report identical
records as skipped and preserve the same source checksums.

### Checksum Report Review

The import report records source paths, logical targets, SHA-256 checksums,
per-surface counts, mode, timestamps, and an aggregate manifest checksum.
Notification audits and local reports are checksum-only review artifacts in
P9-11 because PostgreSQL remains canonical control state and Blob remains the
review surface.

### Blob Sync After Accepted Import

Blob synchronization is a separate operator step:

```bash
python scripts/run_marketlab.py paper-blob-sync \
  --config configs/local.qqq-postgres-azure-blob.yaml
```

Do not hide Blob sync inside `paper-state-import`. Review the import report
first, then run Blob sync with the approved environment and deployment ID.

### PostgreSQL Restore Rehearsal

Restore the pre-import backup into a scratch database and run read-only status
checks against the scratch DSN:

```bash
export MARKETLAB_PAPER_POSTGRES_DSN="<scratch-restore-dsn>"
python scripts/run_marketlab.py paper-status \
  --config configs/local.qqq-postgres.yaml
```

The restore rehearsal must prove that proposals, status, submissions, and
order-status payloads can be read through the existing repository APIs.

### Blob Restore Rehearsal

Use approved Azure Blob tooling to copy versioned artifacts into a scratch
prefix such as:

```text
paper/<environment>/<deployment-id>/restore-check/<yyyy-mm-dd>/
```

Compare restored proposal, evidence, approval, submission, order-status,
notification, status, and report artifacts against the checksum report before
P9-12 parity work begins.

### Rollback To Local Scheduler

If import validation fails before cutover, keep the local QQQ scheduler as the
authoritative runtime and restore PostgreSQL from the verified backup. Do not
enable Azure jobs. Do not run broker-facing jobs from Azure. Keep the local
Docker stack available:

```bash
docker compose --env-file .env -f docker/compose.paper.yml up -d --build
```

Production cutover requires P9-13 approval, local scheduler stop, in-flight
state checks, final delta import, and explicit Azure job enablement.

## P9-12 QQQ UAT Parity Runbooks

P9-12 records ten-trading-day QQQ UAT parity evidence and supervised failure
drills. It does not start a live UAT session by itself, enable Azure schedules,
consume Service Bus messages, call providers, send Telegram notifications,
submit broker orders, change strategy behavior, run Terraform plan/apply, or
perform paper-prod cutover. P9-13 owns production cutover, local scheduler
stop, final state delta import, and Azure job enablement.

### Pre-UAT Checklist

- confirm P9-10 dev infrastructure exists with schedules disabled unless a
  separate supervised UAT session explicitly approves them
- confirm P9-11 import and restore runbooks are complete and accepted
- confirm `configs/experiment.qqq_paper_daily.yaml` remains the canonical QQQ
  paper config
- confirm local QQQ artifacts and the shadow/UAT artifact export are complete
  for the reviewed date range
- confirm no broker-facing Azure submission job is enabled during the parity
  window
- record operator, commit SHA, config path, source artifact roots, and reviewed
  date range outside tracked files

### Generate Parity Report

Run the report against explicit local and shadow artifact roots:

```bash
python scripts/run_marketlab.py paper-parity-report \
  --config configs/experiment.qqq_paper_daily.yaml \
  --local-state-dir artifacts/paper/state \
  --shadow-state-dir "<shadow-export-root>/state" \
  --start "<yyyy-mm-dd>" \
  --end "<yyyy-mm-dd>" \
  --report-path artifacts/paper/uat/qqq-parity.json \
  --markdown-path artifacts/paper/uat/qqq-parity.md
```

The default minimum evidence window is `10` consecutive weekdays with proposal
and evidence artifacts present on both sides. The command compares proposal,
evidence, approval, submission, order preview, account snapshot, order status,
status, notification inventory, and report inventory surfaces. It writes local
report artifacts only.

### Difference Explanations

If a difference is expected or accepted by review, record it in an untracked
explanations file keyed by the report difference ID:

```json
{
  "<difference-id>": {
    "status": "accepted",
    "explanation": "<operator-reviewed reason>"
  }
}
```

Allowed statuses are `accepted`, `expected`, and `blocking`. Re-run the report
with the explanations file:

```bash
python scripts/run_marketlab.py paper-parity-report \
  --config configs/experiment.qqq_paper_daily.yaml \
  --local-state-dir artifacts/paper/state \
  --shadow-state-dir "<shadow-export-root>/state" \
  --start "<yyyy-mm-dd>" \
  --end "<yyyy-mm-dd>" \
  --explanations artifacts/paper/uat/qqq-parity-explanations.json \
  --report-path artifacts/paper/uat/qqq-parity-reviewed.json \
  --markdown-path artifacts/paper/uat/qqq-parity-reviewed.md
```

The report is accepted only when the ten-day evidence window passes and every
difference is absent, `accepted`, or `expected`. A `blocking` or unexplained
difference must stop cutover preparation.

### Failure-Drill Evidence

Store supervised drill notes and artifacts outside tracked files. P9-12
acceptance requires reviewed evidence for:

- duplicate delivery
- provider timeout
- broker timeout
- rejected order
- partial fill
- stale data
- missing bar
- queue retry
- dead-letter recovery
- PostgreSQL restore
- Blob restore

Do not use live Azure resource names, DSNs, tfvars, backend files, Terraform
state, Terraform plans, or secret values in tracked documentation.

## P9-13 QQQ Paper-Prod Cutover Runbooks

P9-13 prepares the paper-prod cutover after P9-12 parity and failure-drill
evidence has been accepted. It does not authorize a live cutover by itself.
Stopping the local scheduler, importing the final state delta, applying
Terraform, changing secrets, enabling Azure triggers, and running the first
paper-prod cycle require a separate supervised operator session.

### Pre-Cutover Checklist

- confirm the accepted P9-12 parity report and failure-drill evidence are
  recorded outside tracked files
- confirm `configs/experiment.qqq_paper_daily.yaml` has not changed strategy,
  model, schedule, approval, or broker semantics
- confirm paper-prod Terraform uses `qqq-paper-prod.tfstate` from an untracked
  backend file
- confirm the immutable image digest, Key Vault secret IDs, PostgreSQL DSN,
  operator-approved firewall rules, and paper-prod tfvars are retained outside
  the repository
- confirm Azure QQQ uses a separate Alpaca paper account and a separate
  Telegram bot/chat from the local QQQ runner
- confirm `enable_telegram_notifications = false` for dev and early paper-prod
  smoke tests, then set it to `true` only after the separate Azure QQQ
  Telegram bot/chat and Key Vault secret IDs are accepted
- confirm Alpaca endpoints are paper-only before any broker-facing Azure job is
  smoke-tested
- record operator, commit SHA, P9-12 evidence URI, final import evidence URI,
  backup/restore evidence URI, rollback evidence URI, and alert evidence URI
  outside tracked files

### Sequential Azure Cutover Gates

Use the full
[QQQ Azure cutover operator plan](phase9/QQQ-AZURE-CUTOVER-OPERATOR-PLAN.md)
when moving authority from local Docker to Azure. The required sequence is:

1. merge the selected source commit, including P9-14 and any Terraform
   notification-gate change
2. create separate Azure QQQ Alpaca paper and Telegram credentials outside
   tracked files
3. apply dev infrastructure with jobs, triggers, broker secret refs, and
   Telegram delivery disabled
4. publish an immutable image digest from the accepted commit
5. smoke-test dev jobs with no broker authority
6. accept QQQ parity, failure-drill, backup/restore, rollback, alert, and
   dead-letter evidence while local remains authoritative
7. create paper-prod infrastructure and jobs with triggers and broker secret
   refs disabled
8. attach broker/provider/notification secret refs with triggers disabled
9. stop only `marketlab-paper-scheduler` and `marketlab-paper-agent`, then run
   final import and manual paper-prod smoke
10. enable scheduler and Service Bus triggers only after smoke acceptance
11. accept the first Azure-owned cycle before starting P9-14 closeout

Never run local and Azure QQQ scheduler/agent paths as simultaneous
authorities.

### Stop Local Authority

Stop the local QQQ scheduler and agent before enabling any paper-prod Azure
trigger:

```bash
docker compose --env-file .env -f docker/compose.paper.yml stop marketlab-paper-scheduler marketlab-paper-agent
```

Verify local containers did not leave an unresolved proposal, missing approval,
duplicate submission, or non-terminal order. Use the normal status and artifact
review surfaces:

```bash
python scripts/run_marketlab.py paper-status --config configs/experiment.qqq_paper_daily.yaml
```

Do not continue if an order is still open, pending, partially filled without an
accepted operator note, or otherwise non-terminal.

### Final State Delta Import

Run a final dry-run import from the local QQQ artifact root:

```bash
python scripts/run_marketlab.py paper-state-import \
  --config configs/local.qqq-postgres.yaml \
  --source-state-dir artifacts/paper/state \
  --source-inbox-dir artifacts/paper/inbox \
  --dry-run \
  --report-path artifacts/paper/state/imports/qqq-paper-prod-final-dry-run.json
```

Apply only after the dry-run, backup, restore, and rollback evidence are
accepted:

```bash
python scripts/run_marketlab.py paper-state-import \
  --config configs/local.qqq-postgres.yaml \
  --source-state-dir artifacts/paper/state \
  --source-inbox-dir artifacts/paper/inbox \
  --apply \
  --report-path artifacts/paper/state/imports/qqq-paper-prod-final-apply.json
```

Synchronize review artifacts after the accepted import:

```bash
python scripts/run_marketlab.py paper-blob-sync \
  --config configs/local.qqq-postgres-azure-blob.yaml
```

### Paper-Prod Azure Smoke Test

Create paper-prod jobs with schedules and Service Bus approval triggering still
disabled first. Manually invoke the reviewed jobs in this order:

1. `paper-db-migrate`
2. `paper-blob-sync`
3. `paper-outbox-deliver`
4. `paper-service-bus-receive`
5. `paper-notifications-deliver`
6. `paper-scheduler --once`
7. `paper-agent-approve --once`

Review PostgreSQL, Blob, Service Bus, notification audit, alert, and paper
broker evidence after each job. Do not enable triggers until the smoke test is
accepted.

### Enable Paper-Prod Triggers

Only after the smoke test passes, a supervised apply may set:

```hcl
environment                         = "paper-prod"
create_jobs                         = true
enable_broker_secret_refs           = true
enable_scheduler_schedule           = true
enable_service_bus_approval_trigger = true
```

The paper-prod Terraform gate also requires a non-placeholder immutable image
digest and reviewed HTTPS evidence URIs for P9-12 parity, final import,
backup/restore, rollback, and alerts.

### First Production-Paper Cycle

For the first scheduled paper-prod cycle, verify:

- scheduler due-time enforcement still follows `America/New_York`
- one proposal is written for the expected QQQ trade date
- approval remains approve/reject only
- no duplicate broker submission is created
- Blob artifacts mirror PostgreSQL state
- Service Bus messages are settled or dead-lettered according to the reviewed
  failure-drill contract
- alert evidence shows both job failure and missing-evidence checks are active

### Rollback

If any cutover gate fails, disable Azure scheduler and Service Bus triggers,
restore PostgreSQL from the accepted backup, restore Blob artifacts from the
reviewed snapshot or versioned prefix, and keep the local QQQ scheduler disabled
until the operator explicitly chooses the recovery authority. Restart the local
runner only after the restored state is reviewed:

```bash
docker compose --env-file .env -f docker/compose.paper.yml up -d --build marketlab-paper-scheduler marketlab-paper-agent
```

P9-14 owns the ten-trading-day post-cutover observation window and closeout.

## P9-14 QQQ Post-Cutover Closeout Runbooks

P9-14 documents the post-cutover observation and closeout gates after P9-13
paper-prod cutover has completed and the first production-paper cycle has been
accepted. It does not run the live observation, apply Terraform, change
secrets, enable Azure jobs, submit broker orders, archive state by itself, or
make the P9-15 BTC final evidence decision.

### Observation Checklist

Observe QQQ Azure paper-prod for `10` additional NYSE trading days. Keep
`configs/experiment.qqq_paper_daily.yaml` unchanged and record the operator,
commit SHA, paper-prod cutover evidence URI, first-cycle evidence URI,
observation date range, rollback rehearsal URI, and final closeout report URI
outside tracked files.

For each observed trading day, review exported evidence for:

- PostgreSQL proposal, approval, submission, order-status, and latest-status
  state
- Blob proposal, evidence, approval, submission, order-status, notification,
  and report artifacts
- Service Bus delivery, settlement, retry, and dead-letter state
- notification audit records and Telegram delivery state
- Azure job failure alerts and missing-evidence alerts
- Alpaca paper broker order state and account state

Do not continue closeout if any proposal is unresolved, any order is
non-terminal without an accepted operator note, any broker submission appears
duplicated, or any alert, dead letter, failed job, or reconciliation issue is
unresolved.

### Rollback Runner Rehearsal

Keep local production scheduling disabled during the observation window. The
rollback runner may be preserved only as a reviewed manual recovery path that
uses the same PostgreSQL and Blob adapters. It must not resume local authority
or schedule production paper phases unless a separate supervised incident
decision chooses local recovery.

Record rollback rehearsal evidence outside tracked files. The evidence must
show that the local runner can be started against reviewed PostgreSQL and Blob
state, can read QQQ status, and can remain disabled again after the rehearsal.

### Generate Closeout Report

Export paper-prod evidence to local review roots, then run:

```bash
python scripts/run_marketlab.py paper-closeout-report \
  --config configs/experiment.qqq_paper_daily.yaml \
  --paper-prod-state-dir "<paper-prod-export-root>/state" \
  --paper-prod-artifact-dir "<paper-prod-export-root>/artifacts" \
  --start "<yyyy-mm-dd>" \
  --end "<yyyy-mm-dd>" \
  --rollback-evidence "<paper-prod-export-root>/rollback-evidence.json" \
  --report-path artifacts/paper/closeout/qqq-paper-prod-closeout.json \
  --markdown-path artifacts/paper/closeout/qqq-paper-prod-closeout.md
```

The default minimum evidence window is `10` observed trading days. The command
checks decision, approval, submission, reconciliation, notification inventory,
report inventory, duplicate broker-submission identifiers, alert evidence,
dead-letter evidence, failed-job evidence, non-terminal order evidence, and
rollback rehearsal acceptance. It reads exported files only.

Use these optional exported evidence files under
`<paper-prod-export-root>/artifacts` when issues exist:

```text
alerts.json
dead_letters.json
failed_jobs.json
non_terminal_orders.json
```

Each issue item must have `status` set to `accepted`, `expected`, or
`resolved` before the closeout report can pass. A missing rollback evidence
file, `blocking` status, or unresolved item must stop closeout.

### Archive Local State

Archive the old local QQQ state only after the closeout report is accepted.
Move or snapshot local production artifacts into a dated archive outside the
active scheduler path. Preserve local Docker for development and manual
recovery drills, not for scheduled production QQQ paper phases.

Record the archive location, checksum manifest, operator, commit SHA, closeout
report URI, rollback rehearsal URI, and final acceptance decision outside
tracked files.

### Tracked Documentation Boundary

Do not use live Azure resource names, DSNs, tfvars, backend files, Terraform
state, Terraform plans, Key Vault secret IDs, broker account identifiers, or
secret values in tracked documentation.

## Alpaca Environment

Keep credentials in environment variables, not YAML. For local runs, copy `.env.example` to `.env` in the repo root and fill in the paper credentials:

```bash
cp .env.example .env
```

The local CLI and MCP paper path will load `.env` from the current working directory when those variables are not already present in the process environment.

Example `.env` values:

```bash
ALPACA_API_KEY_ID="..."
ALPACA_API_SECRET_KEY="..."
ALPACA_DATA_BASE_URL="https://data.alpaca.markets"
ALPACA_TRADING_BASE_URL="https://paper-api.alpaca.markets"
ALPACA_DATA_FEED="iex"
ALPACA_TIMEOUT_SECONDS="30"
OPENAI_API_KEY="..."
ANTHROPIC_API_KEY="..."
TELEGRAM_BOT_TOKEN="..."
TELEGRAM_CHAT_ID="..."
MARKETLAB_PAPER_TELEGRAM_ENABLED=false
MARKETLAB_PAPER_TELEGRAM_ALLOWED_EXPERIMENTS=qqq_paper_daily
```

The paper broker path rejects non-paper trading endpoints at runtime unless the base URL is a local test server.

For BTC, copy `.env.btc-paper.example` to `.env.btc-paper` and use a separate Alpaca paper account, separate `OPENAI_API_KEY`, separate `ANTHROPIC_API_KEY`, and separate Telegram bot/chat values. Keep `MARKETLAB_PAPER_TELEGRAM_ALLOWED_EXPERIMENTS=btc_paper_daily` so the BTC container cannot notify for QQQ experiments.

## Local Docker

Run the tracked QQQ local sidecars with:

```bash
docker compose -f docker/compose.paper.yml up --build
```

Run the isolated BTC sidecars with:

```bash
docker compose --env-file .env.btc-paper -f docker/compose.btc-paper.yml up --build
```

The BTC compose file uses `marketlab-btc-paper-scheduler`, `marketlab-btc-paper-agent`, and `marketlab-btc-paper-mcp`, mounts `../artifacts-btc-paper` to `/app/repo/artifacts`, and points only at `configs/experiment.btc_paper_daily.yaml`.

## Telegram Ops Feed

Telegram credentials stay out of YAML. The tracked `QQQ` config already enables the ops feed, and the alternate `VOO` config leaves it explicit but disabled. Set `MARKETLAB_PAPER_TELEGRAM_ENABLED=false` in `.env` to force Telegram delivery off for local CLI, scheduler, agent, and paper MCP runs even when the YAML config enables it. Set it to `true` to force delivery on for a config that leaves Telegram disabled.

Use `MARKETLAB_PAPER_TELEGRAM_ALLOWED_EXPERIMENTS=qqq_paper_daily` to restrict real delivery to the tracked QQQ paper experiment. This keeps fixture and smoke-test configs such as `phase7_paper_fixture` from sending Telegram messages even if their YAML enables notifications.

To enable it in another paper config without an env override:

```yaml
paper:
  notifications:
    telegram:
      enabled: true
```

When enabled, the shared paper service layer sends one plain-text Telegram message per event for:

- `paper-decision`: `proposal_created`, `existing_proposal`, `non_trading_day`, `stale_signal_date`
- `paper-approve`: `approved`, `rejected`
- `paper-submit`: `submitted`, `no_trade_required`, `skipped`, `existing_submission`
- `paper-error`: uncaught scheduler or agent-loop failures, deduplicated until the next successful iteration

Notifications are advisory only. Paper decision, approval, and submit still complete even if Telegram delivery fails or credentials are missing.

Every notification attempt is also persisted under:

- `artifacts/paper/state/notifications/*.json`

These audit records include the stage, outcome, message body, delivery result, and any delivery error. They do not replace the proposal, approval, or submission state files.

## Structured Observability

The paper entrypoints now emit structured JSON logs to `stderr`. This keeps the existing CLI and loop `stdout` contracts unchanged for path outputs, JSON summaries, and MCP protocol traffic.

The persisted paper artifacts remain the canonical audit and debugging surface. Proposal, approval, submission, notification, and status files are still the source of truth for paper state.

For the long-running local stack, `docker logs` remains the operational tail path for scheduler, agent, and MCP execution traces because those structured records are written to container `stderr`.

## Docker Compose Loop

The checked-in local stack is:

- `docker/compose.paper.yml`

It starts:

- `marketlab-paper-scheduler`
- `marketlab-paper-agent`
- `marketlab-paper-mcp`

Start the stack:

```bash
docker compose --env-file .env -f docker/compose.paper.yml up -d --build
```

On Linux, export the host UID and GID first so the bind-mounted directories stay writable:

```bash
export MARKETLAB_UID="$(id -u)"
export MARKETLAB_GID="$(id -g)"
docker compose --env-file .env -f docker/compose.paper.yml up -d --build
```

The scheduler uses the tracked repo config at `/app/repo/configs/experiment.qqq_paper_daily.yaml` and a writable artifact submount at `/app/repo/artifacts`.

On each loop, the scheduler also refreshes the latest persisted submission against Alpaca so later broker-side terminal states such as `filled` or `rejected` are written back into `submission.json` and `order_status.json`.

The agent worker uses the same tracked config and artifact mount, so the approval loop and the scheduler see the same proposal, approval, and submission state.

If `paper.notifications.telegram.enabled` is true, all three paper containers need the Telegram env vars because notifications can be emitted from the scheduler, the agent worker, and MCP-driven approvals.

The matching MCP sidecar should be launched with the same artifact root so it sees the same proposal and submission files:

```bash
docker exec -i marketlab-paper-mcp \
  marketlab-mcp \
  --workspace-root /app/workspace \
  --artifact-root /app/repo/artifacts \
  --repo-root /app/repo
```

The checked-in client samples now include paper-specific entries for this sidecar:

- `docs/codex.config.toml.example`: `marketlab_paper`, `marketlab_paper_online`
- `.vscode/mcp.json.example`: `marketlab-paper-docker-offline`, `marketlab-paper-docker-online`

## MCP Paper Tools

The MCP server now also exposes a narrow paper-review surface:

- `marketlab_list_paper_proposals`
- `marketlab_read_paper_proposal`
- `marketlab_get_paper_status`
- `marketlab_decide_paper_proposal`

These tools intentionally stop at review and approval. Order submission still happens through the CLI-backed scheduler path.

There is no separate Telegram MCP tool. `marketlab_decide_paper_proposal` uses the same shared paper approval service, so MCP approvals trigger the same Telegram notification and audit artifact behavior as CLI or agent approvals.

## Fixed Defaults

Phase 7 defaults are fixed on purpose:

- schedule timezone: `America/New_York`
- decision time: `16:10`
- submission time: `19:05`
- order style: `market` plus `day`
- position sizing: full-equity fractional exposure in the configured ETF when long, `0%` when in cash
- execution policy: deterministic `4-of-6` consensus proposals
- no shorts
- no live-money path
