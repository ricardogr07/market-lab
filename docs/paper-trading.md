# Phase 7 Paper Trading

Phase 7 adds a local, paper-only execution loop around a configurable single-ETF timing strategy. It stays deliberately narrow:

- one configured ETF ticker
- one target: `direction`
- one horizon: `1` trading day
- one execution style: `long` or `cash`
- one broker family: Alpaca paper only
- one deployment shape: local Docker Compose plus a file-backed approval inbox

This is still a paper-trading MVP. It is not a live-money workflow. The tracked unattended-month config uses `QQQ`, and `VOO` ships as the first alternate comparison config.

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
- optional Telegram ops feed: `paper.notifications.telegram.enabled`

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

Behavior:

- `paper-decision` refreshes Alpaca daily bars, rebuilds the latest feature snapshot, retrains all six configured models on the rolling historical window, and writes one consensus proposal plus one evidence artifact.
- `paper-status` reads the latest persisted status plus the latest proposal summary.
- `paper-approve` records an `approve` or `reject` decision by actor `agent` or `manual`.
- `paper-agent-approve` runs the autonomous agent worker once or in a loop. It may use `openai`, `claude`, or `deterministic_consensus`, but it may only approve or reject the existing proposal as written.
- `paper-submit` enforces the approval mode, reconciles against the current paper position, and either submits one fractional `DAY` market order or records a skipped or no-op submission.
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

The tracked config defaults to `openai`, but the worker always falls back to `deterministic_consensus` when:

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
```

The paper broker path rejects non-paper trading endpoints at runtime unless the base URL is a local test server.

## Telegram Ops Feed

Telegram notifications are opt-in and stay out of YAML credentials. Enable them in the paper config:

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
