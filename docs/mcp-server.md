# MCP Server

MarketLab now includes a stdio Model Context Protocol server exposed by the `marketlab-mcp` entrypoint.

Phase 6 keeps the server deliberately narrow:

- stdio transport only
- one user session per server process
- tools only, with no MCP resources or prompts
- repo mounts are read-only
- workspace and artifact mounts are the only writable roots
- long-running workflows execute through the installed `marketlab` CLI in subprocesses

## Tool Surface

Admin:

- `marketlab_server_info`
- `marketlab_workspace_info`

Configs:

- `marketlab_list_templates`
- `marketlab_create_config_from_template`
- `marketlab_copy_repo_config`
- `marketlab_read_config`
- `marketlab_patch_config`
- `marketlab_validate_config`

Jobs:

- `marketlab_plan_run`
- `marketlab_start_job`
- `marketlab_list_jobs`
- `marketlab_get_job_status`
- `marketlab_tail_job_logs`
- `marketlab_cancel_job`

Artifacts:

- `marketlab_list_runs`
- `marketlab_get_run_summary`
- `marketlab_list_artifacts`
- `marketlab_read_table_artifact`
- `marketlab_read_text_artifact`
- `marketlab_get_plot_artifact`
- `marketlab_compare_runs`

## Local Install

Install the MCP extra:

```bash
python -m pip install -e ".[dev]"
marketlab-mcp --help
```

For a minimal local source-tree session:

```bash
marketlab-mcp --workspace-root ./workspace --artifact-root ./artifacts --repo-root .
```

## Docker Sidecar

The generic deployment shape is a long-lived container plus one `docker exec -i` process per LLM session. The container does not keep a shared MCP daemon running.

Example helper files:

- `docker/compose.mcp.yml`
- `docs/codex.config.toml.example`
- `.vscode/mcp.json.example`

Start the container:

```bash
docker compose -f docker/compose.mcp.yml up -d --build
```

On Linux, export the host UID/GID first so the container can write to the bind-mounted `workspace/` and `artifacts/` directories:

```bash
export MARKETLAB_UID="$(id -u)"
export MARKETLAB_GID="$(id -g)"
docker compose -f docker/compose.mcp.yml up -d --build
```

Then launch one MCP stdio session through `docker exec -i`:

```bash
docker exec -i marketlab-mcp \
  marketlab-mcp \
  --workspace-root /app/workspace \
  --artifact-root /app/artifacts \
  --repo-root /app/repo
```

This is the intended bridge for generic MCP clients. The client owns the stdio session lifetime; the container just provides the packaged `marketlab-mcp` executable plus the mounted workspace and artifact volumes.

For the Codex setup flow, see [Codex MCP Setup](codex-mcp.md).
For the supported VS Code stable + GitHub Copilot setup, see [VS Code Copilot MCP Setup](mcp-vscode-copilot.md).

The checked-in compose example runs the container as `MARKETLAB_UID:MARKETLAB_GID` so Linux bind mounts stay writable. Docker Desktop on Windows usually does not need those variables.

## Offline And Network Rules

`marketlab_plan_run` is the confirmation gate before execution.

- A plan is required before `marketlab_start_job`.
- The server allows one active job plus a FIFO queue for the session.
- When `--allow-network` is not set, plans that would require data downloads are rejected.
- If a prepared panel or raw symbol cache is already available inside the workspace, offline execution is allowed.

## Config Authoring Rules

- Configs are always persisted as real YAML files in the workspace.
- Repo configs can be copied into the workspace, but they are never edited in place.
- Validation reuses the existing MarketLab config loader and adds sandbox checks for writable and readable paths.

## VS Code Copilot Contract

The checked-in VS Code helper file is `.vscode/mcp.json.example`. Copy it to `.vscode/mcp.json` in your local checkout and reload VS Code.

The sample exposes two Docker-backed stdio entries:

- `marketlab-docker-offline`
- `marketlab-docker-online`

The only difference is whether the server process receives `--allow-network`.

## Codex Contract

Codex reads MCP server definitions from `~/.codex/config.toml`.

The checked-in Codex helper snippet is `docs/codex.config.toml.example`. Copy its `mcp_servers` entries into your user-local Codex config and start a new session after the `marketlab-mcp` container is running.

The sample exposes two Docker-backed stdio entries:

- `marketlab`
- `marketlab_online`

The only difference is whether the server process receives `--allow-network`.

## Comparison Scope

`marketlab_compare_runs` stays intentionally compact in v1. It compares:

- `metrics.csv`
- `strategy_summary.csv`
- `model_summary.csv`
- Markdown section headings from `report.md`

Time-series comparisons, model binaries, and arbitrary file diffs remain out of scope.
