# Codex MCP Setup

This guide documents the Codex path for attaching the Docker-packaged `marketlab-mcp` server to a new Codex session for both the research workflow and the paper-review sidecar.

The server contract stays the same as the VS Code Copilot path:

- the server is still `marketlab-mcp`
- transport is still stdio only
- the Docker container is still the packaging and volume boundary
- Codex attaches the server through `mcp_servers` in `config.toml`

## Supported Shape

- Codex CLI / Chat session with MCP support enabled
- one MCP session per `docker exec -i` process
- writable `/app/workspace`
- writable `/app/artifacts` for the research sidecar
- writable `/app/repo/artifacts` for the paper sidecar
- read-only `/app/repo`
- offline-safe by default, with an explicit network-enabled alternative

## Start The Container

From the repo root, start the sidecar that matches the flow you want:

- research / experiment workflow:

```bash
docker compose -f docker/compose.mcp.yml up -d --build
```

- paper-review workflow:

```bash
docker compose --env-file .env -f docker/compose.paper.yml up -d --build
```

On Linux, export the host UID/GID before starting either sidecar so the bind-mounted `workspace/` and `artifacts/` directories stay writable inside the container, then run the matching compose command:

```bash
export MARKETLAB_UID="$(id -u)"
export MARKETLAB_GID="$(id -g)"
docker compose -f docker/compose.mcp.yml up -d --build
```

This creates a long-lived container named `marketlab-mcp` for the research sidecar or `marketlab-paper-mcp` for the paper-review sidecar. The MCP server itself is not started as a daemon. Each Codex session starts its own foreground stdio process through `docker exec -i`.

If Docker access fails on Windows with a named-pipe permission error, fix Docker Desktop / daemon access first. Codex cannot attach the MCP server until `docker ps` works for the current user.

## Add The Codex MCP Config

Codex reads MCP server definitions from `~/.codex/config.toml`. The repo includes a checked-in example snippet:

- `docs/codex.config.toml.example`

Copy the `mcp_servers` entries from that file into your user-level Codex config. Keep this in user-local Codex config rather than committing a repo-local `.codex/config.toml`.

The example defines four servers:

- `marketlab`: offline default, no `--allow-network`
- `marketlab_online`: adds `--allow-network` for live data fetches
- `marketlab_paper`: offline default for the paper-review sidecar
- `marketlab_paper_online`: paper-review sidecar plus `--allow-network`

Use `marketlab` for normal cached-panel or cached-raw workflows. Use `marketlab_paper` when you want MCP to inspect the same proposal, approval, and submission state that the paper scheduler and agent containers are writing under `/app/repo/artifacts`.

## Sample Config Contract

The checked-in example uses Codex `config.toml` `mcp_servers` entries with the same Docker foreground stdio rule for both container shapes:

```toml
[mcp_servers.marketlab]
command = "docker"
args = [
  "exec",
  "-i",
  "marketlab-mcp",
  "marketlab-mcp",
  "--workspace-root",
  "/app/workspace",
  "--artifact-root",
  "/app/artifacts",
  "--repo-root",
  "/app/repo",
]

[mcp_servers.marketlab_paper]
command = "docker"
args = [
  "exec",
  "-i",
  "marketlab-paper-mcp",
  "marketlab-mcp",
  "--workspace-root",
  "/app/workspace",
  "--artifact-root",
  "/app/repo/artifacts",
  "--repo-root",
  "/app/repo",
]
```

Do not use `docker exec -d`. Codex MCP stdio servers must stay attached to the foreground process.

## Workspace Rules

- Keep the repo mount read-only.
- Create and patch configs inside `/app/workspace`.
- Write research run outputs under `/app/artifacts`.
- Use `/app/repo/artifacts` only when you intentionally want the paper-review sidecar to read the tracked paper state written by the scheduler and agent.
- Use `marketlab_copy_repo_config` when you want to start from a tracked repo config.
- Do not point Codex at a repo-writable container flow by default.

## Manual Verification Checklist

After updating `~/.codex/config.toml`, start a new Codex session in this repo and verify:

1. Type `/mcp` and confirm `marketlab` and `marketlab_paper` are available.
2. Type `/debug-config` and verify the active config includes the `mcp_servers.marketlab` and `mcp_servers.marketlab_paper` entries you added.
3. Attach to `marketlab` and call `marketlab_server_info`; verify the response reports `transport=stdio` and `allow_network=false`.
4. Attach to `marketlab` and call `marketlab_workspace_info`; verify the roots match `/app/workspace` and `/app/artifacts`.
5. Attach to `marketlab_paper` and call `marketlab_workspace_info`; verify the artifact root is `/app/repo/artifacts`.
6. Attach to `marketlab_paper` and call `marketlab_get_paper_status` with `config_path="configs/experiment.qqq_paper_daily.yaml"`.
7. Use `marketlab_create_config_from_template` and `marketlab_validate_config` on the research sidecar when you want a sandboxed experiment config.
8. Run one offline-safe research job through `marketlab_plan_run` and `marketlab_start_job`, then inspect it with `marketlab_get_run_summary`.

## Troubleshooting

- If Codex does not show the server in `/mcp`, restart Codex after editing `~/.codex/config.toml`.
- If `/debug-config` does not show the new server, check for a syntax error in `config.toml`.
- If Codex cannot start the server, run `docker ps` and verify the container name is exactly `marketlab-mcp`.
- If the paper entry does not start, verify the paper stack is up and the container name is exactly `marketlab-paper-mcp`.
- If Docker access fails with a Windows named-pipe permission error, fix Docker daemon access for the current user before retrying.
- If config writes fail on Linux, make sure `MARKETLAB_UID` and `MARKETLAB_GID` match `id -u` and `id -g` before starting the compose sidecar.
- If job planning says network is required, switch to `marketlab_online` or preload the raw cache / prepared panel.
- If paper review tools do not show the latest state, confirm the paper sidecar is using `--artifact-root /app/repo/artifacts`.
- If config writes fail, verify the host `workspace/` and `artifacts/` directories exist and are writable on the host.
- If the repo copy tool fails, verify the repo mount is present at `/app/repo`.

## Operational Defaults

- Use `marketlab` as the default offline Codex attachment.
- Use `marketlab_paper` when the goal is proposal review and approval against the tracked paper loop state.
- Keep credentials on the host side; do not bake secrets into the image or into repo-tracked config files.
- Treat the Docker stdio command as the single source of truth across Codex and VS Code clients.
