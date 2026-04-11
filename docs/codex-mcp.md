# Codex MCP Setup

This guide documents the Codex path for attaching the Docker-packaged `marketlab-mcp` server to a new Codex session.

The server contract stays the same as the VS Code Copilot path:

- the server is still `marketlab-mcp`
- transport is still stdio only
- the Docker container is still the packaging and volume boundary
- Codex attaches the server through `mcp_servers` in `config.toml`

## Supported Shape

- Codex CLI / Chat session with MCP support enabled
- one MCP session per `docker exec -i` process
- writable `/app/workspace` and `/app/artifacts`
- read-only `/app/repo`
- offline-safe by default, with an explicit network-enabled alternative

## Start The Container

From the repo root:

```bash
docker compose -f docker/compose.mcp.yml up -d --build
```

This creates a long-lived container named `marketlab-mcp`. The MCP server itself is not started as a daemon. Each Codex session starts its own foreground stdio process through `docker exec -i`.

If Docker access fails on Windows with a named-pipe permission error, fix Docker Desktop / daemon access first. Codex cannot attach the MCP server until `docker ps` works for the current user.

## Add The Codex MCP Config

Codex reads MCP server definitions from `~/.codex/config.toml`. The repo includes a checked-in example snippet:

- `docs/codex.config.toml.example`

Copy the `mcp_servers` entries from that file into your user-level Codex config. Keep this in user-local Codex config rather than committing a repo-local `.codex/config.toml`.

The example defines two servers:

- `marketlab`: offline default, no `--allow-network`
- `marketlab_online`: adds `--allow-network` for live data fetches

The offline entry should be your default for cached-panel or cached-raw workflows. Use the online entry only when you intend to let Codex trigger data downloads.

## Sample Config Contract

The checked-in example uses Codex `config.toml` `mcp_servers` entries with the same Docker foreground stdio rule:

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
```

Do not use `docker exec -d`. Codex MCP stdio servers must stay attached to the foreground process.

## Workspace Rules

- Keep the repo mount read-only.
- Create and patch configs inside `/app/workspace`.
- Write all run outputs under `/app/artifacts`.
- Use `marketlab_copy_repo_config` when you want to start from a tracked repo config.
- Do not point Codex at a repo-writable container flow by default.

## Manual Verification Checklist

After updating `~/.codex/config.toml`, start a new Codex session in this repo and verify:

1. Type `/mcp` and confirm the `marketlab` server is available.
2. Type `/debug-config` and verify the active config includes the `mcp_servers.marketlab` entry you added.
3. Call `marketlab_server_info` and verify the response reports `transport=stdio` and `allow_network=false`.
4. Call `marketlab_workspace_info` and verify the mounted workspace and artifact roots match `/app/workspace` and `/app/artifacts`.
5. Create one config from a template with `marketlab_create_config_from_template`.
6. Validate that config with `marketlab_validate_config`.
7. Run one offline-safe job from cached data through `marketlab_plan_run` and `marketlab_start_job`.
8. Inspect the finished run with `marketlab_get_run_summary` and one of the artifact readers.

## Troubleshooting

- If Codex does not show the server in `/mcp`, restart Codex after editing `~/.codex/config.toml`.
- If `/debug-config` does not show the new server, check for a syntax error in `config.toml`.
- If Codex cannot start the server, run `docker ps` and verify the container name is exactly `marketlab-mcp`.
- If Docker access fails with a Windows named-pipe permission error, fix Docker daemon access for the current user before retrying.
- If job planning says network is required, switch to `marketlab_online` or preload the raw cache / prepared panel.
- If config writes fail, verify the host `workspace/` and `artifacts/` directories exist and are writable on the host.
- If the repo copy tool fails, verify the repo mount is present at `/app/repo`.

## Operational Defaults

- Use `marketlab` as the default offline Codex attachment.
- Keep credentials on the host side; do not bake secrets into the image or into repo-tracked config files.
- Treat the Docker stdio command as the single source of truth across Codex and VS Code clients.
