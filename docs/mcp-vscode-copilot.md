# VS Code Copilot MCP Setup

This guide documents the supported VS Code stable path for connecting GitHub Copilot Chat to the Docker-packaged `marketlab-mcp` server.

The design stays generic-MCP-first:

- the server is still `marketlab-mcp`
- transport is still stdio only
- the Docker container is just a packaging and volume boundary
- VS Code Copilot is a documented client, not a special server mode

## Supported Shape

- VS Code stable with GitHub Copilot Chat
- one MCP session per `docker exec -i` process
- writable `/app/workspace` and `/app/artifacts`
- optional read-only `/app/repo`
- offline-safe by default, with an explicit network-enabled alternative

## Start The Container

From the repo root:

```bash
docker compose -f docker/compose.mcp.yml up -d --build
```

This creates a long-lived container named `marketlab-mcp`. The MCP server itself is not started as a daemon. Each Copilot session starts its own foreground stdio process through `docker exec -i`.

## Add The VS Code MCP Config

Copy the checked-in sample file:

```bash
cp .vscode/mcp.json.example .vscode/mcp.json
```

On Windows PowerShell:

```powershell
Copy-Item .vscode\mcp.json.example .vscode\mcp.json
```

The sample defines two servers:

- `marketlab-docker-offline`: no `--allow-network` flag
- `marketlab-docker-online`: adds `--allow-network` for live data fetches

The offline entry should be your default for cached-panel or cached-raw workflows. Use the online entry only when you intend to let Copilot trigger data downloads.

## Sample Config Contract

The checked-in sample uses the current VS Code `mcp.json` workspace format and the Docker foreground stdio rule:

```json
{
  "servers": {
    "marketlab-docker-offline": {
      "type": "stdio",
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "marketlab-mcp",
        "marketlab-mcp",
        "--workspace-root",
        "/app/workspace",
        "--artifact-root",
        "/app/artifacts",
        "--repo-root",
        "/app/repo"
      ]
    }
  }
}
```

Do not use `docker exec -d`. VS Code MCP stdio servers must stay attached to the foreground process.

## Workspace Rules

- Keep the repo mount read-only.
- Create and patch configs inside `/app/workspace`.
- Write all run outputs under `/app/artifacts`.
- Use `marketlab_copy_repo_config` when you want to start from a tracked repo config.
- Do not point Copilot at a repo-writable container flow by default.

## Manual Verification Checklist

After copying `.vscode/mcp.json` and reloading VS Code:

1. Open Copilot Chat in agent mode and confirm the `marketlab-docker-offline` server is available.
2. Call `marketlab_server_info` and verify the response reports `transport=stdio`.
3. Call `marketlab_workspace_info` and verify the mounted workspace and artifact roots match `/app/workspace` and `/app/artifacts`.
4. Create one config from a template with `marketlab_create_config_from_template`.
5. Validate that config with `marketlab_validate_config`.
6. Run one offline-safe job from cached data through `marketlab_plan_run` and `marketlab_start_job`.
7. Inspect the finished run with `marketlab_get_run_summary` and one of the artifact readers.

## Troubleshooting

- If VS Code cannot start the server, run `docker ps` and verify the container name is exactly `marketlab-mcp`.
- If Copilot connects but job planning says network is required, switch to the online server or preload the raw cache / prepared panel.
- If config writes fail, verify the host `workspace/` and `artifacts/` directories exist and are writable on the host.
- If the repo copy tool fails, verify the repo mount is present at `/app/repo`.

## Operational Defaults

- Use `marketlab-docker-offline` for normal review and cached-run flows.
- Keep credentials on the host side; do not bake secrets into the image or commit them into `.vscode/mcp.json`.
- Treat this as the supported VS Code stable path. Other MCP clients can still reuse the same `docker exec -i marketlab-mcp ...` contract.
