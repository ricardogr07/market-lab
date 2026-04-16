# VS Code Copilot MCP Setup

This guide documents the supported VS Code stable path for connecting GitHub Copilot Chat to the Docker-packaged `marketlab-mcp` server for both the research workflow and the paper-review sidecar.

The design stays generic-MCP-first:

- the server is still `marketlab-mcp`
- transport is still stdio only
- the Docker container is just a packaging and volume boundary
- VS Code Copilot is a documented client, not a special server mode

## Supported Shape

- VS Code stable with GitHub Copilot Chat
- one MCP session per `docker exec -i` process
- writable `/app/workspace`
- writable `/app/artifacts` for the research sidecar
- writable `/app/repo/artifacts` for the paper sidecar
- optional read-only `/app/repo`
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

This creates a long-lived container named `marketlab-mcp` for the research sidecar or `marketlab-paper-mcp` for the paper-review sidecar. The MCP server itself is not started as a daemon. Each Copilot session starts its own foreground stdio process through `docker exec -i`.

## Add The VS Code MCP Config

Copy the checked-in sample file:

```bash
cp .vscode/mcp.json.example .vscode/mcp.json
```

On Windows PowerShell:

```powershell
Copy-Item .vscode\mcp.json.example .vscode\mcp.json
```

The sample defines four servers:

- `marketlab-docker-offline`: no `--allow-network` flag
- `marketlab-docker-online`: adds `--allow-network` for live data fetches
- `marketlab-paper-docker-offline`: paper-review sidecar, no `--allow-network`
- `marketlab-paper-docker-online`: paper-review sidecar plus `--allow-network`

Use `marketlab-docker-offline` for normal cached-panel or cached-raw workflows. Use `marketlab-paper-docker-offline` when you want Copilot to inspect the same proposal, approval, and submission state that the paper scheduler and agent containers are writing under `/app/repo/artifacts`.

## Sample Config Contract

The checked-in sample uses the current VS Code `mcp.json` workspace format and the Docker foreground stdio rule for both container shapes:

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
    },
    "marketlab-paper-docker-offline": {
      "type": "stdio",
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "marketlab-paper-mcp",
        "marketlab-mcp",
        "--workspace-root",
        "/app/workspace",
        "--artifact-root",
        "/app/repo/artifacts",
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
- Write research run outputs under `/app/artifacts`.
- Use `/app/repo/artifacts` only when you intentionally want the paper-review sidecar to read the tracked paper state written by the scheduler and agent.
- Use `marketlab_copy_repo_config` when you want to start from a tracked repo config.
- Do not point Copilot at a repo-writable container flow by default.

## Manual Verification Checklist

After copying `.vscode/mcp.json` and reloading VS Code:

1. Open Copilot Chat in agent mode and confirm `marketlab-docker-offline` and `marketlab-paper-docker-offline` are available.
2. Attach to `marketlab-docker-offline`, call `marketlab_server_info`, and verify the response reports `transport=stdio`.
3. Attach to `marketlab-docker-offline`, call `marketlab_workspace_info`, and verify the roots match `/app/workspace` and `/app/artifacts`.
4. Attach to `marketlab-paper-docker-offline`, call `marketlab_workspace_info`, and verify the artifact root is `/app/repo/artifacts`.
5. Attach to `marketlab-paper-docker-offline`, call `marketlab_get_paper_status` with `config_path="configs/experiment.qqq_paper_daily.yaml"`.
6. Use the research sidecar for `marketlab_create_config_from_template` and `marketlab_validate_config`.
7. Run one offline-safe research job through `marketlab_plan_run` and `marketlab_start_job`, then inspect it with `marketlab_get_run_summary`.

## Troubleshooting

- If VS Code cannot start the server, run `docker ps` and verify the container name is exactly `marketlab-mcp`.
- If the paper entry cannot start, verify the paper stack is up and the container name is exactly `marketlab-paper-mcp`.
- If Copilot connects but job planning says network is required, switch to the online server or preload the raw cache / prepared panel.
- If config writes fail on Linux, make sure `MARKETLAB_UID` and `MARKETLAB_GID` match `id -u` and `id -g` before starting the compose sidecar.
- If config writes fail, verify the host `workspace/` and `artifacts/` directories exist and are writable on the host.
- If paper review tools do not show the latest state, confirm the paper sidecar is using `--artifact-root /app/repo/artifacts`.
- If the repo copy tool fails, verify the repo mount is present at `/app/repo`.

## Operational Defaults

- Use `marketlab-docker-offline` for normal review and cached-run flows.
- Use `marketlab-paper-docker-offline` for proposal review and approval against the tracked paper loop state.
- Keep credentials on the host side; do not bake secrets into the image or commit them into `.vscode/mcp.json`.
- Treat this as the supported VS Code stable path. Other MCP clients can still reuse the same `docker exec -i marketlab-mcp ...` contract.
