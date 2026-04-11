from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marketlab-mcp")
    parser.add_argument("--workspace-root", default="/app/workspace")
    parser.add_argument("--artifact-root", default="/app/artifacts")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from marketlab.mcp.server import create_server
    except RuntimeError as exc:
        parser.error(str(exc))

    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    server = create_server(
        workspace_root=Path(args.workspace_root),
        artifact_root=Path(args.artifact_root),
        repo_root=repo_root,
        allow_network=args.allow_network,
        log_level=args.log_level.upper(),
    )
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
