from __future__ import annotations

import sys
from pathlib import Path


def _repo_src_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def main(argv: list[str] | None = None) -> int:
    src_path = str(_repo_src_path())
    if src_path in sys.path:
        sys.path.remove(src_path)
    sys.path.insert(0, src_path)

    from marketlab.mcp.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
