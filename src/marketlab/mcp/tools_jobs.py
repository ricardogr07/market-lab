from __future__ import annotations

from typing import Any

from marketlab.mcp.jobs import MarketLabJobManager


def register_job_tools(
    mcp: Any,
    *,
    jobs: MarketLabJobManager,
) -> None:
    del mcp
    del jobs
