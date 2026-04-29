from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tests._paper_fakes import (
    FakeAlpacaBroker,
    FakeAlpacaProvider,
    write_phase7_paper_config,
)

from marketlab.config import load_config
from marketlab.log import configure_logging
from marketlab.mcp.tools_paper import register_paper_tools
from marketlab.mcp.workspace import WorkspaceSandbox
from marketlab.paper.service import run_paper_decision


class FakeMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name: str, description: str, structured_output: bool):
        del description
        del structured_output

        def decorator(function):
            self.tools[name] = function
            return function

        return decorator


def _stderr_records(stderr: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in stderr.splitlines()
        if line.strip() != ""
    ]


def test_mcp_paper_approval_logs_paper_mcp_execution_context(
    capsys,
    tmp_path: Path,
) -> None:
    configure_logging()
    workspace = tmp_path / "workspace"
    artifact_root = workspace / "artifacts"
    config_path = write_phase7_paper_config(workspace / "configs" / "paper.yaml")
    config = load_config(config_path)
    decision = run_paper_decision(
        config,
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        provider=FakeAlpacaProvider(),
        broker=FakeAlpacaBroker(),
    )

    fake_mcp = FakeMCP()
    sandbox = WorkspaceSandbox(
        workspace_root=workspace,
        artifact_root=artifact_root,
        repo_root=workspace,
    )
    register_paper_tools(fake_mcp, sandbox=sandbox)

    approve = fake_mcp.tools["marketlab_decide_paper_proposal"]
    result = approve(
        config_path="configs/paper.yaml",
        proposal_id=decision["proposal_id"],
        decision="approve",
        actor="agent",
    )

    records = _stderr_records(capsys.readouterr().err)
    start_record = next(record for record in records if record["event"] == "paper.mcp.approval.start")
    finish_record = next(record for record in records if record["event"] == "paper.mcp.approval.finish")

    assert result["status"]["status"] == "approved"
    assert start_record["deployment"] == "paper_mcp"
    assert finish_record["deployment"] == "paper_mcp"
    assert start_record["proposal_id"] == decision["proposal_id"]
    assert finish_record["proposal_id"] == decision["proposal_id"]
    assert finish_record["outcome"] == "approved"
