from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
READINESS_DOC = ROOT / "docs" / "service-extraction-readiness.md"
MKDOCS_CONFIG = ROOT / "mkdocs.yml"


def test_mkdocs_nav_includes_service_extraction_readiness() -> None:
    config = yaml.safe_load(MKDOCS_CONFIG.read_text(encoding="utf-8"))

    assert {"Service Extraction Readiness": "service-extraction-readiness.md"} in config["nav"]


def test_service_extraction_readiness_contains_required_decision_headings() -> None:
    content = READINESS_DOC.read_text(encoding="utf-8")

    required_headings = [
        "## Keep Package-First",
        "## Extract Service",
        "## Introduce Port Or Adapter",
        "## Preserve Artifact Parity",
        "## Defer Extraction",
    ]

    for heading in required_headings:
        assert heading in content


def test_service_extraction_readiness_locks_required_semantic_guardrails() -> None:
    content = READINESS_DOC.read_text(encoding="utf-8")
    normalized = " ".join(content.lower().split())

    required_clauses = [
        "readiness-only prs must not change runtime behavior.",
        "cli, scheduler, agent, mcp, or other inbound callers",
        "proposal, approval, submission, reconciliation, and status semantics",
        "existing artifact paths and reviewable payload meanings",
        "artifact paths, payload meanings, report shapes, paper-state semantics",
        "mcp the execution backend",
    ]

    for clause in required_clauses:
        assert clause in normalized
