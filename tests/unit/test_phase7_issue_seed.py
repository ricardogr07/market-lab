from __future__ import annotations

import json
from pathlib import Path


def test_phase7_issue_seed_covers_expected_epic_and_child_issues() -> None:
    document = json.loads(
        Path(".github/issue-seeds/phase7-voo-paper.json").read_text(encoding="utf-8")
    )

    label_names = {label["name"] for label in document["labels"]}
    issue_titles = [issue["title"] for issue in document["issues"]]

    assert document["project"]["number"] == 3
    assert {"phase-7", "track:broker", "track:ops"} <= label_names
    assert issue_titles == [
        "Phase 7.1: autonomous single-ETF paper loop with OpenAI and Claude backends",
        "Add six-model paper decision and consensus proposal artifacts",
        "Add autonomous paper agent worker and deterministic approval backend",
        "Add OpenAI and Claude approval backends with structured fallback",
        "Add tracked QQQ and VOO paper configs plus month-run paper reporting",
    ]
