from __future__ import annotations

import json
from pathlib import Path


def test_phase8_issue_seed_covers_crypto_hourly_signal_plan() -> None:
    document = json.loads(
        Path(".github/issue-seeds/phase8-crypto-hourly-trend.json").read_text(
            encoding="utf-8"
        )
    )

    label_names = {label["name"] for label in document["labels"]}
    issue_titles = [issue["title"] for issue in document["issues"]]

    assert document["project"]["number"] == 3
    assert {"phase-8", "track:strategy", "track:research"} <= label_names
    assert issue_titles == [
        "Phase 8.1: crypto hourly trend-signal validation against buy-and-hold",
        "Add hourly crypto panel and annualized metric foundations",
        "Add bar-based signal timing for intraday research",
        "Add indicator-stack baseline for influencer-style claims",
        "Document crypto hourly shadow-paper boundaries",
        "Phase 8.6: crypto time-series ML strategy comparison",
        "Phase 8.7: indicator-stack ML time-series tuning",
    ]


def test_phase8_docs_are_linked_from_mkdocs_nav() -> None:
    nav_text = Path("mkdocs.yml").read_text(encoding="utf-8")

    assert "Phase 8 Crypto Hourly Trend: crypto-hourly-trend.md" in nav_text
    assert Path("docs/crypto-hourly-trend.md").exists()
