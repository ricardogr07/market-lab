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
    assert "BTC Phase 8 Methodology: btc-phase8-methodology.md" in nav_text
    assert "BTC Phase 8 Bull Upside: phase8/BTC/bull-upside-methodology.md" in nav_text
    assert "BTC Phase 8 Target/Score Pivot: phase8/BTC/target-score-pivot.md" in nav_text
    assert (
        "BTC Phase 8 Shadow Confirmation: phase8/BTC/shadow-confirmation-plan.md"
        in nav_text
    )
    assert Path("docs/crypto-hourly-trend.md").exists()
    assert Path("docs/btc-phase8-methodology.md").exists()
    assert Path("docs/phase8/BTC/bull-upside-methodology.md").exists()
    assert Path("docs/phase8/BTC/target-score-pivot.md").exists()
    assert Path("docs/phase8/BTC/shadow-confirmation-plan.md").exists()


def test_top_level_docs_describe_current_phase8_and_btc_paper_surface() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    plan = Path("docs/PLAN.md").read_text(encoding="utf-8")

    for content in (readme, plan):
        assert "Phase 8" in content
        assert "phase8-summary" in content
        assert "phase8-selection-probe" in content
        assert "phase8-methodology-review" in content
        assert "phase8-target-diagnostic" in content
        assert "phase8-bull-counterfactual" in content
        assert "phase8-regime-policy-sweep" in content
        assert "phase8-grid-compare" in content
        assert "configs/experiment.btc_paper_daily.yaml" in content
        assert "docker/compose.btc-paper.yml" in content
        assert "marketlab-btc-paper-mcp" in content


def test_mcp_docs_describe_btc_paper_sidecar_without_changing_samples() -> None:
    mcp_server = Path("docs/mcp-server.md").read_text(encoding="utf-8")
    codex = Path("docs/codex-mcp.md").read_text(encoding="utf-8")
    vscode = Path("docs/mcp-vscode-copilot.md").read_text(encoding="utf-8")

    for content in (mcp_server, codex, vscode):
        assert "docker/compose.btc-paper.yml" in content
        assert "marketlab-btc-paper-mcp" in content
        assert "/app/repo/artifacts" in content
