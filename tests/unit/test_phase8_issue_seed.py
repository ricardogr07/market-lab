from __future__ import annotations

from pathlib import Path


def test_phase8_crypto_issue_seed_was_pruned() -> None:
    assert not Path(".github/issue-seeds/phase8-crypto-hourly-trend.json").exists()


def test_phase8_docs_keep_only_retained_methodology_page() -> None:
    nav_text = Path("mkdocs.yml").read_text(encoding="utf-8")

    assert "BTC Phase 8 Methodology: btc-phase8-methodology.md" in nav_text
    assert "Phase 8 Crypto Hourly Trend: crypto-hourly-trend.md" not in nav_text
    assert "BTC Phase 8 Bull Upside: phase8/BTC/bull-upside-methodology.md" not in nav_text
    assert "BTC Phase 8 Target/Score Pivot: phase8/BTC/target-score-pivot.md" not in nav_text
    assert (
        "BTC Phase 8 Shadow Confirmation: phase8/BTC/shadow-confirmation-plan.md"
        not in nav_text
    )
    assert Path("docs/btc-phase8-methodology.md").exists()
    assert not Path("docs/crypto-hourly-trend.md").exists()
    assert not Path("docs/phase8/BTC/bull-upside-methodology.md").exists()
    assert not Path("docs/phase8/BTC/target-score-pivot.md").exists()
    assert not Path("docs/phase8/BTC/shadow-confirmation-plan.md").exists()


def test_top_level_docs_describe_retained_phase8_and_btc_paper_surface() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    plan = Path("docs/PLAN.md").read_text(encoding="utf-8")

    for content in (readme, plan):
        assert "Phase 8" in content
        assert "phase8-summary" in content
        assert "phase8-methodology-review" in content
        assert "phase8-selection-probe" not in content
        assert "phase8-target-diagnostic" not in content
        assert "phase8-bull-counterfactual" not in content
        assert "phase8-regime-policy-sweep" not in content
        assert "phase8-grid-compare" not in content
        assert (
            "configs/experiment.btc_phase8_guarded_gate_bull_risk_off_override_partial_support.yaml"
            in content
        )
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
