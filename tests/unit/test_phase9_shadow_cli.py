from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from marketlab.shadow import cli


def test_shadow_cli_delegates_to_service(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        """{
  "status": "success",
  "selection_source": "strict",
  "fallback_mode": "none",
  "target_allocation": 0.5,
  "input_payload": {"score": 0.61}
}
""",
        encoding="utf-8",
    )
    panel_path = tmp_path / "panel.csv"
    panel_path.write_text("unused", encoding="utf-8")
    contract = SimpleNamespace(
        config_path=Path("config.yaml"),
        config=SimpleNamespace(prepared_panel_path=panel_path),
    )
    panel = pd.DataFrame()
    captured = {}

    monkeypatch.setattr(cli, "verify_shadow_contract", lambda path: contract)
    monkeypatch.setattr(cli, "load_panel_csv", lambda path: panel)
    monkeypatch.setattr(cli, "shadow_bars_from_panel", lambda frame: ("bar",))

    def _run(request, *, evaluator):
        captured["request"] = request
        captured["evaluation"] = evaluator(None)
        return SimpleNamespace(path=tmp_path / "decisions" / "2026-06-11.json")

    monkeypatch.setattr(cli, "run_shadow_decision", _run)

    exit_code = cli.main(
        [
            "--config",
            "config.yaml",
            "--evaluation",
            str(evaluation_path),
            "--panel",
            str(panel_path),
            "--as-of",
            "2026-06-11T01:15:00Z",
        ]
    )

    request = captured["request"]
    assert exit_code == 0
    assert request.contract is contract
    assert request.bars == ("bar",)
    assert request.as_of == datetime(2026, 6, 11, 1, 15, tzinfo=UTC)
    assert captured["evaluation"].target_allocation == 0.5
    assert capsys.readouterr().out.strip().endswith("2026-06-11.json")
