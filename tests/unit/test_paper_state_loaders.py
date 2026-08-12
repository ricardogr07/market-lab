from __future__ import annotations

from pathlib import Path

import pytest
from tests._paper_fakes import build_phase7_paper_config

import marketlab.paper.agent as agent_module
from marketlab.paper import scheduler


@pytest.mark.parametrize(
    ("loader", "state_name"),
    [
        (scheduler._load_scheduler_state, "scheduler.json"),
        (agent_module._load_worker_state, "agent_worker.json"),
    ],
)
def test_corrupt_paper_state_is_quarantined(
    loader,
    state_name: str,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path)
    state_path = config.paper_state_dir / state_name
    corrupt_bytes = b"\x00" * 313
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_bytes(corrupt_bytes)

    with caplog.at_level("WARNING"):
        assert loader(config) == {}

    quarantine_paths = list(state_path.parent.glob(f"{state_name}.corrupt-*"))
    assert not state_path.exists()
    assert len(quarantine_paths) == 1
    assert quarantine_paths[0].read_bytes() == corrupt_bytes
    assert str(state_path) in caplog.text
    assert str(quarantine_paths[0]) in caplog.text


@pytest.mark.parametrize(
    ("loader", "state_name"),
    [
        (scheduler._load_scheduler_state, "scheduler.json"),
        (agent_module._load_worker_state, "agent_worker.json"),
    ],
)
@pytest.mark.parametrize("non_dict_json", ["null", "[]", '"a string"', "42"])
def test_non_object_paper_state_is_quarantined(
    loader,
    state_name: str,
    non_dict_json: str,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path)
    state_path = config.paper_state_dir / state_name
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(non_dict_json, encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert loader(config) == {}

    quarantine_paths = list(state_path.parent.glob(f"{state_name}.corrupt-*"))
    assert not state_path.exists()
    assert len(quarantine_paths) == 1
    assert quarantine_paths[0].read_text(encoding="utf-8") == non_dict_json
