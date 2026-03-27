from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import yaml

from marketlab.cli import main

pytestmark = [pytest.mark.real_data, pytest.mark.network]


def _load_smoke_config() -> dict:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "experiment.weekly_rank.smoke.yaml"
    )
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _latest_run_dir(run_root: Path) -> Path:
    run_dirs = list(run_root.iterdir())
    assert run_dirs, f"No run directories were created under {run_root}"
    return max(run_dirs, key=lambda path: path.stat().st_mtime_ns)


def _assert_run_artifacts(run_dir: Path) -> None:
    for artifact_name in (
        "metrics.csv",
        "performance.csv",
        "report.md",
        "cumulative_returns.png",
        "drawdown.png",
    ):
        assert (run_dir / artifact_name).exists(), (
            f"Expected artifact {artifact_name} in {run_dir}"
        )


@pytest.mark.skipif(
    os.getenv("MARKETLAB_RUN_REAL_DATA") != "1",
    reason="Set MARKETLAB_RUN_REAL_DATA=1 to run real-data smoke tests.",
)
def test_real_data_cli_smoke_creates_cache_and_artifacts(tmp_path: Path) -> None:
    config_payload = _load_smoke_config()
    config_payload["experiment_name"] = "weekly_rank_real_data_smoke"
    config_payload["data"]["cache_dir"] = str(tmp_path / "e2e-data")
    config_payload["artifacts"]["output_dir"] = str(tmp_path / "e2e-runs")

    config_path = tmp_path / "real-data-smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    panel_path = Path(config_payload["data"]["cache_dir"]) / config_payload["data"]["prepared_panel_filename"]
    run_root = Path(config_payload["artifacts"]["output_dir"]) / config_payload["experiment_name"]

    assert main(["prepare-data", "--config", str(config_path)]) == 0
    assert panel_path.exists()

    first_panel_mtime = panel_path.stat().st_mtime_ns

    assert main(["prepare-data", "--config", str(config_path)]) == 0
    assert panel_path.stat().st_mtime_ns == first_panel_mtime

    assert main(["backtest", "--config", str(config_path)]) == 0
    backtest_run_dir = _latest_run_dir(run_root)
    _assert_run_artifacts(backtest_run_dir)

    time.sleep(1)

    assert main(["run-experiment", "--config", str(config_path)]) == 0
    experiment_run_dir = _latest_run_dir(run_root)
    _assert_run_artifacts(experiment_run_dir)

