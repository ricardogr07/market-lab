from __future__ import annotations

from pathlib import Path

import yaml

from marketlab.cli import main


def test_run_experiment_produces_metrics_report_and_plots(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    fixture_panel = Path(__file__).resolve().parents[1] / "fixtures" / "market_panel.csv"
    (cache_dir / "panel.csv").write_text(fixture_panel.read_text(encoding="utf-8"), encoding="utf-8")

    config = {
        "experiment_name": "integration_fixture",
        "data": {
            "symbols": ["VOO", "QQQ", "SMH", "XLV", "IEMG"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "interval": "1d",
            "cache_dir": str(cache_dir),
            "prepared_panel_filename": "panel.csv",
        },
        "features": {
            "return_windows": [2, 3],
            "ma_windows": [2, 3],
            "vol_windows": [2],
            "momentum_window": 2,
        },
        "portfolio": {
            "costs": {"bps_per_trade": 10},
        },
        "baselines": {
            "buy_hold": True,
            "sma": {"enabled": True, "fast_window": 2, "slow_window": 3},
        },
        "artifacts": {
            "output_dir": str(tmp_path / "runs"),
            "save_predictions": False,
            "save_metrics_csv": True,
            "save_report_md": True,
            "save_plots": True,
        },
    }

    config_path = tmp_path / "integration.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = main(["run-experiment", "--config", str(config_path)])

    assert result == 0

    run_root = tmp_path / "runs" / "integration_fixture"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "performance.csv").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "cumulative_returns.png").exists()
    assert (run_dir / "drawdown.png").exists()
