from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_local_launcher_prefers_repo_src_over_pythonpath(tmp_path: Path) -> None:
    fake_package = tmp_path / "marketlab"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text("", encoding="utf-8")
    (fake_package / "cli.py").write_text(
        "\n".join(
            [
                "def main(argv=None):",
                "    print('fake-marketlab-cli')",
                "    return 99",
            ]
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[2]
    launcher_path = repo_root / "scripts" / "run_marketlab.py"

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, str(launcher_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0
    assert "prepare-data" in result.stdout
    assert "fake-marketlab-cli" not in result.stdout

