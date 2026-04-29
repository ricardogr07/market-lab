from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolate_paper_telegram_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty_env_file = tmp_path / ".marketlab-test.env"
    empty_env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("MARKETLAB_ENV_FILE", str(empty_env_file))
    monkeypatch.delenv("MARKETLAB_PAPER_TELEGRAM_ENABLED", raising=False)
    monkeypatch.delenv("MARKETLAB_PAPER_TELEGRAM_ALLOWED_EXPERIMENTS", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
