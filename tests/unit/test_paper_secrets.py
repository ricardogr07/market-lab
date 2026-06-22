from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tests._paper_fakes import build_phase7_paper_config

from marketlab.paper.alpaca import AlpacaCredentials
from marketlab.paper.notifications import deliver_telegram_notification
from marketlab.paper.secrets import EnvironmentPaperSecretProvider, resolve_paper_secret


class _MappingSecretProvider:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def get(self, name: str) -> str | None:
        return self._mapping.get(name)


def test_environment_secret_provider_reads_process_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MARKETLAB_TEST_SECRET", "  secret-value  ")
    provider = EnvironmentPaperSecretProvider(load_env=False)

    assert provider.get("MARKETLAB_TEST_SECRET") == "secret-value"
    assert resolve_paper_secret("MARKETLAB_TEST_SECRET", secret_provider=provider) == "secret-value"


def test_alpaca_credentials_can_be_loaded_from_secret_provider() -> None:
    provider = _MappingSecretProvider(
        {
            "ALPACA_API_KEY_ID": "provider-key",
            "ALPACA_API_SECRET_KEY": "provider-secret",
            "ALPACA_DATA_BASE_URL": "https://data.example.test",
            "ALPACA_TRADING_BASE_URL": "https://paper.example.test",
            "ALPACA_DATA_FEED": "sip",
            "ALPACA_TIMEOUT_SECONDS": "45",
        }
    )

    credentials = AlpacaCredentials.from_env(secret_provider=provider)

    assert credentials.api_key_id == "provider-key"
    assert credentials.api_secret_key == "provider-secret"
    assert credentials.data_base_url == "https://data.example.test"
    assert credentials.trading_base_url == "https://paper.example.test"
    assert credentials.data_feed == "sip"
    assert credentials.timeout_seconds == 45


def test_telegram_delivery_can_use_secret_provider(
    tmp_path: Path,
) -> None:
    config = build_phase7_paper_config(tmp_path, telegram_enabled=True)
    provider = _MappingSecretProvider(
        {
            "TELEGRAM_BOT_TOKEN": "provider-bot",
            "TELEGRAM_CHAT_ID": "provider-chat",
            "MARKETLAB_TELEGRAM_API_BASE_URL": "https://telegram.example.test",
        }
    )
    calls: list[tuple[str, dict[str, object], int]] = []

    record = deliver_telegram_notification(
        config,
        stage="paper-decision",
        outcome="proposal_created",
        message="paper-decision\noutcome: proposal_created",
        now=datetime(2026, 4, 10, 20, 10, tzinfo=UTC),
        transport=lambda url, payload, timeout: calls.append((url, payload, timeout)) or (200, '{"ok": true}'),
        secret_provider=provider,
    )

    assert record["delivery_status"] == "delivered"
    assert calls == [
        (
            "https://telegram.example.test/botprovider-bot/sendMessage",
            {
                "chat_id": "provider-chat",
                "text": "paper-decision\noutcome: proposal_created",
                "disable_web_page_preview": True,
            },
            10,
        )
    ]
