from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from marketlab.paper.alpaca import (
    AlpacaCredentials,
    AlpacaMarketDataProvider,
    AlpacaPaperBrokerClient,
)


class _FakeAlpacaHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/v2/stocks/bars":
            _ = parse_qs(parsed.query)
            self._json_response(
                {
                    "bars": {
                        "VOO": [
                            {
                                "t": "2026-04-09T00:00:00Z",
                                "o": 100.0,
                                "h": 101.0,
                                "l": 99.0,
                                "c": 100.5,
                                "v": 1000,
                            },
                            {
                                "t": "2026-04-10T00:00:00Z",
                                "o": 100.5,
                                "h": 101.5,
                                "l": 100.0,
                                "c": 101.0,
                                "v": 1200,
                            },
                        ]
                    }
                }
            )
            return
        if parsed.path == "/v2/calendar":
            self._json_response(
                [
                    {"date": "2026-04-10"},
                    {"date": "2026-04-13"},
                ]
            )
            return
        if parsed.path == "/v2/account":
            self._json_response(
                {
                    "equity": "10000.00",
                    "buying_power": "10000.00",
                    "cash": "10000.00",
                    "status": "ACTIVE",
                }
            )
            return
        if parsed.path == "/v2/positions/VOO":
            self._json_response(
                {
                    "symbol": "VOO",
                    "qty": "0.250000",
                    "market_value": "25.00",
                }
            )
            return
        if parsed.path == "/v2/orders/order-1":
            self._json_response(
                {
                    "id": "order-1",
                    "status": "accepted",
                    "client_order_id": "marketlab-test-order",
                }
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/v2/orders":
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            response = {
                "id": "order-1",
                "status": "accepted",
                "client_order_id": payload["client_order_id"],
            }
            if "qty" in payload:
                response["qty"] = payload["qty"]
            if "notional" in payload:
                response["notional"] = payload["notional"]
            self._json_response(response)
            return
        self.send_error(404)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _json_response(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def fake_alpaca_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeAlpacaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_alpaca_clients_parse_local_fake_server(
    fake_alpaca_server: str,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY_ID", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_DATA_BASE_URL", fake_alpaca_server)
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", fake_alpaca_server)

    provider = AlpacaMarketDataProvider()
    broker = AlpacaPaperBrokerClient()

    frame = provider.download_symbol_history("VOO", "2026-04-09", "2026-04-10", "1d")
    calendar = broker.get_calendar(start_date=frame["Date"].min().date(), end_date=frame["Date"].max().date())
    account = broker.get_account()
    position = broker.get_position("VOO")
    order = broker.submit_fractional_day_market_order(
        symbol="VOO",
        qty=0.5,
        side="buy",
        client_order_id="marketlab-test-order",
    )
    order_status = broker.get_order("order-1")
    notional_order = broker.submit_notional_day_market_order(
        symbol="VOO",
        notional=500.0,
        side="buy",
        client_order_id="marketlab-test-notional",
    )

    assert list(frame.columns) == ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert len(frame) == 2
    assert [item["date"] for item in calendar] == ["2026-04-10", "2026-04-13"]
    assert account["equity"] == "10000.00"
    assert position["qty"] == "0.250000"
    assert order["id"] == "order-1"
    assert order_status["status"] == "accepted"
    assert notional_order["notional"] == "500.00"


def test_alpaca_client_reads_configured_symbol_position(
    fake_alpaca_server: str,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY_ID", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_DATA_BASE_URL", fake_alpaca_server)
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", fake_alpaca_server)

    broker = AlpacaPaperBrokerClient()
    position = broker.get_position("VOO")

    assert position is not None
    assert position["symbol"] == "VOO"


def test_paper_broker_rejects_live_endpoint() -> None:
    credentials = AlpacaCredentials(
        api_key_id="test-key",
        api_secret_key="test-secret",
        trading_base_url="https://api.alpaca.markets",
    )

    with pytest.raises(RuntimeError, match="paper endpoint"):
        AlpacaPaperBrokerClient(credentials)


def test_alpaca_credentials_load_local_env_file(
    tmp_path,
    monkeypatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "ALPACA_API_KEY_ID=from-env-file-key",
                "ALPACA_API_SECRET_KEY=from-env-file-secret",
                "ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)
    monkeypatch.delenv("MARKETLAB_ENV_FILE", raising=False)

    credentials = AlpacaCredentials.from_env()

    assert credentials.api_key_id == "from-env-file-key"
    assert credentials.api_secret_key == "from-env-file-secret"
    assert credentials.trading_base_url == "https://paper-api.alpaca.markets"
