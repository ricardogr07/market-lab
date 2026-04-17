from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd

from marketlab.data.market import MarketDataProvider
from marketlab.env import load_env_file

DEFAULT_ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
DEFAULT_ALPACA_TRADING_BASE_URL = "https://paper-api.alpaca.markets"
PAPER_HOSTS = {"paper-api.alpaca.markets", "127.0.0.1", "localhost"}


@dataclass(slots=True, frozen=True)
class AlpacaCredentials:
    api_key_id: str
    api_secret_key: str
    data_base_url: str = DEFAULT_ALPACA_DATA_BASE_URL
    trading_base_url: str = DEFAULT_ALPACA_TRADING_BASE_URL
    data_feed: str = "iex"
    timeout_seconds: int = 30

    @classmethod
    def from_env(cls) -> "AlpacaCredentials":
        import os

        load_env_file()
        api_key_id = os.environ.get("ALPACA_API_KEY_ID", "").strip()
        api_secret_key = os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
        if not api_key_id or not api_secret_key:
            raise RuntimeError(
                "Alpaca credentials are required. Set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY in the environment or add them to a local .env file."
            )

        data_base_url = os.environ.get(
            "ALPACA_DATA_BASE_URL",
            DEFAULT_ALPACA_DATA_BASE_URL,
        ).strip() or DEFAULT_ALPACA_DATA_BASE_URL
        trading_base_url = os.environ.get(
            "ALPACA_TRADING_BASE_URL",
            DEFAULT_ALPACA_TRADING_BASE_URL,
        ).strip() or DEFAULT_ALPACA_TRADING_BASE_URL
        data_feed = os.environ.get("ALPACA_DATA_FEED", "iex").strip() or "iex"
        timeout_seconds = int(os.environ.get("ALPACA_TIMEOUT_SECONDS", "30"))
        return cls(
            api_key_id=api_key_id,
            api_secret_key=api_secret_key,
            data_base_url=data_base_url.rstrip("/"),
            trading_base_url=trading_base_url.rstrip("/"),
            data_feed=data_feed,
            timeout_seconds=timeout_seconds,
        )


def _json_request(
    *,
    method: str,
    base_url: str,
    path: str,
    api_key_id: str,
    api_secret_key: str,
    timeout_seconds: int,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> Any:
    query = ""
    if params:
        query = "?" + urlencode(params, doseq=True)
    url = f"{base_url.rstrip('/')}{path}{query}"
    data = None
    headers = {
        "APCA-API-KEY-ID": api_key_id,
        "APCA-API-SECRET-KEY": api_secret_key,
        "Accept": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url=url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Alpaca API request failed: {exc.code} {detail}") from exc

    if body == "":
        return {}
    return json.loads(body)


def _normalize_daily_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True)
    if isinstance(timestamp, pd.Timestamp):
        return timestamp.tz_convert(None).normalize()
    raise TypeError(f"Unsupported timestamp value: {value!r}")


class AlpacaMarketDataProvider(MarketDataProvider):
    def __init__(self, credentials: AlpacaCredentials | None = None) -> None:
        self._credentials = credentials or AlpacaCredentials.from_env()

    def download_symbol_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> pd.DataFrame:
        if interval != "1d":
            raise ValueError("The Alpaca paper-data path currently supports interval='1d' only.")

        payload = _json_request(
            method="GET",
            base_url=self._credentials.data_base_url,
            path="/v2/stocks/bars",
            api_key_id=self._credentials.api_key_id,
            api_secret_key=self._credentials.api_secret_key,
            timeout_seconds=self._credentials.timeout_seconds,
            params={
                "symbols": symbol,
                "timeframe": "1Day",
                "start": f"{start_date}T00:00:00Z",
                "end": f"{end_date}T23:59:59Z",
                "adjustment": "all",
                "feed": self._credentials.data_feed,
                "sort": "asc",
                "limit": 10000,
            },
        )

        bars = (payload.get("bars") or {}).get(symbol, [])
        if not bars:
            raise RuntimeError(f"No Alpaca market data returned for {symbol}.")

        frame = pd.DataFrame(
            [
                {
                    "Date": _normalize_daily_timestamp(bar["t"]),
                    "Open": float(bar["o"]),
                    "High": float(bar["h"]),
                    "Low": float(bar["l"]),
                    "Close": float(bar["c"]),
                    "Adj Close": float(bar["c"]),
                    "Volume": float(bar["v"]),
                }
                for bar in bars
            ]
        )
        return frame.sort_values("Date").reset_index(drop=True)


class AlpacaPaperBrokerClient:
    def __init__(self, credentials: AlpacaCredentials | None = None) -> None:
        self._credentials = credentials or AlpacaCredentials.from_env()
        self._ensure_paper_endpoint()

    def _ensure_paper_endpoint(self) -> None:
        parsed = urlparse(self._credentials.trading_base_url)
        if parsed.hostname not in PAPER_HOSTS and "paper" not in self._credentials.trading_base_url:
            raise RuntimeError(
                "The paper-trading path requires an Alpaca paper endpoint. "
                f"Configured trading base URL: {self._credentials.trading_base_url}"
            )

    def get_calendar(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        payload = _json_request(
            method="GET",
            base_url=self._credentials.trading_base_url,
            path="/v2/calendar",
            api_key_id=self._credentials.api_key_id,
            api_secret_key=self._credentials.api_secret_key,
            timeout_seconds=self._credentials.timeout_seconds,
            params={
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
        )
        if not isinstance(payload, list):
            raise RuntimeError("Alpaca calendar response must be a list.")
        return payload

    def get_account(self) -> dict[str, Any]:
        payload = _json_request(
            method="GET",
            base_url=self._credentials.trading_base_url,
            path="/v2/account",
            api_key_id=self._credentials.api_key_id,
            api_secret_key=self._credentials.api_secret_key,
            timeout_seconds=self._credentials.timeout_seconds,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Alpaca account response must be an object.")
        return payload

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        try:
            payload = _json_request(
                method="GET",
                base_url=self._credentials.trading_base_url,
                path=f"/v2/positions/{symbol}",
                api_key_id=self._credentials.api_key_id,
                api_secret_key=self._credentials.api_secret_key,
                timeout_seconds=self._credentials.timeout_seconds,
            )
        except RuntimeError as exc:
            if "404" in str(exc):
                return None
            raise

        if not isinstance(payload, dict):
            raise RuntimeError("Alpaca position response must be an object.")
        return payload

    def submit_fractional_day_market_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        client_order_id: str,
    ) -> dict[str, Any]:
        payload = _json_request(
            method="POST",
            base_url=self._credentials.trading_base_url,
            path="/v2/orders",
            api_key_id=self._credentials.api_key_id,
            api_secret_key=self._credentials.api_secret_key,
            timeout_seconds=self._credentials.timeout_seconds,
            payload={
                "symbol": symbol,
                "qty": f"{qty:.6f}",
                "side": side,
                "type": "market",
                "time_in_force": "day",
                "client_order_id": client_order_id,
            },
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Alpaca order submission response must be an object.")
        return payload

    def submit_notional_day_market_order(
        self,
        *,
        symbol: str,
        notional: float,
        side: str,
        client_order_id: str,
    ) -> dict[str, Any]:
        payload = _json_request(
            method="POST",
            base_url=self._credentials.trading_base_url,
            path="/v2/orders",
            api_key_id=self._credentials.api_key_id,
            api_secret_key=self._credentials.api_secret_key,
            timeout_seconds=self._credentials.timeout_seconds,
            payload={
                "symbol": symbol,
                "notional": f"{notional:.2f}",
                "side": side,
                "type": "market",
                "time_in_force": "day",
                "client_order_id": client_order_id,
            },
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Alpaca order submission response must be an object.")
        return payload

    def get_order(self, order_id: str) -> dict[str, Any]:
        payload = _json_request(
            method="GET",
            base_url=self._credentials.trading_base_url,
            path=f"/v2/orders/{order_id}",
            api_key_id=self._credentials.api_key_id,
            api_secret_key=self._credentials.api_secret_key,
            timeout_seconds=self._credentials.timeout_seconds,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Alpaca order response must be an object.")
        return payload
