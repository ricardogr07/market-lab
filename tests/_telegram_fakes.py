from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any


@dataclass(slots=True)
class TelegramRequest:
    path: str
    headers: dict[str, str]
    payload: dict[str, Any]


class FakeTelegramServer:
    def __init__(
        self,
        *,
        status_code: int = 200,
        response_payload: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_payload = response_payload or {"ok": True, "result": {"message_id": 1}}
        self.requests: list[TelegramRequest] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._build_handler())
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "FakeTelegramServer":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        owner = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(body)
                owner.requests.append(
                    TelegramRequest(
                        path=self.path,
                        headers={key: value for key, value in self.headers.items()},
                        payload=payload,
                    )
                )
                encoded = json.dumps(owner.response_payload).encode("utf-8")
                self.send_response(owner.status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

        return _Handler
