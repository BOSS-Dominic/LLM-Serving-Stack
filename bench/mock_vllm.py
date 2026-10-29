#!/usr/bin/env python3
"""Minimal OpenAI-compatible mock of a vLLM server.

It exists so that the benchmark harness and the gateway can be exercised
end-to-end on a laptop or in CI, where no GPU is available. It is NOT a
performance model of vLLM: it simulates a fixed prefill delay plus a fixed
per-token delay, which is enough to verify that TTFT, inter-token timing and
token accounting are measured correctly.

    python bench/mock_vllm.py --port 18000 --prefill-ms 40 --token-ms 4

Endpoints:
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions   (streaming and non-streaming)
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    text = "".join(
        str(message.get("content", "")) for message in payload.get("messages", [])
    )
    # Crude but stable: roughly one token per two CJK characters.
    return max(1, len(text) // 2)


class MockVLLMHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mock-vllm/0.1"

    # Injected by make_server.
    prefill_ms: float = 20.0
    token_ms: float = 3.0
    max_tokens_cap: int = 256

    def log_message(self, fmt: str, *args: Any) -> None:  # silence per-request logs
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    # -- helpers ----------------------------------------------------------- #

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}

    # -- routes ------------------------------------------------------------ #

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/health":
            self._json(200, {"status": "ok"})
            return
        if self.path.startswith("/v1/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": getattr(self.server, "model_name", "mock-model"),
                            "object": "model",
                            "owned_by": "mock",
                        }
                    ],
                },
            )
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path.split("?")[0] != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return

        payload = self._read_json()
        model = payload.get("model") or getattr(self.server, "model_name", "mock-model")
        max_tokens = min(int(payload.get("max_tokens") or 32), self.max_tokens_cap)
        prompt_tokens = _estimate_prompt_tokens(payload)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if payload.get("stream"):
            self._stream_chat(request_id, model, max_tokens, prompt_tokens, payload)
        else:
            self._full_chat(request_id, model, max_tokens, prompt_tokens)

    # -- response shapes ---------------------------------------------------- #

    def _full_chat(self, request_id: str, model: str, max_tokens: int, prompt_tokens: int) -> None:
        time.sleep(self.prefill_ms / 1000.0)
        time.sleep(self.token_ms * max_tokens / 1000.0)
        body = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "mock " * max_tokens},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max_tokens,
                "total_tokens": prompt_tokens + max_tokens,
            },
        }
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _stream_chat(
        self,
        request_id: str,
        model: str,
        max_tokens: int,
        prompt_tokens: int,
        payload: dict[str, Any],
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(obj: dict[str, Any]) -> None:
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        # Time to first token = prefill; every following token costs token_ms.
        # The sleep happens BEFORE the first chunk so TTFT reflects prefill.
        time.sleep(self.prefill_ms / 1000.0)

        for i in range(max_tokens):
            try:
                emit(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"token{i} "},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
                return
            time.sleep(self.token_ms / 1000.0)

        try:
            emit(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                }
            )
            if (payload.get("stream_options") or {}).get("include_usage"):
                emit(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": max_tokens,
                            "total_tokens": prompt_tokens + max_tokens,
                        },
                    }
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True


def make_server(
    host: str,
    port: int,
    prefill_ms: float,
    token_ms: float,
    model_name: str,
    verbose: bool = False,
    max_tokens_cap: int = 256,
) -> ThreadingHTTPServer:
    handler = type(
        "BoundMockHandler",
        (MockVLLMHandler,),
        {"prefill_ms": prefill_ms, "token_ms": token_ms, "max_tokens_cap": max_tokens_cap},
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.model_name = model_name  # type: ignore[attr-defined]
    server.verbose = verbose  # type: ignore[attr-defined]

    # Clients that disconnect mid-stream are normal in these tests; do not dump
    # a traceback for every one of them.
    def handle_error(request, client_address):  # noqa: ANN001
        import sys
        from socketserver import BaseServer

        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        BaseServer.handle_error(server, request, client_address)  # type: ignore[arg-type]

    server.handle_error = handle_error  # type: ignore[method-assign]
    return server


def serve_in_thread(port: int = 0, **kwargs: Any) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = make_server("127.0.0.1", port, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible mock of a vLLM server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--prefill-ms", type=float, default=20.0, help="simulated time to first token")
    parser.add_argument("--token-ms", type=float, default=3.0, help="simulated per-token decode time")
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    server = make_server(
        args.host, args.port, args.prefill_ms, args.token_ms, args.model, args.verbose
    )
    print(f"mock vLLM listening on http://{args.host}:{server.server_address[1]} (model={args.model})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
