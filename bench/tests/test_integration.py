"""End-to-end test of the benchmark harness against the mock vLLM server.

This is the test that would have caught a broken SSE parser, a wrong token
accounting path, or a QPS formula that divides by the wrong wall time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_pro import main  # noqa: E402
from mock_vllm import serve_in_thread  # noqa: E402


def test_benchmark_against_mock(tmp_path):
    server, thread = serve_in_thread(
        port=0, prefill_ms=5.0, token_ms=0.5, model_name="mock-model"
    )
    try:
        port = server.server_address[1]
        out_dir = tmp_path / "results"
        rc = main(
            [
                "--base-url", f"http://127.0.0.1:{port}",
                "--model", "mock-model",
                "--concurrency", "1,2",
                "--num-requests", "4",
                "--max-tokens", "8",
                "--prompt-chars", "200",
                "--warmup", "1",
                "--cooldown", "0",
                "--output-dir", str(out_dir),
            ]
        )
        assert rc == 0

        results = list(out_dir.glob("*.json"))
        assert len(results) == 1
        payload = json.loads(results[0].read_text(encoding="utf-8"))

        assert payload["model"] == "mock-model"
        assert len(payload["levels"]) == 2

        for level in payload["levels"]:
            assert level["completed"] == 4
            assert level["errors"] == 0
            assert level["qps"] > 0
            assert level["ttft"]["p50"] > 0
            assert level["completion_tokens_mean"] == 8
            assert level["tokens_estimated"] is False  # usage chunk was honoured
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_non_streaming_mode(tmp_path):
    server, thread = serve_in_thread(
        port=0, prefill_ms=2.0, token_ms=0.2, model_name="mock-model"
    )
    try:
        port = server.server_address[1]
        out_dir = tmp_path / "results"
        rc = main(
            [
                "--base-url", f"http://127.0.0.1:{port}",
                "--model", "mock-model",
                "--concurrency", "2",
                "--num-requests", "3",
                "--max-tokens", "4",
                "--prompt-chars", "100",
                "--warmup", "0",
                "--cooldown", "0",
                "--no-stream",
                "--output-dir", str(out_dir),
            ]
        )
        assert rc == 0
        payload = json.loads(next(out_dir.glob("*.json")).read_text(encoding="utf-8"))
        level = payload["levels"][0]
        assert level["completed"] == 3
        assert level["completion_tokens_mean"] == 4
    finally:
        server.shutdown()
        thread.join(timeout=5)
