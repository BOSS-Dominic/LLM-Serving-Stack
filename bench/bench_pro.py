#!/usr/bin/env python3
"""Closed-loop concurrency benchmark for OpenAI-compatible LLM servers.

Why this script exists
----------------------
The "obvious" benchmark sends N requests with asyncio.gather and divides N by
wall-clock time. That measures the drain rate of one batch, not the sustained
throughput of a server under load, and it understates latency because nothing
queues behind the batch.

This script instead runs a closed loop: exactly `concurrency` requests are in
flight at any moment, and a new request starts the instant one finishes. That is
what a real client pool does, and it is the only way the numbers stay comparable
across concurrency levels.

Metrics produced per level
--------------------------
* QPS             completed requests / measurement wall time
* output tok/s    completion tokens / wall time
* total tok/s     (prompt + completion) tokens / wall time
* TTFT            submission -> first streamed chunk (queueing + prefill)
* TPOT            (end - first chunk) / (completion_tokens - 1), i.e. decode speed
* E2E             submission -> last chunk
* error rate      non-2xx responses and transport errors (never retried here;
                  retries would silently distort latency percentiles)

Usage
-----
    python bench/bench_pro.py --base-url http://127.0.0.1:8000 \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --concurrency 1,4,16 --num-requests 100 --max-tokens 128

Results are written to bench/results/<label>-<timestamp>.json plus a Markdown
table, so docs can be regenerated instead of hand-copied.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import httpx
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - exercised only on misconfigured hosts
    sys.exit("missing dependency: pip install -r bench/requirements.txt")


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (the same definition NumPy uses).

    The naive `sorted(xs)[int(len(xs) * p) - 1]` is wrong twice over: it floors
    instead of interpolating, and for small samples it can index a value far
    below the requested quantile (P95 of 4 samples becomes P75).
    """
    if not values:
        return float("nan")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"percentile must be in [0, 1], got {p}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class RequestResult:
    ok: bool
    status: int
    ttft: float | None = None
    e2e: float = 0.0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    tokens_estimated: bool = False
    error: str | None = None


@dataclass
class LevelResult:
    concurrency: int
    num_requests: int
    wall_seconds: float
    completed: int
    errors: int
    error_rate: float
    qps: float
    output_tokens_per_second: float
    total_tokens_per_second: float
    ttft: dict[str, float]
    tpot: dict[str, float]
    e2e: dict[str, float]
    prompt_tokens_mean: float
    completion_tokens_mean: float
    tokens_estimated: bool
    sample_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    label: str
    base_url: str
    model: str
    started_at: str
    finished_at: str
    environment: dict[str, Any]
    config: dict[str, Any]
    levels: list[LevelResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "base_url": self.base_url,
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "environment": self.environment,
            "config": self.config,
            "levels": [lvl.as_dict() for lvl in self.levels],
        }


# --------------------------------------------------------------------------- #
# Benchmark core
# --------------------------------------------------------------------------- #


def build_prompt(target_chars: int, seed: int) -> str:
    """Build a deterministic prompt of roughly `target_chars` characters.

    Exact token counts differ per tokenizer, so the report records the prompt
    token count the server reports in `usage` rather than assuming one.
    """
    rng = random.Random(seed)
    words = [
        "分布式", "系统", "负载均衡", "推理", "服务", "延迟", "吞吐",
        "缓存", "调度", "并发", "GPU", "显存", "批处理", "排队",
    ]
    parts: list[str] = ["请阅读下面的背景材料，然后用一句话总结其核心观点："]
    total = len(parts[0])
    while total < target_chars:
        chunk = f"{rng.choice(words)}的{rng.choice(words)}会影响{rng.choice(words)}的{rng.choice(words)}。"
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts)


class Benchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        # trust_env=False is deliberate. On machines with a system-wide proxy
        # (corporate MITM proxy, Privoxy, a VPN client), the OpenAI/httpx client
        # would otherwise route even 127.0.0.1 through it. That either fails or,
        # worse, adds a proxy hop to every request and silently inflates TTFT.
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(args.timeout),
            trust_env=args.use_env_proxy,
        )
        self.client = AsyncOpenAI(
            base_url=args.base_url.rstrip("/"),
            api_key=args.api_key,
            max_retries=0,  # measure failures, do not silently retry them
            http_client=self._http,
        )
        self.prompt = build_prompt(args.prompt_chars, args.seed)
        self.model = args.model

    async def aclose(self) -> None:
        await self._http.aclose()

    async def one_request(self, sem: asyncio.Semaphore) -> RequestResult:
        async with sem:
            started = time.perf_counter()
            ttft: float | None = None
            chunks = 0
            usage_tokens = 0
            prompt_tokens = 0

            try:
                if self.args.no_stream:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": self.prompt}],
                        max_tokens=self.args.max_tokens,
                        temperature=self.args.temperature,
                        stream=False,
                    )
                    ended = time.perf_counter()
                    usage = getattr(response, "usage", None)
                    completion = usage.completion_tokens if usage else 0
                    prompt_tokens = usage.prompt_tokens if usage else 0
                    return RequestResult(
                        ok=True,
                        status=200,
                        ttft=ended - started,
                        e2e=ended - started,
                        completion_tokens=completion,
                        prompt_tokens=prompt_tokens,
                        tokens_estimated=usage is None,
                    )

                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": self.prompt}],
                    max_tokens=self.args.max_tokens,
                    temperature=self.args.temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for chunk in stream:
                    choices = getattr(chunk, "choices", None) or []
                    if choices:
                        delta = choices[0].delta
                        content = getattr(delta, "content", None)
                        if content:
                            chunks += 1
                            if ttft is None:
                                ttft = time.perf_counter() - started
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        usage_tokens = getattr(usage, "completion_tokens", 0) or 0
                        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0

                ended = time.perf_counter()
                estimated = usage_tokens == 0
                return RequestResult(
                    ok=True,
                    status=200,
                    ttft=ttft if ttft is not None else ended - started,
                    e2e=ended - started,
                    completion_tokens=usage_tokens if usage_tokens else chunks,
                    prompt_tokens=prompt_tokens,
                    tokens_estimated=estimated,
                )
            except Exception as exc:  # noqa: BLE001 - diagnostics are the point
                ended = time.perf_counter()
                status = getattr(exc, "status_code", 0) or 0
                return RequestResult(
                    ok=False,
                    status=int(status),
                    e2e=ended - started,
                    error=f"{type(exc).__name__}: {exc}",
                )

    async def warmup(self) -> None:
        count = self.args.warmup
        if count <= 0:
            return
        sem = asyncio.Semaphore(max(1, min(count, 8)))
        await asyncio.gather(*(self.one_request(sem) for _ in range(count)))
        await asyncio.sleep(self.args.cooldown)

    async def run_level(self, concurrency: int) -> LevelResult:
        sem = asyncio.Semaphore(concurrency)
        total = self.args.num_requests
        results: list[RequestResult] = []

        # Fixed concurrency, refilled as each request completes: this is a
        # closed loop, unlike "fire N at once and divide".
        async def worker() -> None:
            results.append(await self.one_request(sem))

        pending: set[asyncio.Task[None]] = set()
        started = time.perf_counter()
        launched = 0
        try:
            while launched < total or pending:
                while launched < total and len(pending) < concurrency:
                    task = asyncio.create_task(worker())
                    pending.add(task)
                    launched += 1
                if pending:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                else:
                    break
        finally:
            for task in pending:
                task.cancel()
        wall = time.perf_counter() - started

        return summarize_level(concurrency, total, wall, results)


def summarize_level(
    concurrency: int, total: int, wall: float, results: Sequence[RequestResult]
) -> LevelResult:
    ok_results = [r for r in results if r.ok]
    errors = len(results) - len(ok_results)
    completion_tokens = sum(r.completion_tokens for r in ok_results)
    prompt_tokens = sum(r.prompt_tokens for r in ok_results)

    ttft = [r.ttft for r in ok_results if r.ttft is not None]
    # TPOT excludes the first token: (e2e - ttft) / (n_tokens - 1).
    tpot = [
        max(r.e2e - (r.ttft or 0.0), 0.0) / max(r.completion_tokens - 1, 1)
        for r in ok_results
        if r.completion_tokens > 1
    ]
    e2e = [r.e2e for r in ok_results]

    def q(values: Iterable[float]) -> dict[str, float]:
        vals = list(values)
        return {
            "p50": round(percentile(vals, 0.50), 4),
            "p95": round(percentile(vals, 0.95), 4),
            "p99": round(percentile(vals, 0.99), 4),
            "mean": round(statistics.fmean(vals), 4) if vals else float("nan"),
        }

    return LevelResult(
        concurrency=concurrency,
        num_requests=total,
        wall_seconds=round(wall, 4),
        completed=len(ok_results),
        errors=errors,
        error_rate=round(errors / max(len(results), 1), 4),
        qps=round(len(ok_results) / wall, 4) if wall > 0 else 0.0,
        output_tokens_per_second=round(completion_tokens / wall, 4) if wall > 0 else 0.0,
        total_tokens_per_second=round((completion_tokens + prompt_tokens) / wall, 4) if wall > 0 else 0.0,
        ttft=q(ttft),
        tpot=q(tpot),
        e2e=q(e2e),
        prompt_tokens_mean=round(prompt_tokens / len(ok_results), 2) if ok_results else 0.0,
        completion_tokens_mean=round(completion_tokens / len(ok_results), 2) if ok_results else 0.0,
        tokens_estimated=any(r.tokens_estimated for r in ok_results),
        sample_error=next((r.error for r in results if not r.ok), None),
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def markdown_table(run: RunResult) -> str:
    lines = [
        "| 并发 | 完成请求 | QPS | 输出 tok/s | 总 tok/s | TTFT P50 (s) | TTFT P95 (s) | E2E P50 (s) | E2E P99 (s) | TPOT P50 (ms) | 错误率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lvl in run.levels:
        lines.append(
            f"| {lvl.concurrency} | {lvl.completed}/{lvl.num_requests} | {lvl.qps:.2f} | "
            f"{lvl.output_tokens_per_second:.1f} | {lvl.total_tokens_per_second:.1f} | "
            f"{lvl.ttft['p50']:.3f} | {lvl.ttft['p95']:.3f} | "
            f"{lvl.e2e['p50']:.3f} | {lvl.e2e['p99']:.3f} | "
            f"{lvl.tpot['p50'] * 1000:.1f} | {lvl.error_rate * 100:.1f}% |"
        )
    return "\n".join(lines)


def environment_info(args: argparse.Namespace) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "client_host": platform.node(),
        "same_host_as_server": args.same_host,
    }
    try:
        import openai

        info["openai_client"] = openai.__version__
    except Exception:  # noqa: BLE001
        info["openai_client"] = "unknown"
    return info


async def fetch_server_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Read /v1/models so the report records what the server actually served."""
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=args.use_env_proxy) as client:
            resp = await client.get(args.base_url.rstrip("/") + "/models")
            resp.raise_for_status()
            return {"models_endpoint": resp.json()}
    except Exception as exc:  # noqa: BLE001
        return {"models_endpoint_error": f"{type(exc).__name__}: {exc}"}


def parse_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "concurrency": args.concurrency,
        "num_requests": args.num_requests,
        "max_tokens": args.max_tokens,
        "prompt_chars": args.prompt_chars,
        "temperature": args.temperature,
        "stream": not args.no_stream,
        "warmup": args.warmup,
        "cooldown_s": args.cooldown,
        "timeout_s": args.timeout,
        "same_host": args.same_host,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed-loop concurrency benchmark for OpenAI-compatible servers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="server URL (the /v1 root; appended if omitted)")
    parser.add_argument("--model", required=True, help="model name as served")
    parser.add_argument("--concurrency", default="1,4,16", help="comma separated levels")
    parser.add_argument("--num-requests", type=int, default=100, help="requests per level (>=50 recommended)")
    parser.add_argument("--max-tokens", type=int, default=128, help="fixed output length")
    parser.add_argument("--prompt-chars", type=int, default=900, help="approximate prompt length in characters")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 keeps output length stable")
    parser.add_argument("--warmup", type=int, default=4, help="discarded warm-up requests")
    parser.add_argument("--cooldown", type=float, default=2.0, help="seconds between levels")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request client timeout")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--label", default="run", help="label used in output filenames")
    parser.add_argument("--seed", type=int, default=42, help="prompt seed")
    parser.add_argument("--no-stream", action="store_true", help="use non-streaming requests (no TTFT granularity)")
    parser.add_argument(
        "--same-host",
        action="store_true",
        help="assert the client runs on the same machine as the server (recommended)",
    )
    parser.add_argument("--output-dir", default="bench/results", help="where to write JSON/Markdown")
    parser.add_argument("--no-files", action="store_true", help="print only, write nothing")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="honour HTTP(S)_PROXY for the benchmark client (off by default: localhost must never be proxied)",
    )
    return parser


def normalize_base_url(url: str) -> str:
    """Return the OpenAI API root, appending /v1 when the user omitted it.

    The OpenAI SDK appends `/chat/completions` directly to base_url, so passing
    `http://host:8000` produces `http://host:8000/chat/completions` (404 against
    vLLM). Accept both forms to avoid a confusing failure.
    """
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


async def async_main(args: argparse.Namespace) -> int:
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    if not levels:
        raise SystemExit("--concurrency is empty")

    args.base_url = normalize_base_url(args.base_url)

    bench = Benchmark(args)
    started_at = datetime.now(timezone.utc).isoformat()
    run = RunResult(
        label=args.label,
        base_url=args.base_url,
        model=args.model,
        started_at=started_at,
        finished_at="",
        environment=await fetch_server_metadata(args) | environment_info(args),
        config=parse_config(args),
    )

    print(f"server   : {args.base_url}  model={args.model}")
    print(f"levels   : {levels}  requests/level={args.num_requests}  max_tokens={args.max_tokens}")
    if args.warmup:
        print(f"warmup   : {args.warmup} requests (discarded)")
    await bench.warmup()

    try:
        for concurrency in levels:
            print(f"\n--- concurrency={concurrency} ---", flush=True)
            level = await bench.run_level(concurrency)
            run.levels.append(level)
            print(
                f"QPS={level.qps:.2f}  out_tok/s={level.output_tokens_per_second:.1f}  "
                f"TTFT p50={level.ttft['p50']:.3f}s p95={level.ttft['p95']:.3f}s  "
                f"E2E p99={level.e2e['p99']:.3f}s  errors={level.errors}",
                flush=True,
            )
            if level.sample_error:
                print(f"  sample error: {level.sample_error}", flush=True)
            await asyncio.sleep(args.cooldown)
    finally:
        await bench.aclose()

    run.finished_at = datetime.now(timezone.utc).isoformat()
    table = markdown_table(run)
    print("\n" + table)

    if not args.no_files:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = f"{args.label}-{stamp}-{uuid.uuid4().hex[:6]}"
        json_path = out_dir / f"{stem}.json"
        md_path = out_dir / f"{stem}.md"
        json_path.write_text(json.dumps(run.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(
            f"# Benchmark: {args.label}\n\n"
            f"- model: `{args.model}`\n- base_url: `{args.base_url}`\n"
            f"- started: {run.started_at}\n- finished: {run.finished_at}\n\n"
            f"{table}\n",
            encoding="utf-8",
        )
        print(f"\nwrote {json_path}\nwrote {md_path}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_requests < 1:
        raise SystemExit("--num-requests must be >= 1")
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
