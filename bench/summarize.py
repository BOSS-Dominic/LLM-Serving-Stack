#!/usr/bin/env python3
"""Turn bench/results/*.json into the Markdown tables used in the report.

Hand-copying numbers from a terminal into a report is how typos and stale
figures get into a resume. This script is the single source of truth for the
tables: re-run it after every benchmark.

    python bench/summarize.py --include-mock          # show everything
    python bench/summarize.py --out docs/generated/benchmark-tables.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Run:
    path: Path
    label: str
    model: str
    levels: list[dict[str, Any]]
    config: dict[str, Any]
    environment: dict[str, Any]
    finished_at: str


def load_runs(results_dir: Path, include_mock: bool) -> list[Run]:
    runs: list[Run] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"skip unreadable {path}", file=sys.stderr)
            continue
        label = str(doc.get("label", path.stem))
        if "mock" in label.lower() and not include_mock:
            continue
        runs.append(
            Run(
                path=path,
                label=label,
                model=doc.get("model", "?"),
                levels=doc.get("levels", []),
                config=doc.get("config", {}),
                environment=doc.get("environment", {}),
                finished_at=doc.get("finished_at", ""),
            )
        )
    return runs


def level_table(run: Run) -> str:
    lines = [
        f"#### `{run.label}` — model `{run.model}`",
        "",
        "| 并发 | 完成/总数 | QPS | 输出 tok/s | 总 tok/s | TTFT P50 (s) | TTFT P95 (s) | TTFT P99 (s) | E2E P50 (s) | E2E P95 (s) | E2E P99 (s) | TPOT P50 (ms) | 错误率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lvl in run.levels:
        ttft, e2e, tpot = lvl["ttft"], lvl["e2e"], lvl["tpot"]
        lines.append(
            f"| {lvl['concurrency']} | {lvl['completed']}/{lvl['num_requests']} | {lvl['qps']:.2f} | "
            f"{lvl['output_tokens_per_second']:.1f} | {lvl['total_tokens_per_second']:.1f} | "
            f"{ttft['p50']:.3f} | {ttft['p95']:.3f} | {ttft['p99']:.3f} | "
            f"{e2e['p50']:.3f} | {e2e['p95']:.3f} | {e2e['p99']:.3f} | "
            f"{tpot['p50'] * 1000:.1f} | {lvl['error_rate'] * 100:.1f}% |"
        )
    if any(lvl.get("tokens_estimated") for lvl in run.levels):
        lines += ["", "> token 数由 chunk 计数估算（服务端未返回 usage），吞吐数字仅作参考。"]
    return "\n".join(lines)


def capacity_baseline(run: Run) -> str:
    """Derive "max concurrency that still satisfies each SLA" from the levels."""
    levels = [lvl for lvl in run.levels if lvl["completed"] > 0]
    if not levels:
        return "_没有可用档位_"

    def best(predicate):
        ok = [lvl for lvl in levels if predicate(lvl)]
        return max(ok, key=lambda lvl: lvl["concurrency"]) if ok else None

    ttft_1s = best(lambda lvl: lvl["ttft"]["p99"] < 1.0)
    strict = best(lambda lvl: lvl["ttft"]["p99"] < 1.0 and lvl["e2e"]["p99"] < 5.0)
    throughput = max(levels, key=lambda lvl: lvl["output_tokens_per_second"])

    def row(name, lvl, bottleneck):
        if not lvl:
            return f"| {name} | — | — | — | — | 所有档位都不满足，需降低并发或扩容 |"
        return (
            f"| {name} | {lvl['concurrency']} | {lvl['output_tokens_per_second']:.1f} | "
            f"{lvl['ttft']['p99']:.3f} | {lvl['e2e']['p99']:.3f} | {bottleneck} |"
        )

    # A rough bottleneck heuristic, explicitly labelled as a heuristic.
    def bottleneck_for(lvl):
        if not lvl:
            return "—"
        if lvl["ttft"]["p99"] > 2 * max(lvl["e2e"]["p50"] - lvl["ttft"]["p50"], 1e-6):
            return "prefill / 排队（TTFT 占比高）"
        return "decode / 显存（TPOT 占比高）"

    lines = [
        "| 约束（可接受上限） | 最大可持续并发 | 对应吞吐 (tok/s) | TTFT P99 (s) | E2E P99 (s) | 瓶颈判断 |",
        "|---|---:|---:|---:|---:|---|",
        row("TTFT P99 < 1s", ttft_1s, bottleneck_for(ttft_1s)),
        row("TTFT P99 < 1s 且 E2E P99 < 5s", strict, bottleneck_for(strict)),
        row("吞吐优先（不看延迟）", throughput, "显存 / 调度上限"),
        "",
        "> 瓶颈一栏是根据 TTFT 与 TPOT 占比的启发式判断，最终结论应结合 Grafana 的 "
        "`vllm:gpu_cache_usage_perc` 与 `vllm:num_requests_waiting` 复核。",
    ]
    return "\n".join(lines)


def render(runs: list[Run]) -> str:
    if not runs:
        return "_没有找到结果文件_"

    parts = [
        "# 压测结果（自动生成）",
        "",
        "> 由 `python bench/summarize.py` 从 `bench/results/*.json` 生成，请勿手工修改。",
        "",
    ]
    for run in runs:
        parts += [
            f"## {run.label}",
            "",
            f"- model: `{run.model}`",
            f"- 完成时间: {run.finished_at}",
            f"- 请求总数/档位: {run.config.get('num_requests')} × {run.config.get('concurrency')}",
            f"- max_tokens: {run.config.get('max_tokens')}，stream: {run.config.get('stream')}",
            "",
            level_table(run),
            "",
            "### 容量基线",
            "",
            capacity_baseline(run),
            "",
        ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="bench/results", help="directory with benchmark JSON files")
    parser.add_argument("--include-mock", action="store_true", help="include mock-engine runs")
    parser.add_argument("--out", default="", help="write Markdown here instead of stdout")
    args = parser.parse_args(argv)

    runs = load_runs(Path(args.results), args.include_mock)
    if not runs:
        print(
            "no results found. Run bench_pro.py first, or pass --include-mock for harness validation runs.",
            file=sys.stderr,
        )
        return 1

    text = render(runs)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out} ({len(runs)} run(s))")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
