#!/usr/bin/env python3
"""Aggregate repeated benchmark runs into a median run (plus a spread report).

One benchmark run on a laptop GPU varies by 5-20% because of clocks, thermals
and background GPU users. Publishing the fastest run would overstate the
result; this script takes the per-level median of N independent runs and also
records the min/max, so the report can state both the number and its stability.

    python bench/aggregate.py \
      --pattern "bench/results/repeats/rtx5070ti-*.json" \
      --label rtx5070ti-qwen1.5b-median \
      --out bench/results/rtx5070ti-qwen1.5b-median.json

The output JSON has the same shape as a single run, so `summarize.py` and
`plot.py` consume it unchanged. It is explicitly marked as an aggregate; it is
NOT a measured run.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any

PERCENTILE_KEYS = ("p50", "p95", "p99", "mean")


def load_runs(pattern: str) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(pattern))
    runs = []
    for p in paths:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        if doc.get("levels"):
            doc["_path"] = p
            runs.append(doc)
    if not runs:
        raise SystemExit(f"no runs matched {pattern}")
    return runs


def median_dict(values: list[dict[str, float]]) -> dict[str, float]:
    return {k: round(statistics.median([v[k] for v in values]), 4) for k in PERCENTILE_KEYS}


def median_scalar(values: list[float]) -> float:
    return round(statistics.median(values), 4)


def spread(values: list[float]) -> dict[str, float]:
    lo, hi = min(values), max(values)
    med = statistics.median(values)
    return {
        "min": round(lo, 4),
        "median": round(med, 4),
        "max": round(hi, 4),
        "spread_pct": round((hi - lo) / med * 100, 1) if med else 0.0,
    }


def aggregate(runs: list[dict[str, Any]], label: str) -> tuple[dict[str, Any], str]:
    base = runs[0]
    by_concurrency: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        for level in run["levels"]:
            by_concurrency.setdefault(int(level["concurrency"]), []).append(level)

    levels: list[dict[str, Any]] = []
    variability: list[dict[str, Any]] = []

    for concurrency in sorted(by_concurrency):
        group = by_concurrency[concurrency]
        ttft = median_dict([g["ttft"] for g in group])
        tpot = median_dict([g["tpot"] for g in group])
        e2e = median_dict([g["e2e"] for g in group])

        qps = [g["qps"] for g in group]
        outs = [g["output_tokens_per_second"] for g in group]

        levels.append(
            {
                "concurrency": concurrency,
                "num_requests": int(statistics.median([g["num_requests"] for g in group])),
                "wall_seconds": median_scalar([g["wall_seconds"] for g in group]),
                "completed": int(statistics.median([g["completed"] for g in group])),
                "errors": int(statistics.median([g["errors"] for g in group])),
                "error_rate": median_scalar([g["error_rate"] for g in group]),
                "qps": median_scalar(qps),
                "output_tokens_per_second": median_scalar(outs),
                "total_tokens_per_second": median_scalar(
                    [g["total_tokens_per_second"] for g in group]
                ),
                "ttft": ttft,
                "tpot": tpot,
                "e2e": e2e,
                "prompt_tokens_mean": median_scalar([g["prompt_tokens_mean"] for g in group]),
                "completion_tokens_mean": median_scalar(
                    [g["completion_tokens_mean"] for g in group]
                ),
                "tokens_estimated": any(g.get("tokens_estimated") for g in group),
                "sample_error": None,
            }
        )
        variability.append(
            {
                "concurrency": concurrency,
                "runs": len(group),
                "qps": spread(qps),
                "output_tokens_per_second": spread(outs),
                "ttft_p99": spread([g["ttft"]["p99"] for g in group]),
                "e2e_p99": spread([g["e2e"]["p99"] for g in group]),
            }
        )

    doc = {
        "label": label,
        "model": base.get("model"),
        "base_url": base.get("base_url"),
        "started_at": min(r.get("started_at", "") for r in runs),
        "finished_at": max(r.get("finished_at", "") for r in runs),
        "environment": dict(base.get("environment", {})),
        "config": dict(base.get("config", {})),
        "aggregate": {
            "method": "median",
            "run_count": len(runs),
            "sources": [Path(r["_path"]).name for r in runs],
            "note": "Per-level median across independent runs. Not a single measured run.",
        },
        "levels": levels,
    }

    lines = [
        f"# Aggregate: {label}",
        "",
        f"- method: **median of {len(runs)} runs**",
        "- sources: " + ", ".join(f"`{s}`" for s in doc["aggregate"]["sources"]),
        "",
        "## Run-to-run variability",
        "",
        "| 并发 | 运行次数 | QPS (min/median/max) | 输出 tok/s (min/median/max) | tok/s 波动 | TTFT P99 波动 | E2E P99 波动 |",
        "|---:|---:|---|---|---:|---:|---:|",
    ]
    for v in variability:
        q, o = v["qps"], v["output_tokens_per_second"]
        lines.append(
            f"| {v['concurrency']} | {v['runs']} | {q['min']:.2f} / {q['median']:.2f} / {q['max']:.2f} | "
            f"{o['min']:.0f} / {o['median']:.0f} / {o['max']:.0f} | {o['spread_pct']:.1f}% | "
            f"{v['ttft_p99']['spread_pct']:.1f}% | {v['e2e_p99']['spread_pct']:.1f}% |"
        )
    return doc, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", required=True, help="glob of independent run JSONs")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, help="output aggregate JSON")
    parser.add_argument("--markdown", default="", help="optional variability Markdown")
    args = parser.parse_args()

    runs = load_runs(args.pattern)
    doc, md = aggregate(runs, args.label)

    out = Path(args.out)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} (median of {len(runs)} runs)")
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
