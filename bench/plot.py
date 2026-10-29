#!/usr/bin/env python3
"""Plot benchmark results as PNGs (no GPU needed, no display needed).

    python bench/plot.py --results bench/results/rtx4090-*.json --out docs/figures

Produces:
    latency-vs-concurrency.png   TTFT P95 and E2E P99 per level
    throughput-vs-concurrency.png   output tok/s per level

Multiple runs are drawn as separate series, so a vLLM run and an HF baseline
run can share one chart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def load(paths: list[Path]) -> list[dict]:
    runs = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            doc = json.load(fh)
        if not doc.get("levels"):
            continue
        runs.append(doc)
    return runs


def plot_latency(runs: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    for run in runs:
        levels = run["levels"]
        xs = [lvl["concurrency"] for lvl in levels]
        ax.plot(xs, [lvl["ttft"]["p95"] for lvl in levels], marker="o",
                label=f"{run['label']} · TTFT P95")
        ax.plot(xs, [lvl["e2e"]["p99"] for lvl in levels], marker="s", linestyle="--",
                label=f"{run['label']} · E2E P99")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency vs concurrency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_throughput(runs: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    width = 0.8 / max(len(runs), 1)
    for i, run in enumerate(runs):
        levels = run["levels"]
        xs = [j + i * width for j in range(len(levels))]
        vals = [lvl["output_tokens_per_second"] for lvl in levels]
        ax.bar(xs, vals, width=width, label=run["label"])
        ax.set_xticks([j + width * (len(runs) - 1) / 2 for j in range(len(levels))])
        ax.set_xticklabels([str(lvl["concurrency"]) for lvl in levels])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Output tokens / s")
    ax.set_title("Throughput vs concurrency")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True, help="benchmark JSON files")
    parser.add_argument("--out", default="docs/figures", help="output directory")
    args = parser.parse_args()

    paths = [Path(p) for p in args.results]
    runs = load(paths)
    if not runs:
        raise SystemExit("no usable result files")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_latency(runs, out_dir / "latency-vs-concurrency.png")
    plot_throughput(runs, out_dir / "throughput-vs-concurrency.png")
    print(f"wrote {out_dir / 'latency-vs-concurrency.png'}")
    print(f"wrote {out_dir / 'throughput-vs-concurrency.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
