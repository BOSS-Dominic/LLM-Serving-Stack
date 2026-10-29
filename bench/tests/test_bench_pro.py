"""Unit tests for bench_pro.py.

These run without a GPU: they cover the statistics and summarisation logic,
which is where a benchmark silently lies if it is wrong.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_pro import (  # noqa: E402
    LevelResult,
    RequestResult,
    RunResult,
    build_prompt,
    markdown_table,
    percentile,
    summarize_level,
)


class TestPercentile:
    def test_interpolates_between_neighbours(self):
        # numpy.percentile([1,2,3,4], 95) == 3.85
        assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)

    def test_p95_of_four_is_not_p75(self):
        # The classic bug: sorted(xs)[int(4*0.95)-1] == xs[2] == P75.
        assert percentile([1, 2, 3, 4], 0.95) > 3.0

    def test_edges(self):
        assert percentile([5.0], 0.99) == 5.0
        assert percentile([1.0, 2.0], 0.0) == 1.0
        assert percentile([1.0, 2.0], 1.0) == 2.0

    def test_empty_is_nan(self):
        assert math.isnan(percentile([], 0.5))

    def test_rejects_invalid_quantile(self):
        with pytest.raises(ValueError):
            percentile([1.0], 1.5)


class TestSummarizeLevel:
    def _results(self):
        return [
            RequestResult(ok=True, status=200, ttft=0.10, e2e=1.10, completion_tokens=101, prompt_tokens=50),
            RequestResult(ok=True, status=200, ttft=0.20, e2e=2.20, completion_tokens=101, prompt_tokens=50),
            RequestResult(ok=False, status=500, e2e=0.5, error="boom"),
            RequestResult(ok=False, status=0, e2e=0.5, error="timeout"),
        ]

    def test_error_rate_and_throughput(self):
        level = summarize_level(4, 4, wall=4.0, results=self._results())
        assert level.completed == 2
        assert level.errors == 2
        assert level.error_rate == pytest.approx(0.5)
        assert level.qps == pytest.approx(0.5)  # 2 ok / 4s
        assert level.output_tokens_per_second == pytest.approx(202 / 4.0)
        assert level.total_tokens_per_second == pytest.approx((202 + 100) / 4.0)

    def test_tpot_excludes_prefill(self):
        # (e2e - ttft)/100: 1.0/100 = 0.01 for the first, 2.0/100 = 0.02 for the
        # second; the interpolated median of two samples is 0.015.
        level = summarize_level(1, 2, wall=3.3, results=self._results()[:2])
        assert level.tpot["p50"] == pytest.approx(0.015, rel=1e-6)

    def test_tokens_estimated_flag(self):
        results = [RequestResult(ok=True, status=200, ttft=0.1, e2e=0.2, completion_tokens=10, tokens_estimated=True)]
        level = summarize_level(1, 1, wall=0.2, results=results)
        assert level.tokens_estimated is True

    def test_no_successful_requests(self):
        results = [RequestResult(ok=False, status=503, e2e=1.0, error="down")]
        level = summarize_level(1, 1, wall=1.0, results=results)
        assert level.completed == 0
        assert level.error_rate == 1.0
        assert level.qps == 0.0
        assert math.isnan(level.ttft["p50"])


class TestPrompt:
    def test_deterministic(self):
        assert build_prompt(500, seed=1) == build_prompt(500, seed=1)

    def test_respects_approximate_length(self):
        prompt = build_prompt(600, seed=2)
        assert len(prompt) >= 600
        assert len(prompt) < 700 + 80


class TestMarkdown:
    def test_table_has_header_and_rows(self):
        level = LevelResult(
            concurrency=8,
            num_requests=100,
            wall_seconds=10.0,
            completed=100,
            errors=0,
            error_rate=0.0,
            qps=10.0,
            output_tokens_per_second=1280.0,
            total_tokens_per_second=1500.0,
            ttft={"p50": 0.1, "p95": 0.2, "p99": 0.3, "mean": 0.15},
            tpot={"p50": 0.01, "p95": 0.02, "p99": 0.03, "mean": 0.015},
            e2e={"p50": 1.0, "p95": 2.0, "p99": 3.0, "mean": 1.5},
            prompt_tokens_mean=50,
            completion_tokens_mean=128,
            tokens_estimated=False,
        )
        run = RunResult(
            label="test",
            base_url="http://localhost",
            model="m",
            started_at="now",
            finished_at="later",
            environment={},
            config={},
            levels=[level],
        )
        table = markdown_table(run)
        assert "| 并发 |" in table
        assert "| 8 |" in table
        assert "10.00" in table
