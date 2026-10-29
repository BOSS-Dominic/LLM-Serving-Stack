# docs/figures

报告里引用的图放这里。

| 文件 | 内容 | 状态 |
|---|---|---|
| `throughput-vs-concurrency.png` | 并发 vs 输出 tok/s（实测中位数） | ✅ 已生成 |
| `latency-vs-concurrency.png` | 并发 vs TTFT P95 / E2E P99（实测中位数） | ✅ 已生成 |
| `dashboard.png` | Grafana 看板截图（压测进行中，能看到 running/waiting、KV cache、TTFT） | ⏳ 待补（需手动截图） |

生成方式：

```bash
python bench/plot.py --results bench/results/rtx5070ti-qwen1.5b-median.json --out docs/figures
```

`bench/plot.py` 读取压测 JSON，输出上面两张曲线图；多个 `--results` 文件会画在同一张图上，
可直接用于 vLLM vs HF 对比。看板截图需要真正启动 `deploy/docker-compose.yml` 后手动截取。
