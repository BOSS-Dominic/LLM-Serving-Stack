# docs/figures

报告里引用的图放这里。

| 文件 | 内容 | 状态 |
|---|---|---|
| `dashboard.png` | Grafana 看板截图（压测进行中，能看到 running/waiting、KV cache、TTFT） | ⏳ 待补（需手动截图） |
| `latency-vs-concurrency.png` | 并发 vs TTFT P95 / E2E P99 | 由 `bench/plot.py` 生成 |
| `throughput-vs-concurrency.png` | 并发 vs 输出 tok/s | 由 `bench/plot.py` 生成 |

生成方式（真实 GPU 数据跑完后）：

```bash
python bench/plot.py --results bench/results/rtx4090-qwen7b-*.json --out docs/figures
```

`bench/plot.py` 读取压测 JSON，输出上面两张曲线图；多个 `--results` 文件会画在同一张图上，
可直接用于 vLLM vs HF 对比。看板截图需要真正启动 `deploy/docker-compose.yml` 后手动截取。
