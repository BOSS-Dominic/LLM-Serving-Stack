# bench/results

每次压测的原始结果。**文件名决定它能不能进报告**：

| 前缀 | 含义 | 能否用于容量结论 |
|---|---|---|
| `rtx5070ti-*` 等真实标签 | 真实 vLLM 引擎 | ✅ 可以 |
| `e2e-*mock*`、`overhead-*`、`*-mock` | `bench/mock_vllm.py` 假引擎 | ❌ 只能证明工具链正确 |

规则：

- 真实引擎的结果用 `--label <机器>-<模型>-<日期>` 命名，例如 `rtx5070ti-qwen1.5b-1029`。
- mock 结果必须带 `mock` 字样（label 或 model），`bench/summarize.py` 默认会过滤掉；
  需要看时加 `--include-mock`。
- 不要手工编辑 JSON；要改结论就重跑，或改 `summarize.py`。
- JSON 里记录了 `environment`、`config`、每档分位数和错误率，报告中的表格必须能从这里重新生成。
- 同一配置建议跑 3 次，用 `bench/aggregate.py` 取中位数，原始 run 放 `repeats/`。

当前文件：

- `rtx5070ti-qwen1.5b-median.json`：**报告使用的数据集**，3 次独立运行的逐档中位数
  （由 `bench/aggregate.py` 生成，文件内 `aggregate` 字段标明来源）。
- `repeats/rtx5070ti-*.json`：三次独立运行的原始数据，用于复算中位数与波动。
- `e2e-gateway-mock-*`：2 个 mock 后端 + 真实 Go 网关，验证均衡与故障摘除（报告 §3.1）。
- `overhead-direct-*` / `overhead-gateway-*`：同一后端直连 vs 经网关，测网关开销（报告 §3.2）。

> 第一批 GPU 数据（2026-10-29 15:18–15:21）因后台游戏占用 GPU 导致吞吐腰斩，已整批删除，
> 不参与任何统计。这也是压测前必须用 `nvidia-smi` 确认 GPU 空闲的原因（见 docs/decisions.md B7）。
