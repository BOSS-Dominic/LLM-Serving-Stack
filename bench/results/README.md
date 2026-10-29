# bench/results

每次压测的原始结果。**文件名决定它能不能进报告**：

| 前缀 | 含义 | 能否用于简历/容量结论 |
|---|---|---|
| `rtx4090-*`、`a100-*`…（真实标签） | 真实 vLLM 引擎 | ✅ 可以 |
| `e2e-*mock*`、`overhead-*`、`*-mock` | `bench/mock_vllm.py` 假引擎 | ❌ 只能证明工具链正确 |

规则：

- 真实引擎的结果用 `--label <机器>-<模型>-<日期>` 命名，例如 `rtx4090-qwen7b-1029`。
- mock 结果必须带 `mock` 字样，`bench/summarize.py` 默认会过滤掉它们；
  需要看时加 `--include-mock`。
- 不要手工编辑 JSON；要改结论就重跑，或改 `summarize.py`。
- JSON 里记录了 `environment`、`config`、每档分位数和错误率，报告中的表格必须能从这里重新生成。

当前文件：

- `e2e-gateway-mock-*`：2 个 mock 后端 + 真实 Go 网关，验证均衡与故障摘除（报告 §3.1）。
- `overhead-direct-*` / `overhead-gateway-*`：同一后端直连 vs 经网关，测网关开销（报告 §3.2）。
