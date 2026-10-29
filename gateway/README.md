# gateway — 轻量 LLM 推理网关

一个只依赖标准库 + Prometheus client 的 OpenAI 兼容网关，放在多个 vLLM 实例前面，
负责健康检查、负载均衡、SSE 流式透传、失败重试和指标暴露。

```
client ──▶ gateway :9000 ──┬──▶ vllm-1 :8000
                           └──▶ vllm-2 :8000
```

## 快速开始

```bash
cd gateway
go build -o gateway .
./gateway -config config.yaml

curl -s localhost:9000/healthz
curl -s localhost:9000/readyz
curl -s localhost:9000/debug/backends | jq
curl -s localhost:9000/metrics | grep '^gateway_'
```

没有 GPU 也可以验证：用 `bench/mock_vllm.py` 起两个假后端。

```bash
python ../bench/mock_vllm.py --port 18001 &
python ../bench/mock_vllm.py --port 18002 &
GATEWAY_BACKENDS="m1=http://127.0.0.1:18001,m2=http://127.0.0.1:18002" ./gateway -config config.yaml
# 然后 kill 掉其中一个，观察 /debug/backends 里的 healthy 变成 false
```

## HTTP 接口

| 路径 | 说明 |
|---|---|
| `/*` | 透传到后端（`/v1/chat/completions`、`/v1/models`、`/v1/completions` 等） |
| `/healthz` | 网关自身存活，恒返回 200 |
| `/readyz` | 至少有一个健康后端才 200，否则 503 |
| `/debug/backends` | 每个后端的 healthy / running / total_requests 快照 |
| `/metrics` | Prometheus 指标 |

## 配置（config.yaml）

| 字段 | 默认 | 说明 |
|---|---|---|
| `listen` | `:9000` | 监听地址 |
| `strategy` | `least_conn` | `least_conn` 或 `round_robin`（平滑加权轮询） |
| `backends[].weight` | `1` | 权重，参与两种策略 |
| `health_check.interval` | `10s` | 探活周期 |
| `health_check.path` | `/health` | vLLM 的 `/health` 在权重加载完成后才 200 |
| `health_check.unhealthy_after` | `2` | 连续失败几次才摘除 |
| `health_check.healthy_after` | `2` | 连续成功几次才恢复 |
| `timeouts.connect` | `5s` | 建连超时 |
| `timeouts.request` | `60s` | **非流式**请求总超时 |
| `timeouts.stream_first_byte` | `30s` | **流式**请求首字节超时（之后不设总超时） |
| `retry.max` | `0` | 最大重试次数（0 = 不重试，需要显式开启） |
| `retry.max_body_bytes` | `8 MiB` | 请求体上限，也是重试时可缓存的上限 |
| `retry.max_response_body_bytes` | `4 MiB` | 非流式响应缓冲上限 |
| `rate_limit.rps` | `0` | 全局限流，0 = 关闭 |

环境变量覆盖：`GATEWAY_LISTEN`、`GATEWAY_BACKENDS`（`name=url,name=url` 或纯 URL 列表）。

## 设计要点

- **原子选主**：`选择 + Running++` 在同一把 `selMu` 内完成，避免"两个请求同时看到最小连接数、
  都选同一台"的竞态；顺序请求下用 `TotalRequests` 打破平局，避免低并发时全打第一台。
- **流式不缓冲**：`http.Flusher` 逐块下发；响应头在**读到第一个 chunk 之后**才提交，
  这样"后端在 prefill 阶段挂掉"仍然可以安全重试。
- **首字节后不重试**：客户端已经收到 token，换后端会造成重复/断裂。
- **超时分离**：流式用首字节超时，非流式用总超时，`http.Client.Timeout` 保持 0。
- **软摘除**：健康检查只影响新请求的选择，在途请求继续跑完。
- **probe 不复用连接**：探活要验证"新连接"是否可用，keep-alive 会掩盖停止 accept 的实例。

更多取舍见 [`../docs/decisions.md`](../docs/decisions.md)。

## 测试

```bash
go vet ./...
go test -race -count=1 ./...
```

`go test` 覆盖了：重试时 body 非空（最容易假成功的一项）、4xx 不重试、首字节前重试、
首字节后不重试、SSE 增量下发（经过 access log 包装）、客户端断连取消后端、超时映射 504、
并发选择分布、健康检查摘除/恢复、配置校验。

## 限制

- 不支持 backend 热加载（改配置需重启，重启时健康状态会重新探测）。
- 限流是进程级全局令牌桶，不是按租户/按模型。
- 没有实现 prefill/decode 分离、按 KV Cache 利用率路由；这些是 vLLM Router 的方向。
