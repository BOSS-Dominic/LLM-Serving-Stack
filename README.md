# LLM Serving Stack

[![ci](https://github.com/BOSS-Dominic/LLM-Serving-Stack/actions/workflows/ci.yml/badge.svg)](https://github.com/BOSS-Dominic/LLM-Serving-Stack/actions/workflows/ci.yml)
[![repo](https://img.shields.io/badge/github-BOSS--Dominic%2FLLM--Serving--Stack-181717?logo=github)](https://github.com/BOSS-Dominic/LLM-Serving-Stack)

> vLLM serving · closed-loop benchmarking · Go multi-instance gateway · Prometheus/Grafana observability

**English** | [中文](#llm-推理服务工程化实践)

A two-layer project about running an LLM service in production:

- **Engine layer** — vLLM serving Qwen2.5 behind an OpenAI-compatible API, benchmarked with a closed-loop
  concurrency sweep (TTFT, TPOT, P99, throughput) and turned into a capacity baseline.
- **Access layer** — a Go gateway in front of multiple vLLM instances: active health checks with soft
  removal, least-connections balancing, SSE pass-through, retries that only happen before the first byte,
  and Prometheus metrics.

Every number in this repository is produced by scripts under `bench/`. Nothing is hand-copied.

## Results at a glance

Measured on an RTX 5070 Ti Laptop (12 GB) with Qwen2.5-1.5B-Instruct and vLLM 0.31.0.
Each level is the median of 3 independent runs; 100 requests per level, `max_tokens=128` (cap —
actual completions were ≈ 47–50 tokens because the prompt asks for a one-sentence summary),
temperature 0.

| Concurrency | QPS | Output tok/s | TTFT P50 | TTFT P99 | E2E P99 | TPOT P50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.34 | 114.6 | 20 ms | 24 ms | 0.459 s | 8.3 ms |
| 4 | 9.67 | 457.2 | 33 ms | 44 ms | 0.537 s | 8.1 ms |
| 8 | 18.17 | 836.7 | 43 ms | 84 ms | 0.528 s | 8.2 ms |
| 16 | 29.11 | 1363.1 | 66 ms | 80 ms | 0.628 s | 8.9 ms |
| 32 | 44.60 | 2110.0 | 108 ms | 128 ms | 0.807 s | 10.9 ms |

- 1 → 32 concurrent requests: **114.6 → 2110 output tok/s (18.4×)**, 2.34 → 44.6 QPS.
- TTFT P99 stays under 128 ms; E2E P99 stays under 807 ms; 500/500 requests succeeded.
- 32 concurrency is the KV-cache ceiling at 2048 context (31.24 sequences); beyond that requests queue.

## Architecture

```
                          ┌──────────────────────────────────────────┐
   client                 │            gateway (Go, :9000)           │
  (OpenAI SDK)            │                                          │
      │  /v1/chat/completions  ┌─────────────┐  ┌──────────────────┐ │
      ├──────────────────────▶ │  balancer   │  │ health checker    │ │
      │   stream: true         │ least_conn/ │  │ 10s probe, remove │ │
      │                        │ round_robin │  │ after 2 failures  │ │
      │                        └──────┬──────┘  └────────┬─────────┘ │
      │                               │                  │           │
      │                        ┌──────▼──────────────────▼─────────┐ │
      │                        │ proxy: buffered body, retry before │ │
      │                        │ first byte, SSE flush, ctx cancel  │ │
      │                        └──────┬───────────────────┬─────────┘ │
      └───────────────────────────────┘                   │           │
                                                          │           │
                    ┌─────────────────────────────────────┴──────┐    │
                    │                                            │    │
              ┌─────▼──────┐                              ┌──────▼──┐ │
              │ vllm-1     │   PagedAttention             │ vllm-2  │ │
              │ :8000      │   continuous batching        │ :8000   │ │
              └─────┬──────┘                              └────┬────┘ │
                    └──────────────┬───────────────────────────┘      │
                                   │ /metrics                          │
                          ┌────────▼─────────┐   ┌──────────────────┐  │
                          │  Prometheus      │──▶│  Grafana         │  │
                          │  :9090           │   │  :3000           │  │
                          └──────────────────┘   └──────────────────┘  │
                                                                       │
                          └───────────────────────────────────────────┘
```

Request lifecycle inside the engine:

```
Client → OpenAI API → scheduler (waiting queue / admission)
       → batch assembly every iteration (continuous batching)
       → prefill (compute-bound) → decode (memory-bound, one token at a time)
       → KV blocks allocated / recycled on demand (PagedAttention, 16 tokens/block)
       → streamed back to the client, blocks released at the end
```

## Repository layout

```
llm-serving-stack/
├── README.md                     # this file
├── docs/
│   ├── benchmark-report.md       # method, results, capacity baseline
│   └── decisions.md              # design trade-offs ("why not X")
├── bench/
│   ├── bench_pro.py              # closed-loop benchmark (TTFT/TPOT/P50/P95/P99/QPS)
│   ├── mock_vllm.py              # OpenAI-compatible mock for CI and local testing
│   ├── summarize.py              # JSON → Markdown tables + capacity baseline
│   ├── aggregate.py              # N runs → median + min/median/max spread
│   ├── plot.py                   # JSON → latency / throughput charts
│   ├── results/                  # benchmark JSON (source of truth for the report)
│   └── tests/                    # statistics + end-to-end tests
├── gateway/
│   ├── main.go / config.go       # entry point, config, graceful shutdown
│   ├── backend.go                # backend model, pool, health checks
│   ├── balancer.go               # least-connections / smooth weighted round-robin
│   ├── proxy.go                  # forwarding, SSE, retries, timeouts
│   ├── metrics.go                # Prometheus collectors
│   ├── *_test.go                 # concurrency / retry / streaming / disconnect tests
│   ├── config.yaml
│   └── Dockerfile
├── deploy/
│   ├── docker-compose.yml        # vllm / gateway / prometheus / grafana
│   ├── prometheus/               # scrape config + 7 alert rules
│   └── grafana/                  # datasource, dashboard provider, dashboard JSON
└── .github/workflows/ci.yml      # Go tests + Python tests + config validation
```

## Quick start

Requirements: Go 1.26+ for the gateway, Python 3.10+ for the benchmark tooling.
A real vLLM needs Linux + an NVIDIA GPU (vLLM does not run natively on Windows) + Docker.

### 1. No GPU: mock engine + gateway (about 2 minutes)

```bash
# two fake vLLM backends
python bench/mock_vllm.py --port 18001 --token-ms 2 &
python bench/mock_vllm.py --port 18002 --token-ms 2 &

# build and start the gateway
(cd gateway && go build -o gateway .)
(cd gateway && GATEWAY_BACKENDS="m1=http://127.0.0.1:18001,m2=http://127.0.0.1:18002" ./gateway -config config.yaml) &

# benchmark through the gateway
python bench/bench_pro.py --base-url http://127.0.0.1:9000 --model mock-qwen \
  --concurrency 1,4,8 --num-requests 40 --max-tokens 32

# backend states (kill one mock and watch it leave rotation)
curl -s localhost:9000/debug/backends
```

### 2. GPU: real vLLM + monitoring stack

```bash
cd deploy
cp .env.example .env          # defaults target a 12 GB card + Qwen2.5-1.5B
docker compose --profile gpu up -d
docker compose logs -f vllm   # wait for model loading to finish
```

### 3. Benchmark a real engine

```bash
# the client must run on the same machine as vLLM
python bench/bench_pro.py --base-url http://127.0.0.1:8000 --model qwen-1.5b \
  --concurrency 1,4,8,16,32 --num-requests 100 --max-tokens 128 \
  --label local-qwen1.5b --same-host

# regenerate the report tables
python bench/summarize.py --out docs/generated/benchmark-tables.md
```

Grafana: <http://localhost:3000> (admin/admin) → pre-provisioned dashboard **LLM Serving Stack**.

## Design highlights

- **Atomic selection + reservation.** Choosing the least-loaded backend and incrementing its in-flight
  counter happen under one short lock; otherwise two concurrent requests pick the same instance and the
  balancer degenerates into "everything to one box".
- **Tie-break by lifetime assignments.** Two idle backends compare equal; without an explicit tie-break the
  first one wins every time. An end-to-end run showed an 83:41 skew before this was fixed.
- **SSE headers are committed only after the first chunk is read.** If a backend dies during prefill, the
  client has received nothing yet, so the request can still be retried on another instance.
- **Never retry after the first byte.** Once a token has reached the client, replaying the generation
  elsewhere would duplicate or corrupt the output.
- **Split timeouts.** Streams get a first-byte deadline, non-streams get a total deadline, and
  `http.Client.Timeout` stays zero so healthy multi-minute generations are not cut off.
- **Health probes open fresh connections.** A keep-alive probe can keep reporting an instance healthy after
  it stopped accepting new connections.

## Observability

The gateway exposes `gateway_requests_total{backend,code,stream}`, `gateway_request_duration_seconds`,
`gateway_ttft_seconds`, `gateway_backend_running_requests`, `gateway_backend_healthy` and
`gateway_retries_total`. vLLM exposes its native `vllm:*` metrics (running/waiting requests, KV-cache
utilisation, TTFT histogram, token counters). The provisioned Grafana dashboard has 16 panels across both
layers, and `deploy/prometheus/alerts.yml` ships 7 alert rules (backend down, high TTFT, queue backlog,
KV cache nearly full, no healthy backends, 5xx ratio, retry storm).

## Verification

```bash
(cd gateway && gofmt -l . && go vet ./... && go test -race -count=1 ./...)
python -m pytest bench/tests -q
docker compose -f deploy/docker-compose.yml config -q
```

CI runs the same checks on every push.

## Limitations

- 7B and the HuggingFace baseline are not measured yet; 7B FP16 does not fit in 12 GB.
- The measured numbers come from a laptop GPU and vary 5–20% between runs; the reported values are medians.
- Backends cannot be hot-reloaded; a config change requires a restart (health state is re-probed in seconds).
- Rate limiting is a single process-wide token bucket, not per-tenant or per-model.
- No prefill/decode disaggregation or KV-cache-aware routing (that is the direction of vLLM Router).

## Documentation

- [Benchmark report](docs/benchmark-report.md) — method, control variables, measured results, capacity baseline
- [Design decisions](docs/decisions.md) — 20+ trade-offs with the alternatives that were rejected
- [Gateway notes](gateway/README.md) — config reference and endpoint list
- [Generated tables](docs/generated/benchmark-tables.md) — produced from `bench/results/*.json`

Detailed docs are primarily written in Chinese; each starts with an English summary.

---

# LLM 推理服务工程化实践

[English](#llm-serving-stack) | **中文**

> vLLM 推理服务部署 · 闭环并发压测 · Go 多实例网关 · Prometheus/Grafana 可观测

这个项目分两层，回答"一个大模型怎么在线上跑得稳"：

- **引擎层**：用 vLLM 部署 Qwen2.5，提供 OpenAI 兼容 API；用闭环保压量化 TTFT、TPOT、P99 和吞吐，
  并推导出容量基线。
- **接入层**：Go 网关放在多个 vLLM 实例前面，负责健康检查与软摘除、最少连接负载均衡、SSE 流式透传、
  只在首字节之前发生的重试，以及 Prometheus 指标。

仓库里的每个数字都由 `bench/` 下的脚本生成，不存在手工抄写。

## 实测结果

环境：RTX 5070 Ti Laptop（12 GB）+ Qwen2.5-1.5B-Instruct + vLLM 0.31.0。
每档 100 请求、`max_tokens=128`（这是上限，实际完成约 47–50 token——prompt 要求“一句话总结”，
模型会提前停止）、temperature 0，取 3 次独立运行的中位数。

| 并发 | QPS | 输出 tok/s | TTFT P50 | TTFT P99 | E2E P99 | TPOT P50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.34 | 114.6 | 20 ms | 24 ms | 0.459 s | 8.3 ms |
| 4 | 9.67 | 457.2 | 33 ms | 44 ms | 0.537 s | 8.1 ms |
| 8 | 18.17 | 836.7 | 43 ms | 84 ms | 0.528 s | 8.2 ms |
| 16 | 29.11 | 1363.1 | 66 ms | 80 ms | 0.628 s | 8.9 ms |
| 32 | 44.60 | 2110.0 | 108 ms | 128 ms | 0.807 s | 10.9 ms |

- 并发 1 → 32：输出吞吐 **114.6 → 2110 tok/s（18.4×）**，QPS 2.34 → 44.6。
- TTFT P99 不超过 128 ms，E2E P99 不超过 807 ms，500/500 请求成功。
- 32 并发正好是 2048 上下文下 KV Cache 的并发上限（31.24 条），再往上请求开始排队。

## 架构

```
                          ┌──────────────────────────────────────────┐
   client                 │            gateway (Go, :9000)           │
  (OpenAI SDK)            │                                          │
      │  /v1/chat/completions  ┌─────────────┐  ┌──────────────────┐ │
      ├──────────────────────▶ │  balancer   │  │ health checker    │ │
      │   stream: true         │ least_conn/ │  │ 10s 探测，2 次    │ │
      │                        │ round_robin │  │ 失败摘除/2 次恢复 │ │
      │                        └──────┬──────┘  └────────┬─────────┘ │
      │                               │                  │           │
      │                        ┌──────▼──────────────────▼─────────┐ │
      │                        │ proxy: 缓冲 body、首字节前重试、   │ │
      │                        │ SSE 逐块 flush、ctx 感知断连       │ │
      │                        └──────┬───────────────────┬─────────┘ │
      └───────────────────────────────┘                   │           │
                                                          │           │
                    ┌─────────────────────────────────────┴──────┐    │
                    │                                            │    │
              ┌─────▼──────┐                              ┌──────▼──┐ │
              │ vllm-1     │   PagedAttention             │ vllm-2  │ │
              │ :8000      │   continuous batching        │ :8000   │ │
              └─────┬──────┘                              └────┬────┘ │
                    └──────────────┬───────────────────────────┘      │
                                   │ /metrics                          │
                          ┌────────▼─────────┐   ┌──────────────────┐  │
                          │  Prometheus      │──▶│  Grafana         │  │
                          │  :9090           │   │  :3000           │  │
                          └──────────────────┘   └──────────────────┘  │
                                                                       │
                          └───────────────────────────────────────────┘
```

请求生命周期（引擎层）：`Client → OpenAI API → 调度器排队/准入 → 每个 iteration 组 batch
（continuous batching）→ prefill（计算密集）→ decode（访存密集，逐 token）→ KV block 按需分配/回收
（PagedAttention，16 token/块）→ 流式返回，结束释放 block`。

## 快速开始

前置：Go 1.26+（网关）、Python 3.10+（压测）；跑真实 vLLM 需要 Linux + NVIDIA GPU（vLLM 不支持
Windows 原生运行）+ Docker。

### 1. 无 GPU：mock 引擎 + 网关（约 2 分钟）

```bash
# 两个假 vLLM 后端
python bench/mock_vllm.py --port 18001 --token-ms 2 &
python bench/mock_vllm.py --port 18002 --token-ms 2 &

# 编译并启动网关
(cd gateway && go build -o gateway .)
(cd gateway && GATEWAY_BACKENDS="m1=http://127.0.0.1:18001,m2=http://127.0.0.1:18002" ./gateway -config config.yaml) &

# 压测（走网关）
python bench/bench_pro.py --base-url http://127.0.0.1:9000 --model mock-qwen \
  --concurrency 1,4,8 --num-requests 40 --max-tokens 32

# 查看后端状态（杀掉一个 mock，观察它被摘除）
curl -s localhost:9000/debug/backends
```

Windows PowerShell 等价写法（三个终端）：两个终端各跑一个 `python bench\mock_vllm.py --port 1800x`，
第三个终端执行 `$env:GATEWAY_BACKENDS="m1=http://127.0.0.1:18001,m2=http://127.0.0.1:18002"; cd gateway;
go build -o gateway.exe .; .\gateway.exe -config config.yaml`，最后另开终端跑压测。

### 2. 有 GPU：真实 vLLM + 监控栈

```bash
cd deploy
cp .env.example .env          # 默认 12 GB 卡 + Qwen2.5-1.5B
docker compose --profile gpu up -d
docker compose logs -f vllm   # 等模型加载完成
```

### 3. 压测真实引擎

```bash
# 客户端必须和 vLLM 同机（否则测的是网络，不是 GPU）
python bench/bench_pro.py --base-url http://127.0.0.1:8000 --model qwen-1.5b \
  --concurrency 1,4,8,16,32 --num-requests 100 --max-tokens 128 \
  --label local-qwen1.5b --same-host

# 重新生成报告表格
python bench/summarize.py --out docs/generated/benchmark-tables.md
```

Grafana：<http://localhost:3000>（admin/admin）→ 预置看板 **LLM Serving Stack**。

## 关键设计

- **选择与计数在同一把锁内**：读 `Running`、选最小、`Running+1` 必须原子完成，否则高并发下会退化成
  "全打一台"。
- **平局显式打破**：两个空闲实例比较永远相等，不破平局就会永远选第一个；端到端实测曾出现 83:41 的偏斜。
- **流式响应读到第一个 chunk 才提交响应头**：后端在 prefill 阶段挂掉时客户端还没收到字节，请求仍可安全重试。
- **首字节之后绝不重试**：客户端已收到 token，换实例重放会造成重复或语义断裂。
- **超时分层**：流式用首字节超时，非流式用总超时，`http.Client.Timeout` 保持 0，避免掐断正常的长生成。
- **健康探针不复用连接**：keep-alive 探针会在实例停止 accept 新连接后继续报健康。

## 可观测性

网关暴露 `gateway_requests_total{backend,code,stream}`、`gateway_request_duration_seconds`、
`gateway_ttft_seconds`、`gateway_backend_running_requests`、`gateway_backend_healthy`、
`gateway_retries_total`；vLLM 原生指标包括运行/等待请求数、KV Cache 使用率、TTFT 直方图和 token 计数。
Grafana 看板共 16 个图，跨引擎层与网关层；`deploy/prometheus/alerts.yml` 带 7 条告警规则。

## 验证

```bash
(cd gateway && gofmt -l . && go vet ./... && go test -race -count=1 ./...)
python -m pytest bench/tests -q
docker compose -f deploy/docker-compose.yml config -q
```

CI 在每次 push 时跑同样的检查。

## 已知限制

- 7B 与 HF 基线尚未测；7B FP16 放不进 12 GB 显存。
- 实测数字来自笔记本 GPU，run-to-run 波动 5–20%，报告使用中位数。
- 后端列表不支持热加载，改配置需重启（重启后几秒内重新探测健康状态）。
- 限流是进程级全局令牌桶，不是按租户/按模型。
- 未做 prefill/decode 分离与按 KV Cache 利用率路由。

## 文档

- [压测报告](docs/benchmark-report.md)：方法、控制变量、实测结果、容量基线
- [技术取舍](docs/decisions.md)：20+ 条"为什么不用 X"
- [网关说明](gateway/README.md)：配置项与接口列表
- [自动生成表格](docs/generated/benchmark-tables.md)：由 `bench/results/*.json` 生成

## License

[MIT](LICENSE)
