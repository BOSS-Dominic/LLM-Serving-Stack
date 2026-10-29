# Makefile — convenience targets; every one is a thin wrapper over a command
# documented in README.md.

GO ?= go
PYTHON ?= python
COMPOSE ?= docker compose -f deploy/docker-compose.yml

# Used by `make bench-gpu` / `make report`.
BASE_URL ?= http://127.0.0.1:8000
MODEL ?= qwen-7b
LABEL ?= local-gpu

.PHONY: help fmt vet test build bench-test bench-smoke bench-gpu report compose-config up up-mock down logs

help:
	@echo "make fmt             gofmt the gateway"
	@echo "make vet             go vet ./..."
	@echo "make test            go test -race"
	@echo "make build           build gateway/gateway"
	@echo "make bench-test      pytest bench/tests"
	@echo "make bench-smoke     run the mock engine and benchmark it end-to-end"
	@echo "make bench-gpu       benchmark a real vLLM (BASE_URL/MODEL/LABEL)"
	@echo "make report          regenerate docs/generated/benchmark-tables.md"
	@echo "make compose-config  validate deploy/docker-compose.yml"
	@echo "make up              start the stack with the GPU profile"
	@echo "make up-mock         start the stack with the mock engine (no GPU)"
	@echo "make down            stop everything and delete volumes"

fmt:
	cd gateway && gofmt -w .

vet:
	cd gateway && $(GO) vet ./...

test:
	cd gateway && $(GO) test -race -count=1 ./...

build:
	cd gateway && $(GO) build -o gateway .

bench-test:
	$(PYTHON) -m pytest bench/tests -q

bench-smoke:
	$(PYTHON) bench/mock_vllm.py --port 18000 --prefill-ms 5 --token-ms 1 & \
	MOCK_PID=$$!; \
	sleep 1; \
	$(PYTHON) bench/bench_pro.py --base-url http://127.0.0.1:18000 --model mock-model \
		--concurrency 1,4 --num-requests 20 --max-tokens 32 --warmup 2 \
		--output-dir bench/results; \
	kill $$MOCK_PID

bench-gpu:
	$(PYTHON) bench/bench_pro.py --base-url $(BASE_URL) --model $(MODEL) \
		--concurrency 1,4,16,32 --num-requests 100 --max-tokens 128 \
		--label $(LABEL) --same-host

report:
	$(PYTHON) bench/summarize.py --out docs/generated/benchmark-tables.md

compose-config:
	$(COMPOSE) config -q

up:
	$(COMPOSE) --profile gpu up -d --build

up-mock:
	BACKENDS=mock-1=http://mock-vllm:18000 $(COMPOSE) --profile mock up -d --build

down:
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f --tail=100
