# AI Field Developer / Solutions Engineer Assessment

Four runnable Python implementations, one per task, with a test suite for each.

| Task | Folder | What it is | Tests |
|------|--------|------------|-------|
| 1 | [`task1_mcp_server/`](task1_mcp_server/README.md) | MCP server over stdio with strict Pydantic validation and standard JSON-RPC error codes | 54 |
| 2 | [`task2_mcp_gateway/`](task2_mcp_gateway/README.md) | MCP security gateway: HTTP/JSON-RPC reverse proxy with bearer roles and `admin_*` tool authorization | 38 |
| 3 | [`task3_llm_streaming_guardrail/`](task3_llm_streaming_guardrail/README.md) | LLM gateway that redacts PII inside a live SSE stream with bounded holdback | 49 |
| 4 | [`task4_model_router/`](task4_model_router/README.md) | Token-aware sliding-window rate limiter on SQLite plus timeout-bounded primary/secondary failover | 37 |

## Setup

Requires **Python 3.11 or newer** (developed and tested on 3.13). Nothing else needs to be installed system-wide; SQLite ships with Python.

```bash
# 1. unzip, then open a terminal in this folder
cd ai-solutions-engineer-assessment

# 2. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate              # Windows (PowerShell or cmd)
source .venv/bin/activate           # macOS / Linux

# 3. install pinned dependencies (includes pytest)
pip install -r requirements-dev.txt
```

`requirements.txt` holds only the runtime dependencies (the official `mcp` SDK 2.x, `pydantic`, `starlette`, `uvicorn`, `httpx`); `requirements-dev.txt` adds `pytest`. Both are pinned to the exact versions the tests were run with. If you use `uv`, `uv sync` does the same from `pyproject.toml` / `uv.lock`.

## Run the tests

```bash
python -m pytest            # all four tasks, ~25 s (Task 4 includes one real 3000 ms timeout test)
python -m pytest task1_mcp_server
python -m pytest task2_mcp_gateway
python -m pytest task3_llm_streaming_guardrail
python -m pytest task4_model_router
```

Every command below is run from this directory; the packages are imported as `task1_mcp_server.server` and so on.

## Run the services

```bash
# Task 1: MCP server on stdio (speak JSON-RPC on stdin; logs appear on stderr)
python -m task1_mcp_server.server

# Task 2: downstream mock on :9100, gateway on :9000
python -m task2_mcp_gateway.downstream_mock
python -m task2_mcp_gateway.gateway

# Task 3: mock LLM provider on :9200, redacting gateway on :9300
python -m task3_llm_streaming_guardrail.mock_provider
python -m task3_llm_streaming_guardrail.gateway

# Task 4: mock providers on :9401/:9402, routing gateway on :9400 (SQLite at .data/gateway.db)
python -m task4_model_router.mock_providers
python -m task4_model_router.server
```

Each task README has `curl` examples and a design section covering the trade-offs the evaluation criteria ask about.

## Conventions shared across tasks

- **No framework magic.** Starlette + httpx only; every proxy decision is visible in one request handler.
- **Errors never leak internals.** Upstream bodies, hostnames and tracebacks go to the server log keyed by a request id; clients get a stable code and a plain message.
- **Tests exercise the wire**, not just the functions: raw stdio subprocess for Task 1, real TCP sockets for the streaming assertions in Task 3, in-process ASGI everywhere speed matters.
- **Mocks are part of the deliverable.** Each gateway ships with the downstream it was tested against, with switchable failure modes.
