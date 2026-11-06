# Task 1: MCP server with strict validation and clean stdio

An MCP server over the stdio transport, built with the official Python SDK (`mcp` 2.x), exposing:

| Tool | Input | Output |
|------|-------|--------|
| `get_customer_record` | `customer_id` matching `^CUST-\d{5}$` | customer profile + balance (`structuredContent` and text) |
| `trigger_refund` | `customer_id` (same pattern), `amount` positive number ≤ 1,000,000 with ≤ 2 decimals, `reason` ≥ 10 chars after trimming | refund receipt |

## Run

```bash
python -m task1_mcp_server.server          # from the repository root
MCP_LOG_LEVEL=DEBUG python -m task1_mcp_server.server
```

Or point an MCP client at it. For Claude Desktop / Inspector style configs:

```json
{
  "mcpServers": {
    "customer-ops": {
      "command": "python",
      "args": ["-m", "task1_mcp_server.server"],
      "cwd": "C:/Projects/New folder"
    }
  }
}
```

Hand-drive it from a shell (one JSON-RPC message per line on stdin):

```bash
printf '%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"cli","version":"0"}}}' \
 '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_customer_record","arguments":{"customer_id":"cust-42"}}}' \
 | python -m task1_mcp_server.server 2>/dev/null
```

The last line of output is the validation failure as a protocol error:

```json
{"jsonrpc":"2.0","id":2,"error":{"code":-32602,"message":"Invalid arguments for tool 'get_customer_record': customer_id: String should match pattern '^CUST-\\d{5}$'","data":{"tool":"get_customer_record","issues":[{"field":"customer_id","message":"String should match pattern '^CUST-\\d{5}$'","type":"string_pattern_mismatch"}]}}}
```

## Test

```bash
python -m pytest task1_mcp_server -q
```

- `tests/test_stdio_isolation.py` spawns the server as a **real subprocess** with `MCP_LOG_LEVEL=DEBUG`, feeds raw JSON-RPC lines (including one that is not JSON), and asserts that every stdout line parses as a JSON-RPC message and that the logs landed on stderr.
- `tests/test_validation.py` connects the official SDK `Client` in-process and runs the full edge-case matrix (wrong types, extra fields, whitespace-only reason, 3-decimal amounts, `NaN`, `inf`, booleans, unknown tool...).

## Design

### stdout is the wire

- All diagnostics go through `logging` with an explicit `stream=sys.stderr` handler (`force=True` evicts any handler a library might have attached to stdout).
- While serving, the SDK's `stdio_server` re-points file descriptor 1 at stderr, so even a stray `print()` in a handler cannot corrupt the stream.
- The test suite verifies the invariant at the process boundary, not through the SDK client (which would silently discard a bad line).

### Error mapping

| Where it fails | Client sees |
|----------------|-------------|
| Not JSON | `-32700 Parse error`, `id: null` |
| Bad JSON-RPC envelope | `-32600 Invalid Request` |
| Unknown method | `-32601 Method not found` |
| Unknown tool name | `-32602 Invalid params` with `known_tools` in `data` |
| Arguments fail the schema | `-32602 Invalid params` with field-level `issues` in `data` |
| Tool ran, business rule failed (unknown customer, refund > balance) | `CallToolResult` with `isError: true` |
| Unexpected exception | `-32603 Internal error`, message sanitized |

The split between the last three rows matters. A malformed request is a protocol error: the caller must fix the call. A refund that exceeds the balance is a *successful* protocol exchange whose *outcome* is a failure the model should read and reason about, which is what the MCP spec reserves `isError` for.

The SDK's high-level `MCPServer` class reports schema failures as `isError` results. This server is built on the low-level `Server` so that validation failures come back with the standard `-32602` code the task asks for (`server.py`, `on_call_tool`).

### Validation (`schemas.py`)

- Pydantic models with `extra="forbid"` and `strict=True`: unknown fields are rejected, and `"amount": "25"` (a string) is rejected rather than coerced.
- `amount` is checked for positivity, finiteness, a ceiling, and at most two decimals (via `Decimal(str(v))`, so `0.1` is not misread as `0.1000000000000000055`).
- `reason` is trimmed before the length check, so ten spaces do not pass.
- Error `data` carries the field path and rule but **not** the rejected value, so PII the caller sent is not echoed back or logged downstream.
- Input and output schemas are published through `tools/list` (`inputSchema` with `additionalProperties: false`, `outputSchema`), with `readOnlyHint` / `destructiveHint` annotations.

### Transport guard (`transport.py`)

A thin layer between the SDK's stdio streams and the server loop adds two behaviours the stock transport lacks:

1. A line that is not valid JSON gets a `-32700` response with `id: null`, as JSON-RPC 2.0 requires. The SDK only logs it at DEBUG.
2. When the client closes stdin while tool calls are still running, the guard holds the server loop open until every forwarded request has been answered (bounded by 5 s). Without this, the last responses before EOF are dropped. The raw-subprocess test covers exactly this case.

## Layout

```
task1_mcp_server/
  server.py      Server wiring, tool registry, error mapping
  schemas.py     Pydantic input/output models
  store.py       In-memory customer store with BusinessError
  transport.py   Parse-error responses and drain-on-EOF
  tests/
```
