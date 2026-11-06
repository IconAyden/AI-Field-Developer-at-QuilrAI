# Task 2: MCP security gateway (tool filtering and auth)

A reverse proxy that sits between an AI agent and a downstream MCP server, authenticates the caller from a bearer token, and enforces method-level authorization on the JSON-RPC payload before anything is forwarded.

```
agent client --Bearer <token>--> [ gateway :9000 ] --> downstream MCP server :9100
```

## Run

```bash
python -m task2_mcp_gateway.downstream_mock        # terminal 1, :9100, no auth of its own
python -m task2_mcp_gateway.gateway                # terminal 2, :9000
```

Demo tokens: `admin-token-123` (alice, admin) and `viewer-token-456` (bob, viewer). Override with `GATEWAY_TOKENS='{"<token>":{"subject":"x","role":"admin"}}'`.

```bash
# viewer calls an admin_ tool: intercepted, downstream never called
curl -s localhost:9000/mcp -H 'Authorization: Bearer viewer-token-456' -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{}}}'
# -> 403 {"jsonrpc":"2.0","id":1,"error":{"code":-32001,"message":"Unauthorized Tool Call","data":{"tool":"admin_reset_key","required_role":"admin","role":"viewer"}}}

# admin: forwarded
curl -s localhost:9000/mcp -H 'Authorization: Bearer admin-token-123' -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"tenant":"acme"}}}'
# -> 200 {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"API key rotated for tenant acme"}]}}

# tools/list: forwarded transparently for every role
curl -s localhost:9000/mcp -H 'Authorization: Bearer viewer-token-456' -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/list"}'
```

Environment: `DOWNSTREAM_URL` (default `http://127.0.0.1:9100/mcp`), `DOWNSTREAM_TOKEN` (service token to present downstream), `DOWNSTREAM_TIMEOUT` (seconds, default 10), `FILTER_TOOL_LIST=1` (opt-in, see below), `PORT`.

## Test

```bash
python -m pytest task2_mcp_gateway -q
```

Policy unit tests, authentication cases, every wire-format edge case (bad JSON, bad envelopes, batches, notifications, oversized bodies), downstream failure handling, and one end-to-end run over real TCP with uvicorn.

## Design

### Request pipeline (`gateway.py`)

1. **Authenticate.** `Authorization: Bearer <token>` is resolved to a principal (subject, role). Token comparison uses `hmac.compare_digest` across the whole store, so lookup time does not reveal how many leading bytes matched. Failure: HTTP 401, JSON-RPC `-32002 Authentication required`.
2. **Parse.** Body must be a JSON-RPC 2.0 message or batch. Failures: `-32700` (not JSON), `-32600` (bad envelope), both HTTP 400; 413 above 1 MiB.
3. **Authorize** each message with the pure policy function `policy.decide(message, principal)`:
   - `tools/list` → forward.
   - `tools/call` → `params.name` must be a non-empty string (`-32602` otherwise). If it starts with `admin_` and the role is not `admin` → **`-32001 Unauthorized Tool Call`**, answered by the gateway, HTTP 403. The downstream is never contacted.
   - Everything else (`initialize`, `ping`, `notifications/*`, ...) → forward.
4. **Forward** with httpx. Hop-by-hop headers are stripped, the client's token is **not** passed through (zero trust: the downstream gets the gateway's own service token plus `X-Gateway-Subject` / `X-Gateway-Role` / `X-Forwarded-For` for its audit log). Status, body and safe headers come back unchanged.
5. **Downstream failures** become `-32003 Upstream unavailable`: 502 for connection errors, 504 when the gateway's own deadline (`anyio.fail_after`) expires. No hostnames or exception text reach the client.

### Batches

A JSON-RPC batch may mix allowed and denied calls. The gateway keeps a slot per incoming message, answers the denied ones locally, forwards only the allowed subset, and splices the downstream's responses back into the client's original order by `id`. A batch that is entirely denied never touches the network. Notifications (no `id`) produce no response; a batch of only notifications returns 202.

### Error codes

| Code | Meaning | HTTP |
|------|---------|------|
| `-32700` | Parse error | 400 |
| `-32600` | Invalid Request | 400 / 413 |
| `-32602` | Invalid params (malformed `tools/call`) | 400 |
| `-32001` | Unauthorized Tool Call | 403 |
| `-32002` | Authentication required | 401 |
| `-32003` | Upstream unavailable / timed out | 502 / 504 |

Gateway-specific codes sit in the JSON-RPC reserved server-error range (-32000 to -32099), so they can never collide with a downstream server's application codes.

### Optional response filtering

The task asks for `tools/list` to be forwarded transparently, which is the default. `FILTER_TOOL_LIST=1` additionally strips `admin_*` entries from the listing for non-admins, so a viewer's agent does not plan a call it cannot make. It is an opt-in because hiding tools changes what the model believes exists, which is a product decision rather than a security requirement: the `tools/call` check already enforces the boundary either way.

### Why no framework middleware

The authorization decision depends on the *body* of the request, not just its headers, and on each element of a batch separately. Doing it inside one handler keeps the parse → decide → forward → splice flow readable in a single place; the policy itself is a pure function with its own unit tests.

## Layout

```
task2_mcp_gateway/
  gateway.py          HTTP handler: auth, parse, authorize, forward, splice
  policy.py           Pure decision function and error-code constants
  auth.py             Bearer token store and principal lookup
  downstream_mock.py  Unprotected JSON-RPC MCP server used as the upstream
  tests/
```
