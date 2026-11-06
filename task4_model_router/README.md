# Task 4: Rate-limiting and model-fallback router

A resilient routing module for an LLM gateway: a token-aware sliding-window rate limiter per tenant API key, persisted in on-disk SQLite, in front of a primary → secondary model failover with a hard 3000 ms deadline and sanitized error responses.

```
client --Bearer <tenant key>--> [ gateway :9400 ] --> primary :9401   (3000 ms deadline)
                                       └─ on 429 / 5xx / timeout / conn error --> secondary :9402
```

## Run

```bash
python -m task4_model_router.mock_providers     # terminal 1: primary :9401 and secondary :9402
python -m task4_model_router.server             # terminal 2: gateway :9400, SQLite at .data/gateway.db

curl -s -D - localhost:9400/v1/chat/completions -H 'Authorization: Bearer key-A' -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":20}'
# x-gateway-provider: primary
# x-gateway-attempts: primary:ok:200

curl -s localhost:9401/_control -d '{"mode":"hang"}'        # make the primary hang
curl -s -D - localhost:9400/v1/chat/completions -H 'Authorization: Bearer key-A' -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":20}'
# ~3.2 s later:
# x-gateway-provider: secondary
# x-gateway-attempts: primary:timeout,secondary:ok:200

curl -s localhost:9400/v1/usage -H 'Authorization: Bearer key-A'
# {"tenant":"t_b7930bd94b2ed34d","used_tokens":19,"limit_tokens":50000,"window_seconds":60.0}

curl -s localhost:9400/v1/chat/completions -H 'Authorization: Bearer key-A' -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":60000}'
# 429 {"error":{"code":"rate_limited","message":"Token budget for this API key is exhausted for the current window",
#      "request_id":"req_...","retryable":true,"retry_after_seconds":60.0,
#      "details":{"limit_tokens":50000,"window_seconds":60.0,"used_tokens":19,"requested_tokens":60002}}}
```

Mock provider modes (`POST /_control {"mode": ...}`): `ok`, `429`, `500` (returns a fake stack trace that must never reach the client), `400`, `hang` (60 s), `slow` (`slow_ms`). `GET /_stats` shows calls, in-flight and cancelled counts.

Environment: `LIMIT_TOKENS` (50000), `WINDOW_SECONDS` (60), `PROVIDER_TIMEOUT_MS` (3000), `PRIMARY_URL` / `PRIMARY_MODEL` / `PRIMARY_API_KEY`, the same for `SECONDARY_*`, `GATEWAY_DB`, `PORT`.

## Test

```bash
python -m pytest task4_model_router -q      # ~6 s; one test uses the real 3000 ms deadline
```

- `tests/test_rate_limiter.py`: an injectable clock drives window sliding, partial expiry, eviction, `retry_after` arithmetic, reserve/settle/release, persistence across instances, and two concurrency tests (50 threads on one connection; 40 threads over four separate connections to the same file) asserting that exactly the budgeted number of reservations win.
- `tests/test_router.py`: failover on 429, 5xx, timeout and connection error; no failover on 4xx; both-down → 503 with a payload proven free of the mock's fake traceback and IPs; **the timeout test asserts the hung primary request was actually cancelled** (the mock counts cancellations); full gateway tests for headers, settle-to-actual usage, refunds on failure, per-key tenancy and sanitized 500s.

## Design

### Sliding-window limiter (`rate_limiter.py`)

**Model.** One row per request (`tenant, ts, tokens`). Usage at time *t* is `SUM(tokens) WHERE ts > t - window`. This is a sliding *log*, which gives exact semantics (a burst at 12:00:59 still counts at 12:01:58), unlike fixed buckets that reset on the minute.

**Reserve then settle.** Token usage is only known after the model answers, but the limit must be enforced before sending. So `try_reserve` inserts the *estimate* (prompt chars / 4 + `max_tokens`), `settle` overwrites it with the provider's `usage.total_tokens`, and `release` deletes it if no provider consumed anything. Over-estimating is the safe direction: it can only delay a tenant briefly until settle.

**Atomicity.** `try_reserve` runs inside `BEGIN IMMEDIATE`, taking SQLite's write lock *before* the SELECT, so two concurrent reservations cannot both read the same usage and both succeed. That holds across threads and across processes sharing the file, which is why SQLite (WAL mode, `busy_timeout`) was chosen over an in-memory structure: the limit survives restarts and is shared by every gateway worker on the host.

**Eviction.** Rows older than the window are deleted for the tenant on every reservation (an indexed range delete), and `evict_expired()` sweeps all tenants. Correctness never depends on eviction having run, because every read is bounded by `ts > now - window`.

**Retry-After.** On denial, the limiter walks the tenant's rows oldest-first and reports when enough tokens will have slid out for the request to fit, so the client gets a precise `Retry-After` instead of a guess.

**Event loop.** SQLite calls are synchronous; the server runs them in `anyio.to_thread` so disk I/O never blocks request handling.

### Failover router (`router.py`)

- The primary attempt runs under `anyio.fail_after(timeout)`. Expiry **cancels** the in-flight `send`, closing the socket; the hung request does not linger while the secondary answers. The httpx client carries the same read timeout as a second line of defence.
- Failover triggers: HTTP 429, any 5xx, timeout, connection failure, non-JSON body. These say "this endpoint cannot serve me now".
- No failover on other 4xx (400, 401, 404, 413): the request itself is wrong, the backup would reject it too, and retrying burns the tenant's budget twice.
- Each endpoint has its own model id (`primary-large` → `backup-small`), set per attempt.
- Every attempt is recorded (provider, outcome, status, elapsed ms) and surfaced in `X-Gateway-Attempts` and the log.

### Standardized errors (`errors.py`)

Every error has one shape:

```json
{"error": {"code": "providers_unavailable", "message": "No model provider could complete the request",
           "request_id": "req_6c1d...", "retryable": true, "retry_after_seconds": 2.0,
           "details": {"attempts": ["primary:upstream_error:500", "secondary:timeout"]}}}
```

| Situation | HTTP | `code` |
|-----------|------|--------|
| No / bad bearer key | 401 | `unauthenticated` |
| Bad JSON / missing `messages` / `stream: true` | 400 | `invalid_json`, `invalid_request`, `unsupported` |
| Tenant window exhausted | 429 | `rate_limited` (+ `Retry-After` header) |
| Provider rejected the request as invalid | 4xx | `invalid_request` |
| Both providers rate-limited | 429 | `providers_rate_limited` |
| Both providers failed / timed out | 503 | `providers_unavailable` |
| Unexpected exception | 500 | `internal_error` |

What never appears in a response: upstream bodies, hostnames, URLs, exception text, stack traces. Those go to the server log under the same `request_id`. The mock's `500` mode returns a fake traceback with an internal IP precisely so the tests can prove none of it leaks.

### Trade-offs

- The reservation is based on `max_tokens`; a client that always sets a large ceiling will be throttled early until settle corrects it. A tokenizer-based estimate would be more precise at the cost of a dependency.
- Failover is sequential (primary deadline, then secondary deadline), so worst-case latency is two deadlines. Hedged requests (start the secondary at, say, 1500 ms) would cut tail latency at the cost of extra provider spend; the router is structured so that is a local change in `complete()`.
- Streaming is rejected on this route; the streaming guardrail lives in Task 3 and composing the two is a matter of wiring the router's endpoint selection in front of Task 3's stream proxy.

## Layout

```
task4_model_router/
  rate_limiter.py    SQLite sliding-window limiter, reserve/settle/release, token estimate
  router.py          Primary/secondary attempts with cancel-on-timeout
  errors.py          GatewayError and the standard payload
  server.py          HTTP front: tenant auth, limiter, router, exception handlers
  mock_providers.py  Two providers with switchable failure modes
  tests/
```
