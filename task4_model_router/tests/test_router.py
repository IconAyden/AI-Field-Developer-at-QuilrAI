"""Router and end-to-end gateway tests.

Providers run in-process behind httpx's ASGI transport: each ``_attempt`` is an
ASGI call, so cancellation from ``anyio.fail_after`` propagates straight into
the hung provider's ``sleep``, which lets us assert the timeout really cancels
the in-flight request.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from task4_model_router import mock_providers
from task4_model_router.errors import GatewayError
from task4_model_router.rate_limiter import SlidingWindowRateLimiter
from task4_model_router.router import ModelEndpoint, ModelRouter
from task4_model_router.server import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Providers:
    """Primary + secondary mock providers reachable through one httpx client."""

    def __init__(self) -> None:
        self.primary = mock_providers.ProviderState("primary")
        self.secondary = mock_providers.ProviderState("secondary")
        apps = {
            "primary": mock_providers.create_provider_app(self.primary),
            "secondary": mock_providers.create_provider_app(self.secondary),
        }

        async def dispatch(scope, receive, send):
            host = dict(scope["headers"]).get(b"host", b"").decode()
            await apps[host.split(":")[0]](scope, receive, send)

        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=dispatch))

    def router(self, timeout_ms: int = 300) -> ModelRouter:
        return ModelRouter(
            ModelEndpoint("primary", "http://primary/v1/chat/completions", "primary-large", api_key="pk"),
            ModelEndpoint("secondary", "http://secondary/v1/chat/completions", "backup-small", api_key="sk"),
            timeout_ms=timeout_ms,
            http_client=self.client,
        )


@pytest.fixture
async def providers():
    p = Providers()
    async with p.client:
        yield p


PAYLOAD = {"messages": [{"role": "user", "content": "Hello there"}], "max_tokens": 50}


# --------------------------------------------------------------------------- #
# Router                                                                      #
# --------------------------------------------------------------------------- #
async def test_primary_ok_uses_primary(providers):
    r = await providers.router().complete(PAYLOAD, request_id="r1")
    assert r.provider == "primary" and r.model == "primary-large"
    assert r.body["choices"][0]["message"]["content"].startswith("[primary:primary-large]")
    assert [a.summary() for a in r.attempts] == ["primary:ok:200"]
    assert r.total_tokens == r.body["usage"]["total_tokens"]
    assert providers.secondary.calls == 0


async def test_primary_429_fails_over(providers):
    providers.primary.mode = "429"
    r = await providers.router().complete(PAYLOAD, request_id="r2")
    assert r.provider == "secondary"
    assert [a.summary() for a in r.attempts] == ["primary:rate_limited:429", "secondary:ok:200"]
    assert r.body["model"] == "backup-small", "secondary gets its own model id"


async def test_primary_500_fails_over(providers):
    providers.primary.mode = "500"
    r = await providers.router().complete(PAYLOAD, request_id="r3")
    assert r.provider == "secondary"
    assert r.attempts[0].outcome == "upstream_error"


async def test_primary_timeout_fails_over_and_cancels_in_flight(providers):
    providers.primary.mode = "hang"
    t0 = time.perf_counter()
    r = await providers.router(timeout_ms=300).complete(PAYLOAD, request_id="r4")
    elapsed = time.perf_counter() - t0
    assert r.provider == "secondary"
    assert r.attempts[0].outcome == "timeout"
    assert 0.25 < elapsed < 1.5, f"failover should happen right at the deadline, took {elapsed:.2f}s"
    assert providers.primary.cancelled == 1, "the hung request must be cancelled, not left running"
    assert providers.primary.in_flight == 0


async def test_default_timeout_is_3000ms(providers):
    providers.primary.mode = "hang"
    t0 = time.perf_counter()
    r = await providers.router(timeout_ms=3000).complete(PAYLOAD, request_id="r5")
    elapsed = time.perf_counter() - t0
    assert r.provider == "secondary"
    assert 2.9 < elapsed < 4.0


async def test_slow_but_within_deadline_stays_on_primary(providers):
    providers.primary.mode = "slow"
    providers.primary.slow_ms = 100
    r = await providers.router(timeout_ms=500).complete(PAYLOAD, request_id="r6")
    assert r.provider == "primary"


async def test_both_down_is_503_with_sanitized_error(providers):
    providers.primary.mode = "500"
    providers.secondary.mode = "hang"
    with pytest.raises(GatewayError) as info:
        await providers.router(timeout_ms=200).complete(PAYLOAD, request_id="r7")
    err = info.value
    assert err.status == 503 and err.code == "providers_unavailable" and err.retryable
    payload = json.dumps(err.to_payload("r7"))
    assert err.details["attempts"] == ["primary:upstream_error:500", "secondary:timeout"]
    for leak in ("Traceback", "10.0.3.14", "shard", "http://", "RuntimeError", "primary/v1"):
        assert leak not in payload, f"error payload leaks {leak!r}"


async def test_both_rate_limited_is_429(providers):
    providers.primary.mode = providers.secondary.mode = "429"
    with pytest.raises(GatewayError) as info:
        await providers.router().complete(PAYLOAD, request_id="r8")
    assert info.value.status == 429 and info.value.code == "providers_rate_limited"
    assert info.value.retry_after_seconds


async def test_client_error_does_not_fail_over(providers):
    providers.primary.mode = "400"
    with pytest.raises(GatewayError) as info:
        await providers.router().complete(PAYLOAD, request_id="r9")
    assert info.value.status == 400 and info.value.code == "invalid_request"
    assert providers.secondary.calls == 0


async def test_unreachable_primary_fails_over():
    primary_state = mock_providers.ProviderState("primary")
    secondary_state = mock_providers.ProviderState("secondary")

    async def dispatch(scope, receive, send):
        host = dict(scope["headers"]).get(b"host", b"").decode()
        if host.startswith("primary"):
            raise httpx.ConnectError("connection refused")
        await mock_providers.create_provider_app(secondary_state)(scope, receive, send)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=dispatch)) as client:
        router = ModelRouter(
            ModelEndpoint("primary", "http://primary/v1/chat/completions", "p"),
            ModelEndpoint("secondary", "http://secondary/v1/chat/completions", "s"),
            timeout_ms=300,
            http_client=client,
        )
        r = await router.complete(PAYLOAD, request_id="r10")
    assert r.provider == "secondary"
    assert r.attempts[0].outcome in ("unavailable", "upstream_error")
    del primary_state


# --------------------------------------------------------------------------- #
# Full gateway (limiter + router + HTTP)                                      #
# --------------------------------------------------------------------------- #
class FakeClock:
    def __init__(self) -> None:
        self.t = 5_000_000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
async def gateway(tmp_path: Path, providers):
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(tmp_path / "gw.db", limit_tokens=1_000, window_seconds=60, clock=clock)
    app = create_app(limiter, providers.router(timeout_ms=300))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        yield c, limiter, clock
    limiter.close()


AUTH = {"authorization": "Bearer tenant-key-1"}


async def test_gateway_happy_path_sets_headers_and_settles_actual_usage(gateway):
    c, limiter, _ = gateway
    r = await c.post("/v1/chat/completions", json=PAYLOAD, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "primary"
    assert r.headers["x-gateway-attempts"] == "primary:ok:200"
    assert r.headers["x-request-id"].startswith("req_")
    actual = r.json()["usage"]["total_tokens"]
    usage = (await c.get("/v1/usage", headers=AUTH)).json()
    assert usage["used_tokens"] == actual, "estimate must be replaced by the provider's real count"
    assert actual < 50 + 3  # far less than the max_tokens-based estimate


async def test_gateway_rate_limits_with_standard_payload(gateway):
    c, limiter, _ = gateway
    big = {"messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 500}  # estimate 600
    assert (await c.post("/v1/chat/completions", json=big, headers=AUTH)).status_code == 200
    # settled to the real (small) usage, so a second 600 fits; a third request pushes over.
    assert (await c.post("/v1/chat/completions", json=big, headers=AUTH)).status_code == 200
    limiter.try_reserve(tenant_for("tenant-key-1"), 700)  # fill it up behind the scenes
    r = await c.post("/v1/chat/completions", json=big, headers=AUTH)
    assert r.status_code == 429
    body = r.json()["error"]
    assert body["code"] == "rate_limited" and body["retryable"] is True
    assert body["request_id"].startswith("req_")
    assert body["details"]["limit_tokens"] == 1_000
    assert "retry_after_seconds" in body
    assert int(r.headers["retry-after"]) >= 1


def tenant_for(key: str) -> str:
    import hashlib

    return "t_" + hashlib.sha256(key.encode()).hexdigest()[:16]


async def test_gateway_tenants_are_keyed_by_api_key(gateway):
    c, limiter, _ = gateway
    limiter.try_reserve(tenant_for("tenant-key-1"), 1_000)
    assert (await c.post("/v1/chat/completions", json=PAYLOAD, headers=AUTH)).status_code == 429
    other = {"authorization": "Bearer tenant-key-2"}
    assert (await c.post("/v1/chat/completions", json=PAYLOAD, headers=other)).status_code == 200


async def test_gateway_refunds_reservation_when_all_providers_fail(gateway, providers):
    c, limiter, _ = gateway
    providers.primary.mode = providers.secondary.mode = "500"
    r = await c.post("/v1/chat/completions", json=PAYLOAD, headers=AUTH)
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "providers_unavailable"
    assert "Traceback" not in r.text and "10.0.3.14" not in r.text
    assert limiter.usage(tenant_for("tenant-key-1")) == 0


async def test_gateway_failover_visible_in_headers(gateway, providers):
    c, _, _ = gateway
    providers.primary.mode = "hang"
    r = await c.post("/v1/chat/completions", json=PAYLOAD, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "secondary"
    assert r.headers["x-gateway-attempts"] == "primary:timeout,secondary:ok:200"


async def test_gateway_auth_and_validation_errors(gateway):
    c, _, _ = gateway
    r = await c.post("/v1/chat/completions", json=PAYLOAD)
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthenticated"
    r = await c.post("/v1/chat/completions", content=b"{", headers={**AUTH, "content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_json"
    r = await c.post("/v1/chat/completions", json={"prompt": "no messages"}, headers=AUTH)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


async def test_gateway_unexpected_exception_is_sanitized(tmp_path, providers):
    class ExplodingLimiter(SlidingWindowRateLimiter):
        def try_reserve(self, tenant, tokens):
            raise RuntimeError("disk on fire at /var/lib/secret")

    limiter = ExplodingLimiter(tmp_path / "x.db")
    app = create_app(limiter, providers.router())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions", json=PAYLOAD, headers=AUTH)
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "disk on fire" not in r.text and "/var/lib" not in r.text
    limiter.close()
