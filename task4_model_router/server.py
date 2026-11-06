"""HTTP front for the rate-limiting, failover model router.

    POST /v1/chat/completions   Authorization: Bearer <tenant api key>
    GET  /v1/usage              current window usage for the caller
    GET  /healthz

Request lifecycle::

    authenticate tenant
      └─ estimate tokens  ─► limiter.try_reserve  ──denied──► 429 (standard payload, Retry-After)
             └─ router.complete (primary → secondary, 3000 ms each)
                   ├─ ok     ─► limiter.settle(actual usage) ─► 200 + X-Gateway-Provider
                   └─ failed ─► limiter.release()            ─► 4xx/5xx standard payload

SQLite calls are synchronous and run in a worker thread (``anyio.to_thread``)
so the event loop never blocks on disk I/O. Unexpected exceptions are caught
by a global handler that returns ``internal_error`` with a request id and logs
the traceback server-side only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .errors import GatewayError, internal_error_payload
from .rate_limiter import SlidingWindowRateLimiter, estimate_tokens
from .router import ModelEndpoint, ModelRouter

log = logging.getLogger("llm-gateway")

MAX_BODY_BYTES = 2 * 1024 * 1024


def tenant_from_request(request: Request) -> str:
    """The API key *is* the tenant identity. Stored hashed so the DB never holds raw keys."""
    auth = request.headers.get("authorization", "")
    scheme, _, key = auth.partition(" ")
    key = key.strip()
    if scheme.lower() != "bearer" or not key:
        raise GatewayError(401, "unauthenticated", "Provide an API key as 'Authorization: Bearer <key>'")
    return "t_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def create_app(limiter: SlidingWindowRateLimiter, router: ModelRouter) -> Starlette:
    async def chat_completions(request: Request) -> Response:
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        tenant = tenant_from_request(request)

        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            raise GatewayError(413, "payload_too_large", "Request body exceeds 2 MiB")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise GatewayError(400, "invalid_json", "Request body must be valid JSON") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            raise GatewayError(400, "invalid_request", "Body must be an object with a 'messages' array")
        if payload.get("stream"):
            raise GatewayError(400, "unsupported", "Streaming is not supported on this route")

        estimate = estimate_tokens(payload)
        decision = await anyio.to_thread.run_sync(limiter.try_reserve, tenant, estimate)
        if not decision.allowed:
            log.info("request_id=%s tenant=%s rate_limited used=%d requested=%d", request_id, tenant,
                     decision.used_before, estimate)
            raise GatewayError(
                429,
                "rate_limited",
                "Token budget for this API key is exhausted for the current window",
                retryable=True,
                retry_after_seconds=decision.retry_after_seconds,
                details={
                    "limit_tokens": decision.limit,
                    "window_seconds": decision.window_seconds,
                    "used_tokens": decision.used_before,
                    "requested_tokens": estimate,
                },
            )
        assert decision.reservation_id is not None

        try:
            result = await router.complete(payload, request_id=request_id)
        except BaseException:
            # Nothing was consumed upstream: refund the reservation. (BaseException so a
            # client disconnect / cancellation also refunds.)
            await anyio.to_thread.run_sync(limiter.release, decision.reservation_id)
            raise

        actual = result.total_tokens
        if actual is not None:
            await anyio.to_thread.run_sync(limiter.settle, decision.reservation_id, actual)
        log.info("request_id=%s tenant=%s provider=%s attempts=%s estimate=%d actual=%s", request_id, tenant,
                 result.provider, [a.summary() for a in result.attempts], estimate, actual)
        return JSONResponse(
            result.body,
            status_code=200,
            headers={
                "x-request-id": request_id,
                "x-gateway-provider": result.provider,
                "x-gateway-attempts": ",".join(a.summary() for a in result.attempts),
                "x-ratelimit-limit-tokens": str(decision.limit),
                "x-ratelimit-remaining-tokens": str(max(0, decision.limit - decision.used_before - (actual or estimate))),
            },
        )

    async def usage(request: Request) -> Response:
        tenant = tenant_from_request(request)
        used = await anyio.to_thread.run_sync(limiter.usage, tenant)
        return JSONResponse({"tenant": tenant, "used_tokens": used, "limit_tokens": limiter.limit, "window_seconds": limiter.window})

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def gateway_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, GatewayError)
        request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:12]}"
        headers = {"x-request-id": request_id}
        if exc.retry_after_seconds is not None:
            headers["retry-after"] = str(max(1, int(exc.retry_after_seconds + 0.999)))
        return JSONResponse(exc.to_payload(request_id), status_code=exc.status, headers=headers)

    async def unexpected_error(request: Request, exc: Exception) -> Response:
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        log.exception("request_id=%s unhandled error", request_id)
        return JSONResponse(internal_error_payload(request_id), status_code=500, headers={"x-request-id": request_id})

    @asynccontextmanager
    async def lifespan(_: Starlette):
        try:
            yield
        finally:
            await router.aclose()
            limiter.close()

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/usage", usage, methods=["GET"]),
            Route("/healthz", health, methods=["GET"]),
        ],
        exception_handlers={GatewayError: gateway_error, Exception: unexpected_error},
        lifespan=lifespan,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    db_path = Path(os.environ.get("GATEWAY_DB", ".data/gateway.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    limiter = SlidingWindowRateLimiter(
        db_path,
        limit_tokens=int(os.environ.get("LIMIT_TOKENS", "50000")),
        window_seconds=float(os.environ.get("WINDOW_SECONDS", "60")),
    )
    router = ModelRouter(
        ModelEndpoint(
            name="primary",
            url=os.environ.get("PRIMARY_URL", "http://127.0.0.1:9401/v1/chat/completions"),
            model=os.environ.get("PRIMARY_MODEL", "primary-large"),
            api_key=os.environ.get("PRIMARY_API_KEY"),
        ),
        ModelEndpoint(
            name="secondary",
            url=os.environ.get("SECONDARY_URL", "http://127.0.0.1:9402/v1/chat/completions"),
            model=os.environ.get("SECONDARY_MODEL", "backup-small"),
            api_key=os.environ.get("SECONDARY_API_KEY"),
        ),
        timeout_ms=int(os.environ.get("PROVIDER_TIMEOUT_MS", "3000")),
    )
    uvicorn.run(create_app(limiter, router), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "9400")), log_level="info")


if __name__ == "__main__":
    main()
