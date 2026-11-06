"""Model routing with timeout-bounded primary attempt and automatic failover.

    complete(payload)
        ├─ primary   (deadline: timeout_ms)  ── 200 ──────────────► return
        │     429 / 5xx / timeout / connection error
        └─ secondary (deadline: timeout_ms)  ── 200 ──────────────► return
              429 / 5xx / timeout / connection error ─────────────► GatewayError 503

Rules
-----
* **Failover triggers**: HTTP 429, any 5xx, a timeout (no response within
  ``timeout_ms``), or a connection failure. These say "this endpoint cannot
  serve me right now", which is exactly what a backup is for.
* **No failover** on other 4xx (400, 401, 404, 413...). Those say "your request
  is wrong"; the backup would reject it too, and retrying a bad request burns
  the tenant's budget twice.
* **Timeout is a hard cancel.** ``anyio.fail_after`` cancels the in-flight
  ``send``; httpx then closes the connection, so a hung primary does not keep
  a socket and a task alive while the secondary is answering. The httpx client
  also gets a read timeout of the same length as belt-and-braces.
* **Every attempt is recorded** (provider name, outcome, elapsed ms) and
  returned to the caller for the ``X-Gateway-Attempts`` header and the log.
  The error returned to the client contains the outcome *codes* only, never
  the provider's URL, body, or exception text.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import anyio
import httpx

from .errors import GatewayError

log = logging.getLogger("model-router")

Outcome = Literal["ok", "rate_limited", "timeout", "upstream_error", "unavailable", "client_error"]


@dataclass(frozen=True, slots=True)
class ModelEndpoint:
    name: str  # e.g. "primary"
    url: str  # e.g. "https://api.provider.com/v1/chat/completions"
    model: str  # model id to request at this endpoint
    api_key: str | None = None


@dataclass(slots=True)
class Attempt:
    provider: str
    outcome: Outcome
    status: int | None
    elapsed_ms: float

    def summary(self) -> str:
        return f"{self.provider}:{self.outcome}" + (f":{self.status}" if self.status else "")


@dataclass(slots=True)
class RouteResult:
    status: int
    body: dict[str, Any]
    provider: str
    model: str
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def total_tokens(self) -> int | None:
        usage = self.body.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
            return usage["total_tokens"]
        return None


class ModelRouter:
    def __init__(
        self,
        primary: ModelEndpoint,
        secondary: ModelEndpoint,
        *,
        timeout_ms: int = 3000,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.timeout = timeout_ms / 1000.0
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 2.0))
        )
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------ #
    async def complete(self, payload: dict[str, Any], *, request_id: str) -> RouteResult:
        attempts: list[Attempt] = []
        for endpoint in (self.primary, self.secondary):
            result = await self._attempt(endpoint, payload, request_id, attempts)
            if result is not None:
                result.attempts = attempts
                return result
            last = attempts[-1]
            if last.outcome == "client_error":
                # The request itself is bad; do not try the backup.
                raise GatewayError(
                    status=last.status or 400,
                    code="invalid_request",
                    message="The model provider rejected the request as invalid",
                    retryable=False,
                    details={"attempts": [a.summary() for a in attempts]},
                )
            log.warning("request_id=%s provider=%s outcome=%s status=%s elapsed_ms=%.0f -> failing over",
                        request_id, last.provider, last.outcome, last.status, last.elapsed_ms)

        # Both endpoints failed.
        outcomes = {a.outcome for a in attempts}
        if outcomes == {"rate_limited"}:
            raise GatewayError(
                status=429,
                code="providers_rate_limited",
                message="All model providers are rate limiting requests; retry shortly",
                retryable=True,
                retry_after_seconds=5.0,
                details={"attempts": [a.summary() for a in attempts]},
            )
        raise GatewayError(
            status=503,
            code="providers_unavailable",
            message="No model provider could complete the request",
            retryable=True,
            retry_after_seconds=2.0,
            details={"attempts": [a.summary() for a in attempts]},
        )

    async def _attempt(
        self, endpoint: ModelEndpoint, payload: dict[str, Any], request_id: str, attempts: list[Attempt]
    ) -> RouteResult | None:
        body = {**payload, "model": endpoint.model}
        headers = {"content-type": "application/json", "x-request-id": request_id}
        if endpoint.api_key:
            headers["authorization"] = f"Bearer {endpoint.api_key}"
        started = time.perf_counter()

        def record(outcome: Outcome, status: int | None) -> None:
            attempts.append(Attempt(endpoint.name, outcome, status, (time.perf_counter() - started) * 1000))

        try:
            with anyio.fail_after(self.timeout):
                resp = await self._client.post(endpoint.url, json=body, headers=headers)
        except (TimeoutError, httpx.TimeoutException):
            record("timeout", None)
            return None
        except httpx.HTTPError as exc:
            log.warning("request_id=%s provider=%s transport error %s", request_id, endpoint.name, exc.__class__.__name__)
            record("unavailable", None)
            return None

        if resp.status_code == 429:
            record("rate_limited", 429)
            return None
        if resp.status_code >= 500:
            record("upstream_error", resp.status_code)
            return None
        if resp.status_code >= 400:
            record("client_error", resp.status_code)
            return None

        try:
            data = resp.json()
        except ValueError:
            log.warning("request_id=%s provider=%s returned non-JSON body", request_id, endpoint.name)
            record("upstream_error", resp.status_code)
            return None
        if not isinstance(data, dict):
            record("upstream_error", resp.status_code)
            return None

        record("ok", resp.status_code)
        return RouteResult(status=resp.status_code, body=data, provider=endpoint.name, model=endpoint.model)
