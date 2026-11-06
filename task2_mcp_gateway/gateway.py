"""MCP security gateway: an HTTP/JSON-RPC reverse proxy with role-based tool authorization.

    agent client  --Bearer token-->  [ gateway :9000 ]  --->  downstream MCP server :9100

Request pipeline (``POST /mcp``):

1. **Authenticate** the ``Authorization: Bearer <token>`` header to a principal
   (subject + role). Failure: HTTP 401, JSON-RPC ``-32002``.
2. **Parse** the body as JSON-RPC 2.0: a single message or a batch. Failure:
   HTTP 400, ``-32700`` (not JSON) or ``-32600`` (bad envelope).
3. **Authorize** each message with the policy (``policy.decide``).
   ``tools/call`` on an ``admin_*`` tool by a non-admin: ``-32001 Unauthorized
   Tool Call``, answered here, never forwarded.
4. **Forward** what remains to the downstream and splice the responses back
   together in the client's order. The client token is stripped; the downstream
   sees a gateway service token plus ``X-Gateway-Subject`` / ``X-Gateway-Role``
   for its own audit trail.
5. Downstream unreachable or slow: HTTP 502, ``-32003`` with no internal detail.

Run from the repository root::

    python -m task2_mcp_gateway.downstream_mock      # terminal 1
    python -m task2_mcp_gateway.gateway              # terminal 2
    curl -s localhost:9000/mcp -H 'Authorization: Bearer viewer-token-456' \
         -H 'Content-Type: application/json' \
         -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{}}}'
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .auth import AuthError, Principal, TokenStore, authenticate
from .policy import (
    AUTHENTICATION_REQUIRED,
    INVALID_REQUEST,
    PARSE_ERROR,
    UPSTREAM_UNAVAILABLE,
    JsonRpcError,
    Reject,
    decide,
    filter_tool_listing,
    validate_envelope,
)

log = logging.getLogger("mcp-gateway")
audit = logging.getLogger("mcp-gateway.audit")

# Headers that are hop-by-hop or that the gateway owns; never copied in either direction.
_STRIP_REQUEST_HEADERS = {
    "authorization",
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
    "keep-alive",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
}
_STRIP_RESPONSE_HEADERS = {"content-length", "connection", "transfer-encoding", "keep-alive", "server", "date"}

MAX_BODY_BYTES = 1 * 1024 * 1024


class GatewayConfig:
    def __init__(
        self,
        *,
        downstream_url: str,
        token_store: TokenStore,
        downstream_token: str | None = None,
        timeout_seconds: float = 10.0,
        filter_tool_list: bool = False,
    ) -> None:
        self.downstream_url = downstream_url
        self.token_store = token_store
        self.downstream_token = downstream_token
        self.timeout_seconds = timeout_seconds
        self.filter_tool_list = filter_tool_list


def _rpc_error_response(err: JsonRpcError, *, id_: Any = None, status: int) -> JSONResponse:
    return JSONResponse(err.to_wire(id_), status_code=status)


def _forward_headers(request: Request, principal: Principal, cfg: GatewayConfig) -> dict[str, str]:
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _STRIP_REQUEST_HEADERS}
    headers.setdefault("content-type", "application/json")
    headers.setdefault("accept", "application/json")
    headers["x-gateway-subject"] = principal.subject
    headers["x-gateway-role"] = principal.role
    client_ip = request.client.host if request.client else "unknown"
    prior = request.headers.get("x-forwarded-for")
    headers["x-forwarded-for"] = f"{prior}, {client_ip}" if prior else client_ip
    if cfg.downstream_token:
        headers["authorization"] = f"Bearer {cfg.downstream_token}"
    return headers


def create_app(cfg: GatewayConfig, *, http_client: httpx.AsyncClient | None = None) -> Starlette:
    """Build the ASGI app. ``http_client`` is injectable so tests can point it at an in-process downstream."""

    client = http_client or httpx.AsyncClient(timeout=cfg.timeout_seconds)

    async def forward(request: Request, principal: Principal, payload: Any) -> httpx.Response:
        return await client.post(
            cfg.downstream_url,
            content=json.dumps(payload).encode(),
            headers=_forward_headers(request, principal, cfg),
        )

    def _audit(principal: Principal, msg: dict[str, Any], decision: str, extra: str = "") -> None:
        tool = ""
        if msg.get("method") == "tools/call" and isinstance(msg.get("params"), dict):
            tool = str(msg["params"].get("name", ""))
        audit.info(
            "decision=%s subject=%s role=%s method=%s tool=%s id=%s %s",
            decision, principal.subject, principal.role, msg.get("method"), tool, msg.get("id"), extra,
        )

    async def mcp(request: Request) -> Response:
        started = time.perf_counter()

        # 1. Authenticate --------------------------------------------------
        try:
            principal = authenticate(request.headers.get("authorization"), cfg.token_store)
        except AuthError as exc:
            audit.info("decision=unauthenticated reason=%r", exc.reason)
            return _rpc_error_response(
                JsonRpcError(AUTHENTICATION_REQUIRED, "Authentication required", {"reason": exc.reason}),
                status=401,
            )

        # 2. Parse ---------------------------------------------------------
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return _rpc_error_response(JsonRpcError(INVALID_REQUEST, "Request body too large"), status=413)
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _rpc_error_response(JsonRpcError(PARSE_ERROR, "Parse error"), status=400)

        is_batch = isinstance(body, list)
        messages: list[Any] = body if is_batch else [body]
        if is_batch and not messages:
            return _rpc_error_response(JsonRpcError(INVALID_REQUEST, "Invalid Request: empty batch"), status=400)

        # 3. Authorize each message -----------------------------------------
        # `slots` keeps the client's ordering: each entry is either a local answer
        # or a placeholder for a response we expect back from the downstream.
        slots: list[dict[str, Any] | None] = []
        to_forward: list[dict[str, Any]] = []
        forwarded_ids: dict[Any, int] = {}  # id -> slot index
        for msg in messages:
            envelope_error = validate_envelope(msg)
            if envelope_error is not None:
                slots.append(envelope_error.to_wire(msg.get("id") if isinstance(msg, dict) else None))
                continue
            decision = decide(msg, principal)
            if isinstance(decision, Reject):
                _audit(principal, msg, "reject", f"code={decision.error.code}")
                if "id" in msg:
                    slots.append(decision.error.to_wire(msg["id"]))
                else:
                    slots.append(None)  # a rejected notification has nothing to answer
                continue
            _audit(principal, msg, "forward")
            if "id" in msg:
                forwarded_ids[msg["id"]] = len(slots)
            slots.append(None)
            to_forward.append(msg)

        # Single-message denial: answer directly, same shape the downstream would use.
        if not is_batch and not to_forward:
            answer = slots[0]
            if answer is None:
                return Response(status_code=202)
            code = answer["error"]["code"]
            status = 403 if code == -32001 else 400
            return JSONResponse(answer, status_code=status)

        # 4. Forward -------------------------------------------------------
        upstream_body: Any = None
        upstream: httpx.Response | None = None
        if to_forward:
            payload: Any = to_forward if is_batch else to_forward[0]
            try:
                # The gateway owns the deadline, independent of the HTTP client's own timeouts.
                with anyio.fail_after(cfg.timeout_seconds):
                    upstream = await forward(request, principal, payload)
            except (TimeoutError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
                log.warning("downstream timeout after %.1fs", cfg.timeout_seconds)
                return _rpc_error_response(
                    JsonRpcError(UPSTREAM_UNAVAILABLE, "Upstream MCP server timed out"), status=504
                )
            except httpx.HTTPError as exc:  # ConnectError, ConnectTimeout, protocol errors
                log.warning("downstream error: %s", exc.__class__.__name__)
                return _rpc_error_response(
                    JsonRpcError(UPSTREAM_UNAVAILABLE, "Upstream MCP server unavailable"), status=502
                )

            if not is_batch:
                # Transparent pass-through of status, body and safe headers.
                headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _STRIP_RESPONSE_HEADERS}
                content = upstream.content
                if cfg.filter_tool_list and to_forward[0].get("method") == "tools/list":
                    try:
                        parsed = upstream.json()
                        if isinstance(parsed, dict) and "result" in parsed:
                            parsed["result"] = filter_tool_listing(parsed["result"], principal)
                            content = json.dumps(parsed).encode()
                            headers.pop("content-encoding", None)
                    except ValueError:
                        pass
                log.info("proxied %s -> %s in %.1fms", to_forward[0].get("method"), upstream.status_code,
                         (time.perf_counter() - started) * 1000)
                return Response(content=content, status_code=upstream.status_code, headers=headers)

            if upstream.status_code == 202 or not upstream.content.strip():
                upstream_body = []  # batch of notifications only: nothing comes back
            else:
                try:
                    upstream_body = upstream.json()
                except ValueError:
                    return _rpc_error_response(
                        JsonRpcError(UPSTREAM_UNAVAILABLE, "Upstream returned a non-JSON response"), status=502
                    )

        # 5. Splice batch responses back into the client's order --------------
        upstream_list = upstream_body if isinstance(upstream_body, list) else ([upstream_body] if upstream_body else [])
        for resp in upstream_list:
            if not isinstance(resp, dict):
                continue
            idx = forwarded_ids.get(resp.get("id"))
            if idx is None:
                continue
            if cfg.filter_tool_list and messages[idx].get("method") == "tools/list" and "result" in resp:
                resp = {**resp, "result": filter_tool_listing(resp["result"], principal)}
            slots[idx] = resp
        answers = [s for s in slots if s is not None]
        log.info("proxied batch of %d (%d forwarded) in %.1fms", len(messages), len(to_forward),
                 (time.perf_counter() - started) * 1000)
        if not answers:
            return Response(status_code=202)
        return JSONResponse(answers)

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok", "downstream": cfg.downstream_url})

    @asynccontextmanager
    async def lifespan(_: Starlette):
        try:
            yield
        finally:
            if http_client is None:
                await client.aclose()

    return Starlette(
        routes=[Route("/mcp", mcp, methods=["POST"]), Route("/healthz", health, methods=["GET"])],
        lifespan=lifespan,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = GatewayConfig(
        downstream_url=os.environ.get("DOWNSTREAM_URL", "http://127.0.0.1:9100/mcp"),
        token_store=TokenStore.from_env(),
        downstream_token=os.environ.get("DOWNSTREAM_TOKEN"),
        timeout_seconds=float(os.environ.get("DOWNSTREAM_TIMEOUT", "10")),
        filter_tool_list=os.environ.get("FILTER_TOOL_LIST", "0") == "1",
    )
    app = create_app(cfg)
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "9000")), log_level="info")


if __name__ == "__main__":
    main()
