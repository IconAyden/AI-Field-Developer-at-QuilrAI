"""Two mock model providers with switchable failure modes.

Each provider exposes:

* ``POST /v1/chat/completions``: behaves according to its current mode.
* ``POST /_control`` ``{"mode": "ok" | "429" | "500" | "hang" | "slow"}``: switch mode.
* ``GET /_stats``: ``{"mode": ..., "calls": n}``.

Modes:

* ``ok``    normal completion with a ``usage`` block.
* ``429``   Too Many Requests, with ``Retry-After: 1``.
* ``500``   Internal Server Error with a *fake stack trace* in the body. The
            gateway must never forward that body.
* ``hang``  never answers (sleeps 60 s). Exercises the 3000 ms timeout.
* ``slow``  answers after ``slow_ms`` (default 1000 ms); under the timeout.

Run standalone (primary on 9401, secondary on 9402)::

    python -m task4_model_router.mock_providers
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Any

import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

log = logging.getLogger("mock-provider")

FAKE_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/srv/provider/app.py", line 231, in complete\n'
    "    raise RuntimeError('shard 7 at 10.0.3.14:8443 out of memory')\n"
    "RuntimeError: shard 7 at 10.0.3.14:8443 out of memory\n"
)


class ProviderState:
    def __init__(self, name: str, mode: str = "ok", slow_ms: int = 1000) -> None:
        self.name = name
        self.mode = mode
        self.slow_ms = slow_ms
        self.calls = 0
        self.in_flight = 0
        self.cancelled = 0  # requests whose client went away before we answered


def create_provider_app(state: ProviderState) -> Starlette:
    async def completions(request: Request) -> Response:
        state.calls += 1
        state.in_flight += 1
        try:
            try:
                body = json.loads(await request.body())
            except json.JSONDecodeError:
                return JSONResponse({"error": {"message": "bad json"}}, status_code=400)

            mode = state.mode
            if mode == "429":
                return JSONResponse(
                    {"error": {"message": "Rate limit reached", "type": "rate_limit_error"}},
                    status_code=429,
                    headers={"retry-after": "1"},
                )
            if mode == "500":
                return PlainTextResponse(FAKE_TRACEBACK, status_code=500)
            if mode == "400":
                return JSONResponse({"error": {"message": "model does not exist", "type": "invalid_request_error"}}, status_code=400)
            if mode == "hang":
                try:
                    await anyio.sleep(60)
                except anyio.get_cancelled_exc_class():
                    state.cancelled += 1
                    raise
                return JSONResponse({"error": "unreachable"}, status_code=500)
            if mode == "slow":
                await anyio.sleep(state.slow_ms / 1000)

            prompt = " ".join(str(m.get("content", "")) for m in body.get("messages", []) if isinstance(m, dict))
            text = f"[{state.name}:{body.get('model', '?')}] echo: {prompt[:80]}"
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(text) // 4)
            return JSONResponse(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "?"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
        finally:
            state.in_flight -= 1

    async def control(request: Request) -> Response:
        body: dict[str, Any] = json.loads(await request.body())
        if "mode" in body:
            state.mode = str(body["mode"])
        if "slow_ms" in body:
            state.slow_ms = int(body["slow_ms"])
        return JSONResponse({"name": state.name, "mode": state.mode, "slow_ms": state.slow_ms})

    async def stats(_: Request) -> Response:
        return JSONResponse({"name": state.name, "mode": state.mode, "calls": state.calls, "in_flight": state.in_flight, "cancelled": state.cancelled})

    return Starlette(
        routes=[
            Route("/v1/chat/completions", completions, methods=["POST"]),
            Route("/_control", control, methods=["POST"]),
            Route("/_stats", stats, methods=["GET"]),
        ]
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("HOST", "127.0.0.1")
    primary = create_provider_app(ProviderState("primary", mode=os.environ.get("PRIMARY_MODE", "ok")))
    secondary = create_provider_app(ProviderState("secondary", mode=os.environ.get("SECONDARY_MODE", "ok")))
    p1, p2 = int(os.environ.get("PRIMARY_PORT", "9401")), int(os.environ.get("SECONDARY_PORT", "9402"))
    t = threading.Thread(target=lambda: uvicorn.run(secondary, host=host, port=p2, log_level="info"), daemon=True)
    t.start()
    log.info("primary on %s:%s, secondary on %s:%s", host, p1, host, p2)
    uvicorn.run(primary, host=host, port=p1, log_level="info")


if __name__ == "__main__":
    main()
