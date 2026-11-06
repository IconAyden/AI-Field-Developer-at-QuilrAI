"""LLM gateway with an in-flight PII redaction guardrail.

    client  --POST /v1/chat/completions-->  [ gateway :9300 ]  -->  provider (OpenAI-compatible SSE)
            <--- SSE, PII replaced with [REDACTED] ---

Design for low latency and bounded memory:

* The provider response is consumed with ``aiter_bytes`` and parsed by a small
  incremental SSE parser (``SSEParser``) that only ever holds one partial event.
* Each delta's text goes through ``StreamRedactor``, which releases text as soon
  as it cannot be part of a pattern (bounded holdback, see ``redactor.py``).
  Nothing accumulates the full response.
* Output events are written as soon as there is redacted text to send, so the
  gateway adds at most "one boundary character" of delay on top of the provider.
* Response headers disable proxy buffering (``X-Accel-Buffering: no``) and the
  response is a Starlette ``StreamingResponse``, so uvicorn flushes each event.

Run from the repository root::

    python -m task3_llm_streaming_guardrail.mock_provider     # terminal 1
    python -m task3_llm_streaming_guardrail.gateway           # terminal 2
    curl -N localhost:9300/v1/chat/completions -H 'Content-Type: application/json' \
         -d '{"model":"mock-1","stream":true,"messages":[{"role":"user","content":"hi"}]}'
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .redactor import DEFAULT_PATTERNS, Pattern, StreamRedactor, redact_text

log = logging.getLogger("llm-gateway")

SSE_HEADERS = {
    "cache-control": "no-cache, no-transform",
    "x-accel-buffering": "no",  # nginx: do not buffer this response
    "connection": "keep-alive",
}


# --------------------------------------------------------------------------- #
# Incremental SSE parsing                                                     #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SSEEvent:
    data: str
    event: str | None = None
    id: str | None = None


@dataclass(slots=True)
class SSEParser:
    """Feed raw bytes, get complete events. Holds at most one partial event."""

    _buf: bytes = field(default=b"", init=False)
    max_event_bytes: int = 1 * 1024 * 1024

    def feed(self, data: bytes) -> list[SSEEvent]:
        self._buf += data
        if len(self._buf) > self.max_event_bytes:
            raise ValueError("SSE event exceeds size limit")
        events: list[SSEEvent] = []
        while True:
            # An event ends at a blank line: \n\n or \r\n\r\n.
            idx_n = self._buf.find(b"\n\n")
            idx_rn = self._buf.find(b"\r\n\r\n")
            if idx_n == -1 and idx_rn == -1:
                break
            if idx_rn != -1 and (idx_n == -1 or idx_rn < idx_n):
                raw, self._buf = self._buf[:idx_rn], self._buf[idx_rn + 4 :]
            else:
                raw, self._buf = self._buf[:idx_n], self._buf[idx_n + 2 :]
            ev = self._parse_block(raw.decode("utf-8", errors="replace"))
            if ev is not None:
                events.append(ev)
        return events

    @staticmethod
    def _parse_block(block: str) -> SSEEvent | None:
        data_lines: list[str] = []
        event: str | None = None
        id_: str | None = None
        for line in block.splitlines():
            if not line or line.startswith(":"):
                continue
            name, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if name == "data":
                data_lines.append(value)
            elif name == "event":
                event = value
            elif name == "id":
                id_ = value
        if not data_lines and event is None:
            return None
        return SSEEvent(data="\n".join(data_lines), event=event, id=id_)


def _sse(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


# --------------------------------------------------------------------------- #
# Gateway                                                                     #
# --------------------------------------------------------------------------- #
class GatewayConfig:
    def __init__(
        self,
        *,
        provider_url: str,
        provider_api_key: str | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
        patterns: tuple[Pattern, ...] = DEFAULT_PATTERNS,
        max_holdback: int = 64,
    ) -> None:
        self.provider_url = provider_url
        self.provider_api_key = provider_api_key
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.patterns = patterns
        self.max_holdback = max_holdback


def _error(status: int, message: str, type_: str = "gateway_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": type_}}, status_code=status)


def _rewrite_delta_event(event_json: dict[str, Any], text: str) -> bytes:
    """Clone an OpenAI chunk event, replacing the delta content with redacted text."""
    choices = event_json.get("choices") or [{}]
    first = dict(choices[0]) if choices else {}
    delta = dict(first.get("delta") or {})
    delta["content"] = text
    first["delta"] = delta
    first.setdefault("index", 0)
    first["finish_reason"] = None
    return _sse(json.dumps({**event_json, "choices": [first]}))


def create_app(cfg: GatewayConfig, *, http_client: httpx.AsyncClient | None = None) -> Starlette:
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(cfg.read_timeout, connect=cfg.connect_timeout)
    )

    def provider_headers(request: Request) -> dict[str, str]:
        headers = {"content-type": "application/json", "accept": request.headers.get("accept", "*/*")}
        if cfg.provider_api_key:
            headers["authorization"] = f"Bearer {cfg.provider_api_key}"
        elif auth := request.headers.get("authorization"):
            headers["authorization"] = auth  # pass-through mode: client brings its own provider key
        return headers

    async def stream_redacted(
        upstream: httpx.Response, started: float, redactor: StreamRedactor
    ) -> AsyncIterator[bytes]:
        """Translate provider SSE into client SSE with redacted deltas."""
        parser = SSEParser()
        first_byte_out: float | None = None
        chunks_in = chunks_out = 0
        last_event_json: dict[str, Any] | None = None
        try:
            async for raw in upstream.aiter_bytes():
                for ev in parser.feed(raw):
                    chunks_in += 1
                    if ev.data.strip() == "[DONE]":
                        tail = redactor.flush()
                        if tail and last_event_json is not None:
                            chunks_out += 1
                            yield _rewrite_delta_event(last_event_json, tail)
                        yield b"data: [DONE]\n\n"
                        continue
                    try:
                        ej = json.loads(ev.data)
                    except json.JSONDecodeError:
                        yield _sse(ev.data)  # not ours to interpret; forward verbatim
                        continue

                    choices = ej.get("choices") or []
                    first = choices[0] if choices else {}
                    delta = first.get("delta") or {}
                    content = delta.get("content")
                    finish = first.get("finish_reason")

                    if isinstance(content, str) and content:
                        last_event_json = ej
                        out = redactor.feed(content)
                        if out:
                            chunks_out += 1
                            if first_byte_out is None:
                                first_byte_out = time.perf_counter()
                            yield _rewrite_delta_event(ej, out)
                        if finish:
                            # Content and finish in the same chunk: flush before signalling the end.
                            tail = redactor.flush()
                            if tail:
                                chunks_out += 1
                                yield _rewrite_delta_event(ej, tail)
                            yield _sse(json.dumps({**ej, "choices": [{**first, "delta": {}}]}))
                        continue

                    if finish:
                        tail = redactor.flush()
                        if tail and last_event_json is not None:
                            chunks_out += 1
                            yield _rewrite_delta_event(last_event_json, tail)
                    # Role-only, finish-only, usage-only events pass through untouched.
                    yield _sse(ev.data)
            # Provider closed without [DONE]: still release what we hold.
            tail = redactor.flush()
            if tail and last_event_json is not None:
                yield _rewrite_delta_event(last_event_json, tail)
        finally:
            await upstream.aclose()
            ttft = (first_byte_out - started) * 1000 if first_byte_out else None
            log.info(
                "stream done in=%d out=%d ttft_ms=%s total_ms=%.0f peak_holdback=%d",
                chunks_in, chunks_out, f"{ttft:.0f}" if ttft is not None else "n/a",
                (time.perf_counter() - started) * 1000, redactor.peak_held,
            )

    async def chat_completions(request: Request) -> Response:
        started = time.perf_counter()
        try:
            body = json.loads(await request.body())
        except json.JSONDecodeError:
            return _error(400, "Request body must be valid JSON", "invalid_request_error")
        if not isinstance(body, dict):
            return _error(400, "Request body must be a JSON object", "invalid_request_error")

        streaming = bool(body.get("stream"))
        req = client.build_request(
            "POST", cfg.provider_url, content=json.dumps(body).encode(), headers=provider_headers(request)
        )

        try:
            upstream = await client.send(req, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            log.warning("provider unreachable: %s", exc.__class__.__name__)
            return _error(502, "The model provider is unavailable")
        except httpx.TimeoutException:
            return _error(504, "The model provider did not respond in time")
        except httpx.HTTPError as exc:
            log.warning("provider error: %s", exc.__class__.__name__)
            return _error(502, "The model provider is unavailable")

        if upstream.status_code != 200:
            # Never stream a provider error page through; summarise it.
            await upstream.aread()
            await upstream.aclose()
            log.warning("provider returned %s", upstream.status_code)
            status = 429 if upstream.status_code == 429 else 502
            return _error(status, f"The model provider rejected the request (status {upstream.status_code})")

        if not streaming:
            await upstream.aread()
            await upstream.aclose()
            try:
                payload = upstream.json()
            except ValueError:
                return _error(502, "The model provider returned a malformed response")
            for choice in payload.get("choices") or []:
                msg = choice.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    msg["content"] = redact_text(msg["content"], cfg.patterns)
            return JSONResponse(payload)

        redactor = StreamRedactor(patterns=cfg.patterns, max_holdback=cfg.max_holdback)
        return StreamingResponse(
            stream_redacted(upstream, started, redactor),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok", "provider": cfg.provider_url})

    @asynccontextmanager
    async def lifespan(_: Starlette):
        try:
            yield
        finally:
            if http_client is None:
                await client.aclose()

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/healthz", health, methods=["GET"]),
        ],
        lifespan=lifespan,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = GatewayConfig(
        provider_url=os.environ.get("PROVIDER_URL", "http://127.0.0.1:9200/v1/chat/completions"),
        provider_api_key=os.environ.get("PROVIDER_API_KEY"),
        max_holdback=int(os.environ.get("MAX_HOLDBACK", "64")),
    )
    uvicorn.run(create_app(cfg), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "9300")), log_level="info")


if __name__ == "__main__":
    main()
