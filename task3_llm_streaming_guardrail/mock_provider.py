"""Mock LLM provider speaking the OpenAI chat-completions wire format, streaming via SSE.

It exists so the gateway can be exercised without network access or API keys,
and so tests can control *exactly* how the text is chunked. Sensitive values
are deliberately split across chunk boundaries (``"john.d" | "oe@exa" | "mple.com"``)
because that is the case a naive per-chunk regex misses.

Request extensions (ignored by real providers, used here for testing):

* ``mock_text``: the assistant text to stream (defaults to a PII-laden sample).
* ``mock_chunks``: explicit list of chunk strings; overrides ``mock_text``.
* ``mock_delay_ms``: pause between chunks, to make latency measurable.

Run standalone::

    python -m task3_llm_streaming_guardrail.mock_provider   # 127.0.0.1:9200
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator

import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

log = logging.getLogger("mock-provider")

DEFAULT_TEXT = (
    "Sure! The customer's email is john.doe@example.com and her SSN is 123-45-6789. "
    "The card on file is 4111 1111 1111 1111 (exp 12/29). "
    "Order number 123456789012345 is not a card. Reach the team at support@corp.io."
)


def _default_chunks(text: str) -> list[str]:
    """Split roughly every 5-7 characters so PII straddles chunk boundaries."""
    out: list[str] = []
    i = 0
    sizes = [5, 7, 6, 4, 8]
    k = 0
    while i < len(text):
        n = sizes[k % len(sizes)]
        out.append(text[i : i + n])
        i += n
        k += 1
    return out


def _chunk_event(id_: str, model: str, created: int, delta: dict[str, Any], finish: str | None = None) -> bytes:
    payload = {
        "id": id_,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


async def chat_completions(request: Request) -> Response:
    try:
        body = json.loads(await request.body())
    except json.JSONDecodeError:
        return JSONResponse({"error": {"message": "invalid JSON", "type": "invalid_request_error"}}, status_code=400)

    model = body.get("model", "mock-1")
    text: str = body.get("mock_text", DEFAULT_TEXT)
    chunks: list[str] = body.get("mock_chunks") or _default_chunks(text)
    delay = float(body.get("mock_delay_ms", 0)) / 1000.0
    id_ = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if not body.get("stream"):
        full = "".join(chunks)
        return JSONResponse(
            {
                "id": id_,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": full}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": len(full) // 4, "total_tokens": 10 + len(full) // 4},
            }
        )

    async def stream() -> AsyncIterator[bytes]:
        yield _chunk_event(id_, model, created, {"role": "assistant", "content": ""})
        for piece in chunks:
            if delay:
                await anyio.sleep(delay)
            yield _chunk_event(id_, model, created, {"content": piece})
        yield _chunk_event(id_, model, created, {}, finish="stop")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"cache-control": "no-cache"})


def create_app() -> Starlette:
    return Starlette(routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])])


app = create_app()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "9200")), log_level="info")


if __name__ == "__main__":
    main()
