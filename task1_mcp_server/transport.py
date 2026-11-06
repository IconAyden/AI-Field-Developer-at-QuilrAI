"""Transport guard that sits between the SDK's stdio streams and the server loop.

It adds two behaviours the stock stdio transport does not have:

1. **Parse errors are answered.** JSON-RPC 2.0 says a request that is not valid
   JSON must get ``-32700 Parse error`` with ``"id": null``. The SDK's stdio
   reader hands the exception to the dispatcher, which only logs it at DEBUG.
   This guard turns it into a proper wire response.

2. **Graceful drain on EOF.** Tool calls are dispatched concurrently. When the
   client closes stdin while calls are still running, the server loop shuts
   down and their responses are dropped on the floor. The guard keeps the
   inner read stream open until every request it forwarded has been answered
   (bounded by ``drain_timeout``), so the last requests before EOF still get
   their responses.

Everything that passes through is a ``SessionMessage``; nothing here touches
stdout directly, so the stdout-is-only-JSON-RPC invariant is preserved.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from mcp import types
from mcp.shared._context_streams import ContextReceiveStream, ContextSendStream, create_context_streams
from mcp.shared.message import SessionMessage

log = logging.getLogger("customer-ops.transport")

InboundItem = SessionMessage | Exception


def parse_error_response(exc: Exception) -> SessionMessage:
    """``-32700`` with a null id. The offending bytes are not echoed back."""
    detail = exc.__class__.__name__
    return SessionMessage(
        types.JSONRPCError(
            jsonrpc="2.0",
            id=None,
            error=types.ErrorData(code=types.PARSE_ERROR, message="Parse error", data={"reason": detail}),
        )
    )


@asynccontextmanager
async def guarded_streams(
    wire_read: ContextReceiveStream[InboundItem],
    wire_write: ContextSendStream[SessionMessage],
    *,
    drain_timeout: float = 5.0,
) -> AsyncIterator[tuple[ContextReceiveStream[InboundItem], ContextSendStream[SessionMessage]]]:
    inner_read_tx, inner_read_rx = create_context_streams[InboundItem](0)
    inner_write_tx, inner_write_rx = create_context_streams[SessionMessage](0)

    pending: set[types.RequestId] = set()
    # Re-created each time the pending set refills, so a late request after an
    # earlier quiet period still gets drained on EOF.
    drained = {"event": anyio.Event()}

    async def inbound() -> None:
        try:
            async for item in wire_read:
                if isinstance(item, Exception):
                    log.warning("unparseable line on stdin: %s", item.__class__.__name__)
                    await wire_write.send(parse_error_response(item))
                    continue
                msg = item.message
                if isinstance(msg, types.JSONRPCRequest):
                    if drained["event"].is_set():
                        drained["event"] = anyio.Event()
                    pending.add(msg.id)
                await inner_read_tx.send(item)
        finally:
            # stdin is closed. Let in-flight requests finish before the server loop sees EOF.
            if pending:
                log.info("stdin closed with %d request(s) in flight; draining", len(pending))
                with anyio.move_on_after(drain_timeout) as scope:
                    await drained["event"].wait()
                if scope.cancelled_caught:
                    log.warning("drain timeout; %d request(s) unanswered", len(pending))
            await inner_read_tx.aclose()

    async def outbound() -> None:
        try:
            async for item in inner_write_rx:
                msg = item.message
                if isinstance(msg, (types.JSONRPCResponse, types.JSONRPCError)):
                    pending.discard(msg.id)
                    if not pending:
                        drained["event"].set()
                await wire_write.send(item)
        finally:
            await wire_write.aclose()

    async with anyio.create_task_group() as tg:
        tg.start_soon(inbound)
        tg.start_soon(outbound)
        try:
            yield inner_read_rx, inner_write_tx
        finally:
            # The server loop has returned; closing its write side lets outbound() finish.
            await inner_write_tx.aclose()
