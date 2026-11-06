"""MCP server over stdio exposing ``get_customer_record`` and ``trigger_refund``.

Run from the repository root::

    python -m task1_mcp_server.server

stdout carries JSON-RPC only. All diagnostics go to stderr (see ``_configure_logging``).

Error mapping
-------------
Where a request fails determines what the client sees:

=====  ===========================================  =====================================
Stage  Failure                                      Response
=====  ===========================================  =====================================
1      Malformed JSON on the wire                   -32700 Parse error (SDK transport)
2      Valid JSON, invalid JSON-RPC envelope        -32600 Invalid request (SDK)
3      Unknown method (``tools/explode``)           -32601 Method not found (SDK)
4      ``tools/call`` with an unknown tool name     -32602 Invalid params (this module)
5      Arguments fail the Pydantic schema           -32602 Invalid params, with field-level
                                                    detail in ``error.data`` (this module)
6      Tool ran; business rule failed               ``CallToolResult(is_error=True)``
7      Unexpected exception inside a handler        -32603 Internal error (SDK), message
                                                    sanitised so no traceback leaks
=====  ===========================================  =====================================

Stages 1-5 are protocol errors: the *request* was wrong. Stage 6 is a successful
protocol exchange whose *outcome* was a failure the model should see and act
on. The MCP specification reserves ``isError`` for exactly that case, and tells
servers to use ``-32602`` for unknown tools and invalid arguments.

The high-level ``MCPServer`` class in the ``mcp`` SDK reports schema failures
as ``is_error`` results instead. This server is built on the low-level
``Server`` so that validation failures can be surfaced with the standard
JSON-RPC code the task requires.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import anyio
from mcp import types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ValidationError

from .schemas import CustomerRecord, GetCustomerRecordInput, RefundReceipt, TriggerRefundInput
from .store import BusinessError, CustomerStore
from .transport import guarded_streams

SERVER_NAME = "customer-ops"
SERVER_VERSION = "1.0.0"

log = logging.getLogger(SERVER_NAME)


# --------------------------------------------------------------------------- #
# Logging: stderr only.                                                       #
# --------------------------------------------------------------------------- #
def _configure_logging() -> None:
    """Route every log record to stderr; never let anything touch stdout.

    Two layers of protection:

    1. ``logging`` is configured with an explicit ``stream=sys.stderr`` handler,
       and ``force=True`` tears down any handler a library may have attached to
       stdout.
    2. While the SDK's ``stdio_server`` is active it re-points file descriptor 1
       at stderr, so even a stray ``print()`` in a handler (or a child process)
       lands on stderr instead of the wire. The test-suite verifies the wire
       stays clean with a raw subprocess, not through the SDK client.
    """
    level_name = os.environ.get("MCP_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


# --------------------------------------------------------------------------- #
# Tool registry                                                               #
# --------------------------------------------------------------------------- #
class ToolSpec:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        annotations: types.ToolAnnotations,
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.output_model = output_model
        self.annotations = annotations

    def as_mcp_tool(self) -> types.Tool:
        return types.Tool(
            name=self.name,
            title=self.annotations.title,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
            output_schema=self.output_model.model_json_schema(),
            annotations=self.annotations,
        )


TOOLS: dict[str, ToolSpec] = {
    "get_customer_record": ToolSpec(
        name="get_customer_record",
        description=(
            "Fetch a customer's profile and current balance. "
            "customer_id must be in the form CUST-XXXXX (five digits)."
        ),
        input_model=GetCustomerRecordInput,
        output_model=CustomerRecord,
        annotations=types.ToolAnnotations(
            title="Get customer record", read_only_hint=True, idempotent_hint=True, open_world_hint=False
        ),
    ),
    "trigger_refund": ToolSpec(
        name="trigger_refund",
        description=(
            "Issue a refund to a customer. amount must be a positive number with at most 2 decimals; "
            "reason must be at least 10 characters."
        ),
        input_model=TriggerRefundInput,
        output_model=RefundReceipt,
        annotations=types.ToolAnnotations(
            title="Trigger refund",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    ),
}


# --------------------------------------------------------------------------- #
# Error helpers                                                               #
# --------------------------------------------------------------------------- #
def _validation_error_to_mcp(tool: str, exc: ValidationError) -> MCPError:
    """Translate a Pydantic failure into ``-32602`` with structured, value-free detail.

    ``include_input=False`` keeps the caller's data out of the error (it may be
    PII or be logged downstream). The field path and the rule are enough for the
    model to correct the call.
    """
    issues = [
        {
            "field": ".".join(str(p) for p in err["loc"]) or "<root>",
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors(include_url=False, include_input=False)
    ]
    summary = "; ".join(f"{i['field']}: {i['message']}" for i in issues)
    return MCPError(
        code=types.INVALID_PARAMS,
        message=f"Invalid arguments for tool '{tool}': {summary}",
        data={"tool": tool, "issues": issues},
    )


def _business_error_result(exc: BusinessError) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=f"{exc.code}: {exc.message}")],
        structured_content={"error": {"code": exc.code, "message": exc.message}},
        is_error=True,
    )


def _ok_result(model: BaseModel) -> types.CallToolResult:
    payload = model.model_dump(mode="json")
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=model.model_dump_json())],
        structured_content=payload,
    )


# --------------------------------------------------------------------------- #
# Server                                                                      #
# --------------------------------------------------------------------------- #
def build_server(store: CustomerStore | None = None) -> Server[Any]:
    store = store or CustomerStore()

    async def on_list_tools(
        _ctx: ServerRequestContext[Any], _params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[spec.as_mcp_tool() for spec in TOOLS.values()])

    async def on_call_tool(
        _ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        spec = TOOLS.get(params.name)
        if spec is None:
            log.warning("tools/call for unknown tool %r", params.name)
            raise MCPError(
                code=types.INVALID_PARAMS,
                message=f"Unknown tool: {params.name}",
                data={"tool": params.name, "known_tools": sorted(TOOLS)},
            )

        raw_args = params.arguments if params.arguments is not None else {}
        try:
            args = spec.input_model.model_validate(raw_args)
        except ValidationError as exc:
            log.info("tools/call %s rejected: %d validation issue(s)", params.name, exc.error_count())
            raise _validation_error_to_mcp(params.name, exc) from None

        log.info("tools/call %s", params.name)
        try:
            if isinstance(args, GetCustomerRecordInput):
                return _ok_result(store.get(args.customer_id))
            if isinstance(args, TriggerRefundInput):
                receipt = store.refund(args.customer_id, args.amount, args.reason)
                log.info("refund processed id=%s customer=%s amount=%.2f", receipt.refund_id, args.customer_id, args.amount)
                return _ok_result(receipt)
        except BusinessError as exc:
            log.info("tools/call %s business error %s", params.name, exc.code)
            return _business_error_result(exc)

        # Unreachable: every registered input model is handled above.
        raise MCPError(code=types.INTERNAL_ERROR, message=f"No handler for tool {params.name}")

    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions=(
            "Customer operations. Look up a customer with get_customer_record before "
            "calling trigger_refund so you know the available balance."
        ),
        get_tool_input_schema=lambda name: TOOLS[name].input_model.model_json_schema() if name in TOOLS else None,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def serve() -> None:
    server = build_server()
    async with stdio_server() as (wire_read, wire_write):
        log.info("server started transport=stdio pid=%s", os.getpid())
        async with guarded_streams(wire_read, wire_write) as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    log.info("server stopped")


def main() -> None:
    _configure_logging()
    try:
        anyio.run(serve)
    except KeyboardInterrupt:
        log.info("interrupted")


if __name__ == "__main__":
    main()
