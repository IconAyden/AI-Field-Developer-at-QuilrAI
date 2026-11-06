"""Mock downstream MCP server speaking JSON-RPC 2.0 over HTTP POST.

It has *no* authorization of its own, on purpose: the point of the exercise
is that the gateway in front of it is the only thing standing between a
viewer and ``admin_reset_key``. Every call is recorded in ``CALLS`` so tests
can prove a blocked call never reached this process.

Run standalone::

    python -m task2_mcp_gateway.downstream_mock   # listens on 127.0.0.1:9100
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

log = logging.getLogger("downstream-mock")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    {
        "name": "echo",
        "description": "Echo the input back.",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    {
        "name": "admin_reset_key",
        "description": "Rotate the tenant API key. Destructive.",
        "inputSchema": {"type": "object", "properties": {"tenant": {"type": "string"}}, "required": ["tenant"]},
    },
    {
        "name": "admin_delete_user",
        "description": "Permanently delete a user. Destructive.",
        "inputSchema": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
    },
]

# Every tools/call that reached this server, in order. Tests assert on this.
CALLS: list[dict[str, Any]] = []


def _error(id_: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": err}


def _result(id_: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    CALLS.append({"name": name, "arguments": arguments})
    if name == "get_weather":
        return {"content": [{"type": "text", "text": f"Sunny, 21C in {arguments.get('city', '?')}"}]}
    if name == "echo":
        return {"content": [{"type": "text", "text": str(arguments.get("text", ""))}]}
    if name == "admin_reset_key":
        return {"content": [{"type": "text", "text": f"API key rotated for tenant {arguments.get('tenant', '?')}"}]}
    if name == "admin_delete_user":
        return {"content": [{"type": "text", "text": f"Deleted user {arguments.get('user_id', '?')}"}]}
    return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}


def handle_message(msg: Any) -> dict[str, Any] | None:
    """Handle one JSON-RPC message. Returns None for notifications (no id)."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return _error(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST, "Invalid Request")
    id_ = msg.get("id")
    method = msg["method"]
    params = msg.get("params") or {}

    if "id" not in msg:  # notification
        log.info("notification %s", method)
        return None

    if method == "initialize":
        return _result(
            id_,
            {
                "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "downstream-mock", "version": "1.0.0"},
            },
        )
    if method == "ping":
        return _result(id_, {})
    if method == "tools/list":
        return _result(id_, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        if not isinstance(name, str):
            return _error(id_, INVALID_PARAMS, "params.name must be a string")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _error(id_, INVALID_PARAMS, "params.arguments must be an object")
        return _result(id_, _call_tool(name, arguments))
    return _error(id_, METHOD_NOT_FOUND, "Method not found", {"method": method})


async def mcp_endpoint(request: Request) -> Response:
    try:
        body = json.loads(await request.body())
    except json.JSONDecodeError:
        return JSONResponse(_error(None, PARSE_ERROR, "Parse error"), status_code=400)

    if isinstance(body, list):
        if not body:
            return JSONResponse(_error(None, INVALID_REQUEST, "Invalid Request: empty batch"), status_code=400)
        responses = [r for r in (handle_message(m) for m in body) if r is not None]
        if not responses:
            return Response(status_code=202)
        return JSONResponse(responses)

    response = handle_message(body)
    if response is None:
        return Response(status_code=202)
    return JSONResponse(response)


def create_app() -> Starlette:
    return Starlette(routes=[Route("/mcp", mcp_endpoint, methods=["POST"])])


app = create_app()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "9100")), log_level="info")


if __name__ == "__main__":
    main()
