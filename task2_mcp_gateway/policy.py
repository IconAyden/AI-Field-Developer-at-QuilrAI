"""Method-level authorization policy.

The policy is a pure function from (JSON-RPC message, principal) to a
decision. Keeping it free of HTTP and I/O makes it trivial to unit-test and
to reason about: the gateway asks "may this principal send this message?"
and acts on the answer.

Decisions
---------
* ``Forward``       send to the downstream as-is.
* ``Reject(error)`` answer the client directly; the downstream never sees it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .auth import Principal

# JSON-RPC standard codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602

# Gateway-specific codes live in the JSON-RPC "server error" range (-32000..-32099).
UNAUTHORIZED_TOOL_CALL = -32001  # authenticated, but the role may not call this tool
AUTHENTICATION_REQUIRED = -32002  # no usable bearer token
UPSTREAM_UNAVAILABLE = -32003  # downstream MCP server unreachable / timed out

ADMIN_TOOL_PREFIX = "admin_"


@dataclass(frozen=True, slots=True)
class JsonRpcError:
    code: int
    message: str
    data: Any = None

    def to_wire(self, id_: Any) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return {"jsonrpc": "2.0", "id": id_, "error": err}


@dataclass(frozen=True, slots=True)
class Forward:
    pass


@dataclass(frozen=True, slots=True)
class Reject:
    error: JsonRpcError


Decision = Forward | Reject


def is_admin_tool(name: str) -> bool:
    return name.startswith(ADMIN_TOOL_PREFIX)


def validate_envelope(msg: Any) -> JsonRpcError | None:
    """Structural check of a single JSON-RPC 2.0 message. None means it is well-formed."""
    if not isinstance(msg, dict):
        return JsonRpcError(INVALID_REQUEST, "Invalid Request: message must be an object")
    if msg.get("jsonrpc") != "2.0":
        return JsonRpcError(INVALID_REQUEST, "Invalid Request: jsonrpc must be '2.0'")
    if not isinstance(msg.get("method"), str) or not msg["method"]:
        return JsonRpcError(INVALID_REQUEST, "Invalid Request: method must be a non-empty string")
    if "id" in msg and not isinstance(msg["id"], (str, int, type(None))):
        return JsonRpcError(INVALID_REQUEST, "Invalid Request: id must be a string, number, or null")
    if "params" in msg and not isinstance(msg["params"], (dict, list)):
        return JsonRpcError(INVALID_REQUEST, "Invalid Request: params must be an object or array")
    return None


def decide(msg: dict[str, Any], principal: Principal) -> Decision:
    """Authorize one well-formed JSON-RPC message for the given principal."""
    method = msg["method"]

    if method == "tools/list":
        return Forward()

    if method == "tools/call":
        params = msg.get("params")
        if not isinstance(params, dict):
            return Reject(JsonRpcError(INVALID_PARAMS, "tools/call requires params object"))
        name = params.get("name")
        if not isinstance(name, str) or not name:
            # Reject here rather than forward: a missing name can never be authorized,
            # and the downstream should not have to defend against malformed calls.
            return Reject(JsonRpcError(INVALID_PARAMS, "tools/call params.name must be a non-empty string"))
        if is_admin_tool(name) and principal.role != "admin":
            return Reject(
                JsonRpcError(
                    UNAUTHORIZED_TOOL_CALL,
                    "Unauthorized Tool Call",
                    {"tool": name, "required_role": "admin", "role": principal.role},
                )
            )
        return Forward()

    # initialize, ping, notifications/*, prompts/*, resources/*: no tool semantics, forward.
    return Forward()


def filter_tool_listing(result: Any, principal: Principal) -> Any:
    """Optional response filter: hide admin_ tools from non-admin listings.

    Off by default (the task asks for transparent forwarding of tools/list).
    Enabling it stops a viewer's agent from even planning an admin_ call.
    """
    if principal.role == "admin" or not isinstance(result, dict):
        return result
    tools = result.get("tools")
    if not isinstance(tools, list):
        return result
    return {**result, "tools": [t for t in tools if not (isinstance(t, dict) and is_admin_tool(str(t.get("name", ""))))]}
