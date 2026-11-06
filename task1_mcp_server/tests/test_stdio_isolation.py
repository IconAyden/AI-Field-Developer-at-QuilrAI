"""Wire-level tests: drive the server as a real subprocess over stdin/stdout.

These deliberately do *not* use the SDK client. The SDK client would silently
discard non-JSON lines on stdout; the point here is to prove there are none.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HANDSHAKE_PROTOCOL_VERSION = "2025-06-18"


def _rpc(id_: int | None, method: str, params: dict | None = None) -> str:
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if id_ is not None:
        msg["id"] = id_
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


def run_server(lines: list[str], *, env_extra: dict[str, str] | None = None) -> tuple[list[str], str]:
    """Feed raw lines to the server, close stdin, return (stdout_lines, stderr_text)."""
    env = {**os.environ, "MCP_LOG_LEVEL": "DEBUG", "PYTHONIOENCODING": "utf-8", **(env_extra or {})}
    proc = subprocess.run(
        [sys.executable, "-m", "task1_mcp_server.server"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO_ROOT,
        env=env,
        timeout=30,
    )
    stdout_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return stdout_lines, proc.stderr


def _handshake() -> list[str]:
    return [
        _rpc(
            1,
            "initialize",
            {
                "protocolVersion": HANDSHAKE_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "raw-test", "version": "0"},
            },
        ),
        _rpc(None, "notifications/initialized"),
    ]


@pytest.fixture(scope="module")
def session() -> tuple[list[dict], str]:
    """One subprocess run exercising the happy path and every error stage."""
    lines = _handshake() + [
        _rpc(2, "tools/list"),
        _rpc(3, "tools/call", {"name": "get_customer_record", "arguments": {"customer_id": "CUST-00001"}}),
        _rpc(4, "tools/call", {"name": "get_customer_record", "arguments": {"customer_id": "bad-id"}}),
        _rpc(5, "tools/call", {"name": "no_such_tool", "arguments": {}}),
        _rpc(6, "no/such/method"),
        '{"jsonrpc": "2.0", "id": 7, "method": "tools/list"   this is not json',
        _rpc(
            8,
            "tools/call",
            {
                "name": "trigger_refund",
                "arguments": {"customer_id": "CUST-00001", "amount": 10.5, "reason": "Duplicate charge on invoice"},
            },
        ),
        _rpc(9, "tools/call", {"name": "trigger_refund", "arguments": {"customer_id": "CUST-00001", "amount": 99999, "reason": "Larger than the balance"}}),
    ]
    stdout_lines, stderr = run_server(lines)
    parsed = [json.loads(ln) for ln in stdout_lines]
    return parsed, stderr


def test_stdout_contains_only_jsonrpc(session):
    messages, _ = session
    assert messages, "server produced no output"
    for msg in messages:
        assert msg.get("jsonrpc") == "2.0", f"non-JSON-RPC line on stdout: {msg!r}"
        assert ("result" in msg) ^ ("error" in msg) or "method" in msg, msg


def test_logs_went_to_stderr_not_stdout(session):
    _, stderr = session
    assert "server started" in stderr
    assert "tools/call" in stderr


def test_stdout_lines_all_parse_even_with_debug_logging():
    """Independent run with aggressive logging: every stdout line must still parse."""
    stdout_lines, _ = run_server(_handshake() + [_rpc(2, "tools/list")])
    for ln in stdout_lines:
        json.loads(ln)  # raises if any log line leaked


def _by_id(messages: list[dict], id_: int) -> dict:
    for m in messages:
        if m.get("id") == id_:
            return m
    raise AssertionError(f"no response with id {id_}; got ids {[m.get('id') for m in messages]}")


def test_initialize_and_list(session):
    messages, _ = session
    init = _by_id(messages, 1)
    assert init["result"]["serverInfo"]["name"] == "customer-ops"
    tools = _by_id(messages, 2)["result"]["tools"]
    assert {t["name"] for t in tools} == {"get_customer_record", "trigger_refund"}
    refund_schema = next(t for t in tools if t["name"] == "trigger_refund")["inputSchema"]
    assert refund_schema["additionalProperties"] is False
    assert set(refund_schema["required"]) == {"customer_id", "amount", "reason"}


def test_valid_call_returns_result(session):
    messages, _ = session
    res = _by_id(messages, 3)["result"]
    assert res.get("isError") is not True
    assert res["structuredContent"]["customer_id"] == "CUST-00001"


def test_invalid_customer_id_is_minus_32602(session):
    messages, _ = session
    err = _by_id(messages, 4)["error"]
    assert err["code"] == -32602
    assert err["data"]["issues"][0]["field"] == "customer_id"
    assert "bad-id" not in json.dumps(err), "caller input must not be echoed back"


def test_unknown_tool_is_minus_32602(session):
    messages, _ = session
    assert _by_id(messages, 5)["error"]["code"] == -32602


def test_unknown_method_is_minus_32601(session):
    messages, _ = session
    assert _by_id(messages, 6)["error"]["code"] == -32601


def test_malformed_json_is_minus_32700(session):
    messages, _ = session
    parse_errors = [m for m in messages if "error" in m and m["error"]["code"] == -32700]
    assert parse_errors, "malformed line should produce a -32700 Parse error response"


def test_refund_happy_path_and_business_error(session):
    messages, _ = session
    ok = _by_id(messages, 8)["result"]
    assert ok.get("isError") is not True
    assert ok["structuredContent"]["status"] == "processed"
    assert ok["structuredContent"]["new_balance"] == 1240.0

    biz = _by_id(messages, 9)["result"]
    assert biz["isError"] is True
    assert "INSUFFICIENT_BALANCE" in biz["content"][0]["text"]
