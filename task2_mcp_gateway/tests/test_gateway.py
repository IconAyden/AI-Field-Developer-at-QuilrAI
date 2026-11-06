"""Gateway tests.

Most tests run the gateway and the downstream mock in-process via httpx's
ASGI transport (fast, deterministic). The final test boots both on real TCP
sockets with uvicorn to prove the HTTP forwarding works end to end.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import httpx
import pytest
import uvicorn

from task2_mcp_gateway import downstream_mock
from task2_mcp_gateway.auth import Principal, TokenStore
from task2_mcp_gateway.gateway import GatewayConfig, create_app
from task2_mcp_gateway.policy import UNAUTHORIZED_TOOL_CALL, decide, Forward, Reject

pytestmark = pytest.mark.anyio

ADMIN = "Bearer admin-token-123"
VIEWER = "Bearer viewer-token-456"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def clear_downstream_calls():
    downstream_mock.CALLS.clear()
    yield
    downstream_mock.CALLS.clear()


@pytest.fixture
def store() -> TokenStore:
    return TokenStore(
        {
            "admin-token-123": Principal(subject="alice", role="admin"),
            "viewer-token-456": Principal(subject="bob", role="viewer"),
        }
    )


def _make_gateway_client(store: TokenStore, downstream_app, **cfg_kwargs) -> httpx.AsyncClient:
    downstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=downstream_app), base_url="http://downstream"
    )
    cfg = GatewayConfig(downstream_url="http://downstream/mcp", token_store=store, **cfg_kwargs)
    app = create_app(cfg, http_client=downstream_client)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")


@pytest.fixture
async def client(store):
    c = _make_gateway_client(store, downstream_mock.create_app())
    async with c:
        yield c


def rpc(id_: Any, method: str, params: dict | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if id_ is not None:
        msg["id"] = id_
    if params is not None:
        msg["params"] = params
    return msg


def call(id_: Any, name: str, arguments: dict | None = None) -> dict[str, Any]:
    return rpc(id_, "tools/call", {"name": name, "arguments": arguments or {}})


async def post(client: httpx.AsyncClient, body: Any, auth: str | None = VIEWER, **kw) -> httpx.Response:
    headers = {"content-type": "application/json"}
    if auth is not None:
        headers["authorization"] = auth
    content = body if isinstance(body, bytes) else json.dumps(body)
    return await client.post("/mcp", content=content, headers=headers, **kw)


# --------------------------------------------------------------------------- #
# Policy unit tests (pure)                                                    #
# --------------------------------------------------------------------------- #
def test_policy_forwards_tools_list_for_everyone():
    for role in ("admin", "viewer"):
        assert isinstance(decide(rpc(1, "tools/list"), Principal("x", role)), Forward)


def test_policy_blocks_admin_tool_for_viewer():
    d = decide(call(1, "admin_reset_key"), Principal("bob", "viewer"))
    assert isinstance(d, Reject)
    assert d.error.code == UNAUTHORIZED_TOOL_CALL
    assert d.error.data == {"tool": "admin_reset_key", "required_role": "admin", "role": "viewer"}


def test_policy_allows_admin_tool_for_admin():
    assert isinstance(decide(call(1, "admin_reset_key"), Principal("alice", "admin")), Forward)


def test_policy_prefix_is_exact():
    viewer = Principal("bob", "viewer")
    assert isinstance(decide(call(1, "administer"), viewer), Forward)  # no underscore
    assert isinstance(decide(call(1, "Admin_reset"), viewer), Forward)  # case-sensitive by spec
    assert isinstance(decide(call(1, "not_admin_tool"), viewer), Forward)  # prefix, not substring
    assert isinstance(decide(call(1, "admin_"), viewer), Reject)


def test_policy_rejects_malformed_tools_call():
    viewer = Principal("bob", "viewer")
    assert isinstance(decide(rpc(1, "tools/call"), viewer), Reject)
    assert isinstance(decide(rpc(1, "tools/call", {"name": 42}), viewer), Reject)
    assert isinstance(decide(rpc(1, "tools/call", {"name": ""}), viewer), Reject)
    assert decide(rpc(1, "tools/call", {"name": 42}), viewer).error.code == -32602


# --------------------------------------------------------------------------- #
# Authentication                                                              #
# --------------------------------------------------------------------------- #
async def test_missing_token_is_401(client):
    r = await post(client, rpc(1, "tools/list"), auth=None)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == -32002
    assert downstream_mock.CALLS == []


@pytest.mark.parametrize("auth", ["Bearer", "Bearer ", "Basic admin-token-123", "admin-token-123", "Bearer nope"])
async def test_bad_tokens_are_401(client, auth):
    r = await post(client, rpc(1, "tools/list"), auth=auth)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == -32002


async def test_bearer_scheme_is_case_insensitive(client):
    r = await post(client, rpc(1, "tools/list"), auth="bearer admin-token-123")
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
# tools/list and tools/call forwarding                                        #
# --------------------------------------------------------------------------- #
async def test_tools_list_forwarded_transparently(client):
    r = await post(client, rpc(1, "tools/list"), auth=VIEWER)
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 1
    names = [t["name"] for t in body["result"]["tools"]]
    assert "admin_reset_key" in names, "default mode forwards the listing unmodified"


async def test_tools_list_filtered_when_enabled(store):
    async with _make_gateway_client(store, downstream_mock.create_app(), filter_tool_list=True) as c:
        viewer = (await post(c, rpc(1, "tools/list"), auth=VIEWER)).json()
        admin = (await post(c, rpc(1, "tools/list"), auth=ADMIN)).json()
    assert not any(t["name"].startswith("admin_") for t in viewer["result"]["tools"])
    assert any(t["name"].startswith("admin_") for t in admin["result"]["tools"])


async def test_viewer_can_call_normal_tool(client):
    r = await post(client, call(7, "get_weather", {"city": "Lisbon"}), auth=VIEWER)
    assert r.status_code == 200
    assert "Lisbon" in r.json()["result"]["content"][0]["text"]
    assert downstream_mock.CALLS == [{"name": "get_weather", "arguments": {"city": "Lisbon"}}]


async def test_viewer_blocked_from_admin_tool_and_downstream_never_called(client):
    r = await post(client, call("req-9", "admin_reset_key", {"tenant": "acme"}), auth=VIEWER)
    assert r.status_code == 403
    body = r.json()
    assert body == {
        "jsonrpc": "2.0",
        "id": "req-9",
        "error": {
            "code": -32001,
            "message": "Unauthorized Tool Call",
            "data": {"tool": "admin_reset_key", "required_role": "admin", "role": "viewer"},
        },
    }
    assert downstream_mock.CALLS == [], "blocked call must not reach the downstream"


async def test_admin_can_call_admin_tool(client):
    r = await post(client, call(2, "admin_reset_key", {"tenant": "acme"}), auth=ADMIN)
    assert r.status_code == 200
    assert "rotated" in r.json()["result"]["content"][0]["text"]
    assert downstream_mock.CALLS[-1]["name"] == "admin_reset_key"


async def test_other_methods_are_forwarded(client):
    r = await post(client, rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}}), auth=VIEWER)
    assert r.json()["result"]["serverInfo"]["name"] == "downstream-mock"
    r = await post(client, rpc(2, "ping"), auth=VIEWER)
    assert r.json()["result"] == {}
    r = await post(client, rpc(3, "resources/list"), auth=VIEWER)
    assert r.json()["error"]["code"] == -32601  # downstream's answer, passed through


async def test_notification_passes_through_with_202(client):
    r = await post(client, rpc(None, "notifications/initialized"), auth=VIEWER)
    assert r.status_code == 202


async def test_client_token_not_leaked_downstream(store):
    seen: dict[str, str] = {}

    async def spy_app(scope, receive, send):
        if scope["type"] == "lifespan":
            return
        seen.update({k.decode(): v.decode() for k, v in scope["headers"]})
        app = downstream_mock.create_app()
        await app(scope, receive, send)

    async with _make_gateway_client(store, spy_app, downstream_token="svc-secret") as c:
        await post(c, rpc(1, "tools/list"), auth=VIEWER)
    assert seen["authorization"] == "Bearer svc-secret"
    assert seen["x-gateway-subject"] == "bob"
    assert seen["x-gateway-role"] == "viewer"
    assert "x-forwarded-for" in seen


# --------------------------------------------------------------------------- #
# Wire-format edge cases                                                      #
# --------------------------------------------------------------------------- #
async def test_malformed_json_is_400_parse_error(client):
    r = await post(client, b'{"jsonrpc": "2.0", "id": 1, "method": ', auth=VIEWER)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32700


@pytest.mark.parametrize(
    "body",
    [
        {"id": 1, "method": "tools/list"},  # missing jsonrpc
        {"jsonrpc": "1.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 1},  # missing method
        {"jsonrpc": "2.0", "id": 1, "method": 5},
        {"jsonrpc": "2.0", "id": {"nested": 1}, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": "string"},
        "just a string",
        42,
        [],
    ],
)
async def test_invalid_envelopes_are_400(client, body):
    r = await post(client, body, auth=ADMIN)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32600
    assert downstream_mock.CALLS == []


async def test_tools_call_without_name_is_invalid_params(client):
    r = await post(client, rpc(1, "tools/call", {"arguments": {}}), auth=ADMIN)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32602


async def test_batch_splices_denials_and_forwarded_results_in_order(client):
    batch = [
        call(1, "get_weather", {"city": "Oslo"}),
        call(2, "admin_delete_user", {"user_id": "u1"}),
        rpc(None, "notifications/progress"),
        rpc(3, "tools/list"),
        call(4, "admin_reset_key"),
        {"jsonrpc": "2.0", "id": 5},  # invalid envelope inside a batch
    ]
    r = await post(client, batch, auth=VIEWER)
    assert r.status_code == 200
    answers = r.json()
    assert [a["id"] for a in answers] == [1, 2, 3, 4, 5]
    assert "Oslo" in answers[0]["result"]["content"][0]["text"]
    assert answers[1]["error"]["code"] == -32001
    assert "tools" in answers[2]["result"]
    assert answers[3]["error"]["code"] == -32001
    assert answers[4]["error"]["code"] == -32600
    assert [c["name"] for c in downstream_mock.CALLS] == ["get_weather"]


async def test_batch_entirely_denied_never_hits_downstream(client):
    r = await post(client, [call(1, "admin_reset_key"), call(2, "admin_delete_user")], auth=VIEWER)
    assert r.status_code == 200
    assert [a["error"]["code"] for a in r.json()] == [-32001, -32001]
    assert downstream_mock.CALLS == []


async def test_batch_of_only_notifications_is_202(client):
    r = await post(client, [rpc(None, "notifications/initialized")], auth=VIEWER)
    assert r.status_code == 202


async def test_oversized_body_is_413(client):
    big = json.dumps(call(1, "echo", {"text": "x" * (1024 * 1024 + 1)})).encode()
    r = await post(client, big, auth=ADMIN)
    assert r.status_code == 413


# --------------------------------------------------------------------------- #
# Downstream failure handling                                                 #
# --------------------------------------------------------------------------- #
async def test_downstream_unreachable_is_502_without_internal_detail(store):
    # A port that was just released: the OS refuses the connection outright.
    cfg = GatewayConfig(downstream_url=f"http://127.0.0.1:{_free_port()}/mcp", token_store=store, timeout_seconds=2.0)
    app = create_app(cfg, http_client=httpx.AsyncClient(timeout=1.0))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as c:
        r = await post(c, rpc(1, "tools/list"), auth=ADMIN)
    assert r.status_code == 502
    body = r.json()
    assert body["error"]["code"] == -32003
    assert "127.0.0.1" not in json.dumps(body)
    assert "Traceback" not in r.text


async def test_downstream_timeout_is_504(store):
    async def slow_app(scope, receive, send):
        if scope["type"] == "lifespan":
            return
        import anyio

        await anyio.sleep(2)

    downstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=slow_app), base_url="http://downstream", timeout=0.2
    )
    cfg = GatewayConfig(downstream_url="http://downstream/mcp", token_store=store, timeout_seconds=0.2)
    app = create_app(cfg, http_client=downstream_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as c:
        r = await post(c, rpc(1, "tools/list"), auth=ADMIN)
    assert r.status_code == 504
    assert r.json()["error"]["code"] == -32003


# --------------------------------------------------------------------------- #
# Real sockets, end to end                                                    #
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ServerThread:
    def __init__(self, app, port: int) -> None:
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)


def test_end_to_end_over_tcp(store):
    down_port, gw_port = _free_port(), _free_port()
    cfg = GatewayConfig(downstream_url=f"http://127.0.0.1:{down_port}/mcp", token_store=store)
    with _ServerThread(downstream_mock.create_app(), down_port), _ServerThread(create_app(cfg), gw_port):
        base = f"http://127.0.0.1:{gw_port}/mcp"
        with httpx.Client(timeout=5) as c:
            denied = c.post(base, json=call(1, "admin_reset_key"), headers={"authorization": VIEWER})
            allowed = c.post(base, json=call(2, "admin_reset_key", {"tenant": "t"}), headers={"authorization": ADMIN})
            listing = c.post(base, json=rpc(3, "tools/list"), headers={"authorization": VIEWER})
            unauth = c.post(base, json=rpc(4, "tools/list"))
    assert denied.status_code == 403 and denied.json()["error"]["code"] == -32001
    assert allowed.status_code == 200 and "rotated" in allowed.json()["result"]["content"][0]["text"]
    assert listing.status_code == 200 and len(listing.json()["result"]["tools"]) == 4
    assert unauth.status_code == 401
    assert [c["name"] for c in downstream_mock.CALLS] == ["admin_reset_key"]
