"""Validation matrix via the official SDK client connected in-process.

Fast (no subprocess) and exhaustive: every edge case the schema is meant to
catch, plus the business-error path and the happy path.
"""

from __future__ import annotations

from typing import Any

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

from task1_mcp_server.server import build_server
from task1_mcp_server.store import CustomerStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with Client(build_server(CustomerStore())) as c:
        yield c


async def _expect_invalid_params(client: Client, tool: str, args: dict[str, Any], field: str) -> MCPError:
    with pytest.raises(MCPError) as info:
        await client.call_tool(tool, args)
    err = info.value
    assert err.code == -32602, f"expected -32602 for {args!r}, got {err.code}: {err.message}"
    fields = {i["field"] for i in err.data["issues"]}
    assert field in fields, f"expected issue on {field!r}, got {fields}"
    return err


# --------------------------------------------------------------------------- #
# get_customer_record                                                         #
# --------------------------------------------------------------------------- #
async def test_get_customer_record_happy_path(client):
    res = await client.call_tool("get_customer_record", {"customer_id": "CUST-00042"})
    assert res.is_error is False
    assert res.structured_content["name"] == "Grace Hopper"


@pytest.mark.parametrize(
    "customer_id",
    [
        "cust-00042",  # lowercase prefix
        "CUST-0042",  # four digits
        "CUST-000420",  # six digits
        "CUST-ABCDE",  # letters
        "CUST_00042",  # wrong separator
        " CUST-00042",  # leading whitespace (strict: no trimming of ids)
        "CUST-00042\n",  # trailing newline
        "",
    ],
)
async def test_get_customer_record_rejects_bad_formats(client, customer_id):
    await _expect_invalid_params(client, "get_customer_record", {"customer_id": customer_id}, "customer_id")


@pytest.mark.parametrize("value", [42, None, ["CUST-00042"], {"id": "CUST-00042"}])
async def test_get_customer_record_rejects_wrong_types(client, value):
    await _expect_invalid_params(client, "get_customer_record", {"customer_id": value}, "customer_id")


async def test_get_customer_record_rejects_missing_and_extra_fields(client):
    await _expect_invalid_params(client, "get_customer_record", {}, "customer_id")
    await _expect_invalid_params(client, "get_customer_record", {"customer_id": "CUST-00042", "verbose": True}, "verbose")


async def test_get_customer_record_unknown_customer_is_tool_error_not_protocol_error(client):
    res = await client.call_tool("get_customer_record", {"customer_id": "CUST-99999"})
    assert res.is_error is True
    assert res.structured_content["error"]["code"] == "CUSTOMER_NOT_FOUND"


# --------------------------------------------------------------------------- #
# trigger_refund                                                              #
# --------------------------------------------------------------------------- #
VALID_REFUND = {"customer_id": "CUST-00001", "amount": 25.0, "reason": "Duplicate charge on invoice #123"}


async def test_trigger_refund_happy_path(client):
    res = await client.call_tool("trigger_refund", VALID_REFUND)
    assert res.is_error is False
    sc = res.structured_content
    assert sc["status"] == "processed"
    assert sc["amount"] == 25.0
    assert sc["new_balance"] == 1225.5
    assert sc["refund_id"].startswith("rf_")


async def test_trigger_refund_accepts_integer_amount(client):
    res = await client.call_tool("trigger_refund", {**VALID_REFUND, "amount": 10})
    assert res.is_error is False


@pytest.mark.parametrize(
    "amount",
    [0, -0.01, -100, 1_000_000.01, 0.001, 12.345, "25.00", None, True, float("inf"), float("nan")],
)
async def test_trigger_refund_rejects_bad_amounts(client, amount):
    await _expect_invalid_params(client, "trigger_refund", {**VALID_REFUND, "amount": amount}, "amount")


@pytest.mark.parametrize(
    "reason",
    ["", "too short", "123456789", "          ", "   short   ", "x" * 2001, None, 12345, ["a reason that is long"]],
)
async def test_trigger_refund_rejects_bad_reasons(client, reason):
    await _expect_invalid_params(client, "trigger_refund", {**VALID_REFUND, "reason": reason}, "reason")


async def test_trigger_refund_reason_exactly_ten_chars_ok(client):
    res = await client.call_tool("trigger_refund", {**VALID_REFUND, "reason": "0123456789"})
    assert res.is_error is False


async def test_trigger_refund_reports_all_issues_at_once(client):
    err = await _expect_invalid_params(
        client, "trigger_refund", {"customer_id": "nope", "amount": -1, "reason": "x"}, "customer_id"
    )
    assert {i["field"] for i in err.data["issues"]} == {"customer_id", "amount", "reason"}


async def test_trigger_refund_missing_fields(client):
    err = await _expect_invalid_params(client, "trigger_refund", {"customer_id": "CUST-00001"}, "amount")
    assert {i["field"] for i in err.data["issues"]} == {"amount", "reason"}


async def test_trigger_refund_extra_field_rejected(client):
    await _expect_invalid_params(client, "trigger_refund", {**VALID_REFUND, "currency": "EUR"}, "currency")


async def test_trigger_refund_insufficient_balance_is_tool_error(client):
    res = await client.call_tool("trigger_refund", {**VALID_REFUND, "customer_id": "CUST-12345", "amount": 1})
    assert res.is_error is True
    assert res.structured_content["error"]["code"] == "INSUFFICIENT_BALANCE"


async def test_unknown_tool_is_invalid_params(client):
    with pytest.raises(MCPError) as info:
        await client.call_tool("admin_delete_everything", {})
    assert info.value.code == -32602
    assert "known_tools" in info.value.data


async def test_tool_listing_exposes_strict_schemas(client):
    listing = await client.list_tools()
    by_name = {t.name: t for t in listing.tools}
    assert set(by_name) == {"get_customer_record", "trigger_refund"}
    schema = by_name["trigger_refund"].input_schema
    assert schema["additionalProperties"] is False
    assert schema["properties"]["amount"]["exclusiveMinimum"] == 0
    assert schema["properties"]["reason"]["minLength"] == 10
    assert schema["properties"]["customer_id"]["pattern"] == r"^CUST-\d{5}$"
    assert by_name["trigger_refund"].annotations.destructive_hint is True
    assert by_name["get_customer_record"].annotations.read_only_hint is True
