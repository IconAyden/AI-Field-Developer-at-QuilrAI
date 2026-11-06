"""Strict input/output schemas for the two tools.

Design notes
------------
* ``extra="forbid"``: an unexpected field is a protocol error, not something to
  silently drop. ``{"customer_id": ..., "amont": 5}`` must fail loudly.
* ``strict=True``: no type coercion. ``"amount": "25.00"`` (a string) is rejected
  rather than parsed, because a model that sends the wrong JSON type has not
  understood the schema and should be told so.
* Error messages are written for the model that will read them: they say what
  the rule is, not just that it was broken.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CUSTOMER_ID_PATTERN = r"^CUST-\d{5}$"
_CUSTOMER_ID_RE = re.compile(CUSTOMER_ID_PATTERN)

MIN_REASON_LENGTH = 10
MAX_REASON_LENGTH = 2_000
MAX_REFUND_AMOUNT = 1_000_000.0


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GetCustomerRecordInput(_StrictModel):
    customer_id: str = Field(
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier in the form CUST-XXXXX, where X is a digit (e.g. CUST-00042).",
        examples=["CUST-00042"],
    )


class TriggerRefundInput(_StrictModel):
    customer_id: str = Field(
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier in the form CUST-XXXXX, where X is a digit (e.g. CUST-00042).",
        examples=["CUST-00042"],
    )
    amount: float = Field(
        gt=0,
        le=MAX_REFUND_AMOUNT,
        allow_inf_nan=False,
        description="Refund amount in USD. Must be positive, finite, and have at most 2 decimal places.",
        examples=[25.0],
    )
    reason: str = Field(
        min_length=MIN_REASON_LENGTH,
        max_length=MAX_REASON_LENGTH,
        description=f"Why the refund is being issued. At least {MIN_REASON_LENGTH} characters after trimming whitespace.",
        examples=["Duplicate charge on 2025-10-01 invoice"],
    )

    @field_validator("amount")
    @classmethod
    def _at_most_two_decimals(cls, v: float) -> float:
        # Decimal(str(v)) avoids binary-float artefacts such as 0.1 -> 0.1000000000000000055
        if Decimal(str(v)).as_tuple().exponent < -2:
            raise ValueError("amount may have at most 2 decimal places")
        return v

    @field_validator("reason")
    @classmethod
    def _reason_has_substance(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < MIN_REASON_LENGTH:
            raise ValueError(
                f"reason must be at least {MIN_REASON_LENGTH} characters after trimming whitespace "
                f"(got {len(stripped)})"
            )
        return stripped


class CustomerRecord(BaseModel):
    customer_id: str = Field(pattern=CUSTOMER_ID_PATTERN)
    name: str
    email: str
    tier: Literal["free", "pro", "enterprise"]
    balance: float
    currency: Literal["USD"] = "USD"
    created_at: str


class RefundReceipt(BaseModel):
    refund_id: str
    customer_id: str = Field(pattern=CUSTOMER_ID_PATTERN)
    amount: float
    currency: Literal["USD"] = "USD"
    status: Literal["processed"] = "processed"
    new_balance: float
    processed_at: str


def is_valid_customer_id(value: str) -> bool:
    return _CUSTOMER_ID_RE.fullmatch(value) is not None
