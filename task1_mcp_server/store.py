"""In-memory stand-in for the customer system of record.

Business failures (unknown customer, refund larger than balance) are raised as
``BusinessError``. The server turns those into tool results with
``is_error=True``: the request was well-formed, the tool ran, and the outcome
is something the model should read and reason about. Protocol errors
(malformed input) never get this far; see ``server.py``.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Literal

from .schemas import CustomerRecord, RefundReceipt

BusinessErrorCode = Literal["CUSTOMER_NOT_FOUND", "INSUFFICIENT_BALANCE"]


class BusinessError(Exception):
    def __init__(self, code: BusinessErrorCode, message: str) -> None:
        super().__init__(message)
        self.code: BusinessErrorCode = code
        self.message = message


class CustomerStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._customers: dict[str, CustomerRecord] = {
            "CUST-00001": CustomerRecord(
                customer_id="CUST-00001",
                name="Ada Lovelace",
                email="ada@example.com",
                tier="enterprise",
                balance=1250.50,
                created_at="2024-01-15T09:30:00Z",
            ),
            "CUST-00042": CustomerRecord(
                customer_id="CUST-00042",
                name="Grace Hopper",
                email="grace@example.com",
                tier="pro",
                balance=80.00,
                created_at="2024-06-02T14:05:00Z",
            ),
            "CUST-12345": CustomerRecord(
                customer_id="CUST-12345",
                name="Linus Torvalds",
                email="linus@example.com",
                tier="free",
                balance=0.00,
                created_at="2025-03-21T18:45:00Z",
            ),
        }

    def get(self, customer_id: str) -> CustomerRecord:
        with self._lock:
            rec = self._customers.get(customer_id)
            if rec is None:
                raise BusinessError("CUSTOMER_NOT_FOUND", f"No customer with id {customer_id}")
            return rec.model_copy()

    def refund(self, customer_id: str, amount: float, reason: str) -> RefundReceipt:
        with self._lock:
            rec = self._customers.get(customer_id)
            if rec is None:
                raise BusinessError("CUSTOMER_NOT_FOUND", f"No customer with id {customer_id}")
            if amount > rec.balance:
                raise BusinessError(
                    "INSUFFICIENT_BALANCE",
                    f"Refund of {amount:.2f} exceeds available balance of {rec.balance:.2f}",
                )
            rec.balance = round(rec.balance - amount, 2)
            del reason  # a real system would persist it against the ledger entry
            return RefundReceipt(
                refund_id=f"rf_{uuid.uuid4().hex}",
                customer_id=customer_id,
                amount=amount,
                new_balance=rec.balance,
                processed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )
