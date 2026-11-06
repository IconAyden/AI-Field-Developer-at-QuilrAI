"""Standardized gateway error payload.

Every error the gateway returns, whatever its origin, has this shape::

    {
      "error": {
        "code": "rate_limited",          # stable, machine-readable
        "message": "Token budget exhausted for this key",   # human-readable, no internals
        "request_id": "req_7f3c...",     # correlate with gateway logs
        "retryable": true,
        "retry_after_seconds": 12.5,     # optional
        "details": {...}                 # optional, curated, never raw upstream data
      }
    }

What never appears: stack traces, exception text, upstream hostnames or URLs,
upstream response bodies, internal model names unless the client chose them.
Those go to the log, keyed by ``request_id``.
"""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.details = details

    def to_payload(self, request_id: str) -> dict[str, Any]:
        err: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "request_id": request_id,
            "retryable": self.retryable,
        }
        if self.retry_after_seconds is not None:
            err["retry_after_seconds"] = round(self.retry_after_seconds, 3)
        if self.details:
            err["details"] = self.details
        return {"error": err}


def internal_error_payload(request_id: str) -> dict[str, Any]:
    """For unexpected exceptions: say nothing about what went wrong."""
    return {
        "error": {
            "code": "internal_error",
            "message": "The gateway encountered an internal error",
            "request_id": request_id,
            "retryable": True,
        }
    }
