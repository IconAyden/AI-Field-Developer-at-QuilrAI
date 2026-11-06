"""Token-aware sliding-window rate limiter backed by on-disk SQLite.

Model
-----
Each tenant has a budget of ``limit_tokens`` per trailing ``window_seconds``
(default 50,000 tokens / 60 s). Usage is a *sliding log*: one row per request
with its timestamp and token count. Usage at time ``t`` is the sum of rows with
``ts > t - window``. A log (rather than fixed buckets) gives exact sliding
semantics: a burst at 12:00:59 still counts at 12:01:58.

Reserve / settle
----------------
Token usage is only known after the model answers, but the limit must be
enforced before the request is sent. So:

1. ``try_reserve(tenant, estimate)`` atomically checks the window and inserts a
   row for the *estimate* (prompt size + ``max_tokens``). Denied requests
   insert nothing.
2. ``settle(reservation_id, actual)`` replaces the estimate with the provider's
   reported ``usage.total_tokens`` once the response arrives.
3. ``release(reservation_id)`` deletes the row if the request failed before any
   tokens were consumed (every provider was down), so the tenant is not charged
   for the gateway's failure.

Atomicity
---------
``try_reserve`` runs inside ``BEGIN IMMEDIATE``, which takes SQLite's write
lock before the SELECT, so two concurrent reservations cannot both read the
same usage and both succeed. This holds across threads *and* across processes
sharing the file, which is why SQLite was chosen over an in-memory dict.

Eviction
--------
Rows older than the window are deleted on every reservation for that tenant
(cheap: it is an indexed range delete) and by ``evict_expired`` for everyone.
Rows are never counted by mistake even before eviction, because every read is
bounded by ``ts > now - window``.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant  TEXT    NOT NULL,
    ts      REAL    NOT NULL,
    tokens  INTEGER NOT NULL CHECK (tokens >= 0),
    settled INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS token_usage_tenant_ts ON token_usage (tenant, ts);
"""


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    tenant: str
    requested: int
    used_before: int
    limit: int
    window_seconds: float
    reservation_id: int | None = None
    retry_after_seconds: float | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used_before - (self.requested if self.allowed else 0))


class SlidingWindowRateLimiter:
    def __init__(
        self,
        db_path: str | Path,
        *,
        limit_tokens: int = 50_000,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if limit_tokens <= 0 or window_seconds <= 0:
            raise ValueError("limit_tokens and window_seconds must be positive")
        self.db_path = str(db_path)
        self.limit = limit_tokens
        self.window = float(window_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None, timeout=5.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ #
    def try_reserve(self, tenant: str, tokens: int) -> Decision:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        now = self._clock()
        cutoff = now - self.window
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM token_usage WHERE tenant = ? AND ts <= ?", (tenant, cutoff))
                (used,) = self._conn.execute(
                    "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND ts > ?",
                    (tenant, cutoff),
                ).fetchone()
                if used + tokens > self.limit:
                    retry_after = self._retry_after(tenant, cutoff, now, used, tokens)
                    self._conn.execute("COMMIT")
                    return Decision(
                        allowed=False,
                        tenant=tenant,
                        requested=tokens,
                        used_before=used,
                        limit=self.limit,
                        window_seconds=self.window,
                        retry_after_seconds=retry_after,
                    )
                cur = self._conn.execute(
                    "INSERT INTO token_usage (tenant, ts, tokens) VALUES (?, ?, ?)", (tenant, now, tokens)
                )
                self._conn.execute("COMMIT")
                return Decision(
                    allowed=True,
                    tenant=tenant,
                    requested=tokens,
                    used_before=used,
                    limit=self.limit,
                    window_seconds=self.window,
                    reservation_id=cur.lastrowid,
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def _retry_after(self, tenant: str, cutoff: float, now: float, used: int, requested: int) -> float:
        """Seconds until enough of the oldest usage has slid out of the window.

        Walks the tenant's rows oldest-first, subtracting each until the request
        would fit. If it can never fit (requested > limit), returns the full window.
        """
        if requested > self.limit:
            return self.window
        need_to_free = used + requested - self.limit
        freed = 0
        for ts, tokens in self._conn.execute(
            "SELECT ts, tokens FROM token_usage WHERE tenant = ? AND ts > ? ORDER BY ts ASC", (tenant, cutoff)
        ):
            freed += tokens
            if freed >= need_to_free:
                return max(0.0, ts + self.window - now)
        return self.window

    def settle(self, reservation_id: int, actual_tokens: int) -> None:
        """Replace the estimate with what the provider actually billed."""
        if actual_tokens < 0:
            raise ValueError("actual_tokens must be non-negative")
        with self._lock:
            self._conn.execute(
                "UPDATE token_usage SET tokens = ?, settled = 1 WHERE id = ?", (actual_tokens, reservation_id)
            )

    def release(self, reservation_id: int) -> None:
        """The request never consumed tokens (all providers failed): refund the estimate."""
        with self._lock:
            self._conn.execute("DELETE FROM token_usage WHERE id = ?", (reservation_id,))

    def usage(self, tenant: str) -> int:
        cutoff = self._clock() - self.window
        with self._lock:
            (used,) = self._conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND ts > ?", (tenant, cutoff)
            ).fetchone()
        return int(used)

    def evict_expired(self) -> int:
        """Delete every row outside the window, for all tenants. Returns rows removed."""
        cutoff = self._clock() - self.window
        with self._lock:
            cur = self._conn.execute("DELETE FROM token_usage WHERE ts <= ?", (cutoff,))
            return cur.rowcount

    def row_count(self) -> int:
        with self._lock:
            (n,) = self._conn.execute("SELECT COUNT(*) FROM token_usage").fetchone()
        return int(n)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def estimate_tokens(payload: dict) -> int:
    """Pre-flight estimate: ~4 chars per token for the prompt, plus the requested output ceiling.

    Over-estimating is the safe direction: the estimate is only a reservation
    and is replaced by the provider's real count on settle.
    """
    chars = 0
    for m in payload.get("messages") or []:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):  # multimodal parts
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chars += len(part["text"])
    if isinstance(payload.get("prompt"), str):
        chars += len(payload["prompt"])
    prompt_tokens = (chars + 3) // 4
    max_tokens = payload.get("max_tokens")
    if max_tokens is None:
        max_tokens = payload.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = 256
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = 256
    return prompt_tokens + max(0, max_tokens)
