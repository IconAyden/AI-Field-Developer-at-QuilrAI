from __future__ import annotations

import threading
from pathlib import Path

import pytest

from task4_model_router.rate_limiter import SlidingWindowRateLimiter, estimate_tokens


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def limiter(tmp_path: Path, clock: FakeClock):
    lim = SlidingWindowRateLimiter(tmp_path / "rl.db", limit_tokens=50_000, window_seconds=60, clock=clock)
    yield lim
    lim.close()


def test_db_is_on_disk(tmp_path, clock):
    path = tmp_path / "on-disk.db"
    lim = SlidingWindowRateLimiter(path, clock=clock)
    lim.try_reserve("a", 10)
    lim.close()
    assert path.exists() and path.stat().st_size > 0


def test_allows_up_to_limit_then_denies(limiter):
    assert limiter.try_reserve("t1", 20_000).allowed
    assert limiter.try_reserve("t1", 20_000).allowed
    d = limiter.try_reserve("t1", 10_001)
    assert not d.allowed
    assert d.used_before == 40_000 and d.requested == 10_001 and d.limit == 50_000
    assert d.reservation_id is None
    assert limiter.try_reserve("t1", 10_000).allowed  # exactly fills the budget
    assert not limiter.try_reserve("t1", 1).allowed


def test_denied_request_consumes_nothing(limiter):
    limiter.try_reserve("t1", 45_000)
    limiter.try_reserve("t1", 10_000)  # denied
    assert limiter.usage("t1") == 45_000


def test_tenants_are_isolated(limiter):
    limiter.try_reserve("t1", 50_000)
    assert not limiter.try_reserve("t1", 1).allowed
    assert limiter.try_reserve("t2", 50_000).allowed


def test_window_slides_and_evicts(limiter, clock):
    limiter.try_reserve("t1", 50_000)
    clock.advance(59.9)
    assert not limiter.try_reserve("t1", 1).allowed, "still inside the window"
    clock.advance(0.2)  # t = 60.1 s after the first reservation
    assert limiter.try_reserve("t1", 50_000).allowed, "old usage has slid out"
    # Eviction happened as part of the reservation: only the new row remains.
    assert limiter.row_count() == 1


def test_partial_expiry(limiter, clock):
    limiter.try_reserve("t1", 30_000)  # t=0
    clock.advance(30)
    limiter.try_reserve("t1", 20_000)  # t=30
    clock.advance(31)  # t=61: the 30k row is out, the 20k row remains
    assert limiter.usage("t1") == 20_000
    assert limiter.try_reserve("t1", 30_000).allowed
    assert not limiter.try_reserve("t1", 1).allowed


def test_retry_after_points_at_oldest_usage_that_frees_enough(limiter, clock):
    limiter.try_reserve("t1", 20_000)  # expires at t=60
    clock.advance(10)
    limiter.try_reserve("t1", 20_000)  # expires at t=70
    clock.advance(10)
    limiter.try_reserve("t1", 10_000)  # expires at t=80  (now full, t=20)
    clock.advance(5)  # t=25
    d = limiter.try_reserve("t1", 15_000)
    assert not d.allowed
    # Need to free 15k: the first row (20k) expiring at t=60 is enough -> 35 s from now.
    assert d.retry_after_seconds == pytest.approx(35.0)
    d2 = limiter.try_reserve("t1", 25_000)
    # Need 25k: rows 1+2 (40k) -> second expires at t=70 -> 45 s.
    assert d2.retry_after_seconds == pytest.approx(45.0)


def test_retry_after_for_request_larger_than_limit(limiter):
    d = limiter.try_reserve("t1", 60_000)
    assert not d.allowed and d.retry_after_seconds == 60


def test_settle_replaces_estimate_with_actual(limiter):
    d = limiter.try_reserve("t1", 10_000)
    assert limiter.usage("t1") == 10_000
    limiter.settle(d.reservation_id, 1_234)
    assert limiter.usage("t1") == 1_234
    assert limiter.try_reserve("t1", 48_000).allowed


def test_release_refunds_reservation(limiter):
    d = limiter.try_reserve("t1", 50_000)
    assert not limiter.try_reserve("t1", 1).allowed
    limiter.release(d.reservation_id)
    assert limiter.usage("t1") == 0
    assert limiter.try_reserve("t1", 50_000).allowed


def test_state_persists_across_instances(tmp_path, clock):
    path = tmp_path / "persist.db"
    a = SlidingWindowRateLimiter(path, limit_tokens=100, clock=clock)
    a.try_reserve("t1", 80)
    a.close()
    b = SlidingWindowRateLimiter(path, limit_tokens=100, clock=clock)
    assert b.usage("t1") == 80
    assert not b.try_reserve("t1", 30).allowed
    b.close()


def test_evict_expired_across_tenants(limiter, clock):
    limiter.try_reserve("a", 1)
    limiter.try_reserve("b", 1)
    clock.advance(61)
    limiter.try_reserve("c", 1)
    assert limiter.evict_expired() == 2
    assert limiter.row_count() == 1


def test_concurrent_reservations_never_exceed_limit(tmp_path, clock):
    """50 threads x 2000 tokens against a 50k budget: exactly 25 may win."""
    lim = SlidingWindowRateLimiter(tmp_path / "conc.db", limit_tokens=50_000, clock=clock)
    results: list[bool] = []
    lock = threading.Lock()
    start = threading.Barrier(50)

    def worker():
        start.wait()
        d = lim.try_reserve("t1", 2_000)
        with lock:
            results.append(d.allowed)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 25
    assert lim.usage("t1") == 50_000
    lim.close()


def test_concurrent_across_separate_connections(tmp_path, clock):
    """Two limiter instances (as two processes would have) on the same file stay consistent."""
    path = tmp_path / "multi.db"
    lims = [SlidingWindowRateLimiter(path, limit_tokens=10_000, clock=clock) for _ in range(4)]
    results: list[bool] = []
    lock = threading.Lock()
    start = threading.Barrier(40)

    def worker(lim):
        start.wait()
        d = lim.try_reserve("t1", 1_000)
        with lock:
            results.append(d.allowed)

    threads = [threading.Thread(target=worker, args=(lims[i % 4],)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 10
    for lim in lims:
        lim.close()


def test_invalid_args(tmp_path):
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(tmp_path / "x.db", limit_tokens=0)
    lim = SlidingWindowRateLimiter(tmp_path / "y.db")
    with pytest.raises(ValueError):
        lim.try_reserve("t", -1)
    lim.close()


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 100}, 200),
        ({"messages": [{"role": "user", "content": "hi"}]}, 1 + 256),  # default ceiling
        ({"messages": [{"role": "user", "content": [{"type": "text", "text": "a" * 40}]}], "max_tokens": 0}, 10),
        ({"messages": [], "max_tokens": "not a number"}, 256),
        ({"prompt": "a" * 8, "messages": [], "max_completion_tokens": 5}, 7),
    ],
)
def test_estimate_tokens(payload, expected):
    assert estimate_tokens(payload) == expected
