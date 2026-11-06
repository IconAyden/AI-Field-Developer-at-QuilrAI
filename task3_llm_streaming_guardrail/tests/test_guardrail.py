from __future__ import annotations

import json
import random
import socket
import threading
import time
from typing import Any

import httpx
import pytest
import uvicorn

from task3_llm_streaming_guardrail import mock_provider
from task3_llm_streaming_guardrail.gateway import GatewayConfig, SSEParser, create_app
from task3_llm_streaming_guardrail.redactor import REDACTED, StreamRedactor, luhn_ok, redact_text

# --------------------------------------------------------------------------- #
# One-shot redaction                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected",
    [
        ("mail me at john.doe@example.com please", f"mail me at {REDACTED} please"),
        ("x+tag@sub.domain.co.uk,", f"{REDACTED},"),
        ("SSN 123-45-6789.", f"SSN {REDACTED}."),
        ("card 4111 1111 1111 1111 ok", f"card {REDACTED} ok"),
        ("card 4111-1111-1111-1111 ok", f"card {REDACTED} ok"),
        ("card 4111111111111111 ok", f"card {REDACTED} ok"),
        ("amex 3782 822463 10005 ok", f"amex {REDACTED} ok"),
        ("two: a@b.io and 123-45-6789", f"two: {REDACTED} and {REDACTED}"),
        ("no pii here, 42 apples", "no pii here, 42 apples"),
        ("order 123456789012345 is not a card", "order 123456789012345 is not a card"),  # fails Luhn
        ("phone 555-123-4567", "phone 555-123-4567"),  # not an SSN shape
        ("id 1234-56-7890", "id 1234-56-7890"),  # 4-2-4 is not an SSN
        ("version 1.2.3@build", "version 1.2.3@build"),  # no TLD -> not an email
        ("", ""),
    ],
)
def test_redact_text(text, expected):
    assert redact_text(text) == expected


def test_luhn():
    assert luhn_ok("4111111111111111")
    assert luhn_ok("378282246310005")
    assert not luhn_ok("4111111111111112")
    assert not luhn_ok("123456789012345")


# --------------------------------------------------------------------------- #
# Streaming redaction: the chunk boundary must not matter                     #
# --------------------------------------------------------------------------- #
SAMPLES = [
    "Contact john.doe@example.com today.",
    "Her SSN is 123-45-6789, keep it safe.",
    "Card: 4111 1111 1111 1111 expires soon",
    "Card 4111-1111-1111-1111 and mail a@b.co",
    mock_provider.DEFAULT_TEXT,
    "Email at the very end: final@domain.org",
    "123-45-6789",
    "a@b.io",
    "Nothing sensitive in this sentence at all!",
    "Order 123456789012345 shipped. Card 5555 5555 5555 4444 charged.",
]


def _stream(text: str, cuts: list[int], **kw) -> str:
    r = StreamRedactor(**kw)
    out = ""
    prev = 0
    for c in cuts + [len(text)]:
        out += r.feed(text[prev:c])
        prev = c
    return out + r.flush()


@pytest.mark.parametrize("text", SAMPLES)
def test_every_single_split_point_matches_one_shot(text):
    expected = redact_text(text)
    for cut in range(len(text) + 1):
        assert _stream(text, [cut]) == expected, f"split at {cut}: {text[:cut]!r} | {text[cut:]!r}"


@pytest.mark.parametrize("text", SAMPLES)
def test_random_multi_way_splits_match_one_shot(text):
    expected = redact_text(text)
    rng = random.Random(1234)
    for _ in range(200):
        k = rng.randint(1, min(12, len(text)))
        cuts = sorted(rng.sample(range(len(text) + 1), k))
        assert _stream(text, cuts) == expected, f"cuts {cuts}"


def test_character_at_a_time():
    text = mock_provider.DEFAULT_TEXT
    assert _stream(text, list(range(1, len(text)))) == redact_text(text)


def test_prose_is_released_at_the_next_space():
    r = StreamRedactor()
    assert r.feed("Hello") == ""  # could be the start of hello@...
    assert r.feed(" world") == "Hello "  # the space after a letter is a boundary
    assert r.feed(", ") == "world, "
    assert r.flush() == ""


def test_digits_followed_by_space_are_held_for_card_groups():
    r = StreamRedactor()
    assert r.feed("pay 4111 ") == "pay "
    assert r.feed("1111 1111 ") == ""
    assert r.feed("1111 now") == ""  # 'now' continues the run (letters are in the alphabet)
    assert r.feed(".") == ""  # '.' is legal inside an email, still held
    assert r.feed("! ") == f"{REDACTED} now.! "  # '!' is a hard boundary


def test_holdback_is_bounded():
    r = StreamRedactor(max_holdback=32)
    run = "a" * 500  # a single 500-char token with no boundary
    out = r.feed(run)
    assert len(out) == 500 - 32
    assert r.held == 32
    assert r.peak_held <= 500
    out2 = r.feed(" tail")
    assert out2 == "a" * 32 + " "
    assert r.flush() == "tail"


def test_peak_memory_independent_of_stream_length():
    r = StreamRedactor(max_holdback=64)
    for _ in range(2000):
        r.feed("some ordinary words with numbers 42 and emails a@b.io ")
    assert r.peak_held <= 64 + 60  # one chunk plus the cap, never the whole stream


def test_flush_redacts_trailing_pii():
    r = StreamRedactor()
    assert r.feed("reach me: bob@corp.io") == "reach me: "
    assert r.flush() == REDACTED


# --------------------------------------------------------------------------- #
# SSE parser                                                                  #
# --------------------------------------------------------------------------- #
def test_sse_parser_reassembles_split_events():
    p = SSEParser()
    raw = b'data: {"a": 1}\n\ndata: [DONE]\n\n'
    got = []
    for i in range(len(raw)):
        got += p.feed(raw[i : i + 1])
    assert [e.data for e in got] == ['{"a": 1}', "[DONE]"]


def test_sse_parser_handles_crlf_comments_and_multiline_data():
    p = SSEParser()
    events = p.feed(b": keepalive\r\n\r\nevent: delta\r\nid: 7\r\ndata: line1\r\ndata: line2\r\n\r\n")
    assert len(events) == 1
    assert events[0].event == "delta" and events[0].id == "7" and events[0].data == "line1\nline2"


# --------------------------------------------------------------------------- #
# Gateway over real sockets (httpx's ASGI transport buffers bodies, so a real  #
# server is needed to observe streaming behaviour).                            #
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
            time.sleep(0.02)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture(scope="module")
def stack():
    prov_port, gw_port = _free_port(), _free_port()
    cfg = GatewayConfig(provider_url=f"http://127.0.0.1:{prov_port}/v1/chat/completions")
    with _ServerThread(mock_provider.create_app(), prov_port), _ServerThread(create_app(cfg), gw_port):
        yield f"http://127.0.0.1:{gw_port}"


def _collect_stream(base: str, body: dict[str, Any]) -> tuple[list[dict], list[float], bool, float]:
    """Return (chunk events, arrival times relative to request start, saw_done, total time)."""
    events: list[dict] = []
    times: list[float] = []
    saw_done = False
    t0 = time.perf_counter()
    with httpx.Client(timeout=30) as c, c.stream("POST", f"{base}/v1/chat/completions", json=body) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers.get("x-accel-buffering") == "no"
        parser = SSEParser()
        for raw in r.iter_bytes():
            for ev in parser.feed(raw):
                if ev.data == "[DONE]":
                    saw_done = True
                    continue
                events.append(json.loads(ev.data))
                times.append(time.perf_counter() - t0)
    return events, times, saw_done, time.perf_counter() - t0


def _content(events: list[dict]) -> str:
    return "".join((e["choices"][0].get("delta") or {}).get("content") or "" for e in events)


def test_streaming_redacts_pii_split_across_chunks(stack):
    events, _, saw_done, _ = _collect_stream(stack, {"model": "mock-1", "stream": True, "messages": []})
    text = _content(events)
    assert text == redact_text(mock_provider.DEFAULT_TEXT)
    assert "john.doe@example.com" not in text
    assert "123-45-6789" not in text
    assert "4111" not in text
    assert "123456789012345" in text  # Luhn-invalid order number is kept
    assert saw_done
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert len(events) > 5, "response must be streamed as multiple events, not buffered into one"


def test_streaming_with_adversarial_chunking(stack):
    chunks = ["The email is j", "ohn.d", "oe@", "exa", "mple.c", "om and SSN 1", "23-4", "5-6789 and card 4", "111 1", "111 1111 11", "11, done"]
    events, _, _, _ = _collect_stream(stack, {"model": "mock-1", "stream": True, "mock_chunks": chunks})
    assert _content(events) == f"The email is {REDACTED} and SSN {REDACTED} and card {REDACTED}, done"


def test_stream_is_progressive_not_buffered(stack):
    """With the provider pausing 120 ms between 8 chunks, the first redacted event must arrive long before the end."""
    chunks = ["Hello ", "there ", "friend, ", "contact ", "a@b.io ", "now ", "or ", "later."]
    events, times, _, total = _collect_stream(
        stack, {"model": "mock-1", "stream": True, "mock_chunks": chunks, "mock_delay_ms": 120}
    )
    assert _content(events) == f"Hello there friend, contact {REDACTED} now or later."
    assert total >= 0.8, "provider delays should dominate total time"
    first_content_time = next(t for e, t in zip(events, times) if (e["choices"][0]["delta"].get("content")))
    assert first_content_time < total / 2, f"TTFT {first_content_time:.3f}s vs total {total:.3f}s: gateway is buffering"
    # consecutive content events should be spaced out, i.e. relayed as they arrive
    content_times = [t for e, t in zip(events, times) if e["choices"][0]["delta"].get("content")]
    assert len(content_times) >= 5
    assert content_times[-1] - content_times[0] > 0.4


def test_non_streaming_path_is_redacted(stack):
    r = httpx.post(f"{stack}/v1/chat/completions", json={"model": "mock-1", "stream": False, "messages": []}, timeout=10)
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content == redact_text(mock_provider.DEFAULT_TEXT)


def test_provider_down_is_sanitized_502():
    cfg = GatewayConfig(provider_url=f"http://127.0.0.1:{_free_port()}/v1/chat/completions", connect_timeout=1.0)
    port = _free_port()
    with _ServerThread(create_app(cfg), port):
        r = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={"stream": True}, timeout=10)
    assert r.status_code == 502
    assert "127.0.0.1" not in r.text and "Traceback" not in r.text
    assert r.json()["error"]["type"] == "gateway_error"


def test_bad_json_is_400(stack):
    r = httpx.post(f"{stack}/v1/chat/completions", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400
