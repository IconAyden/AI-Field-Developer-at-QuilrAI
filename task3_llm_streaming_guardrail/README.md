# Task 3: LLM gateway streaming guardrail (PII redaction)

An LLM gateway endpoint that forwards chat-completion requests to an OpenAI-compatible provider and streams the answer back while replacing emails, SSNs and credit-card numbers with `[REDACTED]` **in flight**, even when a value is split across several stream chunks.

```
client --POST /v1/chat/completions--> [ gateway :9300 ] --> provider (SSE)
       <------- SSE, PII redacted ----
```

## Run

```bash
python -m task3_llm_streaming_guardrail.mock_provider     # terminal 1, :9200
python -m task3_llm_streaming_guardrail.gateway           # terminal 2, :9300

curl -N localhost:9300/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"mock-1","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

The mock provider streams a sentence containing an email, an SSN, a Luhn-valid card number and a 15-digit order number, in chunks of 4 to 8 characters so every sensitive value straddles a boundary. The gateway output contains `[REDACTED]` for the first three and leaves the order number alone. Point the gateway at a real provider with `PROVIDER_URL=https://api.openai.com/v1/chat/completions PROVIDER_API_KEY=sk-...`.

Testing knobs the mock accepts in the request body: `mock_text`, `mock_chunks` (explicit chunk list), `mock_delay_ms` (pause per chunk).

## Test

```bash
python -m pytest task3_llm_streaming_guardrail -q
```

The central test is a property: for each sample string, **every possible split point** (and 200 random multi-way splits, and character-at-a-time) must produce exactly the same output as one-shot redaction of the whole string. The streaming tests run the gateway and provider on real TCP sockets, because httpx's in-process ASGI transport buffers response bodies and would hide buffering bugs. One test has the provider pause 120 ms between chunks and asserts the first redacted event arrives well before the stream ends, and that events arrive spaced out rather than in one burst.

## Design

### The holdback window (`redactor.py`)

Regex on each chunk alone misses `"john.d" | "oe@exa" | "mple.com"`. Buffering the whole reply finds it but kills latency and uses unbounded memory. The redactor does neither:

- Every pattern it redacts is composed of characters from a small alphabet: letters, digits, `. _ % + - @`, plus a space or dash *between digits* for card groups.
- Any character outside that alphabet (a space after a letter, a comma, a newline...) can never be inside a match, so it is a **safe cut point**: everything before it is scanned and released immediately. Only the trailing run of alphabet characters is held.
- The held tail is capped at `max_holdback` (64 chars). A longer unbroken run is force-released from the front. Memory is O(cap + chunk), independent of response length (asserted by `test_peak_memory_independent_of_stream_length`).

Latency consequences: in prose, text is released at the next space, so time-to-first-token is the first token itself (`"Hello "` is released at once; `"Hello"` waits for the next boundary). Digit runs wait a little longer because `4111 ` may be the first group of a card.

Credit-card candidates (13 to 19 digits with optional single spaces/dashes) are **Luhn-validated** before redaction, so order numbers, timestamps and tracking ids are not eaten. SSNs require the exact 3-2-4 shape with no adjacent digits or dashes.

### Stream plumbing (`gateway.py`)

- The provider response is consumed with `aiter_bytes` and parsed by a small incremental **SSE parser** that holds at most one partial event (handles `\n\n` and `\r\n\r\n`, comments, multi-line `data:`).
- Each delta's `content` goes through the redactor. If any text is released, a new chunk event is emitted with the same `id`/`model`/`created` and the redacted text; role-only, finish-only and usage events pass through untouched. On `finish_reason` or `[DONE]` the redactor is flushed first, so no held text is lost, and `[DONE]` is forwarded last.
- The response is a `StreamingResponse` with `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no`, so uvicorn and any nginx in front flush per event.
- Per-stream metrics (events in/out, TTFT, total, peak holdback) are logged at the end.
- Non-streaming requests (`stream: false`) are redacted in one shot on `choices[].message.content`.
- Provider errors are summarized (`429` stays `429`, everything else becomes `502` / `504`); provider error pages are never streamed through.

### Known limits (deliberate)

- The email regex accepts the common local-part characters, not the full RFC 5322 set (`!#$&'*/=?^{|}~` are treated as boundaries).
- A single value longer than `max_holdback` with no boundary could be split by the forced release. 64 covers realistic emails and the longest formatted card number (23 chars); raise `MAX_HOLDBACK` if your domain needs it.
- Redaction operates on the assistant's text only; tool-call arguments in deltas are passed through unchanged.

## Layout

```
task3_llm_streaming_guardrail/
  redactor.py        Patterns, Luhn check, StreamRedactor with bounded holdback
  gateway.py         SSE parser, streaming proxy endpoint
  mock_provider.py   OpenAI-compatible SSE provider with controllable chunking
  tests/
```
