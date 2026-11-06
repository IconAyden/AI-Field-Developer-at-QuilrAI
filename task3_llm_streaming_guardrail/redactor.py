"""Streaming PII redactor.

The problem: a token stream delivers ``"john.d"``, ``"oe@exa"``, ``"mple.com"``
as three chunks. Regex on each chunk alone finds nothing. Buffering the whole
response finds it but destroys streaming latency and uses unbounded memory.

The approach: keep only the *tail* of the stream that could still be the
beginning of a sensitive pattern, and emit everything before it immediately.

Every pattern we redact (emails, SSNs, card numbers) is made of characters from
a small alphabet: letters, digits and ``. _ % + - @``, plus a space or dash
between digits in card numbers. A character outside that alphabet (a plain
space after a letter, a comma, a newline, ...) can never be inside a match, so
it is a safe cut point: text before it can be scanned and released. The
held-back tail is bounded (``max_holdback``) so memory and latency stay
constant regardless of response length.

Latency characteristics
-----------------------
* Text is released at the first boundary character after it, which in prose is
  the next space, so Time-To-First-Token is typically the first token itself.
* The worst case is a single unbroken run of alphabet characters, held up to
  ``max_holdback`` characters before a forced release.
* Memory is O(max_holdback + chunk size), independent of response length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

REDACTED = "[REDACTED]"

# --------------------------------------------------------------------------- #
# Patterns                                                                    #
# --------------------------------------------------------------------------- #
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*\.[A-Za-z]{2,}")
SSN_RE = re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])")
# 13-19 digits, optionally separated by single spaces or dashes. Validated with Luhn
# before redaction, so an order number or a timestamp is not eaten by mistake.
CARD_RE = re.compile(r"(?<![\dA-Za-z])\d(?:[ -]?\d){12,18}(?![\dA-Za-z])")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card_replacement(m: re.Match[str]) -> str:
    digits = re.sub(r"[ -]", "", m.group(0))
    return REDACTED if luhn_ok(digits) else m.group(0)


@dataclass(frozen=True, slots=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    replacement: str | Callable[[re.Match[str]], str] = REDACTED


DEFAULT_PATTERNS: tuple[Pattern, ...] = (
    Pattern("email", EMAIL_RE),
    Pattern("credit_card", CARD_RE, _card_replacement),
    Pattern("ssn", SSN_RE),
)


def redact_text(text: str, patterns: tuple[Pattern, ...] = DEFAULT_PATTERNS) -> str:
    """One-shot redaction of a complete string."""
    for p in patterns:
        text = p.regex.sub(p.replacement, text)
    return text


# --------------------------------------------------------------------------- #
# Streaming                                                                   #
# --------------------------------------------------------------------------- #
# Characters that can appear inside any of the patterns above.
_ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._%+-@")


def _holdback_start(buf: str) -> int:
    """Index where the 'could still be part of a match' tail begins.

    Scans backwards from the end. A character is part of the tail if it is in
    the pattern alphabet, or if it is a space immediately preceded by a digit
    (a card number group separator whose next group has not arrived yet).
    """
    i = len(buf)
    while i > 0:
        ch = buf[i - 1]
        if ch in _ALPHABET:
            i -= 1
        elif ch == " " and i >= 2 and buf[i - 2].isdigit():
            i -= 1
        else:
            break
    return i


@dataclass(slots=True)
class StreamRedactor:
    """Feed text chunks in, get redacted text out, with bounded holdback.

    >>> r = StreamRedactor()
    >>> r.feed("Contact john.d") + r.feed("oe@exa") + r.feed("mple.com today.") + r.flush()
    'Contact [REDACTED] today.'
    """

    patterns: tuple[Pattern, ...] = DEFAULT_PATTERNS
    max_holdback: int = 64
    _buf: str = field(default="", init=False, repr=False)
    _held_chars_peak: int = field(default=0, init=False, repr=False)

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        cut = _holdback_start(self._buf)

        if len(self._buf) - cut > self.max_holdback:
            # An unbroken run longer than any pattern we expect. Force-release the
            # oldest part, keeping the last max_holdback characters in case a
            # pattern is still forming at the very end.
            cut = len(self._buf) - self.max_holdback

        if cut == 0:
            self._held_chars_peak = max(self._held_chars_peak, len(self._buf))
            return ""

        safe, self._buf = self._buf[:cut], self._buf[cut:]
        self._held_chars_peak = max(self._held_chars_peak, len(self._buf))
        return redact_text(safe, self.patterns)

    def flush(self) -> str:
        """End of stream: scan and release whatever is still held."""
        out, self._buf = redact_text(self._buf, self.patterns), ""
        return out

    @property
    def held(self) -> int:
        return len(self._buf)

    @property
    def peak_held(self) -> int:
        return self._held_chars_peak
