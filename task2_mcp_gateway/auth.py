"""Bearer-token authentication and role lookup.

Tokens are opaque strings mapped to a principal (subject + role). In
production this would be a JWT verification or an introspection call; the
interface (``authenticate(authorization_header) -> Principal``) is what the
gateway depends on, so swapping the backing store does not touch the proxy.

Comparison uses ``hmac.compare_digest`` so a timing side-channel cannot be
used to guess tokens character by character.
"""

from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass
from typing import Literal

Role = Literal["admin", "viewer"]
VALID_ROLES: frozenset[str] = frozenset({"admin", "viewer"})


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    role: Role


class AuthError(Exception):
    """Raised when a request cannot be attributed to a principal."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class TokenStore:
    def __init__(self, tokens: dict[str, Principal]) -> None:
        self._tokens = tokens

    @classmethod
    def from_env(cls, var: str = "GATEWAY_TOKENS") -> TokenStore:
        """Load ``{"<token>": {"subject": "...", "role": "admin|viewer"}, ...}`` from JSON.

        Falls back to two demo tokens so the gateway runs out of the box.
        """
        raw = os.environ.get(var)
        if not raw:
            return cls(
                {
                    "admin-token-123": Principal(subject="alice", role="admin"),
                    "viewer-token-456": Principal(subject="bob", role="viewer"),
                }
            )
        parsed = json.loads(raw)
        tokens: dict[str, Principal] = {}
        for token, entry in parsed.items():
            role = entry["role"]
            if role not in VALID_ROLES:
                raise ValueError(f"token for {entry.get('subject')!r} has unknown role {role!r}")
            tokens[token] = Principal(subject=entry["subject"], role=role)
        return cls(tokens)

    def lookup(self, presented: str) -> Principal | None:
        # Constant-time scan over every token: do not short-circuit on the first byte mismatch.
        found: Principal | None = None
        for token, principal in self._tokens.items():
            if hmac.compare_digest(token.encode(), presented.encode()):
                found = principal
        return found


def authenticate(authorization_header: str | None, store: TokenStore) -> Principal:
    if not authorization_header:
        raise AuthError("missing Authorization header")
    scheme, _, credentials = authorization_header.strip().partition(" ")
    if scheme.lower() != "bearer" or not credentials.strip():
        raise AuthError("Authorization header must be 'Bearer <token>'")
    principal = store.lookup(credentials.strip())
    if principal is None:
        raise AuthError("unknown or revoked token")
    return principal
