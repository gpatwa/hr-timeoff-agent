"""Who is signed in.

The app depends only on `IdentityProvider`. `PersonaSwitcher` is the
implementation for synthetic data: pick a person from the directory, no
password, and the choice is carried in a signed cookie. Replacing it with real
sign-in (OIDC/SAML mapped to worker ids) changes this file only; every
permission check downstream already keys on the worker id it returns.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Protocol

from fastapi import Request, Response

COOKIE = "hr_persona"


class IdentityProvider(Protocol):
    def current(self, request: Request) -> str | None:
        """The signed-in worker id, or None."""

    def sign_in(self, response: Response, worker_id: str) -> None: ...

    def sign_out(self, response: Response) -> None: ...


class PersonaSwitcher:
    """Demo identity for synthetic data. The cookie is HMAC-signed so it can't be
    edited to impersonate someone; the secret is per process unless set."""

    def __init__(self, secret: str | None = None):
        self._key = (secret or os.environ.get("HR_WEB_SECRET") or secrets.token_hex(32)).encode()

    def _sign(self, worker_id: str) -> str:
        mac = hmac.new(self._key, worker_id.encode(), hashlib.sha256).hexdigest()
        return f"{worker_id}.{mac}"

    def current(self, request: Request) -> str | None:
        raw = request.cookies.get(COOKIE, "")
        worker_id, _, mac = raw.rpartition(".")
        if worker_id and hmac.compare_digest(self._sign(worker_id), raw):
            return worker_id
        return None

    def sign_in(self, response: Response, worker_id: str) -> None:
        response.set_cookie(COOKIE, self._sign(worker_id), httponly=True, samesite="strict")

    def sign_out(self, response: Response) -> None:
        response.delete_cookie(COOKIE)
