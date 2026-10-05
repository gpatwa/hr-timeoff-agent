"""Who is signed in.

The app depends only on `IdentityProvider`. `PersonaSwitcher` is the
implementation for synthetic data: pick a person from the directory, no
password, and the choice is carried in a signed cookie. Replacing it with real
sign-in (OIDC/SAML mapped to worker ids) changes this file only; every
permission check downstream already keys on the worker id it returns.

`OIDCIdentity` is that replacement: the authorization-code flow with PKCE against
an OpenID Connect provider. It only establishes who someone is (a verified email
that names one worker in this tenant); what they may do still comes from the
directory and the graph.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Protocol
from urllib.parse import urlencode

import httpx
from fastapi import Request, Response
from fastapi.responses import RedirectResponse

COOKIE = "hr_persona"


class IdentityProvider(Protocol):
    kind: str   # "persona" (demo picker) or "oidc"

    def current(self, request: Request) -> str | None:
        """The signed-in worker id, or None."""

    def sign_in(self, response: Response, worker_id: str) -> None: ...

    def sign_out(self, response: Response) -> None: ...


class PersonaSwitcher:
    """Demo identity for synthetic data. The cookie is HMAC-signed so it can't be
    edited to impersonate someone; the secret is per process unless set."""

    kind = "persona"

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


SESSION = "hr_session"
LOGIN_TX = "hr_login"
SESSION_TTL_S = 8 * 3600
LOGIN_TTL_S = 600


class ConfigError(RuntimeError):
    pass


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class OIDCIdentity:
    """Sign in with OpenID Connect (authorization code + PKCE), session in a signed cookie.

    Routes it adds: GET /login (to the IdP), GET /auth/callback (back from it).
    The session cookie holds only a worker id and an expiry, signed with
    HR_WEB_SECRET; that secret is required here, because a per-process random
    one would end every session on restart and break a second instance.
    """

    kind = "oidc"

    def __init__(self, cfg, workspace, *, secret: str | None = None, provider=None, verifier=None,
                 http: httpx.Client | None = None, now=time.time):
        from ...adapters.oidc import Provider, TokenVerifier

        key = secret or os.environ.get("HR_WEB_SECRET")
        if not key:
            raise ConfigError("HR_WEB_SECRET must be set when HR_OIDC_ISSUER is: it signs the session cookie.")
        self._key = key.encode()
        self.cfg, self.ws, self._now = cfg, workspace, now
        self._http = http or httpx.Client(timeout=5.0)
        self.provider = provider or Provider(cfg, self._http)
        self.verifier = verifier or TokenVerifier(cfg, self.provider)
        self.secure = cfg.public_url.startswith("https://")

    # signed, expiring cookie values

    def _seal(self, purpose: str, payload: dict, ttl: int) -> str:
        body = _b64(json.dumps({**payload, "p": purpose, "exp": int(self._now()) + ttl}, separators=(",", ":")).encode())
        return f"{body}.{_b64(hmac.new(self._key, body.encode(), hashlib.sha256).digest())}"

    def _unseal(self, raw: str | None, purpose: str) -> dict | None:
        body, _, mac = (raw or "").partition(".")
        if not body or not hmac.compare_digest(mac, _b64(hmac.new(self._key, body.encode(), hashlib.sha256).digest())):
            return None
        try:
            payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        except ValueError:
            return None
        # The purpose keeps a session cookie from standing in for a login-attempt cookie, and back.
        return payload if payload.get("p") == purpose and payload.get("exp", 0) > self._now() else None

    # IdentityProvider

    def current(self, request: Request) -> str | None:
        session = self._unseal(request.cookies.get(SESSION), "session")
        return session.get("w") if session else None

    def sign_in(self, response: Response, worker_id: str) -> None:
        response.set_cookie(SESSION, self._seal("session", {"w": worker_id}, SESSION_TTL_S), httponly=True,
                            samesite="lax", secure=self.secure, max_age=SESSION_TTL_S)

    def sign_out(self, response: Response) -> None:
        response.delete_cookie(SESSION)

    def logout_url(self) -> str | None:
        end = self.provider.front_endpoint("end_session_endpoint")
        if not end:
            return None
        return end + "?" + urlencode({"client_id": self.cfg.client_id, "post_logout_redirect_uri": f"{self.cfg.public_url}/signin"})

    # the authorization-code flow

    def login(self) -> RedirectResponse:
        state, nonce, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(24), secrets.token_urlsafe(48)
        challenge = _b64(hashlib.sha256(verifier.encode()).digest())
        url = self.provider.front_endpoint("authorization_endpoint") + "?" + urlencode({
            "response_type": "code", "client_id": self.cfg.client_id, "redirect_uri": f"{self.cfg.public_url}/auth/callback",
            "scope": "openid email profile", "state": state, "nonce": nonce,
            "code_challenge": challenge, "code_challenge_method": "S256",
        })
        response = RedirectResponse(url, status_code=303)
        response.set_cookie(LOGIN_TX, self._seal("login", {"s": state, "n": nonce, "v": verifier}, LOGIN_TTL_S),
                            httponly=True, samesite="lax", secure=self.secure, max_age=LOGIN_TTL_S)
        return response

    def complete(self, request: Request) -> str:
        """Finish the flow from the callback request. Returns the worker id, or raises
        `SignInFailed` with a message that is safe to show."""
        from ...adapters.oidc import InvalidToken, Unauthorized, caller_from_claims

        tx = self._unseal(request.cookies.get(LOGIN_TX), "login")
        q = request.query_params
        if q.get("error"):
            raise SignInFailed(f"The identity provider refused the sign-in ({q['error']}).", "provider_refused")
        if not tx or not q.get("state") or not hmac.compare_digest(str(tx["s"]), q["state"]):
            raise SignInFailed("This sign-in attempt is not valid or has expired. Start again.", "invalid_attempt")
        if not q.get("code"):
            raise SignInFailed("The identity provider returned no code.", "invalid_attempt")
        data = {
            "grant_type": "authorization_code", "code": q["code"], "redirect_uri": f"{self.cfg.public_url}/auth/callback",
            "client_id": self.cfg.client_id, "code_verifier": tx["v"],
        }
        if self.cfg.client_secret:
            data["client_secret"] = self.cfg.client_secret
        try:
            r = self._http.post(self.provider.endpoint("token_endpoint"), data=data)
            r.raise_for_status()
            id_token = r.json()["id_token"]
            claims = self.verifier.verify(id_token, audience=self.cfg.client_id)
        except (httpx.HTTPError, KeyError, ValueError, InvalidToken):
            raise SignInFailed("The sign-in could not be completed.", "token_exchange")
        if not hmac.compare_digest(str(claims.get("nonce", "")), str(tx["n"])):
            raise SignInFailed("This sign-in attempt is not valid. Start again.", "invalid_attempt")
        try:
            caller = caller_from_claims(claims, tenant_id=self.ws.tenant_id(), directory=self.ws.worker_id_for_email)
        except Unauthorized as exc:
            raise SignInFailed(f"Signed in, but not allowed here: {exc}.", exc.code)
        if caller.kind != "user":
            raise SignInFailed("A service account cannot sign in to the web app.", "service_account")
        return caller.name


class SignInFailed(Exception):
    """Safe to show to the person signing in. `code` is a fixed label for metrics."""

    def __init__(self, message: str, code: str = "failed"):
        super().__init__(message)
        self.code = code


def identity_from_env(workspace) -> IdentityProvider:
    from ...adapters.oidc import OIDCConfig

    cfg = OIDCConfig.from_env()
    return OIDCIdentity(cfg, workspace) if cfg else PersonaSwitcher()
