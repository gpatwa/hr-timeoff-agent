"""Real sign-in: OpenID Connect tokens, checked and mapped to the HR directory.

The identity provider (Keycloak here; anything that issues OIDC tokens) answers
exactly one question: who is this? It never decides what they may do. Roles,
manager relationships and approval rights stay in the tenant's own data, so the
guarantees in the graph (only the direct manager decides) hold whatever the IdP
claims. Three things are checked on every token before it means anything:

  signature + issuer + audience + expiry   (RS256 only, keys fetched from the IdP's JWKS)
  tenant      the token's tenant_id claim must be this deployment's tenant
  directory   a person needs a verified email that names exactly one worker

A token with no email that belongs to one of the configured service clients is a
service caller (the time-off agent calling the payroll agent), also bound to a tenant.

Everything is opt-in through HR_OIDC_ISSUER. Unset, the app keeps its demo
persona picker and static bearer tokens, so the offline tests and a laptop demo
need no identity provider.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

import httpx
import jwt

ALGORITHMS = ["RS256"]
LEEWAY_S = 30
HTTP_TIMEOUT_S = 5.0


class InvalidToken(Exception):
    """The token is not acceptable. The message is safe to log, not to show."""


class Unauthorized(Exception):
    """The token is genuine, but it does not identify someone this tenant knows.
    `code` is a short, fixed label for metrics (never the person's details)."""

    def __init__(self, message: str, code: str = "unauthorized"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class OIDCConfig:
    issuer: str
    client_id: str                      # the web app's client; its id tokens are for this audience
    client_secret: str | None
    audience: str                       # what access tokens presented to the A2A agents must name
    public_url: str                     # where browsers reach this app (the redirect URI's origin)
    internal_url: str | None            # where this process reaches the IdP, when that differs (docker)
    service_clients: frozenset[str]     # client ids that may call as a service

    @classmethod
    def from_env(cls, env: dict | None = None) -> "OIDCConfig | None":
        env = os.environ if env is None else env
        issuer = (env.get("HR_OIDC_ISSUER") or "").rstrip("/")
        if not issuer:
            return None
        return cls(
            issuer=issuer,
            client_id=env.get("HR_OIDC_CLIENT_ID", "hr-web"),
            client_secret=env.get("HR_OIDC_CLIENT_SECRET") or None,
            audience=env.get("HR_OIDC_AUDIENCE", "hr-a2a"),
            public_url=(env.get("HR_PUBLIC_URL") or "http://localhost:8000").rstrip("/"),
            internal_url=(env.get("HR_OIDC_INTERNAL_URL") or "").rstrip("/") or None,
            service_clients=frozenset(c for c in env.get("HR_OIDC_SERVICE_CLIENTS", "timeoff-agent").split(",") if c),
        )

    def backchannel(self, url: str) -> str:
        """An endpoint the IdP advertised, as this process should call it."""
        if self.internal_url and url.startswith(self.issuer):
            return self.internal_url + url[len(self.issuer):]
        return url


class Provider:
    """The IdP's discovery document, fetched once and refreshed hourly."""

    def __init__(self, cfg: OIDCConfig, http: httpx.Client | None = None, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self._http = http or httpx.Client(timeout=HTTP_TIMEOUT_S)
        self._clock = clock
        self._doc: dict | None = None
        self._at = 0.0
        self._lock = threading.Lock()

    def metadata(self) -> dict:
        with self._lock:
            if self._doc is None or self._clock() - self._at > 3600:
                base = self.cfg.internal_url or self.cfg.issuer
                r = self._http.get(f"{base}/.well-known/openid-configuration")
                r.raise_for_status()
                doc = r.json()
                if doc.get("issuer") != self.cfg.issuer:
                    raise InvalidToken(f"the identity provider says it is {doc.get('issuer')!r}, not {self.cfg.issuer!r}")
                self._doc, self._at = doc, self._clock()
            return self._doc

    def endpoint(self, name: str) -> str:
        return self.cfg.backchannel(self.metadata()[name])

    def front_endpoint(self, name: str) -> str | None:
        """An endpoint for the browser: exactly as advertised."""
        return self.metadata().get(name)


class TokenVerifier:
    """Checks a JWT's signature, issuer, audience and lifetime. Nothing else."""

    def __init__(self, cfg: OIDCConfig, provider: Provider | None = None, key_for: Callable[[str], object] | None = None):
        self.cfg = cfg
        self.provider = provider or Provider(cfg)
        self._key_for = key_for
        self._jwks: jwt.PyJWKClient | None = None

    def _key(self, token: str):
        if self._key_for is not None:
            return self._key_for(token)
        if self._jwks is None:
            uri = self.provider.endpoint("jwks_uri")
            self._jwks = jwt.PyJWKClient(uri, cache_keys=True, lifespan=300, timeout=HTTP_TIMEOUT_S)
        return self._jwks.get_signing_key_from_jwt(token).key

    def verify(self, token: str, *, audience: str) -> dict:
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") not in ALGORITHMS:   # never "none", never an HMAC keyed by a public key
                raise InvalidToken(f"algorithm {header.get('alg')!r} is not accepted")
            return jwt.decode(
                token, self._key(token), algorithms=ALGORITHMS, audience=audience, issuer=self.cfg.issuer,
                leeway=LEEWAY_S, options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except InvalidToken:
            raise
        except jwt.PyJWTError as exc:
            raise InvalidToken(f"{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise InvalidToken(f"identity provider unreachable: {type(exc).__name__}") from exc


@dataclass(frozen=True)
class Caller:
    kind: str                 # "user" | "service"
    name: str                 # worker id for a user, client id for a service
    tenant_id: str
    subject: str


def caller_from_claims(claims: dict, *, tenant_id: str, directory: Callable[[str], str | None],
                       service_clients: frozenset[str] | set[str] = frozenset()) -> Caller:
    """Who a verified token is, in this tenant's terms, or Unauthorized.

    `directory(email)` returns the one worker id for that email, or None.
    """
    if claims.get("tenant_id") != tenant_id:
        raise Unauthorized(f"the token is for tenant {claims.get('tenant_id')!r}, not {tenant_id!r}", "wrong_tenant")
    client = claims.get("azp") or claims.get("client_id")
    email = claims.get("email")
    if not email and client in service_clients:
        return Caller("service", client, tenant_id, claims["sub"])
    if not email or claims.get("email_verified") is not True:
        raise Unauthorized("the token has no verified email to match to a worker", "unverified_email")
    worker_id = directory(email)
    if worker_id is None:
        raise Unauthorized("this account is not a worker in this tenant", "not_in_directory")
    return Caller("user", worker_id, tenant_id, claims["sub"])


class BearerGuard:
    """ASGI middleware: no valid bearer, no request. For an HTTP server that has no auth of its own.

    `authenticate(authorization_header)` returns something truthy for an accepted caller.
    """

    def __init__(self, app, authenticate: Callable[[str | None], object]):
        self.app, self.authenticate = app, authenticate

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
            if not self.authenticate(headers.get("authorization")):
                body = b'{"error":"A valid bearer token is required."}'
                await send({"type": "http.response.start", "status": 401, "headers": [
                    (b"content-type", b"application/json"), (b"www-authenticate", b"Bearer"), (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


# ── tokens this process presents to others ──────────────────────────────────

class ServiceTokens:
    """Client-credentials access tokens for this service, cached until shortly before they expire."""

    def __init__(self, cfg: OIDCConfig, client_id: str, client_secret: str, *, provider: Provider | None = None,
                 http: httpx.Client | None = None, clock: Callable[[], float] = time.monotonic):
        self.cfg, self.client_id, self.client_secret = cfg, client_id, client_secret
        self.provider = provider or Provider(cfg)
        self._http = http or httpx.Client(timeout=HTTP_TIMEOUT_S)
        self._clock = clock
        self._token: str | None = None
        self._expires = 0.0
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            if self._token is None or self._clock() >= self._expires - 30:
                r = self._http.post(self.provider.endpoint("token_endpoint"), data={
                    "grant_type": "client_credentials", "client_id": self.client_id, "client_secret": self.client_secret,
                })
                r.raise_for_status()
                body = r.json()
                self._token = body["access_token"]
                self._expires = self._clock() + float(body.get("expires_in", 60))
            return self._token


class BearerAuth(httpx.Auth):
    """Sets the Authorization header from a fixed token or a function that returns a fresh one."""

    def __init__(self, token: str | Callable[[], str]):
        self._token = token

    def auth_flow(self, request):
        token = self._token() if callable(self._token) else self._token
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


def password_grant(cfg: OIDCConfig, username: str, password: str, *, client_id: str = "hr-cli",
                   provider: Provider | None = None, http: httpx.Client | None = None) -> str:
    """A user's access token by password, for scripts and tests against a development realm.
    The production flow is the browser's authorization-code flow."""
    provider = provider or Provider(cfg)
    r = (http or httpx.Client(timeout=HTTP_TIMEOUT_S)).post(provider.endpoint("token_endpoint"), data={
        "grant_type": "password", "client_id": client_id, "username": username, "password": password, "scope": "openid",
    })
    r.raise_for_status()
    return r.json()["access_token"]
