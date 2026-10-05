"""Real sign-in, against a fake identity provider that lives in the test.

Offline. The provider signs tokens with a key generated here and serves discovery,
JWKS and the token endpoint through an httpx mock transport, so the real client
code (PKCE, state, nonce, signature, issuer, audience, tenant, directory) is what
runs. tests/test_keycloak.py runs the same flows against a real Keycloak.

Run: .venv/bin/python tests/test_oidc.py
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")
for var in ("HR_OIDC_ISSUER", "HR_DATABASE_URL"):
    os.environ.pop(var, None)
logging.getLogger("a2a").setLevel(logging.ERROR)

import httpx  # noqa: E402
import jwt  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from hr_timeoff_agent.services.a2a.client import A2AAgent  # noqa: E402
from hr_timeoff_agent.services.a2a.common import OIDCBearer  # noqa: E402
from hr_timeoff_agent.services.a2a.payroll import create_payroll_app  # noqa: E402
from hr_timeoff_agent.services.a2a.server import create_timeoff_app  # noqa: E402
from hr_timeoff_agent.adapters.oidc import InvalidToken, OIDCConfig, Provider, ServiceTokens, TokenVerifier, Unauthorized, caller_from_claims
from hr_timeoff_agent.services.web.app import create_app  # noqa: E402
from hr_timeoff_agent.services.web.identity import ConfigError, OIDCIdentity  # noqa: E402

ISSUER = "https://idp.test/realms/hr"
CFG = OIDCConfig(issuer=ISSUER, client_id="hr-web", client_secret="web-secret", audience="hr-a2a",
                 public_url="http://testserver", internal_url=None, service_clients=frozenset({"timeoff-agent"}))
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
REVIEW = {"skill": "review_time_off_request", "request_id": "REQ-2004"}


def claims(**over) -> dict:
    now = int(time.time())
    base = {"iss": ISSUER, "aud": "hr-a2a", "sub": str(uuid.uuid4()), "iat": now, "exp": now + 300,
            "tenant_id": "TEN-001", "azp": "hr-cli", "email": "aiko.tanaka@acme.example", "email_verified": True}
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


def token(key=KEY, headers=None, alg="RS256", **over) -> str:
    return jwt.encode(claims(**over), key, algorithm=alg, headers={"kid": "k1", **(headers or {})})


def verifier() -> TokenVerifier:
    return TokenVerifier(CFG, key_for=lambda t: KEY.public_key())


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


# ── the token ───────────────────────────────────────────────────────────────

def test_a_valid_token_verifies():
    c = verifier().verify(token(), audience="hr-a2a")
    assert c["email"] == "aiko.tanaka@acme.example"


def test_every_way_a_token_can_be_wrong_is_refused():
    v = verifier()
    cases = {
        "expired": token(exp=int(time.time()) - 3600, iat=int(time.time()) - 7200),
        "wrong audience": token(aud="somebody-else"),
        "wrong issuer": token(iss="https://evil.test/realms/hr"),
        "signed by another key": token(key=OTHER_KEY),
        "no expiry": token(exp=None),
        "no subject": token(sub=None),
        "alg none": jwt.encode(claims(), None, algorithm="none"),
        "hmac (alg confusion)": jwt.encode(claims(), "x" * 64, algorithm="HS256"),
        "garbage": "not-a-jwt",
    }
    for name, t in cases.items():
        raises(InvalidToken, v.verify, t, audience="hr-a2a")
    good = token()
    head, body, sig = good.split(".")
    forged = base64.urlsafe_b64encode(json.dumps({**claims(), "email": "grace.kim@acme.example"}).encode()).rstrip(b"=").decode()
    raises(InvalidToken, v.verify, f"{head}.{forged}.{sig}", audience="hr-a2a")


def test_the_id_token_audience_is_the_web_client_not_the_agents():
    raises(InvalidToken, verifier().verify, token(aud="hr-web"), audience="hr-a2a")
    assert verifier().verify(token(aud="hr-web"), audience="hr-web")


# ── who the token is ────────────────────────────────────────────────────────

DIRECTORY = {"aiko.tanaka@acme.example": "W-100236"}.get
SERVICES = frozenset({"timeoff-agent"})


def test_a_verified_email_in_the_directory_is_that_worker():
    c = caller_from_claims(claims(), tenant_id="TEN-001", directory=DIRECTORY)
    assert (c.kind, c.name) == ("user", "W-100236")


def test_identities_that_do_not_map_to_a_worker_are_refused():
    for over in ({"email_verified": False}, {"email_verified": None}, {"email": "stranger@acme.example"},
                 {"email": None, "azp": "hr-cli"}, {"tenant_id": "TEN-002"}, {"tenant_id": None}):
        raises(Unauthorized, caller_from_claims, claims(**over), tenant_id="TEN-001", directory=DIRECTORY, service_clients=SERVICES)


def test_a_service_client_is_a_service_in_its_own_tenant_only():
    svc = claims(email=None, email_verified=None, azp="timeoff-agent")
    c = caller_from_claims(svc, tenant_id="TEN-001", directory=DIRECTORY, service_clients=SERVICES)
    assert (c.kind, c.name) == ("service", "timeoff-agent")
    raises(Unauthorized, caller_from_claims, {**svc, "tenant_id": "TEN-002"}, tenant_id="TEN-001", directory=DIRECTORY, service_clients=SERVICES)
    raises(Unauthorized, caller_from_claims, claims(email=None, azp="some-other-client"), tenant_id="TEN-001", directory=DIRECTORY, service_clients=SERVICES)
    # a person cannot pass as the service by naming its client: with an email they are matched as a person
    p = caller_from_claims(claims(azp="timeoff-agent"), tenant_id="TEN-001", directory=DIRECTORY, service_clients=SERVICES)
    assert p.kind == "user"


# ── the browser flow ────────────────────────────────────────────────────────

class FakeIdP:
    """Discovery, JWKS and a token endpoint that enforces PKCE and the client secret."""

    def __init__(self):
        self.codes: dict[str, dict] = {}
        self.exchanges = 0

    def authorize(self, location: str, *, email="priya.raman@acme.example", **claim_over) -> str:
        """The part the person's browser and the IdP do: returns the code for this authorization request."""
        q = {k: v[0] for k, v in parse_qs(urlparse(location).query).items()}
        assert q["code_challenge_method"] == "S256" and q["response_type"] == "code" and q["client_id"] == "hr-web"
        assert q["redirect_uri"] == "http://testserver/auth/callback" and {"openid", "email"} <= set(q["scope"].split())
        code = uuid.uuid4().hex
        self.codes[code] = {"challenge": q["code_challenge"], "nonce": q["nonce"], "email": email, "over": claim_over}
        return code

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/.well-known/openid-configuration"):
            return httpx.Response(200, json={
                "issuer": ISSUER, "authorization_endpoint": f"{ISSUER}/auth", "token_endpoint": f"{ISSUER}/token",
                "jwks_uri": f"{ISSUER}/certs", "end_session_endpoint": f"{ISSUER}/logout"})
        if path.endswith("/token"):
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            if form.get("grant_type") == "client_credentials":
                self.exchanges += 1
                return httpx.Response(200, json={"access_token": token(email=None, email_verified=None, azp=form["client_id"]), "expires_in": 300})
            grant = self.codes.pop(form.get("code", ""), None)
            if grant is None or form.get("client_secret") != "web-secret":
                return httpx.Response(400, json={"error": "invalid_grant"})
            verifier_hash = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).rstrip(b"=").decode()
            if verifier_hash != grant["challenge"]:
                return httpx.Response(400, json={"error": "invalid_grant", "error_description": "PKCE verification failed"})
            fields = {"aud": "hr-web", "azp": "hr-web", "email": grant["email"], "nonce": grant["nonce"], **grant["over"]}
            return httpx.Response(200, json={"id_token": token(**fields)})
        return httpx.Response(404)


class Web:
    def __init__(self, *, now=time.time, secret="a-long-test-secret"):
        self.idp = FakeIdP()
        http = httpx.Client(transport=httpx.MockTransport(self.idp.handle))
        provider = Provider(CFG, http)
        self.app = create_app(
            tempfile.mkdtemp(),
            identity=lambda ws: OIDCIdentity(CFG, ws, secret=secret, provider=provider, http=http,
                                             verifier=TokenVerifier(CFG, provider, key_for=lambda t: KEY.public_key()), now=now),
        )
        self.ws = self.app.state.workspace
        self.c = TestClient(self.app, base_url="http://testserver")

    def start(self):
        r = self.c.get("/login", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith(f"{ISSUER}/auth?")
        return r.headers["location"], parse_qs(urlparse(r.headers["location"]).query)["state"][0]

    def sign_in(self, **kw):
        location, state = self.start()
        code = self.idp.authorize(location, **kw)
        return self.c.get(f"/auth/callback?code={code}&state={state}", follow_redirects=False)


def test_signing_in_makes_the_session_that_worker_and_permissions_still_follow_the_directory():
    w = Web()
    r = w.sign_in()
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert "Priya Raman" in w.c.get("/requests?scope=mine").text
    assert w.c.get("/requests/REQ-2004").status_code == 403, "signing in as Priya does not let her see Samuel's request"
    assert w.c.get("/admin").status_code == 403
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie


def test_there_is_no_persona_picker_once_real_sign_in_is_on():
    w = Web()
    page = w.c.get("/signin").text
    assert "/login" in page and "W-100234" not in page
    assert w.c.post("/signin", data={"worker_id": "W-100001"}, follow_redirects=False).status_code in (404, 405)
    assert w.c.get("/requests?scope=inbox", follow_redirects=False).status_code == 303


def test_the_attacks_on_the_callback_all_fail_closed():
    w = Web()
    location, state = w.start()
    code = w.idp.authorize(location)
    assert w.c.get(f"/auth/callback?code={code}&state=not-the-state").status_code == 403, "state mismatch (CSRF / login fixation)"
    fresh = TestClient(w.app, base_url="http://testserver")
    assert fresh.get(f"/auth/callback?code={code}&state={state}").status_code == 403, "no login cookie: a callback this browser did not start"
    assert w.c.get("/auth/callback?error=access_denied").status_code == 403
    assert w.c.get("/auth/callback").status_code == 403
    # a code the IdP minted for a different PKCE verifier is refused by the IdP, and so by us
    location2, state2 = w.start()
    stolen = w.idp.authorize(location2)
    w.idp.codes[stolen]["challenge"] = "someone-elses-challenge"
    assert w.c.get(f"/auth/callback?code={stolen}&state={state2}").status_code == 403
    assert w.c.get("/requests?scope=mine", follow_redirects=False).status_code == 303, "none of that signed anyone in"


def test_an_id_token_for_another_flow_or_the_wrong_person_does_not_sign_in():
    for over, email in (({"nonce": "replayed-from-another-login"}, "priya.raman@acme.example"),
                        ({}, "stranger@acme.example"), ({"email_verified": False}, "priya.raman@acme.example"),
                        ({"tenant_id": "TEN-002"}, "priya.raman@acme.example"), ({"aud": "someone-else"}, "priya.raman@acme.example")):
        w = Web()
        r = w.sign_in(email=email, **over)
        assert r.status_code == 403, (over, email, r.status_code)
        assert w.c.get("/requests?scope=mine", follow_redirects=False).status_code == 303


def test_a_login_attempt_works_once():
    w = Web()
    location, state = w.start()
    code = w.idp.authorize(location)
    assert w.c.get(f"/auth/callback?code={code}&state={state}", follow_redirects=False).status_code == 303
    again = TestClient(w.app, base_url="http://testserver")
    again.cookies.set("hr_login", w.c.cookies.get("hr_login") or "x")
    assert again.get(f"/auth/callback?code={code}&state={state}").status_code == 403, "the code is single-use"


def test_sessions_are_signed_and_expire():
    clock = [1_000_000.0]
    w = Web(now=lambda: clock[0])
    assert w.sign_in().status_code == 303
    assert w.c.get("/requests?scope=mine", follow_redirects=False).status_code == 200
    real = w.c.cookies.get("hr_session")
    # clear first: setting a same-named cookie beside the original would send both, and which the server read would be luck
    w.c.cookies.clear()
    w.c.cookies.set("hr_session", real[:-3] + ("AAA" if not real.endswith("AAA") else "BBB"))
    assert w.c.get("/requests?scope=mine", follow_redirects=False).status_code == 303, "a tampered cookie is not a session"
    w.c.cookies.clear()
    w.c.cookies.set("hr_session", real)
    clock[0] += 9 * 3600
    assert w.c.get("/requests?scope=mine", follow_redirects=False).status_code == 303, "and a session ends"


def test_a_session_cookie_cannot_stand_in_for_a_login_attempt():
    w = Web()
    w.sign_in()
    session = w.c.cookies.get("hr_session")
    other = TestClient(w.app, base_url="http://testserver")
    other.cookies.set("hr_login", session)
    assert other.get("/auth/callback?code=x&state=y").status_code == 403
    assert other.get("/requests?scope=mine", follow_redirects=False).status_code == 303
    login_tx = Web()
    login_tx.start()
    forged = TestClient(login_tx.app, base_url="http://testserver")
    forged.cookies.set("hr_session", login_tx.c.cookies.get("hr_login"))
    assert forged.get("/requests?scope=mine", follow_redirects=False).status_code == 303


def test_signing_out_clears_the_session_and_ends_the_idp_session():
    w = Web()
    w.sign_in()
    r = w.c.post("/signout", follow_redirects=False)
    assert r.status_code == 303
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert r.headers["location"].startswith(f"{ISSUER}/logout?") and q["client_id"] == ["hr-web"]
    assert q["post_logout_redirect_uri"] == ["http://testserver/signin"]
    assert w.c.get("/requests?scope=mine", follow_redirects=False).status_code == 303


def test_it_refuses_to_start_without_a_session_secret():
    saved = os.environ.pop("HR_WEB_SECRET", None)
    try:
        raises(ConfigError, lambda: Web(secret=None))
    finally:
        if saved:
            os.environ["HR_WEB_SECRET"] = saved


# ── the agents ──────────────────────────────────────────────────────────────

class Agents:
    """Both A2A agents behind OIDC bearer checks, in process."""

    def __init__(self):
        from hr_timeoff_agent.agent.workspace import Workspace

        self.ws = Workspace(tempfile.mkdtemp())
        v = verifier()
        self.pay_app = create_payroll_app(OIDCBearer(CFG, "TEN-001", lambda e: None, verifier=v), base_url="http://payroll")
        self.idp = FakeIdP()
        provider = Provider(CFG, httpx.Client(transport=httpx.MockTransport(self.idp.handle)))
        self.service = ServiceTokens(CFG, "timeoff-agent", "secret", provider=provider,
                                     http=httpx.Client(transport=httpx.MockTransport(self.idp.handle)))
        payroll = A2AAgent("http://payroll", self.service.get,
                           httpx_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=self.pay_app), base_url="http://payroll"))
        self.app = create_timeoff_app(self.ws, OIDCBearer(CFG, "TEN-001", self.ws.worker_id_for_email, verifier=v),
                                      payroll=payroll, base_url="http://timeoff")

    def as_(self, bearer: str | None, app=None, base="http://timeoff") -> A2AAgent:
        return A2AAgent(base, bearer, httpx_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app or self.app), base_url=base))


def test_the_agent_accepts_a_workers_access_token_and_calls_payroll_as_a_service():
    a = Agents()
    r = asyncio.run(a.as_(token()).send(REVIEW))   # Aiko, REQ-2004: her report's request
    assert r.state == "input-required" and r.data["can_decide"] is True
    pay = r.data["payroll_impact"]
    assert pay["available"] is True and pay["source"] == "payroll agent over A2A", pay
    assert a.idp.exchanges == 1, "the service token was fetched by client credentials"


def test_the_agents_refuse_tokens_that_are_not_for_them():
    a = Agents()
    bad = {
        "none": None,
        "expired": token(exp=int(time.time()) - 3600, iat=int(time.time()) - 7200),
        "wrong audience": token(aud="hr-web"),
        "other tenant": token(tenant_id="TEN-002"),
        "unverified email": token(email_verified=False),
        "not a worker": token(email="stranger@acme.example"),
        "signed by another key": token(key=OTHER_KEY),
    }
    async def go():
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=a.app), base_url="http://timeoff")
        for name, t in bad.items():
            headers = {"Authorization": f"Bearer {t}"} if t else {}
            assert (await http.post("/a2a/jsonrpc", json={}, headers=headers)).status_code == 401, name
        assert (await http.get("/.well-known/agent-card.json")).status_code == 200, "discovery stays public"

    asyncio.run(go())


def test_a_person_cannot_ask_the_payroll_agent_and_neither_can_a_service_of_another_tenant():
    a = Agents()
    ask = {"skill": "assess_unpaid_leave_impact", "tenant_id": "TEN-001", "worker_id": "W-100237",
           "unpaid_hours": 80, "start": "2026-10-19", "end": "2026-11-06"}
    try:
        asyncio.run(a.as_(token(), a.pay_app, "http://payroll").send(ask))
    except Exception as exc:  # noqa: BLE001
        assert "401" in str(exc), exc   # the payroll agent does not even map people: a worker's token is no credential there
    else:
        raise AssertionError("a person's token must not reach the payroll agent")
    service = asyncio.run(a.as_(token(email=None, email_verified=None, azp="timeoff-agent"), a.pay_app, "http://payroll").send(ask))
    assert service.state == "completed", service
    stranger = httpx.AsyncClient(transport=httpx.ASGITransport(app=a.pay_app), base_url="http://payroll")
    other = token(email=None, email_verified=None, azp="timeoff-agent", tenant_id="TEN-002")
    r = asyncio.run(stranger.post("/a2a/jsonrpc", json={}, headers={"Authorization": f"Bearer {other}"}))
    assert r.status_code == 401


def test_a_service_token_is_not_a_worker_at_the_time_off_agent():
    a = Agents()
    r = asyncio.run(a.as_(token(email=None, email_verified=None, azp="timeoff-agent")).send(REVIEW))
    assert r.state == "rejected" and "does not belong to a worker" in r.text


def test_service_tokens_are_cached_and_refreshed_before_they_expire():
    now = [0.0]
    idp = FakeIdP()
    http = httpx.Client(transport=httpx.MockTransport(idp.handle))
    t = ServiceTokens(CFG, "timeoff-agent", "secret", provider=Provider(CFG, http), http=http, clock=lambda: now[0])
    first = t.get()
    assert t.get() == first and idp.exchanges == 1
    now[0] = 280.0   # inside the 30s margin of a 300s token
    t.get()
    assert idp.exchanges == 2


def test_the_mcp_http_server_needs_a_valid_token():
    from hr_timeoff_agent.hr_tools.server import HRToolServer
    from hr_timeoff_agent.adapters.oidc import BearerGuard

    server = HRToolServer()
    by_email = {w["email"]: w["worker_id"] for w in server.tenant.workers.values()}
    auth = OIDCBearer(CFG, server.tenant.tenant_id, by_email.get, verifier=verifier())
    app = BearerGuard(server.server.streamable_http_app(), auth.authenticate)
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
    accept = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    with TestClient(app, base_url="http://localhost:8000") as c:
        for label, headers in (("none", {}), ("garbage", {"Authorization": "Bearer nope"}),
                               ("wrong tenant", {"Authorization": f"Bearer {token(tenant_id='TEN-002')}"}),
                               ("not a worker", {"Authorization": f"Bearer {token(email='stranger@acme.example')}"})):
            assert c.post("/mcp", json=init, headers={**accept, **headers}).status_code == 401, label
        ok = c.post("/mcp", json=init, headers={**accept, "Authorization": f"Bearer {token()}"})
        assert ok.status_code == 200, (ok.status_code, ok.text[:200])


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
