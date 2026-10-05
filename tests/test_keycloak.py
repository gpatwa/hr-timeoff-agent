"""The same identity flows, against a real Keycloak with the development realm.

    docker compose -f docker-compose.dev.yml up -d keycloak
    HR_OIDC_ISSUER=http://localhost:8080/realms/hr HR_OIDC_CLIENT_SECRET=hr-web-dev-secret \\
    HR_OIDC_SERVICE_CLIENT_SECRET=timeoff-agent-dev-secret \\
        .venv/bin/python tests/test_keycloak.py

The browser is played by httpx: it follows the redirect to Keycloak, fills in the
real login form and comes back to the callback with the real code.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")
logging.getLogger("a2a").setLevel(logging.ERROR)

if not os.environ.get("HR_OIDC_ISSUER"):
    print("SKIPPED: set HR_OIDC_ISSUER (and the client secrets) to run the Keycloak tests.")
    raise SystemExit(0)
os.environ.setdefault("HR_WEB_SECRET", "keycloak-test-session-secret")
os.environ.setdefault("HR_PUBLIC_URL", "http://localhost:8000")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from hr_timeoff_agent.a2a_client import A2AAgent  # noqa: E402
from hr_timeoff_agent.a2a_common import OIDCBearer  # noqa: E402
from hr_timeoff_agent.a2a_payroll import DATA as PAYROLL_DATA, create_payroll_app  # noqa: E402
from hr_timeoff_agent.a2a_server import create_timeoff_app  # noqa: E402
from hr_timeoff_agent.oidc import OIDCConfig, Provider, ServiceTokens, password_grant  # noqa: E402
from hr_timeoff_agent.web.app import create_app  # noqa: E402
from hr_timeoff_agent.workspace import Workspace  # noqa: E402

CFG = OIDCConfig.from_env()
PASSWORD = "hr-demo-pass"
REVIEW = {"skill": "review_time_off_request", "request_id": "REQ-2004"}
ASK = {"skill": "assess_unpaid_leave_impact", "tenant_id": "TEN-001", "worker_id": "W-100237",
       "unpaid_hours": 80, "start": "2026-10-19", "end": "2026-11-06"}
SERVICE_SECRET = os.environ.get("HR_OIDC_SERVICE_CLIENT_SECRET", "timeoff-agent-dev-secret")


def user_token(email: str) -> str:
    return password_grant(CFG, email, PASSWORD)


def service_token(client: str, secret: str) -> str:
    return ServiceTokens(CFG, client, secret).get()


# ── the agents ──────────────────────────────────────────────────────────────

def agents():
    ws = Workspace(tempfile.mkdtemp())
    payroll_tenant = __import__("json").loads(PAYROLL_DATA.read_text())["tenant_id"]
    pay_app = create_payroll_app(OIDCBearer(CFG, payroll_tenant, lambda e: None), base_url="http://payroll")
    payroll = A2AAgent("http://payroll", ServiceTokens(CFG, "timeoff-agent", SERVICE_SECRET).get,
                       httpx_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=pay_app), base_url="http://payroll"))
    app = create_timeoff_app(ws, OIDCBearer(CFG, ws.tenant_id(), ws.worker_id_for_email), payroll=payroll, base_url="http://timeoff")
    return ws, app, pay_app


def call(app, base, bearer, data):
    client = A2AAgent(base, bearer, httpx_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base))
    return asyncio.run(client.send(data))


def status(app, bearer, path="/a2a/jsonrpc") -> int:
    async def go():
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://x")
        return (await http.post(path, json={}, headers={"Authorization": f"Bearer {bearer}"})).status_code

    return asyncio.run(go())


def test_a_real_access_token_reaches_the_approver_and_payroll_is_called_as_a_service():
    _, app, _ = agents()
    r = call(app, "http://timeoff", user_token("aiko.tanaka@acme.example"), REVIEW)
    assert r.state == "input-required" and r.data["can_decide"] is True, r
    assert r.data["payroll_impact"]["available"] is True, r.data["payroll_impact"]


def test_the_right_person_still_decides_and_others_only_view():
    ws, app, _ = agents()
    grace = call(app, "http://timeoff", user_token("grace.kim@acme.example"), REVIEW)
    assert grace.state == "completed" and grace.data["can_decide"] is False
    priya = call(app, "http://timeoff", user_token("priya.raman@acme.example"), REVIEW)
    assert priya.state == "rejected"
    assert ws.request("REQ-2004")["status"] == "pending"


def test_real_tokens_that_do_not_identify_a_worker_get_a_401():
    _, app, pay_app = agents()
    assert status(app, user_token("stranger@acme.example")) == 401, "authenticates at the IdP, not in the directory"
    assert status(app, user_token("unverified@acme.example")) == 401, "claims Priya's email without verifying it"
    assert status(app, "not-a-token") == 401
    assert status(pay_app, user_token("aiko.tanaka@acme.example")) == 401, "a person's token is no credential at payroll"
    other = service_token("other-tenant-agent", "other-tenant-dev-secret")
    assert status(pay_app, other) == 401 and status(app, other) == 401, "another tenant's service is refused by both"
    assert status(pay_app, service_token("timeoff-agent", SERVICE_SECRET)) != 401


# ── the browser ─────────────────────────────────────────────────────────────

class Browser:
    """An httpx client that keeps cookies like a browser does on http://localhost: Keycloak
    marks its cookies Secure, which browsers accept for localhost and httpx's jar would not send."""

    def __init__(self):
        self.http, self.jar = httpx.Client(follow_redirects=False), {}

    def request(self, method, url, **kw):
        headers = {"Cookie": "; ".join(f"{k}={v}" for k, v in self.jar.items())} if self.jar else {}
        r = self.http.request(method, url, headers=headers, **kw)
        for line in r.headers.get_list("set-cookie"):
            name, _, rest = line.partition("=")
            value = rest.split(";", 1)[0]
            if value in ("", '""') or "Max-Age=0" in line:
                self.jar.pop(name, None)
            else:
                self.jar[name] = value
        return r

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def post(self, url, **kw):
        return self.request("POST", url, **kw)


def login_form(browser: Browser, authorize_url: str):
    page = browser.get(authorize_url)
    assert page.status_code == 200, page.status_code
    return html.unescape(re.search(r'<form[^>]*id="kc-form-login"[^>]*action="([^"]+)"', page.text).group(1))


def browser_sign_in(app_client: TestClient, email: str, password: str = PASSWORD) -> httpx.Response:
    """Log in through the real Keycloak pages and return the app's callback response."""
    r = app_client.get("/login", follow_redirects=False)
    assert r.status_code == 303 and "/protocol/openid-connect/auth" in r.headers["location"]
    kc = Browser()
    action = login_form(kc, r.headers["location"])
    done = kc.post(action, data={"username": email, "password": password})
    assert done.status_code == 302, (done.status_code, done.text[:300])
    callback = done.headers["location"]
    assert callback.startswith("http://localhost:8000/auth/callback?"), callback
    return app_client.get(callback.removeprefix("http://localhost:8000"), follow_redirects=False)


def web():
    app = create_app(tempfile.mkdtemp())
    return TestClient(app, base_url="http://localhost:8000")


def test_signing_in_through_keycloak_is_that_worker():
    c = web()
    r = browser_sign_in(c, "priya.raman@acme.example")
    assert r.status_code == 303 and r.headers["location"] == "/", r.text[:300]
    assert "Priya Raman" in c.get("/requests?scope=mine").text
    assert c.get("/requests/REQ-2004").status_code == 403, "permissions still come from the directory"


def test_the_manager_signs_in_and_decides_through_the_web_app():
    c = web()
    assert browser_sign_in(c, "aiko.tanaka@acme.example").status_code == 303
    assert "REQ-2004" in c.get("/requests?scope=inbox").text
    assert c.post("/requests/REQ-2004/decide", data={"outcome": "returned"}, follow_redirects=False).status_code == 303
    assert c.app.state.workspace.request("REQ-2004")["status"] == "returned"


def test_an_account_that_is_not_a_worker_cannot_sign_in():
    for email in ("stranger@acme.example", "unverified@acme.example"):
        c = web()
        r = browser_sign_in(c, email)
        assert r.status_code == 403, (email, r.status_code)
        assert c.get("/requests?scope=mine", follow_redirects=False).status_code == 303


def test_a_wrong_password_never_reaches_the_app():
    c = web()
    r = c.get("/login", follow_redirects=False)
    kc = Browser()
    action = login_form(kc, r.headers["location"])
    bad = kc.post(action, data={"username": "priya.raman@acme.example", "password": "wrong"})
    assert bad.status_code == 200 and "/auth/callback" not in bad.headers.get("location", "")


def test_the_logout_url_points_at_keycloak_and_back():
    c = web()
    browser_sign_in(c, "priya.raman@acme.example")
    r = c.post("/signout", follow_redirects=False)
    assert r.headers["location"].startswith(CFG.issuer + "/protocol/openid-connect/logout?")
    assert "post_logout_redirect_uri=http%3A%2F%2Flocalhost%3A8000%2Fsignin" in r.headers["location"]
    assert c.get("/requests?scope=mine", follow_redirects=False).status_code == 303


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
            print(f"  FAIL  {name}: {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
