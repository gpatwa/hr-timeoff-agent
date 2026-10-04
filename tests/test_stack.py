"""A smoke test through the whole running stack, from the outside, over real HTTP.

    python scripts/make_secrets.py
    docker compose up -d --build --wait
    .venv/bin/python tests/test_stack.py

It signs in through the real Keycloak pages, decides a request in the web app, reviews and decides
another over A2A (the time-off agent asks the payroll agent, in the same container, for the pay
impact), calls the MCP server with a service token, and looks for the trace in Jaeger and the
numbers in Prometheus. It needs only httpx and the repo's A2A and MCP clients; the stack's
secrets are read from ./secrets (or HR_STACK_SECRETS).

It decides seeded requests, so it runs once per fresh stack: `docker compose down -v` first to
run it again. Run it a second time and it says so, rather than failing in a confusing place.
"""

from __future__ import annotations

import asyncio
import html
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SECRETS = Path(os.environ.get("HR_STACK_SECRETS") or ROOT / "secrets")
HOST = os.environ.get("HR_STACK_HOST", "localhost")
WEB, TIMEOFF, PAYROLL, MCP = (f"http://{HOST}:{p}" for p in (8000, 8100, 8101, 8200))
KEYCLOAK = f"http://{HOST}:8080/realms/hr"
JAEGER, PROM = f"http://{HOST}:16686", f"http://{HOST}:9090"

if not (SECRETS / "demo_password").exists():
    print(f"SKIPPED: no secrets in {SECRETS}. Run scripts/make_secrets.py and bring the stack up (docker compose up -d --wait).")
    raise SystemExit(0)


def secret(name: str) -> str:
    return (SECRETS / name).read_text().strip()


PASSWORD = secret("demo_password")


def poll(fn, what: str, timeout: float = 90.0):
    deadline, last = time.time() + timeout, None
    while time.time() < deadline:
        try:
            last = fn()
            if last:
                return last
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(2)
    raise AssertionError(f"timed out waiting for {what} (last: {last!r})")


# ── tokens and a browser ────────────────────────────────────────────────────

def user_token(email: str) -> str:
    r = httpx.post(f"{KEYCLOAK}/protocol/openid-connect/token", data={
        "grant_type": "password", "client_id": "hr-cli", "username": email, "password": PASSWORD, "scope": "openid"})
    r.raise_for_status()
    return r.json()["access_token"]


def service_token(client: str, secret_name: str) -> str:
    r = httpx.post(f"{KEYCLOAK}/protocol/openid-connect/token", data={
        "grant_type": "client_credentials", "client_id": client, "client_secret": secret(secret_name)})
    r.raise_for_status()
    return r.json()["access_token"]


class Browser:
    """An httpx client that keeps cookies like a browser on http://localhost (Keycloak marks its
    cookies Secure, which browsers accept for localhost and httpx's own jar would not send)."""

    def __init__(self):
        self.http, self.jar = httpx.Client(follow_redirects=False, timeout=30), {}

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


def sign_in(email: str) -> Browser:
    """The real authorization-code flow: the app redirects to Keycloak, the person logs in, Keycloak redirects back."""
    b = Browser()
    r = b.get(f"{WEB}/login")
    assert r.status_code == 303 and "/protocol/openid-connect/auth" in r.headers["location"], (r.status_code, r.headers.get("location"))
    page = b.get(r.headers["location"])
    action = html.unescape(re.search(r'<form[^>]*id="kc-form-login"[^>]*action="([^"]+)"', page.text).group(1))
    done = b.post(action, data={"username": email, "password": PASSWORD})
    assert done.status_code == 302, (done.status_code, done.text[:200])
    back = b.get(done.headers["location"])
    assert back.status_code == 303 and back.headers["location"] == "/", (back.status_code, back.text[:200])
    return b


# ── the tests, in order: later ones rely on what earlier ones changed ───────

def test_01_every_service_reports_ready():
    for url in (f"{WEB}/readyz", f"{TIMEOFF}/readyz", f"{PAYROLL}/healthz", f"{MCP}/healthz"):
        r = httpx.get(url, timeout=10)
        assert r.status_code == 200, (url, r.status_code, r.text[:200])
    assert httpx.get(f"{WEB}/readyz", timeout=10).json() == {"ready": True, "failing": {}}


def test_02_the_endpoints_that_need_a_token_refuse_without_one_and_with_the_wrong_tenant():
    assert httpx.post(f"{TIMEOFF}/a2a/jsonrpc", json={}, timeout=10).status_code == 401
    assert httpx.post(f"{PAYROLL}/a2a/jsonrpc", json={}, timeout=10).status_code == 401
    assert httpx.post(f"{MCP}/mcp", json={}, timeout=10).status_code == 401
    other = service_token("other-tenant-agent", "other_tenant_client_secret")
    for url in (f"{TIMEOFF}/a2a/jsonrpc", f"{PAYROLL}/a2a/jsonrpc", f"{MCP}/mcp"):
        r = httpx.post(url, json={}, headers={"Authorization": f"Bearer {other}"}, timeout=10)
        assert r.status_code in (401, 403), (url, r.status_code)


def test_03_the_agent_card_is_public_and_advertises_the_published_address():
    card = httpx.get(f"{TIMEOFF}/.well-known/agent-card.json", timeout=10).json()
    assert {s["id"] for s in card["skills"]} == {"file_time_off_request", "review_time_off_request"}
    assert any(i["url"] == f"http://localhost:8100/a2a/jsonrpc" for i in card["supportedInterfaces"]), card["supportedInterfaces"]


def test_04_a_manager_signs_in_through_keycloak_and_decides_in_the_web_app():
    dana = sign_in("dana.whitfield@acme.example")
    inbox = dana.get(f"{WEB}/requests?scope=inbox").text
    assert "REQ-2001" in inbox, "the stack has already been used: docker compose down -v, then up again"
    r = dana.post(f"{WEB}/requests/REQ-2001/decide", data={"outcome": "approved", "note": "enjoy"})
    assert r.status_code == 303, (r.status_code, r.text[:200])
    page = dana.get(f"{WEB}/requests/REQ-2001").text
    assert "approved" in page and "chain verified" in page


def test_05_an_agent_reviews_and_decides_over_a2a_and_the_payroll_agent_prices_the_unpaid_hours():
    from hr_timeoff_agent.a2a_client import A2AAgent

    async def go():
        aiko = A2AAgent(TIMEOFF, user_token("aiko.tanaka@acme.example"))
        try:
            r = await aiko.send({"skill": "review_time_off_request", "request_id": "REQ-2004"})
            assert r.state == "input-required", (r.state, r.text)
            impact = r.data["payroll_impact"]
            assert impact["available"] and impact["unpaid_hours"] == 80.0, impact
            d = await aiko.send({"outcome": "approved", "note": "40 paid, 80 unpaid"}, task_id=r.task_id, context_id=r.context_id)
            art = d.artifacts["decision"]
            assert d.state == "completed" and (art["paid_hours"], art["unpaid_hours"]) == (40.0, 80.0), (d.state, art)
            assert art["evidence"]["chain_verified"] is True
        finally:
            await aiko.close()

    asyncio.run(go())


def test_06_what_the_agent_decided_is_visible_in_the_web_app():
    aiko = sign_in("aiko.tanaka@acme.example")
    page = aiko.get(f"{WEB}/requests/REQ-2004").text
    assert "approved" in page and "chain verified" in page, "the web app and the A2A agent should share one database"


def test_07_the_mcp_server_answers_a_service_caller_over_http():
    import httpx2
    from mcp.client import Client
    from mcp.client.streamable_http import streamable_http_client

    async def go():
        http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {service_token('timeoff-agent', 'oidc_service_client_secret')}"})
        async with Client(streamable_http_client(f"{MCP}/mcp", http_client=http)) as c:
            assert len((await c.list_tools()).tools) == 6
            bal = (await c.call_tool("get_balance", {"worker_id": "W-100234", "plan": "PTO"})).structured_content
            assert bal["balance_hours"] == 96.0, bal
            hits = (await c.call_tool("search_handbook", {"query": "can a request be approved when the balance is short"})).structured_content
            assert hits["passages"] and {p["tenant_id"] for p in hits["passages"]} == {"TEN-001"}, hits

    asyncio.run(go())


def test_08_the_trace_is_in_jaeger_and_the_numbers_are_in_prometheus():
    def services():
        names = set(httpx.get(f"{JAEGER}/api/services", timeout=10).json()["data"])
        return names if {"hr-web", "hr-a2a"} <= names else None

    poll(services, "hr-web and hr-a2a in Jaeger")

    def a2a_trace():
        r = httpx.get(f"{JAEGER}/api/traces", params={"service": "hr-a2a", "limit": 20, "lookback": "1h"}, timeout=10).json()["data"]
        for trace in r:
            ops = {sp["operationName"] for sp in trace["spans"]}
            if {"a2a-timeoff POST", "a2a-payroll POST"} <= ops:
                return ops
        return None

    ops = poll(a2a_trace, "one trace covering both A2A hops")
    assert "a2a.client.send" in ops, ops

    def query(q):
        return httpx.get(f"{PROM}/api/v1/query", params={"query": q}, timeout=10).json()["data"]["result"] or None

    poll(lambda: query("sum(hr_http_requests_total)"), "hr_http_requests_total in Prometheus")

    # The two decisions are exported on a timer, so the first can be there before the second:
    # wait for both rather than asserting on whatever has arrived.
    def two_decisions():
        r = query("sum(hr_decision_total)")
        return r if r and float(r[0]["value"][1]) >= 2 else None

    poll(two_decisions, "both decisions counted in Prometheus (hr_decision_total >= 2)")

    def alert_rules():
        groups = httpx.get(f"{PROM}/api/v1/rules", timeout=10).json()["data"]["groups"]
        names = {r["name"] for g in groups for r in g["rules"]}
        return names if {"HRModelDegraded", "HREvidenceChainBroken"} <= names else None

    poll(alert_rules, "the alert rules loaded in Prometheus")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {str(exc)[:400]}", flush=True)
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
