"""What happens when things go wrong: outages, crashes, repeats, races.

Offline, in process. Runs on local files by default and on Postgres when
HR_DATABASE_URL is set (CI's durable job), where two workspaces stand in for two
processes sharing one database.

Run: .venv/bin/python tests/test_resilience.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")
if os.environ.get("HR_DATABASE_URL"):
    os.environ["HR_DATABASE_SCHEMA"] = "auto"
logging.getLogger("a2a").setLevel(logging.ERROR)

import httpx  # noqa: E402

from hr_timeoff_agent.agent import agents as agents_mod, graph as graph_mod
from hr_timeoff_agent.core import evidence, policy
from hr_timeoff_agent.adapters import llm  # noqa: E402
from hr_timeoff_agent.services.a2a.client import A2AAgent  # noqa: E402
from hr_timeoff_agent.services.a2a.common import PeerBreaker  # noqa: E402
from hr_timeoff_agent.services.a2a.server import create_timeoff_app  # noqa: E402
from hr_timeoff_agent.core.models import Recommendation  # noqa: E402
from hr_timeoff_agent.agent.workspace import Invalid, Workspace  # noqa: E402

PRIYA, DANA, AIKO, SAMUEL = "W-100234", "W-100001", "W-100236", "W-100237"
ALL_PASS = {"start": "2026-11-30", "end": "2026-12-02", "hours": "", "note": "Family trip, booked months ago.", "plan": "PTO"}
SHARED_DB = bool(os.environ.get("HR_DATABASE_URL"))


def _stub(*, system, user, schema, model=llm.AGENT_MODEL, record=False, label=""):
    return Recommendation(action="approve", rationale="Stub.", cited_rule_ids=["BAL-01"], cited_passage_ids=[], confidence="high")


def _down(*, system, user, schema, model=llm.AGENT_MODEL, record=False, label=""):
    raise llm.ModelUnavailable("APIConnectionError: connection refused")


def _human_entries(ws, rid):
    d = ws.detail(ws.persona(DANA if rid == "REQ-2001" else AIKO), rid)
    return d, [e for e in d["evidence"] if e["actor"] == "human"], [e for e in d["evidence"] if e["node"] == "record"]


# ── the model is down ───────────────────────────────────────────────────────

def test_a_model_outage_escalates_to_the_manager_instead_of_failing_triage():
    graph_mod.structured = _down
    ws = Workspace(tempfile.mkdtemp())
    rid = ws.submit(ws.persona(PRIYA), **ALL_PASS)
    r = ws.request(rid)
    assert r["status"] == "pending" and not r.get("triage_error"), r
    d = ws.detail(ws.persona(DANA), rid)
    assert d["recommendation"]["action"] == "escalate" and d["recommendation"]["confidence"] == "low"
    assert "could not be reached" in d["recommendation"]["rationale"]
    degraded = [e for e in d["evidence"] if e["data"].get("degraded")]
    assert degraded and degraded[0]["actor"] == "system" and d["chain_ok"]
    assert d["findings"], "the deterministic findings are still there"
    graph_mod.structured = _stub
    ws.decide(ws.persona(DANA), rid, "approved", "Fine.")  # and the manager can still decide
    assert ws.request(rid)["status"] == "approved"
    ws.close()


def test_a_bad_request_is_not_mistaken_for_an_outage():
    import anthropic

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    for exc in (anthropic.APIConnectionError(request=req), anthropic.APITimeoutError(request=req),
                anthropic.InternalServerError("overloaded", response=httpx.Response(529, request=req), body=None),
                anthropic.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)):
        try:
            with llm.translate_outages():
                raise exc
        except llm.ModelUnavailable:
            pass
        else:
            raise AssertionError(f"{type(exc).__name__} should be an outage")
    for exc in (anthropic.BadRequestError("bad", response=httpx.Response(400, request=req), body=None),
                anthropic.AuthenticationError("key", response=httpx.Response(401, request=req), body=None), ValueError("x")):
        try:
            with llm.translate_outages():
                raise exc
        except llm.ModelUnavailable:
            raise AssertionError(f"{type(exc).__name__} is a bug to fix, not an outage")
        except type(exc):
            pass
    assert llm.api_client().max_retries == llm.MAX_RETRIES and llm.api_client().timeout.read == llm.TIMEOUT_S


def test_multi_agent_outage_stops_after_one_timeout_and_escalates():
    calls = []

    def down(**kw):
        calls.append(kw.get("label"))
        raise llm.ModelUnavailable("timeout")

    ws = Workspace(tempfile.mkdtemp())  # seeded first, from the recordings, in single mode
    saved = (agents_mod.run_agent, agents_mod.structured, os.environ.get("HR_AGENT_MODE"))
    agents_mod.run_agent = agents_mod.structured = down
    os.environ["HR_AGENT_MODE"] = "multi"
    try:
        rid = ws.submit(ws.persona(PRIYA), **ALL_PASS)
        d = ws.detail(ws.persona(DANA), rid)
    finally:
        agents_mod.run_agent, agents_mod.structured = saved[0], saved[1]
        os.environ.pop("HR_AGENT_MODE") if saved[2] is None else os.environ.__setitem__("HR_AGENT_MODE", saved[2])
    assert len(calls) == 1, f"the coverage specialist and coordinator should not retry the outage: {calls}"
    assert ws.request(rid)["status"] == "pending" and d["recommendation"]["action"] == "escalate" and d["chain_ok"]
    assert {e["node"] for e in d["evidence"] if e["data"].get("degraded")} == {"policy_specialist", "assess"}
    ws.close()


# ── crashes in the middle of a decision ─────────────────────────────────────

def _balance(ws):
    return ws.worker(SAMUEL)["time_off_plans"]["PTO"]["balance_hours"]


def test_a_crash_after_the_graph_recorded_the_decision_is_finished_by_the_retry():
    ws = Workspace(tempfile.mkdtemp())
    aiko, real, calls = ws.persona(AIKO), ws._commit, [0]

    def crash_once(*a, **kw):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("process died before committing")
        return real(*a, **kw)

    ws._commit = crash_once
    try:
        ws.decide(aiko, "REQ-2004", "approved", "")
    except RuntimeError:
        pass
    else:
        raise AssertionError("the crash should surface")
    assert ws.request("REQ-2004")["status"] == "pending" and _balance(ws) == 40.0, "nothing half-applied"
    ws.decide(aiko, "REQ-2004", "approved", "")  # the retry
    assert ws.request("REQ-2004")["status"] == "approved" and _balance(ws) == 0.0
    assert len([a for a in ws._read("absences.json") if a["absence_id"] == "ABS-REQ-2004"]) == 1
    d, human, record = _human_entries(ws, "REQ-2004")
    assert len(human) == 1 and len(record) == 1 and d["chain_ok"], "one decision, one record, intact chain"
    ws.close()


def test_a_crash_between_the_gate_and_record_is_finished_by_the_retry():
    ws = Workspace(tempfile.mkdtemp())
    aiko, real, calls = ws.persona(AIKO), policy.authorize_approver, [0]

    def crash_in_record(*a, **kw):
        calls[0] += 1
        if calls[0] == 2:  # the gate is the first call, `record` the second
            raise RuntimeError("process died inside record")
        return real(*a, **kw)

    policy.authorize_approver = crash_in_record
    try:
        ws.decide(aiko, "REQ-2004", "approved", "")
    except RuntimeError:
        pass
    finally:
        policy.authorize_approver = real
    assert ws.request("REQ-2004")["status"] == "pending"
    ws.decide(aiko, "REQ-2004", "approved", "")
    assert ws.request("REQ-2004")["status"] == "approved" and _balance(ws) == 0.0
    d, human, record = _human_entries(ws, "REQ-2004")
    assert len(human) == 1 and len(record) == 1 and d["chain_ok"]
    ws.close()


def test_a_restart_reconciles_a_decision_the_tenant_never_received():
    home = tempfile.mkdtemp()
    ws = Workspace(home)
    ws._commit = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("died"))
    try:
        ws.decide(ws.persona(AIKO), "REQ-2004", "approved", "")
    except RuntimeError:
        pass
    ws.close()
    ws = Workspace(home)  # the next start finishes it, with nobody asking
    assert ws.reconciled == ["REQ-2004"], ws.reconciled
    assert ws.request("REQ-2004")["status"] == "approved" and _balance(ws) == 0.0
    ws.close()


def test_the_same_decision_twice_is_a_no_op_and_a_different_one_is_refused():
    ws = Workspace(tempfile.mkdtemp())
    dana = ws.persona(DANA)
    hours = ws.request("REQ-2001")["hours"]
    ws.decide(dana, "REQ-2001", "approved", "")
    after = ws.worker(PRIYA)["time_off_plans"]["PTO"]["balance_hours"]
    assert after == 96.0 - hours
    ws.decide(dana, "REQ-2001", "approved", "")  # a double click or a retried message
    assert ws.worker(PRIYA)["time_off_plans"]["PTO"]["balance_hours"] == after
    try:
        ws.decide(dana, "REQ-2001", "declined", "")
    except Invalid as exc:
        assert "approved" in str(exc)
    else:
        raise AssertionError("a conflicting second decision must be refused")
    assert ws.request("REQ-2001")["status"] == "approved"
    ws.close()


def test_racing_decisions_take_turns_and_apply_once():
    home = tempfile.mkdtemp()
    spaces = [Workspace(home), Workspace(home) if SHARED_DB else None]
    spaces[1] = spaces[1] or spaces[0]
    hours = spaces[0].request("REQ-2001")["hours"]
    errors = []

    def go(i):
        try:
            spaces[i % 2].decide(spaces[i % 2].persona(DANA), "REQ-2001", "approved", "")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    ws = spaces[0]
    assert not errors, errors
    assert ws.worker(PRIYA)["time_off_plans"]["PTO"]["balance_hours"] == 96.0 - hours, "deducted exactly once"
    d, human, record = _human_entries(ws, "REQ-2001")
    assert len(human) == 1 and len(record) == 1 and d["chain_ok"]
    assert len([a for a in ws._read("absences.json") if a["absence_id"] == "ABS-REQ-2001"]) == 1
    [s.close() for s in {id(s): s for s in spaces}.values()]


def test_a_slow_triage_does_not_block_other_requests():
    entered, release = threading.Event(), threading.Event()

    def slow(**kw):
        entered.set()
        release.wait(10)
        return _stub(**kw)

    graph_mod.structured = slow
    ws = Workspace(tempfile.mkdtemp())
    t = threading.Thread(target=lambda: ws.submit(ws.persona(PRIYA), **ALL_PASS))
    t.start()
    assert entered.wait(10)
    start = time.monotonic()
    ws.decide(ws.persona(AIKO), "REQ-2004", "returned", "")  # another request, while one model call is in flight
    assert time.monotonic() - start < 5, "a decision waited for someone else's model call"
    release.set()
    t.join()
    ws.close()


# ── repeats ─────────────────────────────────────────────────────────────────

def test_the_same_idempotency_key_files_one_request():
    graph_mod.structured = _stub
    ws = Workspace(tempfile.mkdtemp())
    priya = ws.persona(PRIYA)
    first = ws.submit(priya, **ALL_PASS, idempotency_key="form-1")
    again = ws.submit(priya, **ALL_PASS, idempotency_key="form-1")
    other = ws.submit(priya, **ALL_PASS, idempotency_key="form-2")
    assert first == again and other != first
    assert len([r for r in ws.requests() if r.get("idempotency_key") == "form-1"]) == 1
    ws.close()


# ── the payroll peer ────────────────────────────────────────────────────────

def test_the_breaker_opens_after_repeated_failures_and_closes_after_a_success():
    now = [0.0]
    b = PeerBreaker(threshold=3, cooldown=30, clock=lambda: now[0])
    for _ in range(3):
        assert not b.open
        b.failure()
    assert b.open
    now[0] = 29.0
    assert b.open
    now[0] = 31.0
    assert not b.open, "after the cool-down one call is let through"
    b.failure()
    assert b.open, "and one more failure re-opens it"
    now[0] = 70.0
    assert not b.open
    b.success()
    b.failure()
    assert not b.open


def _stack_with_payroll(transport):
    from test_a2a import SERVICE, Stack

    s = Stack()
    s.payroll = A2AAgent("http://payroll", s.tokens.for_principal(SERVICE),
                         httpx_client=httpx.AsyncClient(transport=transport, base_url="http://payroll"))
    s.app = create_timeoff_app(s.ws, s.tokens, payroll=s.payroll, base_url="http://timeoff")
    return s


def test_a_slow_payroll_peer_costs_a_bounded_wait_then_is_skipped():
    import hr_timeoff_agent.services.a2a.server as srv
    from test_a2a import AIKO as A_AIKO, REVIEW

    seen = [0]

    async def hang(request):
        seen[0] += 1
        await asyncio.sleep(30)

    saved, srv.PEER_TIMEOUT_S = srv.PEER_TIMEOUT_S, 0.2
    try:
        s = _stack_with_payroll(httpx.MockTransport(hang))
        manager = s.as_(A_AIKO)
        started = time.monotonic()
        for _ in range(3):
            r = asyncio.run(manager.send(REVIEW))
            pay = r.data["payroll_impact"]
            assert r.state == "input-required", "an advisory peer must not block the review"
            assert pay["available"] is False and "unreachable" in pay["reason"]
        assert time.monotonic() - started < 10, "each review waits a bounded time, not the peer's 30s"
        before = seen[0]
        r = asyncio.run(manager.send(REVIEW))
        assert seen[0] == before, "after repeated failures the peer is not called at all"
        assert "skipping" in r.data["payroll_impact"]["reason"] and r.state == "input-required"
    finally:
        srv.PEER_TIMEOUT_S = saved


class _FailsFirst(httpx.AsyncBaseTransport):
    """A link that drops its first request, then behaves."""

    def __init__(self, inner, drops: int = 1):
        self.inner, self.drops = inner, drops

    async def handle_async_request(self, request):
        if self.drops:
            self.drops -= 1
            raise httpx.ConnectError("blip")
        return await self.inner.handle_async_request(request)


def test_a_transient_peer_failure_is_retried_once():
    from test_a2a import AIKO as A_AIKO, REVIEW, Stack

    s = Stack()
    s = _stack_with_payroll(_FailsFirst(httpx.ASGITransport(app=s.pay_app)))
    r = asyncio.run(s.as_(A_AIKO).send(REVIEW))
    assert r.data["payroll_impact"]["available"] is True, r.data["payroll_impact"]


def test_a_retried_a2a_message_files_one_request():
    from test_a2a import ALL_PASS as FILE, PRIYA as A_PRIYA, Stack

    graph_mod.structured = _stub
    s = Stack()
    agent = s.as_(A_PRIYA)

    async def go():
        a = await agent.send(FILE, message_id="message-1")
        b = await agent.send(FILE, message_id="message-1")  # the caller timed out and sent it again
        return a.artifacts["request"]["request_id"], b.artifacts["request"]["request_id"]

    first, second = asyncio.run(go())
    assert first == second, (first, second)
    assert len([r for r in s.ws.requests() if r["worker_id"] == A_PRIYA and r.get("idempotency_key") == "message-1"]) == 1


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        saved = graph_mod.structured
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        finally:
            graph_mod.structured = saved
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
