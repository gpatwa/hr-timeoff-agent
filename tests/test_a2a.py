"""The two A2A agents, through the real A2A protocol (JSON-RPC over an in-process ASGI transport).

Run: .venv/bin/python tests/test_a2a.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")
logging.getLogger("a2a").setLevel(logging.ERROR)

import httpx  # noqa: E402

from hr_timeoff_agent.agent import graph as graph_mod
from hr_timeoff_agent.adapters import llm
from hr_timeoff_agent.core import policy  # noqa: E402
from hr_timeoff_agent.services.a2a.client import A2AAgent  # noqa: E402
from hr_timeoff_agent.services.a2a.common import BearerTokens  # noqa: E402
from hr_timeoff_agent.services.a2a.payroll import PayrollError, assess, create_payroll_app  # noqa: E402
from hr_timeoff_agent.services.a2a.server import TimeOffExecutor, create_timeoff_app  # noqa: E402
from hr_timeoff_agent.core.models import Recommendation  # noqa: E402
from hr_timeoff_agent.agent.workspace import Workspace  # noqa: E402

PRIYA, AIKO, GRACE, SAMUEL, DANA = "W-100234", "W-100236", "W-100003", "W-100237", "W-100001"
SERVICE = "timeoff-agent"
REVIEW = {"skill": "review_time_off_request", "request_id": "REQ-2004"}
ALL_PASS = {"skill": "file_time_off_request", "from": "2026-11-30", "to": "2026-12-02", "note": "Family trip, booked months ago."}


class Stack:
    """A workspace, both agents, and clients for any caller, all in process."""

    def __init__(self, *, payroll_up: bool = True):
        self.home = tempfile.mkdtemp()
        self.ws = Workspace(self.home)
        self.tokens = BearerTokens.derive("test-secret", [p.worker_id for p in self.ws.personas()] + [SERVICE])
        self.pay_app = create_payroll_app(self.tokens, base_url="http://payroll")
        transport = httpx.ASGITransport(app=self.pay_app) if payroll_up else httpx.MockTransport(self._down)
        self.payroll = A2AAgent("http://payroll", self.tokens.for_principal(SERVICE),
                                httpx_client=httpx.AsyncClient(transport=transport, base_url="http://payroll"))
        self.app = create_timeoff_app(self.ws, self.tokens, payroll=self.payroll, base_url="http://timeoff")

    @staticmethod
    def _down(request):
        raise httpx.ConnectError("payroll agent is down")

    def as_(self, worker_id: str) -> A2AAgent:
        return A2AAgent("http://timeoff", self.tokens.for_principal(worker_id),
                        httpx_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://timeoff"))

    def status(self, rid: str) -> str:
        return self.ws.request(rid)["status"]


def run(coro):
    return asyncio.run(coro)


def test_the_agent_card_is_public_and_the_endpoint_is_not():
    s = Stack()

    async def go():
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://timeoff")
        card = await http.get("/.well-known/agent-card.json")
        assert card.status_code == 200
        body = card.json()
        assert {sk["id"] for sk in body["skills"]} == {"file_time_off_request", "review_time_off_request"}
        assert "bearer" in body["securitySchemes"]
        assert (await http.post("/a2a/jsonrpc", json={})).status_code == 401
        assert (await http.post("/a2a/jsonrpc", json={}, headers={"Authorization": "Bearer nope"})).status_code == 401

    run(go())


def test_a_service_token_is_not_a_worker():
    s = Stack()
    agent = A2AAgent("http://timeoff", s.tokens.for_principal(SERVICE),
                     httpx_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://timeoff"))
    r = run(agent.send(REVIEW))
    assert r.state == "rejected" and "does not belong to a worker" in r.text


def test_the_approver_gets_input_required_with_an_advisory_recommendation_and_payroll_impact():
    s = Stack()
    r = run(s.as_(AIKO).send(REVIEW))
    assert r.state == "input-required" and r.data["advisory_only"] is True and r.data["can_decide"] is True
    assert r.data["recommendation"]["action"] == "escalate"
    assert r.data["approver_required"]["worker_id"] == AIKO
    pay = r.data["payroll_impact"]
    assert pay["available"] and pay["source"] == "payroll agent over A2A" and pay["unpaid_hours"] == 80.0
    assert pay["estimated_pay_reduction"] == 4923.08 and pay["payroll_signoff_required"] is True
    assert s.status("REQ-2004") == "pending", "reviewing decides nothing"


def test_someone_who_is_not_the_approver_gets_a_view_only_review():
    s = Stack()
    for who in (GRACE, SAMUEL):
        r = run(s.as_(who).send(REVIEW))
        assert r.state == "completed" and r.data["can_decide"] is False, who
    assert s.status("REQ-2004") == "pending"


def test_a_peer_who_cannot_see_the_request_is_rejected():
    s = Stack()
    r = run(s.as_(PRIYA).send(REVIEW))
    assert r.state == "rejected" and "can't see" in r.text


def test_another_callers_task_cannot_be_taken_over():
    s = Stack()

    async def go():
        r = await s.as_(AIKO).send(REVIEW)
        try:
            await s.as_(GRACE).send({"outcome": "approved"}, task_id=r.task_id, context_id=r.context_id)
        except Exception as exc:
            assert "not found" in str(exc).lower()
        else:
            raise AssertionError("HR continued the manager's task")

    run(go())
    assert s.status("REQ-2004") == "pending"


def test_the_managers_decision_completes_the_task_and_is_applied():
    s = Stack()

    async def go():
        aiko = s.as_(AIKO)
        r = await aiko.send(REVIEW)
        return await aiko.send({"outcome": "approved", "note": "40 paid, 80 unpaid"}, task_id=r.task_id, context_id=r.context_id)

    d = run(go())
    art = d.artifacts["decision"]
    assert d.state == "completed" and art["outcome"] == "approved" and art["decided_by"]["worker_id"] == AIKO
    assert (art["paid_hours"], art["unpaid_hours"]) == (40.0, 80.0)
    assert art["evidence"]["chain_verified"] is True and art["payroll_impact"]["available"]
    assert s.status("REQ-2004") == "approved"
    assert s.ws.worker(SAMUEL)["time_off_plans"]["PTO"]["balance_hours"] == 0.0


def test_a_bad_outcome_keeps_the_task_waiting():
    s = Stack()

    async def go():
        aiko = s.as_(AIKO)
        r = await aiko.send(REVIEW)
        again = await aiko.send({"outcome": "maybe"}, task_id=r.task_id, context_id=r.context_id)
        done = await aiko.send({"outcome": "returned"}, task_id=r.task_id, context_id=r.context_id)
        return again, done

    again, done = run(go())
    assert again.state == "input-required" and "approved, declined or returned" in again.text
    assert done.state == "completed" and done.artifacts["decision"]["outcome"] == "returned"


class _Recorder:
    """Captures what the executor tells the task, to check the wrong-approver path directly."""

    def __init__(self):
        self.calls = []

    def new_agent_message(self, parts):
        return parts

    async def requires_input(self, message):
        self.calls.append(("input-required", message))

    async def reject(self, message):
        self.calls.append(("rejected", message))

    async def add_artifact(self, *a, **k):
        self.calls.append(("artifact", None))

    async def complete(self, message=None):
        self.calls.append(("completed", message))


def test_a_refused_decision_leaves_the_task_waiting_and_the_request_pending():
    """The gate, reached over A2A: HR may view REQ-2004 but is not Samuel's manager."""
    s = Stack()
    ex = TimeOffExecutor(s.ws, None)
    rec = _Recorder()
    hr = s.ws.persona(GRACE)
    run(ex._decide(rec, hr, "REQ-2004", {"outcome": "approved"}))
    assert [c[0] for c in rec.calls] == ["input-required"], rec.calls
    assert "not Samuel Ortiz's direct manager" in str(rec.calls[0][1]) or "direct manager" in str(rec.calls[0][1])
    assert s.status("REQ-2004") == "pending"


def _fake_structured(*, system, user, schema, model=llm.AGENT_MODEL, record=False, label=""):
    return Recommendation(action="approve", rationale="Stub.", cited_rule_ids=["BAL-01"], cited_passage_ids=[], confidence="high")


def test_an_employee_files_a_request_that_the_web_workspace_and_the_manager_then_see():
    s = Stack()
    real, graph_mod.structured = graph_mod.structured, _fake_structured
    try:
        filed = run(s.as_(PRIYA).send(ALL_PASS))
    finally:
        graph_mod.structured = real
    art = filed.artifacts["request"]
    assert filed.state == "completed" and art["status"] == "pending" and art["approver"]["worker_id"] == DANA
    assert "recommendation" not in art, "the employee's result carries findings, not the advisory recommendation"
    assert s.status(art["request_id"]) == "pending", "the same workspace the web app reads"
    reviewed = run(s.as_(DANA).send({"skill": "review_time_off_request", "request_id": art["request_id"]}))
    assert reviewed.state == "input-required" and reviewed.data["payroll_impact"] is None, "no unpaid hours, so no payroll call"


def test_bad_filing_input_is_rejected_with_the_reason():
    s = Stack()
    r = run(s.as_(PRIYA).send({**ALL_PASS, "from": "2026-12-05", "to": "2026-12-01"}))
    assert r.state == "rejected" and "before the start" in r.text
    assert len(s.ws.requests()) == 5


def test_a_payroll_agent_outage_does_not_block_the_review():
    s = Stack(payroll_up=False)
    r = run(s.as_(AIKO).send(REVIEW))
    assert r.state == "input-required"
    assert r.data["payroll_impact"]["available"] is False and "unreachable" in r.data["payroll_impact"]["reason"]


def test_the_payroll_agent_refuses_another_tenant_and_unauthenticated_callers():
    s = Stack()

    async def go():
        ok = await s.payroll.send({"skill": "assess_unpaid_leave_impact", "tenant_id": "TEN-001", "worker_id": SAMUEL,
                                   "unpaid_hours": 8, "start": "2026-10-19", "end": "2026-10-19"})
        other = await s.payroll.send({"skill": "assess_unpaid_leave_impact", "tenant_id": "TEN-002", "worker_id": SAMUEL,
                                      "unpaid_hours": 8, "start": "2026-10-19", "end": "2026-10-19"})
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=s.pay_app), base_url="http://payroll")
        return ok, other, (await http.post("/a2a/jsonrpc", json={})).status_code

    ok, other, status = run(go())
    assert ok.state == "completed" and other.state == "rejected" and "TEN-001" in other.text and status == 401


def test_payroll_arithmetic_comes_from_data():
    data = json.loads((Path(__file__).resolve().parent.parent / "data" / "payroll.json").read_text())
    out = assess(data, {"tenant_id": "TEN-001", "worker_id": SAMUEL, "unpaid_hours": 80, "start": "2026-10-19", "end": "2026-11-06"})
    assert out["hourly_rate"] == 61.54 and out["estimated_pay_reduction"] == 4923.08
    assert [p["unpaid_hours"] for p in out["pay_periods"]] == [40.0, 40.0]
    assert out["payroll_signoff_required"] and not out["benefits_review_required"]
    data["rules"]["payroll_signoff_over_unpaid_hours"] = 100  # a data change, not a code change
    assert not assess(data, {"tenant_id": "TEN-001", "worker_id": SAMUEL, "unpaid_hours": 80, "start": "2026-10-19", "end": "2026-11-06"})["payroll_signoff_required"]
    try:
        assess(data, {"tenant_id": "TEN-001", "worker_id": "W-NOPE", "unpaid_hours": 8, "start": "2026-10-19", "end": "2026-10-19"})
    except PayrollError:
        return
    raise AssertionError("an unknown worker was priced")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
