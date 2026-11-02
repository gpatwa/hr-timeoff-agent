"""Regression tests for twelve findings of the manual assessment (Aveto playbook, docs/research/2026-10-assessment-hr-timeoff-agent.md
and the calibration run of 2026-11-02). Each test names the finding it pins and fails on the code before the fix.

Run: .venv/bin/python tests/test_assessment_fixes.py
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")
for var in ("HR_OIDC_ISSUER", "HR_DATABASE_URL"):
    os.environ.pop(var, None)

from fastapi.testclient import TestClient  # noqa: E402

from hr_timeoff_agent.core import policy  # noqa: E402
from hr_timeoff_agent.hr_tools.server import HRToolServer  # noqa: E402
from hr_timeoff_agent.services.a2a import payroll  # noqa: E402
from hr_timeoff_agent.services.a2a.client import TaskResult  # noqa: E402
from hr_timeoff_agent.services.web.app import create_app  # noqa: E402
from hr_timeoff_agent.services.web.identity import PersonaSwitcher  # noqa: E402

AIKO, GRACE, PRIYA = "W-100236", "W-100003", "W-100234"


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f"{exc.__name__} not raised")


# ── web (F-01-1 .. F-01-4) ───────────────────────────────────────────────────────────────────────

def test_f01_1_demo_sign_in_is_visible_to_a_monitor_and_logged():
    records = []

    class Catch(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("hr_timeoff_agent.services.web.app")
    h = Catch()
    log.addHandler(h)
    try:
        app = create_app(tempfile.mkdtemp())
    finally:
        log.removeHandler(h)
    assert any("DEMO sign-in" in m for m in records), records
    assert TestClient(app).get("/healthz").json()["auth"] == "persona"


def test_f01_2_non_ascii_input_is_invalid_not_a_server_error():
    from hr_timeoff_agent.services.web.identity import _same

    assert _same("abc", "abc") and not _same("abc", "abd") and not _same("é", "e") and _same("é", "é")
    ps = PersonaSwitcher("k")
    for raw in ("é.é", f"{PRIYA}.é", "ünï.cödé"):
        assert ps.current(SimpleNamespace(cookies={"hr_persona": raw})) is None, raw


def test_f01_3_a_link_cannot_make_the_app_show_its_text_as_a_notice():
    app = create_app(tempfile.mkdtemp())
    c = TestClient(app)
    c.post("/signin", data={"worker_id": AIKO}, follow_redirects=False)
    assert "Your account is locked" not in c.get("/requests?scope=mine&msg=Your account is locked").text
    # the app's own notices still work, once
    r = c.post("/requests/REQ-2004/decide", data={"outcome": "approved"}, follow_redirects=True)
    assert "Recorded: approved" in r.text
    assert "Recorded: approved" not in c.get("/requests?scope=inbox").text, "the notice is shown once"


def test_f01_4_only_a_missing_request_is_a_404():
    app = create_app(tempfile.mkdtemp())

    @app.get("/boom")
    def boom():
        raise KeyError("internal-key-name")

    c = TestClient(app, raise_server_exceptions=False)
    c.post("/signin", data={"worker_id": AIKO}, follow_redirects=False)
    assert c.get("/boom").status_code == 500, "a stray KeyError is a bug, not a 404"
    assert c.get("/requests/REQ-9999").status_code == 404
    assert "internal-key-name" not in c.get("/boom").text


# ── policy and the tool server (F-04-3, F-04-4, F-05-1, F-05-2) ──────────────────────────────────

def _request(**over):
    return {"worker_id": AIKO, "plan": "PTO", "from": "2026-11-30", "to": "2026-12-02", "hours": 24.0, "submitted_at": "2026-10-01", "note": "", **over}


def test_f04_a_request_that_cannot_be_judged_fails_closed():
    tenant = policy.Tenant()
    w = tenant.workers[AIKO]
    for bad in ({"hours": -8.0}, {"hours": 0}, {"hours": float("nan")}, {"hours": "lots"}, {"from": "2026-12-05", "to": "2026-12-01"}, {"to": "not-a-date"}):
        findings = policy.evaluate(tenant, _request(**bad), w)
        assert len(findings) == 1 and findings[0].rule_id == "INP-00" and findings[0].status == "fail", (bad, findings)
        assert policy.has_blocking_failure(findings), bad
    assert [f.rule_id for f in policy.evaluate(tenant, _request(), w)] == [r.rule_id for r in policy.evaluate(tenant, _request(), w)]
    assert all(f.rule_id != "INP-00" for f in policy.evaluate(tenant, _request(), w)), "a good request is judged by the real rules"


def test_f05_1_the_tool_boundary_rejects_reversed_windows_and_bad_hours():
    from hr_timeoff_agent.hr_tools.client import ToolCallError, call_tool

    srv = HRToolServer()
    ok = dict(worker_id=AIKO, plan="PTO", start="2026-11-30", end="2026-12-02", hours=24.0, submitted_at="2026-10-01")
    assert call_tool("evaluate_policy", ok, srv)["findings"]
    for bad in ({"start": "2026-12-05", "end": "2026-12-01"}, {"hours": -1.0}, {"hours": 0.0}):
        raises(ToolCallError, call_tool, "evaluate_policy", {**ok, **bad}, srv)
    raises(ToolCallError, call_tool, "team_availability", {"worker_id": AIKO, "start": "2026-12-05", "end": "2026-12-01"}, srv)
    assert call_tool("team_availability", {"worker_id": AIKO, "start": "2026-11-30", "end": "2026-12-02"}, srv)


def test_f05_2_every_failed_call_reaches_the_audit_log():
    srv = HRToolServer()

    def explode(worker_id: str):
        raise KeyError("missing field")

    explode.__name__ = "explode"
    raises(KeyError, srv._audited(explode), worker_id="W-1")
    assert srv.audit and srv.audit[-1]["tool"] == "explode" and "KeyError" in srv.audit[-1]["error"]


# ── payroll agent (F-03-3, F-03-4) ───────────────────────────────────────────────────────────────

def _payroll_args(**over):
    data = payroll.json.loads(payroll.DATA.read_text())
    args = {"tenant_id": data["tenant_id"], "worker_id": next(iter(data["workers"])), "unpaid_hours": 16, "start": "2026-11-02", "end": "2026-11-06", **over}
    return data, args


def test_f03_3_hours_that_do_not_fit_are_refused_not_dropped():
    data, args = _payroll_args(unpaid_hours=80, start="2026-11-02", end="2026-11-04")   # 3 working days = 24h
    e = raises(payroll.PayrollError, payroll.assess, data, args)
    assert "do not fit" in str(e)
    data, args = _payroll_args(unpaid_hours=24, start="2026-11-02", end="2026-11-04")
    out = payroll.assess(data, args)
    assert sum(p["unpaid_hours"] for p in out["pay_periods"]) == out["unpaid_hours"], "periods add up to the headline figure"


def test_f03_4_bad_inputs_are_refused_with_a_message():
    for bad in ({"unpaid_hours": float("nan")}, {"unpaid_hours": float("inf")}, {"unpaid_hours": [1]}, {"unpaid_hours": None}, {"start": ["x"]},
                {"end": {"a": 1}}, {"start": "2026-01-01", "end": "9999-12-31"}):
        data, args = _payroll_args(**bad)
        raises(payroll.PayrollError, payroll.assess, data, args)


# ── time-off agent (F-03-5, F-03-6) ──────────────────────────────────────────────────────────────

def test_f03_5_a_malformed_payroll_answer_does_not_undo_a_decision():
    from test_a2a import REVIEW, Stack, run

    s = Stack()
    for artifacts in ({}, {"payroll_impact": "text, not data"}, {"payroll_impact": None}):
        async def fake(message, artifacts=artifacts):
            return TaskResult("t", "c", "completed", "", None, artifacts)

        s.payroll.send = fake
        r = run(s.as_(AIKO).send(REVIEW))
        assert r.state == "input-required", (artifacts, r.state, r.text)
        assert r.data["payroll_impact"]["available"] is False and "no usable impact" in r.data["payroll_impact"]["reason"]


def test_f03_6_an_internal_failure_does_not_leak_its_text_to_the_caller():
    from test_a2a import REVIEW, Stack, run

    s = Stack()

    def broken(*a, **k):
        raise RuntimeError("select secret_column from payroll where token='abc'")

    s.ws.detail = broken
    r = run(s.as_(AIKO).send(REVIEW))
    assert r.state == "failed", r.state
    assert "secret_column" not in r.text and "RuntimeError" not in r.text and "could not complete" in r.text


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
