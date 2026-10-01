"""The web app, end to end through HTTP, offline.

Every permission is checked server-side, so these drive the real routes as
different people and assert what the server allows. New-request triage uses a
stubbed model call (the rest — graph, authorization, SQLite, retrieval — is
real); the seeded requests replay from the recorded fixtures.

Run: .venv/bin/python tests/test_web.py
"""

from __future__ import annotations

import html
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")

from fastapi.testclient import TestClient  # noqa: E402

from hr_timeoff_agent import graph as graph_mod, llm  # noqa: E402
from hr_timeoff_agent.models import Recommendation  # noqa: E402
from hr_timeoff_agent.web.app import create_app  # noqa: E402

PRIYA, MARCUS, AIKO, SAMUEL, DANA, GRACE = "W-100234", "W-100235", "W-100236", "W-100237", "W-100001", "W-100003"
# Same note as REQ-2001 and every rule passing, so the retrieval query is one
# whose embedding is already cached: no network, no model download.
ALL_PASS = {"start": "2026-11-30", "end": "2026-12-02", "hours": "", "note": "Family trip, booked months ago.", "plan": "PTO"}


def _fake_structured(*, system, user, schema, model=llm.AGENT_MODEL, record=False, label=""):
    """Stands in for a live model call, including the spend hooks a real one runs."""
    if llm.before_live_call:
        llm.before_live_call(model, label)
    if llm.after_live_call:
        llm.after_live_call(model, label, 0.01, "stub")
    return Recommendation(action="approve", rationale="Stub.", cited_rule_ids=["BAL-01"],
                          cited_passage_ids=[], confidence="high")


class App:
    def __init__(self, home: str | None = None, **kw):
        self.home = home or tempfile.mkdtemp()
        self.app = create_app(self.home, **kw)
        self.ws = self.app.state.workspace
        self.c = TestClient(self.app)

    def as_(self, worker_id: str) -> "App":
        self.c.cookies.clear()
        r = self.c.post("/signin", data={"worker_id": worker_id}, follow_redirects=False)
        assert r.status_code == 303
        return self

    def status(self, rid: str) -> str:
        return self.ws.request(rid)["status"]


def test_signed_out_visitors_are_sent_to_sign_in():
    a = App()
    r = a.c.get("/requests?scope=mine", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/signin"


def test_a_forged_identity_cookie_is_ignored():
    a = App()
    a.c.cookies.set("hr_persona", f"{DANA}.not-a-valid-signature")
    assert a.c.get("/requests?scope=inbox", follow_redirects=False).status_code == 303


def test_a_manager_sees_only_their_reports():
    a = App().as_(AIKO)
    ids = set(re.findall(r"REQ-\d+", a.c.get("/requests?scope=inbox").text))
    assert ids == {"REQ-2004"}, ids


def test_people_cannot_view_requests_that_are_not_theirs():
    a = App().as_(MARCUS)
    assert a.c.get("/requests/REQ-2004").status_code == 403
    assert a.c.get("/requests/REQ-2002").status_code == 200  # his own


def test_the_wrong_person_is_refused_and_the_request_stays_pending():
    a = App().as_(GRACE)  # HR can see it, but is not Samuel's manager
    r = a.c.post("/requests/REQ-2004/decide", data={"outcome": "approved"})
    assert r.status_code == 403 and "not Samuel Ortiz's direct manager" in html.unescape(r.text)
    assert a.status("REQ-2004") == "pending"
    # and the real manager can still decide afterwards
    a.as_(AIKO).c.post("/requests/REQ-2004/decide", data={"outcome": "approved"})
    assert a.status("REQ-2004") == "approved"


def test_nobody_approves_their_own_request():
    a = App().as_(AIKO)  # REQ-2003 is Aiko's own
    r = a.c.post("/requests/REQ-2003/decide", data={"outcome": "approved"})
    assert r.status_code == 403 and "cannot decide their own request" in r.text
    assert a.status("REQ-2003") == "pending"


def test_approval_deducts_balance_and_records_unpaid_leave():
    a = App().as_(AIKO)
    a.c.post("/requests/REQ-2004/decide", data={"outcome": "approved", "note": "40 paid, rest unpaid"})
    assert a.ws.worker(SAMUEL)["time_off_plans"]["PTO"]["balance_hours"] == 0.0
    absence = next(x for x in a.ws._read("absences.json") if x["absence_id"] == "ABS-REQ-2004")
    assert (absence["paid_hours"], absence["unpaid_hours"]) == (40.0, 80.0)
    page = a.c.get("/requests/REQ-2004").text
    assert "80h unpaid leave" in page and "chain verified" in page


def test_decline_changes_nothing_but_status():
    a = App().as_(DANA)
    a.c.post("/requests/REQ-2001/decide", data={"outcome": "declined"})
    assert a.status("REQ-2001") == "declined"
    assert a.ws.worker(PRIYA)["time_off_plans"]["PTO"]["balance_hours"] == 96.0
    assert "overrides the agent" in a.c.get("/requests/REQ-2001").text  # it recommended approve


def test_a_pending_approval_survives_a_restart():
    first = App()
    home = first.home
    first.ws.close()
    second = App(home).as_(AIKO)  # same files, new process state
    second.c.post("/requests/REQ-2004/decide", data={"outcome": "returned"})
    assert second.status("REQ-2004") == "returned"


def test_hr_sees_hr_only_guidance_and_managers_do_not():
    a = App().as_(GRACE)
    assert "HB-7.2" in a.c.get("/requests/REQ-2004").text
    assert "HB-7.2" not in a.as_(AIKO).c.get("/requests/REQ-2004").text


def test_only_admins_reach_admin_and_change_policy():
    a = App().as_(PRIYA)
    assert a.c.get("/admin").status_code == 403
    assert a.c.post("/admin/policy", data={"min_notice_days": "1", "min_available_pct": "0",
                                           "max_consecutive_days": "99", "blackouts": ""}).status_code == 403
    a.as_(GRACE).c.post("/admin/policy", data={"min_notice_days": "21", "min_available_pct": "60",
                                               "max_consecutive_days": "15", "blackouts": "Close | 2026-12-28 | 2027-01-06"})
    pol = a.ws.policy()
    assert pol["revision"] == 1
    assert next(r for r in pol["rules"] if r["id"] == "NOT-01")["min_notice_days"] == 21


def test_bad_policy_input_is_rejected():
    a = App().as_(GRACE)
    r = a.c.post("/admin/policy", data={"min_notice_days": "x", "min_available_pct": "60",
                                        "max_consecutive_days": "15", "blackouts": ""})
    assert r.status_code == 400 and a.ws.policy().get("revision") is None


def test_submit_validation():
    a = App().as_(PRIYA)
    bad = [
        {**ALL_PASS, "start": "2026-12-05", "end": "2026-12-01"},
        {**ALL_PASS, "hours": "500"},
        {**ALL_PASS, "start": "2026-12-05", "end": "2026-12-06"},  # a weekend
        {**ALL_PASS, "plan": "SABBATICAL"},
    ]
    for form in bad:
        assert a.c.post("/requests", data=form).status_code == 400, form
    assert len(a.ws.requests()) == 5, "nothing should have been saved"


def _with_stub(fn):
    real = graph_mod.structured
    graph_mod.structured = _fake_structured
    try:
        return fn()
    finally:
        graph_mod.structured = real


def test_an_employee_submission_is_triaged_and_reaches_their_manager():
    a = App().as_(PRIYA)
    r = _with_stub(lambda: a.c.post("/requests", data=ALL_PASS, follow_redirects=False))
    rid = r.headers["location"].split("/")[2].split("?")[0]
    assert a.status(rid) == "pending"
    assert rid in a.as_(DANA).c.get("/requests?scope=inbox").text
    a.c.post(f"/requests/{rid}/decide", data={"outcome": "approved"})
    assert a.ws.worker(PRIYA)["time_off_plans"]["PTO"]["balance_hours"] == 72.0


def test_the_daily_spend_cap_refuses_live_triage():
    a = App(daily_cap_usd=0.0).as_(PRIYA)
    r = _with_stub(lambda: a.c.post("/requests", data=ALL_PASS))
    assert r.status_code == 429 and "cap" in r.text
    rid = a.ws.requests()[-1]["request_id"]
    assert a.status(rid) == "needs_triage", "kept, and retryable once the cap allows"


def test_the_per_person_rate_limit():
    a = App(triages_per_hour=1).as_(PRIYA)
    assert _with_stub(lambda: a.c.post("/requests", data=ALL_PASS)).status_code == 200
    r = _with_stub(lambda: a.c.post("/requests", data={**ALL_PASS, "start": "2026-12-07", "end": "2026-12-09"}))
    assert r.status_code == 429 and "per person per hour" in r.text


def test_failed_triage_is_kept_and_can_be_retried():
    a = App().as_(PRIYA)
    a.c.post("/requests", data=ALL_PASS)  # offline, no stub: the model call can't happen
    rid = a.ws.requests()[-1]["request_id"]
    assert a.status(rid) == "needs_triage"
    assert "Retry triage" in a.c.get(f"/requests/{rid}").text
    _with_stub(lambda: a.c.post(f"/requests/{rid}/retriage"))
    assert a.status(rid) == "pending"


def test_reset_restores_the_seeded_tenant():
    a = App().as_(AIKO)
    a.c.post("/requests/REQ-2004/decide", data={"outcome": "approved"})
    a.as_(GRACE).c.post("/admin/reset")
    assert a.status("REQ-2004") == "pending"
    assert a.ws.worker(SAMUEL)["time_off_plans"]["PTO"]["balance_hours"] == 40.0
    assert len(a.ws.requests()) == 5


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
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
