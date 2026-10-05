"""Tests for the claims this project actually makes.

Run: .venv/bin/python -m pytest tests/ -q     (or: .venv/bin/python tests/test_guarantees.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.types import Command  # noqa: E402

from hr_timeoff_agent.agent import assembly, graph as graph_mod
from hr_timeoff_agent.core import evidence, policy  # noqa: E402
from hr_timeoff_agent.core.models import Decision  # noqa: E402

DANA = "W-100001"   # Priya's manager (REQ-2001, REQ-2005)
AIKO = "W-100236"   # Samuel's manager (REQ-2004); also the requester in REQ-2003
MARCUS = "W-100235"  # Priya's peer, not a manager


def _app_and_request(request_id="REQ-2001"):
    tenant = policy.Tenant()
    return assembly.build(tenant), tenant.requests[request_id]


def _paused(app, request, thread_id):
    cfg = {"configurable": {"thread_id": thread_id}}
    app.invoke(graph_mod.initial_state(request), config=cfg)
    return cfg


def _refused(result: dict) -> str:
    """A refused resume comes back paused again, carrying the reason."""
    assert "__interrupt__" in result, "expected the approval to be refused and the run to stay paused"
    assert result["decision"] is None, "nothing may be recorded on a refused approval"
    return result["__interrupt__"][0].value["refused"]


def _raises_permission(fn) -> str:
    try:
        fn()
    except PermissionError as exc:
        return str(exc)
    raise AssertionError("expected a PermissionError")


def test_graph_halts_before_deciding():
    """The agent reaches a recommendation and stops. No decision is produced."""
    app, request = _app_and_request()
    cfg = {"configurable": {"thread_id": "t-halt"}}
    state = app.invoke(graph_mod.initial_state(request), config=cfg)

    assert "__interrupt__" in state, "graph should pause at approval_gate"
    assert state["recommendation"] is not None, "a recommendation should exist"
    assert state["decision"] is None, "the agent must not produce a decision"


def test_human_can_override_the_recommendation():
    """The human is the decider, not a rubber stamp for the agent.

    Whatever the agent recommends, the manager chooses something else, and the
    trail shows both. (Does not depend on which action a given model picks.)
    """
    app, request = _app_and_request("REQ-2004")
    cfg = {"configurable": {"thread_id": "t-override"}}
    state = app.invoke(graph_mod.initial_state(request), config=cfg)
    recommended = state["recommendation"]["action"]
    outcome = "declined" if recommended == "approve" else "approved"

    final = app.invoke(
        Command(resume={"outcome": outcome, "decided_by_id": AIKO, "note": "Manager's call."}),
        config=cfg,
    )
    assert final["decision"]["outcome"] == outcome
    assert final["decision"]["decided_by"] == "Aiko Tanaka", "the name comes from the directory"

    overrides = [
        e for e in final["evidence"]
        if e["actor"] == "human" and f"recommended {recommended}" in e["summary"]
    ]
    assert overrides, "the override should be visible in the evidence trail"


def test_decision_cannot_be_attributed_to_an_agent():
    """Decision pins actor_type to 'human' at the type level."""
    import pydantic

    try:
        Decision(
            outcome="approved", decided_by_id=DANA, decided_by="bot", actor_type="agent", at="2026-01-01"
        )
    except pydantic.ValidationError:
        return
    raise AssertionError("Decision accepted a non-human actor_type")


def test_only_the_direct_manager_can_decide():
    """A peer cannot approve; the run stays paused and the manager can still decide."""
    app, request = _app_and_request("REQ-2001")  # Priya, managed by Dana
    cfg = _paused(app, request, "t-peer")

    reason = _refused(app.invoke(Command(resume={"outcome": "approved", "decided_by_id": MARCUS}), config=cfg))
    assert "not Priya Raman's direct manager" in reason
    pending = [t for t in app.get_state(cfg).tasks if t.name == "approval_gate" and t.interrupts]
    assert pending, "a refused approval must leave the run paused at approval_gate"

    # The refusal must not jam the run: the real manager can still decide.
    final = app.invoke(Command(resume={"outcome": "declined", "decided_by_id": DANA}), config=cfg)
    assert final["decision"]["decided_by_id"] == DANA
    assert final["decision"]["outcome"] == "declined"

    refused = [e for e in final["evidence"] if e["node"] == "approval_gate" and e["actor"] == "system"]
    assert len(refused) == 1 and refused[0]["data"]["attempted_by_id"] == MARCUS, (
        "the refused attempt should be in the evidence trail"
    )
    ok, _ = evidence.verify(final["evidence"])
    assert ok


def test_nobody_decides_their_own_request():
    """Aiko is a manager, but REQ-2003 is her own request."""
    app, request = _app_and_request("REQ-2003")
    cfg = _paused(app, request, "t-self")
    reason = _refused(app.invoke(Command(resume={"outcome": "approved", "decided_by_id": AIKO}), config=cfg))
    assert "cannot decide their own request" in reason


def test_unknown_approver_is_refused():
    app, request = _app_and_request("REQ-2001")
    cfg = _paused(app, request, "t-unknown")
    reason = _refused(app.invoke(Command(resume={"outcome": "approved", "decided_by_id": "W-999999"}), config=cfg))
    assert "is not a worker" in reason


def _record_node():
    tenant = policy.Tenant()
    *_, record = graph_mod.make_nodes(tenant)
    return tenant, record


def test_recorder_rechecks_the_approver():
    """Even if a decision reached `record` without passing the gate, it is refused."""
    tenant, record = _record_node()
    request = tenant.requests["REQ-2001"]
    state = {
        "request": request,
        "worker": tenant.workers[request["worker_id"]],
        "decision": Decision(
            outcome="approved", decided_by_id=MARCUS, decided_by="Marcus Vogel", at="2026-09-29"
        ).model_dump(),
        "evidence": [],
    }
    reason = _raises_permission(lambda: record(state))
    assert "Decision not recorded" in reason


def test_recorder_refuses_a_tampered_chain():
    """A valid approver is not enough: the evidence chain must verify before commit."""
    app, request = _app_and_request("REQ-2001")
    cfg = _paused(app, request, "t-chain")
    paused = app.get_state(cfg).values

    tenant, record = _record_node()
    tampered = [dict(e) for e in paused["evidence"]]
    tampered[1]["summary"] = "BAL-01 PASS: plenty of balance, nothing to see here"
    state = {
        **paused,
        "decision": Decision(
            outcome="approved", decided_by_id=DANA, decided_by="Dana Whitfield", at="2026-09-29"
        ).model_dump(),
        "evidence": tampered,
    }
    try:
        record(state)
    except RuntimeError as exc:
        assert "failed verification" in str(exc)
        return
    raise AssertionError("record committed despite a tampered evidence chain")


def test_evidence_chain_detects_tampering():
    """Editing a recorded step breaks verification."""
    app, request = _app_and_request()
    cfg = _paused(app, request, "t-tamper")
    final = app.invoke(
        Command(resume={"outcome": "approved", "decided_by_id": DANA, "note": ""}),
        config=cfg,
    )

    ok, reason = evidence.verify(final["evidence"])
    assert ok, f"clean chain should verify: {reason}"

    tampered = [dict(e) for e in final["evidence"]]
    tampered[1]["summary"] = "BAL-01 PASS: plenty of balance, nothing to see here"
    ok, reason = evidence.verify(tampered)
    assert not ok, "tampering should be detected"
    assert "entry 1" in reason


def test_a_shortfall_states_its_size_in_working_days_so_no_model_has_to_convert():
    t = policy.Tenant()
    worker = t.workers["W-100237"]   # PTO balance 40h
    base = {"request_id": "X", "worker_id": "W-100237", "plan": "PTO", "from": "2026-10-26", "to": "2026-11-06",
            "submitted_at": "2026-09-21", "note": ""}
    f = policy.check_balance(t, {**base, "hours": 80.0}, worker)
    assert f.status == "fail" and f.detail.endswith("by 40h (5 working days at 8h a day).")
    assert (f.evidence["shortfall_hours"], f.evidence["shortfall_working_days"]) == (40.0, 5.0)
    assert "4 working days" in policy.check_balance(t, {**base, "hours": 72.0}, worker).detail
    assert "0.5 working days" in policy.check_balance(t, {**base, "hours": 44.0}, worker).detail
    ok = policy.check_balance(t, {**base, "hours": 40.0}, worker)
    assert ok.status == "pass" and ok.detail == "Requested 40h against a PTO balance of 40h."
    assert ok.evidence["shortfall_hours"] == 0.0


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
