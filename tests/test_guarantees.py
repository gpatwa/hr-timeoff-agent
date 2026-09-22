"""Tests for the two claims this project actually makes.

Run: .venv/bin/python -m pytest tests/ -q     (or: .venv/bin/python tests/test_guarantees.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.types import Command  # noqa: E402

from hr_timeoff_agent import evidence, graph as graph_mod, policy  # noqa: E402
from hr_timeoff_agent.models import Decision  # noqa: E402


def _app_and_request(request_id="REQ-2001"):
    tenant = policy.Tenant()
    return graph_mod.build(tenant), tenant.requests[request_id]


def test_graph_halts_before_deciding():
    """The agent reaches a recommendation and stops. No decision is produced."""
    app, request = _app_and_request()
    cfg = {"configurable": {"thread_id": "t-halt"}}
    state = app.invoke(graph_mod.initial_state(request), config=cfg)

    assert "__interrupt__" in state, "graph should pause at approval_gate"
    assert state["recommendation"] is not None, "a recommendation should exist"
    assert state["decision"] is None, "the agent must not produce a decision"


def test_human_can_override_the_recommendation():
    """The human is the decider, not a rubber stamp for the agent."""
    app, request = _app_and_request("REQ-2004")  # agent recommends decline
    cfg = {"configurable": {"thread_id": "t-override"}}
    state = app.invoke(graph_mod.initial_state(request), config=cfg)
    assert state["recommendation"]["action"] == "decline"

    final = app.invoke(
        Command(resume={"outcome": "approved", "decided_by": "Aiko Tanaka", "note": "Unpaid leave agreed."}),
        config=cfg,
    )
    assert final["decision"]["outcome"] == "approved"
    assert final["decision"]["decided_by"] == "Aiko Tanaka"

    overrides = [
        e for e in final["evidence"]
        if e["actor"] == "human" and "recommended decline" in e["summary"]
    ]
    assert overrides, "the override should be visible in the evidence trail"


def test_decision_cannot_be_attributed_to_an_agent():
    """Decision pins actor_type to 'human' at the type level."""
    import pydantic

    try:
        Decision(outcome="approved", decided_by="bot", actor_type="agent", at="2026-01-01")
    except pydantic.ValidationError:
        return
    raise AssertionError("Decision accepted a non-human actor_type")


def test_evidence_chain_detects_tampering():
    """Editing a recorded step breaks verification."""
    app, request = _app_and_request()
    cfg = {"configurable": {"thread_id": "t-tamper"}}
    app.invoke(graph_mod.initial_state(request), config=cfg)
    final = app.invoke(
        Command(resume={"outcome": "approved", "decided_by": "Dana Whitfield", "note": ""}),
        config=cfg,
    )

    ok, reason = evidence.verify(final["evidence"])
    assert ok, f"clean chain should verify: {reason}"

    tampered = [dict(e) for e in final["evidence"]]
    tampered[1]["summary"] = "BAL-01 PASS: plenty of balance, nothing to see here"
    ok, reason = evidence.verify(tampered)
    assert not ok, "tampering should be detected"
    assert "entry 1" in reason


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
