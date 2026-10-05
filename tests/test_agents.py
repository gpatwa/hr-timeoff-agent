"""The multi-agent path: replayable trajectories, the tool allowlist, the API loop,
and the graph wiring. No network; the model is faked where a live call would be.

Run: .venv/bin/python tests/test_agents.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")

from pydantic import BaseModel  # noqa: E402

from hr_timeoff_agent.agent import agentloop, agents, assembly, graph as graph_mod
from hr_timeoff_agent.core import evidence, policy
from hr_timeoff_agent.adapters import llm, retrieval  # noqa: E402
from hr_timeoff_agent.agent.agentloop import AgentRun, StaleTrajectory, ToolCall, parse_claude_stream, run_agent  # noqa: E402
from hr_timeoff_agent.hr_tools.server import HRToolServer, digest  # noqa: E402
from hr_timeoff_agent.core.models import Recommendation  # noqa: E402

PRIYA = "W-100234"
HR_ONLY_TEXT = (
    "For requests over 15 days, check leave-of-absence eligibility, statutory family and "
    "medical leave entitlements, benefits continuation and the return-to-work date."
)
_INDEX = None


class Report(BaseModel):
    answer: str


def _server() -> HRToolServer:
    global _INDEX
    if _INDEX is None:
        _INDEX = retrieval.PolicyIndex()
    return HRToolServer(index=_INDEX, reader="manager")


class scratch_cache:
    """Points the fixture cache at an empty temp file for the duration of a test."""

    def __enter__(self):
        self._dir = tempfile.TemporaryDirectory()
        self._saved = llm.FIXTURES
        llm.FIXTURES = Path(self._dir.name) / "cache.json"
        return self

    def __exit__(self, *exc):
        llm.FIXTURES = self._saved
        self._dir.cleanup()


def _record(server, tools, call_args, *, system="s", user="u") -> str:
    """Hand-write a trajectory the way a live run would have recorded it."""
    from hr_timeoff_agent.hr_tools.client import call_tool

    trace = [
        {"tool": t, "arguments": a, "result_sha256": digest(call_tool(t, a, server))} for t, a in call_args
    ]
    key = agentloop.agent_key(llm.AGENT_MODEL, system, user, Report, tools)
    llm._save_cache({key: {"kind": "agent", "trace": trace, "response": {"answer": "ok"}, "source": "test", "served_by": ["m"]}})
    return key


def test_a_recorded_trajectory_replays_through_the_real_tools():
    with scratch_cache():
        server = _server()
        tools = ["get_balance", "get_worker"]
        _record(server, tools, [("get_balance", {"worker_id": PRIYA, "plan": "PTO"})])
        run = run_agent(system="s", user="u", schema=Report, tools=tools, server=server)
        assert run.replayed and run.output.answer == "ok"
        assert [c.tool for c in run.calls] == ["get_balance"] and run.calls[0].result["balance_hours"] == 96
        assert server.audit and server.audit[-1]["tool"] == "get_balance", "replay goes through the server"


def test_replay_notices_when_the_data_behind_a_result_changed():
    with scratch_cache():
        server = _server()
        tools = ["get_balance"]
        _record(server, tools, [("get_balance", {"worker_id": PRIYA, "plan": "PTO"})])
        server.tenant.workers[PRIYA]["time_off_plans"]["PTO"]["balance_hours"] = 12.0
        try:
            run_agent(system="s", user="u", schema=Report, tools=tools, server=server)
        except llm.OfflineCacheMiss as exc:
            assert "stale" in str(exc) and "get_balance" in str(exc)
        else:
            raise AssertionError("a trajectory whose result changed was replayed")


def test_a_recorded_call_outside_the_allowlist_is_rejected():
    with scratch_cache():
        server = _server()
        key = _record(server, ["get_balance", "get_worker"], [("get_worker", {"worker_id": PRIYA})])
        cache = llm._load_cache()
        cache[agentloop.agent_key(llm.AGENT_MODEL, "s", "u", Report, ["get_balance"])] = cache[key]
        llm._save_cache(cache)
        try:
            run_agent(system="s", user="u", schema=Report, tools=["get_balance"], server=server)
        except llm.OfflineCacheMiss as exc:
            assert "allowlist" in str(exc)
        else:
            raise AssertionError("a call outside the allowlist was replayed")


def test_offline_with_nothing_recorded_says_so():
    with scratch_cache():
        try:
            run_agent(system="s", user="u", schema=Report, tools=["get_balance"], server=_server())
        except llm.OfflineCacheMiss as exc:
            assert "no recorded trajectory" in str(exc)
        else:
            raise AssertionError("ran without a recording")


def test_the_allowlist_is_part_of_the_cache_key():
    a = agentloop.agent_key("m", "s", "u", Report, ["get_balance"])
    assert a != agentloop.agent_key("m", "s", "u", Report, ["get_balance", "get_worker"])
    assert a == agentloop.agent_key("m", "s", "u", Report, ["get_balance"])


def test_a_claude_code_stream_becomes_a_trace_and_ignores_the_schema_tool():
    stream = "\n".join(json.dumps(e) for e in [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "mcp__hr__get_balance", "input": {"worker_id": PRIYA, "plan": "PTO"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": '{"worker_id":"W-100234","plan":"PTO","balance_hours":96}'}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t2", "name": "mcp__hr__get_worker", "input": {"worker_id": "W-NOPE"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t2", "is_error": True, "content": [{"type": "text", "text": "not a worker"}]}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t3", "name": "StructuredOutput", "input": {"answer": "x"}}]}},
        {"type": "result", "subtype": "success", "structured_output": {"answer": "x"}, "modelUsage": {"m": {}}, "total_cost_usd": 0.01},
    ])
    final, calls = parse_claude_stream(stream)
    assert final["structured_output"] == {"answer": "x"}
    assert [c.tool for c in calls] == ["get_balance", "get_worker"]
    # 96 vs 96.0: the same digest however the number was serialized
    assert calls[0].result_sha256 == digest({"worker_id": PRIYA, "plan": "PTO", "balance_hours": 96.0})
    assert calls[1].error == "not a worker" and calls[1].result_sha256 is None


# ── the Messages-API loop, with the SDK faked ───────────────────────────────

def _msg(blocks, stop):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0)
    return SimpleNamespace(content=blocks, stop_reason=stop, model="claude-sonnet-5-5", usage=usage)


def _use(i, name, **inp):
    return SimpleNamespace(type="tool_use", id=f"tu{i}", name=name, input=inp)


class FakeAnthropic:
    script: list = []
    seen: list = []

    def __init__(self, *a, **k):
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        FakeAnthropic.seen.append(json.loads(json.dumps(kwargs["messages"], default=lambda o: getattr(o, "__dict__", str(o)))))
        return FakeAnthropic.script.pop(0)


def _live_api(script, tools, *, user="u"):
    import anthropic

    FakeAnthropic.script, FakeAnthropic.seen = list(script), []
    real = anthropic.Anthropic
    anthropic.Anthropic = FakeAnthropic
    saved = llm.BACKEND
    llm.BACKEND = "api"
    os.environ["ANTHROPIC_API_KEY"] = "test"
    os.environ.pop("HR_AGENT_OFFLINE", None)
    try:
        return run_agent(system="s", user=user, schema=Report, tools=tools, server=_server())
    finally:
        anthropic.Anthropic = real
        llm.BACKEND = saved
        os.environ["HR_AGENT_OFFLINE"] = "1"
        os.environ.pop("ANTHROPIC_API_KEY", None)


def test_the_api_loop_runs_tools_through_mcp_and_records_the_trajectory():
    with scratch_cache():
        run = _live_api([
            _msg([SimpleNamespace(type="text", text="Checking."), _use(1, "get_balance", worker_id=PRIYA, plan="PTO")], "tool_use"),
            _msg([SimpleNamespace(type="text", text='{"answer": "96h"}')], "end_turn"),
        ], ["get_balance"])
        assert run.output.answer == "96h" and not run.replayed
        assert run.calls[0].result["balance_hours"] == 96 and run.usd > 0
        sent = FakeAnthropic.seen[1][-1]["content"][0]
        assert sent["type"] == "tool_result" and sent["tool_use_id"] == "tu1" and "96" in sent["content"]
        entry = next(iter(llm._load_cache().values()))
        assert entry["trace"][0]["tool"] == "get_balance" and entry["response"] == {"answer": "96h"}
        # and the recording replays offline
        again = run_agent(system="s", user="u", schema=Report, tools=["get_balance"], server=_server())
        assert again.replayed and again.output.answer == "96h"


def test_the_api_loop_refuses_tools_outside_the_allowlist():
    with scratch_cache():
        run = _live_api([
            _msg([_use(1, "get_worker", worker_id=PRIYA)], "tool_use"),
            _msg([SimpleNamespace(type="text", text='{"answer": "done"}')], "end_turn"),
        ], ["get_balance"])
        result = FakeAnthropic.seen[1][-1]["content"][0]
        assert result["is_error"] and "not available" in result["content"]
        assert run.calls == [], "a refused call is not a recorded call"


def test_a_tool_error_is_shown_to_the_agent_not_raised():
    with scratch_cache():
        run = _live_api([
            _msg([_use(1, "get_balance", worker_id="W-NOPE", plan="PTO")], "tool_use"),
            _msg([SimpleNamespace(type="text", text='{"answer": "could not look it up"}')], "end_turn"),
        ], ["get_balance"])
        assert run.calls[0].error and "not a worker" in run.calls[0].error
        assert FakeAnthropic.seen[1][-1]["content"][0]["is_error"] is True


# ── the graph in multi-agent mode ───────────────────────────────────────────

def _fake_agents(monkey):
    """Stands in for the two specialists and the coordinator, keeping the real graph,
    ledger, tools and gate."""
    server = _server()

    def fake_run_agent(*, system, user, schema, tools, server, model=llm.AGENT_MODEL, record=False, label=""):
        from hr_timeoff_agent.agent.agentloop import _execute

        if "search_handbook" in tools:
            calls = _execute(server, [("search_handbook", {"query": HR_ONLY_TEXT}), ("get_worker", {"worker_id": "W-100236"})])
        else:
            calls = _execute(server, [("team_availability", {"worker_id": "W-100237", "start": "2026-10-19", "end": "2026-11-06"})])
        out = agents.SpecialistReport(summary="s", options=["use unpaid leave"], cited_rule_ids=["BAL-01"], cited_passage_ids=[], gaps=[])
        return AgentRun(out, calls, ["claude-sonnet-5-5"], 0.0, "test", True)

    def fake_structured(*, system, user, schema, model=llm.AGENT_MODEL, record=False, label=""):
        assert "SPECIALIST REPORTS" in user and "use unpaid leave" in user
        return Recommendation(action="escalate", rationale="Stub.", cited_rule_ids=["BAL-01"], cited_passage_ids=[], confidence="medium")

    monkey["run_agent"], agents.run_agent = agents.run_agent, fake_run_agent
    monkey["structured"], agents.structured = agents.structured, fake_structured


def _multi_run():
    tenant = policy.Tenant()
    tenant._policy_index = _INDEX or retrieval.PolicyIndex()
    saved: dict = {}
    _fake_agents(saved)
    try:
        app = assembly.build(tenant, agents="multi")
        state = app.invoke(graph_mod.initial_state(tenant.requests["REQ-2004"]), config={"configurable": {"thread_id": "t-multi"}})
        return tenant, app, state
    finally:
        agents.run_agent, agents.structured = saved["run_agent"], saved["structured"]


def test_multi_agent_run_pauses_at_the_gate_with_every_tool_call_in_the_ledger():
    tenant, app, state = _multi_run()
    assert "__interrupt__" in state and state["decision"] is None, "specialists must not decide"
    nodes = [e["node"] for e in state["evidence"]]
    assert nodes.index("policy_specialist") < nodes.index("coverage_specialist") < nodes.index("assess")
    calls = [e for e in state["evidence"] if e["data"].get("via") == "mcp"]
    assert [(c["data"]["agent"], c["data"]["tool"]) for c in calls] == [
        ("policy_specialist", "search_handbook"), ("policy_specialist", "get_worker"), ("coverage_specialist", "team_availability"),
    ]
    assert all(c["actor"] == "agent" and c["data"]["tenant_id"] == "TEN-001" and c["data"]["reader"] == "manager" for c in calls)
    assert all(c["data"]["result_sha256"] for c in calls)
    assert evidence.verify(state["evidence"])[0]


def test_multi_agent_passages_are_exactly_what_the_tools_returned():
    _, _, state = _multi_run()
    ids = {p["passage_id"] for p in state["passages"]}
    assert ids and "HB-7.2" not in ids, "the manager-reader server must not surface HR-only guidance"
    assert {p["tenant_id"] for p in state["passages"]} == {"TEN-001"}


def test_only_the_direct_manager_can_resume_a_multi_agent_run():
    from langgraph.types import Command

    tenant, app, _ = _multi_run()
    cfg = {"configurable": {"thread_id": "t-multi"}}
    refused = app.invoke(Command(resume={"outcome": "approved", "decided_by_id": "W-100003"}), config=cfg)
    assert "__interrupt__" in refused and refused["decision"] is None
    done = app.invoke(Command(resume={"outcome": "returned", "decided_by_id": "W-100236", "note": "x"}), config=cfg)
    assert done["decision"]["decided_by_id"] == "W-100236" and evidence.verify(done["evidence"])[0]


def test_single_agent_is_still_the_default():
    tenant = policy.Tenant()
    assert assembly.agent_mode() == "single" or os.environ.get("HR_AGENT_MODE")
    try:
        assembly.build(tenant, agents="swarm")
    except ValueError:
        return
    raise AssertionError("an unknown mode was accepted")


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
        except Exception as exc:  # a crash is a failure too, with its type
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
