"""The eval gate's own logic, offline: thresholds, metering, refusal to pass untested, plumbing.

The live gate needs a real model and runs from CI (.github/workflows/eval-gate.yml); this checks
everything about it that does not.

Run: .venv/bin/python tests/test_gate.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["HR_AGENT_OFFLINE"] = "1"
for var in ("HR_DATABASE_URL", "HR_OIDC_ISSUER", "ANTHROPIC_API_KEY"):
    os.environ.pop(var, None)

from hr_timeoff_agent.agent import agents
from hr_timeoff_agent.adapters import llm  # noqa: E402
from hr_timeoff_agent.eval_harness import gate  # noqa: E402
from hr_timeoff_agent.agent.agentloop import AgentRun  # noqa: E402
from hr_timeoff_agent.cli import main  # noqa: E402
from hr_timeoff_agent.core.models import Recommendation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = json.loads((ROOT / "evals" / "thresholds.json").read_text())
HR_ONLY_TEXT = (
    "For requests over 15 days, check leave-of-absence eligibility, statutory family and "
    "medical leave entitlements, benefits continuation and the return-to-work date."
)


def metric(mode="single", **kw):
    base = dict(mode=mode, case_id="EV-01", expected="approve", actual="approve", action_match=True, never_self_approved=True,
                scores={"rationale_grounded": 3, "citations_correct": 3, "tone_appropriate": 3, "justification": "x"},
                error=None, agent_usd=0.01, judge_usd=0.05, triage_seconds=5.0, tool_calls=0)
    base.update(kw)
    return gate.CaseMetrics(**base)


def test_the_thresholds_file_covers_every_mode_and_every_limit_the_gate_checks():
    assert set(THRESHOLDS["modes"]) == {"single", "multi"}
    expected = {"min_action_match_rate", "min_never_self_approved_rate", "min_no_approve_on_blocking_rate", "min_cites_failures_rate",
                "min_injection_resisted_rate", "min_rationale_grounded", "min_citations_correct", "min_tone_appropriate",
                "max_mean_agent_usd_per_case", "max_triage_seconds"}
    for mode, limits in THRESHOLDS["modes"].items():
        assert expected == set(limits), (mode, expected ^ set(limits))
    assert {"max_usd", "max_seconds", "min_tool_calls"} == set(THRESHOLDS["a2a_multi"])
    assert THRESHOLDS["total_budget_usd"] > 0


def test_a_run_that_meets_every_threshold_passes():
    rows = [metric(case_id=f"EV-0{i}") for i in range(1, 6)] + [metric("multi", case_id=f"EV-0{i}", tool_calls=4) for i in range(1, 6)]
    a2a = {"passed": True, "detail": "ok", "usd": 0.1, "seconds": 30.0, "tool_calls": 4}
    assert gate.evaluate(gate.summarize(rows), a2a, THRESHOLDS) == []


def test_each_kind_of_regression_is_named():
    rows = [metric(case_id=f"EV-0{i}", action_match=(i > 2), never_self_approved=(i != 1), agent_usd=0.5, triage_seconds=500.0,
                   no_approve_on_blocking=(i != 2), cites_failures=(i > 2), attack=(i >= 4), injection_resisted=(i == 5),
                   scores={"rationale_grounded": 1, "citations_correct": 1, "tone_appropriate": 1, "justification": "x"}) for i in range(1, 6)]
    bad = " | ".join(gate.evaluate(gate.summarize(rows), None, THRESHOLDS))
    lim = THRESHOLDS["modes"]["single"]
    for fragment in (f"action_match_rate 0.6 is below the floor {lim['min_action_match_rate']}",
                     f"never_self_approved_rate 0.8 is below the floor {lim['min_never_self_approved_rate']}",
                     f"no_approve_on_blocking_rate 0.8 is below the floor {lim['min_no_approve_on_blocking_rate']}",
                     f"cites_failures_rate 0.6 is below the floor {lim['min_cites_failures_rate']}",
                     f"injection_resisted_rate 0.5 is below the floor {lim['min_injection_resisted_rate']} (fell to: EV-04)", "rationale_grounded 1.0",
                     "citations_correct", "tone_appropriate", "per triage is over the ceiling", "the slowest triage took 500.0 s"):
        assert fragment in bad, (fragment, bad)


def test_a_case_that_errored_fails_the_gate_even_if_the_rest_look_fine():
    rows = [metric(case_id=f"EV-0{i}") for i in range(1, 5)] + [metric(case_id="EV-05", actual=None, action_match=False, scores=None, error="TimeoutError: model")]
    bad = gate.evaluate(gate.summarize(rows), None, THRESHOLDS)
    assert any("a case errored" in v and "EV-05" in v for v in bad), bad


def test_the_a2a_result_is_judged_on_passing_cost_time_and_tool_use():
    ok = {"passed": True, "detail": "d", "usd": 0.05, "seconds": 20.0, "tool_calls": 3}
    assert gate.evaluate({}, ok, THRESHOLDS) == []
    assert any("did not pass" in v or "boom" in v for v in gate.evaluate({}, {"passed": False, "detail": "boom"}, THRESHOLDS))
    assert any("over the ceiling" in v for v in gate.evaluate({}, {**ok, "usd": 9.0}, THRESHOLDS))
    assert any("over the ceiling" in v for v in gate.evaluate({}, {**ok, "seconds": 9999.0}, THRESHOLDS))
    assert any("only 0 tool calls" in v for v in gate.evaluate({}, {**ok, "tool_calls": 0}, THRESHOLDS))


def test_the_meter_separates_agent_from_judge_spend_and_stops_at_the_budget():
    m = gate.Meter(budget_usd=0.10)
    m.before("m", "assess:REQ-1")
    m.after("m", "assess:REQ-1", 0.02, "t")
    m.before("m", "judge:EV-01")
    m.after("m", "judge:EV-01", 0.05, "t")
    assert (round(m.agent_usd, 3), round(m.judge_usd, 3), round(m.spent, 3)) == (0.02, 0.05, 0.07)
    assert m.triage_seconds >= 0.0
    m.after("m", "coordinator:REQ-1", 0.04, "t")
    try:
        m.before("m", "judge:EV-02")
    except gate.OverBudget as exc:
        assert "budget" in str(exc) and "judge:EV-02" in str(exc)
    else:
        raise AssertionError("the meter let a call through after the budget was spent")
    m.reset_case()
    assert (m.agent_usd, m.judge_usd, m.tool_calls) == (0.0, 0.0, 0)


def test_a_live_gate_with_no_model_refuses_instead_of_passing():
    try:
        gate.run_gate(modes=["single"], repeats=1)
    except RuntimeError as exc:
        assert "needs a model" in str(exc) and "--replay" in str(exc)
    else:
        raise AssertionError("a live gate ran with no model available")
    assert main(["gate"]) == 2


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_replay_gate_checks_the_plumbing_against_the_fixtures_and_never_edits_them():
    watched = [ROOT / "fixtures" / "llm_cache.json", ROOT / "fixtures" / "embeddings.json"]
    before = [_digest(p) for p in watched]
    report = gate.run_gate(modes=["single", "multi"], repeats=1, replay=True)
    assert report["passed"] and not report["live"] and report["a2a"] is None
    n = report["summary"]["single"]["cases"]
    assert n >= 40 and report["summary"]["single"]["action_match"] == n and report["summary"]["multi"]["never_self_approved"] == n
    assert report["summary"]["single"]["injection_resisted"] == report["summary"]["single"]["attacks"] >= 14
    assert report["summary"]["multi"]["mean_tool_calls"] > 0, "replayed trajectories still run their tools"
    assert [_digest(p) for p in watched] == before
    assert "replay: no model was called" in gate.render_markdown(report)


def test_child_processes_are_pointed_at_scratch_caches_during_a_run_and_restored_after():
    """Claude Code starts the MCP server as a child process; it learns where to record new
    embeddings from the environment, and a gate run must never aim it at the repo's file."""
    seen = {}
    real = gate.run_eval

    def spy(*a, **kw):
        seen.update({k: os.environ.get(k) for k in ("HR_AGENT_EMBEDDINGS", "HR_AGENT_FIXTURES")})
        return real(*a, **kw)

    before = {k: os.environ.get(k) for k in seen or ("HR_AGENT_EMBEDDINGS", "HR_AGENT_FIXTURES")}
    gate.run_eval = spy
    try:
        gate.run_gate(modes=["single"], repeats=1, replay=True)
    finally:
        gate.run_eval = real
    for k, v in seen.items():
        assert v and str(ROOT / "fixtures") not in v, (k, v)
    assert {k: os.environ.get(k) for k in before} == before


def test_the_cli_exit_codes_are_pass_zero_fail_one_cannot_run_two():
    with tempfile.TemporaryDirectory() as d:
        strict = json.loads(json.dumps(THRESHOLDS))
        strict["modes"]["single"]["min_citations_correct"] = 3.0   # the recorded single run scores below this
        path = Path(d) / "strict.json"
        path.write_text(json.dumps(strict))
        out = Path(d) / "report.json"
        assert main(["gate", "--replay", "--out", str(out), "--summary", str(Path(d) / "s.md")]) == 0
        assert json.loads(out.read_text())["passed"] is True and (Path(d) / "s.md").read_text().startswith("## Eval gate: PASS")
        assert main(["gate", "--replay", "--thresholds", str(path), "--out", str(out)]) == 1
        failed = json.loads(out.read_text())
        assert failed["passed"] is False and any("citations_correct" in v for v in failed["violations"])
    assert main(["gate", "--replay", "--modes", "bogus"]) == 2


# ── the multi-agent run through the A2A agents, with the model stubbed ──────

def _stub_agents(*, coverage_uses_tools: bool):
    from hr_timeoff_agent.agent.agentloop import _execute

    def fake_run_agent(*, system, user, schema, tools, server, model=llm.AGENT_MODEL, record=False, label=""):
        if "search_handbook" in tools:
            calls = _execute(server, [("search_handbook", {"query": HR_ONLY_TEXT}), ("get_worker", {"worker_id": "W-100234"})])
        elif coverage_uses_tools:
            calls = _execute(server, [("team_availability", {"worker_id": "W-100234", "start": "2026-12-07", "end": "2026-12-25"})])
        else:
            calls = []
        out = agents.SpecialistReport(summary="s", options=["use unpaid leave"], cited_rule_ids=["BAL-01"], cited_passage_ids=[], gaps=[])
        return AgentRun(out, calls, [llm.AGENT_MODEL], 0.0, "test", True)

    def fake_structured(*, system, user, schema, model=llm.AGENT_MODEL, record=False, label=""):
        return Recommendation(action="escalate", rationale="Stub: 24h over balance.", cited_rule_ids=["BAL-01"], cited_passage_ids=[], confidence="medium")

    return fake_run_agent, fake_structured


def _a2a_run(coverage_uses_tools: bool) -> dict:
    run, structured = _stub_agents(coverage_uses_tools=coverage_uses_tools)
    saved = (agents.run_agent, agents.structured)
    agents.run_agent, agents.structured = run, structured
    try:
        with tempfile.TemporaryDirectory() as d:
            return gate.run_a2a_multi(Path(d))
    finally:
        agents.run_agent, agents.structured = saved


def test_the_a2a_multi_agent_check_passes_when_the_whole_path_works():
    r = _a2a_run(coverage_uses_tools=True)
    assert r["passed"], r
    assert r["tool_calls"] == 3 and r["recommendation"] == "escalate" and "payroll consulted" in r["detail"]


def test_the_a2a_multi_agent_check_catches_a_specialist_that_never_used_its_tools():
    r = _a2a_run(coverage_uses_tools=False)
    assert not r["passed"] and "not both specialists" in r["detail"], r


def test_parallel_cases_give_the_same_results_and_quick_mode_runs_the_spread():
    serial = gate.run_gate(modes=["single"], repeats=1, replay=True)
    par = gate.run_gate(modes=["single"], repeats=1, replay=True, workers=6)
    key = lambda r: sorted((c["case_id"], c["actual"], c["injection_resisted"]) for c in r["cases"])  # noqa: E731
    assert par["passed"] and key(par) == key(serial) and par["workers"] == 6
    quick = gate.run_gate(modes=["single"], repeats=1, replay=True, workers=4, quick=True)
    assert quick["quick"] and sorted(c["case_id"] for c in quick["cases"]) == sorted(gate.QUICK_CASES)


def test_the_meter_keeps_each_threads_case_apart_but_one_shared_total():
    from concurrent.futures import ThreadPoolExecutor

    m = gate.Meter(budget_usd=100)

    def case(n):
        m.reset_case()
        for _ in range(50):
            m.before("m", "agent")
            m.after("m", "agent", 0.01 * n, "x")
        return round(m.agent_usd, 2), m.calls

    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(case, [1, 2, 3, 4]))
    assert got == [(0.5, 50), (1.0, 50), (1.5, 50), (2.0, 50)] and round(m.spent, 2) == 5.0


def test_concurrent_recordings_do_not_lose_each_others_fixtures():
    from concurrent.futures import ThreadPoolExecutor

    saved = llm.FIXTURES
    with tempfile.TemporaryDirectory() as d:
        llm.FIXTURES = Path(d) / "c.json"
        try:
            with ThreadPoolExecutor(8) as pool:
                list(pool.map(lambda i: llm.store(f"k{i}", {"n": i}), range(60)))
            assert len(json.loads(llm.FIXTURES.read_text())) == 60
        finally:
            llm.FIXTURES = saved


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
            print(f"  FAIL  {name}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
