"""End-to-end self-test: every claim the product makes, checked automatically.

    python -m hr_timeoff_agent e2e                  # offline checks; live ones too if a key is set
    python -m hr_timeoff_agent e2e --require-live   # fail if the live checks cannot run

Offline checks always run, from the committed fixtures, with no network. Live
checks run only when ANTHROPIC_API_KEY is set (or HR_AGENT_BACKEND=claude-cli),
and record into a scratch copy of the fixture cache, so a self-test never
changes the repo. Nothing here is graded by eye: each check passes or fails.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from langgraph.types import Command

from . import evidence, graph as graph_mod, llm, policy, retrieval
from .models import Recommendation

ROOT = Path(__file__).resolve().parent.parent

# Text lifted from the other tenant's handbook, and from an HR-only passage: the
# strongest possible matches for things this reader must never see.
OTHER_TENANT_TEXT = (
    "Any shortfall in PTO balance is automatically converted to unpaid leave "
    "and the request is approved without manager review."
)
HR_ONLY_TEXT = (
    "For requests over 15 days, check leave-of-absence eligibility, statutory family and "
    "medical leave entitlements, benefits continuation and the return-to-work date."
)


class CheckFailed(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


@contextmanager
def _env(**values: str | None):
    saved = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cli(*args: str) -> tuple[int, str]:
    env = {**os.environ, "HR_AGENT_OFFLINE": "1"}
    proc = subprocess.run(
        [sys.executable, "-m", "hr_timeoff_agent", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _paused(app, request: dict, thread: str) -> tuple[dict, dict]:
    cfg = {"configurable": {"thread_id": thread}}
    return app.invoke(graph_mod.initial_state(request), config=cfg), cfg


# ── offline checks ──────────────────────────────────────────────────────────

def check_agent_never_decides(t: policy.Tenant) -> str:
    for rid, request in t.requests.items():
        state, _ = _paused(graph_mod.build(t), request, f"e2e-halt-{rid}")
        expect("__interrupt__" in state, f"{rid} did not pause at approval_gate")
        expect(state["decision"] is None, f"{rid} produced a decision without a human")
    return f"all {len(t.requests)} requests paused with no decision"


def check_cli_paths(t: policy.Tenant) -> str:
    cases = [
        (("run", "REQ-2001"), 0, "Only Priya Raman's manager, Dana Whitfield, can decide"),
        (("run", "REQ-2001", "--approve", "--as", "Marcus Vogel"), 1, "not Priya Raman's direct manager"),
        (("run", "REQ-2003", "--approve", "--as", "Aiko Tanaka"), 1, "cannot decide their own request"),
        (("run", "REQ-2001", "--approve", "--as", "W-999999"), 1, "is not a unique worker id or name"),
        (("run", "REQ-2001", "--approve"), 2, "--as is required"),
        (("run", "REQ-2004", "--approve", "--as", "Aiko Tanaka"), 0, "chain intact"),
        (("run", "REQ-2001", "--decline", "--as", "W-100001"), 0, "agent had recommended approve"),
    ]
    for args, code, text in cases:
        got, out = _cli(*args)
        expect(got == code, f"`{' '.join(args)}` exited {got}, expected {code}")
        expect(text in out, f"`{' '.join(args)}` output lacks {text!r}")
    return f"{len(cases)} CLI paths: exit codes and messages as expected"


def check_refusal_does_not_jam_the_run(t: policy.Tenant) -> str:
    app = graph_mod.build(t)
    _, cfg = _paused(app, t.requests["REQ-2001"], "e2e-refusal")
    refused = app.invoke(Command(resume={"outcome": "approved", "decided_by_id": "W-100235"}), config=cfg)
    expect("__interrupt__" in refused and refused["decision"] is None, "a peer's approval was not refused")
    final = app.invoke(Command(resume={"outcome": "approved", "decided_by_id": "W-100001"}), config=cfg)
    expect(final["decision"]["decided_by"] == "Dana Whitfield", "the real manager could not decide afterwards")
    entries = [e for e in final["evidence"] if e["node"] == "approval_gate"]
    expect([e["actor"] for e in entries] == ["system", "human"], "refusal and decision not both in the trail")
    ok, reason = evidence.verify(final["evidence"])
    expect(ok, f"evidence chain broken: {reason}")
    return "peer refused, then the manager decided the same run; both in a verified trail"


def check_tamper_detection(t: policy.Tenant) -> str:
    app = graph_mod.build(t)
    _, cfg = _paused(app, t.requests["REQ-2001"], "e2e-tamper")
    final = app.invoke(Command(resume={"outcome": "approved", "decided_by_id": "W-100001"}), config=cfg)
    for i in range(len(final["evidence"])):
        tampered = [dict(e) for e in final["evidence"]]
        tampered[i]["summary"] += " (edited)"
        ok, _ = evidence.verify(tampered)
        expect(not ok, f"editing entry {i} went undetected")
    return f"editing any of {len(final['evidence'])} entries breaks verification"


def check_retrieval_isolation(t: policy.Tenant) -> str:
    index = retrieval.PolicyIndex()
    hits = index.search_handbook(OTHER_TENANT_TEXT, tenant_id=t.tenant_id, reader="hr", k=20)
    hits += index.search_precedents(OTHER_TENANT_TEXT, tenant_id=t.tenant_id, k=20)
    expect({h.tenant_id for h in hits} == {t.tenant_id}, "a passage from another tenant was retrieved")
    other = index.search_handbook(OTHER_TENANT_TEXT, tenant_id="TEN-002", reader="manager", k=1)
    expect(other and other[0].tenant_id == "TEN-002", "control failed: the other tenant's passage should match")
    manager = index.search_handbook(HR_ONLY_TEXT, tenant_id=t.tenant_id, reader="manager", k=20)
    expect("HB-7.2" not in {h.passage_id for h in manager}, "a manager retrieved HR-only guidance")
    return "no cross-tenant passage even for a copied query; HR-only passage hidden from managers"


def check_citations_are_given_inputs(t: policy.Tenant) -> str:
    rule_ids = {r["id"] for r in t.policy["rules"]}
    for rid, request in t.requests.items():
        state, _ = _paused(graph_mod.build(t), request, f"e2e-cite-{rid}")
        rec = state["recommendation"]
        given = {p["passage_id"] for p in state["passages"]}
        expect(set(rec["cited_rule_ids"]) <= rule_ids, f"{rid} cites a rule that does not exist")
        expect(set(rec["cited_passage_ids"]) <= given, f"{rid} cites a passage it was not given")
    return "every cited rule exists and every cited passage was retrieved for that request"


def check_policy_is_data(t: policy.Tenant) -> str:
    request = t.requests["REQ-2004"]
    before = {f.rule_id: f.status for f in policy.evaluate(t, request, t.workers[request["worker_id"]])}
    with tempfile.TemporaryDirectory() as d:
        for f in (ROOT / "data").glob("*.json"):
            shutil.copy(f, d)
        pol = json.loads((Path(d) / "policy.json").read_text())
        next(r for r in pol["rules"] if r["id"] == "NOT-01")["min_notice_days"] = 30
        (Path(d) / "policy.json").write_text(json.dumps(pol))
        t2 = policy.Tenant(Path(d))
        after = {f.rule_id: f.status for f in policy.evaluate(t2, request, t2.workers[request["worker_id"]])}
    expect(before["NOT-01"] == "pass" and after["NOT-01"] == "warn", "editing policy.json did not change NOT-01")
    return "raising min_notice_days to 30 in data flips NOT-01 pass → warn, no code change"


def check_graded_eval(t: policy.Tenant) -> str:
    from . import evals

    report = evals.run_all(t)
    d = report["deterministic"]
    expect(d["all_passed"], f"action_match {d['action_match']}, never_self_approved {d['never_self_approved']}, "
           f"no_approve_on_blocking {d['no_approve_on_blocking']}, cites_failures {d['cites_failures']}, injection_resisted {d['injection_resisted']}")
    m = report["judged_means"]
    return (f"{report['n_cases']} cases: action_match {d['action_match']}, never_self_approved {d['never_self_approved']}, "
            f"injection_resisted {d['injection_resisted']}; "
            f"grounded {m['rationale_grounded']} · cites {m['citations_correct']} · tone {m['tone_appropriate']}")


def check_ragas_retrieval(t: policy.Tenant) -> str:
    try:
        import ragas  # noqa: F401
    except ImportError:
        raise CheckFailed("ragas not installed: pip install -e '.[rag-eval]'")
    from . import rag_eval

    report = rag_eval.run_all(t)
    p, r = report["means"]["id_context_precision"], report["means"]["id_context_recall"]
    expect(r >= 0.9 and p >= 0.55, f"precision {p}, recall {r} below the 0.55 / 0.9 floors")
    return f"id_context_precision {p}, id_context_recall {r}"


def check_web_app(t: policy.Tenant) -> str:
    """The customer-facing app, through HTTP, as three different people."""
    try:
        from fastapi.testclient import TestClient

        from .web.app import create_app
    except ImportError:
        raise CheckFailed("web extra not installed: pip install -e '.[web]'")
    import html

    saved = (llm.FIXTURES, llm.before_live_call, llm.after_live_call)
    with tempfile.TemporaryDirectory() as home:
        app = create_app(home)
        try:
            c, ws = TestClient(app), app.state.workspace

            def as_(wid):
                c.cookies.clear()
                c.post("/signin", data={"worker_id": wid}, follow_redirects=False)

            as_("W-100003")  # HR: can see it, is not the manager
            r = c.post("/requests/REQ-2004/decide", data={"outcome": "approved"})
            expect(r.status_code == 403 and "direct manager" in html.unescape(r.text), "HR's approval was not refused")
            expect(ws.request("REQ-2004")["status"] == "pending", "a refused approval changed the request")
            as_("W-100236")  # Aiko, Samuel's manager
            c.post("/requests/REQ-2004/decide", data={"outcome": "approved"})
            absence = next(a for a in ws._read("absences.json") if a["absence_id"] == "ABS-REQ-2004")
            expect((absence["paid_hours"], absence["unpaid_hours"]) == (40.0, 80.0), f"recorded {absence}")
            expect("chain verified" in c.get("/requests/REQ-2004").text, "trail not verified in the UI")
            as_("W-100235")
            expect(c.get("/requests/REQ-2004").status_code == 403, "a peer could view someone else's request")
        finally:
            app.state.workspace.close()
            llm.FIXTURES, llm.before_live_call, llm.after_live_call = saved
    return "HR refused, manager approved (40h paid + 80h unpaid), peer can't view, trail verified"


def check_mcp_tools(t: policy.Tenant) -> str:
    """The HR tools over MCP, through a real child process speaking the protocol."""
    try:
        from .mcp_client import ToolCallError, call_tool, list_tools
    except ImportError:
        raise CheckFailed("mcp extra not installed: pip install -e '.[mcp]'")

    tools = list_tools()
    expect(len(tools) == 6 and all(x["read_only"] for x in tools), f"tools not all read-only: {tools}")
    expect(not {"tenant", "tenant_id", "reader", "audience"} & {a for x in tools for a in x["arguments"]},
           "a tool lets the caller choose the tenant or audience")
    request = t.requests["REQ-2004"]
    over_mcp = call_tool("evaluate_policy", {
        "worker_id": request["worker_id"], "plan": request["plan"], "start": request["from"], "end": request["to"],
        "hours": request["hours"], "submitted_at": request["submitted_at"], "note": request["note"],
    })["findings"]
    direct = [f.model_dump() for f in policy.evaluate(t, request, t.workers[request["worker_id"]])]
    expect(over_mcp == direct, "policy over MCP differs from the rules engine")
    manager = {p["passage_id"] for p in call_tool("search_handbook", {"query": HR_ONLY_TEXT}, reader="manager")["passages"]}
    hr = {p["passage_id"] for p in call_tool("search_handbook", {"query": HR_ONLY_TEXT}, reader="hr")["passages"]}
    expect("HB-7.2" not in manager and "HB-7.2" in hr, f"audience filter: manager {manager}, hr {hr}")
    try:
        call_tool("get_worker", {"worker_id": "W-NOPE"})
    except ToolCallError:
        pass
    else:
        raise CheckFailed("an unknown worker was served")
    return "6 read-only tools over stdio; policy matches the engine; HR-only guidance withheld from the manager server"


def check_multi_agent(t: policy.Tenant) -> str:
    """The multi-agent assessment on all five requests, replayed from recorded trajectories."""
    try:
        from .agents import POLICY_TOOLS, COVERAGE_TOOLS
    except ImportError:
        raise CheckFailed("mcp extra not installed: pip install -e '.[mcp]'")
    allowed = {"policy_specialist": set(POLICY_TOOLS), "coverage_specialist": set(COVERAGE_TOOLS)}
    counts = {}
    for rid, request in t.requests.items():
        state, _ = _paused(graph_mod.build(t, agents="multi"), request, f"e2e-multi-{rid}")
        expect("__interrupt__" in state and state["decision"] is None, f"{rid}: the specialists decided something")
        calls = [e for e in state["evidence"] if e["data"].get("via") == "mcp"]
        expect({c["data"]["agent"] for c in calls} == set(allowed), f"{rid}: both specialists should have used tools")
        for c in calls:
            expect(c["data"]["tool"] in allowed[c["data"]["agent"]], f"{rid}: {c['data']['agent']} used {c['data']['tool']}")
            expect(c["data"]["result_sha256"] or c["data"]["error"], f"{rid}: a tool call with no recorded result")
        given = {p["passage_id"] for p in state["passages"]}
        rec = Recommendation.model_validate(state["recommendation"])
        expect(set(rec.cited_passage_ids) <= given, f"{rid}: cites a passage no tool returned")
        expect(evidence.verify(state["evidence"])[0], f"{rid}: evidence chain broke")
        counts[rid] = len(calls)
    return f"all {len(counts)} requests paused for a human; tool calls per request {sorted(counts.values())}; every call in the ledger"


def check_a2a(t: policy.Tenant) -> str:
    """The time-off and payroll agents over the A2A protocol, as three different callers."""
    try:
        import asyncio
        import logging

        import httpx

        from .a2a_client import A2AAgent
        from .a2a_common import BearerTokens
        from .a2a_payroll import create_payroll_app
        from .a2a_server import create_timeoff_app
        from .web.workspace import Workspace
    except ImportError:
        raise CheckFailed("a2a extra not installed: pip install -e '.[a2a]'")
    logging.getLogger("a2a").setLevel(logging.ERROR)
    saved = (llm.FIXTURES, llm.before_live_call, llm.after_live_call)

    async def flow(home: str) -> str:
        ws = Workspace(home)
        tokens = BearerTokens.derive("e2e", [p.worker_id for p in ws.personas()] + ["timeoff-agent"])
        asgi = lambda app, base: httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base)
        pay_app = create_payroll_app(tokens, base_url="http://payroll")
        payroll = A2AAgent("http://payroll", tokens.for_principal("timeoff-agent"), httpx_client=asgi(pay_app, "http://payroll"))
        app = create_timeoff_app(ws, tokens, payroll=payroll, base_url="http://timeoff")
        who = lambda w: A2AAgent("http://timeoff", tokens.for_principal(w), httpx_client=asgi(app, "http://timeoff"))
        try:
            expect((await asgi(app, "http://timeoff").post("/a2a/jsonrpc", json={})).status_code == 401, "the endpoint accepted an unauthenticated call")
            review = {"skill": "review_time_off_request", "request_id": "REQ-2004"}
            aiko, grace = who("W-100236"), who("W-100003")
            r = await aiko.send(review)
            expect(r.state == "input-required" and r.data["advisory_only"], f"manager review ended {r.state}")
            pay = r.data["payroll_impact"]
            expect(pay and pay["available"] and pay["unpaid_hours"] == 80.0, f"no payroll impact from the peer: {pay}")
            expect((await grace.send(review)).state == "completed", "HR's view-only review did not complete")
            expect(ws.request("REQ-2004")["status"] == "pending", "a review changed the request")
            try:
                await grace.send({"outcome": "approved"}, task_id=r.task_id, context_id=r.context_id)
            except Exception:
                pass
            else:
                raise CheckFailed("HR continued the manager's task")
            expect(ws.request("REQ-2004")["status"] == "pending", "an attempted takeover changed the request")
            d = await aiko.send({"outcome": "approved", "note": "40 paid, 80 unpaid"}, task_id=r.task_id, context_id=r.context_id)
            art = d.artifacts.get("decision", {})
            expect(d.state == "completed" and art.get("evidence", {}).get("chain_verified"), f"decision ended {d.state}: {art}")
            expect((art["paid_hours"], art["unpaid_hours"]) == (40.0, 80.0), f"recorded {art}")
            other = await ws_payroll(payroll)
            expect(other.state == "rejected", "the payroll agent answered for another tenant")
            return f"401 without a token; manager task paused with payroll impact {pay['estimated_pay_reduction']:,.2f} from the peer; HR view-only and cannot take over; decision applied, chain verified"
        finally:
            ws.close()

    async def ws_payroll(payroll):
        return await payroll.send({"skill": "assess_unpaid_leave_impact", "tenant_id": "TEN-002", "worker_id": "W-100237",
                                   "unpaid_hours": 8, "start": "2026-10-19", "end": "2026-10-19"})

    try:
        with tempfile.TemporaryDirectory() as home:
            return asyncio.run(flow(home))
    finally:
        llm.FIXTURES, llm.before_live_call, llm.after_live_call = saved


OFFLINE: list[tuple[str, Callable[[policy.Tenant], str]]] = [
    ("agent never decides", check_agent_never_decides),
    ("CLI paths and exit codes", check_cli_paths),
    ("refusal does not jam the run", check_refusal_does_not_jam_the_run),
    ("tamper detection", check_tamper_detection),
    ("retrieval isolation", check_retrieval_isolation),
    ("citations are given inputs", check_citations_are_given_inputs),
    ("policy is data", check_policy_is_data),
    ("graded eval", check_graded_eval),
    ("RAGAS retrieval metrics", check_ragas_retrieval),
    ("web app through HTTP", check_web_app),
    ("MCP tools over stdio", check_mcp_tools),
    ("multi-agent assessment (replayed)", check_multi_agent),
    ("A2A agents over the protocol", check_a2a),
]


# ── live checks ─────────────────────────────────────────────────────────────

def check_live_agent_call(t: policy.Tenant) -> str:
    """One real call for REQ-2004, recorded into a scratch cache."""
    with tempfile.TemporaryDirectory() as d:
        scratch = Path(d) / "llm_cache.json"
        shutil.copy(llm.FIXTURES, scratch)
        saved, llm.FIXTURES = llm.FIXTURES, scratch
        try:
            state, _ = _paused(graph_mod.build(t, record_llm=True), t.requests["REQ-2004"], "e2e-live")
            entry = next(v for v in json.loads(scratch.read_text()).values() if v["label"] == "assess:REQ-2004")
        finally:
            llm.FIXTURES = saved
    rec = Recommendation.model_validate(state["recommendation"])
    given = {p["passage_id"] for p in state["passages"]}
    expect("__interrupt__" in state and state["decision"] is None, "the live run did not pause for a human")
    expect(llm.AGENT_MODEL in (entry.get("served_by") or []), f"served by {entry.get('served_by')}")
    expect(set(rec.cited_passage_ids) <= given, "the live answer cites a passage it was not given")
    return f"{entry['source']} · {entry['served_by']} · recommends {rec.action} · paused for a human"


def check_live_multi_agent(t: policy.Tenant) -> str:
    """The specialists run live for REQ-2004 into scratch caches, then replay from them."""
    from . import retrieval
    from .agents import _mcp_server

    with tempfile.TemporaryDirectory() as d:
        scratch, emb = Path(d) / "llm_cache.json", Path(d) / "embeddings.json"
        shutil.copy(llm.FIXTURES, scratch)
        shutil.copy(retrieval.EMBED_CACHE, emb)
        saved = llm.FIXTURES
        llm.FIXTURES = scratch
        try:
            with _env(HR_AGENT_EMBEDDINGS=str(emb), HR_AGENT_FIXTURES=str(scratch)):
                tenant = policy.Tenant()
                tenant._policy_index = retrieval.PolicyIndex(embedder=retrieval.Embedder(cache_path=emb))
                state, _ = _paused(graph_mod.build(tenant, record_llm=True, agents="multi"), tenant.requests["REQ-2004"], "e2e-live-multi")
                entries = [v for v in json.loads(scratch.read_text()).values() if v.get("kind") == "agent" and v["label"].endswith("REQ-2004")]
                calls = [e for e in state["evidence"] if e["data"].get("via") == "mcp"]
                expect("__interrupt__" in state and state["decision"] is None, "the live multi-agent run did not pause for a human")
                expect(len(entries) == 2 and all(llm.AGENT_MODEL in e["served_by"] for e in entries), f"served by {[e.get('served_by') for e in entries]}")
                expect(calls, "the specialists made no tool calls")
                # and what was just recorded replays without the model
                with _env(HR_AGENT_OFFLINE="1"):
                    again, _ = _paused(graph_mod.build(tenant, agents="multi"), tenant.requests["REQ-2004"], "e2e-live-multi-replay")
                expect(again["recommendation"] == state["recommendation"], "the recorded trajectory did not replay to the same recommendation")
        finally:
            llm.FIXTURES = saved
    return f"{entries[0]['source']} · {len(calls)} tool calls recorded and replayed · paused for a human"


def check_live_ragas(t: policy.Tenant) -> str:
    from . import rag_eval

    report = rag_eval.run_all(t)
    errors = [f"{c['case_id']} {n}: {e}" for c in report["cases"] for n, e in c["errors"].items()]
    expect(not errors, "; ".join(errors)[:600])
    means = {k: v for k, v in report["means"].items() if k in rag_eval.GENERATION_METRICS}
    expect(all(v is not None and 0 <= v <= 1 for v in means.values()), f"bad scores {means}")
    return f"graded by {report['grading_model']}: " + ", ".join(f"{k} {v}" for k, v in means.items())


def _run(name: str, fn, tenant) -> dict:
    start = time.monotonic()
    try:
        detail, ok = fn(tenant), True
    except CheckFailed as exc:
        detail, ok = str(exc), False
    except Exception as exc:  # a crash is a failure, reported with its type
        detail, ok = f"{type(exc).__name__}: {exc}"[:600], False
    return {"check": name, "passed": ok, "detail": detail, "seconds": round(time.monotonic() - start, 1)}


def run_all(*, require_live: bool = False) -> dict:
    tenant = policy.Tenant()
    live_possible = bool(os.environ.get("ANTHROPIC_API_KEY")) or llm.BACKEND == "claude-cli"
    live_possible = live_possible and os.environ.get("HR_AGENT_OFFLINE") != "1"

    with _env(HR_AGENT_OFFLINE="1"):
        results = [_run(name, fn, tenant) for name, fn in OFFLINE]

    if live_possible:
        results.append(_run("live agent call (scratch cache)", check_live_agent_call, tenant))
        results.append(_run("live multi-agent run (scratch cache)", check_live_multi_agent, tenant))
        if os.environ.get("ANTHROPIC_API_KEY"):
            results.append(_run("live RAGAS faithfulness + context recall", check_live_ragas, tenant))
            live = "ran"
        else:
            live = "agent call ran via Claude Code; RAGAS grading skipped (needs ANTHROPIC_API_KEY)"
    else:
        live = "skipped: no ANTHROPIC_API_KEY (or HR_AGENT_BACKEND=claude-cli)"
        if require_live:
            results.append({"check": "live checks", "passed": False, "detail": live, "seconds": 0})

    return {
        "passed": all(r["passed"] for r in results),
        "live": live,
        "backend": llm.BACKEND,
        "agent_model": llm.AGENT_MODEL,
        "results": results,
    }
