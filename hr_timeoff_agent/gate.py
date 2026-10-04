"""The eval gate: does the agent still meet the bar, on the real model?

The offline eval replays recorded fixtures, so it proves the code still handles what was once
recorded. It cannot notice that the model, the prompts or the tools have drifted. This runs the
graded eval live, in single-agent and multi-agent mode, measures what each triage costs and how
long it takes, drives the multi-agent path through the A2A agents with the real model, and fails if
any number crosses the thresholds in evals/thresholds.json.

    python -m hr_timeoff_agent gate                 # live: needs ANTHROPIC_API_KEY (or HR_AGENT_BACKEND=claude-cli)
    python -m hr_timeoff_agent gate --replay        # offline plumbing check against the fixtures: no model, no cost

It never writes to the repo: calls are recorded into scratch copies of the caches. A live run that
cannot actually call a model refuses to start rather than pass having tested nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import evals, llm, policy, retrieval

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = ROOT / "evals" / "thresholds.json"
JUDGE_PREFIX = "judge:"
# A spread for routine checks: every category, four injections, one poisoned handbook.
QUICK_CASES = ["EV-01", "EV-03", "EV-06", "EV-10", "EV-19", "EV-22", "EV-23", "EV-26", "EV-29", "EV-32", "EV-35", "EV-40"]


class OverBudget(RuntimeError):
    pass


# ── measuring ───────────────────────────────────────────────────────────────

class Meter:
    """Sees every live model call: adds up what it cost, when the agent's calls started and
    ended, and refuses the next call once the whole run has spent its budget.

    Cases run in parallel threads, so the run's total is shared (under a lock) while each case's
    own cost, timing and tool calls are kept per thread."""

    def __init__(self, budget_usd: float):
        self.budget_usd = budget_usd
        self.spent = 0.0
        self._lock = threading.Lock()
        self._case = threading.local()
        self.reset_case()

    def reset_case(self) -> None:
        c = self._case
        c.agent_usd = c.judge_usd = 0.0
        c.first_start = c.last_end = None
        c.tool_calls = c.calls = 0

    agent_usd = property(lambda self: self._case.agent_usd)
    judge_usd = property(lambda self: self._case.judge_usd)
    calls = property(lambda self: self._case.calls)
    tool_calls = property(lambda self: self._case.tool_calls)

    def count_tool_calls(self, n: int) -> None:
        self._case.tool_calls += n

    def before(self, model: str, label: str) -> None:
        if self.spent >= self.budget_usd:
            raise OverBudget(f"the run's budget of ${self.budget_usd:.2f} is spent (${self.spent:.2f}); stopping before {label}")
        if not label.startswith(JUDGE_PREFIX) and self._case.first_start is None:
            self._case.first_start = time.monotonic()

    def after(self, model: str, label: str, usd: float, source: str) -> None:
        with self._lock:
            self.spent += usd
        c = self._case
        c.calls += 1
        if label.startswith(JUDGE_PREFIX):
            c.judge_usd += usd
        else:
            c.agent_usd += usd
            c.last_end = time.monotonic()

    @property
    def triage_seconds(self) -> float:
        c = self._case
        return round(c.last_end - c.first_start, 2) if c.first_start and c.last_end else 0.0


@dataclass
class CaseMetrics:
    mode: str
    case_id: str
    expected: str
    actual: str | None
    action_match: bool
    never_self_approved: bool
    scores: dict | None
    error: str | None
    agent_usd: float
    judge_usd: float
    triage_seconds: float
    tool_calls: int
    category: str = "original"
    attack: bool = False
    injection_resisted: bool | None = None
    no_approve_on_blocking: bool = True
    cites_failures: bool = True
    out_of_scope: list = field(default_factory=list)


def _count_tool_calls(meter: Meter):
    """Wrap run_agent so each specialist's tool calls are counted."""
    from . import agents

    real = agents.run_agent

    def counting(**kw):
        run = real(**kw)
        meter.count_tool_calls(len(run.calls))
        return run

    agents.run_agent = counting
    return lambda: setattr(agents, "run_agent", real)


def _one_case(tenant: policy.Tenant, case: dict, mode: str, live: bool, meter: Meter) -> CaseMetrics:
    meter.reset_case()
    r = evals.run_case(tenant, case, record=live)
    return CaseMetrics(
        mode=mode, case_id=r.case_id, expected=r.expected_action, actual=r.actual_action,
        action_match=r.action_match, never_self_approved=r.never_self_approved,
        scores=r.scores.model_dump() if r.scores else None, error=r.error,
        agent_usd=round(meter.agent_usd, 5), judge_usd=round(meter.judge_usd, 5),
        triage_seconds=meter.triage_seconds, tool_calls=meter.tool_calls,
        category=r.category, attack=r.attack, injection_resisted=r.injection_resisted,
        no_approve_on_blocking=r.no_approve_on_blocking, cites_failures=r.cites_failures,
        out_of_scope=list(r.out_of_scope),
    )


def run_eval(tenant: policy.Tenant, modes: list[str], repeats: int, *, live: bool, meter: Meter,
             workers: int = 1, only: list[str] | None = None) -> list[CaseMetrics]:
    """Run the suite. `workers` cases run at once. The cases that plant a hostile handbook change an
    environment variable that tool servers read, so they run alone, after the rest."""
    out: list[CaseMetrics] = []
    undo = _count_tool_calls(meter)
    saved_mode = os.environ.get("HR_AGENT_MODE")
    cases = [c for c in evals._cases() if only is None or c["case_id"] in only]
    jobs = [c for c in cases for _ in range(repeats)]
    shared = [c for c in jobs if not c.get("corpus")]
    alone = [c for c in jobs if c.get("corpus")]
    try:
        for mode in modes:
            os.environ["HR_AGENT_MODE"] = mode
            if workers > 1 and len(shared) > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    out += list(pool.map(lambda c: _one_case(tenant, c, mode, live, meter), shared))
            else:
                out += [_one_case(tenant, c, mode, live, meter) for c in shared]
            out += [_one_case(tenant, c, mode, live, meter) for c in alone]
    finally:
        undo()
        if saved_mode is None:
            os.environ.pop("HR_AGENT_MODE", None)
        else:
            os.environ["HR_AGENT_MODE"] = saved_mode
    return out


# ── judging the results ─────────────────────────────────────────────────────

def summarize(metrics: list[CaseMetrics]) -> dict:
    by_mode: dict[str, list[CaseMetrics]] = {}
    for m in metrics:
        by_mode.setdefault(m.mode, []).append(m)
    out = {}
    for mode, rows in by_mode.items():
        scored = [r for r in rows if r.scores]

        def mean(key: str) -> float:
            return round(sum(r.scores[key] for r in scored) / len(scored), 2) if scored else 0.0

        attacks = [r for r in rows if r.attack]
        n = len(rows)
        out[mode] = {
            "cases": n,
            "errors": [f"{r.case_id}: {r.error}" for r in rows if r.error],
            "action_match": sum(r.action_match for r in rows),
            "action_match_rate": round(sum(r.action_match for r in rows) / n, 3) if n else 0.0,
            "never_self_approved": sum(r.never_self_approved for r in rows),
            "never_self_approved_rate": round(sum(r.never_self_approved for r in rows) / n, 3) if n else 0.0,
            "no_approve_on_blocking_rate": round(sum(r.no_approve_on_blocking for r in rows) / n, 3) if n else 0.0,
            "cites_failures_rate": round(sum(r.cites_failures for r in rows) / n, 3) if n else 0.0,
            "attacks": len(attacks),
            "injection_resisted": sum(bool(r.injection_resisted) for r in attacks),
            "injection_resisted_rate": round(sum(bool(r.injection_resisted) for r in attacks) / len(attacks), 3) if attacks else 1.0,
            "fell_to": [f"{r.case_id}" for r in attacks if not r.injection_resisted],
            "rationale_grounded": mean("rationale_grounded"),
            "citations_correct": mean("citations_correct"),
            "tone_appropriate": mean("tone_appropriate"),
            "mean_agent_usd_per_case": round(statistics.fmean(r.agent_usd for r in rows), 5) if rows else 0.0,
            "mean_judge_usd_per_case": round(statistics.fmean(r.judge_usd for r in rows), 5) if rows else 0.0,
            "max_triage_seconds": max((r.triage_seconds for r in rows), default=0.0),
            "mean_tool_calls": round(statistics.fmean(r.tool_calls for r in rows), 2) if rows else 0.0,
        }
    return out


def evaluate(summary: dict, a2a: dict | None, thresholds: dict) -> list[str]:
    """Every way the numbers fail the thresholds, as sentences."""
    bad: list[str] = []
    for mode, limits in thresholds["modes"].items():
        got = summary.get(mode)
        if got is None:
            continue
        for e in got["errors"]:
            bad.append(f"{mode}: a case errored ({e})")
        floors = {"min_action_match_rate": "action_match_rate", "min_never_self_approved_rate": "never_self_approved_rate",
                  "min_no_approve_on_blocking_rate": "no_approve_on_blocking_rate", "min_cites_failures_rate": "cites_failures_rate",
                  "min_injection_resisted_rate": "injection_resisted_rate",
                  "min_rationale_grounded": "rationale_grounded", "min_citations_correct": "citations_correct",
                  "min_tone_appropriate": "tone_appropriate"}
        for key, metric in floors.items():
            if key in limits and got[metric] < limits[key]:
                extra = f" (fell to: {', '.join(got['fell_to'])})" if metric == "injection_resisted_rate" and got["fell_to"] else ""
                bad.append(f"{mode}: {metric} {got[metric]} is below the floor {limits[key]}{extra}")
        if got["mean_agent_usd_per_case"] > limits.get("max_mean_agent_usd_per_case", float("inf")):
            bad.append(f"{mode}: ${got['mean_agent_usd_per_case']:.4f} per triage is over the ceiling ${limits['max_mean_agent_usd_per_case']}")
        if got["max_triage_seconds"] > limits.get("max_triage_seconds", float("inf")):
            bad.append(f"{mode}: the slowest triage took {got['max_triage_seconds']} s, over the ceiling {limits['max_triage_seconds']} s")
    if a2a is not None and "a2a_multi" in thresholds:
        lim = thresholds["a2a_multi"]
        if not a2a["passed"]:
            bad.append(f"a2a multi-agent: {a2a['detail']}")
        else:
            if a2a["usd"] > lim.get("max_usd", float("inf")):
                bad.append(f"a2a multi-agent: ${a2a['usd']:.4f} is over the ceiling ${lim['max_usd']}")
            if a2a["seconds"] > lim.get("max_seconds", float("inf")):
                bad.append(f"a2a multi-agent: {a2a['seconds']} s is over the ceiling {lim['max_seconds']} s")
            if a2a["tool_calls"] < lim.get("min_tool_calls", 0):
                bad.append(f"a2a multi-agent: only {a2a['tool_calls']} tool calls; the specialists should have used their tools")
    return bad


# ── the multi-agent path through the A2A agents, with the real model ─────────

def run_a2a_multi(scratch: Path) -> dict:
    """An employee's agent files a request over A2A (so triage runs live, in multi-agent mode), the
    manager's agent reviews it (the task pauses with the recommendation and the payroll agent's
    price), then decides on the same task. Checks what only a live run can show: the specialists
    used their tools, the evidence names them, and the human gate still holds."""
    import asyncio

    import httpx

    from .a2a_client import A2AAgent
    from .a2a_common import BearerTokens
    from .a2a_payroll import create_payroll_app
    from .a2a_server import create_timeoff_app
    from .web.workspace import Workspace

    start = time.monotonic()
    saved_mode = os.environ.get("HR_AGENT_MODE")
    os.environ["HR_AGENT_MODE"] = "multi"
    saved_llm = (llm.FIXTURES, llm.before_live_call, llm.after_live_call)

    async def flow(ws: Workspace) -> dict:
        tokens = BearerTokens.derive("gate", [p.worker_id for p in ws.personas()] + ["timeoff-agent"])
        asgi = lambda app, base: httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base, timeout=600)
        payroll = A2AAgent("http://payroll", tokens.for_principal("timeoff-agent"), httpx_client=asgi(create_payroll_app(tokens, base_url="http://payroll"), "http://payroll"))
        app = create_timeoff_app(ws, tokens, payroll=payroll, base_url="http://timeoff")
        who = lambda w: A2AAgent("http://timeoff", tokens.for_principal(w), httpx_client=asgi(app, "http://timeoff"))
        priya, dana = who("W-100234"), who("W-100001")
        try:
            filed = await priya.send({"skill": "file_time_off_request", "plan": "PTO", "from": "2026-12-07", "to": "2026-12-25",
                                      "note": "Family trip abroad, booked months ago."})
            if filed.state != "completed" or filed.artifacts["request"]["status"] != "pending":
                return {"passed": False, "detail": f"filing ended {filed.state}/{filed.artifacts.get('request', {}).get('status')}: {filed.text}"}
            rid = filed.artifacts["request"]["request_id"]
            review = await dana.send({"skill": "review_time_off_request", "request_id": rid})
            if review.state != "input-required":
                return {"passed": False, "detail": f"the manager's task ended {review.state}, not input-required: {review.text}"}
            rec = review.data["recommendation"]
            if rec["action"] not in ("approve", "decline", "escalate") or not rec["rationale"]:
                return {"passed": False, "detail": f"no usable recommendation: {rec}"}
            pay = review.data.get("payroll_impact") or {}
            if not pay.get("available"):
                return {"passed": False, "detail": f"the payroll agent was not consulted over A2A: {pay}"}
            detail = ws.detail(ws.persona("W-100001"), rid)
            calls = [e for e in detail["evidence"] if e["data"].get("via") == "mcp"]
            agents_used = {e["data"]["agent"] for e in calls}
            if agents_used != {"policy_specialist", "coverage_specialist"}:
                return {"passed": False, "detail": f"tool calls came from {sorted(agents_used)}, not both specialists"}
            if not detail["chain_ok"]:
                return {"passed": False, "detail": "the evidence chain does not verify after triage"}
            given = {p["passage_id"] for p in detail["passages"]}
            if not set(rec["cited_passage_ids"]) <= given:
                return {"passed": False, "detail": "the recommendation cites a passage no tool returned"}
            done = await dana.send({"outcome": "approved", "note": "gate"}, task_id=review.task_id, context_id=review.context_id)
            if done.state != "completed" or not done.artifacts["decision"]["evidence"]["chain_verified"]:
                return {"passed": False, "detail": f"the decision ended {done.state}"}
            return {"passed": True, "detail": f"{rid}: {len(calls)} tool calls by both specialists, payroll consulted, recommendation {rec['action']}, decided by the manager",
                    "tool_calls": len(calls), "recommendation": rec["action"]}
        finally:
            await priya.close()
            await dana.close()

    try:
        ws = Workspace(scratch / "a2a-home", daily_cap_usd=float(os.environ.get("HR_GATE_A2A_CAP_USD", "2.0")), triages_per_hour=50)
        try:
            result = asyncio.run(flow(ws))
            result["usd"] = round(ws.spent_today(), 5)
        finally:
            ws.close()
    except Exception as exc:  # noqa: BLE001
        result = {"passed": False, "detail": f"{type(exc).__name__}: {exc}"[:400], "usd": 0.0}
    finally:
        llm.FIXTURES, llm.before_live_call, llm.after_live_call = saved_llm
        if saved_mode is None:
            os.environ.pop("HR_AGENT_MODE", None)
        else:
            os.environ["HR_AGENT_MODE"] = saved_mode
    result.setdefault("tool_calls", 0)
    result["seconds"] = round(time.monotonic() - start, 1)
    return result


# ── the run ─────────────────────────────────────────────────────────────────

def can_call_a_model() -> bool:
    return not llm.is_offline()


def render_markdown(report: dict) -> str:
    live = report["live"]
    lines = [f"## Eval gate: {'PASS' if report['passed'] else 'FAIL'}" + ("" if live else " (replay: no model was called)"), ""]
    lines.append(f"Backend `{report['backend']}` · agent `{report['agent_model']}` · judge `{report['judge_model']}` · "
                 f"spent ${report['spent_usd']:.3f} of ${report['budget_usd']:.2f} · {report['seconds']} s")
    lines += ["", "| mode | action ok | never self-approved | no approve on blocking | cites failures | attacks held | grounded | cites | tone | $/triage | slowest triage | tool calls |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, s in report["summary"].items():
        lines.append(f"| {mode} | {s['action_match']}/{s['cases']} | {s['never_self_approved']}/{s['cases']} | {s['no_approve_on_blocking_rate']:.0%} | "
                     f"{s['cites_failures_rate']:.0%} | {s['injection_resisted']}/{s['attacks']} | {s['rationale_grounded']} | "
                     f"{s['citations_correct']} | {s['tone_appropriate']} | ${s['mean_agent_usd_per_case']:.4f} | {s['max_triage_seconds']} s | {s['mean_tool_calls']} |")
    if report["a2a"] is not None:
        a = report["a2a"]
        lines += ["", f"**Multi-agent over A2A:** {'pass' if a['passed'] else 'FAIL'} · {a['detail']} · ${a.get('usd', 0):.4f} · {a.get('seconds')} s"]
    elif live:
        lines += ["", "**Multi-agent over A2A:** not run (`--no-a2a`)"]
    lines.append("")
    lines += [f"- violation: {v}" for v in report["violations"]] or ["No threshold was crossed."]
    return "\n".join(lines) + "\n"


def run_gate(*, modes: list[str], repeats: int, thresholds_path: Path = THRESHOLDS, replay: bool = False,
             a2a: bool = True, budget_usd: float | None = None, workers: int = 1, quick: bool = False) -> dict:
    thresholds = json.loads(Path(thresholds_path).read_text())
    live = not replay
    if live and not can_call_a_model():
        raise RuntimeError("The live gate needs a model: set ANTHROPIC_API_KEY (or HR_AGENT_BACKEND=claude-cli). "
                           "Use --replay to check the plumbing against the recorded fixtures.")
    budget = budget_usd if budget_usd is not None else float(thresholds.get("total_budget_usd", 5.0))
    meter = Meter(budget_usd=budget)
    start = time.monotonic()
    saved = (llm.FIXTURES, llm.before_live_call, llm.after_live_call)

    with tempfile.TemporaryDirectory() as d:
        scratch = Path(d)
        # Live calls are recorded into scratch copies, so a gate run never edits the repo's fixtures.
        cache, emb = scratch / "llm_cache.json", scratch / "embeddings.json"
        shutil.copy(llm.FIXTURES, cache)
        shutil.copy(retrieval.EMBED_CACHE, emb)
        llm.FIXTURES = cache
        llm.before_live_call, llm.after_live_call = meter.before, meter.after
        # The MCP server that Claude Code starts is a child process: it learns where to record
        # new query embeddings from the environment, and must not write to the repo's file.
        saved_env = {k: os.environ.get(k) for k in ("HR_AGENT_EMBEDDINGS", "HR_AGENT_FIXTURES")}
        os.environ["HR_AGENT_EMBEDDINGS"], os.environ["HR_AGENT_FIXTURES"] = str(emb), str(cache)
        report: dict = {}
        try:
            tenant = policy.Tenant()
            tenant._policy_index = retrieval.PolicyIndex(embedder=retrieval.Embedder(cache_path=emb))
            try:
                metrics = run_eval(tenant, modes, repeats, live=live, meter=meter, workers=workers,
                                   only=QUICK_CASES if quick else None)
                halted = None
            except OverBudget as exc:
                metrics, halted = [], str(exc)
            summary = summarize(metrics)
            a2a_result = None
            if live and a2a and halted is None:
                saved_hooks = (llm.before_live_call, llm.after_live_call)
                a2a_result = run_a2a_multi(scratch)
                llm.before_live_call, llm.after_live_call = saved_hooks
                meter.spent += a2a_result.get("usd", 0.0)
            violations = evaluate(summary, a2a_result, thresholds)
            if halted:
                violations.insert(0, halted)
            report = {
                "passed": not violations,
                "live": live,
                "quick": quick,
                "workers": workers,
                "backend": llm.BACKEND if live else "replay",
                "agent_model": llm.AGENT_MODEL,
                "judge_model": llm.JUDGE_MODEL,
                "spent_usd": round(meter.spent, 4),
                "budget_usd": budget,
                "seconds": round(time.monotonic() - start, 1),
                "summary": summary,
                "a2a": a2a_result,
                "violations": violations,
                "cases": [asdict(m) for m in metrics],
                "thresholds": thresholds,
            }
        finally:
            llm.FIXTURES, llm.before_live_call, llm.after_live_call = saved
            for k, v in saved_env.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return report
