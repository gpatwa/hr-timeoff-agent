"""The multi-agent assessment: two tool-calling specialists and a coordinator.

    check_policy → policy_specialist → coverage_specialist → assess (coordinator)

The specialists investigate with read-only MCP tools and write a typed report.
The coordinator makes no tool calls: it reads the findings, the passages the
policy specialist retrieved and both reports, and writes the same `Recommendation`
the single-agent path writes. Everything downstream is unchanged: the rules
engine still decides what passes, the approval gate still waits for the
requester's direct manager, and none of these agents can decide anything.

Each tool call a specialist makes lands in the evidence ledger, with its
arguments and a digest of what it returned.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import evidence, policy
from .agentloop import AgentRun, run_agent
from .graph import ASSESS_SYSTEM, _index, build_assess_prompt
from .llm import structured
from .mcp_server import HRToolServer
from .models import Finding, Passage, Recommendation

POLICY_TOOLS = ["search_handbook", "search_precedents", "get_worker"]
COVERAGE_TOOLS = ["team_availability", "get_balance", "get_worker"]


class SpecialistReport(BaseModel):
    summary: str = Field(description="Two or three sentences for the coordinator.")
    options: list[str] = Field(description="Concrete things the approving manager could do, each grounded in a tool result.")
    cited_rule_ids: list[str] = Field(description="Rule ids from the findings that the report rests on.")
    cited_passage_ids: list[str] = Field(description="Passage ids returned by your search tools that the report rests on.")
    gaps: list[str] = Field(description="Anything you could not establish with your tools.")


_SHARED = """You are a specialist assistant on an HR time-off review. You investigate \
with the read-only tools you are given and write a short report for a coordinator. \
You never decide anything: a human manager approves or declines.

The policy findings in the request were computed by a deterministic rules engine. \
Treat them as ground truth; do not re-derive or dispute them. Only state facts that \
the findings, the request or a tool result you actually received support, and cite \
rule and passage ids exactly as given. Make as few tool calls as the job needs (at \
most six). If a tool returns an error, read it and either correct the call or note \
the gap."""

POLICY_SYSTEM = _SHARED + """

YOUR JOB: find what the handbook and past decisions say the manager can do about \
the situation, using search_handbook and search_precedents. Search on the specific \
issue (for example a balance shortfall, a blackout period, thin coverage), not on \
the dates. If every rule passes, one search for routine guidance is enough. List \
the manager's options, each tied to a passage id you retrieved."""

COVERAGE_SYSTEM = _SHARED + """

YOUR JOB: establish how the request affects the team, using team_availability, and \
whether a nearby alternative window would hold coverage when it does not. Check the \
requested window first. Only when coverage is below the floor, try one or two \
alternative windows of the same length (for example one week earlier or later) and \
report the best one. Use get_balance only if the findings leave the balance unclear. \
If coverage is fine, say so and stop."""

COORDINATOR_ADDENDUM = """

Two specialist assistants have investigated with read-only tools; their reports are \
below the findings. They advise you: the findings remain ground truth, and a \
specialist's claim that is not backed by a finding or a listed passage is not \
something to rely on. Cite passage ids only from the lists given."""


def _case(state: dict) -> str:
    findings = [Finding.model_validate(f) for f in state["findings"]]
    full = build_assess_prompt(state["request"], state["worker"], findings, [])
    case = full.split("\n\nHANDBOOK GUIDANCE")[0]
    # The tools take a worker id, so the specialists need to be told which one.
    return case.replace("REQUEST\n", f"REQUEST\n  worker id: {state['worker']['worker_id']}\n", 1)


def _describe(call) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
    if call.error is not None:
        return f"{call.tool}({args}) failed: {call.error}"
    if call.tool.startswith("search_"):
        ids = ", ".join(p["passage_id"] for p in call.result["passages"]) or "nothing"
        return f"{call.tool}({args}) returned {ids}"
    return f"{call.tool}({args}) ok"


def coordinator_prompt(state: dict) -> str:
    """Exactly what the coordinator is given; the eval judge is shown the same text."""
    findings = [Finding.model_validate(f) for f in state["findings"]]
    passages = [Passage.model_validate(p) for p in state["passages"]]
    lines = [build_assess_prompt(state["request"], state["worker"], findings, passages), "", "SPECIALIST REPORTS (advisory)"]
    for label, key in (("Policy specialist", "policy"), ("Coverage specialist", "coverage")):
        r = state["agent_reports"][key]
        lines += [f"  {label}: {r['summary']}"]
        lines += [f"    option: {o}" for o in r["options"]]
        lines += [f"    gap: {g}" for g in r["gaps"]]
    return "\n".join(lines)


def _mcp_server(tenant: policy.Tenant) -> HRToolServer:
    if not hasattr(tenant, "_mcp_server"):
        tenant._mcp_server = HRToolServer(tenant, index=_index(tenant), reader="manager")
    return tenant._mcp_server


def make_multi_nodes(tenant: policy.Tenant, *, record_llm: bool = False):
    server = _mcp_server(tenant)

    def investigate(state: dict, name: str, system: str, tools: list[str]) -> tuple[AgentRun, list[dict]]:
        run = run_agent(
            system=system,
            user=_case(state),
            schema=SpecialistReport,
            tools=tools,
            server=server,
            record=record_llm,
            label=f"{name}:{state['request']['request_id']}",
        )
        ledger = state["evidence"]
        for c in run.calls:
            ledger = evidence.append(
                ledger,
                actor="agent",
                node=name,
                summary=_describe(c),
                data={
                    "agent": name, "via": "mcp", "tool": c.tool, "arguments": c.arguments,
                    "result_sha256": c.result_sha256, "error": c.error,
                    "tenant_id": tenant.tenant_id, "reader": server.reader,
                },
            )
        ledger = evidence.append(
            ledger,
            actor="agent",
            node=name,
            summary=f"{name} reported after {len(run.calls)} tool call(s): {run.output.summary}",
            data={"agent": name, "advisory_only": True, **run.output.model_dump()},
        )
        return run, ledger

    def policy_specialist(state: dict) -> dict:
        run, ledger = investigate(state, "policy_specialist", POLICY_SYSTEM, POLICY_TOOLS)
        seen: dict[str, dict] = {}
        for c in run.calls:
            if c.error is None and c.tool.startswith("search_"):
                for p in c.result["passages"]:
                    seen.setdefault(p["passage_id"], p)
        reports = {**(state.get("agent_reports") or {}), "policy": run.output.model_dump()}
        return {"passages": list(seen.values()), "agent_reports": reports, "evidence": ledger}

    def coverage_specialist(state: dict) -> dict:
        run, ledger = investigate(state, "coverage_specialist", COVERAGE_SYSTEM, COVERAGE_TOOLS)
        reports = {**(state.get("agent_reports") or {}), "coverage": run.output.model_dump()}
        return {"agent_reports": reports, "evidence": ledger}

    def assess(state: dict) -> dict:
        rec = structured(
            system=ASSESS_SYSTEM + COORDINATOR_ADDENDUM,
            user=coordinator_prompt(state),
            schema=Recommendation,
            record=record_llm,
            label=f"coordinator:{state['request']['request_id']}",
        )
        ledger = evidence.append(
            state["evidence"],
            actor="agent",
            node="assess",
            summary=f"Coordinator recommended {rec.action.upper()} (confidence {rec.confidence}).",
            data={**rec.model_dump(), "advisory_only": True, "agent": "coordinator"},
        )
        return {"recommendation": rec.model_dump(), "evidence": ledger}

    return policy_specialist, coverage_specialist, assess
