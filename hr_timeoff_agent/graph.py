"""The time-off triage graph.

    load_context → check_policy → assess → approval_gate ⏸ → record

The graph physically cannot reach `record` without a human resuming it at
`approval_gate`, and `record` rejects any decision not attributed to a human.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import evidence, policy
from .llm import structured
from .models import Decision, Finding, Recommendation

ASSESS_SYSTEM = """You review time off requests for an HR system and produce a \
RECOMMENDATION for a human approver. You never make the decision yourself.

You are given policy findings that were computed deterministically by a rules \
engine. Treat them as ground truth: do not re-derive, dispute, or infer findings \
that are not present.

Choose exactly one action:
- approve: every finding passes, or the only issues are advisory and clearly minor.
- decline: a blocking finding failed and there is no plausible remedy.
- escalate: a blocking finding failed but an exception is plausible, or advisory \
findings need human judgement.

Cite the rule ids your reasoning actually rests on. Write the rationale for the \
approving manager: two or three sentences, specific, no restating of the whole \
request back to them."""


class AgentState(TypedDict):
    request: dict
    worker: Optional[dict]
    findings: list[dict]
    recommendation: Optional[dict]
    decision: Optional[dict]
    evidence: list[dict]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_assess_prompt(request: dict, worker: dict, findings: list[Finding]) -> str:
    """Deterministic given its inputs — no clock, no ids that vary per run.

    That stability is what lets the fixture cache key on prompt content.
    """
    lines = [
        "REQUEST",
        f"  id: {request['request_id']}",
        f"  plan: {request['plan']}",
        f"  window: {request['from']} to {request['to']} ({request['hours']:g} hours)",
        f"  submitted: {request['submitted_at']}",
        f"  worker note: {request['note']}",
        "",
        "WORKER",
        f"  position: {worker['position']}",
        f"  supervisory org: {worker['supervisory_org']}",
        f"  hire date: {worker['hire_date']}",
        f"  {request['plan']} balance: "
        f"{worker['time_off_plans'].get(request['plan'], {}).get('balance_hours', 0):g} hours",
        "",
        "POLICY FINDINGS (computed by the rules engine — treat as ground truth)",
    ]
    for f in findings:
        lines.append(f"  [{f.rule_id}] {f.rule_name} — {f.status.upper()} ({f.severity})")
        lines.append(f"      {f.detail}")
    return "\n".join(lines)


def make_nodes(tenant: policy.Tenant, *, record_llm: bool = False):
    def load_context(state: AgentState) -> dict:
        request = state["request"]
        worker = tenant.workers[request["worker_id"]]
        ledger = evidence.append(
            state["evidence"],
            actor="system",
            node="load_context",
            summary=f"Resolved worker {worker['worker_id']} for {request['request_id']}.",
            data={
                "worker_id": worker["worker_id"],
                "supervisory_org": worker["supervisory_org"],
                "policy_id": tenant.policy["policy_id"],
            },
        )
        return {"worker": worker, "evidence": ledger}

    def check_policy(state: AgentState) -> dict:
        findings = policy.evaluate(tenant, state["request"], state["worker"])
        ledger = state["evidence"]
        for f in findings:
            ledger = evidence.append(
                ledger,
                actor="rule",
                node="check_policy",
                summary=f"{f.rule_id} {f.status.upper()}: {f.detail}",
                data=f.model_dump(),
            )
        return {"findings": [f.model_dump() for f in findings], "evidence": ledger}

    def assess(state: AgentState) -> dict:
        findings = [Finding.model_validate(f) for f in state["findings"]]
        prompt = build_assess_prompt(state["request"], state["worker"], findings)
        rec = structured(
            system=ASSESS_SYSTEM,
            user=prompt,
            schema=Recommendation,
            record=record_llm,
            label=f"assess:{state['request']['request_id']}",
        )
        ledger = evidence.append(
            state["evidence"],
            actor="agent",
            node="assess",
            summary=f"Recommended {rec.action.upper()} (confidence {rec.confidence}).",
            data={**rec.model_dump(), "advisory_only": True},
        )
        return {"recommendation": rec.model_dump(), "evidence": ledger}

    def approval_gate(state: AgentState) -> dict:
        """Hard stop. Execution cannot continue without a human resume value."""
        rec = state["recommendation"]
        response: dict[str, Any] = interrupt(
            {
                "kind": "approval_required",
                "request_id": state["request"]["request_id"],
                "worker": state["worker"]["legal_name"],
                "window": f"{state['request']['from']} to {state['request']['to']}",
                "agent_recommendation": rec["action"],
                "agent_rationale": rec["rationale"],
                "cited_rules": rec["cited_rule_ids"],
                "findings": [
                    {"rule_id": f["rule_id"], "status": f["status"], "detail": f["detail"]}
                    for f in state["findings"]
                ],
                "awaiting": "outcome (approved | declined | returned), decided_by, note",
            }
        )
        decision = Decision(
            outcome=response["outcome"],
            decided_by=response["decided_by"],
            note=response.get("note", ""),
            at=_now(),
        )
        ledger = evidence.append(
            state["evidence"],
            actor="human",
            node="approval_gate",
            summary=(
                f"{decision.decided_by} {decision.outcome.upper()} "
                f"(agent had recommended {rec['action']})."
            ),
            data=decision.model_dump(),
        )
        return {"decision": decision.model_dump(), "evidence": ledger}

    def record(state: AgentState) -> dict:
        decision = Decision.model_validate(state["decision"])
        if decision.actor_type != "human":
            raise PermissionError("Only a human decision can be recorded.")
        if not decision.decided_by:
            raise PermissionError("A recorded decision must name its approver.")

        ok, reason = evidence.verify(state["evidence"])
        if not ok:
            raise RuntimeError(f"Evidence chain failed verification: {reason}")

        ledger = evidence.append(
            state["evidence"],
            actor="system",
            node="record",
            summary=f"Committed {decision.outcome} to the worker record.",
            data={
                "request_id": state["request"]["request_id"],
                "outcome": decision.outcome,
                "authorized_by": decision.decided_by,
                "chain_verified": True,
            },
        )
        return {"evidence": ledger}

    return load_context, check_policy, assess, approval_gate, record


def build(tenant: policy.Tenant, *, record_llm: bool = False):
    load_context, check_policy, assess, approval_gate, record = make_nodes(
        tenant, record_llm=record_llm
    )
    g = StateGraph(AgentState)
    g.add_node("load_context", load_context)
    g.add_node("check_policy", check_policy)
    g.add_node("assess", assess)
    g.add_node("approval_gate", approval_gate)
    g.add_node("record", record)

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "check_policy")
    g.add_edge("check_policy", "assess")
    g.add_edge("assess", "approval_gate")
    g.add_edge("approval_gate", "record")
    g.add_edge("record", END)
    return g.compile(checkpointer=InMemorySaver())


def initial_state(request: dict) -> AgentState:
    return {
        "request": request,
        "worker": None,
        "findings": [],
        "recommendation": None,
        "decision": None,
        "evidence": [],
    }
