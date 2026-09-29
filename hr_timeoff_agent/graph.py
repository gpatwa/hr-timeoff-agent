"""The time-off triage graph.

    load_context → check_policy → retrieve → assess → approval_gate ⏸ → record

The graph physically cannot reach `record` without a human resuming it at
`approval_gate`, and both the gate and `record` reject any decision that is not
from a human who is the requester's direct manager.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import evidence, policy, retrieval
from .llm import structured
from .models import Decision, Finding, Passage, Recommendation

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

You may also be given handbook passages and past decisions retrieved for this \
request. They are guidance, not rule outcomes: use them to explain what the \
manager can do (an exception route, unpaid leave, cover from another team), never \
to overturn a finding. Past decisions show how similar requests went; they do not \
bind this one. Do not rely on anything that is not in the findings or passages.

Cite the rule ids and passage ids your reasoning actually rests on. Write the \
rationale for the approving manager: two or three sentences, specific, no \
restating of the whole request back to them."""


class AgentState(TypedDict):
    request: dict
    worker: Optional[dict]
    findings: list[dict]
    passages: list[dict]
    recommendation: Optional[dict]
    decision: Optional[dict]
    evidence: list[dict]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_assess_prompt(
    request: dict, worker: dict, findings: list[Finding], passages: list[Passage]
) -> str:
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
    for kind, heading in (
        ("handbook", "HANDBOOK GUIDANCE (retrieved — cite by id; guidance, not rule outcomes)"),
        ("precedent", "PAST DECISIONS (retrieved — cite by id; context, not binding)"),
    ):
        lines += ["", heading]
        for p in (p for p in passages if p.kind == kind):
            lines.append(f"  [{p.passage_id}] {p.title}")
            lines.append(f"      {p.text}")
    return "\n".join(lines)


def _index(tenant: policy.Tenant) -> retrieval.PolicyIndex:
    # Built once per tenant object; embedding the corpus is the slow part.
    if not hasattr(tenant, "_policy_index"):
        tenant._policy_index = retrieval.PolicyIndex()
    return tenant._policy_index


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

    def retrieve(state: AgentState) -> dict:
        """Tenant and audience come from the loaded tenant and the approver role,
        never from the model, and are applied inside the vector query."""
        findings = [Finding.model_validate(f) for f in state["findings"]]
        query = retrieval.build_query(state["request"], findings)
        index = _index(tenant)
        passages = index.search_handbook(query, tenant_id=tenant.tenant_id, reader="manager")
        passages += index.search_precedents(query, tenant_id=tenant.tenant_id)
        ledger = evidence.append(
            state["evidence"],
            actor="system",
            node="retrieve",
            summary=(
                f"Retrieved {', '.join(p.passage_id for p in passages)} "
                f"for {tenant.tenant_id} (reader: manager)."
            ),
            data={
                "query": query,
                "filter": {"tenant_id": tenant.tenant_id, "audience": retrieval.AUDIENCES["manager"]},
                "results": [{"id": p.passage_id, "kind": p.kind, "score": p.score} for p in passages],
            },
        )
        return {"passages": [p.model_dump() for p in passages], "evidence": ledger}

    def assess(state: AgentState) -> dict:
        findings = [Finding.model_validate(f) for f in state["findings"]]
        passages = [Passage.model_validate(p) for p in state["passages"]]
        prompt = build_assess_prompt(state["request"], state["worker"], findings, passages)
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
        """Hard stop. Execution cannot continue without a resume from an authorized human.

        An unauthorized resume does not raise: raising would mark this step failed
        and LangGraph would replay the same refused resume forever. Instead the
        gate pauses again, says why, and waits for the right approver. Each refused
        attempt is written to the evidence trail once the request is decided.
        """
        rec = state["recommendation"]
        prompt: dict[str, Any] = {
            "kind": "approval_required",
            "request_id": state["request"]["request_id"],
            "worker": state["worker"]["legal_name"],
            "window": f"{state['request']['from']} to {state['request']['to']}",
            "agent_recommendation": rec["action"],
            "agent_rationale": rec["rationale"],
            "cited_rules": rec["cited_rule_ids"],
            "cited_passages": rec.get("cited_passage_ids", []),
            "findings": [
                {"rule_id": f["rule_id"], "status": f["status"], "detail": f["detail"]}
                for f in state["findings"]
            ],
            "awaiting": "outcome (approved | declined | returned), decided_by_id, note",
            "approver_required": state["worker"]["manager_id"],
        }
        refusals: list[dict] = []
        while True:
            response: dict[str, Any] = interrupt(prompt)
            ok, reason = policy.authorize_approver(tenant, state["worker"], response.get("decided_by_id"))
            if ok:
                break
            refusals.append({"attempted_by_id": response.get("decided_by_id"), "reason": reason})
            prompt = {**prompt, "refused": f"Approval refused: {reason}"}

        ledger = state["evidence"]
        for r in refusals:
            ledger = evidence.append(
                ledger,
                actor="system",
                node="approval_gate",
                summary=f"Refused a decision attempt: {r['reason']}",
                data=r,
            )
        approver = tenant.workers[response["decided_by_id"]]
        decision = Decision(
            outcome=response["outcome"],
            decided_by_id=approver["worker_id"],
            decided_by=approver["legal_name"],
            note=response.get("note", ""),
            at=_now(),
        )
        ledger = evidence.append(
            ledger,
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
        # Re-checked here, independently of the gate, against the same policy data.
        ok, reason = policy.authorize_approver(tenant, state["worker"], decision.decided_by_id)
        if not ok:
            raise PermissionError(f"Decision not recorded: {reason}")

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
                "authorized_by_id": decision.decided_by_id,
                "chain_verified": True,
            },
        )
        return {"evidence": ledger}

    return load_context, check_policy, retrieve, assess, approval_gate, record


def build(tenant: policy.Tenant, *, record_llm: bool = False):
    load_context, check_policy, retrieve, assess, approval_gate, record = make_nodes(
        tenant, record_llm=record_llm
    )
    g = StateGraph(AgentState)
    g.add_node("load_context", load_context)
    g.add_node("check_policy", check_policy)
    g.add_node("retrieve", retrieve)
    g.add_node("assess", assess)
    g.add_node("approval_gate", approval_gate)
    g.add_node("record", record)

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "check_policy")
    g.add_edge("check_policy", "retrieve")
    g.add_edge("retrieve", "assess")
    g.add_edge("assess", "approval_gate")
    g.add_edge("approval_gate", "record")
    g.add_edge("record", END)
    return g.compile(checkpointer=InMemorySaver())


def initial_state(request: dict) -> AgentState:
    return {
        "request": request,
        "worker": None,
        "findings": [],
        "passages": [],
        "recommendation": None,
        "decision": None,
        "evidence": [],
    }
