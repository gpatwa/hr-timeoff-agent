"""The time-off agent as an A2A agent: a second front door onto the same product.

The web app and this agent are two clients of the same `Workspace`: the same
requests, the same graph, the same authorization, the same evidence trail. A
request filed here appears in the web app, and a decision made here is checked by
the same approval gate.

Two skills, one per side of the conversation:

  file_time_off_request     for an employee's agent. Files and triages a request.
                            The task completes with the request id and who must
                            approve it.
  review_time_off_request   for the approving manager's agent. The task pauses in
                            `input-required` with the recommendation (advisory), the
                            findings and the payroll impact, and resumes when the
                            manager's decision arrives as a follow-up message. A
                            wrong approver gets `rejected`; the request stays pending.

The mapping that makes this fit A2A rather than bend it: `input-required` means
"the agent needs a decision from the caller", and that caller is the approver.
The employee's task has nothing left to wait for, so it completes.

When an approval would leave unpaid hours, the agent asks a separate payroll agent
over A2A what that costs (see a2a_payroll.py) so the manager sees it before deciding.
"""

from __future__ import annotations

import asyncio
import logging

from a2a.helpers import new_data_part, new_text_part
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities, AgentCard, AgentInterface, AgentProvider, AgentSkill, HTTPAuthSecurityScheme,
    SecurityRequirement, SecurityScheme, StringList, TaskState,
)
from a2a.helpers import get_data_parts
from fastapi import FastAPI

from .a2a_client import A2AAgent, TaskResult
from .a2a_common import BearerContextBuilder, BearerTokens, begin, request_data, require_bearer, text_and_data
from .web.workspace import BudgetExceeded, Forbidden, Invalid, Refused, Workspace

log = logging.getLogger(__name__)

RPC_PATH = "/a2a/jsonrpc"
FILE, REVIEW = "file_time_off_request", "review_time_off_request"


class TimeOffExecutor(AgentExecutor):
    def __init__(self, workspace: Workspace, payroll: A2AAgent | None = None):
        self.ws = workspace
        self.payroll = payroll

    # ── entry point ──────────────────────────────────────────────────────

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = await begin(context, event_queue)
        persona = self.ws.persona(context.call_context.user.user_name)
        if persona is None:
            await self._reject(updater, "The bearer token does not belong to a worker in this tenant.")
            return
        data = request_data(context)
        try:
            if context.current_task and context.current_task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED:
                await self._decide(updater, persona, self._request_id_of(context.current_task), data)
            elif data.get("skill") == FILE:
                await self._file(updater, persona, data)
            elif data.get("skill") == REVIEW:
                await self._review(updater, persona, data)
            else:
                await self._reject(updater, f"Send structured data with skill {FILE!r} or {REVIEW!r}.")
        except (Forbidden, Invalid, BudgetExceeded) as exc:
            await self._reject(updater, str(exc))
        except KeyError as exc:
            await self._reject(updater, f"No such request: {exc}.")
        except Exception as exc:
            log.exception("time-off agent failed")
            await updater.failed(updater.new_agent_message([new_text_part(f"{type(exc).__name__}: {exc}"[:300])]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()

    async def _reject(self, updater: TaskUpdater, reason: str) -> None:
        await updater.reject(updater.new_agent_message([new_text_part(reason)]))

    @staticmethod
    def _request_id_of(task) -> str:
        for part in get_data_parts(task.status.message.parts):
            if isinstance(part, dict) and part.get("request_id"):
                return part["request_id"]
        raise KeyError("the paused task does not name a request")

    # ── file ─────────────────────────────────────────────────────────────

    async def _file(self, updater: TaskUpdater, persona, data: dict) -> None:
        rid = await asyncio.to_thread(
            self.ws.submit, persona,
            start=str(data.get("from", "")), end=str(data.get("to", "")),
            hours=str(data.get("hours") or ""), note=str(data.get("note") or ""), plan=str(data.get("plan") or "PTO"),
        )
        detail = await asyncio.to_thread(self.ws.detail, persona, rid)
        approver = detail["manager"] or {}
        result = {
            "request_id": rid,
            "status": detail["request"]["status"],
            "approver": {"worker_id": approver.get("worker_id"), "name": approver.get("legal_name")},
            "findings": [{"rule_id": f["rule_id"], "status": f["status"], "detail": f["detail"]} for f in detail["findings"]],
        }
        await updater.add_artifact([new_data_part(result)], name="request", last_chunk=True)
        waiting = "waiting for triage to be retried" if result["status"] == "needs_triage" else f"pending approval by {approver.get('legal_name')}"
        await updater.complete(updater.new_agent_message(text_and_data(f"{rid} filed; {waiting}.", result)))

    # ── review: pause for the approver ───────────────────────────────────

    async def _payroll_impact(self, request: dict, unpaid: float | None = None) -> dict | None:
        """What the unpaid part of an approval would cost, from the payroll agent.
        Before a decision the unpaid part is whatever exceeds the balance; after one it is what was recorded."""
        if unpaid is None:
            balance = self.ws.worker(request["worker_id"])["time_off_plans"].get(request["plan"], {}).get("balance_hours", 0.0)
            unpaid = max(0.0, float(request["hours"]) - balance)
        if unpaid <= 0:
            return None
        if self.payroll is None:
            return {"available": False, "reason": "no payroll agent configured"}
        try:
            r: TaskResult = await self.payroll.send({
                "skill": "assess_unpaid_leave_impact", "tenant_id": self.ws.tenant().tenant_id,
                "worker_id": request["worker_id"], "unpaid_hours": unpaid, "start": request["from"], "end": request["to"],
            })
        except Exception as exc:  # advisory: a down peer must not block the review
            log.warning("payroll agent unavailable: %s", exc)
            return {"available": False, "reason": f"payroll agent unreachable ({type(exc).__name__})"}
        if r.state != "completed":
            return {"available": False, "reason": f"payroll agent {r.state}: {r.text}"}
        return {"available": True, "source": "payroll agent over A2A", **r.artifacts["payroll_impact"]}

    async def _review(self, updater: TaskUpdater, persona, data: dict) -> None:
        rid = str(data.get("request_id", ""))
        detail = await asyncio.to_thread(self.ws.detail, persona, rid)   # raises Forbidden if they may not see it
        request = detail["request"]
        rec = detail["recommendation"] or {}
        approver = detail["manager"] or {}
        summary = {
            "request_id": rid,
            "status": request["status"],
            "requester": {"worker_id": request["worker_id"], "name": detail["requester"]["legal_name"]},
            "window": {k: request[k] for k in ("plan", "from", "to", "hours")},
            "findings": [{"rule_id": f["rule_id"], "status": f["status"], "detail": f["detail"]} for f in detail["findings"]],
            "recommendation": {k: rec.get(k) for k in ("action", "confidence", "rationale", "cited_rule_ids", "cited_passage_ids")},
            "advisory_only": True,
            "approver_required": {"worker_id": approver.get("worker_id"), "name": approver.get("legal_name")},
            "can_decide": detail["can_decide"],
            "payroll_impact": await self._payroll_impact(request) if request["status"] == "pending" else None,
        }
        who = f"{summary['requester']['name']}, {request['from']} to {request['to']} ({request['hours']:g}h {request['plan']})"
        await updater.add_artifact([new_data_part(summary)], name="review", last_chunk=True)
        if request["status"] != "pending":
            await updater.complete(updater.new_agent_message(text_and_data(f"{rid} ({who}) is already {request['status']}.", summary)))
        elif not detail["can_decide"]:
            await updater.complete(updater.new_agent_message(text_and_data(
                f"{rid} ({who}): recommendation {rec.get('action')}. View only: {approver.get('legal_name')} decides.", summary)))
        else:
            summary["awaiting"] = "outcome: approved | declined | returned, and an optional note"
            await updater.requires_input(updater.new_agent_message(text_and_data(
                f"{rid} ({who}): the agent recommends {rec.get('action')} (advisory). Awaiting your decision.", summary)))

    # ── decide: the follow-up message ────────────────────────────────────

    async def _decide(self, updater: TaskUpdater, persona, rid: str, data: dict) -> None:
        outcome = str(data.get("outcome", ""))
        try:
            await asyncio.to_thread(self.ws.decide, persona, rid, outcome, str(data.get("note") or ""))
        except Refused as exc:
            # The gate pauses again rather than failing, and so does the task.
            await updater.requires_input(updater.new_agent_message(text_and_data(
                f"Not recorded. {exc} The request is still pending.", {"request_id": rid, "refused": str(exc)})))
            return
        except Invalid as exc:
            await updater.requires_input(updater.new_agent_message(text_and_data(
                f"{exc} Send outcome approved, declined or returned.", {"request_id": rid, "error": str(exc)})))
            return
        detail = await asyncio.to_thread(self.ws.detail, persona, rid)
        request, decision, absence = detail["request"], detail["decision"], detail["absence"]
        result = {
            "request_id": rid,
            "outcome": decision["outcome"],
            "decided_by": {"worker_id": decision["decided_by_id"], "name": decision["decided_by"]},
            "agent_recommended": (detail["recommendation"] or {}).get("action"),
            "overrides_agent": detail["overrides"],
            "paid_hours": absence["paid_hours"] if absence else None,
            "unpaid_hours": absence["unpaid_hours"] if absence else None,
            "evidence": {"entries": len(detail["evidence"]), "chain_verified": detail["chain_ok"]},
        }
        if absence and absence["unpaid_hours"]:
            result["payroll_impact"] = await self._payroll_impact(request, unpaid=absence["unpaid_hours"])
        await updater.add_artifact([new_data_part(result)], name="decision", last_chunk=True)
        await updater.complete(updater.new_agent_message(text_and_data(
            f"{rid} {decision['outcome']} by {decision['decided_by']}; evidence chain "
            f"{'verified' if detail['chain_ok'] else 'FAILED verification'}.", result)))


def agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="Time-off triage agent",
        description=(
            "Files and triages time-off requests, and prepares them for the approving manager. "
            "The agent recommends; only the requester's direct manager decides, and the decision "
            "is checked by an approval gate and recorded in a hash-chained evidence trail."
        ),
        provider=AgentProvider(organization="hr-timeoff-agent (synthetic data)", url=base_url),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["application/json"],
        default_output_modes=["application/json", "text/plain"],
        skills=[
            AgentSkill(
                id=FILE, name="File a time-off request",
                description="For an employee. Data: plan (default PTO), from, to (YYYY-MM-DD), hours (optional), note. Completes with the request id and the required approver.",
                tags=["time-off", "employee"],
                examples=['{"skill": "file_time_off_request", "plan": "PTO", "from": "2026-11-30", "to": "2026-12-02", "note": "Family trip"}'],
                input_modes=["application/json"], output_modes=["application/json"],
            ),
            AgentSkill(
                id=REVIEW, name="Review a time-off request as the approver",
                description=("For a manager or HR. Data: request_id. For the requester's direct manager the task pauses in input-required with the "
                             "recommendation, findings and payroll impact; reply on the same task with data {outcome: approved|declined|returned, note}. "
                             "Anyone else who may see the request gets a completed, view-only review."),
                tags=["time-off", "approver", "human-in-the-loop"],
                examples=['{"skill": "review_time_off_request", "request_id": "REQ-2004"}'],
                input_modes=["application/json"], output_modes=["application/json"],
            ),
        ],
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", protocol_version="1.0", url=f"{base_url}{RPC_PATH}")],
        security_schemes={"bearer": SecurityScheme(http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="Bearer", description="Per-worker token"))},
        security_requirements=[SecurityRequirement(schemes={"bearer": StringList()})],
    )


def create_timeoff_app(workspace: Workspace, tokens: BearerTokens, *, payroll: A2AAgent | None = None, base_url: str = "http://127.0.0.1:8100") -> FastAPI:
    card = agent_card(base_url)
    handler = DefaultRequestHandler(agent_executor=TimeOffExecutor(workspace, payroll), task_store=InMemoryTaskStore(), agent_card=card)
    app = FastAPI(title="Time-off triage agent")
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card=card),
        jsonrpc_routes=create_jsonrpc_routes(request_handler=handler, rpc_url=RPC_PATH, context_builder=BearerContextBuilder(tokens)),
    )
    require_bearer(app, tokens, RPC_PATH)
    return app
