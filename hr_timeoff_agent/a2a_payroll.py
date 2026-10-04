"""A separate A2A agent: payroll impact of unpaid leave.

It is deliberately deterministic. Money arithmetic should not be a model's
guess, and an A2A agent is defined by its interface, not by what runs behind it.
The time-off agent calls it over A2A when an approval would leave unpaid hours,
so the approving manager sees the pay effect before deciding.

It is its own service with its own data (data/payroll.json), its own token and
its own tenant check: a request for another tenant is refused, not answered.

    python -m hr_timeoff_agent a2a          # starts it alongside the time-off agent
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.types import (
    AgentCapabilities, AgentCard, AgentInterface, AgentProvider, AgentSkill, HTTPAuthSecurityScheme,
    SecurityRequirement, SecurityScheme, StringList,
)
from a2a.helpers import new_data_part, new_text_part
from fastapi import FastAPI

from . import telemetry
from .a2a_common import BearerContextBuilder, BearerTokens, begin, make_task_store, request_data, require_bearer, text_and_data

DATA = Path(__file__).resolve().parent.parent / "data" / "payroll.json"
RPC_PATH = "/a2a/jsonrpc"
SKILL = "assess_unpaid_leave_impact"


class PayrollError(ValueError):
    """The request cannot be answered; the message says why."""


def _workdays(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1) if (start + timedelta(days=i)).weekday() < 5]


def assess(data: dict, args: dict) -> dict:
    """Pure function: payroll data and a request in, the pay impact out."""
    if args.get("tenant_id") != data["tenant_id"]:
        raise PayrollError(f"This agent serves tenant {data['tenant_id']}, not {args.get('tenant_id')!r}.")
    comp = data["workers"].get(args.get("worker_id", ""))
    if comp is None:
        raise PayrollError(f"No compensation record for {args.get('worker_id')!r}.")
    try:
        unpaid = float(args["unpaid_hours"])
        start, end = date.fromisoformat(args["start"]), date.fromisoformat(args["end"])
    except (KeyError, ValueError):
        raise PayrollError("unpaid_hours (number), start and end (YYYY-MM-DD) are required.")
    if unpaid <= 0 or end < start:
        raise PayrollError("unpaid_hours must be positive and end must not precede start.")

    hourly = comp["annual_salary"] / data["hours_per_year"]
    # Spread the unpaid hours over the window's working days (8h each) and assign
    # each day to its biweekly pay period.
    anchor, period = date.fromisoformat(data["period_anchor"]), timedelta(days=14)
    per_period: dict[str, float] = {}
    remaining = unpaid
    for day in _workdays(start, end):
        if remaining <= 0:
            break
        hours = min(8.0, remaining)
        remaining -= hours
        first = anchor + ((day - anchor) // period) * period
        key = f"{first.isoformat()} to {(first + period - timedelta(days=1)).isoformat()}"
        per_period[key] = round(per_period.get(key, 0.0) + hours, 2)
    rules = data["rules"]
    return {
        "tenant_id": data["tenant_id"],
        "worker_id": args["worker_id"],
        "unpaid_hours": unpaid,
        "hourly_rate": round(hourly, 2),
        "estimated_pay_reduction": round(unpaid * hourly, 2),
        "currency": comp["currency"],
        "pay_periods": [{"period": k, "unpaid_hours": v, "reduction": round(v * hourly, 2)} for k, v in per_period.items()],
        "payroll_signoff_required": unpaid > rules["payroll_signoff_over_unpaid_hours"],
        "benefits_review_required": unpaid > rules["benefits_review_over_unpaid_hours"],
        "basis": f"annual salary / {data['hours_per_year']} hours; thresholds from payroll rules",
    }


class PayrollExecutor(AgentExecutor):
    def __init__(self, data: dict):
        self.data = data

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = await begin(context, event_queue)
        caller = context.call_context.user
        # Payroll data is for the time-off agent acting as a service, never for a
        # person's own token, whatever tenant that person is in.
        if getattr(caller, "kind", "user") != "service":
            await updater.reject(updater.new_agent_message([new_text_part("Only a service caller may ask the payroll agent.")]))
            return
        args = request_data(context)
        if args.get("skill") != SKILL:
            await updater.reject(updater.new_agent_message(text_and_data(f"Unknown skill {args.get('skill')!r}; this agent offers {SKILL}.")))
            return
        try:
            result = assess(self.data, args)
        except PayrollError as exc:
            await updater.reject(updater.new_agent_message([new_text_part(str(exc))]))
            return
        await updater.add_artifact([new_data_part(result)], name="payroll_impact", last_chunk=True)
        await updater.complete(updater.new_agent_message([new_text_part(
            f"{result['unpaid_hours']:g}h unpaid is about {result['estimated_pay_reduction']:,.2f} {result['currency']}."
        )]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        from a2a.server.tasks.task_updater import TaskUpdater

        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="Payroll impact agent",
        description="Estimates the pay effect of unpaid leave for one tenant. Deterministic; reads its own compensation data.",
        provider=AgentProvider(organization="hr-timeoff-agent (synthetic)", url=base_url),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["application/json"],
        default_output_modes=["application/json", "text/plain"],
        skills=[AgentSkill(
            id=SKILL, name="Assess unpaid-leave impact",
            description="Given tenant_id, worker_id, unpaid_hours, start and end, returns the estimated pay reduction by pay period and whether payroll sign-off or a benefits review is required.",
            tags=["payroll", "unpaid-leave"], examples=['{"skill": "assess_unpaid_leave_impact", "tenant_id": "TEN-001", "worker_id": "W-100237", "unpaid_hours": 80, "start": "2026-10-19", "end": "2026-11-06"}'],
            input_modes=["application/json"], output_modes=["application/json"],
        )],
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", protocol_version="1.0", url=f"{base_url}{RPC_PATH}")],
        security_schemes={"bearer": SecurityScheme(http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="Bearer", description="Service token"))},
        security_requirements=[SecurityRequirement(schemes={"bearer": StringList()})],
    )


def create_payroll_app(tokens: BearerTokens, *, base_url: str = "http://127.0.0.1:8101", data_path: Path = DATA, home: Path | None = None) -> FastAPI:
    data = json.loads(Path(data_path).read_text())
    card = agent_card(base_url)
    handler = DefaultRequestHandler(agent_executor=PayrollExecutor(data), task_store=make_task_store("payroll_tasks", home), agent_card=card)
    app = FastAPI(title="Payroll impact agent")
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card=card),
        jsonrpc_routes=create_jsonrpc_routes(request_handler=handler, rpc_url=RPC_PATH, context_builder=BearerContextBuilder(tokens)),
    )
    require_bearer(app, tokens, RPC_PATH)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "agent": "payroll"}

    @app.get("/readyz")
    def readyz():
        return {"ready": True, "failing": {}}   # no external dependency: its data is loaded at construction

    telemetry.instrument_app(app, "a2a-payroll")
    return app
