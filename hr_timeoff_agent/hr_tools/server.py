"""The HR tools, served over the Model Context Protocol.

This is the one boundary through which an agent reads HR data. The server is
bound to a single tenant and a single reader audience when it is built; neither
appears in any tool's arguments, so a model that calls these tools cannot ask for
another tenant's records or for HR-only guidance. Every tool is read-only, and
every call is appended to an audit log that the caller can fold into the
evidence ledger.

Policy tools are deterministic: `evaluate_policy` runs the same rules engine the
graph runs. Search tools run the tenant- and audience-filtered hybrid query.

    python -m hr_timeoff_agent mcp                 # stdio, for an MCP host
    python -m hr_timeoff_agent mcp --http 8765     # streamable HTTP
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from datetime import date
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from pydantic import BaseModel

from ..core.canonical import canonical, digest  # noqa: F401  (re-exported)
from ..core import policy
from ..adapters import retrieval
from ..core.models import Finding, Passage

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

INSTRUCTIONS = (
    "Read-only HR tools for one tenant. Policy outcomes come from a deterministic rules engine: "
    "call evaluate_policy rather than judging a rule yourself. The tools return facts and "
    "passages; they never approve, decline or change anything."
)


class WorkerRecord(BaseModel):
    worker_id: str
    legal_name: str
    supervisory_org: str
    supervisory_org_id: str
    manager_id: str | None
    time_off_plans: dict


class Balance(BaseModel):
    worker_id: str
    plan: str
    balance_hours: float


class Findings(BaseModel):
    findings: list[Finding]


class Passages(BaseModel):
    passages: list[Passage]


def _dates(*values: str) -> None:
    for v in values:
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ToolError(f"{v!r} is not an ISO date (YYYY-MM-DD).") from None


class HRToolServer:
    """Builds an MCP server over one tenant's data, with an audit log of every call."""

    def __init__(self, tenant: policy.Tenant | None = None, index: retrieval.PolicyIndex | None = None, *, reader: str = "manager"):
        if reader not in retrieval.AUDIENCES:
            raise ValueError(f"unknown reader {reader!r}")
        self.tenant = tenant or policy.Tenant()
        self.reader = reader
        self._index = index
        self.audit: list[dict] = []
        self.server = MCPServer("hr-timeoff-tools", instructions=INSTRUCTIONS, log_level="WARNING")
        self._register()

    @property
    def index(self) -> retrieval.PolicyIndex:
        # Built on first use: embedding the corpus is the slow part.
        if self._index is None:
            # HR_CORPUS_DIR points a tool server at a different handbook and precedents. The eval
            # sets it for the hostile-corpus cases: Claude Code starts this server as a child process,
            # which would otherwise search the real corpus and never meet the planted passage.
            corpus = os.environ.get("HR_CORPUS_DIR")
            self._index = retrieval.PolicyIndex(data_dir=Path(corpus)) if corpus else retrieval.PolicyIndex()
        return self._index

    def _audited(self, fn):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            entry = {"tool": fn.__name__, "arguments": kwargs, "tenant_id": self.tenant.tenant_id, "reader": self.reader}
            try:
                result = fn(*args, **kwargs)
            except ToolError as exc:
                self.audit.append({**entry, "error": str(exc)})
                raise
            self.audit.append({**entry, "result_sha256": digest(result.model_dump() if isinstance(result, BaseModel) else result)})
            return result

        return run

    def _worker(self, worker_id: str) -> dict:
        worker = self.tenant.workers.get(worker_id)
        if worker is None:
            raise ToolError(f"{worker_id!r} is not a worker in tenant {self.tenant.tenant_id}.")
        return worker

    def _register(self) -> None:
        tenant, server = self.tenant, self.server

        def tool(fn):
            return server.tool(annotations=READ_ONLY)(self._audited(fn))

        @tool
        def get_worker(worker_id: str) -> WorkerRecord:
            """A worker's directory record: name, supervisory org, manager and plans."""
            w = self._worker(worker_id)
            return WorkerRecord(**{k: w.get(k) for k in WorkerRecord.model_fields})

        @tool
        def get_balance(worker_id: str, plan: str) -> Balance:
            """Hours available in one time-off plan, such as PTO."""
            w = self._worker(worker_id)
            if plan not in w["time_off_plans"]:
                raise ToolError(f"{w['legal_name']} has no {plan!r} plan.")
            return Balance(worker_id=worker_id, plan=plan, balance_hours=w["time_off_plans"][plan]["balance_hours"])

        @tool
        def team_availability(worker_id: str, start: str, end: str) -> Finding:
            """How many people in the worker's supervisory org are out between two dates, if they were also out."""
            w = self._worker(worker_id)
            _dates(start, end)
            return policy.check_coverage(tenant, {"worker_id": worker_id, "from": start, "to": end}, w)

        @tool
        def evaluate_policy(worker_id: str, plan: str, start: str, end: str, hours: float, submitted_at: str, note: str = "") -> Findings:
            """Run every deterministic time-off rule against a request and return one finding per rule."""
            w = self._worker(worker_id)
            _dates(start, end, submitted_at)
            request = {"worker_id": worker_id, "plan": plan, "from": start, "to": end, "hours": hours, "submitted_at": submitted_at, "note": note}
            return Findings(findings=policy.evaluate(tenant, request, w))

        @tool
        def search_handbook(query: str) -> Passages:
            """Search this tenant's handbook for guidance a manager may read."""
            return Passages(passages=self.index.search_handbook(query, tenant_id=tenant.tenant_id, reader=self.reader))

        @tool
        def search_precedents(query: str) -> Passages:
            """Search this tenant's past time-off decisions for similar situations."""
            return Passages(passages=self.index.search_precedents(query, tenant_id=tenant.tenant_id))

        @server.resource("policy://rules", name="time-off policy rules", mime_type="application/json")
        def rules() -> str:
            """The active policy: rule ids, thresholds, blackout periods and the approval rule."""
            return json.dumps(tenant.policy, indent=2)

    def run(self, transport: str = "stdio", **kwargs) -> None:
        self.server.run(transport, **kwargs)
