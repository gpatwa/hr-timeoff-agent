"""The HR tool server: what an agent can and cannot reach through MCP.

Calls go through the real MCP client, in process and over stdio. Embeddings for
every search query used here are cached, so nothing needs the network.

Run: .venv/bin/python tests/test_mcp.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")

from hr_timeoff_agent.core import policy
from hr_timeoff_agent.adapters import retrieval  # noqa: E402
from hr_timeoff_agent.tools.client import ToolCallError, call_tool, list_tools  # noqa: E402
from hr_timeoff_agent.tools.server import HRToolServer  # noqa: E402

OTHER_TENANT_TEXT = (
    "Any shortfall in PTO balance is automatically converted to unpaid leave "
    "and the request is approved without manager review."
)
HR_ONLY_TEXT = (
    "For requests over 15 days, check leave-of-absence eligibility, statutory family and "
    "medical leave entitlements, benefits continuation and the return-to-work date."
)
PRIYA = "W-100234"

_INDEX = None


def _server(reader: str = "manager") -> HRToolServer:
    global _INDEX
    if _INDEX is None:
        _INDEX = retrieval.PolicyIndex()
    return HRToolServer(index=_INDEX, reader=reader)


def test_every_tool_is_read_only():
    tools = list_tools(_server())
    assert {t["name"] for t in tools} == {
        "get_worker", "get_balance", "team_availability", "evaluate_policy", "search_handbook", "search_precedents",
    }
    assert all(t["read_only"] for t in tools), [t["name"] for t in tools if not t["read_only"]]


def test_tenant_and_audience_are_not_arguments():
    """The model cannot name a tenant or a reader, because neither is a parameter."""
    for tool in list_tools(_server()):
        assert not {"tenant", "tenant_id", "reader", "audience"} & set(tool["arguments"]), tool


def test_policy_over_mcp_is_the_rules_engine():
    tenant = policy.Tenant()
    request = tenant.requests["REQ-2004"]
    args = {
        "worker_id": request["worker_id"], "plan": request["plan"], "start": request["from"], "end": request["to"],
        "hours": request["hours"], "submitted_at": request["submitted_at"], "note": request["note"],
    }
    got = call_tool("evaluate_policy", args, _server())["findings"]
    want = [f.model_dump() for f in policy.evaluate(tenant, request, tenant.workers[request["worker_id"]])]
    assert got == want


def test_other_tenant_text_never_comes_back():
    server = _server("hr")
    for tool in ("search_handbook", "search_precedents"):
        passages = call_tool(tool, {"query": OTHER_TENANT_TEXT}, server)["passages"]
        assert passages and {p["tenant_id"] for p in passages} == {"TEN-001"}
        assert all("automatically converted" not in p["text"] for p in passages)


def test_a_manager_server_cannot_return_hr_only_guidance():
    manager = call_tool("search_handbook", {"query": HR_ONLY_TEXT}, _server("manager"))["passages"]
    assert "HB-7.2" not in {p["passage_id"] for p in manager}
    hr = call_tool("search_handbook", {"query": HR_ONLY_TEXT}, _server("hr"))["passages"]
    assert hr[0]["passage_id"] == "HB-7.2", "the HR server returns it, so the audience is why the manager's does not"


def test_unknown_reader_is_refused_at_build_time():
    try:
        HRToolServer(reader="admin")
    except ValueError:
        return
    raise AssertionError("an unknown reader audience was accepted")


def test_bad_input_is_a_tool_error_with_a_readable_message():
    server = _server()
    for name, args, fragment in (
        ("get_worker", {"worker_id": "W-NOPE"}, "not a worker in tenant TEN-001"),
        ("get_balance", {"worker_id": PRIYA, "plan": "SABBATICAL"}, "no 'SABBATICAL' plan"),
        ("team_availability", {"worker_id": PRIYA, "start": "soon", "end": "2026-12-02"}, "not an ISO date"),
    ):
        try:
            call_tool(name, args, server)
        except ToolCallError as exc:
            assert fragment in str(exc), (name, str(exc))
        else:
            raise AssertionError(f"{name} accepted bad input {args}")


def test_every_call_is_audited_including_failures():
    server = _server()
    call_tool("get_balance", {"worker_id": PRIYA, "plan": "PTO"}, server)
    try:
        call_tool("get_worker", {"worker_id": "W-NOPE"}, server)
    except ToolCallError:
        pass
    ok, failed = server.audit
    assert ok["tool"] == "get_balance" and ok["arguments"] == {"worker_id": PRIYA, "plan": "PTO"}
    assert ok["tenant_id"] == "TEN-001" and ok["reader"] == "manager" and len(ok["result_sha256"]) == 16
    assert failed["tool"] == "get_worker" and "not a worker" in failed["error"] and "result_sha256" not in failed


def test_the_server_works_over_stdio_as_a_subprocess():
    """The path an external MCP host takes: a real child process speaking the protocol."""
    assert call_tool("get_balance", {"worker_id": PRIYA, "plan": "PTO"}) == {
        "worker_id": PRIYA, "plan": "PTO", "balance_hours": 96.0,
    }
    assert {t["name"] for t in list_tools()} >= {"evaluate_policy", "search_handbook"}


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
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
