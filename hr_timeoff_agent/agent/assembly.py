"""Assembling the graph: wires the nodes in `graph` (and, in multi-agent mode, the specialists in
`agents`) into one compiled workflow. This is the one place that knows about both, so the two no longer
import each other.
"""

from __future__ import annotations

import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ..core import policy
from .graph import AgentState, _index, make_nodes, traced_node


def tool_host(tenant: policy.Tenant):
    """The HR tool server for this tenant, built once. Needs the mcp extra."""
    from ..tools.server import HRToolServer

    if not hasattr(tenant, "_mcp_server"):
        tenant._mcp_server = HRToolServer(tenant, index=_index(tenant), reader="manager")
    return tenant._mcp_server


def agent_mode() -> str:
    """'single' (default): one assessment call. 'multi': tool-calling specialists
    plus a coordinator. Chosen per process with HR_AGENT_MODE or per graph with build()."""
    return os.environ.get("HR_AGENT_MODE", "single")


def build(tenant: policy.Tenant, *, record_llm: bool = False, checkpointer=None, agents: str | None = None):
    """Compile the graph. The checkpointer holds paused runs: in memory by default
    (CLI, tests), or durable (the web app's SQLite) so a pending approval
    survives a restart. `agents` picks the single-agent or multi-agent assessment."""
    mode = agents or agent_mode()
    if mode not in ("single", "multi"):
        raise ValueError(f"agents must be 'single' or 'multi', not {mode!r}")
    load_context, check_policy, retrieve, assess, approval_gate, record = make_nodes(
        tenant, record_llm=record_llm
    )
    g = StateGraph(AgentState)
    g.add_node("load_context", traced_node("load_context", load_context))
    g.add_node("check_policy", traced_node("check_policy", check_policy))
    g.add_node("approval_gate", traced_node("approval_gate", approval_gate))
    g.add_node("record", traced_node("record", record))
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "check_policy")
    if mode == "single":
        g.add_node("retrieve", traced_node("retrieve", retrieve))
        g.add_node("assess", traced_node("assess", assess))
        g.add_edge("check_policy", "retrieve")
        g.add_edge("retrieve", "assess")
        g.add_edge("assess", "approval_gate")
    else:
        from .agents import make_multi_nodes  # needs the mcp extra

        policy_specialist, coverage_specialist, coordinate = make_multi_nodes(tenant, tool_host(tenant), record_llm=record_llm)
        g.add_node("policy_specialist", traced_node("policy_specialist", policy_specialist))
        g.add_node("coverage_specialist", traced_node("coverage_specialist", coverage_specialist))
        g.add_node("assess", traced_node("assess", coordinate))
        g.add_edge("check_policy", "policy_specialist")
        g.add_edge("policy_specialist", "coverage_specialist")
        g.add_edge("coverage_specialist", "assess")
        g.add_edge("assess", "approval_gate")
    g.add_edge("approval_gate", "record")
    g.add_edge("record", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
