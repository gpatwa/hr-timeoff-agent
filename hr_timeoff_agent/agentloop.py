"""Tool-calling agent loops over the MCP tools, with replayable trajectories.

A specialist agent is given a task and a short list of MCP tools and decides for
itself which to call and with what arguments, then returns a typed report. The
loop runs one of two ways, the same backends `llm.py` offers:

  api         a manual Messages-API tool loop; the tools are the MCP server's.
  claude-cli  headless Claude Code, given the MCP server and an allowlist, which
              runs the loop itself and reports every tool call.

What gets cached is the trajectory, not just the answer: the tool calls the model
chose (name and arguments) and a digest of each result, plus the final report.
Replaying re-runs those tool calls against the real server and checks every result
digest, so an offline run still exercises the tools and notices when the data they
read has changed since recording. Only the model's choices come from the fixture.

Tools outside the allowlist are never offered, and a trajectory that uses one is
rejected on replay.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from . import llm
from .mcp_client import stdio_params, tool_definitions, try_tools
from .mcp_server import HRToolServer, digest

T = TypeVar("T", bound=BaseModel)

MAX_TOOL_CALLS = 8
MCP_SERVER_NAME = "hr"


class StaleTrajectory(RuntimeError):
    """A recorded trajectory no longer reproduces: a tool result changed or a call is not allowed."""


@dataclass
class ToolCall:
    tool: str
    arguments: dict
    result: Any = None
    result_sha256: str | None = None
    error: str | None = None

    def trace(self) -> dict:
        t: dict[str, Any] = {"tool": self.tool, "arguments": self.arguments}
        if self.error is not None:
            t["error"] = self.error
        else:
            t["result_sha256"] = self.result_sha256
        return t


@dataclass
class AgentRun:
    output: Any
    calls: list[ToolCall] = field(default_factory=list)
    served_by: list[str] = field(default_factory=list)
    usd: float = 0.0
    source: str = ""
    replayed: bool = False


def agent_key(model: str, system: str, user: str, schema: Type[BaseModel], tools: list[str]) -> str:
    return "agent-" + llm._key(model, system + "\n#tools:" + ",".join(sorted(tools)), user, schema)


def _execute(server: HRToolServer, wanted: list[tuple[str, dict]]) -> list[ToolCall]:
    out = []
    for (tool, arguments), (ok, value) in zip(wanted, try_tools(wanted, server)):
        out.append(
            ToolCall(tool, arguments, result=value, result_sha256=digest(value))
            if ok else ToolCall(tool, arguments, error=str(value))
        )
    return out


def _replay(server: HRToolServer, entry: dict, tools: list[str]) -> tuple[Any, list[ToolCall]]:
    trace = entry["trace"]
    for t in trace:
        if t["tool"] not in tools:
            raise StaleTrajectory(f"recorded call to {t['tool']!r}, which is not in this agent's allowlist {tools}")
    calls = _execute(server, [(t["tool"], t["arguments"]) for t in trace])
    for want, got in zip(trace, calls):
        if want.get("error") != got.error or want.get("result_sha256") != got.result_sha256:
            raise StaleTrajectory(
                f"{want['tool']}({json.dumps(want['arguments'], sort_keys=True)}) no longer returns what was recorded"
            )
    return entry["response"], calls


def run_agent(
    *,
    system: str,
    user: str,
    schema: Type[T],
    tools: list[str],
    server: HRToolServer,
    model: str = llm.AGENT_MODEL,
    record: bool = False,
    label: str = "",
) -> AgentRun:
    """Run one tool-calling agent: replay a recorded trajectory, or run it live."""
    cache = llm._load_cache()
    key = agent_key(model, system, user, schema, tools)
    stale: str | None = None

    entry = cache.get(key)
    if entry is not None and not record:
        try:
            response, calls = _replay(server, entry, tools)
            return AgentRun(schema.model_validate(response), calls, entry.get("served_by", []), 0.0, entry.get("source", ""), True)
        except StaleTrajectory as exc:
            stale = str(exc)
    if llm.is_offline() and not record:
        why = f"its recorded trajectory is stale: {stale}" if stale else "there is no recorded trajectory for it"
        raise llm.OfflineCacheMiss(
            f"Cannot replay agent {label or schema.__name__} (key {key}): {why}.\n"
            "Re-record with HR_AGENT_BACKEND=claude-cli (or ANTHROPIC_API_KEY) and --record."
        )

    if llm.before_live_call:
        llm.before_live_call(model, label or schema.__name__)
    with llm.translate_outages():
        if llm.BACKEND == "claude-cli":
            output, calls, served_by, usd = _via_claude_cli(model=model, system=system, user=user, schema=schema, tools=tools, reader=server.reader)
            source = "claude-code-cli"
        else:
            output, calls, served_by, usd = _via_api(model=model, system=system, user=user, schema=schema, tools=tools, server=server)
            source = "anthropic-api"
    if llm.after_live_call:
        llm.after_live_call(model, label or schema.__name__, usd, source)
    if len(calls) > MAX_TOOL_CALLS:
        raise RuntimeError(f"{label or schema.__name__} made {len(calls)} tool calls; the limit is {MAX_TOOL_CALLS}")

    cache[key] = {
        "kind": "agent",
        "label": label or schema.__name__,
        "model": model,
        "schema": schema.__name__,
        "tools": sorted(tools),
        "source": source,
        "served_by": served_by,
        "trace": [c.trace() for c in calls],
        "response": output.model_dump(),
    }
    llm._save_cache(cache)
    return AgentRun(output, calls, served_by, usd, source, False)


# ── live: the Messages API ──────────────────────────────────────────────────

def _via_api(*, model: str, system: str, user: str, schema: Type[T], tools: list[str], server: HRToolServer):
    import anthropic

    client = llm.api_client()
    defs = tool_definitions(tools, server)
    output_format = {"type": "json_schema", "schema": anthropic.transform_schema(schema.model_json_schema())}
    messages: list[dict] = [{"role": "user", "content": user}]
    calls: list[ToolCall] = []
    served: set[str] = set()
    usd = 0.0

    for _ in range(MAX_TOOL_CALLS + 2):
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            messages=messages,
            tools=defs,
            output_config={"effort": llm.EFFORT, "format": output_format},
        )
        served.add(response.model)
        u = response.usage
        usd += llm.cost_usd(model, (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0), u.output_tokens or 0)
        if response.stop_reason != "tool_use":
            if response.stop_reason != "end_turn":
                raise RuntimeError(f"agent stopped with {response.stop_reason!r}")
            text = next(b.text for b in response.content if b.type == "text")
            return schema.model_validate_json(text), calls, sorted(served), usd

        uses = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})  # thinking blocks must go back unchanged
        allowed = [b for b in uses if b.name in tools]
        ran = iter(_execute(server, [(b.name, dict(b.input)) for b in allowed]) if len(calls) + len(allowed) <= MAX_TOOL_CALLS else [])
        blocks = []
        for b in uses:
            if b.name not in tools:
                blocks.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True, "content": f"{b.name} is not available."})
                continue
            call = next(ran, None)
            if call is None:
                blocks.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True, "content": "Tool call limit reached; write your report now."})
                continue
            calls.append(call)
            blocks.append(
                {"type": "tool_result", "tool_use_id": b.id, "is_error": True, "content": call.error}
                if call.error is not None
                else {"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(call.result)}
            )
        messages.append({"role": "user", "content": blocks})
    raise RuntimeError("agent did not finish within the turn limit")


# ── live: headless Claude Code with the MCP server attached ─────────────────

def claude_cli_agent_command(*, model: str, system: str, schema: Type[BaseModel], tools: list[str], mcp_config: str) -> list[str]:
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("HR_AGENT_BACKEND=claude-cli needs the `claude` CLI on PATH.")
    return [
        exe, "-p",
        "--model", model,
        "--system-prompt", system,
        "--tools", "",                       # no built-in tools: only the MCP ones below
        "--strict-mcp-config", "--mcp-config", mcp_config,
        "--allowedTools", ",".join(f"mcp__{MCP_SERVER_NAME}__{t}" for t in tools),
        "--setting-sources", "",
        "--no-session-persistence",
        "--output-format", "stream-json", "--verbose",
        "--json-schema", json.dumps(schema.model_json_schema()),
    ]


def parse_claude_stream(stdout: str) -> tuple[dict, list[ToolCall]]:
    """The final result event, and every MCP tool call with its result, in order."""
    uses: dict[str, dict] = {}
    results: dict[str, tuple[bool, str]] = {}
    order: list[str] = []
    final: dict | None = None
    prefix = f"mcp__{MCP_SERVER_NAME}__"
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            final = event
        elif event.get("type") in ("assistant", "user"):
            content = event.get("message", {}).get("content")
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "tool_use" and block.get("name", "").startswith(prefix):
                    uses[block["id"]] = {"tool": block["name"][len(prefix):], "arguments": block.get("input") or {}}
                    order.append(block["id"])
                elif block.get("type") == "tool_result" and block.get("tool_use_id") in uses:
                    body = block.get("content")
                    if isinstance(body, list):
                        body = "".join(b.get("text", "") for b in body if isinstance(b, dict))
                    results[block["tool_use_id"]] = (bool(block.get("is_error")), body or "")
    if final is None:
        raise RuntimeError("claude -p produced no result event")
    calls = []
    for tid in order:
        is_error, body = results.get(tid, (True, "no result recorded"))
        use = uses[tid]
        if is_error:
            calls.append(ToolCall(use["tool"], use["arguments"], error=body))
            continue
        try:
            value = json.loads(body)
        except json.JSONDecodeError:
            value = body
        calls.append(ToolCall(use["tool"], use["arguments"], result=value, result_sha256=digest(value)))
    return final, calls


def _via_claude_cli(*, model: str, system: str, user: str, schema: Type[T], tools: list[str], reader: str):
    p = stdio_params(reader)
    with tempfile.TemporaryDirectory() as cwd:
        config = f"{cwd}/mcp.json"
        with open(config, "w") as f:
            json.dump({"mcpServers": {MCP_SERVER_NAME: {"command": p.command, "args": p.args, "env": p.env or {}}}}, f)
        cmd = claude_cli_agent_command(model=model, system=system, schema=schema, tools=tools, mcp_config=config)
        proc = subprocess.run(cmd, input=user, capture_output=True, text=True, cwd=cwd, timeout=900)
    try:
        final, calls = parse_claude_stream(proc.stdout)
    except RuntimeError:
        raise RuntimeError(f"claude -p failed (exit {proc.returncode}): {proc.stderr.strip()[:500]}")
    if final.get("is_error") or final.get("structured_output") is None:
        raise RuntimeError(f"claude -p returned an error: {str(final.get('result'))[:500]}")
    served_by = sorted((final.get("modelUsage") or {}).keys())
    if model not in served_by:
        raise RuntimeError(f"asked for {model} but Claude Code used {served_by}; not caching it")
    return schema.model_validate(final["structured_output"]), calls, served_by, float(final.get("total_cost_usd") or 0.0)
