"""A small synchronous client for the HR tool server.

The agent graph is synchronous, so this wraps the async MCP client. Pass an
`ToolHostPort` for an in-process connection (tests, the demo) or leave it out to
spawn the server as a subprocess over stdio, which is how an external MCP host
would reach it.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

from .ports import ToolHostPort


class ToolCallError(RuntimeError):
    """The server refused or failed a tool call; the message is the server's."""


def stdio_params(reader: str = "manager") -> StdioServerParameters:
    # The child gets a minimal environment, so pass through what the package reads.
    env = {k: v for k, v in os.environ.items() if k.startswith(("HR_", "PYTHON"))}
    return StdioServerParameters(
        command=sys.executable, args=["-m", "hr_timeoff_agent", "mcp", "--reader", reader], env=env
    )


async def _call(target, calls: list[tuple[str, dict]]) -> list[Any]:
    out: list[Any] = []
    failure = None
    async with Client(target) as client:
        for name, arguments in calls:
            result = await client.call_tool(name, arguments)
            if result.is_error:
                failure = result.content[0].text if result.content else f"{name} failed"
                break
            out.append(result.structured_content)
    # Raised after the connection closes: inside it, anyio wraps it in an ExceptionGroup.
    if failure is not None:
        raise ToolCallError(failure)
    return out


def call_tools(calls: list[tuple[str, dict]], server: ToolHostPort | None = None, *, reader: str = "manager") -> list[Any]:
    """Run tool calls over one connection and return each structured result, in order."""
    target = server.server if server is not None else stdio_params(reader)
    return asyncio.run(_call(target, calls))


def call_tool(name: str, arguments: dict, server: ToolHostPort | None = None, *, reader: str = "manager") -> Any:
    return call_tools([(name, arguments)], server, reader=reader)[0]


async def _list(target) -> list[dict]:
    async with Client(target) as client:
        tools = await client.list_tools()
        return [
            {"name": t.name, "read_only": bool(t.annotations and t.annotations.read_only_hint), "arguments": sorted(t.input_schema.get("properties", {}))}
            for t in tools.tools
        ]


def list_tools(server: ToolHostPort | None = None, *, reader: str = "manager") -> list[dict]:
    return asyncio.run(_list(server.server if server is not None else stdio_params(reader)))


async def _try(target, calls: list[tuple[str, dict]]) -> list[tuple[bool, Any]]:
    out: list[tuple[bool, Any]] = []
    async with Client(target) as client:
        for name, arguments in calls:
            result = await client.call_tool(name, arguments)
            if result.is_error:
                out.append((False, result.content[0].text if result.content else f"{name} failed"))
            else:
                out.append((True, result.structured_content))
    return out


def try_tools(calls: list[tuple[str, dict]], server: ToolHostPort | None = None, *, reader: str = "manager") -> list[tuple[bool, Any]]:
    """Like call_tools, but a failed call is data, not an exception: (False, message).
    An agent's bad argument is something to show it, not something to crash on."""
    target = server.server if server is not None else stdio_params(reader)
    return asyncio.run(_try(target, calls))


async def _definitions(target, names: list[str]) -> list[dict]:
    async with Client(target) as client:
        listed = {t.name: t for t in (await client.list_tools()).tools}
    missing = [n for n in names if n not in listed]
    if missing:
        raise KeyError(f"no such tools: {missing}")
    return [
        {"name": n, "description": listed[n].description or "", "input_schema": listed[n].input_schema}
        for n in names
    ]


def tool_definitions(names: list[str], server: ToolHostPort | None = None, *, reader: str = "manager") -> list[dict]:
    """The named tools as Messages-API tool definitions, straight from the MCP server."""
    target = server.server if server is not None else stdio_params(reader)
    return asyncio.run(_definitions(target, names))
