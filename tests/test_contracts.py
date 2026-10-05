"""The two contracts other systems code against, as versioned files: the MCP tool definitions and the
A2A agent cards. The files in contracts/ are the source of truth for consumers; this fails if the code
drifts from them, and checks the code that consumes them only uses what they promise.

Changing a contract is allowed, on purpose: bump the version in the file, regenerate, and say so in the PR.

    .venv/bin/python tests/test_contracts.py --update

Run: .venv/bin/python tests/test_contracts.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["HR_AGENT_OFFLINE"] = "1"
for var in ("HR_DATABASE_URL", "HR_OIDC_ISSUER", "ANTHROPIC_API_KEY"):
    os.environ.pop(var, None)

from google.protobuf.json_format import MessageToDict  # noqa: E402
from mcp import Client  # noqa: E402

from hr_timeoff_agent.adapters import retrieval  # noqa: E402
from hr_timeoff_agent.services.a2a.payroll import agent_card as payroll_card  # noqa: E402
from hr_timeoff_agent.services.a2a.server import agent_card as timeoff_card  # noqa: E402
from hr_timeoff_agent.agent.agents import COVERAGE_TOOLS, POLICY_TOOLS  # noqa: E402
from hr_timeoff_agent.tools.server import HRToolServer  # noqa: E402

DIR = Path(__file__).resolve().parent.parent / "contracts"
MCP_FILE, A2A_FILE = DIR / "mcp-tools.json", DIR / "a2a-agent-cards.json"
BASE = "http://example.invalid"   # the cards carry their own address; the contract is everything else


def current_mcp() -> list[dict]:
    server = HRToolServer(index=retrieval.PolicyIndex(), reader="manager")

    async def go():
        async with Client(server.server) as client:
            return [t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in (await client.list_tools()).tools]

    return sorted(asyncio.run(go()), key=lambda t: t["name"])


def current_a2a() -> dict:
    return {"time-off agent": MessageToDict(timeoff_card(BASE)), "payroll agent": MessageToDict(payroll_card(BASE))}


def _wrap(version: str, body):
    return {"contract_version": version, "contract": body}


def test_the_mcp_tool_contract_matches_the_code():
    want = json.loads(MCP_FILE.read_text())
    assert current_mcp() == want["contract"], "an MCP tool's name, description or schema changed: bump contract_version and regenerate"


def test_the_a2a_agent_cards_match_the_code():
    want = json.loads(A2A_FILE.read_text())
    assert current_a2a() == want["contract"], "an agent card changed: bump contract_version and regenerate"


def test_the_agents_only_ask_for_tools_the_contract_offers():
    offered = {t["name"] for t in json.loads(MCP_FILE.read_text())["contract"]}
    assert set(POLICY_TOOLS) | set(COVERAGE_TOOLS) <= offered


def test_the_contract_files_carry_a_version():
    for f in (MCP_FILE, A2A_FILE):
        v = json.loads(f.read_text())["contract_version"]
        assert len(v.split(".")) == 3 and all(p.isdigit() for p in v.split(".")), (f.name, v)


if __name__ == "__main__":
    if "--update" in sys.argv:
        DIR.mkdir(exist_ok=True)
        for f, body in ((MCP_FILE, current_mcp()), (A2A_FILE, current_a2a())):
            old = json.loads(f.read_text())["contract_version"] if f.exists() else "1.0.0"
            f.write_text(json.dumps(_wrap(old, body), indent=2, sort_keys=True) + "\n")
            print(f"wrote {f} (version {old}; bump it by hand if the change is breaking)")
        raise SystemExit(0)
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}: {type(exc).__name__}: {str(exc)[:200]}")
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
