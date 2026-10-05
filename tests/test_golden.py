"""The interfaces other things depend on must not change by accident.

Pins the CLI (every command and its flags) and the MCP tool contract (names, arguments, read-only
flag). Refactors are allowed to move code, not to change these. If a change here is intended, regenerate:

    .venv/bin/python tests/test_golden.py --update

Run: .venv/bin/python tests/test_golden.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["HR_AGENT_OFFLINE"] = "1"
for var in ("HR_DATABASE_URL", "HR_OIDC_ISSUER", "ANTHROPIC_API_KEY"):
    os.environ.pop(var, None)

from hr_timeoff_agent.adapters import retrieval  # noqa: E402
from hr_timeoff_agent.cli import main  # noqa: E402
from hr_timeoff_agent.hr_tools.client import list_tools  # noqa: E402
from hr_timeoff_agent.hr_tools.server import HRToolServer  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "interfaces.json"


def _help(*argv: str) -> str:
    out = io.StringIO()
    os.environ["COLUMNS"] = "100"
    with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
        main([*argv, "-h"])
    return out.getvalue()


def current() -> dict:
    top = _help()
    commands = sorted(set(re.search(r"\{([^}]+)\}", top).group(1).split(",")))
    # Flags only, not the help text: argparse formats help differently across Python versions.
    cli = {name: sorted(set(re.findall(r"--[a-z][a-z0-9-]*", _help(name)))) for name in commands}
    tools = sorted(list_tools(HRToolServer(index=retrieval.PolicyIndex(), reader="manager")), key=lambda t: t["name"])
    return {"cli_commands": commands, "cli_help": cli, "mcp_tools": tools}


def test_the_cli_and_the_mcp_tool_contract_are_unchanged():
    want, got = json.loads(GOLDEN.read_text()), current()
    assert got["cli_commands"] == want["cli_commands"], "a CLI command was added, removed or renamed"
    for name in want["cli_commands"]:
        assert got["cli_help"][name] == want["cli_help"][name], f"`{name}` changed its flags"
    assert got["mcp_tools"] == want["mcp_tools"], "an MCP tool's name, arguments or read-only flag changed"


if __name__ == "__main__":
    if "--update" in sys.argv:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(current(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {GOLDEN}")
        raise SystemExit(0)
    try:
        test_the_cli_and_the_mcp_tool_contract_are_unchanged()
        print("  PASS  test_the_cli_and_the_mcp_tool_contract_are_unchanged\n\nall tests passed")
    except AssertionError as exc:
        print(f"  FAIL  {exc}")
        raise SystemExit(1)
