"""Tests for the claude-cli recording backend, with the subprocess faked.

CI never calls Claude Code; these pin down what we send and what we refuse to cache.

Run: .venv/bin/python tests/test_llm.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hr_timeoff_agent import llm  # noqa: E402
from hr_timeoff_agent.models import Recommendation  # noqa: E402

REC = {"action": "decline", "rationale": "r", "cited_rule_ids": ["BAL-01"], "cited_passage_ids": [], "confidence": "high"}


def _fake_run(payload: dict, calls: list):
    def run(cmd, input, capture_output, text, cwd, timeout):  # noqa: A002 - mirrors subprocess.run
        calls.append({"cmd": cmd, "input": input, "cwd": cwd})
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")
    return run


def _with_fake(payload: dict):
    calls: list = []
    original_run, original_which = llm.subprocess.run, llm.shutil.which
    llm.subprocess.run = _fake_run(payload, calls)
    llm.shutil.which = lambda name: "/usr/local/bin/claude"
    try:
        result = llm._via_claude_cli(
            model="claude-opus-5-5", system="SYSTEM", user="USER", schema=Recommendation
        )
    finally:
        llm.subprocess.run, llm.shutil.which = original_run, original_which
    return result, calls


def test_command_isolates_the_call():
    """Our system prompt, no tools, no MCP servers, no settings, schema-validated output."""
    llm_which = llm.shutil.which
    llm.shutil.which = lambda name: "/usr/local/bin/claude"
    try:
        cmd = llm.claude_cli_command(model="claude-opus-5-5", system="SYSTEM", schema=Recommendation)
    finally:
        llm.shutil.which = llm_which
    pairs = dict(zip(cmd, cmd[1:]))
    assert "-p" in cmd and "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd
    assert pairs["--model"] == "claude-opus-5-5"
    assert pairs["--system-prompt"] == "SYSTEM"
    assert pairs["--tools"] == "" and pairs["--setting-sources"] == ""
    assert pairs["--output-format"] == "json"
    assert json.loads(pairs["--json-schema"])["properties"]["action"]["enum"] == ["approve", "decline", "escalate"]


def test_parses_structured_output_and_sends_the_prompt_on_stdin():
    (rec, served_by, usd), calls = _with_fake(
        {"is_error": False, "structured_output": REC, "modelUsage": {"claude-opus-5-5": {}}, "total_cost_usd": 0.0123}
    )
    assert rec.action == "decline" and served_by == ["claude-opus-5-5"]
    assert usd == 0.0123, "the reported cost feeds the spend cap"
    assert calls[0]["input"] == "USER"
    assert Path(calls[0]["cwd"]).name != "hr-timeoff-agent", "must run outside the repo"


def test_refuses_a_response_from_a_different_model():
    try:
        _with_fake({"is_error": False, "structured_output": REC, "modelUsage": {"claude-sonnet-5": {}}})
    except RuntimeError as exc:
        assert "not caching it" in str(exc)
        return
    raise AssertionError("cached a response served by the wrong model")


def test_refuses_an_error_result():
    try:
        _with_fake({"is_error": True, "result": "API Error: 400", "modelUsage": {}})
    except RuntimeError as exc:
        assert "API Error: 400" in str(exc)
        return
    raise AssertionError("accepted an error result")


def test_cost_uses_the_price_list():
    assert llm.cost_usd("claude-sonnet-5-5", 1_000_000, 0) == 2.0
    assert llm.cost_usd("claude-sonnet-5-5", 0, 1_000_000) == 10.0
    assert llm.cost_usd("some-unknown-model", 10, 10) == llm.UNKNOWN_MODEL_CALL_USD


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
