"""Anthropic access with a fixture-backed offline mode.

The repo has to run for someone who just cloned it and has no API key, so every
model call goes through a content-addressed cache. Offline is the default when
no credentials are present; `--record` repopulates the cache against the live
model.

Two live backends, chosen with HR_AGENT_BACKEND:

  api         (default) the Anthropic API via the SDK; needs ANTHROPIC_API_KEY.
              This is the only backend a deployed service should use.
  claude-cli  headless Claude Code (`claude -p`), using whatever Claude Code is
              logged in with, e.g. a personal subscription. For recording
              fixtures locally. The answer is a real model response to the same
              prompt and schema, but Claude Code manages thinking and effort
              itself, so it is not byte-identical to the API call.

Either way the response lands in the same cache under the same key, tagged with
the backend that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

# Overridable so a self-test can record into a scratch copy, never the committed file.
FIXTURES = Path(
    os.environ.get("HR_AGENT_FIXTURES")
    or Path(__file__).resolve().parent.parent / "fixtures" / "llm_cache.json"
)

# Claude Opus 5.5 always thinks adaptively; effort is the only control, and its
# default is "medium", so we set it explicitly below.
AGENT_MODEL = os.environ.get("HR_AGENT_MODEL", "claude-opus-5-5")

# Kept separate so the judge is never the same instance being graded. Point this
# at a different model to guarantee nothing marks its own homework.
JUDGE_MODEL = os.environ.get("HR_AGENT_JUDGE_MODEL", "claude-opus-5-5")

EFFORT = os.environ.get("HR_AGENT_EFFORT", "medium")

BACKEND = os.environ.get("HR_AGENT_BACKEND", "api")

T = TypeVar("T", bound=BaseModel)


class OfflineCacheMiss(RuntimeError):
    pass


def is_offline() -> bool:
    if os.environ.get("HR_AGENT_OFFLINE") == "1":
        return True
    if BACKEND == "claude-cli":
        return False
    return not os.environ.get("ANTHROPIC_API_KEY")


def _key(model: str, system: str, user: str, schema: Type[BaseModel]) -> str:
    blob = json.dumps(
        {"model": model, "system": system, "user": user, "schema": schema.__name__},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _load_cache() -> dict:
    if not FIXTURES.exists():
        return {}
    return json.loads(FIXTURES.read_text())


def _save_cache(cache: dict) -> None:
    FIXTURES.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def structured(
    *,
    system: str,
    user: str,
    schema: Type[T],
    model: str = AGENT_MODEL,
    record: bool = False,
    label: str = "",
) -> T:
    """Return a validated `schema` instance, from cache or from the live API."""
    cache = _load_cache()
    key = _key(model, system, user, schema)

    # A cached response is replayed unless --record asks for a fresh one. Live
    # mode only calls the model on a miss, so changing one prompt re-records
    # only the calls that changed.
    if key in cache and not record:
        return schema.model_validate(cache[key]["response"])
    if is_offline() and not record:
        raise OfflineCacheMiss(
            f"No cached response for {label or schema.__name__} (key {key}).\n"
            "Offline mode can only replay recorded calls. Either:\n"
            "  • set ANTHROPIC_API_KEY and re-run with --record,\n"
            "  • or HR_AGENT_BACKEND=claude-cli to record through Claude Code, or\n"
            "  • run one of the requests that ships with recorded fixtures."
        )

    if BACKEND == "claude-cli":
        parsed, served_by = _via_claude_cli(model=model, system=system, user=user, schema=schema)
        source = "claude-code-cli"
    else:
        parsed, served_by = _via_api(model=model, system=system, user=user, schema=schema)
        source = "anthropic-api"

    cache[key] = {
        "label": label or schema.__name__,
        "model": model,
        "schema": schema.__name__,
        "source": source,
        "served_by": served_by,
        "response": parsed.model_dump(),
    }
    _save_cache(cache)
    return parsed


def _via_api(*, model: str, system: str, user: str, schema: Type[T]) -> tuple[T, list[str]]:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        # Thinking counts toward max_tokens, so leave room beyond the reply itself.
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": EFFORT},
        output_format=schema,
    )
    return response.parsed_output, [response.model]


def claude_cli_command(*, model: str, system: str, schema: Type[BaseModel]) -> list[str]:
    """The headless Claude Code call: our system prompt, no tools, no MCP servers,
    no settings, no saved session, and output validated against the schema."""
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("HR_AGENT_BACKEND=claude-cli needs the `claude` CLI on PATH.")
    return [
        exe, "-p",
        "--model", model,
        "--system-prompt", system,
        "--tools", "",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--no-session-persistence",
        "--output-format", "json",
        "--json-schema", json.dumps(schema.model_json_schema()),
    ]


def _via_claude_cli(*, model: str, system: str, user: str, schema: Type[T]) -> tuple[T, list[str]]:
    cmd = claude_cli_command(model=model, system=system, schema=schema)
    # An empty working directory, so no project CLAUDE.md or memory is loaded
    # into what should be exactly our prompt.
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(cmd, input=user, capture_output=True, text=True, cwd=cwd, timeout=600)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude -p failed (exit {proc.returncode}): {proc.stderr.strip()[:500]}")
    if out.get("is_error") or out.get("structured_output") is None:
        raise RuntimeError(f"claude -p returned an error: {str(out.get('result'))[:500]}")
    served_by = sorted((out.get("modelUsage") or {}).keys())
    if model not in served_by:
        raise RuntimeError(f"asked for {model} but Claude Code used {served_by}; not caching it")
    return schema.model_validate(out["structured_output"]), served_by
