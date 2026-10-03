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

# The agent runs on Sonnet 5.5: the production tier, at half Opus 5.5's price
# ($2/$10 vs $4/$20 per MTok). Effort is set explicitly below rather than left
# to the model's default.
AGENT_MODEL = os.environ.get("HR_AGENT_MODEL", "claude-sonnet-5-5")

# The judge is deliberately a different, stronger model than the agent, so
# nothing grades its own output.
JUDGE_MODEL = os.environ.get("HR_AGENT_JUDGE_MODEL", "claude-opus-5-5")

EFFORT = os.environ.get("HR_AGENT_EFFORT", "medium")

BACKEND = os.environ.get("HR_AGENT_BACKEND", "api")

# USD per million tokens (input, output), from the published price list. Used to
# account for spend; a model missing here is costed conservatively.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
UNKNOWN_MODEL_CALL_USD = 0.25

# Optional hooks for an embedding application (the web app's spend cap).
#   before_live_call(model, label)            may raise to refuse the call
#   after_live_call(model, label, usd, source) records what it cost
before_live_call = None
after_live_call = None

T = TypeVar("T", bound=BaseModel)


class OfflineCacheMiss(RuntimeError):
    pass


class ModelUnavailable(RuntimeError):
    """The model (or a tool it needed) could not be reached or timed out, after retries.

    Different from a bad request or a bad key, which are bugs to fix: this one is
    an outage to ride out, and the graph answers it by escalating to the human
    without a recommendation instead of failing the triage.
    """


# Bounded, explicit: a request waits at most TIMEOUT_S per attempt and
# MAX_RETRIES retries (the SDK backs off and honours retry-after) before the
# outage path takes over.
TIMEOUT_S = float(os.environ.get("HR_AGENT_TIMEOUT_S", "90"))
MAX_RETRIES = int(os.environ.get("HR_AGENT_MAX_RETRIES", "3"))


def api_client():
    import anthropic

    return anthropic.Anthropic(timeout=anthropic.Timeout(TIMEOUT_S, connect=10.0), max_retries=MAX_RETRIES)


class translate_outages:
    """Context manager: turn "the service is down or slow" into ModelUnavailable.

    Connection failures, timeouts, 429 and 5xx (including 529 overloaded) are
    outages. 4xx such as a bad key or a malformed request are not: they stay as
    they are so they are fixed, not papered over.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            return False
        try:
            import anthropic
        except ImportError:  # pragma: no cover
            return False
        if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.RateLimitError, anthropic.InternalServerError)):
            raise ModelUnavailable(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
        if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
            raise ModelUnavailable(f"HTTP {exc.status_code}: {str(exc)[:200]}") from exc
        if isinstance(exc, subprocess.TimeoutExpired):
            raise ModelUnavailable(f"claude -p timed out after {exc.timeout}s") from exc
        return False


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

    if before_live_call:
        before_live_call(model, label or schema.__name__)
    with translate_outages():
        if BACKEND == "claude-cli":
            parsed, served_by, usd = _via_claude_cli(model=model, system=system, user=user, schema=schema)
            source = "claude-code-cli"
        else:
            parsed, served_by, usd = _via_api(model=model, system=system, user=user, schema=schema)
            source = "anthropic-api"
    if after_live_call:
        after_live_call(model, label or schema.__name__, usd, source)

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


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICES:
        return UNKNOWN_MODEL_CALL_USD
    per_in, per_out = PRICES[model]
    return round((input_tokens * per_in + output_tokens * per_out) / 1_000_000, 6)


def _via_api(*, model: str, system: str, user: str, schema: Type[T]) -> tuple[T, list[str], float]:
    client = api_client()
    response = client.messages.parse(
        model=model,
        # Thinking counts toward max_tokens, so leave room beyond the reply itself.
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": EFFORT},
        output_format=schema,
    )
    usage = response.usage
    input_tokens = (usage.input_tokens or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
    usd = cost_usd(model, input_tokens, usage.output_tokens or 0)
    return response.parsed_output, [response.model], usd


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


def _via_claude_cli(*, model: str, system: str, user: str, schema: Type[T]) -> tuple[T, list[str], float]:
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
    # Claude Code reports an equivalent cost even on a subscription.
    usd = float(out.get("total_cost_usd") or 0.0)
    return schema.model_validate(out["structured_output"]), served_by, usd
