"""Anthropic access with a fixture-backed offline mode.

The repo has to run for someone who just cloned it and has no API key, so every
model call goes through a content-addressed cache. Offline is the default when
no credentials are present; `--record` repopulates the cache against the live
API.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "llm_cache.json"

# Claude Opus 5 runs adaptive thinking when `thinking` is omitted, which is what
# we want for the assessment call.
AGENT_MODEL = os.environ.get("HR_AGENT_MODEL", "claude-opus-5")

# Kept separate so the judge is never the same instance being graded. Point this
# at a different model to guarantee nothing marks its own homework.
JUDGE_MODEL = os.environ.get("HR_AGENT_JUDGE_MODEL", "claude-opus-5")

T = TypeVar("T", bound=BaseModel)


class OfflineCacheMiss(RuntimeError):
    pass


def is_offline() -> bool:
    if os.environ.get("HR_AGENT_OFFLINE") == "1":
        return True
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

    if is_offline() and not record:
        if key not in cache:
            raise OfflineCacheMiss(
                f"No cached response for {label or schema.__name__} (key {key}).\n"
                "Offline mode can only replay recorded calls. Either:\n"
                "  • set ANTHROPIC_API_KEY and re-run with --record, or\n"
                "  • run one of the requests that ships with recorded fixtures."
            )
        return schema.model_validate(cache[key]["response"])

    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=schema,
    )
    parsed = response.parsed_output

    cache[key] = {
        "label": label or schema.__name__,
        "model": model,
        "schema": schema.__name__,
        "response": parsed.model_dump(),
    }
    _save_cache(cache)
    return parsed
