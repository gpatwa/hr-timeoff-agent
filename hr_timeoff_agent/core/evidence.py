"""Append-only, hash-chained evidence ledger.

Every entry commits to the one before it, so any later edit to a recorded step
breaks verification for that entry and everything after it. This is what makes
"the agent recommended X and a human approved it" an auditable claim rather
than a log line someone could quietly rewrite.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

Actor = Literal["rule", "agent", "human", "system"]

GENESIS = "0" * 64


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def append(
    ledger: list[dict],
    *,
    actor: Actor,
    node: str,
    summary: str,
    data: dict | None = None,
    at: str | None = None,
) -> list[dict]:
    """Return a new ledger with one entry appended. Does not mutate the input."""
    prev_hash = ledger[-1]["hash"] if ledger else GENESIS
    body = {
        "seq": len(ledger),
        "at": at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actor": actor,
        "node": node,
        "summary": summary,
        "data": data or {},
        "prev_hash": prev_hash,
    }
    return [*ledger, {**body, "hash": _digest(body)}]


def verify(ledger: list[dict]) -> tuple[bool, str]:
    """Recompute the chain. Returns (ok, reason)."""
    expected_prev = GENESIS
    for i, entry in enumerate(ledger):
        if entry["seq"] != i:
            return False, f"entry {i} has seq {entry['seq']}"
        if entry["prev_hash"] != expected_prev:
            return False, f"entry {i} does not link to entry {i - 1}"
        body = {k: v for k, v in entry.items() if k != "hash"}
        if _digest(body) != entry["hash"]:
            return False, f"entry {i} content does not match its hash"
        expected_prev = entry["hash"]
    return True, "chain intact"


def human_decisions(ledger: list[dict]) -> list[dict]:
    return [e for e in ledger if e["actor"] == "human"]


def render(ledger: list[dict]) -> str:
    """Plain-text rendering for the CLI."""
    lines = []
    for e in ledger:
        lines.append(
            f"  [{e['seq']}] {e['at']}  {e['actor']:<6} {e['node']:<14} {e['summary']}"
        )
        lines.append(f"        hash {e['hash'][:16]}…  prev {e['prev_hash'][:16]}…")
    return "\n".join(lines)
