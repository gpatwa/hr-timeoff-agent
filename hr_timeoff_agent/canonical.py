"""A stable JSON form and short digest of a tool result, so the same data hashes alike wherever it came from."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(value: Any) -> Any:
    """JSON-normal form: whole-number floats become ints, so 96 and 96.0 hash alike
    however a client happened to serialize them."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, default=str).encode()).hexdigest()[:16]
