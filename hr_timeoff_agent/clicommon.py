"""Small things every command group prints or writes the same way."""

from __future__ import annotations

from pathlib import Path

from .llm import is_offline

OUT = Path(__file__).resolve().parent.parent / "out"
RULE = "─" * 74


def mode_banner() -> str:
    return "offline (fixture-backed)" if is_offline() else "live (Anthropic API)"
