"""The four things the product depends on that can vary: the model, the store, retrieval and identity.

These are interfaces only (typing.Protocol), so anything that depends on a port can be given a fake, and a
new adapter has a contract to meet. They import only the core models. tests/test_ports.py checks that every
real adapter in the repo meets its port.

    ModelPort      llm.structured is the adapter: Anthropic API, Claude Code or recorded fixtures
    StorePort      storage.FileStore (SQLite and JSON files) and storage.PostgresStore
    RetrieverPort  retrieval.PolicyIndex (Qdrant)
    IdentityPort   web.identity.PersonaSwitcher and web.identity.OIDCIdentity
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Callable, Protocol, Type, TypeVar, runtime_checkable

from pydantic import BaseModel

from .models import Passage

T = TypeVar("T", bound=BaseModel)


class ModelPort(Protocol):
    def __call__(self, *, system: str, user: str, schema: Type[T], model: str = ..., record: bool = ..., label: str = ...) -> T:
        """A validated `schema` instance for this prompt: from a recording, or from the live model."""


@runtime_checkable
class StorePort(Protocol):
    def ping(self) -> None: ...
    def has_documents(self) -> bool: ...
    def seed_documents(self, source: Path) -> None: ...
    def read(self, name: str) -> Any: ...
    def mutate(self, name: str, fn: Callable[[Any], Any]) -> None: ...
    def mutate_many(self, names: list[str], fn: Callable[[dict[str, Any]], None]) -> None: ...
    def lock(self, key: str) -> contextlib.AbstractContextManager: ...
    def meta_get(self, key: str) -> str | None: ...
    def meta_set(self, key: str, value: str) -> None: ...
    def spend_add(self, at: str, persona: str | None, model: str, label: str, usd: float, source: str) -> None: ...
    def spend_count_since(self, persona: str, since: str) -> int: ...
    def spend_on_day(self, day: str) -> float: ...
    def spend_recent(self, n: int) -> list[tuple]: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class RetrieverPort(Protocol):
    def search_handbook(self, query: str, *, tenant_id: str, reader: str = ..., k: int = ...) -> list[Passage]: ...
    def search_precedents(self, query: str, *, tenant_id: str, k: int = ...) -> list[Passage]: ...


@runtime_checkable
class IdentityPort(Protocol):
    def current(self, request: Any) -> str | None: ...
    def sign_in(self, response: Any, worker_id: str) -> None: ...
    def sign_out(self, response: Any) -> None: ...


@runtime_checkable
class ToolHostPort(Protocol):
    """What the agent runtime needs from a tool server: the MCP server object to connect to, and the
    audience it reads for. The HR tool server is the one adapter."""

    server: Any
    reader: str
