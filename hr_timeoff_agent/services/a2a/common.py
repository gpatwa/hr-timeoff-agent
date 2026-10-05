"""Pieces shared by the two A2A agents: bearer-token identity, request parsing,
and the small helpers that turn a handler's result into A2A task events.

Identity here is a bearer token that maps to a principal (a worker id, or a
service name). The token directory is deliberately small and replaceable: a real
deployment puts OIDC or a gateway in front and fills ServerCallContext.user the
same way. Nothing downstream of `BearerContextBuilder` knows where a token came
from.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from pathlib import Path
from typing import Any

from a2a.auth.user import UnauthenticatedUser, User
from a2a.helpers import get_data_parts, get_message_text, new_data_part, new_text_part
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.routes.common import DefaultServerCallContextBuilder
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import Task, TaskState, TaskStatus
from starlette.requests import Request
from starlette.responses import JSONResponse

from ...adapters import telemetry


def make_task_store(table: str, home: Path | None = None):
    """Where an agent keeps its A2A tasks: Postgres when configured, else memory.

    In memory, a restart forgets every task a caller was waiting on. In Postgres
    the task (and so the owner that scopes it) survives, and either agent
    process can be restarted or scaled without dropping a pending approval.
    """
    from ...adapters.storage import database_settings, ensure_schema

    settings = database_settings(home or Path("."))
    if settings is None:
        from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

        return InMemoryTaskStore()
    import functools

    from a2a.server.tasks import database_task_store as dts
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    url, schema = settings
    ensure_schema(url, schema)
    # One connection per operation: tasks are written a handful of times each,
    # and a pooled async connection cannot move between event loops.
    engine = create_async_engine(
        url.replace("postgresql://", "postgresql+psycopg://", 1),
        poolclass=NullPool, connect_args={"options": f"-csearch_path={schema}"},
    )
    # The SDK defines a new ORM table every time it is asked for a named one,
    # and SQLAlchemy refuses a second definition. Build each table's model once.
    if not getattr(dts.create_task_model, "cache_info", None):
        dts.create_task_model = functools.cache(dts.create_task_model)
    return dts.DatabaseTaskStore(engine, table_name=table)


log = logging.getLogger(__name__)


class PeerBreaker:
    """Stops calling a peer that keeps failing, for a cool-down, then tries again.

    An advisory peer that is down should cost one slow review, not every review.
    After `threshold` failures in a row calls are skipped for `cooldown` seconds;
    one success closes it.
    """

    def __init__(self, threshold: int = 3, cooldown: float = 30.0, clock=None):
        import time

        self.threshold, self.cooldown = threshold, cooldown
        self._clock = clock or time.monotonic
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def open(self) -> bool:
        if self._opened_at is None:
            return False
        if self._clock() - self._opened_at >= self.cooldown:
            self._opened_at, self._failures = None, self.threshold - 1  # half-open: one more failure re-opens
            return False
        return True

    def is_open(self) -> bool:
        """Whether calls are being skipped right now. Unlike `open`, reading it changes nothing."""
        return self._opened_at is not None and self._clock() - self._opened_at < self.cooldown

    def success(self) -> None:
        self._failures, self._opened_at = 0, None

    def failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = self._clock()


SERVICE_PRINCIPALS = frozenset({"timeoff-agent"})


class Principal(User):
    """Who is calling: a worker id (kind "user") or a service name (kind "service")."""

    def __init__(self, name: str, kind: str = "user", tenant_id: str | None = None):
        self._name, self.kind, self.tenant_id = name, kind, tenant_id

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self._name


class BearerTokens:
    """token → principal. Compared in constant time."""

    def __init__(self, tokens: dict[str, str]):
        self._tokens = dict(tokens)

    @classmethod
    def derive(cls, secret: str, principals: list[str]) -> "BearerTokens":
        """Stable demo tokens: HMAC(secret, principal). Anyone with the secret can mint them."""
        return cls({
            hmac.new(secret.encode(), p.encode(), hashlib.sha256).hexdigest()[:32]: p for p in principals
        })

    def principal(self, authorization: str | None) -> str | None:
        if not authorization or not authorization.lower().startswith("bearer "):
            return None
        presented = authorization[7:].strip()
        found = None
        for token, principal in self._tokens.items():
            if hmac.compare_digest(token, presented):
                found = principal
        return found

    def authenticate(self, authorization: str | None) -> "Principal | None":
        name = self.principal(authorization)
        return Principal(name, "service" if name in SERVICE_PRINCIPALS else "user") if name else None

    def for_principal(self, principal: str) -> str:
        return next(t for t, p in self._tokens.items() if p == principal)

    def principals(self) -> list[str]:
        return list(self._tokens.values())


class OIDCBearer:
    """Bearer authentication with OIDC access tokens, for an agent that serves one tenant.

    The same interface as BearerTokens, so the agents do not know which they
    have. A token is accepted when it verifies (signature, issuer, audience,
    expiry), is for this tenant, and names a worker or a configured service.
    """

    def __init__(self, cfg, tenant_id: str, directory, *, verifier=None, surface: str = "a2a"):
        from ...adapters.oidc import TokenVerifier

        self.cfg, self.tenant_id, self.directory, self.surface = cfg, tenant_id, directory, surface
        self.verifier = verifier or TokenVerifier(cfg)

    def authenticate(self, authorization: str | None) -> Principal | None:
        from ...adapters.oidc import InvalidToken, Unauthorized, caller_from_claims

        if not authorization or not authorization.lower().startswith("bearer "):
            return None
        try:
            claims = self.verifier.verify(authorization[7:].strip(), audience=self.cfg.audience)
            caller = caller_from_claims(claims, tenant_id=self.tenant_id, directory=self.directory,
                                        service_clients=self.cfg.service_clients)
        except (InvalidToken, Unauthorized) as exc:
            log.info("bearer token refused: %s", exc)
            telemetry.count("hr.auth.attempts", surface=self.surface, outcome="refused", reason=getattr(exc, "code", "invalid_token"))
            return None
        telemetry.count("hr.auth.attempts", surface=self.surface, outcome="ok", reason=caller.kind)
        return Principal(caller.name, caller.kind, caller.tenant_id)

    def principal(self, authorization: str | None) -> str | None:
        p = self.authenticate(authorization)
        return p.user_name if p else None


class BearerContextBuilder(DefaultServerCallContextBuilder):
    """The SDK's default context (it carries the request headers, which the protocol
    version check reads), with the authenticated principal filled in from the bearer token."""

    def __init__(self, tokens: BearerTokens):
        self.tokens = tokens

    def build_user(self, request: Request) -> User:
        return self.tokens.authenticate(request.headers.get("authorization")) or UnauthenticatedUser()


def require_bearer(app, tokens: BearerTokens, protected_path: str) -> None:
    """Reject unauthenticated calls to the JSON-RPC endpoint with a 401. The agent
    card stays public, as discovery requires."""

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        if request.url.path == protected_path and tokens.authenticate(request.headers.get("authorization")) is None:
            return JSONResponse({"error": "A valid bearer token is required."}, status_code=401, headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)


def request_data(context: RequestContext) -> dict:
    """The first structured (data) part of the incoming message, or {}."""
    message = context.message
    for part in get_data_parts(message.parts) if message else []:
        if isinstance(part, dict):
            return part
    return {}


def text_and_data(text: str, data: dict | None = None):
    parts = [new_text_part(text)]
    if data is not None:
        parts.append(new_data_part(data))
    return parts


async def begin(context: RequestContext, queue: EventQueue) -> TaskUpdater:
    """Open the task (announcing it if this is its first message) and mark it working."""
    if context.current_task is None:
        await queue.enqueue_event(
            Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message],
            )
        )
    updater = TaskUpdater(queue, context.task_id, context.context_id)
    await updater.start_work()
    return updater


def message_text(context: RequestContext) -> str:
    return get_message_text(context.message) if context.message else ""
