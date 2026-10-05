"""A small client for talking to an A2A agent.

`A2AAgent.send` sends one message (structured data, text, or both) and returns
where the task ended up: its state, the agent's message, and any artifacts.
Pass `task_id` and `context_id` back to continue a task that asked for input.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_data_parts, get_message_text, new_data_part, new_text_part
from a2a.types import Message, Role, SendMessageRequest, TaskState

from ...adapters import telemetry
from ...adapters.oidc import BearerAuth


class A2AError(RuntimeError):
    pass


@dataclass
class TaskResult:
    task_id: str
    context_id: str
    state: str                       # "completed", "input-required", "rejected", ...
    text: str = ""                   # the agent's last status message
    data: dict | None = None         # its structured part, if any
    artifacts: dict[str, Any] = field(default_factory=dict)  # name → data (or text)


def _state(value: int) -> str:
    return TaskState.Name(value).removeprefix("TASK_STATE_").lower().replace("_", "-")


def _first_data(parts) -> dict | None:
    return next((d for d in get_data_parts(parts) if isinstance(d, dict)), None)


class A2AAgent:
    def __init__(self, base_url: str, token: "str | Callable[[], str] | None" = None, *, httpx_client: httpx.AsyncClient | None = None):
        """`token` is a bearer token, or a function returning a current one (for
        access tokens that expire: it is called on every request)."""
        self.base_url = base_url.rstrip("/")
        self._http = httpx_client or httpx.AsyncClient(timeout=120)
        if token:
            self._http.auth = BearerAuth(token)
        self._http.event_hooks.setdefault("request", []).append(self._propagate_trace)
        self._client = None
        self.card = None

    @staticmethod
    async def _propagate_trace(request: httpx.Request) -> None:
        """Send the current trace along, so the agent's spans join the caller's trace."""
        telemetry.inject(request.headers)

    async def connect(self) -> "A2AAgent":
        if self._client is None:
            self.card = await A2ACardResolver(self._http, self.base_url).get_agent_card()
            self._client = await create_client(self.card, client_config=ClientConfig(httpx_client=self._http, streaming=False))
        return self

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
        await self._http.aclose()

    async def send(self, data: dict | None = None, text: str | None = None, *, task_id: str | None = None, context_id: str | None = None, message_id: str | None = None) -> TaskResult:
        """Send one message. Resend with the same `message_id` to retry safely: the
        agent treats it as the same request, not a new one."""
        with telemetry.span("a2a.client.send", peer=self.base_url, continuing=bool(task_id)):
            return await self._send(data, text, task_id=task_id, context_id=context_id, message_id=message_id)

    async def _send(self, data, text, *, task_id, context_id, message_id) -> TaskResult:
        await self.connect()
        parts = ([new_text_part(text)] if text else []) + ([new_data_part(data)] if data is not None else [])
        message = Message(
            role=Role.ROLE_USER, message_id=message_id or str(uuid.uuid4()), parts=parts,
            task_id=task_id or "", context_id=context_id or "",
        )
        final = None
        artifacts: dict[str, Any] = {}
        async for event in self._client.send_message(SendMessageRequest(message=message)):
            if event.HasField("task"):
                final = event.task
                for art in event.task.artifacts:
                    artifacts[art.name] = _first_data(art.parts) or get_message_text(Message(parts=art.parts))
            elif event.HasField("status_update"):
                final = final or event.status_update
            elif event.HasField("message"):
                return TaskResult("", "", "completed", get_message_text(event.message), _first_data(event.message.parts))
        if final is None:
            raise A2AError("the agent returned nothing")
        status = final.status
        msg = status.message if status.HasField("message") else None
        return TaskResult(
            task_id=final.id if hasattr(final, "id") else final.task_id,
            context_id=final.context_id,
            state=_state(status.state),
            text=get_message_text(msg) if msg else "",
            data=_first_data(msg.parts) if msg else None,
            artifacts=artifacts,
        )
