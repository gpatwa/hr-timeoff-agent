"""Typed contracts shared across the graph.

The separation that matters here is Recommendation vs Decision. The agent can
only ever produce the former; only a human can produce the latter.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Action = Literal["approve", "decline", "escalate"]
Outcome = Literal["approved", "declined", "returned"]
FindingStatus = Literal["pass", "fail", "warn"]
Severity = Literal["blocking", "advisory"]


class Finding(BaseModel):
    """Result of one deterministic policy rule. Never produced by a model."""

    rule_id: str
    rule_name: str
    status: FindingStatus
    severity: Severity
    detail: str
    evidence: dict = Field(default_factory=dict)


class Passage(BaseModel):
    """A retrieved handbook passage or past decision. Guidance, not a rule outcome."""

    passage_id: str
    tenant_id: str
    kind: Literal["handbook", "precedent"]
    title: str
    text: str
    score: float


class Recommendation(BaseModel):
    """What the agent is allowed to produce. Advisory only."""

    action: Action
    rationale: str
    cited_rule_ids: list[str] = Field(default_factory=list)
    cited_passage_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]


class Decision(BaseModel):
    """What only a human can produce.

    `actor_type` is pinned to "human" at the type level, and the record node
    re-asserts it at runtime. An agent cannot construct a valid Decision.

    `decided_by_id` is the approver's worker id; `decided_by` is their name as
    the directory has it, never free text typed by whoever resumed the graph.
    """

    outcome: Outcome
    decided_by_id: str
    decided_by: str
    actor_type: Literal["human"] = "human"
    note: str = ""
    at: str


class JudgeScore(BaseModel):
    """Output of the LLM-as-judge for one eval case."""

    rationale_grounded: int = Field(ge=0, le=3)
    citations_correct: int = Field(ge=0, le=3)
    tone_appropriate: int = Field(ge=0, le=3)
    justification: str


class InjectionVerdict(BaseModel):
    """Output of the compliance judge for one attack case."""

    complied: bool
    how: str


class CaseResult(BaseModel):
    case_id: str
    expected_action: Action
    category: str = "original"
    acceptable_actions: list[str] = []
    actual_action: Optional[Action] = None
    action_match: bool = False
    never_self_approved: bool = True
    # Label-free invariants: they need no expected action, so they cannot be mislabelled.
    no_approve_on_blocking: bool = True
    cites_failures: bool = True
    # Attack cases only.
    attack: bool = False
    forbidden_action: bool = False
    in_scope_tools: bool = True
    out_of_scope: list[str] = []
    verdict: Optional[InjectionVerdict] = None
    injection_resisted: Optional[bool] = None
    tool_calls: int = 0
    scores: Optional[JudgeScore] = None
    error: Optional[str] = None
