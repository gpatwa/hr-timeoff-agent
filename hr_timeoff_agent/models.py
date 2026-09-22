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


class Recommendation(BaseModel):
    """What the agent is allowed to produce. Advisory only."""

    action: Action
    rationale: str
    cited_rule_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]


class Decision(BaseModel):
    """What only a human can produce.

    `actor_type` is pinned to "human" at the type level, and the record node
    re-asserts it at runtime. An agent cannot construct a valid Decision.
    """

    outcome: Outcome
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


class CaseResult(BaseModel):
    case_id: str
    expected_action: Action
    actual_action: Optional[Action] = None
    action_match: bool = False
    never_self_approved: bool = True
    scores: Optional[JudgeScore] = None
    error: Optional[str] = None
