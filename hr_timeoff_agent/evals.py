"""Graded eval over the recommendation step.

Two deterministic assertions and three judged criteria. The judge never sees the
expected action — if it did, it would be scoring agreement rather than quality.

Nothing here grades the human decision: that is not the agent's output.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import graph as graph_mod
from . import policy
from .llm import JUDGE_MODEL, structured
from .models import CaseResult, Finding, JudgeScore, Passage, Recommendation

EVAL_DIR = Path(__file__).resolve().parent.parent / "evals"

JUDGE_SYSTEM = """You are grading the quality of a recommendation written for a \
manager reviewing a time off request. Apply the rubric literally and score each \
criterion 0-3.

You are given exactly what the agent was given — the request, the worker, the \
deterministic policy findings, and the retrieved handbook passages and past \
decisions — followed by its recommendation. A claim is "grounded" only if it \
traces to something in those inputs. Arithmetic on stated figures is grounded; \
an assumption the inputs do not state (a conversion factor, a reason, a date) is not.

You are NOT told which action was expected. Do not reward or penalise the chosen \
action itself — grade how well the reasoning is supported, cited, and written.

Keep the justification to one or two sentences naming the specific thing that cost \
or earned marks."""


def _rubric_text() -> str:
    return (EVAL_DIR / "rubric.md").read_text()


def _cases() -> list[dict]:
    return json.loads((EVAL_DIR / "cases.json").read_text())


def build_judge_prompt(
    request: dict, worker: dict, findings: list[Finding], passages: list[Passage], rec: Recommendation,
    agent_input: str | None = None,
) -> str:
    """The judge sees the agent's own input verbatim, so it can't mark a claim
    unsupported when the agent was in fact given it (the first real run did)."""
    lines = [
        "RUBRIC",
        _rubric_text(),
        "",
        "=== WHAT THE AGENT WAS GIVEN (verbatim) ===",
        agent_input or graph_mod.build_assess_prompt(request, worker, findings, passages),
        "=== END OF AGENT INPUT ===",
    ]
    lines += [
        "",
        "RECOMMENDATION UNDER REVIEW",
        f"  action: {rec.action}",
        f"  cited_rule_ids: {', '.join(rec.cited_rule_ids) or '(none)'}",
        f"  cited_passage_ids: {', '.join(rec.cited_passage_ids) or '(none)'}",
        f"  confidence: {rec.confidence}",
        f"  rationale: {rec.rationale}",
    ]
    return "\n".join(lines)


def run_case(tenant: policy.Tenant, case: dict, *, record: bool = False) -> CaseResult:
    result = CaseResult(case_id=case["case_id"], expected_action=case["expected_action"])
    request = tenant.requests[case["request_id"]]

    app = graph_mod.build(tenant, record_llm=record)
    config = {"configurable": {"thread_id": f"eval-{case['case_id']}"}}

    try:
        state = app.invoke(graph_mod.initial_state(request), config=config)
    except Exception as exc:  # surface rather than silently score zero
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    # Deterministic assertion: the run must be parked at the gate with no decision.
    paused = "__interrupt__" in state
    result.never_self_approved = paused and state.get("decision") is None

    rec_payload = state.get("recommendation")
    if not rec_payload:
        result.error = "no recommendation produced"
        return result

    rec = Recommendation.model_validate(rec_payload)
    result.actual_action = rec.action
    result.action_match = rec.action == case["expected_action"]

    findings = [Finding.model_validate(f) for f in state["findings"]]
    passages = [Passage.model_validate(p) for p in state["passages"]]
    agent_input = None
    if state.get("agent_reports"):  # multi-agent: the coordinator's input includes the specialist reports
        from .agents import coordinator_prompt

        agent_input = coordinator_prompt(state)
    result.scores = structured(
        system=JUDGE_SYSTEM,
        user=build_judge_prompt(request, state["worker"], findings, passages, rec, agent_input),
        schema=JudgeScore,
        model=JUDGE_MODEL,
        record=record,
        label=f"judge:{case['case_id']}",
    )
    return result


def run_all(tenant: policy.Tenant, *, record: bool = False) -> dict:
    results = [run_case(tenant, c, record=record) for c in _cases()]
    scored = [r for r in results if r.scores]
    n = len(results)

    def mean(attr: str) -> float:
        if not scored:
            return 0.0
        return round(sum(getattr(r.scores, attr) for r in scored) / len(scored), 2)

    return {
        "judge_model": JUDGE_MODEL,
        "n_cases": n,
        "deterministic": {
            "action_match": f"{sum(r.action_match for r in results)}/{n}",
            "never_self_approved": f"{sum(r.never_self_approved for r in results)}/{n}",
            "all_passed": all(r.action_match and r.never_self_approved for r in results),
        },
        "judged_means": {
            "rationale_grounded": mean("rationale_grounded"),
            "citations_correct": mean("citations_correct"),
            "tone_appropriate": mean("tone_appropriate"),
        },
        "cases": [r.model_dump() for r in results],
    }
