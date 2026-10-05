"""RAGAS evaluation of the retrieval step.

The graded eval (evals.py) asks whether the recommendation is right and well
argued. This one asks whether retrieval did its job, using the RAGAS library:

  retrieval (no model, runs offline)
    id_context_precision  share of retrieved passages that were in the reference set
    id_context_recall     share of reference passages that were retrieved

  generation (needs ANTHROPIC_API_KEY; an LLM grades)
    faithfulness          share of claims in the rationale supported by what the
                          agent was given (findings + retrieved passages)
    context_recall        share of the reference answer's statements that the
                          retrieved passages support

Reference passages and answers in evals/rag_cases.json were written before
retrieval was tuned. Every metric is reported on its own; there is no blended
score, for the same reason as the graded eval.

The grading model defaults to a different model from the agent, so nothing
grades its own output.
"""

from __future__ import annotations

import asyncio
import json
import os
import warnings
from pathlib import Path

from .. import assembly, graph as graph_mod
from .. import policy, retrieval
from ..llm import AGENT_MODEL, is_offline
from ..models import Finding, Passage

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "evals"

RAGAS_MODEL = os.environ.get("HR_AGENT_RAGAS_MODEL", "claude-sonnet-5")

RETRIEVAL_METRICS = ("id_context_precision", "id_context_recall")
GENERATION_METRICS = ("faithfulness", "context_recall")


def _cases() -> list[dict]:
    return json.loads((EVAL_DIR / "rag_cases.json").read_text())


def _ragas():
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "The RAG eval needs the ragas extra:\n"
            "  ./.venv/bin/pip install -e '.[rag-eval]'"
        ) from exc


def _retrieval_metrics():
    # ragas 0.4 only ships the ID-based metrics on the legacy import path.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

    return {
        "id_context_precision": IDBasedContextPrecision(),
        "id_context_recall": IDBasedContextRecall(),
    }


def _generation_metrics():
    """LLM-graded metrics, with Claude as the grader.

    ragas passes its default sampling settings straight through to Anthropic,
    and current Claude models reject them, so they are removed. Thinking is off
    so the structured-output tool call can be forced.
    """
    from anthropic import AsyncAnthropic
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextRecall, Faithfulness

    llm = llm_factory(RAGAS_MODEL, provider="anthropic", client=AsyncAnthropic(), max_tokens=4096)
    for key in ("temperature", "top_p"):
        llm.model_args.pop(key, None)
    llm.model_args["thinking"] = {"type": "disabled"}
    return {"faithfulness": Faithfulness(llm=llm), "context_recall": ContextRecall(llm=llm)}


def _agent_context(findings: list[Finding], passages: list[Passage]) -> list[str]:
    return [f"[{f.rule_id}] {f.rule_name}: {f.detail}" for f in findings] + [
        f"[{p.passage_id}] {p.title}. {p.text}" for p in passages
    ]


async def _score_case(tenant: policy.Tenant, case: dict, generation: dict | None) -> dict:
    from ragas import SingleTurnSample

    request = tenant.requests[case["request_id"]]
    app = assembly.build(tenant)
    state = app.invoke(
        graph_mod.initial_state(request),
        config={"configurable": {"thread_id": f"rag-{case['case_id']}"}},
    )
    findings = [Finding.model_validate(f) for f in state["findings"]]
    passages = [Passage.model_validate(p) for p in state["passages"]]
    rationale = state["recommendation"]["rationale"]
    query = retrieval.build_query(request, findings)

    result = {
        "case_id": case["case_id"],
        "retrieved": [p.passage_id for p in passages],
        "reference": case["reference_context_ids"],
        "scores": {},
        "errors": {},
    }

    sample = SingleTurnSample(
        retrieved_context_ids=result["retrieved"],
        reference_context_ids=case["reference_context_ids"],
    )
    for name, metric in _retrieval_metrics().items():
        result["scores"][name] = round(float(await metric.single_turn_ascore(sample)), 3)

    if generation:
        calls = {
            "faithfulness": lambda m: m.ascore(
                user_input=query, response=rationale, retrieved_contexts=_agent_context(findings, passages)
            ),
            "context_recall": lambda m: m.ascore(
                user_input=query,
                retrieved_contexts=[f"{p.title}. {p.text}" for p in passages],
                reference=case["reference"],
            ),
        }
        for name, metric in generation.items():
            try:
                result["scores"][name] = round(float((await calls[name](metric)).value), 3)
            except Exception as exc:  # report, don't silently score zero
                result["errors"][name] = f"{type(exc).__name__}: {exc}"
    return result


def run_all(tenant: policy.Tenant) -> dict:
    _ragas()
    # The graded metrics call the Anthropic API through ragas, so they need a key
    # even when the agent itself is recorded through Claude Code.
    can_grade = not is_offline() and bool(os.environ.get("ANTHROPIC_API_KEY"))
    generation = _generation_metrics() if can_grade else None

    async def run():
        return [await _score_case(tenant, c, generation) for c in _cases()]

    cases = asyncio.run(run())
    metrics = RETRIEVAL_METRICS + (GENERATION_METRICS if generation else ())

    def mean(name: str) -> float | None:
        vals = [c["scores"][name] for c in cases if name in c["scores"]]
        return round(sum(vals) / len(vals), 3) if vals else None

    import ragas

    return {
        "ragas_version": ragas.__version__,
        "agent_model": AGENT_MODEL,
        "grading_model": RAGAS_MODEL if generation else None,
        "generation_metrics": "scored" if generation else "skipped (needs ANTHROPIC_API_KEY)",
        "k": {"handbook": retrieval.HANDBOOK_K, "precedents": retrieval.PRECEDENT_K},
        "means": {m: mean(m) for m in metrics},
        "cases": cases,
    }
