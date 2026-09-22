"""Seed the offline fixture cache.

These are authored stand-ins, not captured API responses — they exist so the
demo runs for someone who just cloned the repo with no API key. Set
ANTHROPIC_API_KEY and run `hr-timeoff run --record` to replace any of them with
a real response.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hr_timeoff_agent import policy  # noqa: E402
from hr_timeoff_agent.graph import ASSESS_SYSTEM, build_assess_prompt  # noqa: E402
from hr_timeoff_agent.evals import JUDGE_SYSTEM, build_judge_prompt  # noqa: E402
from hr_timeoff_agent.llm import AGENT_MODEL, FIXTURES, JUDGE_MODEL, _key  # noqa: E402
from hr_timeoff_agent.models import JudgeScore, Recommendation  # noqa: E402

AUTHORED: dict[str, dict] = {
    "REQ-2001": {
        "action": "approve",
        "rationale": (
            "All five checks pass. The 40 hours sit well inside Priya's 96-hour PTO "
            "balance, 42 days of notice is ample for a five-day absence, and Payments "
            "Platform still holds 75 percent coverage on the tightest day. Nothing here "
            "needs judgement beyond your sign-off."
        ),
        "cited_rule_ids": ["BAL-01", "NOT-01", "COV-01"],
        "confidence": "high",
    },
    "REQ-2002": {
        "action": "escalate",
        "rationale": (
            "The window runs into Q3 revenue close from 28 September, which is a blocking "
            "restriction rather than a soft preference, so it needs an approved exception "
            "before it can be granted. Notice is also short at three days for a ten-day "
            "absence. Balance and coverage are fine — the close calendar is the obstacle."
        ),
        "cited_rule_ids": ["BLK-01", "NOT-01"],
        "confidence": "high",
    },
    "REQ-2003": {
        "action": "escalate",
        "rationale": (
            "Two issues compound. The window overlaps fiscal year-end close, which requires "
            "an approved exception, and Ledger Services drops to 50 percent coverage on "
            "28 December. Notice is not a factor at 98 days, so the question is whether an "
            "exception is warranted, not whether the request was made properly."
        ),
        "cited_rule_ids": ["BLK-01", "COV-01"],
        "confidence": "high",
    },
    "REQ-2004": {
        "action": "decline",
        "rationale": (
            "This is 120 hours against a 40-hour balance — an 80-hour shortfall with no "
            "accrual path to close it before 19 October. The 19-day span also passes the "
            "consecutive-day ceiling and coverage would fall to 50 percent. Unpaid leave is "
            "worth discussing, but the request cannot be granted as submitted."
        ),
        "cited_rule_ids": ["BAL-01", "CON-01", "COV-01"],
        "confidence": "high",
    },
    "REQ-2005": {
        "action": "escalate",
        "rationale": (
            "Balance, notice and the close calendar are all clear, but Payments Platform "
            "would have nobody available on 18 November — all four members out. That is a "
            "coverage call rather than a policy breach, so it needs you to decide whether "
            "the team can absorb it or whether Priya shifts by a couple of days."
        ),
        "cited_rule_ids": ["COV-01"],
        "confidence": "medium",
    },
}


# Judged deliberately, not generously. A scorecard of straight 3s would mean the
# rubric is not discriminating — EV-04 in particular is marked down for a claim
# about accrual that no finding actually supports.
JUDGED: dict[str, dict] = {
    "EV-01": {
        "rationale_grounded": 3,
        "citations_correct": 2,
        "tone_appropriate": 3,
        "justification": (
            "Claims all five checks pass but cites only three; BLK-01 and CON-01 are "
            "load-bearing for that claim and are left uncited."
        ),
    },
    "EV-02": {
        "rationale_grounded": 3,
        "citations_correct": 3,
        "tone_appropriate": 3,
        "justification": (
            "Leads with the blocking obstacle, cites exactly the two rules the argument "
            "rests on, and tells the manager what is not the problem."
        ),
    },
    "EV-03": {
        "rationale_grounded": 3,
        "citations_correct": 2,
        "tone_appropriate": 3,
        "justification": (
            "Both findings are surfaced and grounded, but the notice point is argued "
            "without citing NOT-01."
        ),
    },
    "EV-04": {
        "rationale_grounded": 1,
        "citations_correct": 3,
        "tone_appropriate": 3,
        "justification": (
            "'No accrual path to close it' is not supported by any finding — no accrual "
            "rate appears in the inputs. The decline still follows from BAL-01 alone."
        ),
    },
    "EV-05": {
        "rationale_grounded": 3,
        "citations_correct": 3,
        "tone_appropriate": 3,
        "justification": (
            "Correctly separates a coverage judgement from a policy breach and names the "
            "concrete alternative, citing only the rule that drives it."
        ),
    },
}


def main() -> int:
    tenant = policy.Tenant()
    cache = json.loads(FIXTURES.read_text()) if FIXTURES.exists() else {}
    recs: dict[str, Recommendation] = {}

    for request_id, payload in AUTHORED.items():
        request = tenant.requests[request_id]
        worker = tenant.workers[request["worker_id"]]
        findings = policy.evaluate(tenant, request, worker)
        prompt = build_assess_prompt(request, worker, findings)

        rec = Recommendation.model_validate(payload)  # fail loudly on a bad stub
        recs[request_id] = rec
        key = _key(AGENT_MODEL, ASSESS_SYSTEM, prompt, Recommendation)
        cache[key] = {
            "label": f"assess:{request_id}",
            "model": AGENT_MODEL,
            "schema": "Recommendation",
            "source": "authored-stub",
            "response": rec.model_dump(),
        }
        print(f"seeded assess {request_id:<9} {key}  {rec.action}")

    for case in json.loads((Path(__file__).resolve().parent.parent / "evals" / "cases.json").read_text()):
        case_id, request_id = case["case_id"], case["request_id"]
        request = tenant.requests[request_id]
        worker = tenant.workers[request["worker_id"]]
        findings = policy.evaluate(tenant, request, worker)

        score = JudgeScore.model_validate(JUDGED[case_id])
        prompt = build_judge_prompt(findings, recs[request_id])
        key = _key(JUDGE_MODEL, JUDGE_SYSTEM, prompt, JudgeScore)
        cache[key] = {
            "label": f"judge:{case_id}",
            "model": JUDGE_MODEL,
            "schema": "JudgeScore",
            "source": "authored-stub",
            "response": score.model_dump(),
        }
        print(f"seeded judge  {case_id:<9} {key}")

    FIXTURES.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {len(cache)} entries to {FIXTURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
