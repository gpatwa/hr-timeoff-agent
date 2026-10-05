"""Seed the offline fixture cache with authored stand-ins.

The shipped fixtures are real Opus 5.5 recordings (see README, "Offline mode").
This script is the fallback that existed before them, and it never overwrites a
recorded response: it only fills keys that have no recording.

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

from hr_timeoff_agent import policy, retrieval  # noqa: E402
from hr_timeoff_agent.graph import ASSESS_SYSTEM, build_assess_prompt  # noqa: E402
from hr_timeoff_agent.tooling.evals import JUDGE_SYSTEM, build_judge_prompt  # noqa: E402
from hr_timeoff_agent.llm import AGENT_MODEL, FIXTURES, JUDGE_MODEL, _key  # noqa: E402
from hr_timeoff_agent.models import JudgeScore, Recommendation  # noqa: E402

AUTHORED: dict[str, dict] = {
    "REQ-2001": {
        "action": "approve",
        "rationale": (
            "All five checks pass. The 40 hours sit well inside Priya's 96-hour PTO "
            "balance, 42 days of notice is ample for a five-day absence, and Payments "
            "Platform still holds 75 percent coverage on the tightest day. This is the "
            "same routine case as P-108 and needs nothing beyond your sign-off."
        ),
        "cited_rule_ids": ["BAL-01", "NOT-01", "COV-01"],
        "cited_passage_ids": ["P-108"],
        "confidence": "high",
    },
    "REQ-2002": {
        "action": "escalate",
        "rationale": (
            "The window runs into Q3 revenue close from 28 September, a blocking "
            "restriction, so it needs a close-window exception before it can be granted. "
            "A wedding is the kind of life event HB-4.1 says exceptions are usually granted "
            "for, provided close tasks go to a named backup and the close lead also signs "
            "off (HB-4.2); P-101 was approved on exactly that basis. The three days' notice "
            "can be accepted for a life event (HB-5.1)."
        ),
        "cited_rule_ids": ["BLK-01", "NOT-01"],
        "cited_passage_ids": ["HB-4.1", "HB-4.2", "HB-5.1", "P-101"],
        "confidence": "high",
    },
    "REQ-2003": {
        "action": "escalate",
        "rationale": (
            "Two issues compound. The window overlaps fiscal year-end close, where HB-4.1 "
            "says routine vacation is normally declined or moved, and Ledger Services drops "
            "to 50 percent coverage on 28 December. Notice is not a factor at 98 days; a "
            "similar year-end request (P-102) was returned with a suggestion to move it to "
            "January, which is the likely path here."
        ),
        "cited_rule_ids": ["BLK-01", "COV-01"],
        "cited_passage_ids": ["HB-4.1", "P-102"],
        "confidence": "high",
    },
    "REQ-2004": {
        "action": "decline",
        "rationale": (
            "This is 120 hours against a 40-hour balance, and at 6.67 hours a month (HB-2.1) "
            "accrual cannot close an 80-hour gap before 19 October. HB-3.1 rules out approving "
            "it as fully paid, but you could approve the 40 paid hours and agree the rest as "
            "unpaid leave with the HR Partner, who reviews a 19-day span anyway (HB-7.1). As "
            "submitted, it should be declined, as P-104 was."
        ),
        "cited_rule_ids": ["BAL-01", "CON-01"],
        "cited_passage_ids": ["HB-2.1", "HB-3.1", "HB-7.1", "P-104"],
        "confidence": "high",
    },
    "REQ-2005": {
        "action": "escalate",
        "rationale": (
            "Balance, notice and the close calendar are all clear, but Payments Platform "
            "would have nobody available on 18 November — all four members out. HB-6.1 says "
            "to arrange on-call cover from an adjacent team before approving, or ask Priya to "
            "move the dates; both have worked before (P-105, P-106)."
        ),
        "cited_rule_ids": ["COV-01"],
        "cited_passage_ids": ["HB-6.1", "P-105", "P-106"],
        "confidence": "medium",
    },
}

# Judged deliberately, not generously. A scorecard of straight 3s would mean the
# rubric is not discriminating.
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
            "Leads with the blocking obstacle, then gives the manager the exception route "
            "and a matching precedent, citing exactly the rules and passages it uses."
        ),
    },
    "EV-03": {
        "rationale_grounded": 3,
        "citations_correct": 2,
        "tone_appropriate": 3,
        "justification": (
            "Both findings and the precedent are grounded, but the notice point is argued "
            "without citing NOT-01."
        ),
    },
    "EV-04": {
        "rationale_grounded": 3,
        "citations_correct": 3,
        "tone_appropriate": 2,
        "justification": (
            "The accrual claim is now supported by HB-2.1 and every remedy is cited, but "
            "three options in one long sentence bury the recommendation for a busy manager."
        ),
    },
    "EV-05": {
        "rationale_grounded": 3,
        "citations_correct": 3,
        "tone_appropriate": 3,
        "justification": (
            "Separates a coverage judgement from a policy breach and names two concrete "
            "remedies with the guidance and precedents behind them."
        ),
    },
}


def main() -> int:
    tenant = policy.Tenant()
    index = retrieval.PolicyIndex()
    cache = json.loads(FIXTURES.read_text()) if FIXTURES.exists() else {}
    recs: dict[str, Recommendation] = {}
    passages: dict[str, list] = {}
    recorded: set[str] = set()

    def retrieve(request, findings):
        query = retrieval.build_query(request, findings)
        return index.search_handbook(query, tenant_id=tenant.tenant_id, reader="manager") + (
            index.search_precedents(query, tenant_id=tenant.tenant_id)
        )

    for request_id, payload in AUTHORED.items():
        request = tenant.requests[request_id]
        worker = tenant.workers[request["worker_id"]]
        findings = policy.evaluate(tenant, request, worker)
        passages[request_id] = retrieve(request, findings)
        prompt = build_assess_prompt(request, worker, findings, passages[request_id])

        rec = Recommendation.model_validate(payload)  # fail loudly on a bad stub
        retrieved = {p.passage_id for p in passages[request_id]}
        uncited = set(rec.cited_passage_ids) - retrieved
        assert not uncited, f"{request_id} cites passages it was never given: {uncited}"
        recs[request_id] = rec
        key = _key(AGENT_MODEL, ASSESS_SYSTEM, prompt, Recommendation)
        if cache.get(key, {}).get("source", "authored-stub") != "authored-stub":
            print(f"kept recorded assess {request_id:<9} {key}")
            recorded.add(request_id)
            continue
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

        if request_id in recorded:  # never author a judgment of a real response
            continue
        score = JudgeScore.model_validate(JUDGED[case_id])
        prompt = build_judge_prompt(request, worker, findings, passages[request_id], recs[request_id])
        key = _key(JUDGE_MODEL, JUDGE_SYSTEM, prompt, JudgeScore)
        if cache.get(key, {}).get("source", "authored-stub") != "authored-stub":
            print(f"kept recorded judge  {case_id:<9} {key}")
            continue
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
