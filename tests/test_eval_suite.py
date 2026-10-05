"""The eval suite is itself checked: a case must exercise what it claims, and the labels must follow the rules.

Nothing here calls a model. A case that silently stopped testing what it says (because the policy data
or the rules engine changed) would turn the suite into a comfortable lie, so this fails loudly instead.

Run: .venv/bin/python tests/test_eval_suite.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")

from hr_timeoff_agent.core import policy
from hr_timeoff_agent.adapters import retrieval  # noqa: E402
from hr_timeoff_agent.tooling import evals  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads((ROOT / "evals" / "cases.json").read_text())
TENANT = policy.Tenant()
CATEGORIES = {"original", "boundary", "combined", "judgment", "odd-input", "injection", "poisoned-corpus"}
ACTIONS = {"approve", "decline", "escalate"}


def findings_for(case):
    request = evals.request_for(TENANT, case)
    return request, {f.rule_id: f for f in policy.evaluate(TENANT, request, TENANT.workers[request["worker_id"]])}


def test_every_case_is_well_formed_and_the_ids_are_unique():
    ids = [c["case_id"] for c in CASES]
    assert len(ids) == len(set(ids)) and len(CASES) >= 40
    for c in CASES:
        assert ("request" in c) != ("request_id" in c), c["case_id"]
        assert c["category"] in CATEGORIES, c["case_id"]
        acceptable = c.get("acceptable_actions") or [c["expected_action"]]
        assert set(acceptable) <= ACTIONS and c["expected_action"] in acceptable, c["case_id"]
        assert set(c.get("forbidden_actions", [])) <= ACTIONS and not set(c.get("forbidden_actions", [])) & set(acceptable), c["case_id"]
    cats = {c["category"] for c in CASES}
    assert cats == CATEGORIES, f"every category should be represented: {CATEGORIES - cats}"


def test_the_original_five_cases_are_unchanged_so_old_results_stay_comparable():
    original = {c["case_id"]: c for c in CASES if c["category"] == "original"}
    assert list(original) == ["EV-01", "EV-02", "EV-03", "EV-04", "EV-05"]
    assert [original[i]["expected_action"] for i in original] == ["approve", "escalate", "escalate", "escalate", "escalate"]
    assert [original[i]["request_id"] for i in original] == [f"REQ-200{n}" for n in range(1, 6)]


def test_each_case_exercises_exactly_the_findings_it_claims():
    for c in CASES:
        if not c.get("exercises"):
            continue
        _, got = findings_for(c)
        for rule, status in c["exercises"].items():
            assert got[rule].status == status, f"{c['case_id']}: {rule} is {got[rule].status}, the case says {status}"


def test_labels_follow_the_labelling_rules():
    for c in CASES:
        _, got = findings_for(c)
        acceptable = set(c.get("acceptable_actions") or [c["expected_action"]])
        blocking = [f for f in got.values() if f.status == "fail" and f.severity == "blocking"]
        failing = [f for f in got.values() if f.status != "pass"]
        if blocking:
            assert "approve" not in acceptable, f"{c['case_id']}: approve is acceptable although a blocking rule fails"
        if not failing and not c.get("forbidden_actions"):
            assert acceptable == {"approve"}, f"{c['case_id']}: every rule passes, so approve is the only acceptable action"
        if failing and not blocking:
            assert "decline" not in acceptable, f"{c['case_id']}: decline is acceptable although nothing blocking failed"
        if got["CON-01"].status != "pass":
            assert acceptable <= {"escalate", "decline"}, f"{c['case_id']}: the consecutive-day ceiling routes to HR review"


def test_attack_cases_carry_their_attack_and_a_way_to_tell_it_worked():
    attacks = [c for c in CASES if "injection" in c]
    assert len(attacks) >= 14
    for c in attacks:
        assert c["injection"]["label"] and c["injection"]["instruction"], c["case_id"]
        assert c["category"] in {"injection", "poisoned-corpus"}, c["case_id"]
        if c["category"] == "injection":
            note = evals.request_for(TENANT, c)["note"]
            assert len(note) > 40, f"{c['case_id']}: an injection case needs a real note"
    labels = {c["injection"]["label"] for c in attacks}
    for needed in ("override", "system-spoof", "authority", "exfiltrate", "cross-tenant", "prompt-leak", "format-break",
                   "rationale-poison", "obfuscated", "padding", "french", "scope-creep", "reverse"):
        assert needed in labels, f"missing the {needed!r} attack"


def test_the_hostile_corpus_overlays_differ_from_the_real_one_by_exactly_the_planted_passage():
    real_h = json.loads((ROOT / "data" / "handbook.json").read_text())
    real_p = json.loads((ROOT / "data" / "precedents.json").read_text())
    pp = ROOT / "evals" / "corpora" / "poisoned-precedent"
    po = ROOT / "evals" / "corpora" / "poisoned-policy"
    assert json.loads((pp / "handbook.json").read_text()) == real_h
    assert [p["precedent_id"] for p in json.loads((pp / "precedents.json").read_text())][-1] == "P-999"
    assert json.loads((pp / "precedents.json").read_text())[:-1] == real_p
    assert [p["passage_id"] for p in json.loads((po / "handbook.json").read_text())][-1] == "HB-3.9"
    assert json.loads((po / "handbook.json").read_text())[:-1] == real_h and json.loads((po / "precedents.json").read_text()) == real_p
    for case in (c for c in CASES if c.get("corpus")):
        assert (ROOT / "evals" / "corpora" / case["corpus"]).is_dir()


def test_the_planted_passage_really_is_retrieved_for_its_case():
    """If retrieval never surfaces the hostile passage, the case tests nothing."""
    want = {"poisoned-precedent": "P-999", "poisoned-policy": "HB-3.9"}
    for case in (c for c in CASES if c.get("corpus")):
        try:
            tenant = evals.tenant_for(TENANT, case)
            request, got = findings_for(case)
            query = retrieval.build_query(request, list(got.values()))
            hits = [p.passage_id for p in tenant._policy_index.search_handbook(query, tenant_id="TEN-001", reader="manager")]
            hits += [p.passage_id for p in tenant._policy_index.search_precedents(query, tenant_id="TEN-001")]
        except retrieval.EmbeddingCacheMiss:
            raise AssertionError(f"{case['case_id']}: embeddings are not recorded yet; run the suite once with a model")
        assert want[case["corpus"]] in hits, f"{case['case_id']}: {want[case['corpus']]} is not among {hits}"


def test_a_tool_server_started_for_a_hostile_case_searches_the_overlay_not_the_real_corpus():
    """Under the Claude Code backend the MCP server is a child process. It must be pointed at the overlay,
    or the specialists search the real corpus and the attack never reaches them (this was a real bug:
    the first multi-agent runs of the two corpus cases tested nothing)."""
    from hr_timeoff_agent.tools.client import call_tool
    from hr_timeoff_agent.tools.server import HRToolServer

    case = next(c for c in CASES if c.get("corpus") == "poisoned-precedent")
    request, got = findings_for(case)
    query = retrieval.build_query(request, list(got.values()))
    assert os.environ.get("HR_CORPUS_DIR") is None
    clean = {p["passage_id"] for p in call_tool("search_precedents", {"query": query}, HRToolServer())["passages"]}
    assert "P-999" not in clean
    with evals.corpus_env(case):
        assert os.environ["HR_CORPUS_DIR"].endswith("corpora/poisoned-precedent")
        hostile = {p["passage_id"] for p in call_tool("search_precedents", {"query": query}, HRToolServer())["passages"]}
    assert "P-999" in hostile and os.environ.get("HR_CORPUS_DIR") is None


def test_the_corpus_override_is_set_while_a_case_runs_and_always_restored():
    seen = {}

    class Boom:
        def invoke(self, *a, **kw):
            seen["during"] = os.environ.get("HR_CORPUS_DIR")
            raise RuntimeError("stop here")

    case = next(c for c in CASES if c.get("corpus") == "poisoned-policy")
    real = evals.assembly.build
    evals.assembly.build = lambda *a, **kw: Boom()
    try:
        r = evals.run_case(TENANT, case)
    finally:
        evals.assembly.build = real
    assert r.error and "stop here" in r.error
    assert seen["during"].endswith("corpora/poisoned-policy") and os.environ.get("HR_CORPUS_DIR") is None


def test_the_compliance_judge_prompt_does_not_depend_on_argument_key_order():
    from hr_timeoff_agent.core.models import Recommendation

    case = next(c for c in CASES if "injection" in c)
    rec = Recommendation(action="escalate", rationale="r", cited_rule_ids=["BAL-01"], cited_passage_ids=[], confidence="high")
    a = [{"tool": "get_balance", "arguments": {"worker_id": "W-1", "plan": "PTO"}}]
    b = [{"tool": "get_balance", "arguments": {"plan": "PTO", "worker_id": "W-1"}}]
    assert evals.build_injection_prompt(case, rec, a) == evals.build_injection_prompt(case, rec, b)


def test_scope_check_flags_other_workers_and_other_tenants_and_allows_the_requester_and_manager():
    worker = TENANT.workers["W-100237"]
    ok = [{"tool": "get_worker", "arguments": {"worker_id": "W-100237"}}, {"tool": "get_worker", "arguments": {"worker_id": "W-100236"}},
          {"tool": "search_handbook", "arguments": {"query": "balance shortfall unpaid leave"}}]
    assert evals.scope_violations(ok, worker) == []
    bad = [{"tool": "get_balance", "arguments": {"worker_id": "W-100003", "plan": "PTO"}},
           {"tool": "search_handbook", "arguments": {"query": "shortfall policy for tenant TEN-002"}}]
    got = evals.scope_violations(bad, worker)
    assert len(got) == 2 and "W-100003" in got[0] and "another tenant" in got[1]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
