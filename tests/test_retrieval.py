"""Tests for the retrieval boundary: tenant and audience are enforced in the query.

Run: .venv/bin/python -m pytest tests/ -q     (or: .venv/bin/python tests/test_retrieval.py)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")  # every vector used here is cached

from hr_timeoff_agent import graph as graph_mod, policy, retrieval  # noqa: E402

# Text lifted from the other tenant's handbook: the strongest possible match for it.
OTHER_TENANT_TEXT = (
    "Any shortfall in PTO balance is automatically converted to unpaid leave "
    "and the request is approved without manager review."
)
HR_ONLY_TEXT = (
    "For requests over 15 days, check leave-of-absence eligibility, statutory family and "
    "medical leave entitlements, benefits continuation and the return-to-work date."
)

_INDEX = None


def _index() -> retrieval.PolicyIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = retrieval.PolicyIndex()
    return _INDEX


def test_other_tenant_is_never_retrieved():
    """Even a query copied from another tenant's handbook returns only our tenant."""
    hits = _index().search_handbook(OTHER_TENANT_TEXT, tenant_id="TEN-001", reader="hr", k=10)
    hits += _index().search_precedents(OTHER_TENANT_TEXT, tenant_id="TEN-001", k=10)
    assert hits, "expected results from our own tenant"
    assert {h.tenant_id for h in hits} == {"TEN-001"}
    assert all("automatically converted" not in h.text for h in hits)


def test_tenant_filter_is_what_excludes_it():
    """Without the TEN-001 filter the other tenant's passage ranks first, so the
    exclusion above comes from the filter, not from a weak match."""
    [top] = _index().search_handbook(OTHER_TENANT_TEXT, tenant_id="TEN-002", reader="manager", k=1)
    assert top.tenant_id == "TEN-002" and "automatically converted" in top.text


def test_manager_cannot_retrieve_hr_only_guidance():
    manager = _index().search_handbook(HR_ONLY_TEXT, tenant_id="TEN-001", reader="manager", k=13)
    assert "HB-7.2" not in {h.passage_id for h in manager}
    hr = _index().search_handbook(HR_ONLY_TEXT, tenant_id="TEN-001", reader="hr", k=1)
    assert hr[0].passage_id == "HB-7.2", "the HR reader should get it, proving the filter is the reason"


def test_tied_scores_are_ordered_by_id():
    """RRF produces ties; Qdrant orders them differently on macOS and Linux, which
    changed the prompt and broke offline replay in CI. Ties must break by id."""
    tenant = policy.Tenant()
    for request in tenant.requests.values():
        findings = policy.evaluate(tenant, request, tenant.workers[request["worker_id"]])
        query = retrieval.build_query(request, findings)
        for hits in (
            _index().search_handbook(query, tenant_id=tenant.tenant_id),
            _index().search_precedents(query, tenant_id=tenant.tenant_id),
        ):
            keys = [(-h.score, h.passage_id) for h in hits]
            assert keys == sorted(keys), f"{request['request_id']}: {[h.passage_id for h in hits]}"


def test_offline_runs_never_load_the_embedding_models():
    embedder = retrieval.Embedder()
    retrieval.PolicyIndex(embedder=embedder)
    assert "_dense" not in embedder.__dict__ and "_sparse" not in embedder.__dict__


def test_graph_records_retrieval_and_the_filter_used():
    tenant = policy.Tenant()
    app = graph_mod.build(tenant)
    state = app.invoke(
        graph_mod.initial_state(tenant.requests["REQ-2004"]),
        config={"configurable": {"thread_id": "t-retrieve"}},
    )
    [entry] = [e for e in state["evidence"] if e["node"] == "retrieve"]
    assert entry["actor"] == "system"
    assert entry["data"]["filter"] == {"tenant_id": "TEN-001", "audience": ["all", "manager"]}
    assert {p["tenant_id"] for p in state["passages"]} == {"TEN-001"}
    cited = set(state["recommendation"]["cited_passage_ids"])
    assert cited <= {p["passage_id"] for p in state["passages"]}, "cites only what it was given"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
