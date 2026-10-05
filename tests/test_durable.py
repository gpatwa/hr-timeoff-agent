"""The durable stack: state lives in Postgres (and Qdrant), not in the process.

Needs services, so it only runs when they are configured:

    docker compose -f docker-compose.dev.yml up -d
    HR_DATABASE_URL=postgresql://hr:hr@localhost:5433/hr HR_QDRANT_URL=http://localhost:6334 \\
        .venv/bin/python tests/test_durable.py

Each test gets its own schema (HR_DATABASE_SCHEMA=auto: one per home directory),
so tests do not see each other. The other test files run unchanged against this
stack too; this one covers what only a durable store promises.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")
logging.getLogger("a2a").setLevel(logging.ERROR)

if not os.environ.get("HR_DATABASE_URL"):
    print("SKIPPED: set HR_DATABASE_URL (and HR_QDRANT_URL) to run the durable-stack tests.")
    raise SystemExit(0)
os.environ["HR_DATABASE_SCHEMA"] = "auto"

import httpx  # noqa: E402

from hr_timeoff_agent import graph as graph_mod, llm  # noqa: E402
from hr_timeoff_agent.a2a_client import A2AAgent  # noqa: E402
from hr_timeoff_agent.a2a_common import BearerTokens  # noqa: E402
from hr_timeoff_agent.a2a_server import create_timeoff_app  # noqa: E402
from hr_timeoff_agent.models import Recommendation  # noqa: E402
from hr_timeoff_agent.storage import PostgresStore  # noqa: E402
from hr_timeoff_agent.workspace import Workspace  # noqa: E402

PRIYA, AIKO, SAMUEL = "W-100234", "W-100236", "W-100237"
ALL_PASS = {"start": "2026-11-30", "end": "2026-12-02", "hours": "", "note": "Family trip, booked months ago.", "plan": "PTO"}
REVIEW = {"skill": "review_time_off_request", "request_id": "REQ-2004"}


def _stub(*, system, user, schema, model=llm.AGENT_MODEL, record=False, label=""):
    return Recommendation(action="approve", rationale="Stub.", cited_rule_ids=["BAL-01"],
                          cited_passage_ids=[], confidence="high")


def test_state_is_in_postgres_not_in_the_home_directory():
    home = tempfile.mkdtemp()
    ws = Workspace(home)
    assert isinstance(ws.store, PostgresStore)
    assert not (Path(home) / "tenant").exists(), "documents must not be written to local files"
    assert not (Path(home) / "app.sqlite").exists() and not (Path(home) / "checkpoints.sqlite").exists()
    ws.close()


def test_a_pending_approval_and_the_decision_survive_restarts():
    home = tempfile.mkdtemp()
    ws = Workspace(home)
    ws.close()
    ws = Workspace(home)  # a new process: nothing re-seeded, nothing lost
    manager = ws.persona(AIKO)
    d = ws.detail(manager, "REQ-2004")
    assert d["pending"] and d["can_decide"] and d["chain_ok"]
    ws.decide(manager, "REQ-2004", "approved", "40 paid, 80 unpaid")
    ws.close()

    ws = Workspace(home)  # and again, after the decision
    assert ws.request("REQ-2004")["status"] == "approved"
    assert ws.worker(SAMUEL)["time_off_plans"]["PTO"]["balance_hours"] == 0.0
    assert any(a["absence_id"] == "ABS-REQ-2004" for a in ws._read("absences.json"))
    d = ws.detail(ws.persona(AIKO), "REQ-2004")
    assert d["decision"]["outcome"] == "approved" and d["chain_ok"]
    ws.close()


def test_two_processes_share_one_state():
    home = tempfile.mkdtemp()
    a, b = Workspace(home), Workspace(home)
    a.decide(a.persona(AIKO), "REQ-2004", "returned", "")
    assert b.request("REQ-2004")["status"] == "returned", "the second process must see the first one's write"
    a.close()
    b.close()


def test_concurrent_submissions_get_distinct_ids_across_processes():
    graph_mod.structured = _stub
    home = tempfile.mkdtemp()
    spaces = [Workspace(home), Workspace(home)]
    priya = spaces[0].persona(PRIYA)
    ids, errors = [], []

    def submit(i):
        try:
            ids.append(spaces[i % 2].submit(priya, **ALL_PASS))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    assert len(set(ids)) == 8, f"duplicate request ids: {sorted(ids)}"
    assert len({r["request_id"] for r in spaces[1].requests()}) == len(spaces[1].requests())
    [s.close() for s in spaces]


def test_reset_restores_the_seed_in_the_database():
    home = tempfile.mkdtemp()
    ws = Workspace(home)
    ws.decide(ws.persona(AIKO), "REQ-2004", "approved", "")
    ws.reset()
    assert ws.request("REQ-2004")["status"] == "pending"
    assert ws.worker(SAMUEL)["time_off_plans"]["PTO"]["balance_hours"] == 40.0
    ws.close()


def test_an_a2a_task_survives_an_agent_restart():
    home = tempfile.mkdtemp()
    ws = Workspace(home)
    tokens = BearerTokens.derive("durable-secret", [p.worker_id for p in ws.personas()] + ["timeoff-agent"])

    def client(app, who):
        return A2AAgent("http://timeoff", tokens.for_principal(who),
                        httpx_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://timeoff"))

    async def go():
        first = create_timeoff_app(ws, tokens, base_url="http://timeoff")
        waiting = await client(first, AIKO).send(REVIEW)
        assert waiting.state == "input-required"
        # The agent process restarts: a new app and a new task store, same database.
        second = create_timeoff_app(ws, tokens, base_url="http://timeoff")
        done = await client(second, AIKO).send({"outcome": "approved", "note": "ok"}, task_id=waiting.task_id, context_id=waiting.context_id)
        assert done.state == "completed", done.state
        # Tasks are still owner-scoped after the restart.
        other = client(create_timeoff_app(ws, tokens, base_url="http://timeoff"), "W-100003")
        try:
            await other.send({"outcome": "approved"}, task_id=waiting.task_id, context_id=waiting.context_id)
        except Exception as exc:  # noqa: BLE001
            assert "not found" in str(exc).lower()
        else:
            raise AssertionError("another caller continued the task")

    asyncio.run(go())
    assert ws.request("REQ-2004")["status"] == "approved"
    ws.close()


def test_loading_the_qdrant_index_twice_does_not_duplicate_it():
    url = os.environ.get("HR_QDRANT_URL")
    if not url:
        print("    (HR_QDRANT_URL not set: Qdrant check skipped)")
        return
    from hr_timeoff_agent import retrieval

    first = retrieval.PolicyIndex(qdrant_url=url)
    before = first.client.count("handbook").count, first.client.count("precedents").count
    second = retrieval.PolicyIndex(qdrant_url=url)
    after = second.client.count("handbook").count, second.client.count("precedents").count
    assert before == after and before[0] > 0, (before, after)
    # The server-side index enforces tenant and audience inside the query, as in-process.
    text = ("For requests over 15 days, check leave-of-absence eligibility, statutory family and "
            "medical leave entitlements, benefits continuation and the return-to-work date.")
    hr_ids = {p["passage_id"] for p in second.handbook if p["audience"] == "hr" and p["tenant_id"] == "TEN-001"}
    manager = second.search_handbook(text, tenant_id="TEN-001", reader="manager", k=13)
    assert manager and all(p.tenant_id == "TEN-001" and p.passage_id not in hr_ids for p in manager)
    assert second.search_handbook(text, tenant_id="TEN-001", reader="hr", k=1)[0].passage_id in hr_ids


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        saved = graph_mod.structured
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        finally:
            graph_mod.structured = saved
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
