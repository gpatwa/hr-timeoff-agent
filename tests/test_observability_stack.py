"""Telemetry through the real pipeline: OTLP -> collector -> Jaeger and Prometheus.

    docker compose -f docker-compose.dev.yml up -d otel-collector jaeger prometheus
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 HR_JAEGER_URL=http://localhost:16686 \\
    HR_PROMETHEUS_URL=http://localhost:9090 .venv/bin/python tests/test_observability_stack.py

One trace should cover a filing and its triage and an A2A review that calls the payroll
agent, and the numbers it implies should be queryable in Prometheus, with the alert rules loaded.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")
logging.getLogger("a2a").setLevel(logging.ERROR)

if not (os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") and os.environ.get("HR_JAEGER_URL") and os.environ.get("HR_PROMETHEUS_URL")):
    print("SKIPPED: set OTEL_EXPORTER_OTLP_ENDPOINT, HR_JAEGER_URL and HR_PROMETHEUS_URL to run against the stack.")
    raise SystemExit(0)
os.environ["HR_OTEL_METRIC_INTERVAL_MS"] = "2000"

import httpx  # noqa: E402

from hr_timeoff_agent import graph as graph_mod, llm, telemetry  # noqa: E402
from hr_timeoff_agent.models import Recommendation  # noqa: E402
from hr_timeoff_agent.workspace import Workspace  # noqa: E402

JAEGER, PROM = os.environ["HR_JAEGER_URL"].rstrip("/"), os.environ["HR_PROMETHEUS_URL"].rstrip("/")
SERVICE = f"hr-smoke-{uuid.uuid4().hex[:8]}"
ALL_PASS = {"start": "2026-11-30", "end": "2026-12-02", "hours": "", "note": "Family trip, booked months ago.", "plan": "PTO"}


def poll(fn, what: str, timeout: float = 90.0):
    deadline, last = time.time() + timeout, None
    while time.time() < deadline:
        try:
            last = fn()
            if last:
                return last
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(2)
    raise AssertionError(f"timed out waiting for {what} (last: {last!r})")


def run_workload() -> str:
    from test_a2a import AIKO, REVIEW, Stack

    assert telemetry.init(SERVICE), "telemetry did not start"
    graph_mod.structured = lambda **kw: Recommendation(action="approve", rationale="Stub.", cited_rule_ids=["BAL-01"], cited_passage_ids=[], confidence="high")
    s = Stack()
    with telemetry.span("smoke.root") as root:
        trace_id = telemetry.trace_ids()[0]
        s.ws.submit(s.ws.persona("W-100234"), **ALL_PASS)                    # filing + triage in this trace
        asyncio.run(s.as_(AIKO).send(REVIEW))                               # A2A review that asks payroll
        s.ws.decide(s.ws.persona(AIKO), "REQ-2004", "approved", "")         # a human decision
        root.set("workload", "smoke")
    telemetry.flush()
    return trace_id


def test_one_trace_spans_the_filing_the_agents_and_the_decision_in_jaeger():
    trace_id = run_workload()

    def find():
        r = httpx.get(f"{JAEGER}/api/traces/{trace_id}", timeout=10)
        if r.status_code != 200:
            return None
        spans = r.json()["data"][0]["spans"]
        ops = {sp["operationName"] for sp in spans}
        return ops if {"workspace.triage", "graph.assess", "a2a-timeoff POST", "a2a-payroll POST", "workspace.decide"} <= ops else None

    ops = poll(find, "the trace in Jaeger")
    assert "graph.approval_gate" in ops and "a2a.client.send" in ops
    services = httpx.get(f"{JAEGER}/api/services", timeout=10).json()["data"]
    assert SERVICE in services


def test_the_numbers_are_queryable_in_prometheus_and_the_alert_rules_are_loaded():
    def query(q):
        r = httpx.get(f"{PROM}/api/v1/query", params={"query": q}, timeout=10).json()["data"]["result"]
        return r or None

    decisions = poll(lambda: query(f'sum(hr_decision_total{{service_name="{SERVICE}"}})'), "hr_decision_total in Prometheus")
    assert float(decisions[0]["value"][1]) >= 1
    assert poll(lambda: query(f'sum(hr_triage_total{{service_name="{SERVICE}",outcome="pending"}})'), "hr_triage_total")
    assert poll(lambda: query(f'sum(hr_peer_calls_total{{service_name="{SERVICE}",peer="payroll",outcome="ok"}})'), "hr_peer_calls_total")
    assert poll(lambda: query(f'hr_requests_pending{{service_name="{SERVICE}",status="pending"}}'), "the pending gauge")
    groups = httpx.get(f"{PROM}/api/v1/rules", timeout=10).json()["data"]["groups"]
    names = {r["name"] for g in groups for r in g["rules"]}
    assert {"HRModelDegraded", "HREvidenceChainBroken", "HRPeerBreakerOpen", "HRSpendNearCap"} <= names, names


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {str(exc)[:300]}")
    telemetry.shutdown()
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
