"""Traces, metrics and log correlation, with in-memory exporters (no collector needed).

Skipped when the OpenTelemetry SDK is not installed (pip install -e '.[otel]').

Run: .venv/bin/python tests/test_telemetry.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")
for var in ("HR_OIDC_ISSUER", "HR_DATABASE_URL", "OTEL_EXPORTER_OTLP_ENDPOINT", "HR_OTEL"):
    os.environ.pop(var, None)
logging.getLogger("a2a").setLevel(logging.ERROR)

try:
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
except ImportError:
    print("SKIPPED: the OpenTelemetry SDK is not installed (pip install -e '.[otel]').")
    raise SystemExit(0)

from hr_timeoff_agent.agent import graph as graph_mod
from hr_timeoff_agent.adapters import llm, telemetry  # noqa: E402
from hr_timeoff_agent.core.models import Recommendation  # noqa: E402
from hr_timeoff_agent.agent.workspace import Refused, Workspace  # noqa: E402

PRIYA, DANA, AIKO, GRACE = "W-100234", "W-100001", "W-100236", "W-100003"
NOTE = "Family trip, booked months ago."
ALL_PASS = {"start": "2026-11-30", "end": "2026-12-02", "hours": "", "note": NOTE, "plan": "PTO"}


class Tel:
    """Install in-memory exporters for the duration of a test."""

    def __enter__(self):
        self.spans, self.reader = InMemorySpanExporter(), InMemoryMetricReader()
        assert telemetry.init("hr-test", span_exporter=self.spans, metric_reader=self.reader)
        return self

    def __exit__(self, *exc):
        telemetry.shutdown()

    def finished(self):
        return self.spans.get_finished_spans()

    def named(self, name):
        return [s for s in self.finished() if s.name == name]

    def metric(self, name):
        data, out = self.reader.get_metrics_data(), []
        for rm in (data.resource_metrics if data else []):
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    if m.name == name:
                        for p in m.data.data_points:
                            out.append((dict(p.attributes), getattr(p, "value", None) if hasattr(p, "value") else p.sum))
        return out

    def total(self, name, **where):
        return sum(v for a, v in self.metric(name) if all(a.get(k) == str(x) for k, x in where.items()))


def _stub(*, system, user, schema, model=llm.AGENT_MODEL, record=False, label=""):
    return Recommendation(action="approve", rationale="Stub.", cited_rule_ids=["BAL-01"], cited_passage_ids=[], confidence="high")


def _down(**kw):
    raise llm.ModelUnavailable("connection refused")


# ── traces ──────────────────────────────────────────────────────────────────

def test_a_triage_is_one_trace_through_every_graph_node_and_the_model_call():
    ws = Workspace(tempfile.mkdtemp())
    with Tel() as t:
        ws.reset()   # re-triages the seeded requests from the recordings
        triage = next(s for s in t.named("workspace.triage") if s.attributes["request_id"] == "REQ-2004")
        trace = [s for s in t.finished() if s.context.trace_id == triage.context.trace_id]
        names = [s.name for s in sorted(trace, key=lambda s: s.start_time)]
        for node in ("load_context", "check_policy", "retrieve", "assess", "approval_gate"):
            assert f"graph.{node}" in names, (node, names)
        order = [n for n in names if n.startswith("graph.")]
        assert order.index("graph.load_context") < order.index("graph.check_policy") < order.index("graph.assess") < order.index("graph.approval_gate")
        assert "retrieval.search" in names
        call = next(s for s in trace if s.name == "llm.call")
        assert call.attributes["source"] == "cache" and call.attributes["outcome"] == "ok" and call.attributes["kind"] == "assess"
        gate = next(s for s in trace if s.name == "graph.approval_gate")
        assert gate.attributes.get("paused") is True and gate.status.status_code.name != "ERROR", "pausing for a human is not a failure"
        assert t.total("hr.llm.calls", source="cache", outcome="ok", kind="assess") >= 5
        assert t.total("hr.triage.total", outcome="pending") >= 5
        ws.close()


def test_a_decision_is_counted_once_and_a_refusal_is_counted():
    ws = Workspace(tempfile.mkdtemp())
    with Tel() as t:
        ws.decide(ws.persona(DANA), "REQ-2001", "declined", "")          # the agent recommended approve
        ws.decide(ws.persona(DANA), "REQ-2001", "declined", "")          # the same call again: no new decision
        assert t.total("hr.decision.total", outcome="declined", overrides_agent=True) == 1
        try:
            ws.decide(ws.persona(GRACE), "REQ-2004", "approved", "")    # HR can see it, but is not the manager
        except Refused:
            pass
        assert t.total("hr.gate.refusals") == 1
        spans = t.named("workspace.decide")
        assert len(spans) == 3 and {s.attributes["outcome"] for s in spans} == {"declined", "approved"}
        ws.close()


def test_a_model_outage_and_a_pending_backlog_are_visible():
    ws = Workspace(tempfile.mkdtemp())
    with Tel() as t:
        graph_mod.structured = _down
        rid = ws.submit(ws.persona(PRIYA), **ALL_PASS)
        assert t.total("hr.model.degraded", node="assess") == 1
        assert t.total("hr.triage.total", outcome="pending") == 1
        pending = ws.requests()
        want = sum(1 for r in pending if r["status"] == "pending")
        assert t.total("hr.requests.pending", status="pending") == want and want >= 1, (want, rid)
        ws.close()


def test_trace_context_crosses_both_a2a_hops():
    from test_a2a import AIKO as A_AIKO, REVIEW, Stack

    s = Stack()
    with Tel() as t:
        with telemetry.span("test.root"):
            asyncio.run(s.as_(A_AIKO).send(REVIEW))   # REQ-2004 has unpaid hours: the agent asks payroll
        root = t.named("test.root")[0].context.trace_id
        for name in ("a2a.client.send", "a2a-timeoff POST", "a2a.execute", "a2a-payroll POST"):
            spans = t.named(name)
            assert spans, f"no {name} span"
            assert all(sp.context.trace_id == root for sp in spans), f"{name} is not in the caller's trace"
        assert len(t.named("a2a.client.send")) >= 2, "the manager's call and the time-off agent's call to payroll"
        assert t.total("hr.peer.calls", peer="payroll", outcome="ok") == 1
        assert t.total("hr.http.requests", surface="a2a-payroll") >= 1


def test_the_peer_breaker_and_auth_refusals_are_metrics():
    from test_oidc import CFG, KEY, token, verifier
    from hr_timeoff_agent.services.a2a.common import OIDCBearer
    from hr_timeoff_agent.services.a2a.server import TimeOffExecutor

    ws = Workspace(tempfile.mkdtemp())
    with Tel() as t:
        ex = TimeOffExecutor(ws, None)
        assert t.total("hr.peer.breaker_open", peer="payroll") == 0
        for _ in range(3):
            ex.payroll_breaker.failure()
        assert t.total("hr.peer.breaker_open", peer="payroll") == 1
        auth = OIDCBearer(CFG, "TEN-001", ws.worker_id_for_email, verifier=verifier())
        assert auth.authenticate(f"Bearer {token()}") is not None
        assert auth.authenticate("Bearer garbage") is None
        assert auth.authenticate(f"Bearer {token(tenant_id='TEN-002')}") is None
        assert auth.authenticate(f"Bearer {token(email='stranger@acme.example')}") is None
        assert auth.authenticate(f"Bearer {token(email_verified=False)}") is None
        for reason, n in (("invalid_token", 1), ("wrong_tenant", 1), ("not_in_directory", 1), ("unverified_email", 1)):
            assert t.total("hr.auth.attempts", surface="a2a", outcome="refused", reason=reason) == n, reason
        assert t.total("hr.auth.attempts", surface="a2a", outcome="ok") == 1
        ws.close()


def test_web_requests_are_counted_by_route_and_health_checks_are_not():
    from fastapi.testclient import TestClient

    from hr_timeoff_agent.services.web.app import create_app

    with Tel() as t:
        c = TestClient(create_app(tempfile.mkdtemp()))
        c.get("/healthz")
        c.get("/requests?scope=mine", follow_redirects=False)
        assert t.total("hr.http.requests", surface="web", route="/requests", status="3xx") == 1
        assert not [s for s in t.named("web GET") if s.attributes.get("http.route") == "/healthz"]
        assert t.metric("hr.http.duration")


# ── what must stay out ──────────────────────────────────────────────────────

def _all_text(t: Tel) -> str:
    parts = []
    for s in t.finished():
        parts += [s.name, *map(str, s.attributes.values())]
        for e in s.events:
            parts += [e.name, *map(str, (e.attributes or {}).values())]
    for name in ("hr.llm.calls", "hr.triage.total", "hr.decision.total", "hr.http.requests", "hr.tool.calls"):
        for attrs, _ in t.metric(name):
            parts += list(map(str, attrs.values()))
    return "\n".join(parts)


def test_free_text_prompts_and_rationales_never_reach_telemetry():
    ws = Workspace(tempfile.mkdtemp())
    with Tel() as t:
        graph_mod.structured = _stub
        rid = ws.submit(ws.persona(PRIYA), **ALL_PASS)
        ws.decide(ws.persona(DANA), rid, "approved", "A private remark by the manager")
        ws.reset()   # real recorded rationales flow through the graph
        rationale = ws.detail(ws.persona(GRACE), "REQ-2004")["recommendation"]["rationale"]
        text = _all_text(t)
        for secret in (NOTE, "A private remark by the manager", rationale[:40]):
            assert secret not in text, f"{secret!r} leaked into telemetry"
        assert "REQ-2004" in text and "W-100001" in text, "ids are what telemetry is for"
        ws.close()


# ── logs and the off switch ─────────────────────────────────────────────────

def test_log_records_carry_the_trace_id():
    records = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(record)

    log = logging.getLogger("hr-test-correlation")
    log.addHandler(Grab())
    log.setLevel(logging.INFO)
    with Tel():
        with telemetry.span("work"):
            trace_id, span_id = telemetry.trace_ids()
            log.info("inside")
        log.info("outside")
    inside, outside = records
    assert (inside.otel_trace_id, inside.otel_span_id) == (trace_id, span_id)
    assert outside.otel_trace_id == "-"


def test_with_no_telemetry_everything_is_a_no_op():
    telemetry.shutdown()
    assert telemetry.init("x") is False and not telemetry.enabled()
    with telemetry.span("anything", a=1) as sp:
        sp.set("k", "v")
        telemetry.count("hr.llm.calls", model="m")
        telemetry.observe("hr.llm.duration", 1.0)
        with telemetry.timer("hr.triage.duration"):
            pass
    assert telemetry.trace_ids() is None
    headers = {}
    telemetry.inject(headers)
    assert headers == {}
    with telemetry.continue_trace({"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"}):
        pass
    ws = Workspace(tempfile.mkdtemp())
    assert ws.detail(ws.persona(DANA), "REQ-2001")["chain_ok"]
    ws.close()


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
            print(f"  FAIL  {name}: {type(exc).__name__}: {str(exc)[:300]}")
        finally:
            graph_mod.structured = saved
            telemetry.shutdown()
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
