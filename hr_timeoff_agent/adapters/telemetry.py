"""Traces, metrics and log correlation, off unless asked for.

Everything else in the package calls the small facade below (`span`, `count`,
`observe`, `gauge`, ...). With no OpenTelemetry packages, or no endpoint set, each
call does nothing, so the offline tests and a laptop demo pay no cost and import
nothing. Turn it on with `pip install -e '.[otel]'` and either
OTEL_EXPORTER_OTLP_ENDPOINT (an OTLP/HTTP collector) or HR_OTEL=1.

What goes into telemetry is deliberately narrow: ids, outcomes, counts, model
names and costs. Never a request's free-text note, a prompt, a model's rationale
or a token. Metric labels are low-cardinality on purpose (outcome, model, tool,
surface); anything per-request (a request id) is a span attribute only.

The providers belong to this module rather than to the process, so a test can
install its own in-memory exporters, and nothing here touches OpenTelemetry's
global providers (it does use the global propagator, for W3C trace context).
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import threading
import time
from typing import Any, Callable, Iterable, Iterator

try:  # the API is a few small modules; the SDK and exporters are only needed to export
    from opentelemetry import context as otel_context, metrics as otel_metrics, propagate, trace
    from opentelemetry.trace import Status, StatusCode

    HAVE_API = True
except ImportError:  # pragma: no cover - exercised by the offline CI job, which does not install it
    HAVE_API = False

_tracer = None
_meter = None
_lock = threading.Lock()
_instruments: dict[str, Any] = {}
_gauge_fns: dict[str, Callable[[], Iterable[tuple[float, dict]]]] = {}
_providers: list[Any] = []

# Exceptions that are control flow, not failures: a graph pausing for a human is the product working.
_CONTROL_FLOW = ("GraphInterrupt", "Interrupt", "NodeInterrupt")

DESCRIPTIONS = {
    "hr.llm.calls": "Model calls by model, kind, source (cache, api, cli) and outcome",
    "hr.llm.cost.usd": "Model spend in USD for live calls",
    "hr.llm.duration": "Model call wall time, seconds",
    "hr.tool.calls": "MCP tool calls made by agents, by tool and outcome",
    "hr.triage.total": "Triage runs by outcome (pending, needs_triage)",
    "hr.triage.duration": "Triage wall time, seconds",
    "hr.model.degraded": "Triages that escalated without a recommendation because the model was unavailable",
    "hr.decision.total": "Human decisions recorded, by outcome and whether they override the agent",
    "hr.gate.refusals": "Decisions refused by the approval gate or the workspace",
    "hr.reconciled": "Decisions finished by reconciliation after a crash",
    "hr.evidence.chain_failures": "Evidence chains that failed verification when read",
    "hr.peer.calls": "Calls to a peer agent by peer and outcome (ok, unavailable, skipped)",
    "hr.peer.breaker_open": "1 while the circuit breaker for a peer is open",
    "hr.auth.attempts": "Authentication attempts by surface and outcome",
    "hr.http.requests": "HTTP requests by route and status class",
    "hr.http.duration": "HTTP request wall time, seconds",
    "hr.requests.pending": "Requests currently in each status",
    "hr.spend.utilization": "Today's model spend as a fraction of the daily cap",
}
HISTOGRAMS = {"hr.llm.duration", "hr.triage.duration", "hr.http.duration"}


def enabled() -> bool:
    return _tracer is not None


def configured() -> bool:
    return HAVE_API and bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("HR_OTEL") == "1")


def init(service: str = "hr-timeoff-agent", *, span_exporter=None, metric_reader=None, sample_ratio: float | None = None) -> bool:
    """Start exporting. Returns False (and does nothing) when it isn't configured.

    A test passes its own `span_exporter` and `metric_reader`; the service passes
    neither and gets OTLP/HTTP exporters from the standard OTEL_* environment.
    """
    global _tracer, _meter
    if not HAVE_API or not (configured() or span_exporter or metric_reader):
        return False
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    except ImportError:
        logging.getLogger(__name__).warning("OTEL is configured but the SDK is not installed: pip install -e '.[otel]'")
        return False

    resource = Resource.create({
        "service.name": os.environ.get("OTEL_SERVICE_NAME", service),
        "service.version": os.environ.get("HR_VERSION", "dev"),
    })
    ratio = sample_ratio if sample_ratio is not None else float(os.environ.get("HR_OTEL_SAMPLE_RATIO", "1.0"))
    tp = TracerProvider(resource=resource, sampler=ParentBased(TraceIdRatioBased(ratio)))
    if span_exporter is not None:
        tp.add_span_processor(SimpleSpanProcessor(span_exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    readers = []
    if metric_reader is not None:
        readers.append(metric_reader)
    elif span_exporter is None:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        readers.append(PeriodicExportingMetricReader(
            OTLPMetricExporter(), export_interval_millis=int(os.environ.get("HR_OTEL_METRIC_INTERVAL_MS", "10000"))))
    mp = MeterProvider(resource=resource, metric_readers=readers)
    with _lock:
        shutdown()
        _providers[:] = [tp, mp]
        _tracer, _meter = tp.get_tracer("hr_timeoff_agent"), mp.get_meter("hr_timeoff_agent")
        _instruments.clear()
    for name in list(_gauge_fns):
        _ensure_gauge(name)
    install_log_correlation()
    return True


def flush() -> None:
    for p in _providers:
        with contextlib.suppress(Exception):
            p.force_flush()


def shutdown() -> None:
    global _tracer, _meter
    for p in _providers:
        with contextlib.suppress(Exception):
            p.shutdown()
    _providers.clear()
    _tracer = _meter = None
    _instruments.clear()


# ── traces ──────────────────────────────────────────────────────────────────

class _NullSpan:
    def set(self, key: str, value: Any) -> None:
        pass

    def event(self, name: str, **attrs) -> None:
        pass


class _Span:
    def __init__(self, span):
        self._span = span

    def set(self, key: str, value: Any) -> None:
        if value is not None:
            self._span.set_attribute(key, value if isinstance(value, (bool, int, float, str)) else str(value))

    def event(self, name: str, **attrs) -> None:
        self._span.add_event(name, {k: v for k, v in attrs.items() if isinstance(v, (bool, int, float, str))})


@contextlib.contextmanager
def span(name: str, **attrs) -> Iterator[_Span | _NullSpan]:
    """A span around a block. Exceptions mark it failed, except a graph pausing for a human."""
    if _tracer is None:
        yield _NullSpan()
        return
    with _tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as s:
        wrapped = _Span(s)
        for k, v in attrs.items():
            wrapped.set(k, v)
        try:
            yield wrapped
        except BaseException as exc:
            if type(exc).__name__ in _CONTROL_FLOW:
                wrapped.set("paused", True)
            else:
                s.record_exception(exc)
                s.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}"))
            raise


def traced(name: str, **attrs):
    def wrap(fn):
        @functools.wraps(fn)
        def run(*a, **kw):
            with span(name, **attrs):
                return fn(*a, **kw)

        return run

    return wrap


def trace_ids() -> tuple[str, str] | None:
    if _tracer is None:
        return None
    ctx = trace.get_current_span().get_span_context()
    return (format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")) if ctx.is_valid else None


# ── context across processes (W3C traceparent) ──────────────────────────────

def inject(headers) -> None:
    """Put the current trace context into outgoing headers (a dict or httpx.Headers)."""
    if _tracer is not None:
        propagate.inject(headers)


@contextlib.contextmanager
def continue_trace(headers) -> Iterator[None]:
    """Make an incoming request's trace the parent of what happens inside the block."""
    if _tracer is None:
        yield
        return
    token = otel_context.attach(propagate.extract(dict(headers)))
    try:
        yield
    finally:
        otel_context.detach(token)


def instrument_app(app, surface: str) -> None:
    """A span and request metrics around every HTTP request (Starlette/FastAPI), continuing
    the caller's trace when it sent a traceparent. Health checks and static files are left out."""

    @app.middleware("http")
    async def _telemetry(request, call_next):
        path = request.url.path
        if _tracer is None or path == "/healthz" or path.startswith("/static"):
            return await call_next(request)
        start = time.perf_counter()
        with continue_trace(request.headers), span(f"{surface} {request.method}", surface=surface, method=request.method) as sp:
            response = await call_next(request)
            route = getattr(request.scope.get("route"), "path", None) or "unmatched"
            sp.set("http.route", route)
            sp.set("http.status_code", response.status_code)
            count("hr.http.requests", surface=surface, route=route, status=f"{response.status_code // 100}xx")
            observe("hr.http.duration", time.perf_counter() - start, surface=surface, route=route)
            return response


# ── metrics ─────────────────────────────────────────────────────────────────

def _instrument(kind: str, name: str):
    inst = _instruments.get((kind, name))
    if inst is None:
        with _lock:
            make = {"counter": _meter.create_counter, "histogram": _meter.create_histogram}[kind]
            inst = _instruments[(kind, name)] = make(name, description=DESCRIPTIONS.get(name, ""))
    return inst


def count(name: str, value: float = 1, **attrs) -> None:
    if _meter is not None:
        _instrument("counter", name).add(value, {k: str(v) for k, v in attrs.items()})


def observe(name: str, value: float, **attrs) -> None:
    if _meter is not None:
        _instrument("histogram", name).record(value, {k: str(v) for k, v in attrs.items()})


@contextlib.contextmanager
def timer(name: str, **attrs) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        observe(name, time.perf_counter() - start, **attrs)


def gauge(name: str, fn: Callable[[], Iterable[tuple[float, dict]]]) -> None:
    """An observable gauge: `fn` returns (value, attributes) pairs when metrics are collected.
    Registering the same name again replaces the function (the newest workspace wins), and a
    gauge registered before init() is created when init() runs."""
    _gauge_fns[name] = fn
    _ensure_gauge(name)


def _ensure_gauge(name: str) -> None:
    if _meter is None or ("gauge", name) in _instruments:
        return
    from opentelemetry.metrics import Observation

    def callback(options, _name=name):
        f = _gauge_fns.get(_name)
        try:
            return [Observation(v, {k: str(x) for k, x in a.items()}) for v, a in (f() if f else [])]
        except Exception:  # a metrics callback must never take the service down
            return []

    with _lock:
        _instruments[("gauge", name)] = _meter.create_observable_gauge(name, callbacks=[callback], description=DESCRIPTIONS.get(name, ""))


# ── logs ────────────────────────────────────────────────────────────────────

_factory_installed = False


def install_log_correlation() -> None:
    """Every log record gets `otel_trace_id` and `otel_span_id` ("-" outside a span)."""
    global _factory_installed
    if _factory_installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*a, **kw):
        record = previous(*a, **kw)
        ids = trace_ids()
        record.otel_trace_id, record.otel_span_id = ids if ids else ("-", "-")
        return record

    logging.setLogRecordFactory(factory)
    _factory_installed = True


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s trace=%(otel_trace_id)s span=%(otel_span_id)s %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Service entry points call this: correlated logs when telemetry is on, plain otherwise."""
    if enabled():
        logging.basicConfig(level=level, format=LOG_FORMAT)
    else:
        logging.basicConfig(level=level)
