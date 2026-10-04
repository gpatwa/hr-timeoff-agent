"""Command line entry point.

    python -m hr_timeoff_agent list
    python -m hr_timeoff_agent run REQ-2001 --approve --as "Dana Whitfield"
    python -m hr_timeoff_agent run REQ-2004            # pauses, decides nothing
    python -m hr_timeoff_agent eval
    python -m hr_timeoff_agent rag-eval                # RAGAS over the retrieval step
    python -m hr_timeoff_agent report
    python -m hr_timeoff_agent e2e                     # self-test: every claim, pass/fail
    python -m hr_timeoff_agent web                     # the app, at http://127.0.0.1:8000
    python -m hr_timeoff_agent mcp                     # the HR tools as an MCP server (stdio)
    python -m hr_timeoff_agent a2a                     # the time-off agent and a payroll agent, over A2A
    python -m hr_timeoff_agent a2a-demo                # a walkthrough of the A2A conversation
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from langgraph.types import Command

from . import evals, evidence, graph as graph_mod, policy
from .llm import OfflineCacheMiss, is_offline

OUT = Path(__file__).resolve().parent.parent / "out"

RULE = "─" * 74


def _mode_banner() -> str:
    return "offline (fixture-backed)" if is_offline() else "live (Anthropic API)"


def cmd_list(args) -> int:
    tenant = policy.Tenant()
    print(f"\n{len(tenant.requests)} pending time off requests\n")
    for rid, r in tenant.requests.items():
        w = tenant.workers[r["worker_id"]]
        print(
            f"  {rid}  {w['legal_name']:<16} {r['plan']:<5} "
            f"{r['from']} → {r['to']}  {r['hours']:>5g}h  {w['supervisory_org']}"
        )
    print()
    return 0


def cmd_run(args) -> int:
    tenant = policy.Tenant()
    if args.request_id not in tenant.requests:
        print(f"Unknown request: {args.request_id}. Try `list`.")
        return 2

    request = tenant.requests[args.request_id]
    worker = tenant.workers[request["worker_id"]]
    app = graph_mod.build(tenant, record_llm=args.record)
    config = {"configurable": {"thread_id": f"run-{args.request_id}"}}

    print(f"\n{RULE}\n  {args.request_id} · {worker['legal_name']} · mode: {_mode_banner()}\n{RULE}")

    try:
        state = app.invoke(graph_mod.initial_state(request), config=config)
    except OfflineCacheMiss as exc:
        print(f"\n{exc}\n")
        return 1

    print("\nPOLICY FINDINGS")
    for f in state["findings"]:
        mark = {"pass": "ok  ", "warn": "warn", "fail": "FAIL"}[f["status"]]
        print(f"  {mark}  [{f['rule_id']}] {f['detail']}")

    print(f"\nRETRIEVED  (tenant {tenant.tenant_id}, reader: manager — filtered before ranking)")
    for p in state["passages"]:
        print(f"  {p['passage_id']:<7} {p['kind']:<9} {p['score']:.3f}  {p['title']}")

    rec = state["recommendation"]
    print("\nAGENT RECOMMENDATION  (advisory — the agent cannot decide)")
    print(f"  action     : {rec['action']}")
    print(f"  confidence : {rec['confidence']}")
    print(f"  cites      : {', '.join(rec['cited_rule_ids'] + rec.get('cited_passage_ids', [])) or '(none)'}")
    print(f"  rationale  : {rec['rationale']}")

    if "__interrupt__" not in state:
        print("\nExpected the graph to pause for approval and it did not. Aborting.")
        return 1

    print(f"\n⏸  PAUSED at approval_gate — decision is {state.get('decision')!r}")

    manager = tenant.workers.get(worker["manager_id"] or "")
    manager_name = manager["legal_name"] if manager else worker["manager_id"]
    outcome = args.outcome
    if not outcome:
        print(
            "\n  No human decision supplied, so nothing was committed.\n"
            f"  Only {worker['legal_name']}'s manager, {manager_name}, can decide. To resume:\n"
            f"    python -m hr_timeoff_agent run {args.request_id} --approve --as \"{manager_name}\"\n"
        )
        return 0
    if not args.decided_by:
        print(f"\n  --as is required with a decision: who is deciding? ({manager_name} can.)\n")
        return 2

    approver = tenant.find_worker(args.decided_by)
    if approver is None:
        print(f"\n  REFUSED  {args.decided_by!r} is not a unique worker id or name in {tenant.tenant_id}.\n")
        return 1
    resume = {"outcome": outcome, "decided_by_id": approver["worker_id"], "note": args.note}
    final = app.invoke(Command(resume=resume), config=config)
    if "__interrupt__" in final:
        refused = final["__interrupt__"][0].value.get("refused", "Approval refused.")
        print(f"\n  REFUSED  {refused}\n  Nothing was recorded; the request is still awaiting {manager_name}.\n")
        return 1

    print(f"\n▶  RESUMED by {approver['legal_name']} ({approver['worker_id']}) → {outcome}")
    ok, reason = evidence.verify(final["evidence"])
    print(f"\nEVIDENCE TRAIL  ({len(final['evidence'])} entries · {reason})")
    print(evidence.render(final["evidence"]))

    if args.json:
        OUT.mkdir(exist_ok=True)
        path = OUT / f"run-{args.request_id}.json"
        payload = {k: v for k, v in final.items() if k != "__interrupt__"}
        payload["chain_verified"] = ok
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        print(f"\nwrote {path}")

    print()
    return 0 if ok else 1


def cmd_eval(args) -> int:
    tenant = policy.Tenant()
    print(f"\n{RULE}\n  eval · mode: {_mode_banner()}\n{RULE}\n")
    try:
        report = evals.run_all(tenant, record=args.record)
    except OfflineCacheMiss as exc:
        print(f"{exc}\n")
        return 1

    print(f"{'case':<7} {'expected':<9} {'actual':<9} {'match':<6} {'no-self-approve':<16} scores")
    for c in report["cases"]:
        s = c["scores"]
        scores = (
            f"grounded {s['rationale_grounded']} · cites {s['citations_correct']} · tone {s['tone_appropriate']}"
            if s
            else c.get("error", "—")
        )
        print(
            f"{c['case_id']:<7} {c['expected_action']:<9} {str(c['actual_action']):<9} "
            f"{'yes' if c['action_match'] else 'NO':<6} "
            f"{'yes' if c['never_self_approved'] else 'NO':<16} {scores}"
        )

    d, m = report["deterministic"], report["judged_means"]
    print(f"\n  action_match        {d['action_match']}")
    print(f"  never_self_approved {d['never_self_approved']}")
    print(f"  judged means        grounded {m['rationale_grounded']} · "
          f"cites {m['citations_correct']} · tone {m['tone_appropriate']}   (0–3)")
    print(f"  judge model         {report['judge_model']}")

    OUT.mkdir(exist_ok=True)
    path = OUT / "eval.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nwrote {path}\n")
    return 0 if d["all_passed"] else 1


def cmd_rag_eval(args) -> int:
    from . import rag_eval

    report = rag_eval.run_all(policy.Tenant())
    print(f"\nRAG eval · ragas {report['ragas_version']} · generation metrics: {report['generation_metrics']}")
    if report["grading_model"]:
        print(f"grading model {report['grading_model']} · agent model {report['agent_model']}")
    names = list(report["means"])
    print(f"\n{'case':<7} " + " ".join(f"{n:<21}" for n in names) + " retrieved")
    for c in report["cases"]:
        cells = " ".join(f"{c['scores'].get(n, '—')!s:<21}" for n in names)
        print(f"{c['case_id']:<7} {cells} {', '.join(c['retrieved'])}")
        for name, err in c["errors"].items():
            print(f"        {name} error: {err}")
    print("\n  means (reported separately, never blended)")
    for n, v in report["means"].items():
        print(f"    {n:<21} {v}")
    OUT.mkdir(exist_ok=True)
    path = OUT / "rag_eval.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {path}\n")

    # Each floor gates its own metric; a strong recall can't cover for weak precision.
    floors = {"id_context_recall": args.min_recall, "id_context_precision": args.min_precision}
    below = [
        f"{name} {report['means'][name]} < {floor}"
        for name, floor in floors.items()
        if floor is not None and report["means"][name] < floor
    ]
    errors = [f"{c['case_id']} {n}" for c in report["cases"] for n in c["errors"]]
    for line in below + [f"metric error: {e}" for e in errors]:
        print(f"  GATE FAILED  {line}")
    return 1 if below or errors else 0


def cmd_e2e(args) -> int:
    from . import e2e

    print(f"\n{RULE}\n  end-to-end self-test · backend {_mode_banner()}\n{RULE}\n")
    report = e2e.run_all(require_live=args.require_live)
    for r in report["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  {mark}  {r['check']:<42} {r['seconds']:>5}s  {r['detail']}")
    print(f"\n  live checks: {report['live']}")
    OUT.mkdir(exist_ok=True)
    path = OUT / "e2e.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    failed = [r["check"] for r in report["results"] if not r["passed"]]
    print(f"\n  {'ALL CHECKS PASSED' if not failed else 'FAILED: ' + ', '.join(failed)}  ·  wrote {path}\n")
    return 0 if not failed else 1


def cmd_web(args) -> int:
    try:
        import uvicorn

        from .web.app import create_app
    except ImportError:
        print("The web app needs the web extra:\n  ./.venv/bin/pip install -e '.[web]'")
        return 2
    from . import telemetry

    telemetry.init("hr-web")
    telemetry.setup_logging()
    print(f"\n  Time-off triage · http://{args.host}:{args.port}  (Ctrl-C to stop)\n"
          + ("  Telemetry: exporting to " + os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "OTLP default") + "\n" if telemetry.enabled() else ""))
    try:
        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    finally:
        telemetry.flush()
    return 0


def cmd_mcp(args) -> int:
    try:
        from .mcp_server import HRToolServer
    except ImportError:
        print("The MCP server needs the mcp extra:\n  ./.venv/bin/pip install -e '.[mcp]'")
        return 2
    server = HRToolServer(reader=args.reader)
    if not args.http:
        server.run("stdio")
        return 0
    from .oidc import OIDCConfig

    cfg = OIDCConfig.from_env()
    if cfg is None:
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print(f"Refusing to serve the HR tools on {args.host} without authentication.\n"
                  "Set HR_OIDC_ISSUER (callers then need an access token), or bind to 127.0.0.1.")
            return 2
        server.run("streamable-http", host=args.host, port=args.http)
        return 0
    import uvicorn

    from . import telemetry
    from .a2a_common import OIDCBearer
    from .oidc import BearerGuard

    telemetry.init("hr-mcp")
    telemetry.setup_logging()

    by_email = {(w.get("email") or "").lower(): w["worker_id"] for w in server.tenant.workers.values()}
    auth = OIDCBearer(cfg, server.tenant.tenant_id, lambda email: by_email.get((email or "").lower()), surface="mcp")
    uvicorn.run(BearerGuard(server.server.streamable_http_app(), auth.authenticate), host=args.host, port=args.http, log_level="warning")
    return 0


def _a2a_setup(home: Path):
    """Workspace, token directory and apps for the two A2A agents."""
    import secrets as _secrets

    from .a2a_common import BearerTokens
    from .web.workspace import Workspace

    ws = Workspace(home)
    secret = os.environ.get("HR_A2A_SECRET") or _secrets.token_hex(16)
    people = ws.personas()
    tokens = BearerTokens.derive(secret, [p.worker_id for p in people] + ["timeoff-agent"])
    return ws, tokens, people


def cmd_a2a(args) -> int:
    try:
        import asyncio

        import uvicorn

        from .a2a_client import A2AAgent
        from .a2a_payroll import create_payroll_app
        from .a2a_server import create_timeoff_app
    except ImportError:
        print("The A2A agents need the a2a extra:\n  ./.venv/bin/pip install -e '.[a2a]'")
        return 2
    from . import telemetry

    telemetry.init("hr-a2a")
    telemetry.setup_logging()
    home = Path(args.home or os.environ.get("HR_A2A_HOME", "var/a2a"))
    ws, tokens, people = _a2a_setup(home)
    tf_url, pay_url = f"http://{args.host}:{args.port}", f"http://{args.host}:{args.payroll_port}"
    from .oidc import OIDCConfig

    oidc = OIDCConfig.from_env()
    if oidc:
        # Real identity: callers present OIDC access tokens, and the time-off agent
        # calls payroll with its own client-credentials token, bound to this tenant.
        import json as _json

        from .a2a_common import OIDCBearer
        from .a2a_payroll import DATA as PAYROLL_DATA
        from .oidc import ServiceTokens

        client = os.environ.get("HR_OIDC_SERVICE_CLIENT_ID", "timeoff-agent")
        secret = os.environ.get("HR_OIDC_SERVICE_CLIENT_SECRET")
        if not secret:
            print("HR_OIDC_SERVICE_CLIENT_SECRET must be set: it is the time-off agent's own credential for the payroll agent.")
            return 2
        timeoff_auth = OIDCBearer(oidc, ws.tenant_id(), ws.worker_id_for_email)
        payroll_auth = OIDCBearer(oidc, _json.loads(PAYROLL_DATA.read_text())["tenant_id"], lambda email: None)
        payroll = A2AAgent(pay_url, ServiceTokens(oidc, client, secret).get)
    else:
        timeoff_auth = payroll_auth = tokens
        payroll = A2AAgent(pay_url, tokens.for_principal("timeoff-agent"))
    servers = [
        uvicorn.Server(uvicorn.Config(create_timeoff_app(ws, timeoff_auth, payroll=payroll, base_url=tf_url), host=args.host, port=args.port, log_level="warning")),
        uvicorn.Server(uvicorn.Config(create_payroll_app(payroll_auth, base_url=pay_url, home=home), host=args.host, port=args.payroll_port, log_level="warning")),
    ]
    print(f"\n  Time-off agent   {tf_url}/.well-known/agent-card.json\n  Payroll agent    {pay_url}/.well-known/agent-card.json", flush=True)
    if os.environ.get("HR_DATABASE_URL"):
        print("  State in Postgres (HR_DATABASE_URL): shared with the web app and any other agent process\n", flush=True)
    else:
        print(f"  State in {home}  (local files: separate from the web app's ./var; don't point both at one home at once)\n", flush=True)
    if oidc:
        print(f"  Callers send an access token from {oidc.issuer} (audience {oidc.audience}, tenant_id {ws.tenant_id()}).", flush=True)
    else:
        print("  Demo bearer tokens (set HR_A2A_SECRET to keep them stable across restarts):", flush=True)
        for p in people:
            print(f"    {p.name:<16} {','.join(sorted(p.roles)):<22} {tokens.for_principal(p.worker_id)}", flush=True)
    print("\n  Ctrl-C to stop.\n", flush=True)

    async def serve():
        await asyncio.gather(*(s.serve() for s in servers))

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
    finally:
        telemetry.flush()
    return 0


def cmd_a2a_demo(args) -> int:
    """The A2A conversation end to end, in process (no ports), replayed offline."""
    try:
        import asyncio
        import tempfile

        import httpx

        from .a2a_client import A2AAgent
        from .a2a_payroll import create_payroll_app
        from .a2a_server import create_timeoff_app
    except ImportError:
        print("The A2A agents need the a2a extra:\n  ./.venv/bin/pip install -e '.[a2a]'")
        return 2
    import logging

    logging.getLogger("a2a").setLevel(logging.ERROR)  # the SDK logs a warning when an in-process client closes
    os.environ.setdefault("HR_AGENT_OFFLINE", "1")

    async def run(home):
        ws, tokens, _ = _a2a_setup(Path(home))
        pay_app = create_payroll_app(tokens, base_url="http://payroll")
        client = lambda app, base, who: A2AAgent(base, tokens.for_principal(who), httpx_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base))
        payroll = client(pay_app, "http://payroll", "timeoff-agent")
        tf_app = create_timeoff_app(ws, tokens, payroll=payroll, base_url="http://timeoff")
        manager, hr = client(tf_app, "http://timeoff", "W-100236"), client(tf_app, "http://timeoff", "W-100003")

        def say(who, what):
            print(f"  {who:<22} {what}")

        await manager.connect()
        card = manager.card
        print(f"\n{RULE}\n  A2A · {card.name} · skills: {', '.join(s.id for s in card.skills)}\n{RULE}")
        say("Aiko's agent  →", "review_time_off_request REQ-2004")
        r = await manager.send({"skill": "review_time_off_request", "request_id": "REQ-2004"})
        say("  task state", r.state)
        say("  agent says", r.text)
        pi = (r.data or {}).get("payroll_impact") or {}
        if pi.get("available"):
            say("  payroll agent (A2A)", f"{pi['unpaid_hours']:g}h unpaid ≈ {pi['estimated_pay_reduction']:,.2f} {pi['currency']}; payroll sign-off required: {pi['payroll_signoff_required']}")
        say("Grace's agent (HR) →", "review_time_off_request REQ-2004")
        h = await hr.send({"skill": "review_time_off_request", "request_id": "REQ-2004"})
        say("  task state", f"{h.state}  (view only: she is not Samuel's manager)")
        say("Aiko's agent  →", "outcome=approved on the same task")
        d = await manager.send({"outcome": "approved", "note": "40h paid, 80h unpaid"}, task_id=r.task_id, context_id=r.context_id)
        say("  task state", d.state)
        say("  agent says", d.text)
        dec = d.artifacts.get("decision", {})
        say("  recorded", f"{dec.get('paid_hours'):g}h paid + {dec.get('unpaid_hours'):g}h unpaid; evidence chain verified: {dec['evidence']['chain_verified']}")
        print()
        for a in (manager, hr, payroll):
            await a.close()
        ws.close()

    with tempfile.TemporaryDirectory() as home:
        asyncio.run(run(home))
    return 0


def cmd_report(args) -> int:
    from .report import build

    path = build()
    print(f"\nwrote {path}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hr-timeoff", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show pending requests").set_defaults(func=cmd_list)

    r = sub.add_parser("run", help="triage one request")
    r.add_argument("request_id")
    r.add_argument("--approve", dest="outcome", action="store_const", const="approved")
    r.add_argument("--decline", dest="outcome", action="store_const", const="declined")
    r.add_argument("--return", dest="outcome", action="store_const", const="returned")
    r.add_argument(
        "--as", dest="decided_by",
        help="who is deciding: worker id or exact name; must be the requester's direct manager",
    )
    r.add_argument("--note", default="")
    r.add_argument("--record", action="store_true", help="call the live API and cache the result")
    r.add_argument("--agents", choices=["single", "multi"], help="one assessment call, or tool-calling specialists plus a coordinator (default: HR_AGENT_MODE or single)")
    r.add_argument("--json", action="store_true", help="write out/run-<id>.json")
    r.set_defaults(func=cmd_run, outcome=None)

    e = sub.add_parser("eval", help="run the graded eval")
    e.add_argument("--record", action="store_true")
    e.add_argument("--agents", choices=["single", "multi"], help="which assessment to grade (default: HR_AGENT_MODE or single)")
    e.set_defaults(func=cmd_eval)

    g = sub.add_parser("rag-eval", help="RAGAS eval of retrieval")
    g.add_argument("--min-recall", type=float, help="fail if mean id_context_recall is below this")
    g.add_argument("--min-precision", type=float, help="fail if mean id_context_precision is below this")
    g.set_defaults(func=cmd_rag_eval)

    sub.add_parser("report", help="build the HTML report").set_defaults(func=cmd_report)

    w = sub.add_parser("web", help="run the web app locally")
    w.add_argument("--host", default="127.0.0.1", help="interface to bind (local only by default)")
    w.add_argument("--port", type=int, default=8000)
    w.set_defaults(func=cmd_web)

    m = sub.add_parser("mcp", help="serve the HR tools over MCP")
    m.add_argument("--reader", default="manager", choices=["manager", "hr"], help="audience the handbook search is allowed to read")
    m.add_argument("--http", type=int, metavar="PORT", help="serve streamable HTTP on this port instead of stdio")
    m.add_argument("--host", default="127.0.0.1")
    m.set_defaults(func=cmd_mcp)

    a = sub.add_parser("a2a", help="serve the time-off agent and a payroll agent over A2A")
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=8100, help="time-off agent")
    a.add_argument("--payroll-port", type=int, default=8101, help="payroll agent")
    a.add_argument("--home", help="state directory (default: HR_A2A_HOME or var/a2a)")
    a.set_defaults(func=cmd_a2a)

    sub.add_parser("a2a-demo", help="walk through the A2A conversation, in process").set_defaults(func=cmd_a2a_demo)

    x = sub.add_parser("e2e", help="end-to-end self-test of every guarantee")
    x.add_argument("--require-live", action="store_true", help="fail if live checks cannot run")
    x.set_defaults(func=cmd_e2e)

    args = p.parse_args(argv)
    if getattr(args, "agents", None):
        os.environ["HR_AGENT_MODE"] = args.agents
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
