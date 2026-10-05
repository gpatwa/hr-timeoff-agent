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
    python -m hr_timeoff_agent gate                    # live eval gate: quality, cost and latency on the real model
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from langgraph.types import Command

from .agent import assembly, graph as graph_mod
from .core import evidence, policy
from .clicommon import OUT, RULE, mode_banner as _mode_banner
from .adapters.llm import OfflineCacheMiss



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
    app = assembly.build(tenant, record_llm=args.record)
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


def cmd_web(args) -> int:
    try:
        import uvicorn

        from .services.web.app import create_app
    except ImportError:
        print("The web app needs the web extra:\n  ./.venv/bin/pip install -e '.[web]'")
        return 2
    from .adapters import telemetry

    telemetry.init("hr-web")
    telemetry.setup_logging()
    print(f"\n  Time-off triage · http://{args.host}:{args.port}  (Ctrl-C to stop)\n"
          + ("  Telemetry: exporting to " + os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "OTLP default") + "\n" if telemetry.enabled() else ""))
    try:
        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    finally:
        telemetry.flush()
    return 0


def cmd_migrate(args) -> int:
    from .adapters.storage import database_settings

    settings = database_settings(Path("."))
    if settings is None:
        print("Migrations apply to Postgres. Set HR_DATABASE_URL (local files need none).")
        return 2
    try:
        from .adapters import migrations
    except ImportError:
        print("Postgres support needs the postgres extra:\n  ./.venv/bin/pip install -e '.[postgres]'")
        return 2
    url, schema = settings
    try:
        if args.status:
            st = migrations.status(url, schema)
            print(f"  schema {schema}: at version {st.current} of {st.latest}; applied {st.applied or 'none'}, pending {st.pending or 'none'}")
            return 0 if not st.pending else 1
        done = migrations.migrate(url, schema)
    except (migrations.MigrationFailed, migrations.SchemaTooNew) as exc:
        print(f"  {exc}")
        return 1
    print(f"  applied {done}" if done else "  already up to date")
    return 0


def cmd_init(args) -> int:
    """Migrate and seed once, up front, so the services that share a database start into a finished one."""
    try:
        from .agent.workspace import Workspace
    except ImportError:
        print("Needs the web extra:\n  ./.venv/bin/pip install -e '.[web]'")
        return 2
    home = Path(args.home or os.environ.get("HR_WEB_HOME") or Path.cwd() / "var")
    ws = Workspace(home)   # opening a workspace migrates (Postgres) and seeds if empty
    print(f"  tenant {ws.tenant_id()} ready: {len(ws.requests())} requests, {len(ws.personas())} people")
    ws.close()
    return 0


def cmd_mcp(args) -> int:
    try:
        from .tools.server import HRToolServer
    except ImportError:
        print("The MCP server needs the mcp extra:\n  ./.venv/bin/pip install -e '.[mcp]'")
        return 2
    server = HRToolServer(reader=args.reader)
    if not args.http:
        server.run("stdio")
        return 0
    from .adapters.oidc import OIDCConfig

    cfg = OIDCConfig.from_env()
    if cfg is None:
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print(f"Refusing to serve the HR tools on {args.host} without authentication.\n"
                  "Set HR_OIDC_ISSUER (callers then need an access token), or bind to 127.0.0.1.")
            return 2
        server.run("streamable-http", host=args.host, port=args.http)
        return 0
    import uvicorn

    from .adapters import telemetry
    from .services.a2a.common import OIDCBearer
    from .adapters.oidc import BearerGuard

    telemetry.init("hr-mcp")
    telemetry.setup_logging()

    by_email = {(w.get("email") or "").lower(): w["worker_id"] for w in server.tenant.workers.values()}
    auth = OIDCBearer(cfg, server.tenant.tenant_id, lambda email: by_email.get((email or "").lower()), surface="mcp")
    from mcp.server.transport_security import TransportSecuritySettings

    # DNS-rebinding protection stays on: only these Host values are served. Behind a container
    # network the service name is one (HR_MCP_ALLOWED_HOSTS=mcp:8200,localhost:8200).
    hosts = [h for h in os.environ.get("HR_MCP_ALLOWED_HOSTS", "127.0.0.1:*,localhost:*,[::1]:*").split(",") if h]
    app = server.server.streamable_http_app(
        host=args.host, transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=[]))
    uvicorn.run(BearerGuard(app, auth.authenticate, public_paths=("/healthz",)), host=args.host, port=args.http, log_level="warning")
    return 0


def _a2a_setup(home: Path):
    """Workspace, token directory and apps for the two A2A agents."""
    import secrets as _secrets

    from .services.a2a.common import BearerTokens
    from .agent.workspace import Workspace

    ws = Workspace(home)
    secret = os.environ.get("HR_A2A_SECRET") or _secrets.token_hex(16)
    people = ws.personas()
    tokens = BearerTokens.derive(secret, [p.worker_id for p in people] + ["timeoff-agent"])
    return ws, tokens, people


def cmd_a2a(args) -> int:
    try:
        import asyncio

        import uvicorn

        from .services.a2a.client import A2AAgent
        from .services.a2a.payroll import create_payroll_app
        from .services.a2a.server import create_timeoff_app
    except ImportError:
        print("The A2A agents need the a2a extra:\n  ./.venv/bin/pip install -e '.[a2a]'")
        return 2
    from .adapters import telemetry

    telemetry.init("hr-a2a")
    telemetry.setup_logging()
    home = Path(args.home or os.environ.get("HR_A2A_HOME", "var/a2a"))
    ws, tokens, people = _a2a_setup(home)
    # Where callers reach each agent (the URL on its agent card). Behind a container or a
    # proxy that is not the address it binds, so it can be set.
    tf_url = os.environ.get("HR_A2A_TIMEOFF_URL") or f"http://{args.host}:{args.port}"
    pay_url = os.environ.get("HR_A2A_PAYROLL_URL") or f"http://{args.host}:{args.payroll_port}"
    from .adapters.oidc import OIDCConfig

    oidc = OIDCConfig.from_env()
    if oidc:
        # Real identity: callers present OIDC access tokens, and the time-off agent
        # calls payroll with its own client-credentials token, bound to this tenant.
        import json as _json

        from .services.a2a.common import OIDCBearer
        from .services.a2a.payroll import DATA as PAYROLL_DATA
        from .adapters.oidc import ServiceTokens

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

        from .services.a2a.client import A2AAgent
        from .services.a2a.payroll import create_payroll_app
        from .services.a2a.server import create_timeoff_app
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

    mg = sub.add_parser("migrate", help="apply (or show) the Postgres schema migrations")
    mg.add_argument("--status", action="store_true", help="show applied and pending migrations without applying")
    mg.set_defaults(func=cmd_migrate)

    ini = sub.add_parser("init", help="migrate the database and seed the tenant, then exit (run once before the services)")
    ini.add_argument("--home", help="state directory (default: HR_WEB_HOME or var)")
    ini.set_defaults(func=cmd_init)

    try:  # the eval, gate and self-test commands live in tooling, which a slim install may not ship
        from .tooling import commands as tooling_commands
    except ImportError:
        tooling_commands = None
    if tooling_commands:
        tooling_commands.register(sub)

    args = p.parse_args(argv)
    from .core.config import MissingSecret, load_file_secrets

    try:
        load_file_secrets()   # NAME_FILE=/run/secrets/x fills NAME, for containers
    except MissingSecret as exc:
        print(exc)
        return 2
    if getattr(args, "agents", None):
        os.environ["HR_AGENT_MODE"] = args.agents
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
