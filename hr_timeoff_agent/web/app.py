"""HTTP layer. Thin on purpose: resolve who is signed in, call the workspace,
render. Every permission check lives in the workspace and the graph, so the
same rules hold for the CLI, the tests and the browser.

    python -m hr_timeoff_agent web      # http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import llm, telemetry
from .identity import IdentityProvider, SignInFailed, identity_from_env
from .workspace import BudgetExceeded, Forbidden, Invalid, Persona, Refused, Workspace

HERE = Path(__file__).resolve().parent


class NotSignedIn(Exception):
    pass


def create_app(
    home: Path | None = None,
    *,
    identity: IdentityProvider | Callable[[Workspace], IdentityProvider] | None = None,
    daily_cap_usd: float | None = None,
    triages_per_hour: int | None = None,
) -> FastAPI:
    home = Path(home or os.environ.get("HR_WEB_HOME") or Path.cwd() / "var")
    ws = Workspace(
        home,
        daily_cap_usd=daily_cap_usd if daily_cap_usd is not None else float(os.environ.get("HR_WEB_DAILY_CAP_USD", "2.0")),
        triages_per_hour=triages_per_hour if triages_per_hour is not None else int(os.environ.get("HR_WEB_TRIAGES_PER_HOUR", "5")),
    )
    # `identity` may be a provider, or a function from the workspace to one (it needs the directory).
    ident = (identity(ws) if callable(identity) else identity) or identity_from_env(ws)
    oidc = ident.kind == "oidc"
    templates = Jinja2Templates(directory=str(HERE / "templates"))

    app = FastAPI(title="Time-off triage", docs_url=None, redoc_url=None)
    telemetry.instrument_app(app, "web")
    app.state.workspace = ws
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    def me(request: Request) -> Persona:
        persona = ws.persona(ident.current(request))
        if persona is None:
            raise NotSignedIn()
        return persona

    def render(request: Request, name: str, persona: Persona | None, status: int = 200, **ctx) -> HTMLResponse:
        live = "live" if not llm.is_offline() else "replay only (no API key)"
        return templates.TemplateResponse(
            request, name,
            {"me": persona, "msg": request.query_params.get("msg"), "model": llm.AGENT_MODEL,
             "backend": llm.BACKEND, "live": live, **ctx},
            status_code=status,
        )

    def go(url: str, msg: str | None = None) -> RedirectResponse:
        return RedirectResponse(url + (f"?msg={quote(msg)}" if msg else ""), status_code=303)

    # ── errors ───────────────────────────────────────────────────────────

    @app.exception_handler(NotSignedIn)
    async def _signin(request: Request, exc: NotSignedIn):
        return go("/signin")

    @app.exception_handler(Forbidden)
    async def _forbidden(request: Request, exc: Forbidden):
        return render(request, "error.html", ws.persona(ident.current(request)), 403, title="Not allowed", detail=str(exc))

    @app.exception_handler(Invalid)
    async def _invalid(request: Request, exc: Invalid):
        return render(request, "error.html", ws.persona(ident.current(request)), 400, title="Check your input", detail=str(exc))

    @app.exception_handler(BudgetExceeded)
    async def _budget(request: Request, exc: BudgetExceeded):
        return render(request, "error.html", ws.persona(ident.current(request)), 429, title="Spend limit reached", detail=str(exc))

    @app.exception_handler(KeyError)
    async def _missing(request: Request, exc: KeyError):
        return render(request, "error.html", ws.persona(ident.current(request)), 404, title="Not found", detail=f"No such request: {exc}")

    # ── pages ────────────────────────────────────────────────────────────

    @app.get("/healthz")
    def healthz():
        return JSONResponse({"ok": True, "model": llm.AGENT_MODEL, "live": not llm.is_offline()})

    @app.get("/")
    def home_page(request: Request):
        persona = ws.persona(ident.current(request))
        if persona is None:
            return go("/signin")
        return go("/requests?scope=inbox" if persona.is_manager else "/requests?scope=mine")

    @app.get("/signin")
    def signin_page(request: Request):
        return render(request, "signin.html", ws.persona(ident.current(request)),
                      personas=[] if oidc else ws.personas(), oidc=oidc)

    if oidc:
        # Real sign-in: the only way in is the identity provider. There is no
        # persona picker to post to, so no route that accepts a worker id.
        @app.get("/login")
        def login():
            return ident.login()

        @app.get("/auth/callback")
        def auth_callback(request: Request):
            try:
                worker_id = ident.complete(request)
            except SignInFailed as exc:
                telemetry.count("hr.auth.attempts", surface="web", outcome="refused", reason=exc.code)
                response = render(request, "error.html", None, 403, title="Sign-in failed", detail=str(exc))
                response.delete_cookie("hr_login")
                return response
            telemetry.count("hr.auth.attempts", surface="web", outcome="ok", reason="user")
            response = go("/")
            ident.sign_in(response, worker_id)
            response.delete_cookie("hr_login")
            return response
    else:
        @app.post("/signin")
        def signin(worker_id: str = Form(...)):
            if ws.persona(worker_id) is None:
                raise Invalid("Unknown person.")
            response = go("/")
            ident.sign_in(response, worker_id)
            return response

    @app.post("/signout")
    def signout():
        response = RedirectResponse((ident.logout_url() if oidc else None) or "/signin", status_code=303)
        ident.sign_out(response)
        return response

    @app.get("/requests")
    def list_page(request: Request, scope: str = "mine"):
        persona = me(request)
        if scope not in ("mine", "inbox", "all"):
            scope = "mine"
        rows = ws.visible_requests(persona, scope)
        names = {p.worker_id: p.name for p in ws.personas()}
        return render(request, "list.html", persona, rows=rows, scope=scope, names=names)

    @app.get("/requests/new")
    def new_page(request: Request):
        persona = me(request)
        return render(request, "new.html", persona, worker=ws.worker(persona.worker_id), key=uuid.uuid4().hex)

    @app.post("/requests")
    def submit(request: Request, start: str = Form(...), end: str = Form(...),
               hours: str = Form(""), note: str = Form(""), plan: str = Form("PTO"), key: str = Form("")):
        persona = me(request)
        # `key` is minted when the form is shown, so a double click files one request.
        rid = ws.submit(persona, start=start, end=end, hours=hours, note=note, plan=plan, idempotency_key=key or None)
        status = ws.request(rid)["status"]
        msg = "Submitted. The agent has triaged it." if status == "pending" else "Submitted, but triage failed; see below."
        return go(f"/requests/{rid}", msg)

    @app.get("/requests/{rid}")
    def detail_page(request: Request, rid: str):
        persona = me(request)
        d = ws.detail(persona, rid)
        names = {p.worker_id: p.name for p in ws.personas()}
        return render(request, "detail.html", persona, d=d, names=names)

    @app.post("/requests/{rid}/decide")
    def decide(request: Request, rid: str, outcome: str = Form(...), note: str = Form("")):
        persona = me(request)
        try:
            ws.decide(persona, rid, outcome, note)
        except Refused as exc:
            d = ws.detail(persona, rid)
            names = {p.worker_id: p.name for p in ws.personas()}
            return render(request, "detail.html", persona, 403, d=d, names=names, refused=str(exc))
        return go(f"/requests/{rid}", f"Recorded: {outcome}.")

    @app.post("/requests/{rid}/retriage")
    def retriage(request: Request, rid: str):
        persona = me(request)
        ws.retriage(persona, rid)
        return go(f"/requests/{rid}", "Triage retried.")

    @app.get("/admin")
    def admin_page(request: Request):
        persona = me(request)
        if not persona.is_admin:
            raise Forbidden("Only an admin can open this page.")
        pol = ws.policy()
        rules = {r["id"]: r for r in pol["rules"]}
        return render(request, "admin.html", persona, policy=pol, rules=rules, spend=ws.spend_summary())

    @app.post("/admin/policy")
    def admin_policy(request: Request, min_notice_days: str = Form(...), min_available_pct: str = Form(...),
                     max_consecutive_days: str = Form(...), blackouts: str = Form("")):
        persona = me(request)
        ws.update_policy(persona, min_notice_days=min_notice_days, min_available_pct=min_available_pct,
                         max_consecutive_days=max_consecutive_days, blackouts=blackouts)
        return go("/admin", "Policy saved. It applies to requests triaged from now on.")

    @app.post("/admin/reset")
    def admin_reset(request: Request):
        persona = me(request)
        if not persona.is_admin:
            raise Forbidden("Only an admin can reset the tenant.")
        ws.reset()
        return go("/admin", "Tenant reset to the seeded state.")

    return app
