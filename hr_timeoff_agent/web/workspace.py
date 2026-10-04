"""The web app's domain layer: one tenant's live state, and every operation on it.

State lives in a `Store` (see storage.py): by default under one home
directory, separate from the repo,

  tenant/             working copy of data/*.json (requests, absences, balances, policy)
  checkpoints.sqlite  paused graph runs, so a pending approval survives a restart
  app.sqlite          spend ledger for the daily cap and the per-person rate limit

or, with HR_DATABASE_URL, in Postgres, shared by every process that points at
it. Two caches stay as files in the home directory either way, because they
are only caches: llm_cache.json (recorded model responses) and embeddings.json.

Reset copies the committed data and fixtures back and re-triages the seeded
requests from the recordings, at no cost. The committed files are never written.

Authorization lives here, not in the HTTP layer: who may see a request, and
the graph itself decides who may decide one.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from langgraph.types import Command

from .. import evidence, graph as graph_mod, llm, policy, retrieval, telemetry
from ..storage import Store, open_store
from ..models import Finding

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIXTURES = ROOT / "fixtures"

OUTCOMES = ("approved", "declined", "returned")


class Forbidden(Exception):
    """The signed-in person may not do this."""


class Refused(Exception):
    """The graph refused a decision (wrong approver); the request stays pending."""


class BudgetExceeded(Exception):
    """The daily spend cap or the per-person rate limit would be exceeded."""


class Invalid(ValueError):
    """The submitted data does not make sense."""


@dataclass(frozen=True)
class Persona:
    worker_id: str
    name: str
    position: str
    org: str
    roles: frozenset[str]

    @property
    def is_manager(self) -> bool:
        return "manager" in self.roles

    @property
    def is_hr(self) -> bool:
        return "hr" in self.roles

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _working_days(start: date, end: date) -> int:
    return sum(1 for i in range((end - start).days + 1) if (start + timedelta(days=i)).weekday() < 5)


class Workspace:
    def __init__(self, home: Path, *, daily_cap_usd: float = 2.0, triages_per_hour: int = 5,
                 store: Store | None = None):
        self.home = Path(home)
        self.daily_cap_usd = daily_cap_usd
        self.triages_per_hour = triages_per_hour
        self.lock = threading.RLock()
        self._local = threading.local()
        self.store = store or open_store(self.home)
        with self.lock:
            self._ensure_caches()
            # Several processes can start against one database at once (the web app and the
            # agents). The first seeds it while the others wait, then they find it done.
            with self.store.lock("workspace-init"):
                if not self.store.has_documents():
                    self._seed_documents()
                self._bind()
                if not self.store.meta_get("seeded"):
                    self._seed()
            # A crash between "the graph recorded the decision" and "the request
            # was updated" leaves them disagreeing. Finish those now.
            self.reconciled = self.reconcile()
        telemetry.gauge("hr.requests.pending", self._status_counts)
        telemetry.gauge("hr.spend.utilization", lambda: [(self.spent_today() / self.daily_cap_usd if self.daily_cap_usd else 0.0, {})])

    # ── setup ────────────────────────────────────────────────────────────

    def _ensure_caches(self, *, overwrite: bool = False) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        for name in ("llm_cache.json", "embeddings.json"):
            if overwrite or not (self.home / name).exists():
                # Copy beside, then rename into place: a process starting at the same moment
                # sees the whole file or none of it, never half of one.
                tmp = self.home / f".{name}.{os.getpid()}.{threading.get_ident()}.tmp"
                shutil.copy(FIXTURES / name, tmp)
                os.replace(tmp, self.home / name)

    def _seed_documents(self) -> None:
        self.store.seed_documents(DATA)

        def normalise(requests):
            for r in requests:
                r.setdefault("status", "pending")
                r.setdefault("submitted_by", r["worker_id"])
                r.setdefault("thread_id", r["request_id"])

        self.store.mutate("requests.json", normalise)

    def _bind(self) -> None:
        llm.FIXTURES = self.home / "llm_cache.json"
        self.embedder = retrieval.Embedder(cache_path=self.home / "embeddings.json")
        self.index = retrieval.PolicyIndex(
            docs={n: self.store.read(n) for n in ("handbook.json", "precedents.json")},
            embedder=self.embedder, qdrant_url=os.environ.get("HR_QDRANT_URL") or None,
        )
        self.checkpointer = self.store.checkpointer
        llm.before_live_call = self._before_live_call
        llm.after_live_call = self._after_live_call

    def _seed(self) -> None:
        """Triage every seeded request. They replay from the recordings: no cost."""
        for r in self._read("requests.json"):
            if r["status"] == "pending":
                self.triage(r["request_id"], actor=None)
        self.store.meta_set("seeded", _now())

    def reset(self) -> None:
        with self.lock:
            self.store.reset()
            self._ensure_caches(overwrite=True)
            self._seed_documents()
            self._bind()
            self._seed()

    def close(self) -> None:
        self.store.close()
        if llm.before_live_call == self._before_live_call:
            llm.before_live_call = llm.after_live_call = None

    # ── json files ───────────────────────────────────────────────────────

    def _read(self, name: str):
        return self.store.read(name)

    def tenant(self) -> policy.Tenant:
        t = policy.Tenant(docs={n: self.store.read(n) for n in ("workers.json", "absences.json", "policy.json", "requests.json")})
        t._policy_index = self.index  # shared; the handbook is not edited at runtime
        return t

    def _graph(self, *, record: bool = False):
        return graph_mod.build(self.tenant(), record_llm=record, checkpointer=self.checkpointer)

    # ── people ───────────────────────────────────────────────────────────

    def personas(self) -> list[Persona]:
        t = self.tenant()
        managers = {w["manager_id"] for w in t.workers.values() if w.get("manager_id")}
        out = []
        for w in t.workers.values():
            roles = set(w.get("roles", [])) | {"employee"}
            if w["worker_id"] in managers:
                roles.add("manager")
            out.append(Persona(w["worker_id"], w["legal_name"], w["position"], w["supervisory_org"], frozenset(roles)))
        return sorted(out, key=lambda p: (not p.is_admin, not p.is_manager, p.name))

    def persona(self, worker_id: str | None) -> Persona | None:
        return next((p for p in self.personas() if p.worker_id == worker_id), None)

    def ready(self) -> dict[str, str | None]:
        """Each dependency this workspace needs: None if it answers, else why not."""
        out: dict[str, str | None] = {}
        for name, check in (("store", self.store.ping),
                            ("qdrant", lambda: self.index.client.get_collections() if os.environ.get("HR_QDRANT_URL") else None)):
            try:
                check()
                out[name] = None
            except Exception as exc:  # noqa: BLE001
                out[name] = f"{type(exc).__name__}"
        return out

    def _status_counts(self):
        counts: dict[str, int] = {}
        for r in self.requests():
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return [(n, {"status": s}) for s, n in counts.items()]

    def worker_id_for_email(self, email: str) -> str | None:
        """The one worker with this email, or None (unknown, or ambiguous: both mean no)."""
        wanted = (email or "").strip().lower()
        found = [w["worker_id"] for w in self.tenant().workers.values() if (w.get("email") or "").lower() == wanted]
        return found[0] if wanted and len(found) == 1 else None

    def tenant_id(self) -> str:
        return self.tenant().tenant_id

    def worker(self, worker_id: str) -> dict:
        return self.tenant().workers[worker_id]

    # ── spend control ────────────────────────────────────────────────────

    def _before_live_call(self, model: str, label: str) -> None:
        actor = getattr(self._local, "actor", None)
        spent = self.spent_today()
        if spent >= self.daily_cap_usd:
            raise BudgetExceeded(
                f"Today's model spend (${spent:.2f}) has reached the ${self.daily_cap_usd:.2f} cap. "
                "Raise HR_WEB_DAILY_CAP_USD or try again tomorrow."
            )
        if actor:
            since = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
            n = self.store.spend_count_since(actor, since)
            if n >= self.triages_per_hour:
                raise BudgetExceeded(
                    f"Limit of {self.triages_per_hour} live triages per person per hour reached. Try again later."
                )

    def _after_live_call(self, model: str, label: str, usd: float, source: str) -> None:
        actor = getattr(self._local, "actor", None)
        self.store.spend_add(_now(), actor, model, label, usd, source)

    def spent_today(self) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        return self.store.spend_on_day(today)

    def spend_summary(self) -> dict:
        rows = self.store.spend_recent(20)
        return {
            "today": round(self.spent_today(), 4),
            "cap": self.daily_cap_usd,
            "per_hour": self.triages_per_hour,
            "recent": [dict(zip(("at", "persona", "model", "label", "usd", "source"), r)) for r in rows],
        }

    # ── requests ─────────────────────────────────────────────────────────

    def requests(self) -> list[dict]:
        return self._read("requests.json")

    def request(self, request_id: str) -> dict:
        r = next((r for r in self.requests() if r["request_id"] == request_id), None)
        if r is None:
            raise KeyError(request_id)
        return r

    def _update_request(self, request_id: str, **changes) -> None:
        def apply(rows):
            for r in rows:
                if r["request_id"] == request_id:
                    r.update(changes)

        self.store.mutate("requests.json", apply)

    def can_view(self, persona: Persona, request: dict) -> bool:
        requester = self.worker(request["worker_id"])
        return (
            persona.is_hr
            or persona.worker_id == request["worker_id"]
            or persona.worker_id == requester.get("manager_id")
        )

    def visible_requests(self, persona: Persona, scope: str) -> list[dict]:
        rows = self.requests()
        if scope == "mine":
            rows = [r for r in rows if r["worker_id"] == persona.worker_id]
        elif scope == "inbox":
            reports = {w["worker_id"] for w in self.tenant().workers.values() if w.get("manager_id") == persona.worker_id}
            rows = [r for r in rows if r["worker_id"] in reports]
        elif scope == "all":
            if not persona.is_hr:
                raise Forbidden("Only HR can see every request.")
        rows = [r for r in rows if self.can_view(persona, r)]
        order = {"pending": 0, "needs_triage": 1}
        return sorted(rows, key=lambda r: (order.get(r["status"], 2), r["from"]))

    def submit(self, persona: Persona, *, start: str, end: str, hours: str, note: str, plan: str = "PTO",
               idempotency_key: str | None = None) -> str:
        """File a request and triage it. The same `idempotency_key` from the same
        person returns the request already filed instead of filing another one
        (a double click, a retried A2A message)."""
        with telemetry.span("workspace.submit", worker_id=persona.worker_id) as sp:
            rid = self._submit(persona, start=start, end=end, hours=hours, note=note, plan=plan, idempotency_key=idempotency_key)
            sp.set("request_id", rid)
            return rid

    def _submit(self, persona: Persona, *, start: str, end: str, hours: str, note: str, plan: str, idempotency_key: str | None) -> str:
        try:
            d_from, d_to = date.fromisoformat(start), date.fromisoformat(end)
        except ValueError:
            raise Invalid("Dates must be in YYYY-MM-DD format.")
        if d_to < d_from:
            raise Invalid("The end date is before the start date.")
        workdays = _working_days(d_from, d_to)
        if workdays == 0:
            raise Invalid("The window has no working days.")
        try:
            h = float(hours) if hours else workdays * 8.0
        except ValueError:
            raise Invalid("Hours must be a number.")
        if not 0 < h <= workdays * 8:
            raise Invalid(f"Hours must be between 1 and {workdays * 8:g} for {workdays} working days.")
        worker = self.worker(persona.worker_id)
        if plan not in worker["time_off_plans"]:
            raise Invalid(f"No {plan} plan for {persona.name}.")
        note = (note or "").strip()[:500]

        created: list[str] = []
        existing: list[str] = []

        def add(rows):
            dup = idempotency_key and next(
                (x for x in rows if x.get("idempotency_key") == idempotency_key and x["worker_id"] == persona.worker_id), None)
            if dup:
                existing.append(dup["request_id"])
                return
            # The id is allocated inside the document's write lock, so two
            # submissions (two processes, even) cannot be given the same one.
            rid = f"REQ-{max([int(r['request_id'].split('-')[1]) for r in rows] + [3000]) + 1}"
            rows.append({
                "request_id": rid, "worker_id": persona.worker_id, "plan": plan,
                "from": d_from.isoformat(), "to": d_to.isoformat(), "hours": h,
                "submitted_at": date.today().isoformat(), "note": note,
                "status": "needs_triage", "submitted_by": persona.worker_id, "thread_id": rid,
                **({"idempotency_key": idempotency_key} if idempotency_key else {}),
            })
            created.append(rid)

        self.store.mutate("requests.json", add)
        if existing:
            return existing[0]
        rid = created[0]
        self.triage(rid, actor=persona.worker_id)
        return rid

    def triage(self, request_id: str, *, actor: str | None) -> None:
        """Run the agent up to the approval gate. Failures leave it retryable."""
        with self.store.lock(f"request:{request_id}"), telemetry.span("workspace.triage", request_id=request_id) as sp, \
                telemetry.timer("hr.triage.duration"):
            r = self.request(request_id)
            if r["status"] not in ("pending", "needs_triage"):
                return
            attempt = int(r.get("attempts", 0)) + 1
            sp.set("attempt", attempt)
            thread = r["request_id"] if attempt == 1 else f"{r['request_id']}#{attempt}"
            self._local.actor = actor
            try:
                state = self._graph().invoke(
                    graph_mod.initial_state(r), config={"configurable": {"thread_id": thread}}
                )
            except Exception as exc:  # budget, missing key, API error: keep it retryable
                self._update_request(request_id, status="needs_triage", attempts=attempt,
                                     thread_id=thread, triage_error=f"{type(exc).__name__}: {exc}"[:500])
                telemetry.count("hr.triage.total", outcome="needs_triage", error=type(exc).__name__)
                sp.set("outcome", "needs_triage")
                if isinstance(exc, BudgetExceeded):
                    raise
                return
            finally:
                self._local.actor = None
            if "__interrupt__" not in state:
                raise RuntimeError("the graph did not pause for a human decision")
            self._update_request(request_id, status="pending", attempts=attempt,
                                 thread_id=thread, triage_error=None, triaged_at=_now())
            telemetry.count("hr.triage.total", outcome="pending", error="")
            sp.set("outcome", "pending")

    def retriage(self, persona: Persona, request_id: str) -> None:
        r = self.request(request_id)
        if not self.can_view(persona, r):
            raise Forbidden("You can't see this request.")
        if r["status"] != "needs_triage":
            raise Invalid("Only a request that failed triage can be retried.")
        self.triage(request_id, actor=persona.worker_id)

    def decide(self, persona: Persona, request_id: str, outcome: str, note: str) -> None:
        """Record a manager's decision. Safe to repeat, safe to crash in the middle.

        The graph's recorded decision is the truth; the balance, the absence and
        the request's status are then committed from it in one transaction, and
        committing is idempotent. So the same call twice is a no-op, two callers
        racing take turns on the request's lock, and a crash after the graph
        recorded the decision is finished by the next call (or at startup).
        """
        if outcome not in OUTCOMES:
            raise Invalid(f"Unknown outcome {outcome!r}.")
        with self.store.lock(f"request:{request_id}"), telemetry.span("workspace.decide", request_id=request_id, outcome=outcome, worker_id=persona.worker_id):
            r = self.request(request_id)
            if not self.can_view(persona, r):
                raise Forbidden("You can't see this request.")
            if r["status"] in OUTCOMES:
                if r["status"] == outcome and r.get("decided_by") == persona.worker_id:
                    return  # the same decision again: already done
                raise Invalid(f"This request is {r['status']}, not pending.")
            if r["status"] != "pending":
                raise Invalid(f"This request is {r['status']}, not pending.")
            cfg = {"configurable": {"thread_id": r["thread_id"]}}
            graph = self._graph()
            recorded = (graph.get_state(cfg).values or {}).get("decision")
            if recorded is None:
                # The approver is the signed-in person, never a form field.
                result = graph.invoke(
                    Command(resume={"outcome": outcome, "decided_by_id": persona.worker_id,
                                    "note": (note or "").strip()[:500]}),
                    config=cfg,
                )
                if "__interrupt__" in result:
                    telemetry.count("hr.gate.refusals")
                    raise Refused(result["__interrupt__"][0].value.get("refused", "Approval refused."))
            recorded = self._finish_decided_run(r)
            if recorded["decided_by_id"] != persona.worker_id or recorded["outcome"] != outcome:
                # A decision was already recorded (an earlier attempt that crashed).
                # It stands; it is committed above, and this caller is told.
                raise Invalid(f"This request was already decided: {recorded['outcome']} by {recorded['decided_by']}.")

    def _finish_decided_run(self, r: dict) -> dict:
        """The graph has a recorded decision: let `record` finish if it had not,
        then commit the decision to the tenant. Returns the decision."""
        cfg = {"configurable": {"thread_id": r["thread_id"]}}
        graph = self._graph()
        snap = graph.get_state(cfg)
        if snap.next:  # decided at the gate, crashed before `record`
            graph.invoke(None, config=cfg)
            snap = graph.get_state(cfg)
        decision = snap.values["decision"]
        self._commit(r["request_id"], decision, (snap.values.get("recommendation") or {}).get("action"))
        return decision

    def _commit(self, request_id: str, decision: dict, agent_action: str | None = None) -> None:
        """What `record` committed, made real in the tenant, in one transaction.

        Balances can't go negative (HB-2.1), so an approval pays up to the
        balance and records the rest as unpaid leave (HB-3.1), explicitly.
        Idempotent: a request already decided, or an absence already written,
        is left alone.
        """
        outcome = decision["outcome"]

        changed: list[bool] = []

        def apply(docs):
            r = next(x for x in docs["requests.json"] if x["request_id"] == request_id)
            if r["status"] in OUTCOMES:
                return
            changed.append(True)
            absence_id = f"ABS-{request_id}"
            if outcome == "approved" and not any(a["absence_id"] == absence_id for a in docs["absences.json"]):
                paid = 0.0
                for w in docs["workers.json"]:
                    if w["worker_id"] == r["worker_id"]:
                        plan = w["time_off_plans"][r["plan"]]
                        paid = min(float(r["hours"]), plan["balance_hours"])
                        plan["balance_hours"] = round(plan["balance_hours"] - paid, 2)
                docs["absences.json"].append({
                    "absence_id": absence_id, "worker_id": r["worker_id"],
                    "from": r["from"], "to": r["to"], "status": "approved", "plan": r["plan"],
                    "paid_hours": round(paid, 2), "unpaid_hours": round(float(r["hours"]) - paid, 2),
                })
            r.update(status=outcome, decided_at=decision.get("at") or _now(), decided_by=decision["decided_by_id"])

        self.store.mutate_many(["requests.json", "workers.json", "absences.json"], apply)
        if changed:  # counted once per decision, however many times the commit is retried
            telemetry.count("hr.decision.total", outcome=outcome,
                            overrides_agent=(agent_action, outcome) in {("approve", "declined"), ("decline", "approved")})

    def reconcile(self) -> list[str]:
        """Finish any request whose decision the graph recorded but the tenant did not."""
        fixed = []
        for r in self.requests():
            if r["status"] != "pending":
                continue
            with self.store.lock(f"request:{r['request_id']}"):
                r = self.request(r["request_id"])
                if r["status"] != "pending":
                    continue
                cfg = {"configurable": {"thread_id": r["thread_id"]}}
                if (self._graph().get_state(cfg).values or {}).get("decision") is None:
                    continue
                self._finish_decided_run(r)
                telemetry.count("hr.reconciled")
                fixed.append(r["request_id"])
        return fixed

    def detail(self, persona: Persona, request_id: str) -> dict:
        r = self.request(request_id)
        if not self.can_view(persona, r):
            raise Forbidden("You can't see this request.")
        t = self.tenant()
        requester = t.workers[r["worker_id"]]
        with self.lock:
            snap = self._graph().get_state({"configurable": {"thread_id": r.get("thread_id", r["request_id"])}})
        values = snap.values or {}
        interrupts = [i.value for task in snap.tasks for i in task.interrupts]
        ledger = values.get("evidence", [])
        ok, reason = evidence.verify(ledger) if ledger else (False, "no trail yet")
        if ledger and not ok:
            telemetry.count("hr.evidence.chain_failures")
        manager = t.workers.get(requester.get("manager_id") or "")

        # HR sees guidance written for HR only; the agent (reading as a manager)
        # never retrieves it, by the audience filter in the vector query.
        hr_only = []
        if persona.is_hr and values.get("findings"):
            hr_ids = {
                p["passage_id"] for p in self._read("handbook.json")
                if p["tenant_id"] == t.tenant_id and p["audience"] == "hr"
            }
            query = retrieval.build_query(r, [Finding.model_validate(f) for f in values["findings"]])
            hr_only = [
                p for p in self.index.search_handbook(query, tenant_id=t.tenant_id, reader="hr", k=13)
                if p.passage_id in hr_ids
            ][:2]
        rec, decision = values.get("recommendation"), values.get("decision")
        # "escalate" hands the call to a person, so any outcome follows it. Only a
        # decision that contradicts a firm approve/decline is an override.
        contradicts = {("approve", "declined"), ("decline", "approved")}
        absence = next((a for a in self._read("absences.json") if a["absence_id"] == f"ABS-{request_id}"), None)
        return {
            "overrides": bool(rec and decision and (rec["action"], decision["outcome"]) in contradicts),
            "absence": absence,
            "request": r,
            "requester": requester,
            "manager": manager,
            "findings": values.get("findings", []),
            "passages": values.get("passages", []),
            "recommendation": values.get("recommendation"),
            "decision": values.get("decision"),
            "evidence": ledger,
            "chain_ok": ok,
            "chain_reason": reason,
            "pending": bool(interrupts) and r["status"] == "pending",
            "can_decide": r["status"] == "pending" and requester.get("manager_id") == persona.worker_id,
            "hr_only": hr_only,
        }

    # ── policy ───────────────────────────────────────────────────────────

    def policy(self) -> dict:
        return self._read("policy.json")

    def update_policy(self, persona: Persona, *, min_notice_days: str, min_available_pct: str,
                      max_consecutive_days: str, blackouts: str) -> None:
        if not persona.is_admin:
            raise Forbidden("Only an admin can change policy.")
        try:
            notice, cover, consec = int(min_notice_days), int(min_available_pct), int(max_consecutive_days)
        except ValueError:
            raise Invalid("Notice days, coverage % and consecutive days must be whole numbers.")
        if not (0 <= notice <= 365 and 0 <= cover <= 100 and 1 <= consec <= 365):
            raise Invalid("Values out of range.")
        windows = []
        for line in (blackouts or "").splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3:
                raise Invalid(f"Blackout lines are 'name | YYYY-MM-DD | YYYY-MM-DD': {line!r}")
            name, a, b = parts
            try:
                if date.fromisoformat(b) < date.fromisoformat(a):
                    raise Invalid(f"Blackout {name!r} ends before it starts.")
            except ValueError:
                raise Invalid(f"Bad date in blackout {name!r}.")
            windows.append({"name": name, "from": a, "to": b})
        def revise(pol):
            rules = {r["id"]: r for r in pol["rules"]}
            rules["NOT-01"]["min_notice_days"] = notice
            rules["COV-01"]["min_available_pct"] = cover
            rules["CON-01"]["max_consecutive_days"] = consec
            rules["BLK-01"]["blackout_windows"] = windows
            pol["revision"] = int(pol.get("revision", 0)) + 1
            pol["revised_at"], pol["revised_by"] = _now(), persona.worker_id

        self.store.mutate("policy.json", revise)
