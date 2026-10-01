"""The web app's domain layer: one tenant's live state, and every operation on it.

All state lives under one home directory, separate from the repo:

  tenant/             working copy of data/*.json (requests, absences, balances, policy)
  llm_cache.json      working copy of the recorded model responses
  embeddings.json     working copy of the embedding cache
  checkpoints.sqlite  paused graph runs, so a pending approval survives a restart
  app.sqlite          spend ledger for the daily cap and the per-person rate limit

Reset copies the committed data and fixtures back and re-triages the seeded
requests from the recordings, at no cost. The committed files are never written.

Authorization lives here, not in the HTTP layer: who may see a request, and
the graph itself decides who may decide one.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .. import evidence, graph as graph_mod, llm, policy, retrieval
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
    def __init__(self, home: Path, *, daily_cap_usd: float = 2.0, triages_per_hour: int = 5):
        self.home = Path(home)
        self.daily_cap_usd = daily_cap_usd
        self.triages_per_hour = triages_per_hour
        self.lock = threading.RLock()
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        with self.lock:
            if not (self.home / "tenant" / "policy.json").exists():
                self._fresh_copy()
            self._bind()
            if not self._db.execute("select 1 from meta where k='seeded'").fetchone():
                self._seed()

    # ── setup ────────────────────────────────────────────────────────────

    @property
    def tenant_dir(self) -> Path:
        return self.home / "tenant"

    def _fresh_copy(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        for name in ("tenant", "llm_cache.json", "embeddings.json", "checkpoints.sqlite", "app.sqlite"):
            target = self.home / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        shutil.copytree(DATA, self.tenant_dir)
        shutil.copy(FIXTURES / "llm_cache.json", self.home / "llm_cache.json")
        shutil.copy(FIXTURES / "embeddings.json", self.home / "embeddings.json")
        requests = self._read("requests.json")
        for r in requests:
            r.setdefault("status", "pending")
            r.setdefault("submitted_by", r["worker_id"])
            r.setdefault("thread_id", r["request_id"])
        self._write("requests.json", requests)

    def _bind(self) -> None:
        llm.FIXTURES = self.home / "llm_cache.json"
        self.embedder = retrieval.Embedder(cache_path=self.home / "embeddings.json")
        self.index = retrieval.PolicyIndex(data_dir=self.tenant_dir, embedder=self.embedder)

        ck = sqlite3.connect(self.home / "checkpoints.sqlite", check_same_thread=False)
        self.checkpointer = SqliteSaver(ck)
        self._db = sqlite3.connect(self.home / "app.sqlite", check_same_thread=False)
        self._db.executescript(
            """
            create table if not exists meta (k text primary key, v text);
            create table if not exists spend (
                at text, persona text, model text, label text, usd real, source text);
            """
        )
        self._conns = [ck, self._db]
        llm.before_live_call = self._before_live_call
        llm.after_live_call = self._after_live_call

    def _seed(self) -> None:
        """Triage every seeded request. They replay from the recordings: no cost."""
        for r in self._read("requests.json"):
            if r["status"] == "pending":
                self.triage(r["request_id"], actor=None)
        self._db.execute("insert or replace into meta values ('seeded', ?)", (_now(),))
        self._db.commit()

    def reset(self) -> None:
        with self.lock:
            for c in self._conns:
                c.close()
            self._fresh_copy()
            self._bind()
            self._seed()

    def close(self) -> None:
        for c in self._conns:
            c.close()
        if llm.before_live_call == self._before_live_call:
            llm.before_live_call = llm.after_live_call = None

    # ── json files ───────────────────────────────────────────────────────

    def _read(self, name: str):
        return json.loads((self.tenant_dir / name).read_text())

    def _write(self, name: str, value) -> None:
        tmp = self.tenant_dir / f".{name}.tmp"
        tmp.write_text(json.dumps(value, indent=2) + "\n")
        tmp.replace(self.tenant_dir / name)

    def tenant(self) -> policy.Tenant:
        t = policy.Tenant(self.tenant_dir)
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
            n = self._db.execute(
                "select count(*) from spend where persona=? and at>=?", (actor, since)
            ).fetchone()[0]
            if n >= self.triages_per_hour:
                raise BudgetExceeded(
                    f"Limit of {self.triages_per_hour} live triages per person per hour reached. Try again later."
                )

    def _after_live_call(self, model: str, label: str, usd: float, source: str) -> None:
        actor = getattr(self._local, "actor", None)
        self._db.execute(
            "insert into spend values (?,?,?,?,?,?)", (_now(), actor, model, label, usd, source)
        )
        self._db.commit()

    def spent_today(self) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        return float(self._db.execute(
            "select coalesce(sum(usd),0) from spend where substr(at,1,10)=?", (today,)
        ).fetchone()[0])

    def spend_summary(self) -> dict:
        rows = self._db.execute(
            "select at, persona, model, label, usd, source from spend order by at desc limit 20"
        ).fetchall()
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
        rows = self.requests()
        for r in rows:
            if r["request_id"] == request_id:
                r.update(changes)
        self._write("requests.json", rows)

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

    def submit(self, persona: Persona, *, start: str, end: str, hours: str, note: str, plan: str = "PTO") -> str:
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

        with self.lock:
            rows = self.requests()
            n = max([int(r["request_id"].split("-")[1]) for r in rows] + [3000]) + 1
            rid = f"REQ-{n}"
            rows.append({
                "request_id": rid, "worker_id": persona.worker_id, "plan": plan,
                "from": d_from.isoformat(), "to": d_to.isoformat(), "hours": h,
                "submitted_at": date.today().isoformat(), "note": note,
                "status": "needs_triage", "submitted_by": persona.worker_id, "thread_id": rid,
            })
            self._write("requests.json", rows)
        self.triage(rid, actor=persona.worker_id)
        return rid

    def triage(self, request_id: str, *, actor: str | None) -> None:
        """Run the agent up to the approval gate. Failures leave it retryable."""
        with self.lock:
            r = self.request(request_id)
            if r["status"] not in ("pending", "needs_triage"):
                return
            attempt = int(r.get("attempts", 0)) + 1
            thread = r["request_id"] if attempt == 1 else f"{r['request_id']}#{attempt}"
            self._local.actor = actor
            try:
                state = self._graph().invoke(
                    graph_mod.initial_state(r), config={"configurable": {"thread_id": thread}}
                )
            except Exception as exc:  # budget, missing key, API error: keep it retryable
                self._update_request(request_id, status="needs_triage", attempts=attempt,
                                     thread_id=thread, triage_error=f"{type(exc).__name__}: {exc}"[:500])
                if isinstance(exc, BudgetExceeded):
                    raise
                return
            finally:
                self._local.actor = None
            if "__interrupt__" not in state:
                raise RuntimeError("the graph did not pause for a human decision")
            self._update_request(request_id, status="pending", attempts=attempt,
                                 thread_id=thread, triage_error=None, triaged_at=_now())

    def retriage(self, persona: Persona, request_id: str) -> None:
        r = self.request(request_id)
        if not self.can_view(persona, r):
            raise Forbidden("You can't see this request.")
        if r["status"] != "needs_triage":
            raise Invalid("Only a request that failed triage can be retried.")
        self.triage(request_id, actor=persona.worker_id)

    def decide(self, persona: Persona, request_id: str, outcome: str, note: str) -> None:
        if outcome not in OUTCOMES:
            raise Invalid(f"Unknown outcome {outcome!r}.")
        with self.lock:
            r = self.request(request_id)
            if not self.can_view(persona, r):
                raise Forbidden("You can't see this request.")
            if r["status"] != "pending":
                raise Invalid(f"This request is {r['status']}, not pending.")
            cfg = {"configurable": {"thread_id": r["thread_id"]}}
            # The approver is the signed-in person, never a form field.
            result = self._graph().invoke(
                Command(resume={"outcome": outcome, "decided_by_id": persona.worker_id,
                                "note": (note or "").strip()[:500]}),
                config=cfg,
            )
            if "__interrupt__" in result:
                raise Refused(result["__interrupt__"][0].value.get("refused", "Approval refused."))
            self._apply(r, outcome)
            self._update_request(request_id, status=outcome, decided_at=_now(), decided_by=persona.worker_id)

    def _apply(self, r: dict, outcome: str) -> None:
        """What `record` committed, made real in the tenant: balance and absences.

        Balances can't go negative (HB-2.1), so an approval pays up to the
        balance and records the rest as unpaid leave (HB-3.1), explicitly.
        """
        if outcome != "approved":
            return
        workers = self._read("workers.json")
        paid = 0.0
        for w in workers:
            if w["worker_id"] == r["worker_id"]:
                plan = w["time_off_plans"][r["plan"]]
                paid = min(float(r["hours"]), plan["balance_hours"])
                plan["balance_hours"] = round(plan["balance_hours"] - paid, 2)
        self._write("workers.json", workers)
        absences = self._read("absences.json")
        absences.append({
            "absence_id": f"ABS-{r['request_id']}", "worker_id": r["worker_id"],
            "from": r["from"], "to": r["to"], "status": "approved", "plan": r["plan"],
            "paid_hours": round(paid, 2), "unpaid_hours": round(float(r["hours"]) - paid, 2),
        })
        self._write("absences.json", absences)

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
        with self.lock:
            pol = self.policy()
            rules = {r["id"]: r for r in pol["rules"]}
            rules["NOT-01"]["min_notice_days"] = notice
            rules["COV-01"]["min_available_pct"] = cover
            rules["CON-01"]["max_consecutive_days"] = consec
            rules["BLK-01"]["blackout_windows"] = windows
            pol["revision"] = int(pol.get("revision", 0)) + 1
            pol["revised_at"], pol["revised_by"] = _now(), persona.worker_id
            self._write("policy.json", pol)
