"""Deterministic policy evaluation.

Every Finding in this module is produced by code reading data. The model is
never asked whether a rule passed — only to reason about findings it is given.
That boundary is deliberate: policy outcomes must be reproducible and
explainable without reference to a model version.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from .models import Finding

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HOURS_PER_DAY = 8.0


def _d(value: str) -> date:
    return date.fromisoformat(value)


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _working_days(start: date, end: date) -> list[date]:
    return [d for d in _days(start, end) if d.weekday() < 5]


def _overlaps(a_from: date, a_to: date, b_from: date, b_to: date) -> bool:
    return a_from <= b_to and b_from <= a_to


class Tenant:
    """Mock Workday tenant: workers, approved absences, and the active policy."""

    def __init__(self, data_dir: Path | None = None):
        d = data_dir or DATA_DIR
        self.workers = {w["worker_id"]: w for w in json.loads((d / "workers.json").read_text())}
        self.absences = json.loads((d / "absences.json").read_text())
        self.policy = json.loads((d / "policy.json").read_text())
        self.requests = {r["request_id"]: r for r in json.loads((d / "requests.json").read_text())}

    def rule(self, rule_id: str) -> dict:
        return next(r for r in self.policy["rules"] if r["id"] == rule_id)

    def org_members(self, org_id: str) -> list[dict]:
        return [w for w in self.workers.values() if w["supervisory_org_id"] == org_id]


def _finding(rule: dict, status: str, detail: str, evidence: dict) -> Finding:
    return Finding(
        rule_id=rule["id"],
        rule_name=rule["name"],
        status=status,
        severity=rule["severity_on_fail"],
        detail=detail,
        evidence=evidence,
    )


def check_balance(tenant: Tenant, request: dict, worker: dict) -> Finding:
    rule = tenant.rule("BAL-01")
    plan = request["plan"]
    balance = worker["time_off_plans"].get(plan, {}).get("balance_hours", 0.0)
    requested = request["hours"]
    ok = requested <= balance
    return _finding(
        rule,
        "pass" if ok else "fail",
        (
            f"Requested {requested:g}h against a {plan} balance of {balance:g}h."
            if ok
            else f"Requested {requested:g}h exceeds the {plan} balance of {balance:g}h "
            f"by {requested - balance:g}h."
        ),
        {"plan": plan, "requested_hours": requested, "balance_hours": balance},
    )


def check_notice(tenant: Tenant, request: dict, worker: dict) -> Finding:
    rule = tenant.rule("NOT-01")
    start, submitted = _d(request["from"]), _d(request["submitted_at"])
    notice_days = (start - submitted).days
    duration_days = len(_days(start, _d(request["to"])))
    minimum = rule["min_notice_days"]

    if duration_days <= 3:
        status, detail = "pass", (
            f"{duration_days}-day request is under the 3-day threshold; notice rule does not apply."
        )
    elif notice_days >= minimum:
        status, detail = "pass", f"{notice_days} days of notice meets the {minimum}-day minimum."
    else:
        status, detail = "warn", (
            f"{notice_days} days of notice is short of the {minimum}-day minimum "
            f"for a {duration_days}-day request."
        )
    return _finding(
        rule,
        status,
        detail,
        {"notice_days": notice_days, "minimum_days": minimum, "duration_days": duration_days},
    )


def check_blackout(tenant: Tenant, request: dict, worker: dict) -> Finding:
    rule = tenant.rule("BLK-01")
    start, end = _d(request["from"]), _d(request["to"])
    hits = [
        w
        for w in rule["blackout_windows"]
        if _overlaps(start, end, _d(w["from"]), _d(w["to"]))
    ]
    if not hits:
        return _finding(rule, "pass", "Window does not overlap a restricted period.", {"windows": []})
    names = ", ".join(f"{w['name']} ({w['from']} to {w['to']})" for w in hits)
    return _finding(
        rule,
        "fail",
        f"Window overlaps a restricted period: {names}. An approved exception is required.",
        {"windows": hits},
    )


def check_coverage(tenant: Tenant, request: dict, worker: dict) -> Finding:
    rule = tenant.rule("COV-01")
    start, end = _d(request["from"]), _d(request["to"])
    members = tenant.org_members(worker["supervisory_org_id"])
    total = len(members)
    floor = rule["min_available_pct"]

    worst_pct, worst_day, worst_out = 100.0, None, []
    for day in _working_days(start, end):
        out = [
            a["worker_id"]
            for a in tenant.absences
            if a["status"] == "approved" and _overlaps(day, day, _d(a["from"]), _d(a["to"]))
            and any(m["worker_id"] == a["worker_id"] for m in members)
        ]
        if worker["worker_id"] not in out:
            out = [*out, worker["worker_id"]]  # this request would add the requester
        pct = round((total - len(out)) / total * 100, 1) if total else 100.0
        if pct < worst_pct:
            worst_pct, worst_day, worst_out = pct, day.isoformat(), out

    ok = worst_pct >= floor
    return _finding(
        rule,
        "pass" if ok else "warn",
        (
            f"Coverage holds at {worst_pct:g}% on the tightest day, at or above the {floor}% floor."
            if ok
            else f"Coverage drops to {worst_pct:g}% on {worst_day}, below the {floor}% floor "
            f"({len(worst_out)} of {total} out in {worker['supervisory_org']})."
        ),
        {
            "org": worker["supervisory_org"],
            "org_size": total,
            "worst_day": worst_day,
            "worst_available_pct": worst_pct,
            "floor_pct": floor,
            "workers_out_on_worst_day": worst_out,
        },
    )


def check_consecutive(tenant: Tenant, request: dict, worker: dict) -> Finding:
    rule = tenant.rule("CON-01")
    span = len(_days(_d(request["from"]), _d(request["to"])))
    ceiling = rule["max_consecutive_days"]
    ok = span <= ceiling
    return _finding(
        rule,
        "pass" if ok else "warn",
        (
            f"{span} consecutive days is within the {ceiling}-day ceiling."
            if ok
            else f"{span} consecutive days exceeds the {ceiling}-day ceiling; HR Partner review applies."
        ),
        {"consecutive_days": span, "ceiling_days": ceiling},
    )


CHECKS = (check_balance, check_notice, check_blackout, check_coverage, check_consecutive)


def evaluate(tenant: Tenant, request: dict, worker: dict) -> list[Finding]:
    return [check(tenant, request, worker) for check in CHECKS]


def has_blocking_failure(findings: list[Finding]) -> bool:
    return any(f.status == "fail" and f.severity == "blocking" for f in findings)
