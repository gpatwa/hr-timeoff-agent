"""Command line entry point.

    python -m hr_timeoff_agent list
    python -m hr_timeoff_agent run REQ-2001 --approve --as "Dana Whitfield"
    python -m hr_timeoff_agent run REQ-2004            # pauses, decides nothing
    python -m hr_timeoff_agent eval
    python -m hr_timeoff_agent rag-eval                # RAGAS over the retrieval step
    python -m hr_timeoff_agent report
    python -m hr_timeoff_agent e2e                     # self-test: every claim, pass/fail
"""

from __future__ import annotations

import argparse
import json
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
    r.add_argument("--json", action="store_true", help="write out/run-<id>.json")
    r.set_defaults(func=cmd_run, outcome=None)

    e = sub.add_parser("eval", help="run the graded eval")
    e.add_argument("--record", action="store_true")
    e.set_defaults(func=cmd_eval)

    g = sub.add_parser("rag-eval", help="RAGAS eval of retrieval")
    g.add_argument("--min-recall", type=float, help="fail if mean id_context_recall is below this")
    g.add_argument("--min-precision", type=float, help="fail if mean id_context_precision is below this")
    g.set_defaults(func=cmd_rag_eval)

    sub.add_parser("report", help="build the HTML report").set_defaults(func=cmd_report)

    x = sub.add_parser("e2e", help="end-to-end self-test of every guarantee")
    x.add_argument("--require-live", action="store_true", help="fail if live checks cannot run")
    x.set_defaults(func=cmd_e2e)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
