"""Command line entry point.

    python -m hr_timeoff_agent list
    python -m hr_timeoff_agent run REQ-2001 --approve --as "Dana Whitfield"
    python -m hr_timeoff_agent run REQ-2004            # pauses, decides nothing
    python -m hr_timeoff_agent eval
    python -m hr_timeoff_agent report
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

    rec = state["recommendation"]
    print("\nAGENT RECOMMENDATION  (advisory — the agent cannot decide)")
    print(f"  action     : {rec['action']}")
    print(f"  confidence : {rec['confidence']}")
    print(f"  cites      : {', '.join(rec['cited_rule_ids']) or '(none)'}")
    print(f"  rationale  : {rec['rationale']}")

    if "__interrupt__" not in state:
        print("\nExpected the graph to pause for approval and it did not. Aborting.")
        return 1

    print(f"\n⏸  PAUSED at approval_gate — decision is {state.get('decision')!r}")

    outcome = args.outcome
    if not outcome:
        print(
            "\n  No human decision supplied, so nothing was committed.\n"
            f"  Re-run with --approve, --decline or --return to resume:\n"
            f"    python -m hr_timeoff_agent run {args.request_id} --approve --as \"Dana Whitfield\"\n"
        )
        return 0

    resume = {"outcome": outcome, "decided_by": args.decided_by, "note": args.note}
    final = app.invoke(Command(resume=resume), config=config)

    print(f"\n▶  RESUMED by {args.decided_by} → {outcome}")
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
    r.add_argument("--as", dest="decided_by", default="Dana Whitfield", help="approver name")
    r.add_argument("--note", default="")
    r.add_argument("--record", action="store_true", help="call the live API and cache the result")
    r.add_argument("--json", action="store_true", help="write out/run-<id>.json")
    r.set_defaults(func=cmd_run, outcome=None)

    e = sub.add_parser("eval", help="run the graded eval")
    e.add_argument("--record", action="store_true")
    e.set_defaults(func=cmd_eval)

    sub.add_parser("report", help="build the HTML report").set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
