"""The CLI commands for evals, the live gate and the end-to-end self-test (see hr_timeoff_agent/cli.py)."""

from __future__ import annotations

import json
from pathlib import Path

from .. import policy
from ..clicommon import OUT, RULE, mode_banner
from ..llm import OfflineCacheMiss
from . import evals

def cmd_eval(args) -> int:
    tenant = policy.Tenant()
    print(f"\n{RULE}\n  eval · mode: {mode_banner()}\n{RULE}\n")
    try:
        report = evals.run_all(tenant, record=args.record)
    except OfflineCacheMiss as exc:
        print(f"{exc}\n")
        return 1

    print(f"{'case':<7} {'category':<16} {'acceptable':<17} {'actual':<9} {'ok':<4} {'inv':<4} {'attack':<7} scores")
    for c in report["cases"]:
        s = c["scores"]
        scores = (
            f"grounded {s['rationale_grounded']} · cites {s['citations_correct']} · tone {s['tone_appropriate']}"
            if s
            else c.get("error", "—")
        )
        invariants = c["never_self_approved"] and c["no_approve_on_blocking"] and c["cites_failures"]
        attack = "—" if not c["attack"] else ("held" if c["injection_resisted"] else "FELL")
        print(
            f"{c['case_id']:<7} {c['category']:<16} {'/'.join(c['acceptable_actions']):<17} {str(c['actual_action']):<9} "
            f"{'yes' if c['action_match'] else 'NO':<4} {'yes' if invariants else 'NO':<4} {attack:<7} {scores}"
        )

    d, m = report["deterministic"], report["judged_means"]
    print(f"\n  action_match            {d['action_match']}")
    print(f"  never_self_approved     {d['never_self_approved']}")
    print(f"  no_approve_on_blocking  {d['no_approve_on_blocking']}")
    print(f"  cites_failures          {d['cites_failures']}")
    print(f"  injection_resisted      {d['injection_resisted']}")
    print(f"  judged means            grounded {m['rationale_grounded']} · "
          f"cites {m['citations_correct']} · tone {m['tone_appropriate']}   (0–3)")
    print(f"  judge model             {report['judge_model']}")

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

    print(f"\n{RULE}\n  end-to-end self-test · backend {mode_banner()}\n{RULE}\n")
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


def cmd_gate(args) -> int:
    from . import gate

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not set(modes) <= {"single", "multi"}:
        print("--modes takes single, multi or both (single,multi).")
        return 2
    if "multi" in modes:
        try:
            import mcp  # noqa: F401
        except ImportError:
            print("Multi-agent mode needs the mcp extra:\n  ./.venv/bin/pip install -e '.[mcp]'")
            return 2
    try:
        report = gate.run_gate(modes=modes, repeats=args.repeats, thresholds_path=Path(args.thresholds), replay=args.replay,
                               a2a=not args.no_a2a and not args.replay, budget_usd=args.budget,
                               workers=args.workers, quick=args.quick)
    except RuntimeError as exc:
        print(f"\n  {exc}\n")
        return 2
    text = gate.render_markdown(report)
    print("\n" + text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    if args.summary:
        Path(args.summary).write_text(text)
    return 0 if report["passed"] else 1


def cmd_report(args) -> int:
    from .report import build

    path = build()
    print(f"\nwrote {path}\n")
    return 0


def register(sub) -> None:
    """Add the tooling commands to the CLI's subparsers."""
    e = sub.add_parser("eval", help="run the graded eval")
    e.add_argument("--record", action="store_true")
    e.add_argument("--agents", choices=["single", "multi"], help="which assessment to grade (default: HR_AGENT_MODE or single)")
    e.set_defaults(func=cmd_eval)

    g = sub.add_parser("rag-eval", help="RAGAS eval of retrieval")
    g.add_argument("--min-recall", type=float, help="fail if mean id_context_recall is below this")
    g.add_argument("--min-precision", type=float, help="fail if mean id_context_precision is below this")
    g.set_defaults(func=cmd_rag_eval)

    sub.add_parser("report", help="build the HTML report").set_defaults(func=cmd_report)

    gt = sub.add_parser("gate", help="the live eval gate: quality, cost and latency thresholds on the real model")
    gt.add_argument("--modes", default="single,multi", help="single, multi or both")
    gt.add_argument("--repeats", type=int, default=1, help="runs per case (the model is not deterministic)")
    gt.add_argument("--thresholds", default=str(Path(__file__).resolve().parent.parent.parent / "evals" / "thresholds.json"))
    gt.add_argument("--replay", action="store_true", help="offline: replay the recorded fixtures to check the plumbing; no model is called")
    gt.add_argument("--no-a2a", action="store_true", help="skip the multi-agent run through the A2A agents")
    gt.add_argument("--budget", type=float, help="stop the run when it has spent this many dollars (default: from thresholds.json)")
    gt.add_argument("--workers", type=int, default=1, help="cases to run at once (the full suite is about an hour one at a time)")
    gt.add_argument("--quick", action="store_true", help="a 12-case spread of every category, for routine checks")
    gt.add_argument("--out", help="write the full report as JSON")
    gt.add_argument("--summary", help="write the markdown summary here (for GITHUB_STEP_SUMMARY)")
    gt.set_defaults(func=cmd_gate)

    x = sub.add_parser("e2e", help="end-to-end self-test of every guarantee")
    x.add_argument("--require-live", action="store_true", help="fail if live checks cannot run")
    x.set_defaults(func=cmd_e2e)
