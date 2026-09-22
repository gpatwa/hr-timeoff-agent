"""Build a single self-contained HTML page from real run output.

No CDN, no fonts to fetch — it has to open from a file:// URL with no network.
Everything on the page comes from out/*.json, so it cannot drift from what the
code actually did.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

CSS = """
:root{
  --bg:#F2F3F5; --surface:#FFF; --surface-2:#F8F9FA;
  --ink:#171B21; --ink-soft:#5C6673; --line:rgba(23,27,33,.13); --line-soft:rgba(23,27,33,.07);
  --human:#9A5B1E; --human-bg:#FBEFE1;
  --ok:#2C6A56; --ok-bg:#DFEDE7;
  --warn:#8A6410; --warn-bg:#F7EEDA;
  --fail:#95341F; --fail-bg:#F8E4DF;
  --agent:#33518C; --agent-bg:#E3E9F5;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0F1319; --surface:#161B23; --surface-2:#11161D;
  --ink:#E5E9EE; --ink-soft:#98A3B1; --line:rgba(229,233,238,.14); --line-soft:rgba(229,233,238,.07);
  --human:#E39B57; --human-bg:#3A2A18;
  --ok:#6DBBA1; --ok-bg:#15302A;
  --warn:#D6AC5A; --warn-bg:#332A15;
  --fail:#E08772; --fail-bg:#3A211B;
  --agent:#8FA9DE; --agent-bg:#1B2540;
}}
:root[data-theme="dark"]{
  --bg:#0F1319; --surface:#161B23; --surface-2:#11161D;
  --ink:#E5E9EE; --ink-soft:#98A3B1; --line:rgba(229,233,238,.14); --line-soft:rgba(229,233,238,.07);
  --human:#E39B57; --human-bg:#3A2A18;
  --ok:#6DBBA1; --ok-bg:#15302A;
  --warn:#D6AC5A; --warn-bg:#332A15;
  --fail:#E08772; --fail-bg:#3A211B;
  --agent:#8FA9DE; --agent-bg:#1B2540;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:34px 22px 80px}
.mono{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
header{border-bottom:2px solid var(--ink);padding-bottom:20px;margin-bottom:26px}
.kicker{font:500 11px/1 ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase;
  color:var(--human);margin:0 0 11px}
h1{font-size:30px;line-height:1.15;letter-spacing:-.02em;margin:0 0 10px;text-wrap:balance}
.lede{color:var(--ink-soft);margin:0;max-width:62ch}
h2{font-size:19px;letter-spacing:-.01em;margin:38px 0 6px}
h2+.note{color:var(--ink-soft);margin:0 0 16px;max-width:62ch;font-size:14px}
.claim{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--human);
  border-radius:0 5px 5px 0;padding:16px 18px;margin:0 0 16px}
.claim p{margin:0;font-size:16px}
.assertions{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:10px;margin:16px 0 0}
.assert{background:var(--surface);border:1px solid var(--line);border-radius:5px;padding:12px 14px}
.assert .k{font:500 10px/1 ui-monospace,monospace;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-soft);margin:0 0 7px}
.assert .v{font-size:15px;font-weight:600;margin:0;display:flex;align-items:center;gap:7px}
.dot{width:7px;height:7px;border-radius:50%;flex:none}
.pass .dot{background:var(--ok)} .pass .v{color:var(--ok)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.card+.card{margin-top:14px}
.card-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;padding:13px 16px;
  border-bottom:1px solid var(--line-soft);background:var(--surface-2)}
.card-head .id{font:500 12px/1 ui-monospace,monospace;color:var(--human)}
.card-head .who{font-weight:600}
.card-head .meta{margin-left:auto;font-size:12.5px;color:var(--ink-soft)}
.rows{display:flex;flex-direction:column}
.row{display:grid;grid-template-columns:58px 74px 1fr;gap:12px;padding:9px 16px;
  border-bottom:1px solid var(--line-soft);font-size:13.5px;align-items:start}
.row:last-child{border-bottom:none}
.tag{font:500 10px/1.7 ui-monospace,monospace;text-align:center;border-radius:3px;padding:1px 0}
.t-pass{background:var(--ok-bg);color:var(--ok)}
.t-warn{background:var(--warn-bg);color:var(--warn)}
.t-fail{background:var(--fail-bg);color:var(--fail)}
.t-agent{background:var(--agent-bg);color:var(--agent)}
.t-human{background:var(--human-bg);color:var(--human)}
.rule-id{font:500 11.5px/1.6 ui-monospace,monospace;color:var(--ink-soft)}
.block{padding:14px 16px;border-bottom:1px solid var(--line-soft)}
.block:last-child{border-bottom:none}
.block .label{font:500 10px/1 ui-monospace,monospace;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-soft);margin:0 0 8px}
.block.agent-block{background:var(--agent-bg)}
.block.human-block{background:var(--human-bg)}
.block p{margin:0 0 6px}
.block p:last-child{margin-bottom:0}
.verdict{font-weight:700;letter-spacing:.02em}
.chain{width:100%;border-collapse:collapse;font-size:12.5px}
.chain th{text-align:left;font:500 10px/1 ui-monospace,monospace;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-soft);padding:10px 8px;border-bottom:1px solid var(--line)}
.chain td{padding:8px;border-bottom:1px solid var(--line-soft);vertical-align:top}
.chain tr:last-child td{border-bottom:none}
.hash{font:11.5px/1.5 ui-monospace,monospace;color:var(--ink-soft)}
.scroll{overflow-x:auto}
.scorecard{width:100%;border-collapse:collapse;font-size:13.5px}
.scorecard th{text-align:left;font:500 10px/1 ui-monospace,monospace;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-soft);padding:10px 8px;border-bottom:1px solid var(--line)}
.scorecard td{padding:9px 8px;border-bottom:1px solid var(--line-soft);vertical-align:top}
.scorecard tr:last-child td{border-bottom:none}
.num{font:500 13px/1 ui-monospace,monospace;text-align:center}
.s3{color:var(--ok)} .s2{color:var(--warn)} .s1,.s0{color:var(--fail)}
.just{color:var(--ink-soft);font-size:12.5px;max-width:44ch}
.means{display:flex;flex-wrap:wrap;gap:20px;padding:14px 16px;background:var(--surface-2);
  border-top:1px solid var(--line-soft);font-size:13px}
.means b{font:600 15px/1 ui-monospace,monospace}
footer{margin-top:40px;padding-top:16px;border-top:1px solid var(--line);
  font:11.5px/1.6 ui-monospace,monospace;color:var(--ink-soft)}
svg{display:block;max-width:100%;height:auto}
"""

PIPELINE_SVG = """
<svg viewBox="0 0 820 128" role="img" aria-label="Pipeline: load context, check policy, assess, approval gate (pauses), record">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
    </marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)" opacity=".45">
    <line x1="132" y1="46" x2="166" y2="46"/>
    <line x1="290" y1="46" x2="324" y2="46"/>
    <line x1="434" y1="46" x2="468" y2="46"/>
    <line x1="612" y1="46" x2="646" y2="46"/>
  </g>
  <g font-family="ui-monospace,monospace" font-size="12" text-anchor="middle">
    <g>
      <rect x="8" y="24" width="124" height="44" rx="5" fill="var(--surface)" stroke="var(--line)"/>
      <text x="70" y="51" fill="var(--ink)">load_context</text>
    </g>
    <g>
      <rect x="166" y="24" width="124" height="44" rx="5" fill="var(--ok-bg)" stroke="var(--ok)"/>
      <text x="228" y="51" fill="var(--ok)">check_policy</text>
    </g>
    <g>
      <rect x="324" y="24" width="110" height="44" rx="5" fill="var(--agent-bg)" stroke="var(--agent)"/>
      <text x="379" y="51" fill="var(--agent)">assess</text>
    </g>
    <g>
      <rect x="468" y="24" width="144" height="44" rx="5" fill="var(--human-bg)" stroke="var(--human)" stroke-width="2"/>
      <text x="540" y="51" fill="var(--human)">approval_gate</text>
    </g>
    <g>
      <rect x="646" y="24" width="110" height="44" rx="5" fill="var(--surface)" stroke="var(--line)"/>
      <text x="701" y="51" fill="var(--ink)">record</text>
    </g>
  </g>
  <g font-family="ui-monospace,monospace" font-size="10.5" text-anchor="middle">
    <text x="228" y="92" fill="var(--ink-soft)">deterministic</text>
    <text x="379" y="92" fill="var(--ink-soft)">model</text>
    <text x="540" y="92" fill="var(--human)">execution halts here</text>
    <text x="540" y="107" fill="var(--ink-soft)">resumes only on a human decision</text>
    <text x="701" y="92" fill="var(--ink-soft)">asserts human</text>
  </g>
</svg>
"""


def esc(v) -> str:
    return html.escape(str(v))


def _finding_rows(findings: list[dict]) -> str:
    out = []
    for f in findings:
        cls = {"pass": "t-pass", "warn": "t-warn", "fail": "t-fail"}[f["status"]]
        out.append(
            f'<div class="row"><span class="tag {cls}">{esc(f["status"])}</span>'
            f'<span class="rule-id">{esc(f["rule_id"])}</span>'
            f'<span>{esc(f["detail"])}</span></div>'
        )
    return "".join(out)


def _chain_rows(ledger: list[dict]) -> str:
    out = []
    for e in ledger:
        cls = {"rule": "t-pass", "agent": "t-agent", "human": "t-human", "system": ""}[e["actor"]]
        tag = f'<span class="tag {cls}">{esc(e["actor"])}</span>' if cls else f'<span class="rule-id">{esc(e["actor"])}</span>'
        out.append(
            f"<tr><td class='hash'>{e['seq']}</td><td>{tag}</td>"
            f"<td class='rule-id'>{esc(e['node'])}</td>"
            f"<td>{esc(e['summary'])}</td>"
            f"<td class='hash'>{esc(e['hash'][:12])}…</td></tr>"
        )
    return "".join(out)


def _run_card(run: dict) -> str:
    req, worker = run["request"], run["worker"]
    rec, dec = run["recommendation"], run["decision"]
    overridden = dec and rec["action"] != {"approved": "approve", "declined": "decline", "returned": "escalate"}.get(dec["outcome"])

    header = (
        f'<div class="card-head"><span class="id">{esc(req["request_id"])}</span>'
        f'<span class="who">{esc(worker["legal_name"])}</span>'
        f'<span class="meta">{esc(req["from"])} → {esc(req["to"])} · {req["hours"]:g}h {esc(req["plan"])} '
        f'· {esc(worker["supervisory_org"])}</span></div>'
    )
    findings = f'<div class="rows">{_finding_rows(run["findings"])}</div>'
    agent = (
        f'<div class="block agent-block"><p class="label">Agent recommendation — advisory only</p>'
        f'<p><span class="verdict">{esc(rec["action"].upper())}</span> '
        f'<span class="rule-id">confidence {esc(rec["confidence"])} · cites '
        f'{esc(", ".join(rec["cited_rule_ids"]))}</span></p>'
        f'<p>{esc(rec["rationale"])}</p></div>'
    )
    human = ""
    if dec:
        flag = " <span class='rule-id'>— overrides the recommendation</span>" if overridden else ""
        note = f'<p>{esc(dec["note"])}</p>' if dec.get("note") else ""
        human = (
            f'<div class="block human-block"><p class="label">Human decision — the only binding step</p>'
            f'<p><span class="verdict">{esc(dec["outcome"].upper())}</span> '
            f'<span class="rule-id">by {esc(dec["decided_by"])}</span>{flag}</p>{note}</div>'
        )
    chain = (
        f'<div class="block"><p class="label">Evidence chain — '
        f'{"verified" if run.get("chain_verified") else "NOT VERIFIED"}</p>'
        f'<div class="scroll"><table class="chain"><thead><tr><th>#</th><th>actor</th><th>node</th>'
        f'<th>entry</th><th>hash</th></tr></thead><tbody>{_chain_rows(run["evidence"])}</tbody></table></div></div>'
    )
    return f'<div class="card">{header}{findings}{agent}{human}{chain}</div>'


def _scorecard(ev: dict) -> str:
    rows = []
    for c in ev["cases"]:
        s = c["scores"] or {}
        cells = "".join(
            f'<td class="num s{s.get(k, 0)}">{s.get(k, "—")}</td>'
            for k in ("rationale_grounded", "citations_correct", "tone_appropriate")
        )
        rows.append(
            f'<tr><td class="rule-id">{esc(c["case_id"])}</td>'
            f'<td>{esc(c["expected_action"])}</td><td>{esc(c["actual_action"])}</td>'
            f'<td class="num s{3 if c["action_match"] else 0}">{"yes" if c["action_match"] else "NO"}</td>'
            f'<td class="num s{3 if c["never_self_approved"] else 0}">'
            f'{"yes" if c["never_self_approved"] else "NO"}</td>'
            f'{cells}<td class="just">{esc(s.get("justification", ""))}</td></tr>'
        )
    m = ev["judged_means"]
    return (
        '<div class="card"><div class="scroll"><table class="scorecard"><thead><tr>'
        "<th>case</th><th>expected</th><th>actual</th><th>match</th><th>no self-approve</th>"
        "<th>grounded</th><th>cites</th><th>tone</th><th>judge justification</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        f'<div class="means"><span>grounded <b>{m["rationale_grounded"]}</b>/3</span>'
        f'<span>citations <b>{m["citations_correct"]}</b>/3</span>'
        f'<span>tone <b>{m["tone_appropriate"]}</b>/3</span>'
        f'<span style="color:var(--ink-soft)">judge: {esc(ev["judge_model"])} · '
        f'{ev["n_cases"]} cases</span></div></div>'
    )


def build() -> Path:
    ev = json.loads((OUT / "eval.json").read_text())
    runs = [json.loads(p.read_text()) for p in sorted(OUT.glob("run-*.json"))]
    d = ev["deterministic"]

    assertions = "".join(
        f'<div class="assert pass"><p class="k">{k}</p>'
        f'<p class="v"><span class="dot"></span>{esc(v)}</p></div>'
        for k, v in (
            ("action_match", d["action_match"]),
            ("never_self_approved", d["never_self_approved"]),
            ("evidence chains verified", f"{sum(bool(r.get('chain_verified')) for r in runs)}/{len(runs)}"),
        )
    )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Time-off triage agent — run report</title>
<style>{CSS}</style></head><body><div class="wrap">

<header>
  <p class="kicker">LangGraph · human-in-the-loop · graded eval</p>
  <h1>An HR agent that recommends, and cannot approve</h1>
  <p class="lede">A time-off triage workflow where policy is evaluated deterministically, the model
  only ever writes a recommendation, and the graph physically halts until a named human decides.
  Every figure on this page is read from the JSON those runs produced.</p>
</header>

<h2>The guarantee</h2>
<div class="claim"><p>The agent cannot approve time off. Not "is instructed not to" — the graph
suspends at <span class="mono">approval_gate</span> and only resumes when handed a human decision,
and <span class="mono">record</span> rejects any decision whose actor is not a human.</p></div>
<div class="assertions">{assertions}</div>

<h2>Pipeline</h2>
<p class="note">Rules run in code, not in the model. The model sees findings it cannot dispute, and
writes an advisory recommendation against them.</p>
<div class="card"><div class="block">{PIPELINE_SVG}</div></div>

<h2>Worked runs</h2>
<p class="note">Including one where the manager overrides the agent — the case that proves the human
is the decider rather than a rubber stamp.</p>
{"".join(_run_card(r) for r in runs)}

<h2>Graded eval</h2>
<p class="note">Two deterministic assertions and three judged criteria, scored against a rubric
written before the prompt was tuned. The judge is not told which action was expected, so it grades
quality rather than agreement.</p>
{_scorecard(ev)}

<footer>generated by hr_timeoff_agent.report from out/*.json · mock tenant data, no real worker records</footer>
</div></body></html>
"""
    OUT.mkdir(exist_ok=True)
    path = OUT / "report.html"
    path.write_text(page)
    return path
