# hr-timeoff-agent

A time-off triage agent that **recommends but cannot approve**, with a
tamper-evident evidence trail and a graded eval.

The interesting part is not that an LLM can read a leave policy. It is the
boundary: policy is evaluated in code, the model only ever writes an advisory
recommendation, and the graph physically suspends until a named human decides.

```
load_context → check_policy → retrieve → assess → approval_gate ⏸ → record
                deterministic   hybrid RAG  model    halts here      asserts human
```

**[Interactive architecture diagram →](docs/architecture.html)** — every box labelled by what it
actually is (deterministic rules engine, hybrid retrieval, Qdrant vector store, context
assembly, LLM call, checkpointer, approval interrupt, evidence ledger, eval harness with RAGAS,
judge), with source links into this repo.

## Run it

No API key required. The repo ships with a fixture cache so a fresh clone works
offline.

```bash
python -m venv .venv && ./.venv/bin/pip install -e .

./.venv/bin/python -m hr_timeoff_agent list
./.venv/bin/python -m hr_timeoff_agent run REQ-2004
./.venv/bin/python -m hr_timeoff_agent run REQ-2004 --approve --as "Aiko Tanaka" --json
./.venv/bin/python -m hr_timeoff_agent eval
./.venv/bin/python -m hr_timeoff_agent report   # → docs/report.html

# RAGAS eval of the retrieval step (optional extra)
./.venv/bin/pip install -e '.[rag-eval]'
./.venv/bin/python -m hr_timeoff_agent rag-eval
```

`run REQ-2004` with no decision flag stops at the approval gate and commits
nothing. That is the whole point — there is no flag that makes the agent decide.

To call the model live instead of replaying fixtures (only uncached calls go
out; `--record` forces fresh ones):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./.venv/bin/python -m hr_timeoff_agent run REQ-2001 --record
```

## The web app

The same agent, graph and guarantees, as an app people use in a browser:

```bash
./.venv/bin/pip install -e '.[web]'
export ANTHROPIC_API_KEY=sk-ant-...     # optional: without it, seeded requests still work
./.venv/bin/python -m hr_timeoff_agent web
# → http://127.0.0.1:8000
```

| Who | What they can do |
|---|---|
| **Employee** (e.g. Priya Raman) | File a request; the agent triages it live and routes it to their manager |
| **Manager** (Dana Whitfield, Aiko Tanaka) | See their reports' requests with the recommendation, policy checks and cited passages; approve, decline or return |
| **HR partner + admin** (Grace Kim) | See every request and the HR-only guidance managers can't; edit policy; reset the tenant |

- **Permissions are enforced on the server**, in the same code the CLI uses.
  Anyone may *try* to decide a request; unless they are the requester's direct
  manager the graph refuses, says why, and the request stays pending. People
  can't view requests that aren't theirs, their reports' or (for HR) anyone's.
- **Decisions have real effects.** An approval deducts the balance and records
  the absence, so the next request's coverage check sees it. Balances can't go
  negative (HB-2.1): the paid part is capped at the balance and the rest is
  recorded as unpaid leave (HB-3.1), explicitly.
- **Durable.** Paused runs live in SQLite, so a pending approval survives a
  restart.
- **Spend control.** New requests call the model; seeded ones replay at no cost.
  `HR_WEB_DAILY_CAP_USD` (default 2.00) caps a day's model spend and
  `HR_WEB_TRIAGES_PER_HOUR` (default 5) limits each person. Refused triage is
  kept and can be retried. Spend is recorded per call and shown on the admin
  page.
- **State lives outside the repo**, in `./var` (or `HR_WEB_HOME`): a working copy
  of the tenant and caches. The committed data and fixtures are never written.
  Admin → Reset restores the seeded state.
- **Sign-in is a persona picker** over the synthetic directory, behind an
  `IdentityProvider` interface (`hr_timeoff_agent/web/identity.py`); real SSO
  replaces that file only, since every check keys on the worker id it returns.
  The identity cookie is HMAC-signed; set `HR_WEB_SECRET` to keep sessions
  across restarts.
- **Local by default:** it binds to 127.0.0.1. All data is synthetic.

`tests/test_web.py` drives every flow above through HTTP as different people,
and `e2e` includes a web check.

## The guarantee, and how it is enforced

"The human is in the loop" is usually a prompt instruction, which is to say a
preference. Here it is three separate mechanisms, any one of which would stop an
autonomous approval:

1. **The graph suspends.** `approval_gate` calls LangGraph's `interrupt()`. There
   is no path from `assess` to `record` that does not pass through a resume
   carrying a human decision.
2. **The type system refuses.** `Recommendation` has no outcome field.
   `Decision.actor_type` is pinned to `Literal["human"]`, so an agent-authored
   decision fails validation before it reaches anything.
3. **The recorder re-checks.** `record` asserts the decision names a human
   approver and re-verifies the evidence chain before committing.

A human decision is not enough on its own: it has to be the *right* human. The
approval rule lives in `data/policy.json` (`"approver": "direct_manager"`, no
self-approval) and is checked deterministically, twice — at the gate and again
in `record`. The approver is a worker id from the directory; the name on the
decision comes from the directory, not from whatever was typed. An unauthorized
resume does not fail the run: the gate pauses again with the reason, the
refused attempt goes into the evidence trail, and the real manager can still
decide.

```
$ run REQ-2001 --approve --as "Marcus Vogel"
  REFUSED  Approval refused: Marcus Vogel (W-100235) is not Priya Raman's direct manager (W-100001).
  Nothing was recorded; the request is still awaiting Dana Whitfield.
```

`tests/test_guarantees.py` tests all of this — a peer, a self-approval and an
unknown id are refused, the recorder rejects an unauthorized decision and a
tampered chain on its own — plus the case that matters most:

```bash
./.venv/bin/python tests/test_guarantees.py
```

### One command checks everything

```bash
./.venv/bin/python -m hr_timeoff_agent e2e
```

`e2e` is a self-test built into the product: every claim above becomes a
pass/fail check, with no manual steps. The agent never decides; every CLI path
returns the right exit code and message; a refused approval does not jam the
run; editing any evidence entry is detected; another tenant's passages and
HR-only guidance are never retrieved; every citation was actually given to the
agent; a policy change is a data change; the graded eval and the RAGAS floors
pass. With `ANTHROPIC_API_KEY` set it adds a real agent call and the
model-graded RAGAS metrics, recorded into a scratch cache so the repo never
changes. `--require-live` fails if those can't run. Results go to
`out/e2e.json`; the exit code is 0 only if every check passed.

CI (`.github/workflows/ci.yml`) runs the guarantees, the retrieval isolation
tests, the graded eval and the RAGAS eval on every pull request, on Python 3.10
and 3.12, fully offline from the committed fixtures, then `e2e`. The RAGAS step
gates each metric separately, with floors just under the committed baseline. A
separate `live` job runs `e2e --require-live` against the real API; it is manual
only (Actions → CI → Run workflow) and needs an `ANTHROPIC_API_KEY` repository
secret. Its first run passed all 11 checks, including a real agent call through
the API (paused for a human; on Opus 5.5 at the time) and the model-graded
RAGAS metrics.

## Evidence trail

Every step appends to a hash-chained ledger — each entry commits to the one
before it, so editing a recorded step invalidates that entry and everything
after it.

```
[5] rule   check_policy   CON-01 PASS: 5 consecutive days is within the 15-day ceiling.
[6] agent  assess         Recommended APPROVE (confidence high).
[7] human  approval_gate  Dana Whitfield APPROVED (agent had recommended approve).
[8] system record         Committed approved to the worker record.
```

Actors are distinct (`rule` / `agent` / `human` / `system`), so "what did the
model decide" is answerable after the fact — and the answer is always *nothing*.
When a manager overrides, the ledger says so explicitly.

## Policy lives in data, not prompts

`data/policy.json` holds five rules — balance, notice, restricted periods, team
coverage, consecutive-day ceiling — each with a severity. `policy.py` evaluates
them in plain Python and emits `Finding` objects.

The model never decides whether a rule passed. It receives findings as ground
truth and reasons about what they mean together. That keeps policy outcomes
reproducible and explainable without reference to a model version, and it means
a policy change is a data change.

## Retrieval: guidance, not rule outcomes

Rules still decide whether a request is allowed. After `check_policy`, the
`retrieve` node looks up what a manager would check next: the tenant's leave
handbook (`data/handbook.json`) and past human decisions on similar requests
(`data/precedents.json`). The recommendation cites them by id — for example,
the unpaid-leave route for a balance shortfall, or how a wedding during close
was handled last year.

- **Vector store: Qdrant.** It runs in-process (`:memory:`), so a fresh clone
  needs no server; the same client API talks to a Qdrant server in production.
- **Hybrid search.** A dense embedding (`BAAI/bge-small-en-v1.5` via fastembed)
  plus BM25 for exact policy terms, fused with reciprocal rank fusion. Dense
  alone kept ranking generic PTO passages first. Fusion runs in Python with ties
  broken by id, because Qdrant's built-in fusion ordered tied scores differently
  on macOS and Linux.
- **Tenant and audience are enforced inside the query, before ranking.** Every
  sub-query carries the filter, so a passage from another tenant, or an HR-only
  passage when the reader is a manager, is never a candidate. The tenant comes
  from the loaded tenant, never from the model. `tests/test_retrieval.py` proves
  both, using a query copied from the other tenant's handbook.
- **Recorded.** The ledger gets a `retrieve` entry with the query, the filter
  and every result's score.

## Eval

`evals/rubric.md` was written before the prompt was tuned, and deliberately not
revised afterward. Two deterministic assertions, three judged criteria (0–3),
reported separately — a blended score would let a high average hide an
`action_match` failure.

The agent runs on **Claude Sonnet 5.5**; the judge is **Claude Opus 5.5**, a
different and stronger model, so nothing grades its own output. Recorded:

```
case    expected  actual    match  no-self-approve  scores
EV-01   approve   approve   yes    yes              grounded 3 · cites 3 · tone 3
EV-02   escalate  escalate  yes    yes              grounded 2 · cites 2 · tone 2
EV-03   escalate  escalate  yes    yes              grounded 3 · cites 3 · tone 2
EV-04   escalate  escalate  yes    yes              grounded 3 · cites 3 · tone 2
EV-05   escalate  escalate  yes    yes              grounded 2 · cites 2 · tone 3
means                                               grounded 2.6 · cites 2.6 · tone 2.4
```

Moving the agent from Opus 5.5 to Sonnet 5.5 (half the price) kept every action
correct and raised groundedness from 2.2 to 2.6; citations went from 2.8 to 2.6.
The judge caught one substantive slip worth knowing about: on EV-05 the agent
offered "approve with a documented backup plan" although coverage is 0%, where
HB-6.1 requires on-call cover from an adjacent team first. That is a policy
misreading, not a style point — the kind of thing the eval exists to catch.

What the first real run (on Opus 5.5) found, and what was done about it:

- **EV-04 was relabeled, in the open.** The model escalated an 80-hour
  shortfall the eval expected it to decline. The decline label predates
  retrieval; with the unpaid-leave route (HB-3.1) and HR Partner review (HB-7.1)
  now in its inputs, there is a decision for people to make. The label changed
  to escalate with a dated `relabeled` record in `evals/cases.json` saying who,
  when and why — not silently.
- **The judge had a blind spot.** It only saw findings and passages, so it
  marked claims from the request itself (the worker's note, the start date) as
  unsupported. It now sees the agent's input verbatim; a dated rubric addendum
  records the change. The deductions left are real: EV-02 assumes today's date,
  EV-04 assumes an 8-hour day, EV-05 calls the team "engineers".
- **The model ignores the length limit.** The prompt asks for two or three
  sentences; real rationales run to four or five, and lose a tone point. Left as
  a finding rather than tuned away.

### RAGAS: did retrieval do its job?

`rag-eval` scores the retrieval step with the RAGAS library, separately from the
graded eval. Reference passages and answers (`evals/rag_cases.json`) were
committed before retrieval was tuned.

```
case    id_context_precision  id_context_recall   retrieved
EV-01   0.2                   1.0                 HB-1.1, HB-7.1, HB-4.1, P-103, P-108
EV-02   0.8                   1.0                 HB-4.1, HB-5.1, HB-4.2, P-101, P-108
EV-03   0.6                   1.0                 HB-4.1, HB-6.1, HB-4.2, P-102, P-101
EV-04   0.8                   0.8                 HB-3.1, HB-7.1, HB-2.1, P-107, P-104
EV-05   0.6                   1.0                 HB-6.1, HB-8.1, HB-3.1, P-106, P-105
means   0.6                   0.96
```

The ID-based metrics need no model and run offline. With `ANTHROPIC_API_KEY`
set, RAGAS also scores **faithfulness** (is every claim in the rationale
supported by what the agent was given) and **context recall** against the
reference answer, graded by `claude-sonnet-5` so the grader is not the agent's
model. They call the API directly, so they need `ANTHROPIC_API_KEY`. First live
run, 2026-10-01 ([CI run 36826341416](https://github.com/gpatwa/hr-timeoff-agent/actions/runs/36826341416)):

```
faithfulness     0.778   about 1 claim in 5 is not directly supported by the agent's inputs
context_recall   0.9     retrieval covers most of what the reference answers need
```

Faithfulness agrees with the judge: the unsupported claims are the same
assumptions it marked down (an assumed current date, an assumed 8-hour day).
Five cases, so read these as a demo-scale measurement, not a benchmark. They
were measured with Opus 5.5 as the agent; the manual `live` job re-measures them
for whichever model is current.

What the numbers say: recall is high, precision is low on the easy case. A
fixed five results is wasteful when every rule passes (EV-01 needs one), and
EV-04 misses the closest shortfall precedent (P-103). The next change would be a
score threshold or a reranker, measured against these same references.

The judge is never told which action was expected, so it grades quality rather
than agreement. Set `HR_AGENT_JUDGE_MODEL` to a different model than
`HR_AGENT_MODEL` if you want a hard guarantee that nothing marks its own work.

## Offline mode

Model calls are content-addressed against `fixtures/llm_cache.json`, keyed on a
hash of the exact prompt. Offline is automatic when no API key is present. In
live mode a cached response is still replayed; only a miss calls the model, and
`--record` forces fresh calls.
Embeddings work the same way: `fixtures/embeddings.json` holds every dense and
BM25 vector the demo needs, so a fresh clone never downloads an embedding model.
Set `HR_AGENT_OFFLINE=1` to make any uncached embedding an error instead of a
local fastembed call.

The shipped fixtures are **real model responses** (agent: Claude Sonnet 5.5,
judge: Claude Opus 5.5), recorded through
headless Claude Code rather than the API (each entry says `"source":
"claude-code-cli"` and which model served it). That is the same model, prompt
and output schema, but Claude Code manages thinking and effort itself, so it is
a close stand-in for the API call, not a byte-identical one. To re-record:

```bash
HR_AGENT_BACKEND=claude-cli ./.venv/bin/python -m hr_timeoff_agent eval --record
```

`HR_AGENT_BACKEND=claude-cli` uses whatever Claude Code is logged in with, such
as a personal subscription, and is meant for recording fixtures locally. A
deployed service uses the default `api` backend with an API key. The backend
refuses to cache a response served by any model other than the one asked for.

`scripts/seed_fixtures.py` holds the authored stand-ins the repo shipped with
before the first real run; it only fills keys with no recording and never
overwrites one.

## Layout

```
hr_timeoff_agent/
  models.py     Recommendation vs Decision — the boundary, in types
  policy.py     deterministic rules over data/policy.json
  retrieval.py  hybrid retrieval in Qdrant, tenant/audience filtered before ranking
  evidence.py   append-only hash-chained ledger
  graph.py      the LangGraph workflow and the approval interrupt
  evals.py      graded eval and the LLM judge
  rag_eval.py   RAGAS eval of the retrieval step
  e2e.py        end-to-end self-test: every guarantee as a pass/fail check
  web/          the browser app: workspace (state + rules), identity, routes, pages
  report.py     builds docs/report.html from real run output
  cli.py
data/           mock Workday-shaped tenant: workers, absences, policy, requests,
                leave handbook and past decisions (plus a second tenant, for isolation tests)
evals/          rubric.md (written first), cases.json, rag_cases.json
tests/          the guarantees, including tamper detection and human override,
                and retrieval isolation by tenant and audience
docs/report.html        rendered run report (every figure read from out/*.json)
docs/architecture.html  interactive component diagram (archify; source-linked)
```

All tenant data is fabricated. No real worker records are involved.
