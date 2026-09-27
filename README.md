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
actually is (deterministic rules engine, context assembly, LLM call, checkpointer, approval
interrupt, evidence ledger, eval harness, judge), with source links into this repo.

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

To use the live API instead of fixtures:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./.venv/bin/python -m hr_timeoff_agent run REQ-2001 --record
```

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

`tests/test_guarantees.py` tests all three, plus the case that matters most:

```bash
./.venv/bin/python tests/test_guarantees.py
```

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
  alone kept ranking generic PTO passages first.
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

```
case    expected  actual    match  no-self-approve  scores
EV-01   approve   approve   yes    yes              grounded 3 · cites 2 · tone 3
EV-02   escalate  escalate  yes    yes              grounded 3 · cites 3 · tone 3
EV-03   escalate  escalate  yes    yes              grounded 3 · cites 2 · tone 3
EV-04   decline   decline   yes    yes              grounded 3 · cites 3 · tone 2
EV-05   escalate  escalate  yes    yes              grounded 3 · cites 3 · tone 3
```

EV-04 is the case retrieval changed. Before it, the rationale claimed there was
"no accrual path" to close an 80-hour shortfall while no accrual rate appeared
anywhere in its inputs, and the judge scored it 1 on groundedness. Now the
accrual clause (HB-2.1) is retrieved and cited, so the same claim is grounded —
and the recommendation can offer the unpaid-leave route (HB-3.1) instead of a
flat no. It lost a point on tone instead: three options in one sentence. The
rubric gained a dated addendum saying passages count as sources; the original
wording is unchanged.

### RAGAS: did retrieval do its job?

`rag-eval` scores the retrieval step with the RAGAS library, separately from the
graded eval. Reference passages and answers (`evals/rag_cases.json`) were
committed before retrieval was tuned.

```
case    id_context_precision  id_context_recall   retrieved
EV-01   0.2                   1.0                 HB-1.1, HB-7.1, HB-4.1, P-103, P-108
EV-02   0.8                   1.0                 HB-5.1, HB-4.1, HB-4.2, P-101, P-104
EV-03   0.6                   1.0                 HB-4.1, HB-6.1, HB-4.2, P-102, P-101
EV-04   0.8                   0.8                 HB-3.1, HB-7.1, HB-2.1, P-107, P-104
EV-05   0.6                   1.0                 HB-6.1, HB-8.1, HB-3.1, P-106, P-105
means   0.6                   0.96
```

The ID-based metrics need no model and run offline. With `ANTHROPIC_API_KEY`
set, RAGAS also scores **faithfulness** (is every claim in the rationale
supported by what the agent was given) and **context recall** against the
reference answer, graded by `claude-sonnet-5` so the grader is not the agent's
model. Those two have not been run yet — the repo has only been exercised
offline.

What the numbers say: recall is high, precision is low on the easy case. A
fixed five results is wasteful when every rule passes (EV-01 needs one), and
EV-04 misses the closest shortfall precedent (P-103). The next change would be a
score threshold or a reranker, measured against these same references.

The judge is never told which action was expected, so it grades quality rather
than agreement. Set `HR_AGENT_JUDGE_MODEL` to a different model than
`HR_AGENT_MODEL` if you want a hard guarantee that nothing marks its own work.

## Offline mode

Model calls are content-addressed against `fixtures/llm_cache.json`, keyed on a
hash of the exact prompt. Offline is automatic when no API key is present.
Embeddings work the same way: `fixtures/embeddings.json` holds every dense and
BM25 vector the demo needs, so a fresh clone never downloads an embedding model.
Set `HR_AGENT_OFFLINE=1` to make any uncached embedding an error instead of a
local fastembed call.

The shipped fixtures are **authored stand-ins, not captured responses** — they
exist so the demo runs on a fresh clone. `scripts/seed_fixtures.py` shows exactly
how they were produced, and `--record` replaces any of them with a real call.

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
