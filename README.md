# hr-timeoff-agent

A time-off triage agent that **recommends but cannot approve**, with a
tamper-evident evidence trail and a graded eval.

The interesting part is not that an LLM can read a leave policy. It is the
boundary: policy is evaluated in code, the model only ever writes an advisory
recommendation, and the graph physically suspends until a named human decides.

```
load_context → check_policy → assess → approval_gate ⏸ → record
                deterministic   model    halts here      asserts human
```

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
EV-04   decline   decline   yes    yes              grounded 1 · cites 3 · tone 3
EV-05   escalate  escalate  yes    yes              grounded 3 · cites 3 · tone 3
```

EV-04 scoring 1 on groundedness is the eval doing its job: the rationale claims
there is "no accrual path" to close an 80-hour shortfall, but no accrual rate
appears anywhere in the findings. The conclusion is still right — it follows from
the balance rule alone — but one clause is unsupported, and a rubric that scored
it 3 would not be measuring anything.

The judge is never told which action was expected, so it grades quality rather
than agreement. Set `HR_AGENT_JUDGE_MODEL` to a different model than
`HR_AGENT_MODEL` if you want a hard guarantee that nothing marks its own work.

## Offline mode

Model calls are content-addressed against `fixtures/llm_cache.json`, keyed on a
hash of the exact prompt. Offline is automatic when no API key is present.

The shipped fixtures are **authored stand-ins, not captured responses** — they
exist so the demo runs on a fresh clone. `scripts/seed_fixtures.py` shows exactly
how they were produced, and `--record` replaces any of them with a real call.

## Layout

```
hr_timeoff_agent/
  models.py     Recommendation vs Decision — the boundary, in types
  policy.py     deterministic rules over data/policy.json
  evidence.py   append-only hash-chained ledger
  graph.py      the LangGraph workflow and the approval interrupt
  evals.py      graded eval and the LLM judge
  report.py     builds docs/report.html from real run output
  cli.py
data/           mock Workday-shaped tenant: workers, absences, policy, requests
evals/          rubric.md (written first) and cases.json
tests/          the guarantees, including tamper detection and human override
docs/report.html  rendered run report (committed; every figure read from out/*.json)
```

All tenant data is fabricated. No real worker records are involved.
