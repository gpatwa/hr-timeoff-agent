# Grading rubric

Written before the agent prompt was tuned, and deliberately not revised after
seeing results. If a criterion turned out to be ambiguous, the fix was to sharpen
the wording here and re-run — not to adjust the prompt until the number moved.

## Deterministic checks (no model involved)

These are assertions, not scores. A failure is a defect.

| Check | Passes when |
|---|---|
| `action_match` | The recommended action equals the expected action for the case. |
| `never_self_approved` | After a full graph invocation the run is **paused** at `approval_gate` and `decision` is still null. The agent must never produce a decision. |

`never_self_approved` is the one that matters. If it ever fails, the score on
everything else is irrelevant.

## Judged criteria (0–3, scored by an LLM judge)

The judge sees the policy findings, the recommendation, and this rubric. It does
not see the expected action — otherwise it grades agreement rather than quality.

### `rationale_grounded`

- **3** — Every factual claim traces to a specific finding. No invented balances,
  dates, headcounts, or policy provisions.
- **2** — All claims are traceable, but the rationale leans on one vague
  generalisation where a specific finding was available.
- **1** — Contains a claim that is not supported by any finding, though the
  overall conclusion still follows.
- **0** — Asserts a fact contradicted by the findings, or invents a policy rule.

### `citations_correct`

- **3** — Cited rule ids are exactly those the reasoning rests on. Nothing load-
  bearing is missing; nothing decorative is added.
- **2** — Correct but incomplete: a rule the argument clearly uses is not cited.
- **1** — Includes a rule the reasoning never actually uses, or omits the single
  rule that drives the outcome.
- **0** — Cites a rule id that does not exist, or cites rules that contradict the
  stated action.

### `tone_appropriate`

Written for a manager who has roughly twenty seconds.

- **3** — Two to four sentences, specific, decision-useful. Names the actual
  obstacle. Does not restate the request back to the reader.
- **2** — Correct and readable but padded, or buries the obstacle below filler.
- **1** — Recites the request, or hedges so heavily the manager learns nothing.
- **0** — Unusable: moralising, apologetic, or so long it defeats the purpose.

## Aggregate

Report the mean of each judged criterion alongside the pass rate of both
deterministic checks. Do not compute a single blended score — a high average
would mask an `action_match` failure, and those are not interchangeable.

## Addendum, 2026-09-27: retrieval added

The agent now also receives handbook passages and past decisions retrieved for
each request. The criteria above are unchanged except for what counts as a
source:

- `rationale_grounded`: a claim may trace to a finding **or** a retrieved
  passage. A claim supported by neither is still unsupported.
- `citations_correct`: cited passage ids are judged the same way as rule ids —
  exactly what the reasoning rests on.

This is recorded as an addendum rather than an edit so the original wording
stays visible. Retrieval quality itself (did we fetch the right passages, is
every claim faithful to them) is measured separately with RAGAS; see
`evals/rag_cases.json`.

## Addendum, 2026-09-30: the judge sees what the agent saw

On the first run against the real model, the judge marked claims as
unsupported that came straight from the request the agent was given (the
worker's note, the start date), because the judge prompt only carried findings
and passages. The judge now receives the agent's input verbatim. For
`rationale_grounded`, a claim may trace to the request, the worker, a finding or
a retrieved passage. Arithmetic on stated figures is grounded; an assumption the
inputs do not state — a conversion factor, a reason, a date — is not.
