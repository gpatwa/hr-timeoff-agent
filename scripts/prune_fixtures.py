"""Drop cached model responses that no current prompt can reach.

A cache key hashes the model, system prompt, user prompt and schema, so changing
any of them (a model switch, a prompt edit) orphans the old entries. This keeps
fixtures/llm_cache.json to exactly the responses the demo and evals replay.

Run: .venv/bin/python scripts/prune_fixtures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")

from hr_timeoff_agent.agent import graph as graph_mod
from hr_timeoff_agent.adapters import llm
from hr_timeoff_agent.core import policy  # noqa: E402
from hr_timeoff_agent.tooling import evals  # noqa: E402
from hr_timeoff_agent.core.models import Finding, JudgeScore, Passage, Recommendation  # noqa: E402


def live_keys() -> set[str]:
    tenant = policy.Tenant()
    keys = set()
    for case in json.loads((evals.EVAL_DIR / "cases.json").read_text()):
        request = tenant.requests[case["request_id"]]
        worker = tenant.workers[request["worker_id"]]
        state = graph_mod.build(tenant).invoke(
            graph_mod.initial_state(request),
            config={"configurable": {"thread_id": f"prune-{case['case_id']}"}},
        )
        findings = [Finding.model_validate(f) for f in state["findings"]]
        passages = [Passage.model_validate(p) for p in state["passages"]]
        rec = Recommendation.model_validate(state["recommendation"])
        keys.add(llm._key(llm.AGENT_MODEL, graph_mod.ASSESS_SYSTEM,
                          graph_mod.build_assess_prompt(request, worker, findings, passages), Recommendation))
        keys.add(llm._key(llm.JUDGE_MODEL, evals.JUDGE_SYSTEM,
                          evals.build_judge_prompt(request, worker, findings, passages, rec), JudgeScore))
    return keys


def main() -> int:
    keys = live_keys()
    cache = json.loads(llm.FIXTURES.read_text())
    dropped = sorted(v["label"] for k, v in cache.items() if k not in keys)
    kept = {k: v for k, v in cache.items() if k in keys}
    llm.FIXTURES.write_text(json.dumps(kept, indent=2, sort_keys=True) + "\n")
    print(f"kept {len(kept)}, dropped {len(dropped)}: {', '.join(dropped) or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
