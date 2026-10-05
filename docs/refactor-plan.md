# Refactor plan: from one flat package to enforced boundaries

Status: proposal. Nothing here is started. Each stage is one PR that leaves `main` green and the
behaviour unchanged: the same 40-case suite, the same fixtures, the same `e2e`.

## Where it stands

`hr_timeoff_agent/` is one flat package of about 7,500 lines. The ideas are modular (a deterministic
policy engine, an approval gate, MCP and A2A seams), but the code does not enforce them. Measured from
the real imports:

- `policy` imports only `models`. That is the shape we want everywhere.
- **Cycles:** `graph` imports `agents` and `agents` imports `graph`; `storage` and `migrations` import each other.
- **Wrong direction:** `a2a_server` imports `web` (an agent server depends on the web app's workspace);
  `agentloop` imports `mcp_server` (the agent runtime knows the tool server's internals).
- **Fan-in at the top:** `cli` imports 20 modules, `e2e` 15, `gate` 10. These are test and tooling code
  shipped inside the product package.
- **Process-global state:** `llm.FIXTURES`, `llm.before_live_call`, `llm.after_live_call`, `HR_AGENT_MODE`
  and `HR_CORPUS_DIR` are module or environment globals. The parallel gate needed per-thread metering and a
  locked cache to cope with them.

## Target layout

One repo, several packages, each with a one-way dependency direction:

```
core/            models, policy engine, evidence ledger          imports: nothing
ports/           interfaces: Model, Store, Retriever, Identity   imports: core
adapters/        anthropic, claude-cli, fixtures | sqlite, postgres | qdrant, memory | oidc, personas
agent/           graph, specialists, agent loop, MCP client      imports: core, ports
services/
  tools/         MCP tool server                                 imports: core, ports
  web/           FastAPI app + workspace                         imports: core, ports, agent
  a2a/           time-off agent + payroll peer                   imports: core, ports, agent
tooling/         evals, gate, e2e, report, rag-eval, CLI         imports: everything, shipped by nothing
```

Rules: `core` imports nothing; `ports` imports only `core`; `agent` never imports a service; no service
imports another service; nothing imports `tooling`. "Monorepo" here means one repository with
packages (a `uv` workspace is enough at this size). Bazel, Buck or Pants earns its cost only with a
second team or product.

## Stages

Order is by value per risk. Stages 1 to 3 are the ones worth doing for their own sake.

### Stage 0: a safety net (half a day)
- Record a baseline: `e2e`, `eval` in both modes, `gate --replay`, and the commit hash.
- Add a test that fails if any public CLI command or MCP tool schema changes (golden outputs).
- Exit: baseline numbers saved; golden test green.

### Stage 1: make the rules executable (half a day)
- Add `import-linter` with the contracts above, starting with only the ones that already hold
  (`policy` and `models` import nothing else; nothing imports `cli`, `gate`, `e2e`, `report`, `rag_eval`).
- Known violations go in an explicit allow-list with an owner stage each; the list can only shrink.
- Exit: CI fails on a new bad import. No code moved.

### Stage 2: break the two cycles (1 day)
- `graph` and `agents`: move the node factory contract into a small `nodes` module that both import, so
  `graph` no longer imports `agents`.
- `storage` and `migrations`: move the lock and connection helper to one module both use.
- Exit: the allow-list loses those entries; behaviour unchanged.

### Stage 3: ports for the four things that vary (2 days)
- `Model` (structured call, agent call, cost) with Anthropic, Claude Code CLI and fixture adapters.
- `Store` (requests, paused runs, spend, ledger) with SQLite and Postgres adapters.
- `Retriever` and `Identity` likewise.
- Replace `llm.FIXTURES`, the call hooks and the env switches with objects passed in at construction.
  The gate's `Meter` becomes a wrapper around a `Model`, not a monkey-patched hook.
- Exit: no module-level mutable state in `llm`; the parallel gate no longer needs its thread-local
  and file-lock workarounds in `llm`; all tests unchanged.

### Stage 4: separate what ships from what tests (1 day)
- Move `gate`, `evals`, `e2e`, `report`, `rag_eval` and the eval data into `tooling/`.
- Split `cli.py` into one small module per command group; the product CLI keeps `run`, `web`, `a2a`, `mcp`, `migrate`.
- Exit: the Docker image no longer contains tooling; `tooling` depends on the product, never the reverse.

### Stage 5: services depend on libraries, not on each other (1 to 2 days)
- Move the workspace (authorization, spend cap, paused runs) out of `web/` into `agent/` or its own
  library, so `a2a` and `web` are two front doors on it, as the architecture diagram already claims.
- Move `agentloop`'s use of `mcp_server` behind an interface so the agent runtime depends on a tool
  contract, not on the server.
- Exit: `services/a2a` no longer imports `services/web`.

### Stage 6: packages and contracts (1 to 2 days)
- Turn each box above into a package in a `uv` workspace with its own `pyproject.toml` and extras.
- Publish the MCP tool schemas and A2A agent cards as versioned files; each side tests against them.
- One image per service built from its own package.
- Exit: `pip install` of one package pulls only its dependencies; CI builds and tests changed packages only.

### Stage 7, only if a second team or product appears
- Move to Bazel, Buck or Pants for cached, affected-target builds; add CODEOWNERS per package.

## Risks and how each stage contains them
- **Behaviour drift:** every stage runs `e2e`, both evals and `gate --replay` against the Stage 0 baseline.
  The two live runs (`gate --quick`, then the full gate) happen after Stage 3 and Stage 5.
- **Fixture invalidation:** fixtures are keyed by model, prompt and tool list. Moving code must not change
  any prompt or tool schema; the golden test in Stage 0 enforces that.
- **Big-bang temptation:** if a stage grows past about 600 changed lines of logic, split it.

## What not to do
- Do not start with the directory moves. Without Stage 1's lint they drift back.
- Do not add a plugin or dependency-injection framework. Plain constructors are enough at this size.
- Do not adopt Bazel before Stage 7's trigger.
