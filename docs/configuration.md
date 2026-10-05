# Configuration reference

Every environment variable the code reads, with its default. A variable not listed here is not read.
`tests/test_docs.py` fails if the code starts reading one that is missing from this page.

**Secrets from files.** For `HR_DATABASE_URL`, `HR_WEB_SECRET`, `HR_A2A_SECRET`, `HR_OIDC_CLIENT_SECRET`,
`HR_OIDC_SERVICE_CLIENT_SECRET` and `ANTHROPIC_API_KEY`, setting `NAME_FILE=/path/to/file` reads `NAME` from that
file at startup (for Docker and Kubernetes secrets). A `NAME` already set wins; a `NAME_FILE` that points at nothing
is an error. See `core/config.py`.

## Model and agent

| Variable | Default | What it does |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Key for the Anthropic API backend. Unset means offline (recorded fixtures only), unless the backend is `claude-cli`. |
| `HR_AGENT_BACKEND` | `api` | `api` (Anthropic SDK) or `claude-cli` (Claude Code, using its own login). |
| `HR_AGENT_OFFLINE` | unset | `1` forces offline: replay recordings, never call a model. A cache miss fails the run. |
| `HR_AGENT_MODE` | `single` | `single` (one assessment call) or `multi` (specialists and a coordinator). `--agents` on the CLI sets it. |
| `HR_AGENT_MODEL` | `claude-sonnet-5-5` | The model that writes the assessment. |
| `HR_AGENT_JUDGE_MODEL` | `claude-opus-5-5` | The model that grades outputs in the evals. Deliberately a different model from the agent's. |
| `HR_AGENT_EFFORT` | `medium` | The `effort` setting sent with each model request. |
| `HR_AGENT_TIMEOUT_S` | `90` | Per-attempt timeout for a model request. |
| `HR_AGENT_MAX_RETRIES` | `3` | Retries after a timeout, 429 or 5xx before the outage path takes over. |
| `HR_AGENT_FIXTURES` | `fixtures/llm_cache.json` | Where recorded model calls are read and written. Evals and the gate point it at a scratch copy. |

## Retrieval and embeddings

| Variable | Default | What it does |
|---|---|---|
| `HR_QDRANT_URL` | unset | A running Qdrant to use. Unset means an in-process index. |
| `HR_AGENT_EMBEDDINGS` | `fixtures/embeddings.json` | The recorded embeddings cache. Overridden for scratch runs. |
| `HR_AGENT_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | The dense embedding model. Changing it invalidates the recorded embeddings. |
| `HR_EMBED_LOCAL_ONLY` | unset | `1` makes the embedder use model files already on disk and never go to the network. Set in the container image. |
| `FASTEMBED_CACHE_PATH` | `/tmp/fastembed_cache` | Where the baked-in embedding model files are. Read only when `HR_EMBED_LOCAL_ONLY=1`. |
| `HR_CORPUS_DIR` | unset | Points the MCP tool server at a different handbook and precedents. Used by the hostile-corpus evals. |

## State and storage

| Variable | Default | What it does |
|---|---|---|
| `HR_DATABASE_URL` | unset | Postgres connection URL. Unset means SQLite and JSON files under the home directory. |
| `HR_DATABASE_SCHEMA` | `public` | Postgres schema. `auto` gives each home directory its own schema (how the tests share one database). |
| `HR_LOCK_TIMEOUT_S` | `120` | How long to wait for a migration or schema lock before failing. |
| `HR_WEB_HOME` | `./var` | State directory for the web app and `init`. |
| `HR_A2A_HOME` | `var/a2a` | State directory for the A2A agents. |

## Web app

| Variable | Default | What it does |
|---|---|---|
| `HR_WEB_SECRET` | random per start | Signs the session cookie. Required when OIDC is on, so sessions survive a restart. |
| `HR_WEB_DAILY_CAP_USD` | `2.0` | Model spend the web app allows per day before it refuses new triage. |
| `HR_WEB_TRIAGES_PER_HOUR` | `5` | Triage requests allowed per person per hour. |
| `HR_PUBLIC_URL` | `http://localhost:8000` | The address users reach the web app at; used to build the OIDC redirect. |

## Sign-in (OIDC)

Everything here is off until `HR_OIDC_ISSUER` is set. Unset, the web app has a persona picker and the A2A agents use demo tokens.

| Variable | Default | What it does |
|---|---|---|
| `HR_OIDC_ISSUER` | unset | The provider's issuer URL. Setting it turns OIDC on. |
| `HR_OIDC_INTERNAL_URL` | unset | A different address for server-to-server calls to the provider (for example inside a container network). |
| `HR_OIDC_CLIENT_ID` | `hr-web` | The web app's client id. |
| `HR_OIDC_CLIENT_SECRET` | unset | The web app's client secret. |
| `HR_OIDC_AUDIENCE` | `hr-a2a` | The audience A2A and MCP callers' tokens must carry. |
| `HR_OIDC_SERVICE_CLIENTS` | `timeoff-agent` | Comma-separated client ids allowed to call as a service. |
| `HR_OIDC_SERVICE_CLIENT_ID` | `timeoff-agent` | The id the time-off agent uses to fetch its own token. |
| `HR_OIDC_SERVICE_CLIENT_SECRET` | unset | The secret that goes with it. |

## A2A and MCP services

| Variable | Default | What it does |
|---|---|---|
| `HR_A2A_SECRET` | random per start | Derives the demo bearer tokens. Set it to keep them stable across restarts. |
| `HR_A2A_TIMEOFF_URL` | `http://<host>:<port>` | The URL on the time-off agent's card, when it is reached through a proxy or container network. |
| `HR_A2A_PAYROLL_URL` | `http://<host>:<payroll-port>` | The same, for the payroll agent. |
| `HR_A2A_PEER_TIMEOUT_S` | `10` | How long a review waits for the payroll peer before showing "payroll unavailable". |
| `HR_MCP_ALLOWED_HOSTS` | `127.0.0.1:*,localhost:*,[::1]:*` | Host headers the HTTP MCP server (`mcp --http`) accepts. Add the service name when it runs behind a container network. |

## Observability

Telemetry is off unless an OTLP endpoint is configured or `HR_OTEL=1`.

| Variable | Default | What it does |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Where traces and metrics go. Setting it turns telemetry on. |
| `HR_OTEL` | unset | `1` turns telemetry on even with no endpoint set (the OTLP exporters then use their default address). Needs the `otel` extra installed. |
| `OTEL_SERVICE_NAME` | the service's own name | Overrides the service name on spans and metrics. |
| `HR_VERSION` | `dev` | Reported as the service version. |
| `HR_OTEL_SAMPLE_RATIO` | `1.0` | Fraction of traces kept. |
| `HR_OTEL_METRIC_INTERVAL_MS` | `10000` | How often metrics are exported. |

## Evals and the gate

| Variable | Default | What it does |
|---|---|---|
| `HR_GATE_A2A_CAP_USD` | `2.0` | Spend cap for the gate's multi-agent run through the A2A agents. |
| `HR_AGENT_RAGAS_MODEL` | `claude-sonnet-5` | The grading model for the RAGAS metrics in `rag-eval`. |

The gate's own budget, floors and ceilings are in `evals/thresholds.json`, not the environment.
