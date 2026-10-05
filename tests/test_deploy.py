"""What a deployment relies on: file secrets, migrations, readiness, a safe first start.

The migration and shared-start tests need Postgres and run when HR_DATABASE_URL is set
(CI's durable job); the rest run anywhere.

Run: .venv/bin/python tests/test_deploy.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HR_AGENT_OFFLINE", "1")
PG = os.environ.get("HR_DATABASE_URL")
if PG:
    os.environ["HR_DATABASE_SCHEMA"] = "auto"
for var in ("HR_OIDC_ISSUER",):
    os.environ.pop(var, None)

from fastapi.testclient import TestClient  # noqa: E402

from hr_timeoff_agent.core import config  # noqa: E402
from hr_timeoff_agent.cli import main  # noqa: E402
from hr_timeoff_agent.services.web.app import create_app  # noqa: E402
from hr_timeoff_agent.agent.workspace import Workspace  # noqa: E402


def import_helpers():
    """test_oidc clears HR_DATABASE_URL when imported (it wants file mode); put it back."""
    saved = os.environ.get("HR_DATABASE_URL")
    import test_a2a, test_oidc  # noqa: F401,E401

    if saved is not None:
        os.environ["HR_DATABASE_URL"] = saved


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


def need_pg(fn):
    def run():
        if not PG:
            print("    (HR_DATABASE_URL not set: skipped)")
            return
        fn()

    run.__name__ = fn.__name__
    return run


# ── secrets from files ──────────────────────────────────────────────────────

def test_a_secret_can_come_from_a_file_and_an_explicit_value_wins():
    d = Path(tempfile.mkdtemp())
    (d / "web").write_text("from-the-file\n")
    env = {"HR_WEB_SECRET_FILE": str(d / "web"), "HR_A2A_SECRET_FILE": str(d / "web"), "HR_A2A_SECRET": "explicit"}
    assert sorted(config.load_file_secrets(env=env)) == ["HR_WEB_SECRET"]
    assert env["HR_WEB_SECRET"] == "from-the-file", "trailing newline is stripped"
    assert env["HR_A2A_SECRET"] == "explicit", "a value that is already set is not replaced"


def test_a_secret_file_that_is_missing_or_empty_is_an_error_not_an_empty_secret():
    d = Path(tempfile.mkdtemp())
    (d / "empty").write_text("\n")
    e = raises(config.MissingSecret, config.load_file_secrets, env={"HR_WEB_SECRET_FILE": str(d / "nope")})
    assert "HR_WEB_SECRET_FILE" in str(e) and "from-the-file" not in str(e)
    raises(config.MissingSecret, config.load_file_secrets, env={"HR_WEB_SECRET_FILE": str(d / "empty")})


def test_the_cli_reads_file_secrets_before_anything_else_and_fails_clearly():
    os.environ["HR_WEB_SECRET_FILE"] = "/nonexistent/secret"
    os.environ.pop("HR_WEB_SECRET", None)
    try:
        assert main(["list"]) == 2
    finally:
        os.environ.pop("HR_WEB_SECRET_FILE")


# ── migrations ──────────────────────────────────────────────────────────────

def _schema() -> tuple[str, str]:
    from hr_timeoff_agent.adapters.storage import ensure_schema

    schema = "mig_" + uuid.uuid4().hex[:10]
    ensure_schema(PG, schema)
    return PG, schema


def _exists(url, schema, table) -> bool:
    import psycopg

    with psycopg.connect(url) as c:
        return c.execute("select to_regclass(%s)", (f"{schema}.{table}",)).fetchone()[0] is not None


def test_saving_embeddings_keeps_what_another_process_added_meanwhile():
    """The MCP server Claude Code starts while recording writes the same cache file. A save that wrote
    only this process's copy back deleted its entries, and a later offline replay then failed."""
    import json

    from hr_timeoff_agent.adapters import retrieval

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "embeddings.json"
        path.write_text(json.dumps({"a": {"vector": [1.0]}}))
        mine = retrieval.Embedder(cache_path=path)
        path.write_text(json.dumps({"a": {"vector": [1.0]}, "child": {"vector": [2.0]}}))   # the other process
        mine._cache["mine"] = {"vector": [3.0]}
        mine._dirty = True
        mine.save()
        assert set(json.loads(path.read_text())) == {"a", "child", "mine"}
        assert not list(Path(d).glob("*.tmp"))


def test_embeddings_still_work_when_the_cache_file_cannot_be_written():
    """In a container the committed cache is read-only for the unprivileged user."""
    import json
    import stat

    from hr_timeoff_agent.adapters import retrieval

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "embeddings.json"
        path.write_text(json.dumps({}))
        e = retrieval.Embedder(cache_path=path)
        e._cache["k"] = {"vector": [0.1]}
        e._dirty = True
        path.chmod(stat.S_IRUSR)
        Path(d).chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            e.save()  # must not raise
        finally:
            Path(d).chmod(stat.S_IRWXU)
        assert e._dirty is False and e._cache["k"] == {"vector": [0.1]}


def test_local_only_mode_points_fastembed_at_the_baked_in_models():
    from hr_timeoff_agent.adapters import retrieval

    saved = {k: os.environ.get(k) for k in ("HR_EMBED_LOCAL_ONLY", "FASTEMBED_CACHE_PATH")}
    try:
        os.environ.pop("HR_EMBED_LOCAL_ONLY", None)
        assert retrieval._model_source() == {} and retrieval._sparse_source() == {}, "off by default: normal behaviour on a laptop"
        with tempfile.TemporaryDirectory() as d:
            snap = Path(d) / "models--Qdrant--bm25" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            os.environ.update({"HR_EMBED_LOCAL_ONLY": "1", "FASTEMBED_CACHE_PATH": d})
            assert retrieval._model_source() == {"local_files_only": True}
            assert retrieval._sparse_source() == {"specific_model_path": str(snap), "local_files_only": True}
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@need_pg
def test_migrations_apply_once_in_order_and_are_recorded():
    from hr_timeoff_agent.adapters import migrations as m

    url, schema = _schema()
    assert m.migrate(url, schema) == [1] and m.migrate(url, schema) == []
    more = [*m.MIGRATIONS, (2, "add a table", "create table extra_a (id int)"), (3, "and another", "create table extra_b (id int)")]
    assert m.migrate(url, schema, more) == [2, 3]
    st = m.status(url, schema, more)
    assert (st.applied, st.pending, st.current, st.latest) == ([1, 2, 3], [], 3, 3)
    assert _exists(url, schema, "extra_a") and _exists(url, schema, "extra_b")


@need_pg
def test_a_failing_migration_rolls_back_completely_and_stops():
    from hr_timeoff_agent.adapters import migrations as m

    url, schema = _schema()
    bad = [*m.MIGRATIONS, (2, "half then boom", "create table half_done (id int); select 1/0"), (3, "never reached", "create table after (id int)")]
    e = raises(m.MigrationFailed, m.migrate, url, schema, bad)
    assert e.version == 2
    assert not _exists(url, schema, "half_done"), "the statement before the failure was rolled back"
    assert not _exists(url, schema, "after")
    assert m.status(url, schema, bad).applied == [1], "only the good migration is recorded"


@need_pg
def test_processes_starting_together_apply_each_migration_exactly_once():
    from hr_timeoff_agent.adapters import migrations as m

    url, schema = _schema()
    slow = [*m.MIGRATIONS, (2, "slow", "create table slow_one (id int); select pg_sleep(0.5)")]
    applied, errors = [], []

    def go():
        try:
            applied.extend(m.migrate(url, schema, slow))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    assert sorted(applied) == [1, 2], f"each migration applied once across five starters, not {sorted(applied)}"


@need_pg
def test_waiting_for_a_lock_does_not_stall_a_concurrent_index_build():
    """The deadlock this guards against: a process blocked in pg_advisory_lock is a running
    statement, CREATE INDEX CONCURRENTLY waits for every running statement, and the lock holder
    (LangGraph, building its indexes under the lock) was the one building the index."""
    import psycopg

    from hr_timeoff_agent.adapters.storage import advisory_lock

    url, _ = _schema()
    key, table = f"lock-{uuid.uuid4().hex}", f"cic_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(url, autocommit=True) as c:
        c.execute(f"create table {table} (id int)")
    try:
        built = threading.Event()
        waiter = None
        with advisory_lock(url, key):
            def wait_for_it():
                with advisory_lock(url, key, timeout=60):
                    pass

            waiter = threading.Thread(target=wait_for_it)
            waiter.start()
            time.sleep(0.5)  # the waiter is now waiting for the lock we hold

            def build():
                with psycopg.connect(url, autocommit=True) as c:
                    c.execute(f"create index concurrently on {table} (id)")
                built.set()

            threading.Thread(target=build, daemon=True).start()
            assert built.wait(15), "the index build is stuck behind a process that is only waiting for a lock"
        waiter.join(10)
        assert not waiter.is_alive(), "the waiter never got the lock after the holder let go"
    finally:
        with psycopg.connect(url, autocommit=True) as c:
            c.execute(f"drop table if exists {table}")


@need_pg
def test_a_lock_that_stays_held_times_out_instead_of_hanging():
    from hr_timeoff_agent.adapters.storage import LockTimeout, advisory_lock

    url, _ = _schema()
    key = f"lock-{uuid.uuid4().hex}"
    with advisory_lock(url, key):
        raises(LockTimeout, lambda: advisory_lock(url, key, timeout=0.3).__enter__())


@need_pg
def test_a_database_newer_than_the_code_is_refused():
    from hr_timeoff_agent.adapters import migrations as m

    url, schema = _schema()
    newer = [*m.MIGRATIONS, (2, "from a later release", "create table later (id int)")]
    m.migrate(url, schema, newer)
    raises(m.SchemaTooNew, m.migrate, url, schema)           # this release only knows version 1


@need_pg
def test_the_migrate_command_reports_and_applies():
    schema = "cmd_" + uuid.uuid4().hex[:10]
    os.environ["HR_DATABASE_SCHEMA"] = schema
    try:
        from hr_timeoff_agent.adapters.storage import ensure_schema

        ensure_schema(PG, schema)
        assert main(["migrate", "--status"]) == 1, "pending migrations are a non-zero status"
        assert main(["migrate"]) == 0 and main(["migrate", "--status"]) == 0
    finally:
        os.environ["HR_DATABASE_SCHEMA"] = "auto"


# ── first start and readiness ───────────────────────────────────────────────

@need_pg
def test_two_services_starting_at_once_seed_the_tenant_once():
    home = tempfile.mkdtemp()
    spaces, errors = [], []

    def start():
        try:
            spaces.append(Workspace(home))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=start) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    for r in spaces[0].requests():
        assert r.get("attempts", 1) == 1 and r["thread_id"] == r["request_id"], f"{r['request_id']} was triaged more than once"
    [s.close() for s in spaces]


def test_the_init_command_prepares_a_home_and_a_second_run_changes_nothing():
    home = tempfile.mkdtemp()
    assert main(["init", "--home", home]) == 0
    assert main(["init", "--home", home]) == 0


def test_the_web_app_is_ready_only_when_its_dependencies_answer():
    app = create_app(tempfile.mkdtemp())
    c = TestClient(app)
    assert c.get("/healthz").status_code == 200
    ok = c.get("/readyz")
    assert ok.status_code == 200 and ok.json() == {"ready": True, "failing": {}}
    ws = app.state.workspace

    def down():
        raise ConnectionError("database is gone")

    ws.store.ping = down
    bad = c.get("/readyz")
    assert bad.status_code == 503 and bad.json()["failing"] == {"store": "ConnectionError"}
    assert c.get("/healthz").status_code == 200, "liveness is not readiness: the process is up"


def test_the_agents_expose_health_and_readiness_without_a_token():
    import asyncio

    import httpx

    import_helpers()
    from test_a2a import Stack

    s = Stack()

    async def go(app, path):
        return await httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://x").get(path)

    for app in (s.app, s.pay_app):
        assert asyncio.run(go(app, "/healthz")).status_code == 200
        assert asyncio.run(go(app, "/readyz")).json()["ready"] is True


def test_the_mcp_server_serves_only_the_hosts_it_is_told_to_and_leaves_health_open():
    import_helpers()
    from test_oidc import CFG, token, verifier

    from hr_timeoff_agent.services.a2a.common import OIDCBearer
    from hr_timeoff_agent.tools.server import HRToolServer
    from hr_timeoff_agent.adapters.oidc import BearerGuard
    from mcp.server.transport_security import TransportSecuritySettings

    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json", "Authorization": f"Bearer {token()}"}

    def make_app():   # a session manager runs once, so each client gets its own app
        server = HRToolServer()
        by_email = {w["email"]: w["worker_id"] for w in server.tenant.workers.values()}
        auth = OIDCBearer(CFG, server.tenant.tenant_id, by_email.get, verifier=verifier(), surface="mcp")
        return BearerGuard(server.server.streamable_http_app(
            host="0.0.0.0", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=["mcp:8200"], allowed_origins=[])),
            auth.authenticate, public_paths=("/healthz",))

    with TestClient(make_app(), base_url="http://mcp:8200") as inside:
        assert inside.get("/healthz").status_code == 200
        assert inside.post("/mcp", json=init, headers=headers).status_code == 200
    with TestClient(make_app(), base_url="http://evil.example:8200") as outside:
        assert outside.post("/mcp", json=init, headers=headers).status_code == 421, "a Host that is not allowed is refused even with a valid token"
        assert outside.post("/mcp", json=init).status_code in (401, 421)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
