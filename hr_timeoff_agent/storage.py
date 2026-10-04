"""Where a workspace keeps its state: local files + SQLite, or Postgres.

`Workspace` only talks to a `Store`. The default `FileStore` is the original
local layout (JSON documents under a home directory, SQLite beside them), so
the offline tests and a laptop demo need no services. `PostgresStore`
(`HR_DATABASE_URL=postgresql://...`) keeps the same state in one database that
survives restarts and is shared by the web app and both A2A agents.

What lives in a store:
  documents      the tenant's requests, workers (balances), absences, policy and
                 the handbook, each read and changed as one JSON value
  spend          every live model call, for the daily cap and the hourly limit
  meta           one-off flags such as "seeded"
  checkpointer   LangGraph's paused runs, so a pending approval survives a restart

`mutate` is the only way to change a document: it reads, applies a function
and writes back under a row lock (Postgres) or the workspace lock (files), so
two writers cannot lose each other's update.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

DOCUMENTS = ("workers.json", "absences.json", "policy.json", "requests.json", "handbook.json", "precedents.json")


class Store:
    """Interface both stores implement (documented here, not enforced)."""

    def ping(self) -> None: ...
    def has_documents(self) -> bool: ...
    def seed_documents(self, source: Path) -> None: ...
    def read(self, name: str) -> Any: ...
    def mutate(self, name: str, fn: Callable[[Any], Any]) -> None: ...
    def mutate_many(self, names: list[str], fn: Callable[[dict[str, Any]], None]) -> None: ...
    def lock(self, key: str) -> contextlib.AbstractContextManager: ...
    def meta_get(self, key: str) -> str | None: ...
    def meta_set(self, key: str, value: str) -> None: ...
    def spend_add(self, at: str, persona: str | None, model: str, label: str, usd: float, source: str) -> None: ...
    def spend_count_since(self, persona: str, since: str) -> int: ...
    def spend_on_day(self, day: str) -> float: ...
    def spend_recent(self, n: int) -> list[tuple]: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


class FileStore(Store):
    """JSON documents in `home/tenant/`, SQLite for spend, meta and checkpoints."""

    def __init__(self, home: Path):
        from langgraph.checkpoint.sqlite import SqliteSaver

        self.home = Path(home)
        self.dir = self.home / "tenant"
        self._lock = threading.RLock()
        self._dblock = threading.RLock()   # one sqlite connection is shared by every thread
        self._keys: dict[str, threading.RLock] = {}
        self.home.mkdir(parents=True, exist_ok=True)
        ck = sqlite3.connect(self.home / "checkpoints.sqlite", check_same_thread=False)
        self.checkpointer = SqliteSaver(ck)
        self._db = sqlite3.connect(self.home / "app.sqlite", check_same_thread=False)
        self._db.executescript(
            """
            create table if not exists meta (k text primary key, v text);
            create table if not exists spend (
                at text, persona text, model text, label text, usd real, source text);
            """
        )
        self._conns = [ck, self._db]

    def ping(self) -> None:
        with self._dblock:
            self._db.execute("select 1")

    def has_documents(self) -> bool:
        return (self.dir / "policy.json").exists()

    def seed_documents(self, source: Path) -> None:
        if self.dir.exists():
            shutil.rmtree(self.dir)
        shutil.copytree(source, self.dir)

    def read(self, name: str) -> Any:
        return json.loads((self.dir / name).read_text())

    def mutate(self, name: str, fn: Callable[[Any], Any]) -> None:
        with self._lock:
            value = self.read(name)
            out = fn(value)
            value = value if out is None else out
            tmp = self.dir / f".{name}.tmp"
            tmp.write_text(json.dumps(value, indent=2) + "\n")
            tmp.replace(self.dir / name)

    def mutate_many(self, names: list[str], fn: Callable[[dict[str, Any]], None]) -> None:
        # Files can't be changed together atomically. Each write is atomic and the
        # callers' updates are idempotent, so a crash between two files is repaired
        # by running the operation again. (Postgres does this in one transaction.)
        with self._lock:
            docs = {n: self.read(n) for n in names}
            fn(docs)
            for n in names:
                tmp = self.dir / f".{n}.tmp"
                tmp.write_text(json.dumps(docs[n], indent=2) + "\n")
                tmp.replace(self.dir / n)

    @contextlib.contextmanager
    def lock(self, key: str) -> Iterator[None]:
        """One holder at a time per key, within this process (local files are single-process)."""
        with self._lock:
            k = self._keys.setdefault(key, threading.RLock())
        with k:
            yield

    def meta_get(self, key: str) -> str | None:
        with self._dblock:
            row = self._db.execute("select v from meta where k=?", (key,)).fetchone()
        return row[0] if row else None

    def meta_set(self, key: str, value: str) -> None:
        with self._dblock:
            self._db.execute("insert or replace into meta values (?, ?)", (key, value))
            self._db.commit()

    def spend_add(self, at, persona, model, label, usd, source) -> None:
        with self._dblock:
            self._db.execute("insert into spend values (?,?,?,?,?,?)", (at, persona, model, label, usd, source))
            self._db.commit()

    def spend_count_since(self, persona: str, since: str) -> int:
        with self._dblock:
            return self._db.execute("select count(*) from spend where persona=? and at>=?", (persona, since)).fetchone()[0]

    def spend_on_day(self, day: str) -> float:
        with self._dblock:
            return float(self._db.execute("select coalesce(sum(usd),0) from spend where substr(at,1,10)=?", (day,)).fetchone()[0])

    def spend_recent(self, n: int) -> list[tuple]:
        with self._dblock:
            return self._db.execute(
                "select at, persona, model, label, usd, source from spend order by at desc limit ?", (n,)
            ).fetchall()

    def reset(self) -> None:
        self.close()
        for name in ("tenant", "checkpoints.sqlite", "app.sqlite"):
            target = self.home / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        self.__init__(self.home)

    def close(self) -> None:
        for c in self._conns:
            c.close()


class PostgresStore(Store):
    """The same state in Postgres. Safe to share between processes."""

    def __init__(self, url: str, schema: str = "public"):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        from .migrations import migrate

        self.url, self.schema = url, schema
        migrate(url, schema)   # idempotent and locked: safe when several processes start together
        opts = {"options": f"-csearch_path={schema}", "row_factory": dict_row}
        self.pool = ConnectionPool(url, min_size=1, max_size=8, open=True, kwargs={"autocommit": False, **opts})
        # PostgresSaver wants its own autocommit connections; give it a pool of them.
        self._ck_pool = ConnectionPool(url, min_size=1, max_size=4, open=True, kwargs={"autocommit": True, **opts})
        self.checkpointer = PostgresSaver(self._ck_pool)
        # LangGraph creates and upgrades its own tables, and two processes doing that at once
        # collide on the catalog, so one at a time.
        with advisory_lock(url, f"{schema}:checkpointer-setup"):
            self.checkpointer.setup()

    def ping(self) -> None:
        with self.pool.connection() as c:
            c.execute("select 1")

    def has_documents(self) -> bool:
        with self.pool.connection() as c:
            return c.execute("select 1 from documents where name='policy.json'").fetchone() is not None

    def seed_documents(self, source: Path) -> None:
        with self.pool.connection() as c:
            for name in DOCUMENTS:
                body = (source / name).read_text()
                c.execute(
                    "insert into documents (name, body) values (%s, %s::jsonb) "
                    "on conflict (name) do nothing",
                    (name, body),
                )

    def read(self, name: str) -> Any:
        with self.pool.connection() as c:
            row = c.execute("select body from documents where name=%s", (name,)).fetchone()
        if row is None:
            raise KeyError(name)
        return row["body"]

    def mutate(self, name: str, fn: Callable[[Any], Any]) -> None:
        # The row lock is held from the read to the commit, so a concurrent
        # writer (another request, the A2A agent) waits instead of overwriting.
        with self.pool.connection() as c:
            row = c.execute("select body from documents where name=%s for update", (name,)).fetchone()
            if row is None:
                raise KeyError(name)
            value = row["body"]
            out = fn(value)
            value = value if out is None else out
            c.execute(
                "update documents set body=%s::jsonb, version=version+1, updated_at=now() where name=%s",
                (json.dumps(value), name),
            )

    def mutate_many(self, names: list[str], fn: Callable[[dict[str, Any]], None]) -> None:
        """Several documents changed in one transaction: all of it, or none of it."""
        with self.pool.connection() as c:
            docs = {}
            for n in sorted(names):  # same order everywhere, so two writers cannot deadlock
                row = c.execute("select body from documents where name=%s for update", (n,)).fetchone()
                if row is None:
                    raise KeyError(n)
                docs[n] = row["body"]
            fn(docs)
            for n in names:
                c.execute(
                    "update documents set body=%s::jsonb, version=version+1, updated_at=now() where name=%s",
                    (json.dumps(docs[n]), n),
                )

    @contextlib.contextmanager
    def lock(self, key: str) -> Iterator[None]:
        """A Postgres advisory lock: one holder at a time per key across every process.

        It has its own connection, so a long holder (a model call) cannot starve
        the pool, and the lock is released if the process dies.
        """
        with advisory_lock(self.url, f"{self.schema}:{key}"):
            yield

    def meta_get(self, key: str) -> str | None:
        with self.pool.connection() as c:
            row = c.execute("select v from meta where k=%s", (key,)).fetchone()
        return row["v"] if row else None

    def meta_set(self, key: str, value: str) -> None:
        with self.pool.connection() as c:
            c.execute("insert into meta values (%s,%s) on conflict (k) do update set v=excluded.v", (key, value))

    def spend_add(self, at, persona, model, label, usd, source) -> None:
        with self.pool.connection() as c:
            c.execute(
                "insert into spend (at, persona, model, label, usd, source) values (%s,%s,%s,%s,%s,%s)",
                (at, persona, model, label, usd, source),
            )

    def spend_count_since(self, persona: str, since: str) -> int:
        with self.pool.connection() as c:
            return c.execute(
                "select count(*) as n from spend where persona=%s and at>=%s", (persona, since)
            ).fetchone()["n"]

    def spend_on_day(self, day: str) -> float:
        with self.pool.connection() as c:
            return float(c.execute(
                "select coalesce(sum(usd),0) as s from spend where substr(at,1,10)=%s", (day,)
            ).fetchone()["s"])

    def spend_recent(self, n: int) -> list[tuple]:
        with self.pool.connection() as c:
            rows = c.execute(
                "select at, persona, model, label, usd, source from spend order by at desc, id desc limit %s", (n,)
            ).fetchall()
        return [tuple(r.values()) for r in rows]

    def reset(self) -> None:
        with self.pool.connection() as c:
            c.execute("truncate documents, meta, spend")
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                c.execute(f"truncate {table}")

    def close(self) -> None:
        self.pool.close()
        self._ck_pool.close()


LOCK_POLL_S = 0.02
LOCK_TIMEOUT_S = float(os.environ.get("HR_LOCK_TIMEOUT_S", "120"))


class LockTimeout(TimeoutError):
    pass


@contextlib.contextmanager
def advisory_lock(url: str, key: str, timeout: float | None = None) -> Iterator[None]:
    """A Postgres session advisory lock on its own connection: one holder per key across every
    process, released when the block ends or the process dies.

    It polls with pg_try_advisory_lock instead of blocking in pg_advisory_lock. A blocked
    lock call is a running statement, and a running statement counts as an open transaction
    to CREATE INDEX CONCURRENTLY, which waits for all of them. LangGraph builds its indexes
    that way while holding one of these locks, so a process blocked on the same lock would
    wait for the holder while the holder waited for it: a deadlock. Each try here finishes
    at once, so nothing is left running while we wait.
    """
    import psycopg

    deadline = time.monotonic() + (LOCK_TIMEOUT_S if timeout is None else timeout)
    with psycopg.connect(url, autocommit=True) as c:
        while not c.execute("select pg_try_advisory_lock(hashtextextended(%s, 0))", (key,)).fetchone()[0]:
            if time.monotonic() > deadline:
                raise LockTimeout(f"could not take lock {key!r} within the timeout; another process holds it")
            time.sleep(LOCK_POLL_S)
        try:
            yield
        finally:
            c.execute("select pg_advisory_unlock(hashtextextended(%s, 0))", (key,))


def ensure_schema(url: str, schema: str) -> None:
    if schema == "public":
        return
    import psycopg
    from psycopg import sql

    # "if not exists" is not safe against a second process creating the same schema at the
    # same instant (it fails on the catalog's unique index), so take a lock around it.
    with psycopg.connect(url) as c:
        c.execute("select pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"hr-schema:{schema}",))
        c.execute(sql.SQL("create schema if not exists {}").format(sql.Identifier(schema)))
        c.commit()


def database_settings(home: Path, database_url: str | None = None) -> tuple[str, str] | None:
    """(url, schema) when Postgres is configured, else None (local files)."""
    import os

    url = database_url if database_url is not None else os.environ.get("HR_DATABASE_URL")
    if not url:
        return None
    # HR_DATABASE_SCHEMA=auto gives each home directory its own schema, which is
    # how the tests run against a shared Postgres without seeing each other.
    schema = os.environ.get("HR_DATABASE_SCHEMA", "public")
    if schema == "auto":
        import hashlib

        schema = "ws_" + hashlib.sha1(str(Path(home).resolve()).encode()).hexdigest()[:12]
    return url, schema


def open_store(home: Path, database_url: str | None = None) -> Store:
    settings = database_settings(home, database_url)
    return PostgresStore(*settings) if settings else FileStore(home)
