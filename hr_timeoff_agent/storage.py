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

import json
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable

DOCUMENTS = ("workers.json", "absences.json", "policy.json", "requests.json", "handbook.json", "precedents.json")


class Store:
    """Interface both stores implement (documented here, not enforced)."""

    def has_documents(self) -> bool: ...
    def seed_documents(self, source: Path) -> None: ...
    def read(self, name: str) -> Any: ...
    def mutate(self, name: str, fn: Callable[[Any], Any]) -> None: ...
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

    def meta_get(self, key: str) -> str | None:
        row = self._db.execute("select v from meta where k=?", (key,)).fetchone()
        return row[0] if row else None

    def meta_set(self, key: str, value: str) -> None:
        self._db.execute("insert or replace into meta values (?, ?)", (key, value))
        self._db.commit()

    def spend_add(self, at, persona, model, label, usd, source) -> None:
        self._db.execute("insert into spend values (?,?,?,?,?,?)", (at, persona, model, label, usd, source))
        self._db.commit()

    def spend_count_since(self, persona: str, since: str) -> int:
        return self._db.execute("select count(*) from spend where persona=? and at>=?", (persona, since)).fetchone()[0]

    def spend_on_day(self, day: str) -> float:
        return float(self._db.execute("select coalesce(sum(usd),0) from spend where substr(at,1,10)=?", (day,)).fetchone()[0])

    def spend_recent(self, n: int) -> list[tuple]:
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

    SCHEMA = """
    create table if not exists documents (
        name text primary key, body jsonb not null, version bigint not null default 1,
        updated_at timestamptz not null default now());
    create table if not exists meta (k text primary key, v text not null);
    create table if not exists spend (
        id bigserial primary key, at text not null, persona text, model text,
        label text, usd double precision not null, source text);
    create index if not exists spend_day on spend (substr(at, 1, 10));
    create index if not exists spend_persona_at on spend (persona, at);
    """

    def __init__(self, url: str, schema: str = "public"):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.url, self.schema = url, schema
        ensure_schema(url, schema)
        opts = {"options": f"-csearch_path={schema}", "row_factory": dict_row}
        self.pool = ConnectionPool(url, min_size=1, max_size=8, open=True, kwargs={"autocommit": False, **opts})
        # PostgresSaver wants its own autocommit connections; give it a pool of them.
        self._ck_pool = ConnectionPool(url, min_size=1, max_size=4, open=True, kwargs={"autocommit": True, **opts})
        self.checkpointer = PostgresSaver(self._ck_pool)
        self.checkpointer.setup()
        with self.pool.connection() as c:
            c.execute(self.SCHEMA)

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


def ensure_schema(url: str, schema: str) -> None:
    if schema == "public":
        return
    import psycopg
    from psycopg import sql

    with psycopg.connect(url, autocommit=True) as c:
        c.execute(sql.SQL("create schema if not exists {}").format(sql.Identifier(schema)))


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
