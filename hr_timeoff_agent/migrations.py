"""Versioned schema migrations for the Postgres store.

Each migration is a numbered SQL script, applied once, in order, in its own
transaction, under an advisory lock, so several processes starting at once apply
each exactly once and a failed one leaves nothing half-applied. The applied
versions are recorded in `schema_migrations`.

Two rules keep a rolling deployment safe. A database that is *newer* than this
code (a version the code does not know) is refused, not guessed at. And a
migration is never edited after release: a change is a new number.

Not covered here: LangGraph's checkpoint tables and the A2A task tables, which
their libraries create and upgrade themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

BASELINE = """
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

# (version, name, sql). Append only; never edit a released entry.
MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "baseline: documents, meta, spend", BASELINE),
]


class MigrationFailed(RuntimeError):
    def __init__(self, version: int, name: str, cause: Exception):
        super().__init__(f"migration {version} ({name}) failed and was rolled back: {cause}")
        self.version = version


class SchemaTooNew(RuntimeError):
    """The database has migrations this code does not know: an older release is running against a newer database."""


@dataclass
class Status:
    applied: list[int]
    pending: list[int]
    latest: int

    @property
    def current(self) -> int:
        return max(self.applied, default=0)


def _connect(url: str, schema: str):
    import psycopg

    from .storage import ensure_schema

    ensure_schema(url, schema)
    return psycopg.connect(url, options=f"-csearch_path={schema}", autocommit=False)


def _ensure_table(c) -> None:
    c.execute(
        "create table if not exists schema_migrations ("
        "version integer primary key, name text not null, applied_at timestamptz not null default now())"
    )


def status(url: str, schema: str = "public", migrations=None) -> Status:
    migrations = MIGRATIONS if migrations is None else migrations
    with _connect(url, schema) as c:
        _ensure_table(c)
        applied = sorted(r[0] for r in c.execute("select version from schema_migrations").fetchall())
        c.commit()
    known = [v for v, _, _ in migrations]
    return Status(applied, [v for v in known if v not in applied], max(known, default=0))


def migrate(url: str, schema: str = "public", migrations=None) -> list[int]:
    """Bring the database up to date. Returns the versions applied by this call.

    The advisory lock is taken on its own autocommit connection, so a process that is
    waiting for it has no transaction open. That matters: another process may be
    creating LangGraph's indexes with CREATE INDEX CONCURRENTLY, which waits for every
    older open transaction to finish, and a waiter holding one would deadlock with it.
    The work itself then runs on a second connection, in short transactions.
    """
    from .storage import advisory_lock, ensure_schema

    migrations = sorted(MIGRATIONS if migrations is None else migrations)
    known = {v for v, _, _ in migrations}
    done: list[int] = []
    ensure_schema(url, schema)
    with advisory_lock(url, f"hr-migrate:{schema}"):
        with _connect(url, schema) as c:
            _ensure_table(c)
            c.commit()
            applied = {r[0] for r in c.execute("select version from schema_migrations").fetchall()}
            c.commit()
            unknown = sorted(applied - known)
            if unknown:
                raise SchemaTooNew(f"the database has migration(s) {unknown} that this release does not know; "
                                   "run the release that applied them")
            for version, name, sql in migrations:
                if version in applied:
                    continue
                try:
                    c.execute(sql)
                    c.execute("insert into schema_migrations (version, name) values (%s, %s)", (version, name))
                    c.commit()
                except Exception as exc:
                    c.rollback()
                    raise MigrationFailed(version, name, exc) from exc
                done.append(version)
    return done
