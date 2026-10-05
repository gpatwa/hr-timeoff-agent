"""Postgres helpers shared by the store and the migrations: advisory locks and schema creation.

A leaf module (imports nothing from the app), so `storage` and `migrations` no longer import each other.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Iterator


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
