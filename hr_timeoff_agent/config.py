"""Secrets from files, for containers.

A deployment mounts secrets as files (Docker/Kubernetes secrets) rather than putting
values in the environment. For any variable NAME, `NAME_FILE=/run/secrets/x` means
"read NAME from that file". `load_file_secrets()` runs once at startup and sets
NAME in this process's environment, so the rest of the code (and the Anthropic SDK)
keeps reading plain environment variables. A NAME that is already set wins, and a
NAME_FILE that points at nothing is an error, never a silent empty secret.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

SECRET_NAMES = (
    "HR_DATABASE_URL",
    "HR_WEB_SECRET",
    "HR_A2A_SECRET",
    "HR_OIDC_CLIENT_SECRET",
    "HR_OIDC_SERVICE_CLIENT_SECRET",
    "ANTHROPIC_API_KEY",
)


class MissingSecret(RuntimeError):
    pass


def load_file_secrets(names=SECRET_NAMES, env: dict | None = None) -> list[str]:
    """Fill NAME from NAME_FILE. Returns the names that were loaded (never their values)."""
    env = os.environ if env is None else env
    loaded = []
    for name in names:
        path = env.get(f"{name}_FILE")
        if not path or env.get(name):
            continue
        try:
            value = Path(path).read_text().strip()
        except OSError as exc:
            raise MissingSecret(f"{name}_FILE points at {path}, which cannot be read: {exc.strerror}") from exc
        if not value:
            raise MissingSecret(f"{name}_FILE points at {path}, which is empty")
        env[name] = value
        loaded.append(name)
    return loaded


@contextlib.contextmanager
def scoped_env(**values: str | None):
    """Set environment variables for the block and put each back as it was (unset stays unset).
    A value of None unsets the variable for the block."""
    saved = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
