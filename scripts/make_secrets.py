#!/usr/bin/env python3
"""Generate the secrets the full stack mounts as files, into ./secrets (gitignored).

    python scripts/make_secrets.py            # creates what is missing, leaves the rest alone
    python scripts/make_secrets.py --force    # regenerates everything (and resets the database password!)

Every value is random. The Keycloak realm is rendered from deploy/keycloak/hr-realm.json with the
generated client secrets and demo-user password swapped in for the throwaway ones the dev realm
ships with, so nothing in the full stack uses a guessable credential.

The files are world-readable (0444) on purpose: the containers run as an unprivileged user that is
not the file's owner. This is a local, single-machine stack. A real deployment injects the same
names from its secret manager (the app reads NAME_FILE or NAME, see hr_timeoff_agent/config.py).

ANTHROPIC_API_KEY is deliberately not generated. To let the stack triage new requests with the
model, put your key in secrets/anthropic_api_key yourself and add -f docker-compose.live.yml.
"""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "secrets"
DEV_REALM = ROOT / "deploy" / "keycloak" / "hr-realm.json"

# The throwaway values in the dev realm, and the secret each is replaced by.
CLIENT_SECRETS = {
    "hr-web-dev-secret": "oidc_web_client_secret",
    "timeoff-agent-dev-secret": "oidc_service_client_secret",
    "other-tenant-dev-secret": "other_tenant_client_secret",
}
DEV_USER_PASSWORD = "hr-demo-pass"


def token() -> str:
    return secrets.token_urlsafe(24)


def write(name: str, value: str, force: bool) -> str:
    path = OUT / name
    if path.exists() and not force:
        return path.read_text().strip()
    path.write_text(value + "\n")
    path.chmod(0o444)
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="regenerate every secret")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    OUT.chmod(0o755)

    pg = write("postgres_password", token(), args.force)
    write("database_url", f"postgresql://hr:{pg}@postgres:5432/hr", args.force)
    write("web_secret", secrets.token_hex(32), args.force)
    write("a2a_secret", secrets.token_hex(16), args.force)
    write("keycloak_admin_password", token(), args.force)
    demo = write("demo_password", token(), args.force)
    values = {name: write(name, token(), args.force) for name in CLIENT_SECRETS.values()}

    realm = DEV_REALM.read_text()
    for dev, name in CLIENT_SECRETS.items():
        realm = realm.replace(dev, values[name])
    realm = realm.replace(DEV_USER_PASSWORD, demo)
    json.loads(realm)  # still valid
    target = OUT / "hr-realm.json"
    if target.exists():
        target.chmod(0o644)
    target.write_text(realm)
    target.chmod(0o444)

    print(f"secrets written to {OUT.relative_to(ROOT)}/ ({len(list(OUT.iterdir()))} files)")
    print(f"demo users (e.g. aiko.tanaka@acme.example) sign in with the password in secrets/demo_password")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
