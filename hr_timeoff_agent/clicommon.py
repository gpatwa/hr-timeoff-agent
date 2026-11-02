"""Small things every command group prints or writes the same way."""

from __future__ import annotations

import os
from pathlib import Path

from .adapters.llm import is_offline

OUT = Path(__file__).resolve().parent.parent / "out"
RULE = "─" * 74


def mode_banner() -> str:
    return "offline (fixture-backed)" if is_offline() else "live (Anthropic API)"


LOOPBACK = ("127.0.0.1", "localhost", "::1")
DEMO_AUTH_ENV = "HR_ALLOW_DEMO_AUTH"


def demo_auth_refusal(host: str, surface: str) -> str | None:
    """None when it is safe to start; otherwise the message to print and exit on.

    With no HR_OIDC_ISSUER the web app is a persona picker (anyone chooses anyone, an admin included) and the A2A agents
    use static demo tokens. That is right for a laptop and wrong anywhere else, and an unset variable (a typo, a missing
    secret in a deployment) must not turn into that silently. So: loopback is allowed as before, and any other address
    needs HR_ALLOW_DEMO_AUTH=1, an explicit statement that the data is synthetic and the network is trusted.
    """
    from .adapters.oidc import OIDCConfig

    if OIDCConfig.from_env() is not None or host in LOOPBACK or os.environ.get(DEMO_AUTH_ENV) == "1":
        return None
    return (f"Refusing to serve the {surface} on {host} with demo sign-in: HR_OIDC_ISSUER is not set, so there is no "
            f"authentication.\nSet HR_OIDC_ISSUER for real sign-in, bind to 127.0.0.1, or set {DEMO_AUTH_ENV}=1 if the data "
            "is synthetic and the network is trusted.")


def auth_banner(surface: str) -> str:
    from .adapters.oidc import OIDCConfig

    cfg = OIDCConfig.from_env()
    return (f"  Sign-in ({surface}): OpenID Connect via {cfg.issuer}" if cfg
            else f"  Sign-in ({surface}): DEMO, no authentication (a persona picker and static tokens). Synthetic data only.")
