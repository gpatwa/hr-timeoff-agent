"""Every real adapter meets its port, and a fake can stand in for one.

Run: .venv/bin/python tests/test_ports.py
"""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["HR_AGENT_OFFLINE"] = "1"
for var in ("HR_DATABASE_URL", "HR_OIDC_ISSUER", "ANTHROPIC_API_KEY"):
    os.environ.pop(var, None)

from hr_timeoff_agent import llm, ports, retrieval, storage  # noqa: E402
from hr_timeoff_agent.web import identity  # noqa: E402


def _members(proto) -> set[str]:
    return {n for n, v in vars(proto).items() if callable(v) and not n.startswith("_")}


def test_both_stores_meet_the_store_port():
    with tempfile.TemporaryDirectory() as d:
        store = storage.FileStore(Path(d))
        assert isinstance(store, ports.StorePort)
    assert _members(ports.StorePort) <= set(dir(storage.PostgresStore)), "PostgresStore is missing a StorePort method"


def test_the_retriever_and_both_identity_adapters_meet_their_ports():
    assert _members(ports.RetrieverPort) <= set(dir(retrieval.PolicyIndex))
    assert isinstance(identity.PersonaSwitcher(), ports.IdentityPort)
    assert _members(ports.IdentityPort) <= set(dir(identity.OIDCIdentity))


def test_the_model_adapter_takes_the_arguments_the_port_names():
    wanted = {"system", "user", "schema", "model", "record", "label"}
    assert wanted <= set(inspect.signature(llm.structured).parameters)


def test_a_fake_store_can_replace_a_real_one():
    class Fake:
        pass

    for name in _members(ports.StorePort):
        setattr(Fake, name, lambda self, *a, **k: None)
    assert isinstance(Fake(), ports.StorePort)


def test_the_scoped_overrides_restore_what_they_replaced_even_on_error():
    before = (llm.FIXTURES, llm.before_live_call, llm.after_live_call)
    try:
        with llm.recording_into(Path("/tmp/x.json")), llm.observing(lambda *a: None, lambda *a: None):
            assert llm.FIXTURES == Path("/tmp/x.json") and llm.before_live_call is not None
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert (llm.FIXTURES, llm.before_live_call, llm.after_live_call) == before


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
