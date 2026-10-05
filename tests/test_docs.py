"""The docs stay true to the code: every environment variable the code reads is in the configuration
reference, and every link in the docs index and the README contents resolves.

Run: .venv/bin/python tests/test_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_NAME = re.compile(r"[\"']((?:HR_|OTEL_|FASTEMBED_)[A-Z0-9_]+|ANTHROPIC_API_KEY)[\"']")


def _env_vars_in_code() -> set[str]:
    found: set[str] = set()
    for p in (ROOT / "hr_timeoff_agent").rglob("*.py"):
        found |= set(ENV_NAME.findall(p.read_text()))
    return found


def _anchors(md: Path) -> set[str]:
    out = set()
    for h in re.findall(r"^#{1,6} (.+)$", md.read_text(), flags=re.M):
        a = re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-")
        out.add(a)
    return out


def _links(md: Path) -> list[str]:
    return re.findall(r"\]\(([^)\s]+)\)", md.read_text())


def test_every_environment_variable_the_code_reads_is_documented():
    doc = (ROOT / "docs" / "configuration.md").read_text()
    missing = sorted(v for v in _env_vars_in_code() if f"`{v}`" not in doc)
    assert not missing, f"read by the code but not in docs/configuration.md: {missing}"


def test_the_configuration_reference_lists_nothing_the_code_does_not_read():
    doc = (ROOT / "docs" / "configuration.md").read_text()
    listed = set(re.findall(r"^\| `((?:HR_|OTEL_|FASTEMBED_)[A-Z0-9_]+|ANTHROPIC_API_KEY)` \|", doc, flags=re.M))
    extra = sorted(listed - _env_vars_in_code())
    assert not extra, f"documented but never read: {extra}"


def test_every_link_in_the_docs_index_and_the_readme_contents_resolves():
    bad = []
    for md in (ROOT / "docs" / "README.md", ROOT / "README.md"):
        for link in _links(md):
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, frag = link.partition("#")
            target = (md.parent / path) if path else md
            if not target.exists():
                bad.append(f"{md.name}: {link}")
            elif frag and target.suffix == ".md" and frag not in _anchors(target):
                bad.append(f"{md.name}: {link} (no such heading)")
    assert not bad, bad


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}: {str(exc)[:300]}")
    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
