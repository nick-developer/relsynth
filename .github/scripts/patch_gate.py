# SPDX-License-Identifier: Apache-2.0
"""Refuse an agent-written patch that reaches outside what a security fix may change.

Used by the ``security-patch`` workflow after ``git am`` and before any code of the
patch runs.  ``python patch_gate.py BASE`` compares ``BASE..HEAD`` and exits non-zero,
listing every reason, when the patch

* adds or changes anything but a regular file (a symbolic link, a submodule), which
  could point into a protected tree while its own name looks harmless;
* touches a protected path: anything under ``.github/``, ``.semgrep/`` or ``.claude/``,
  the review and secret-scanning configuration, the agent instructions, ``tests/``
  (a fix must pass the existing tests), the gate scripts in ``tools/``, or any file
  that test, lint or type-check tools read as configuration in place of
  ``pyproject.toml`` (``conftest.py``, ``pytest.toml``, ``setup.cfg``, ``ruff.toml`` ...);
* changes the ``[tool.*]`` settings of ``pyproject.toml`` (ruff, bandit, mypy, pytest):
  a fix may raise dependency floors, never weaken a check.

Paths are read NUL-separated (``-z``), so git never quotes an unusual name past the
check, and ``--no-renames`` lists a moved file under its old path too.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:   # Python 3.10 (the unit tests run there too); pytest brings tomli
    import tomli as tomllib

DENIED = re.compile(
    r"(\.github/|\.semgrep/|\.claude/|\.coderabbit\.yaml$|\.gitleaks\.toml$|CLAUDE\.md$|CONTRIBUTING\.md$"
    r"|tests/|tools/"
    # configuration that pytest, ruff, bandit, mypy or coverage would read in place of
    # (or next to) the settings in pyproject.toml, wherever it is placed
    r"|(.*/)?(conftest\.py|pytest\.ini|\.?pytest\.toml|tox\.ini|setup\.cfg|\.?ruff\.toml|\.bandit"
    r"|mypy\.ini|\.mypy\.ini|\.coveragerc|sitecustomize\.py|usercustomize\.py)$"
    r"|.*\.pth$)")
REGULAR = {"100644", "100755", "000000"}   # 000000: the side of an added or deleted file


def _git(*args: str) -> bytes:
    """Run git and return its standard output."""
    git = shutil.which("git") or "git"
    # argument list, no shell; the arguments are git options, a commit from the workflow and HEAD
    return subprocess.run([git, *args], check=True, capture_output=True).stdout  # noqa: S603 - fixed git argv


def changes(base: str) -> list[tuple[str, str, str]]:
    """(old mode, new mode, path) of every entry changed in ``base..HEAD``."""
    out = _git("diff", "-z", "--raw", "--no-renames", "--no-abbrev", f"{base}..HEAD")
    fields = out.split(b"\0")
    result = []
    for i in range(0, len(fields) - 1, 2):
        meta, path = fields[i].decode("ascii", "replace"), fields[i + 1].decode("utf-8", "surrogateescape")
        if not meta.startswith(":"):
            continue
        old_mode, new_mode = meta[1:].split(" ")[:2]
        result.append((old_mode, new_mode, path))
    return result


def problems(base: str) -> list[str]:
    """Every reason the patch ``base..HEAD`` may not be published (empty when it may)."""
    out = []
    for old_mode, new_mode, path in changes(base):
        if old_mode not in REGULAR or new_mode not in REGULAR:
            out.append(f"not a regular file (mode {old_mode} -> {new_mode}): {path!r}")
        if DENIED.match(path):
            out.append(f"protected path: {path!r}")
    old = tomllib.loads(_git("show", f"{base}:pyproject.toml").decode("utf-8"))
    new = tomllib.loads(_git("show", "HEAD:pyproject.toml").decode("utf-8"))
    if old.get("tool") != new.get("tool"):
        out.append("pyproject.toml: [tool.*] settings changed")
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: ``patch_gate.py BASE``."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: patch_gate.py BASE", file=sys.stderr)
        return 2
    found = problems(args[0])
    for p in found:
        print(p)
    if found:
        print("::error::The patch changes what a security fix may not change (listed above); nothing is published")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
