# SPDX-License-Identifier: Apache-2.0
"""Check (or add) the SPDX licence identifier at the top of every Python file.

    python tools/check_spdx.py          # exit 1 and list the files without it
    python tools/check_spdx.py --fix    # add it after any shebang or encoding line

The policy is in CONTRIBUTING.md: each Python file in the repository carries
``# SPDX-License-Identifier: Apache-2.0`` in its first three lines.  Files are
found with ``git ls-files``, so untracked and ignored files are not checked.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

TAG = "# SPDX-License-Identifier: Apache-2.0"
HEAD = re.compile(r"^#!|^#.*coding[:=]")


def tracked(root: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=root, capture_output=True, text=True, check=True)  # noqa: S603, S607
    return [root / line for line in out.stdout.splitlines() if line]


def has_tag(text: str) -> bool:
    return any(line.strip() == TAG for line in text.splitlines()[:3])


def add_tag(text: str) -> str:
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and i < 2 and HEAD.match(lines[i]):
        i += 1
    return "".join(lines[:i] + [TAG + "\n"] + lines[i:])


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    fix = "--fix" in argv
    missing = []
    for path in tracked(root):
        text = path.read_text(encoding="utf-8")
        if has_tag(text):
            continue
        if fix:
            path.write_text(add_tag(text), encoding="utf-8")
        else:
            missing.append(path.relative_to(root))
    for p in missing:
        print(f"{p}: missing '{TAG}'")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
