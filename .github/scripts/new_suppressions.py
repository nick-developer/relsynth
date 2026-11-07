# SPDX-License-Identifier: Apache-2.0
"""List the ``# nosemgrep`` comments a pull request adds (read from ``git diff -U0`` on stdin).

Each one hides a finding from the Semgrep gate and from code scanning, so it is
annotated on its line and listed in the job summary for the reviewer.  Only the
file and line are printed, never the line's text.
"""
from __future__ import annotations

import os
import re
import sys

# Semgrep's own forms: nosem or nosemgrep after a # or // comment marker, any case
_SUPPRESSION = re.compile(r"(?:#|//)\s*nosem(?:grep)?\b", re.IGNORECASE)


_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}


def _unquote(body: str) -> str:
    """Undo git's C quoting of a path: escapes and octal bytes become bytes, every
    other character stays as its UTF-8 bytes, and the result is decoded as UTF-8."""
    out, i = bytearray(), 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in "01234567" and i + 3 < len(body) + 1 and all(c in "01234567" for c in body[i + 1:i + 4]):
                out.append(int(body[i + 1:i + 4], 8) & 0xFF)
                i += 4
                continue
            if nxt in _ESCAPES:
                out.append(_ESCAPES[nxt])
                i += 2
                continue
        out += ch.encode("utf-8")
        i += 1
    return out.decode("utf-8", "replace")


def added_suppressions(diff: str) -> list[tuple[str, int]]:
    """(path, line) of every added line in a unified diff that carries a Semgrep suppression comment."""
    out, path, line = [], None, 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            # the workflow runs git diff with core.quotePath=false and explicit a/ b/ prefixes;
            # a path with special characters still arrives in C quotes, which are undone here
            name = raw[4:].rstrip("\t")      # git ends the header with a tab when the path has spaces
            if len(name) >= 2 and name.startswith('"') and name.endswith('"'):
                name = _unquote(name[1:-1])
            path = name[2:] if name.startswith("b/") else None
        elif raw.startswith("@@"):
            plus = raw.split(" ")[2]                     # +start[,count]
            line = int(plus[1:].split(",")[0])
        elif raw.startswith("+") and path is not None:
            if _SUPPRESSION.search(raw):
                out.append((path, line))
            line += 1
    return out


def main() -> int:
    """Annotate each new suppression; always succeeds (the reviewer decides)."""
    found = added_suppressions(sys.stdin.read())
    for path, line in found:
        print(f"::warning file={path},line={line}::new nosemgrep: needs a reason and the owner's review")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"### New `# nosemgrep` suppressions: {len(found)}\n\n")
            f.writelines(f"- `{p}:{n}`\n" for p, n in found)
    print(f"{len(found)} new suppression(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
