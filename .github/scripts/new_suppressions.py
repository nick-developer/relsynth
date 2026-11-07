# SPDX-License-Identifier: Apache-2.0
"""List the ``# nosemgrep`` comments a pull request adds (read from ``git diff -U0`` on stdin).

Each one hides a finding from the Semgrep gate and from code scanning, so it is
annotated on its line and listed in the job summary for the reviewer.  Only the
file and line are printed, never the line's text.
"""
from __future__ import annotations

import os
import sys


def added_suppressions(diff: str) -> list[tuple[str, int]]:
    """(path, line) of every added line in a unified diff that contains ``nosemgrep``."""
    out, path, line = [], None, 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            path = raw[6:] if raw.startswith("+++ b/") else None
        elif raw.startswith("@@"):
            plus = raw.split(" ")[2]                     # +start[,count]
            line = int(plus[1:].split(",")[0])
        elif raw.startswith("+") and path is not None:
            if "nosemgrep" in raw:
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
