# SPDX-License-Identifier: Apache-2.0
"""Clean a Semgrep SARIF file before it is uploaded or logged.

* results suppressed in source (``# nosemgrep`` next to a reviewed ``# nosec``
  reason) are dropped: Semgrep keeps them in SARIF and code scanning would open
  alerts for them;
* the matched source text (``snippet`` and ``contextRegion``) is removed, so a
  secret found by a secrets rule is never copied into the artefact or the log;
* with ``--list``, each remaining result is printed as ``rule path:line`` only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def clean(doc: dict) -> list[tuple[str, str, int]]:
    """Drop suppressed results and source text in place; return (rule, path, line) per result."""
    found = []
    for run in doc.get("runs", []):
        run["results"] = [r for r in run.get("results", []) if not r.get("suppressions")]
        for res in run["results"]:
            res.pop("fingerprints", None)
            for loc in res.get("locations", []):
                phys = loc.get("physicalLocation", {})
                phys.pop("contextRegion", None)
                region = phys.get("region", {})
                region.pop("snippet", None)
                found.append((res.get("ruleId", "?"), phys.get("artifactLocation", {}).get("uri", "?"),
                              region.get("startLine", 0)))
    return found


def main(argv: list[str]) -> int:
    """Clean the SARIF file named first; ``--list`` prints the remaining results."""
    if not argv:
        print("usage: sarif_clean.py FILE.sarif [--list]", file=sys.stderr)
        return 2
    path = Path(argv[0])
    if not path.exists():
        return 0
    doc = json.loads(path.read_text(encoding="utf-8"))
    found = clean(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")
    if "--list" in argv[1:]:
        for rule, uri, line in found:
            print(f"::error file={uri},line={line}::{rule}")
        print(f"{len(found)} finding(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
