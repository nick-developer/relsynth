# SPDX-License-Identifier: Apache-2.0
"""Check the structure of the proofs document before it is built.

    python tools/check_proofs.py            # exit 1 and list the problems

Two problems have happened when threads added sections in parallel, and LaTeX
reports neither:

* a ``sec_*.tex`` file in ``docs/proofs/`` that ``relsynth-proofs.tex`` never
  ``\\input``s (its propositions are missing from the PDF), or an ``\\input``
  of a file that does not exist;
* two propositions that claim the same fixed number.  Sections number their
  propositions to match their checks, either in the title
  (``[Worker independence; Prop.~950]``) or with
  ``\\setcounter{proposition}{N-1}`` just before ``\\begin{proposition}``.
  Release 2.0 shipped DP mode and the partitioned runtime both as
  "Prop. 900"; the runtime is now 950-954.

The CI job ``proofs`` runs this and then builds the PDF with latexmk.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "docs" / "proofs"
MAIN = "relsynth-proofs.tex"
INPUT = re.compile(r"^\s*\\input\{([^}]+)\}", re.M)
SETCOUNTER = re.compile(r"\\setcounter\{proposition\}\{(\d+)\}")
BEGIN = re.compile(r"\\begin\{proposition\}(\[[^\]]*\])?")
TITLE_NUMBER = re.compile(r"Prop\.~?\s*(\d{3,5})\]")


def fixed_numbers(path: Path) -> list[tuple[int, int]]:
    """(proposition number, line) for every proposition with a fixed number."""
    text = path.read_text(encoding="utf-8")
    found = []
    pending: int | None = None
    pos_to_line = {}
    line = 1
    for i, ch in enumerate(text):
        pos_to_line[i] = line
        if ch == "\n":
            line += 1
    events = [(m.start(), "set", m) for m in SETCOUNTER.finditer(text)]
    events += [(m.start(), "begin", m) for m in BEGIN.finditer(text)]
    for start, kind, m in sorted(events, key=lambda e: e[0]):
        if kind == "set":
            pending = int(m.group(1)) + 1
            continue
        title = m.group(1) or ""
        t = TITLE_NUMBER.search(title)
        if t:
            found.append((int(t.group(1)), pos_to_line[start]))
        elif pending is not None and pending >= 100:
            found.append((pending, pos_to_line[start]))
        pending = pending + 1 if pending is not None else None
    return found


def main() -> int:
    problems: list[str] = []
    main_tex = (ROOT / MAIN).read_text(encoding="utf-8")
    inputs = [name if name.endswith(".tex") else name + ".tex" for name in INPUT.findall(main_tex)]
    for name in inputs:
        if not (ROOT / name).exists():
            problems.append(f"{MAIN} inputs {name}, which does not exist")
    for path in sorted(ROOT.glob("sec_*.tex")):
        if path.name not in inputs:
            problems.append(f"{path.name} is never \\input in {MAIN}")
    seen: dict[int, list[str]] = defaultdict(list)
    for name in [MAIN, *inputs]:
        path = ROOT / name
        if path.exists():
            for number, line in fixed_numbers(path):
                seen[number].append(f"{name}:{line}")
    for number, where in sorted(seen.items()):
        if len(where) > 1:
            problems.append(f"Proposition {number} is claimed {len(where)} times: {', '.join(where)}")
    for p in problems:
        print(p)
    if not problems:
        print(f"ok: {len(inputs)} sections, {len(seen)} fixed proposition numbers, none twice")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
