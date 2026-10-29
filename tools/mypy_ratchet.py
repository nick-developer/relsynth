# SPDX-License-Identifier: Apache-2.0
"""Type-check ratchet: mypy may not report an error that is not in the baseline.

    python tools/mypy_ratchet.py            # fail on new errors, report fixed ones
    python tools/mypy_ratchet.py --update   # rewrite the baseline (after fixing errors)

relsynth 1.0 was written without a type checker and mypy reports 75 errors on
it, most of them attributes set dynamically on model objects.  Fixing them all
at once would touch the engine for no behavioural gain, so CI holds the line
instead: the baseline (tools/mypy-baseline.txt) lists today's errors without
line numbers (file, error code and message), new code may not add one, and a
change that fixes some should shrink the baseline with --update.
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools" / "mypy-baseline.txt"
LINE = re.compile(r"^(?P<file>[^:]+):\d+(?::\d+)?: error: (?P<msg>.*?)\s+\[(?P<code>[a-z-]+)\]$")


def current() -> Counter:
    out = subprocess.run([sys.executable, "-m", "mypy", "src/relsynth"], cwd=ROOT,  # noqa: S603
                         capture_output=True, text=True, check=False)
    if out.returncode not in (0, 1) or "error: " not in out.stdout and out.returncode == 1 and not out.stdout:
        sys.stderr.write(out.stdout + out.stderr)
        raise SystemExit(2)
    found: Counter = Counter()
    for line in out.stdout.splitlines():
        m = LINE.match(line.strip())
        if m:
            found[f"{m['file']}: [{m['code']}] {m['msg']}"] += 1
    return found


def main(argv: list[str]) -> int:
    now = current()
    if "--update" in argv:
        BASELINE.write_text("".join(f"{k}\n" * n for k, n in sorted(now.items())), encoding="utf-8")
        print(f"baseline: {sum(now.values())} errors")
        return 0
    base = Counter(BASELINE.read_text(encoding="utf-8").splitlines()) if BASELINE.exists() else Counter()
    new, fixed = now - base, base - now
    for k, n in sorted(new.items()):
        print(f"::error::new type error ({n}x): {k}")
    if fixed:
        print(f"{sum(fixed.values())} baseline error(s) are fixed; run 'python tools/mypy_ratchet.py --update'")
    print(f"mypy: {sum(now.values())} errors, baseline {sum(base.values())}, new {sum(new.values())}")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
