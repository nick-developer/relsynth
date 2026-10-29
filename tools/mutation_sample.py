# SPDX-License-Identifier: Apache-2.0
"""Mutation score of the core engine modules from a seeded random sample of mutants.

    python tools/mutation_sample.py --per-module 150 --seed 1 --out mutation.json

mutmut generates every mutant of the modules listed under [tool.mutmut] (about
7,800 on relsynth 1.0, eight hours on four cores), so the baseline and the
weekly job test a simple random sample per module and report the score
killed / (killed + survived) with a Wilson 95% interval.  Mutants that no test
executes count as survived: a line that no fast test reaches is a gap too.
Timeouts count as killed (the mutant changed behaviour).  Following Petrovic
and Ivankovic (ICSE-SEIP 2018), the useful output is the list of surviving
mutants, which reviewers read as missing tests; the score is a trend, not a gate.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from math import sqrt

KILLED = {"killed", "timeout"}
SURVIVED = {"survived", "no tests"}


def mutmut(*args: str) -> str:
    out = subprocess.run([sys.executable, "-m", "mutmut", *args], capture_output=True, text=True, check=False)  # noqa: S603
    return out.stdout


def results() -> dict[str, str]:
    res = {}
    for line in mutmut("results", "--all", "true").splitlines():
        if ": " in line and "__mutmut_" in line:
            name, status = line.strip().rsplit(": ", 1)
            res[name] = status
    return res


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-module", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="mutation.json")
    ap.add_argument("--max-children", default="4")
    a = ap.parse_args(argv)
    allm = results()
    if not allm:                                   # generate the mutants (and run nothing) first
        mutmut("run", "relsynth.posterior.x_run_streams__mutmut_1", "--max-children", a.max_children)
        allm = results()
    by_mod = defaultdict(list)
    for name in sorted(allm):
        by_mod[name.split(".")[1]].append(name)
    rng = random.Random(a.seed)
    sample = {m: sorted(rng.sample(names, min(a.per_module, len(names)))) for m, names in by_mod.items()}
    todo = [n for names in sample.values() for n in names if allm[n] == "not checked"]
    for i in range(0, len(todo), 200):
        mutmut("run", *todo[i:i + 200], "--max-children", a.max_children)
    allm = results()
    report = {"seed": a.seed, "per_module": a.per_module, "modules": {}}
    for m, names in sorted(sample.items()):
        st = [allm[n] for n in names]
        k, s = sum(x in KILLED for x in st), sum(x in SURVIVED for x in st)
        report["modules"][m] = {"mutants_total": len(by_mod[m]), "sampled": len(names), "killed": k, "survived": s,
                                "other": len(st) - k - s, "score": round(k / max(k + s, 1), 3), "ci95": wilson(k, k + s),
                                "survivors": [n for n, x in zip(names, st) if x in SURVIVED]}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("| Module | Mutants | Sampled | Killed | Survived | Score | 95% CI |\n|---|---|---|---|---|---|---|")
    for m, r in report["modules"].items():
        print(f"| {m} | {r['mutants_total']} | {r['sampled']} | {r['killed']} | {r['survived']} | {r['score']:.2f} | "
              f"{r['ci95'][0]:.2f}–{r['ci95'][1]:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
