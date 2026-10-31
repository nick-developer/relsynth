# SPDX-License-Identifier: Apache-2.0
"""CI check of `relsynth autotune` on Chinook (the CI job `settings-autotune`).

    python tools/autotune_ci.py --out tune-out

1. Chinook as given: autotune must finish with a report (exit 0 or 1, never a
   crash or an invalid-input exit), every hard constraint met.
2. Chinook from a planted bad setting (`facts.count_model: poisson` with
   `segments.default_cap: 1`): the search must reach every objective, be
   confirmed on the held-out seed, and undo at least one planted setting.

Writes each run's tuned spec, report.md and report.json under --out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANTED = {"facts.count_model": "poisson", "segments.default_cap": 1}


def autotune(db: Path, out: Path, sets: dict, runs: int) -> tuple[int, dict]:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "relsynth", "autotune", str(ROOT / "examples/chinook_model/spec.yaml"),
           "--set", "source.kind=sqlite", "--set", f"source.path={db}", "--runs", str(runs), "--seeds", "1,2",
           "--holdout", "11", "--history", str(out / "history"), "--out", str(out / "tuned.yaml"),
           "--report", str(out / "report.md"), "--fresh"]
    for k, v in sets.items():
        cmd += ["--set", f"synthesis.{k}={v}"]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env={**os.environ, "RELSYNTH_QUIET": "1"}, check=False)  # noqa: S603
    rep = out / "report.json"
    return p.returncode, json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tune-out")
    ap.add_argument("--runs", type=int, default=24)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from bench.datasets import build_chinook
    db = build_chinook()
    out = Path(a.out)
    bad = []

    rc, r = autotune(db, out / "as_given", {}, a.runs)
    if rc not in (0, 1) or not r:
        bad.append(f"as given: exit {rc}, report {'written' if r else 'missing'}")
    else:
        hard = (r.get("judgement") or {}).get("constraints") or {}
        broken = [k for k, v in hard.items() if v.get("evaluated") and not v.get("met")]
        if broken:
            bad.append(f"as given: hard constraints not met {broken}")
        print(f"as given: met={r.get('met')} confirmed={r.get('confirmed')} delta={r.get('delta')}")

    rc, r = autotune(db, out / "planted", PLANTED, a.runs)
    delta = r.get("delta") or {}
    print(f"planted: exit {rc} met={r.get('met')} confirmed={r.get('confirmed')} delta={delta} runs={r.get('runs')}")
    if rc != 0 or not r.get("met") or r.get("confirmed") is False:
        bad.append(f"planted: objectives not reached (exit {rc}, stopped: {r.get('stopped')})")
    undone = [k for k, v in PLANTED.items() if k in delta and str(delta[k]) != str(v)]
    if not undone:
        bad.append(f"planted: no planted setting was changed ({delta})")

    if bad:
        print("\nFAILED:\n  " + "\n  ".join(bad))
        return 1
    print("\nautotune CI checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
