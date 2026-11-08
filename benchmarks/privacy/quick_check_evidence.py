# SPDX-License-Identifier: Apache-2.0
"""Evidence for the quick privacy check of `relsynth run` (G5, Props. 1740-1741).

    python benchmarks/privacy/quick_check_evidence.py --out docs/assurance/evidence/G5/quick_check.json

1. Known answers on one table (S1 mixed table, 9000 people per seed): a
   database of 3000 people, and releases that copy every member, copy 20% of
   them, or draw people of the same population outside the database.
2. The check on the benchmark databases that are built (bench/.cache; run
   `python -m bench fetch` first), with its time, on the spec's own generator.

Only aggregates are written: AUCs, TPRs, verdicts, people counts, seconds.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "fakedata"))
sys.path.insert(0, str(ROOT))


def known_answers(seeds: int) -> dict:
    import scenarios as S
    from relsynth.privacy import quick as Q
    from relsynth.spec import load_spec
    rows: dict[str, list] = {"copy_all": [], "copy_20pct": [], "copy_5pct": [], "population": []}
    for rep in range(seeds):
        t, s, _ = S.mixed_table(n=9000, seed=100 + rep)
        spec = load_spec(s)
        pop = t["patient"]
        db, pool = {"patient": pop.iloc[:3000].reset_index(drop=True)}, pop.iloc[3000:].reset_index(drop=True)

        def release(tables, seed, frac):
            tr = tables["patient"]
            n, k = len(tr), int(frac * len(tr))
            out = pd.concat([tr.sample(n=k, random_state=seed), pool.sample(n=n - k, random_state=seed)])
            out = out.reset_index(drop=True)
            out["patient_id"] = range(1, n + 1)
            return {"patient": out}
        for name, frac in (("copy_all", 1.0), ("copy_20pct", 0.2), ("copy_5pct", 0.05), ("population", 0.0)):
            r = Q.quick_check(spec, db, seed=rep, unit="patient", people=3000,
                              generator=lambda tb, sd, f=frac: release(tb, sd, f))
            att = r.get("attacks") or {}          # absent when the check was skipped or failed
            rows[name].append({"status": r["status"], **{f"{a}_auc": v.get("auc") for a, v in att.items()},
                               **{f"{a}_tpr_at_fpr_0.01": v.get("tpr_at_fpr_0.01") for a, v in att.items()}})
    out = {}
    for name, rs in rows.items():
        df = pd.DataFrame(rs)
        out[name] = {"runs": len(df), "fail_share": float((df["status"] == "fail").mean()),
                     **{c: round(float(pd.to_numeric(df[c]).mean()), 3) for c in df.columns if c != "status"}}
    return out


def bench_runs(names: list[str]) -> dict:
    import yaml
    from bench.datasets import DATASETS
    from relsynth.api import _resolved
    from relsynth.privacy.quick import quick_check
    from relsynth.pushdown import load_for_fit
    out = {}
    for name in names:
        ds = DATASETS[name]
        try:
            loc = ds.location()
        except Exception as e:  # noqa: BLE001 - a database that is not built is reported, not fatal
            out[name] = {"error": f"not available: {e!r}"[:200]}
            continue
        spec = yaml.safe_load(Path(ds.spec()).read_text(encoding="utf-8"))
        kind, path = loc.split(":", 1)
        spec["source"] = {**(spec.get("source") or {}), "kind": kind, "path": path}
        s = _resolved(spec)
        fit = load_for_fit(s)
        t0 = time.time()
        r = quick_check(s, fit.tables, stats=fit.stats, seed=1)
        out[name] = {"status": r["status"], "unit": r.get("unit"), "people": r.get("people"),
                     "attacks": r.get("attacks"), "reason": r.get("reason"), "seconds": round(time.time() - t0, 1)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--datasets", default="chinook,berka,acs,contoso_sqlite")
    a = ap.parse_args(argv)
    import os
    os.environ.setdefault("RELSYNTH_QUIET", "1")
    doc = {"what": "quick privacy check of relsynth run (G5)", "machine": {"cpus": os.cpu_count(),
                                                                         "python": platform.python_version()},
           "known_answers_one_table": known_answers(a.seeds),
           "bench": bench_runs([x for x in a.datasets.split(",") if x])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x)),
                           encoding="utf-8")
    print(json.dumps(doc, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
