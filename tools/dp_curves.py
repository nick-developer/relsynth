# SPDX-License-Identifier: Apache-2.0
"""Utility of DP mode against epsilon on the bench databases (roadmap T07b).

    python tools/dp_curves.py --db chinook --eps 0.5 1 4 8 --measure copula mst aim --reps 3 --out curves.jsonl

For every run it records the share of scorecard checks that pass (the T07
utility figure) and the workload error (mean two-way total variation
distance, tools/dp_workload.py) of every owned table, synthetic against real.
The bench data must be in the cache first (`python -m bench fetch --datasets ...`).
`--summary` prints mean and a 95 % t interval over repetitions from a jsonl file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

UNITS = {"chinook": "Customer", "contoso_sqlite": "Customer", "adventureworks": "DimCustomer", "acs": None,
         "berka": "client"}


def _source(db: str) -> dict:
    sys.path.insert(0, str(ROOT))
    from bench.datasets import DATASETS
    kind, path = DATASETS[db].location().split(":", 1)
    return {"kind": kind, "path": path}


def one(db: str, eps: float, measure: str, rep: int, extra: dict) -> dict:
    import relsynth.dp as dp
    from dp_workload import workload_error
    from relsynth.api import EvaluationReport
    from relsynth.spec import read_spec
    raw = read_spec(ROOT / "bench" / "specs" / f"{db}.yaml")
    raw["source"] = _source(db)
    raw.setdefault("synthesis", {}).setdefault("privacy", {})["dp"] = {
        "unit": UNITS[db], "epsilon": eps, "delta": 1e-9, "public_tables": "all_unowned", "spec_is_public": True,
        "measure": measure, **extra}
    t = time.time()
    res = dp.synthesize(raw, write=False, privacy=False)
    er = EvaluationReport(res.report["scorecard"])
    work = {}
    for tb in res.statement["unit_meaning"].split("(")[1].rstrip(")").split(", "):
        if tb in res.real and tb in res.data:
            cols = [c for c, cs in raw["tables"][tb]["columns"].items()
                    if cs.get("type") in ("integer", "float", "date", "datetime", "categorical", "boolean")]
            if len(cols) >= 2:
                work[tb] = round(workload_error(res.real[tb], res.data[tb], cols)["tvd2"], 4)
    return {"db": db, "eps": eps, "measure": measure, "rep": rep, "passed": er.passed, "total": er.total,
            "score": round(er.score, 4), "tvd2": work, "eps_stated": res.epsilon, "secs": round(time.time() - t, 1),
            "pairs": {k: v.get("pairs") for k, v in res.report["tables"].items()}}


def summary(path: str) -> None:
    from scipy.stats import t as tdist
    rows = [json.loads(x) for x in open(path) if x.strip()]
    groups: dict = {}
    for r in rows:
        if "error" in r:
            continue
        groups.setdefault((r["db"], r["measure"], r["eps"]), []).append(r)
    print("| database | measure | epsilon | runs | checks passing (95 % CI) | mean two-way TVD (95 % CI) |")
    print("|---|---|---|---|---|---|")
    for (db, m, e), rs in sorted(groups.items()):
        def ci(xs):
            n = len(xs)
            mu = sum(xs) / n
            if n < 2:
                return f"{mu:.3f}"
            sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
            h = tdist.ppf(0.975, n - 1) * sd / math.sqrt(n)
            return f"{mu:.3f} ± {h:.3f}"
        tv = [sum(r["tvd2"].values()) / max(len(r["tvd2"]), 1) for r in rs]
        print(f"| {db} | {m} | {e:g} | {len(rs)} | {ci([r['score'] for r in rs])} | {ci(tv)} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", nargs="+", default=["chinook"])
    ap.add_argument("--eps", nargs="+", type=float, default=[0.5, 1, 4, 8])
    ap.add_argument("--measure", nargs="+", default=["copula", "mst", "aim"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--set", nargs="*", default=[], help="extra dp keys, key=value (YAML values)")
    ap.add_argument("--out", default="dp-curves.jsonl")
    ap.add_argument("--summary", metavar="JSONL")
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
        return 0
    import yaml
    extra = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    for db in a.db:
        for eps in a.eps:
            for m in a.measure:
                for i in range(a.reps):
                    try:
                        rec = one(db, eps, m, i, extra)
                    except Exception as e:  # noqa: BLE001
                        rec = {"db": db, "eps": eps, "measure": m, "rep": i, "error": repr(e)[:500],
                               "tb": traceback.format_exc()[-1500:]}
                    print(json.dumps(rec), flush=True)
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
