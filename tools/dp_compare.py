# SPDX-License-Identifier: Apache-2.0
"""relsynth DP mode against MST and AIM of smartnoise-synth on one table (roadmap T07b).

    python tools/dp_compare.py --engine relsynth   --eps 0.5 1 4 8 --reps 3 --out cmp.jsonl
    python tools/dp_compare.py --engine smartnoise --eps 0.5 1 4 8 --reps 3 --out cmp.jsonl   # other venv
    python tools/dp_compare.py --summary cmp.jsonl

The table is the NIST CRC ACS excerpt of the bench (27,253 people; the data
MST won the NIST 2018 contest on), restricted to its 18 categorical and small
integer columns, every value a category.  Both engines are told the same public
domain (the levels of each column) and protect one row per person at the same
(epsilon, delta = 1e-9).  The measure is the workload error of
tools/dp_workload.py: mean total variation distance over all 153 two-way
marginals.  relsynth is scored on its stage 1 seed (what the measurements
alone give) and on the final synthetic table (after stage 2).

smartnoise-synth pins its own OpenDP (0.14) and pulls PyTorch, so it runs in a
virtual environment of its own; this file needs only pandas and numpy there.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sqlite3
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["PUMA", "AGEP", "SEX", "MSP", "HISP", "RAC1P", "NOC", "NPF", "HOUSING_TYPE", "OWN_RENT", "INDP_CAT", "EDU",
           "PINCP_DECILE", "DVET", "DREM", "DPHY", "DEYE", "DEAR"]


def _workload():
    spec = importlib.util.spec_from_file_location("dp_workload", ROOT / "tools" / "dp_workload.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.workload_error


def load(path: str | None = None):
    import pandas as pd
    db = path or os.environ.get("ACS_DB") or str(ROOT / "bench" / ".cache" / "acs" / "acs.db")
    with sqlite3.connect(db) as c:
        df = pd.read_sql("select * from acs", c)[COLUMNS]

    def s(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "NA"
        return str(int(x)) if isinstance(x, float) and float(x).is_integer() else str(x)
    return df.apply(lambda col: col.map(s))


def run_relsynth(df, eps: float, measure: str):
    import relsynth.dp as dp
    from relsynth.spec import load_spec
    d = df.copy()
    d.insert(0, "pid", range(1, len(d) + 1))
    spec = {"version": 1, "name": "acs_cmp", "source": {"kind": "csv", "path": "."},
            "tables": {"acs": {"role": "entity", "primary_key": "pid",
                               "columns": {"pid": {"type": "id"},
                                           **{c: {"type": "categorical"} for c in COLUMNS}}}},
            "relationships": [],
            "synthesis": {"privacy": {"dp": {
                "unit": "acs", "epsilon": eps, "delta": 1e-9, "spec_is_public": True, "measure": measure,
                "public": {f"acs.{c}": {"levels": sorted(df[c].unique())} for c in COLUMNS}}}}}
    r = dp.synthesize(load_spec(spec), tables={"acs": d}, write=False, evaluate=False)
    pairs = r.report["tables"]["acs"].get("pairs")
    return {"seed": r.seed["acs"][COLUMNS].astype(str), "synth": r.data["acs"][COLUMNS].astype(str)}, pairs


def run_smartnoise(df, eps: float, measure: str):
    from snsynth import Synthesizer
    s = Synthesizer.create(measure, epsilon=eps, delta=1e-9)
    s.fit(df, preprocessor_eps=0.0, categorical_columns=list(df.columns))
    return {"synth": s.sample(len(df)).astype(str)}, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["relsynth", "smartnoise"])
    ap.add_argument("--measure", nargs="+")
    ap.add_argument("--eps", nargs="+", type=float, default=[0.5, 1, 4, 8])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="dp-compare.jsonl")
    ap.add_argument("--acs-db")
    ap.add_argument("--summary", metavar="JSONL")
    a = ap.parse_args()
    if a.summary:
        return summary(a.summary)
    W = _workload()
    df = load(a.acs_db)
    measures = a.measure or (["copula", "mst", "aim"] if a.engine == "relsynth" else ["mst", "aim"])
    fn = run_relsynth if a.engine == "relsynth" else run_smartnoise
    for eps in a.eps:
        for m in measures:
            for i in range(a.reps):
                t = time.time()
                try:
                    outs, pairs = fn(df, eps, m)
                    rec = {"engine": a.engine, "measure": m, "eps": eps, "rep": i, "secs": round(time.time() - t, 1),
                           **{f"tvd2_{k}": round(W(df, v, COLUMNS)["tvd2"], 4) for k, v in outs.items()},
                           **{f"tvd1_{k}": round(W(df, v, COLUMNS)["tvd1"], 4) for k, v in outs.items()}}
                    if pairs:
                        rec["pairs"] = len(pairs)
                except Exception as e:  # noqa: BLE001
                    rec = {"engine": a.engine, "measure": m, "eps": eps, "rep": i, "error": repr(e)[:400],
                           "tb": traceback.format_exc()[-1200:]}
                print(json.dumps(rec), flush=True)
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    return 0


def summary(path: str) -> int:
    rows = [json.loads(x) for x in open(path) if x.strip()]
    g: dict = {}
    for r in rows:
        if "error" not in r:
            g.setdefault((r["engine"], r["measure"], r["eps"]), []).append(r)
    eps = sorted({k[2] for k in g})
    print("| engine / measure | " + " | ".join(f"ε = {e:g}" for e in eps) + " |")
    print("|---|" + "---|" * len(eps))
    for eng, m in sorted({k[:2] for k in g}):
        for key in ("tvd2_synth", "tvd2_seed"):
            cells = []
            for e in eps:
                rs = [r[key] for r in g.get((eng, m, e), []) if key in r]
                if not rs:
                    cells.append("")
                    continue
                mu = sum(rs) / len(rs)
                sd = math.sqrt(sum((x - mu) ** 2 for x in rs) / (len(rs) - 1)) if len(rs) > 1 else 0.0
                cells.append(f"{mu:.3f} ± {1.96 * sd / math.sqrt(len(rs)):.3f}")
            if any(cells):
                label = f"{eng} {m}" + (" (stage 1 seed)" if key == "tvd2_seed" else "")
                print(f"| {label} | " + " | ".join(cells) + " |")
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"\n{len(errs)} runs failed, first: {errs[0]['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
