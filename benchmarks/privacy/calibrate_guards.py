# SPDX-License-Identifier: Apache-2.0
"""Calibrate relsynth's distance guards (DCR ratio, exact-match rate) against
the attacks.

One relsynth release is fitted on the training people.  Leaky releases are made
from it by replacing a share p of its rows with training rows, either exact
copies or near copies (numbers moved by a small relative jitter, categories
kept).  For each p the guards and the attacks are run on the same release, so
the table shows which leak each one detects.

    python benchmarks/privacy/calibrate_guards.py SPEC.yaml OUT.json [--table T] [--reps 3]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import relsynth as rs
from relsynth.evaluate import MODELLED, _eff, dcr, determining_columns
from relsynth.privacy import criteria, mia
from relsynth.privacy.assess import relsynth_generator

SHARES = (0.0, 0.001, 0.005, 0.02, 0.05, 0.2)


def exact_limit(d, base=0.01):
    """The scorecard's exact-match limit (evaluate.scorecard): real repeat rate + 2.5 sd."""
    e = d["exact_match_rate_real_any"]
    sd = float(np.sqrt(max(e * (1 - e), 1 / max(d["n"], 1)) * _eff(d["n"], d["m"])))
    return max(base, e + 2.5 * sd)


def leak(syn, train, p, rng, near):
    k = int(round(p * len(syn)))
    if k == 0:
        return syn
    rows = train.sample(k, random_state=int(rng.integers(2**31))).copy()
    if near:
        for c in rows.columns:
            if rows[c].dtype.kind in "fi" and rows[c].nunique() > 20:
                x = rows[c].astype(float)
                rows[c] = (x * (1 + rng.normal(0, 0.02, len(x)))).round(0 if rows[c].dtype.kind == "i" else 2)
                if train[c].dtype.kind == "i":
                    rows[c] = rows[c].astype("int64")
    out = syn.copy()
    idx = rng.choice(len(out), k, replace=False)
    out.iloc[idx] = rows[out.columns].to_numpy()
    return out.astype(syn.dtypes.to_dict(), errors="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--table")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=20000)
    a = ap.parse_args()
    s = rs.Spec.load(a.spec).resolved
    from relsynth.pushdown import load_for_fit
    t = a.table or next(iter(s["tables"]))
    ts = s["tables"][t]
    df = load_for_fit(s, mode="off").tables[t].reset_index(drop=True)
    df = df.sample(min(len(df), a.max_rows), random_state=0).reset_index(drop=True)
    n = len(df)
    tr, ctl, ref = df.iloc[: int(.4 * n)], df.iloc[int(.4 * n): int(.8 * n)], df.iloc[int(.8 * n):]
    keycols = {r["child_key"] for r in s["relationships"] if r["child"] == t}
    cols = [c for c, cs in ts["columns"].items() if cs["type"] in MODELLED and c not in keycols
            and c != ts.get("primary_key")]
    t0 = time.time()
    base = relsynth_generator(s)({t: tr}, 11)[t]
    fit_s = time.time() - t0
    pc = determining_columns(tr, ts, cols)
    results = []
    for near in (False, True):
        for p in SHARES:
            for rep in range(a.reps):
                r = np.random.default_rng(100 * rep + int(1000 * p))
                syn = leak(base, tr, p, r, near)
                d = dcr(tr, syn, ts, pc)
                X, Y, Z = tr[cols], syn[cols].reset_index(drop=True), ctl[cols]
                so = criteria.record_isolation(X, Y, Z, mode="multivariate", n_attacks=300, seed=rep)
                dom = mia.domias(X, Z, Y, ref[cols], seed=rep)
                row = {"near_copies": near, "share": p, "rep": rep,
                       "dcr_ratio": d["dcr_ratio"], "dcr_flag": d["dcr_ratio"] < 0.5,
                       "exact_match_rate": d["exact_match_rate"], "exact_flag": d["exact_match_rate"] > exact_limit(d),
                       "singling_out_risk": so.risk.value, "singling_out_ci": list(so.risk.ci),
                       "singling_out_flag": so.risk.ci[0] > 0,
                       "domias_auc": dom.get("auc"), "domias_auc_ci": dom.get("auc_ci"),
                       "domias_flag": bool(dom.get("auc_ci") and dom["auc_ci"][0] > 0.5),
                       "domias_tpr_at_fpr_0.01": next((e.get("tpr") for e in dom.get("tpr_at_fpr", [])
                                                       if e["fpr_target"] == 0.01), None)}
                results.append(row)
                print(json.dumps(row), flush=True)
    tab = pd.DataFrame(results)
    summ = tab.groupby(["near_copies", "share"])[["dcr_flag", "exact_flag", "singling_out_flag", "domias_flag",
                                                  "dcr_ratio", "exact_match_rate", "singling_out_risk",
                                                  "domias_auc"]].mean()
    print(summ.to_string())
    json.dump({"table": t, "rows": n, "fit_seconds": fit_s, "results": results,
               "summary": json.loads(summ.reset_index().to_json(orient="records"))}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
