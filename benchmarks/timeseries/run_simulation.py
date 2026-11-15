# SPDX-License-Identifier: Apache-2.0
"""Benchmarks of the simulation features (docs/scenarios.md, docs/timeseries.md):

    python benchmarks/timeseries/run_simulation.py processes --datasets adventureworks contoso_parquet --seeds 1
    python benchmarks/timeseries/run_simulation.py scenarios --datasets contoso_parquet adventureworks
    python benchmarks/timeseries/run_simulation.py electricity

processes: the sequence tables of the golden databases with process renewal
and hawkes (and, where the table has a status column, a status chain), scored
with the calibrated sequence metrics, plus the status transition distance.

scenarios: one fit per database, then the baseline and one dataset per
intervention type; each row says what was asked, what the dataset shows and
what the scenario record says.  Nothing is tuned to a database: the
interventions are written below from the spec's own tables and columns.

electricity: the series table with a date-window intensity (a heat wave: +30%
load for two weeks); the values in and out of the window against the baseline.

Only aggregate numbers are written (--out, JSON); data stays in the cache.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench.datasets import DATASETS  # noqa: E402
from run_timeseries import SEQUENCES, _events, build_electricity, electricity_spec  # noqa: E402

STATES = {}          # no golden database has a status column along its order histories (synthetic checks only)
SCENARIOS = {
    "contoso_parquet": [
        {"type": "scale", "table": "customer", "factor": 1.25},
        {"type": "shares", "table": "customer", "column": "Gender", "shares": {"male": 0.3}},
        {"type": "shares", "table": "customer", "column": "Gender", "factors": {"female": 2.0}},
        {"type": "marginal", "table": "orderrows", "column": "NetPrice", "quantiles": [[0.9, 1.0], [1.0, 1.2]]},
        {"type": "intensity", "table": "orders", "column": "DT", "start": "2019-11-01", "end": "2020-01-01",
         "factor": 2.0},
        {"type": "degree", "table": "orderrows", "key": "ProductKey", "concentration": 2.0},
        {"type": "degree", "table": "orderrows", "key": "OrderKey", "rate": 1.5},
    ],
    "adventureworks": [
        {"type": "scale", "table": "DimCustomer", "factor": 1.5},
        {"type": "shares", "table": "DimCustomer", "column": "MaritalStatus", "shares": {"S": 0.7}},
        {"type": "marginal", "table": "FactInternetSales", "column": "SalesAmount", "multiply": 1.1},
        {"type": "degree", "table": "FactInternetSales", "key": "ProductKey", "concentration": 0.5},
    ],
}


def _spec(ds, seed):
    from relsynth.spec import read_spec
    D = DATASETS[ds]
    kind, path = D.location().split(":", 1)
    spec = read_spec(D.spec())
    spec["source"] = {"kind": kind, "path": path}
    spec.setdefault("synthesis", {})["seed"] = seed
    return spec


def _transitions(df, ent, tc, col, levels):
    d = df[[ent, tc, col]].dropna(subset=[tc]).copy()
    d["_t"] = pd.to_datetime(d[tc], errors="coerce")
    d = d.sort_values([ent, "_t"], kind="stable")
    c = pd.Categorical(d[col].astype(str), categories=levels).codes
    e = d[ent].to_numpy()
    N = np.zeros((len(levels), len(levels)))
    same = (e[1:] == e[:-1]) & (c[1:] >= 0) & (c[:-1] >= 0)
    np.add.at(N, (c[:-1][same], c[1:][same]), 1)
    rows = N.sum(1, keepdims=True)
    return N / np.maximum(rows, 1), rows.ravel()


def run_processes(datasets, seeds, splits):
    import relsynth as rs
    from relsynth.timeseries.metrics import sequence_checks
    out = []
    for ds in datasets:
        for seed in seeds:
            modes = ["renewal", "hawkes"] + (["states"] if ds in STATES else [])
            for mode in modes:
                spec = _spec(ds, seed)
                for t, (e, c) in SEQUENCES[ds].items():
                    sq = {"entity": e, "time": c}
                    if mode == "hawkes":
                        sq["process"] = "hawkes"
                    if mode == "states" and t in STATES.get(ds, {}):
                        sq["states"] = STATES[ds][t]
                    spec["tables"][t]["sequence"] = sq
                t0 = time.time()
                m = rs.fit(spec)
                d = m.sample()
                real = m.fit_data.tables
                row = {"dataset": ds, "seed": seed, "mode": mode, "seconds": round(time.time() - t0, 1),
                       "log": d.sample_log.get("sequences"), "tables": {}}
                for t, (e, c) in SEQUENCES[ds].items():
                    hkey = ((spec["tables"][t].get("fact") or {}).get("header") or {}).get("key")
                    er, es = _events(real[t], e, c, hkey), _events(d[t], e, c, hkey)
                    es.t0, es.t1 = er.t0, er.t1
                    ch = sequence_checks(er, es, splits=splits, seed=seed, predictive=True)
                    info = m.synthesizer.seq.get(t) or next(
                        (v for v in m.synthesizer.seq.values() if v["entity"] == e), None)
                    fl = info["model"].fit_log if info else {}
                    row["tables"][t] = {"passed": sum(1 for x in ch if x["pass"]),
                                        "total": sum(1 for x in ch if x["pass"] is not None),
                                        "checks": {x["metric"]: [x["value"], x["limit"], x["pass"]] for x in ch},
                                        "hawkes": (fl.get("hawkes") or {}).get("pooled")}
                    sc = STATES.get(ds, {}).get(t)
                    if sc:
                        levels = sorted(real[t][sc].dropna().astype(str).unique())
                        Pr, nr = _transitions(real[t], e, c, sc, levels)
                        Ps, _ = _transitions(d[t], e, c, sc, levels)
                        w = nr / nr.sum()
                        row["tables"][t]["transition_tvd"] = round(float((w * 0.5 * np.abs(Pr - Ps).sum(1)).sum()), 4)
                        row["tables"][t]["stay_real"] = round(float((w * np.diag(Pr)).sum()), 4)
                        row["tables"][t]["stay_syn"] = round(float((w * np.diag(Ps)).sum()), 4)
                print(json.dumps(row, default=float), flush=True)
                out.append(row)
    return out


def _measure(iv, base, d):
    """What the dataset shows for intervention iv, against the baseline."""
    from relsynth.scenarios import _gini
    from relsynth.timeseries.calendar import to_days
    t, k = iv["table"], iv["type"]
    if k == "scale":
        return {"rows_ratio": round(len(d[t]) / max(len(base[t]), 1), 4)}
    if k == "shares":
        col = iv["column"]
        sb = base[t][col].astype(str).value_counts(normalize=True).round(4).to_dict()
        ss = d[t][col].astype(str).value_counts(normalize=True).round(4).to_dict()
        return {"shares_base": sb, "shares": ss, "rows_ratio": round(len(d[t]) / max(len(base[t]), 1), 4)}
    if k == "marginal":
        a = pd.to_numeric(base[t][iv["column"]], errors="coerce").dropna()
        b = pd.to_numeric(d[t][iv["column"]], errors="coerce").dropna()
        return {q: round(float(b.quantile(p) / a.quantile(p)), 4) for q, p in
                (("p50_ratio", 0.5), ("p90_ratio", 0.9), ("p99_ratio", 0.99))} | {
                    "mean_ratio": round(float(b.mean() / a.mean()), 4)}
    if k == "intensity":
        lo, hi = to_days(pd.Series([str(iv["start"]), str(iv["end"])]))

        def split(df):
            x = to_days(df[iv["column"]])
            w = (x >= lo) & (x < hi)
            return int(w.sum()), int((~w & np.isfinite(x)).sum())
        (ib, ob), (i2, o2) = split(base[t]), split(d[t])
        return {"in_window_ratio": round(i2 / max(ib, 1), 4), "outside_ratio": round(o2 / max(ob, 1), 4)}
    if k == "degree":
        res = {"rows_ratio": round(len(d[t]) / max(len(base[t]), 1), 4)}
        key = iv["key"]
        for name, df in (("gini_base", base[t]), ("gini", d[t])):
            res[name] = round(_gini(df[key].value_counts().to_numpy()), 4)
        return res
    return {}


def run_scenarios(datasets, seed):
    import relsynth as rs
    out = []
    for ds in datasets:
        spec = _spec(ds, seed)
        ivs = SCENARIOS[ds]
        spec["synthesis"]["scenarios"] = {f"s{i}": {"interventions": [iv]} for i, iv in enumerate(ivs)}
        t0 = time.time()
        m = rs.fit(spec)
        fit_s = round(time.time() - t0, 1)
        base = m.sample(dataset=0)
        for i, iv in enumerate(ivs):
            t1 = time.time()
            d = m.sample(dataset=0, scenario=f"s{i}")
            rec = d.sample_log["scenario"]["interventions"][0]
            row = {"dataset": ds, "seed": seed, "fit_s": fit_s, "sample_s": round(time.time() - t1, 1),
                   "intervention": iv, "measured": _measure(iv, base, d), "record": rec["effect"]}
            print(json.dumps(row, default=float), flush=True)
            out.append(row)
    return out


def run_electricity_scenario(seed, splits):
    import relsynth as rs
    from relsynth.timeseries.calendar import to_days
    from relsynth.timeseries.metrics import series_checks, series_matrix
    db = build_electricity()
    real = pd.read_sql("select * from Load", sqlite3.connect(db))
    t = pd.to_datetime(real["Hour"])
    mid = t.min() + (t.max() - t.min()) / 2
    start, end = mid.normalize(), mid.normalize() + pd.Timedelta(days=14)
    spec = electricity_spec(db)
    spec["synthesis"]["seed"] = seed
    spec["synthesis"]["scenarios"] = {"heatwave": {"interventions": [
        {"type": "intensity", "table": "Load", "start": str(start.date()), "end": str(end.date()), "factor": 1.3}]}}
    m = rs.fit(spec)
    base = m.sample(dataset=0)
    d = m.sample(dataset=0, scenario="heatwave")
    lo, hi = to_days(pd.Series([str(start.date()), str(end.date())]))
    x = to_days(base["Load"]["Hour"])
    w = (x >= lo) & (x < hi)
    rb, rs_ = base["Load"]["kW"].to_numpy(), d["Load"]["kW"].to_numpy()
    Yr, step = series_matrix(real, "MeterKey", "Hour", "kW")
    Ys, _ = series_matrix(base["Load"], "MeterKey", "Hour", "kW")
    checks, _ = series_checks(Yr, Ys, step, splits=splits, seed=seed)
    row = {"dataset": "electricity", "seed": seed, "window": [str(start.date()), str(end.date())],
           "in_window_ratio": round(float(rs_[w].sum() / rb[w].sum()), 4),
           "outside_identical": bool(np.array_equal(rs_[~w], rb[~w])),
           "record": d.sample_log["scenario"]["interventions"][0]["effect"],
           "baseline_checks_passed": f"{sum(1 for c in checks if c['pass'])}/{len(checks)}"}
    print(json.dumps(row, default=float), flush=True)
    return [row]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["processes", "scenarios", "electricity"])
    p.add_argument("--datasets", nargs="*", default=["adventureworks", "contoso_parquet"])
    p.add_argument("--seeds", nargs="*", type=int, default=[1])
    p.add_argument("--splits", type=int, default=20)
    p.add_argument("--out", help="JSON results file")
    a = p.parse_args(argv)
    if a.what == "processes":
        res = run_processes(a.datasets, a.seeds, a.splits)
    elif a.what == "scenarios":
        res = run_scenarios(a.datasets, a.seeds[0])
    else:
        res = run_electricity_scenario(a.seeds[0], a.splits)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
