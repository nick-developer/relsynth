# SPDX-License-Identifier: Apache-2.0
"""Time-series benchmark of roadmap T12.

    python benchmarks/timeseries/run_timeseries.py electricity --seeds 1 2 3
    python benchmarks/timeseries/run_timeseries.py sequences --datasets contoso_sqlite adventureworks --seeds 1 2 3
    python benchmarks/timeseries/run_timeseries.py incremental

electricity: UCI ElectricityLoadDiagrams20112014 (CC BY 4.0) in the hourly
form of Lai et al. (SIGIR 2018; 321 clients, 2012-2014), downloaded once into
the bench cache and checked against a pinned SHA-256; the last year (2014,
321 x 8,760 rows) is synthesised through relsynth (a Meter entity and a Load
series table) and scored with the calibrated series metrics and day-ahead
forecasting TSTR.

sequences: the golden databases of the bench harness with a `sequence` block
on their order tables, run with and without it, scored with the calibrated
sequence metrics (inter-arrival, burstiness, life span, weekday, month, trend).

incremental: a Contoso base extended to 60 days in one run and day by day
(identical rows), and the distribution check of extension against one long
forward run (Prop. 1206).

Only aggregate numbers are written (--out, JSON); data stays in the cache.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.datasets import CACHE, DATASETS, download, DOWNLOADS  # noqa: E402

ELEC = ("electricity.txt.gz", "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/"
        "master/electricity/electricity.txt.gz",
        "3c4c069588198c1fcc95cace7bb69c99922129edfd673b7286661dad20badefa")
SEQUENCES = {"adventureworks": {"FactInternetSales": ("CustomerKey", "OrderDate"),
                                "FactResellerSales": ("ResellerKey", "OrderDate")},
             "contoso_sqlite": {"Orders": ("CustomerKey", "OrderDate")},
             "contoso_parquet": {"orders": ("CustomerKey", "DT")},
             "chinook": {"Invoice": ("CustomerId", "InvoiceDate")}}


def build_electricity(year_hours=8760) -> Path:
    out = CACHE / "electricity" / "electricity.db"
    if out.exists():
        return out
    DOWNLOADS.setdefault(ELEC[0], ELEC[1:])
    src = download(ELEC[0])
    Y = np.loadtxt(gzip.open(src), delimiter=",")[-year_hours:, :].T
    n, T = Y.shape
    ts = pd.date_range("2014-01-01", periods=T, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    pd.DataFrame({"MeterKey": np.arange(1, n + 1)}).to_sql("Meter", con, index=False)
    pd.DataFrame({"MeterKey": np.repeat(np.arange(1, n + 1), T), "Hour": np.tile(ts, n),
                  "kW": Y.reshape(-1)}).to_sql("Load", con, index=False, chunksize=200000)
    con.close()
    tmp.replace(out)
    return out


def electricity_spec(db):
    return {"version": 1, "name": "electricity", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {"Meter": {"role": "entity", "primary_key": "MeterKey", "columns": {"MeterKey": {"type": "id"}}},
                       "Load": {"role": "fact", "columns": {"MeterKey": {"type": "id"}, "Hour": {"type": "datetime"},
                                                            "kW": {"type": "float"}},
                                "fact": {"anchor": "Meter"},
                                "series": {"entity": "MeterKey", "time": "Hour", "values": ["kW"]}}},
            "relationships": [{"child": "Load", "child_key": "MeterKey", "parent": "Meter", "parent_key": "MeterKey"}],
            "synthesis": {"seed": 1}}


def run_electricity(seeds, splits):
    import relsynth as rs
    from relsynth.timeseries.metrics import series_checks, series_matrix
    db = build_electricity()
    real = pd.read_sql("select * from Load", sqlite3.connect(db))
    Yr, step = series_matrix(real, "MeterKey", "Hour", "kW")
    res = []
    for seed in seeds:
        spec = electricity_spec(db)
        spec["synthesis"]["seed"] = seed
        t0 = time.time()
        m = rs.fit(spec)
        t1 = time.time()
        d = m.sample()
        t2 = time.time()
        Ys, _ = series_matrix(d["Load"], "MeterKey", "Hour", "kW")
        checks, ts = series_checks(Yr, Ys, step, splits=splits, seed=seed)
        res.append({"seed": seed, "fit_s": round(t1 - t0, 1), "sample_s": round(t2 - t1, 1),
                    "rows": len(d["Load"]), "checks": [{k: c[k] for k in ("metric", "value", "limit", "pass")}
                                                       for c in checks], "tstr": ts,
                    "model": m.synthesizer.series["Load"]["model"].summary()})
        print(json.dumps(res[-1], default=float), flush=True)
    return res


def _events(df, ent, tc, hkey=None):
    from relsynth.timeseries.metrics import table_events
    if hkey and hkey in df.columns:
        df = df.drop_duplicates(hkey)
    return table_events(df, ent, tc)


def run_sequences(datasets, seeds, splits):
    import relsynth as rs
    from relsynth.spec import read_spec
    from relsynth.timeseries.metrics import sequence_checks
    out = []
    for ds in datasets:
        D = DATASETS[ds]
        loc = D.location()
        kind, path = loc.split(":", 1)
        for seed in seeds:
            for mode in ("1.0", "sequence"):
                spec = read_spec(D.spec())
                spec["source"] = {"kind": kind, "path": path}
                spec.setdefault("synthesis", {})["seed"] = seed
                if mode == "sequence":
                    for t, (e, c) in SEQUENCES[ds].items():
                        spec["tables"][t]["sequence"] = {"entity": e, "time": c}
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
                    ch = sequence_checks(er, es, splits=splits, seed=seed)
                    row["tables"][t] = {"passed": sum(1 for x in ch if x["pass"]),
                                        "total": sum(1 for x in ch if x["pass"] is not None),
                                        "checks": [{k: x[k] for k in ("metric", "value", "limit", "pass")} for x in ch]}
                print(json.dumps(row, default=float), flush=True)
                out.append(row)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("what", choices=["electricity", "sequences"])
    p.add_argument("--datasets", nargs="*", default=["contoso_sqlite", "adventureworks"])
    p.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    p.add_argument("--splits", type=int, default=20)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = run_electricity(a.seeds, a.splits) if a.what == "electricity" else run_sequences(a.datasets, a.seeds, a.splits)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
