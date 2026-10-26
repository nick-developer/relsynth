"""Run a privacy assessment on a benchmark and write the report.

    python benchmarks/privacy/run_assess.py SPEC OUT [--config privacy.yaml] [--csv table=file.csv ...]

The data is read from the spec's source (or from the CSV files given), never
copied into the repository.  Reports hold aggregates only, plus salted person
labels.
"""
import argparse
import json
import sys
import time

import pandas as pd

import relsynth as rs
from relsynth.privacy import assess, write_report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[], help="key=json, e.g. attacks.membership.shadow.n_jobs=4")
    ap.add_argument("--csv", action="append", default=[], help="table=file.csv (instead of the spec's source)")
    a = ap.parse_args(argv)
    cfg = {}
    if a.config:
        cfg = rs.privacy.load_config(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = json.loads(v)
    tables = {t: pd.read_csv(f) for t, f in (x.split("=", 1) for x in a.csv)} or None
    t0 = time.time()
    rep = assess(a.spec, cfg, tables=tables, progress=lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True))
    paths = write_report(rep, a.out)
    for n, c in rep["contexts"].items():
        print(n, c["verdict"])
        for ch in c["checks"]:
            lo, hi = ch["ci"] if ch["ci"] else (ch["value"], ch["value"])
            print(f"   {ch['criterion']:17s} {ch['attack'][:48]:48s} {ch['value']:.3f} [{lo:.3f}, {hi:.3f}] "
                  f"limit {ch['limit']} {ch['status']}")
    print(json.dumps(rep["timings"]), paths)


if __name__ == "__main__":
    sys.exit(main())
