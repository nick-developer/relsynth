# SPDX-License-Identifier: Apache-2.0
"""DP stage 1 in the database against in memory, on the bench databases (gap plan G2).

    python tools/dp_pushdown_compare.py --db contoso_sqlite adventureworks --eps 1 --reps 3 --out g2.jsonl
    python tools/dp_pushdown_compare.py --summary g2.jsonl

Every run is a fresh process, so its peak resident memory is its own: `rss_stage1_mb`
is the peak at the end of stage 1 (the in-memory run has loaded every table by then),
`rss_mb` the peak of the whole run including stage 2 and the scorecard.  It records
the share of scorecard checks that pass, the mean two-way TVD of the owned tables
(tools/dp_workload.py) and the time of stage 1.  The bench data must be in the
cache first (`python -m bench fetch --datasets ...`).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def one(db: str, eps: float, mode: str, measure: str, rep: int) -> dict:
    sys.path.insert(0, str(ROOT / "tools"))
    import resource

    from dp_curves import UNITS, _source
    from dp_workload import workload_error

    import relsynth.dp as dp
    from relsynth.api import EvaluationReport
    from relsynth.spec import read_spec
    raw = read_spec(ROOT / "bench" / "specs" / f"{db}.yaml")
    raw["source"] = _source(db)
    raw.setdefault("synthesis", {}).setdefault("privacy", {})["dp"] = {
        "unit": UNITS[db], "epsilon": eps, "delta": 1e-9, "public_tables": "all_unowned", "spec_is_public": True,
        "measure": measure, "pushdown": mode}
    res = dp.synthesize(raw, write=False, privacy=False)
    er = EvaluationReport(res.report["scorecard"])
    work = {}
    for tb in res.statement["unit_meaning"].split("(")[1].rstrip(")").split(", "):
        if res.real is not None and tb in res.real and tb in res.data:
            cols = [c for c, cs in raw["tables"][tb]["columns"].items()
                    if cs.get("type") in ("integer", "float", "date", "datetime", "categorical", "boolean")]
            if len(cols) >= 2:
                work[tb] = round(workload_error(res.real[tb], res.data[tb], cols)["tvd2"], 4)
    s1 = res.report.get("stage1", {})
    return {"db": db, "eps": eps, "mode": mode, "measure": measure, "rep": rep, "source": s1.get("source"),
            "passed": er.passed, "total": er.total, "score": round(er.score, 4), "tvd2": work,
            "eps_stated": res.epsilon, "secs_stage1": res.report.get("seconds_stage1"),
            "secs": res.report.get("seconds_total"), "rss_stage1_mb": s1.get("peak_rss_mb"),
            "rss_before_mb": s1.get("peak_rss_before_mb"),
            "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "max_degree": {k: v.get("max_degree") for k, v in res.report["tables"].items() if v.get("max_degree")}}


def _ci(xs):
    from scipy.stats import t as tdist
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    return mu, tdist.ppf(0.975, n - 1) * sd / math.sqrt(n)


def summary(path: str) -> None:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    groups: dict = {}
    for r in rows:
        if "error" not in r:
            groups.setdefault((r["db"], r["measure"], r["eps"], r["mode"]), []).append(r)
    print("| database | measure | epsilon | stage 1 | runs | checks passing (95 % CI) | mean two-way TVD | "
          "stage 1 s | peak memory, stage 1 (MB) | peak memory, run (MB) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (db, m, e, _mode), rs in sorted(groups.items()):
        mu, h = _ci([r["score"] for r in rs])
        tv = [sum(r["tvd2"].values()) / len(r["tvd2"]) for r in rs if r["tvd2"]]
        tvs = "" if not tv else "{:.3f} ± {:.3f}".format(*_ci(tv))
        print(f"| {db} | {m} | {e:g} | {rs[0]['source']} | {len(rs)} | {mu:.3f} ± {h:.3f} | {tvs} | "
              f"{_ci([r['secs_stage1'] for r in rs])[0]:.0f} | {max(r['rss_stage1_mb'] for r in rs):.0f} | "
              f"{max(r['rss_mb'] for r in rs):.0f} |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", nargs="+", default=["contoso_sqlite", "adventureworks"])
    ap.add_argument("--eps", nargs="+", type=float, default=[1.0])
    ap.add_argument("--mode", nargs="+", default=["off", "on"])
    ap.add_argument("--measure", nargs="+", default=["copula"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="dp_pushdown.jsonl")
    ap.add_argument("--summary")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.summary:
        summary(a.summary)
        return 0
    if a.one:
        db, eps, mode, measure, rep = json.loads(a.one)
        print(json.dumps(one(db, eps, mode, measure, rep)))
        return 0
    with open(a.out, "a", encoding="utf-8") as f:
        for rep in range(a.reps):
            for db in a.db:
                for eps in a.eps:
                    for measure in a.measure:
                        for mode in a.mode:
                            arg = json.dumps([db, eps, mode, measure, rep])
                            p = subprocess.run([sys.executable, __file__, "--one", arg], capture_output=True,  # noqa: S603
                                               text=True, check=False)
                            line = p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() \
                                else json.dumps({"db": db, "eps": eps, "mode": mode, "measure": measure,
                                                 "rep": rep, "error": p.stderr[-2000:]})
                            print(line, flush=True)
                            f.write(line + "\n")
                            f.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
