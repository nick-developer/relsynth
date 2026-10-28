# SPDX-License-Identifier: Apache-2.0
"""Baseline and regression rule of the benchmark harness.

A baseline summarises history records of the same code over several seeds (at
least 5 for the real databases).  For each dataset it keeps, per seed, the
pass rate (passed / judged checks), and per check the seeds on which it passed.

A new record regresses on a dataset when
  R1  a *calibrated* check fails: one that passed on every baseline seed
      (at least 3) and is judged in the new run, or
  R2  the mean pass rate over the new seeds drops below the baseline mean by
      more than the run-to-run spread, spread = max(2 sd, 1 / judged checks):
      two standard deviations of the per-seed pass rates (a 95% band if the rates
      were normal), and never less than one check, or
  R3  a run errors, or a dataset the baseline has is missing from a full run.

Checks that failed on some baseline seed are "unstable": they count in R2 but
cannot trigger R1, so a borderline statistical check flipping with the seed is
not reported as a regression, while an identity or a check that always passed
is.  The rule is a heuristic chosen for this harness, not a test with a stated
level; docs/development/benchmarks.md explains the choice.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

MIN_SEEDS_CALIBRATED = 3


def _runs(records: list[dict]):
    for rec in records:
        for r in rec.get("runs", []):
            if "checks" in r:
                yield r


def judged(r: dict) -> list[dict]:
    return [c for c in r.get("checks", []) if c.get("pass") is not None]


def rate(r: dict) -> float:
    j = judged(r)
    return sum(c["pass"] for c in j) / len(j) if j else float("nan")


def build(records: list[dict], sources: list[str] | None = None) -> dict:
    by_ds: dict = defaultdict(list)
    for r in _runs(records):
        by_ds[r["dataset"]].append(r)
    out = {"schema": 1, "sources": sources or [], "relsynth": sorted({rec.get("relsynth") for rec in records}),
           "git": sorted({(rec.get("git") or {}).get("sha", "")[:12] for rec in records}),
           "machines": sorted({(rec.get("machine") or {}).get("node", "") for rec in records}), "datasets": {}}
    for ds, runs in sorted(by_ds.items()):
        rates = [rate(r) for r in runs]
        n_judged = statistics.median(len(judged(r)) for r in runs) or 1
        sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
        per_check: dict = defaultdict(lambda: {"passed": 0, "judged": 0, "failed_seeds": []})
        for r in runs:
            for c in judged(r):
                pc = per_check[c["id"]]
                pc["judged"] += 1
                if c["pass"]:
                    pc["passed"] += 1
                else:
                    pc["failed_seeds"].append(r.get("seed"))
        out["datasets"][ds] = {
            "seeds": [r.get("seed") for r in runs], "pass_rates": [round(x, 6) for x in rates],
            "passed": [sum(c["pass"] for c in judged(r)) for r in runs], "judged": [len(judged(r)) for r in runs],
            "mean": round(statistics.fmean(rates), 6), "sd": round(sd, 6),
            "spread": round(max(2 * sd, 1 / n_judged), 6),
            "seconds": [r.get("seconds") for r in runs],
            "checks": {k: {"passed": v["passed"], "judged": v["judged"],
                           "calibrated": v["passed"] == v["judged"] >= MIN_SEEDS_CALIBRATED or
                           (ds == "fake" and v["passed"] == v["judged"] >= 1),
                           **({"failed_seeds": v["failed_seeds"]} if v["failed_seeds"] else {})}
                       for k, v in sorted(per_check.items())},
            "errors": sum(1 for r in runs if r.get("error"))}
    return out


def compare(baseline: dict, record: dict, full: bool = False) -> tuple[list[str], list[str], dict]:
    """(regressions, notes, per-dataset summary) of a record against a baseline."""
    regressions, notes, summary = [], [], {}
    runs: dict = defaultdict(list)
    for r in record.get("runs", []):
        if "skipped" in r:
            notes.append(f"{r['dataset']}: skipped ({r['skipped']})")
            continue
        if r.get("private"):
            notes.append(f"{r['dataset']} seed {r.get('seed')}: private result, not available here")
            continue
        runs[r["dataset"]].append(r)
    for ds, rs in sorted(runs.items()):
        b = baseline["datasets"].get(ds)
        for r in rs:
            if r.get("error"):
                regressions.append(f"R3 {ds} seed {r.get('seed')}: run failed: {r['error'][:300]}")
        if b is None:
            notes.append(f"{ds}: not in the baseline; recorded only")
            continue
        rates = [rate(r) for r in rs if r.get("checks")]
        mean = statistics.fmean(rates) if rates else float("nan")
        summary[ds] = {"seeds": [r.get("seed") for r in rs], "mean": round(mean, 6), "baseline_mean": b["mean"],
                       "spread": b["spread"], "passed": [sum(c["pass"] for c in judged(r)) for r in rs],
                       "judged": [len(judged(r)) for r in rs]}
        if rates and mean < b["mean"] - b["spread"] - 1e-12:
            regressions.append(f"R2 {ds}: mean pass rate {mean:.4f} < baseline {b['mean']:.4f} - spread {b['spread']:.4f}")
        for r in rs:
            seen = set()
            for c in judged(r):
                seen.add(c["id"])
                bc = b["checks"].get(c["id"])
                if bc and bc["calibrated"] and not c["pass"]:
                    v = f" (value {c.get('value')}, limit {c.get('limit')}, real {c.get('real')})"
                    regressions.append(f"R1 {ds} seed {r.get('seed')}: calibrated check failed: {c['id']}{v}")
            gone = [k for k, v in b["checks"].items() if v["calibrated"] and k not in seen]
            if gone and r.get("checks"):
                notes.append(f"{ds} seed {r.get('seed')}: {len(gone)} calibrated check(s) not reported, e.g. {gone[0]}")
    if full:
        for ds in baseline["datasets"]:
            if ds not in runs and not any(n.startswith(f"{ds}: skipped") or n.startswith(f"{ds} seed") for n in notes):
                regressions.append(f"R3 {ds}: in the baseline but not run")
    return regressions, notes, summary


def report(regressions: list[str], notes: list[str], summary: dict) -> str:
    lines = ["## Benchmark comparison", "",
             "| Dataset | Seeds | Passed / judged | Mean pass rate | Baseline mean | Spread |", "|---|---|---|---|---|---|"]
    for ds, s in summary.items():
        pj = ", ".join(f"{p}/{j}" for p, j in zip(s["passed"], s["judged"]))
        lines.append(f"| {ds} | {', '.join(map(str, s['seeds']))} | {pj} | {s['mean']:.4f} | {s['baseline_mean']:.4f} | "
                     f"{s['spread']:.4f} |")
    lines += ["", f"**{'REGRESSION' if regressions else 'No regression'}**", ""]
    lines += [f"- {x}" for x in regressions]
    if notes:
        lines += ["", "Notes:", ""] + [f"- {x}" for x in notes]
    return "\n".join(lines) + "\n"


def load(paths: list[str | Path]) -> list[dict]:
    from .runner import load_record
    out = []
    for p in paths:
        p = Path(p)
        for f in sorted(p.glob("*.json")) if p.is_dir() else [p]:
            out.append(load_record(f))
    return out
