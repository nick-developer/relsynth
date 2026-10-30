# SPDX-License-Identifier: Apache-2.0
"""Baseline and regression rule of the benchmark harness.

A baseline summarises history records of the same code over several seeds (at
least 5 for the real databases).  For each dataset it keeps, per seed, the
pass rate (passed / judged checks), and per check the seeds on which it passed.

A new record regresses on a dataset when
  R1  a *calibrated* check fails: one that passed on every baseline seed
      (at least 3) and is judged in the new run.  An exact check (an identity,
      or a scorecard item whose limit is 0) regresses on any failing seed; a
      statistical check (a distance under a positive limit) only when it fails
      on more than half of the new seeds, or
  R2  the mean pass rate over the new seeds drops below the baseline mean by
      more than the run-to-run spread, spread = max(2 sd, 1 / judged checks):
      two standard deviations of the per-seed pass rates (a 95% band if the rates
      were normal), and never less than one check, or
  R3  a run errors, or a dataset the baseline has is missing from a full run, or
  R4  a tracked value (a `metric:` item: SyntheRela summaries, area pass rates,
      utility gap, attack values) moves the wrong way: its mean over the new
      seeds is worse than the baseline mean by more than
      max(0.02 x max(1, |baseline mean|), 2 sd of the baseline seeds) -- the
      "benchmark regression flag" of docs/acceptance-metrics.md.  Cost values
      (CPU seconds, memory) depend on the machine and are only noted when they
      grow by more than 25%.

Checks that failed on some baseline seed are "unstable": they count in R2 but
cannot trigger R1, so a borderline statistical check flipping with the seed is
not reported as a regression, while an identity or a check that always passed
is.  Five baseline seeds cannot show that a statistical check never fails: with
about 1,300 of them, a few that passed five times by chance fail on one seed of
another machine (BLAS and thread counts change the draws).  A broken table
fails on most seeds, which the majority rule still reports.  The rule is a
heuristic chosen for this harness, not a test with a stated level;
docs/development/benchmarks.md explains the choice.
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


def exact(c: dict) -> bool:
    """An identity (judged against the real value) or a scorecard item with limit 0."""
    return "limit" not in c or not c["limit"]


def tracked(r: dict) -> list[dict]:
    return [c for c in r.get("checks", []) if c.get("pass") is None and c.get("direction") and c.get("value") is not None]


R4_FLOOR, COST_NOTE = 0.02, 0.25


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
        vals: dict = defaultdict(list)
        meta: dict = {}
        for r in runs:
            for c in tracked(r):
                vals[c["id"]].append(float(c["value"]))
                meta[c["id"]] = (c["direction"], c.get("kind"))
        if vals:
            out["datasets"][ds]["tracked"] = {
                k: {"mean": round(statistics.fmean(v), 6), "sd": round(statistics.stdev(v), 6) if len(v) > 1 else 0.0,
                    "n": len(v), "direction": meta[k][0], **({"kind": meta[k][1]} if meta[k][1] else {})}
                for k, v in sorted(vals.items())}
    return out


def drift(b: dict, rs: list[dict]) -> tuple[list[str], list[str]]:
    """R4: tracked values whose new mean is worse than the baseline's beyond the spread."""
    regs, notes = [], []
    vals: dict = defaultdict(list)
    for r in rs:
        for c in tracked(r):
            vals[c["id"]].append(float(c["value"]))
    for k, v in sorted(vals.items()):
        bt = (b.get("tracked") or {}).get(k)
        if not bt:
            continue
        sign = 1 if bt["direction"] == "lower" else -1
        worse = sign * (statistics.fmean(v) - bt["mean"])
        if bt.get("kind") == "cost":
            if worse > COST_NOTE * abs(bt["mean"]):
                notes.append(f"cost {k}: {statistics.fmean(v):.1f} against {bt['mean']:.1f} in the baseline")
            continue
        allowed = max(R4_FLOOR * max(1.0, abs(bt["mean"])), 2 * bt["sd"])
        if worse > allowed + 1e-12:
            regs.append(f"R4 {k}: mean {statistics.fmean(v):.4f} against baseline {bt['mean']:.4f} "
                        f"({'lower' if sign > 0 else 'higher'} is better; allowed drift {allowed:.4f})")
    return regs, notes


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
        reg4, note4 = drift(b, rs)
        regressions += [x.replace("R4 ", f"R4 {ds} ", 1) for x in reg4]
        notes += [f"{ds} {x}" for x in note4]
        n_seeds = sum(1 for r in rs if r.get("checks"))
        failed_on: dict = defaultdict(int)
        for r in rs:
            for c in judged(r):
                failed_on[c["id"]] += not c["pass"]
        for r in rs:
            seen = set()
            for c in judged(r):
                seen.add(c["id"])
                bc = b["checks"].get(c["id"])
                if bc and bc["calibrated"] and not c["pass"] and (exact(c) or 2 * failed_on[c["id"]] > n_seeds):
                    v = f" (value {c.get('value')}, limit {c.get('limit')}, real {c.get('real')})"
                    regressions.append(f"R1 {ds} seed {r.get('seed')}: calibrated check failed: {c['id']}{v}")
                elif bc and bc["calibrated"] and not c["pass"]:
                    notes.append(f"{ds} seed {r.get('seed')}: statistical check {c['id']} failed on "
                                 f"{failed_on[c['id']]} of {n_seeds} seeds (value {c.get('value')}, limit {c.get('limit')})")
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
