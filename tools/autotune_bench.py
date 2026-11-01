# SPDX-License-Identifier: Apache-2.0
"""`relsynth autotune` on a golden database of the bench (the workflow autotune.yml).

    python tools/autotune_bench.py --dataset adventureworks --out tune-out --workers 3 --hours 5

Builds the database (bench/datasets.py), writes the objectives (every hard
constraint, the dataset's domain identities when bench/autotune_domain.py has
them, every statistical family of the scorecard; calibrated), and runs autotune
with its history in <out>/history, so a later call with the same history goes on
where this one stopped.  --set adds spec overrides (a spec experiment, such as a
link_affinity), which change the fingerprint and so start their own history.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adventureworks")
    ap.add_argument("--out", default="tune-out")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--holdout", default="11,12")
    ap.add_argument("--runs", type=int, default=42)
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--halving", action="store_true", help="successive halving (off: every seed of a candidate at once)")
    ap.add_argument("--no-spec-patches", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--summarise", metavar="DIR", help="print a compact summary of DIR's history and report; no runs")
    a = ap.parse_args(argv)
    if a.summarise:
        return summarise(Path(a.summarise))
    sys.path.insert(0, str(ROOT))
    from bench import autotune_domain
    from bench.datasets import DATASETS
    ds = DATASETS[a.dataset]
    kind, _, path = ds.location().partition(":")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    obj = {"constraints": ["referential_integrity", "rules", "unique_keys", "trees", "privacy"],
           "targets": ["all"],
           "seeds": {"search": [int(x) for x in a.seeds.split(",") if x],
                     "holdout": [int(x) for x in a.holdout.split(",") if x]},
           "budget": {"runs": a.runs, "hours": a.hours}, "spec_patches": not a.no_spec_patches,
           "halving": a.halving}
    if hasattr(autotune_domain, a.dataset):
        obj["constraints"].append("domain")
        obj["domain_checks"] = f"{ROOT / 'bench' / 'autotune_domain.py'}:{a.dataset}"
    (out / "objectives.yaml").write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")
    cmd = [sys.executable, "-m", "relsynth", "autotune", str(ds.spec()), "--objectives", str(out / "objectives.yaml"),
           "--set", f"source.kind={kind}", "--set", f"source.path={path}", "--workers", str(a.workers),
           "--history", str(out / "history"), "--out", str(out / "tuned.yaml"), "--report", str(out / "report.md")]
    for kv in a.set:
        cmd += ["--set", kv]
    print("$", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run(cmd, env=env, check=False).returncode  # noqa: S603


def summarise(out: Path) -> int:
    """Every candidate of the history (seeds, hard violations, excess failures,
    families not met), the baseline's and the incumbent's failures per family,
    and the checks that fail systematically: aggregates only, for the job log."""
    import json

    from relsynth.autotune import analysis as A
    from relsynth.autotune.checks import family
    from relsynth.autotune.history import History
    from relsynth.autotune.objectives import load
    obj = load(out / "objectives.yaml")
    hist = out / "history" / "history.jsonl"
    if not hist.exists():
        print(f"no history in {out}")
        return 1
    recs = [json.loads(x) for x in hist.read_text(encoding="utf-8").splitlines() if x.strip()]
    fps = {r["fingerprint"] for r in recs}
    for fp in sorted(fps):
        h = History(out / "history", fp)
        print(f"\n===== fingerprint {fp}: {len(h.records)} runs =====")
        for rs in h.by_key().values():
            J = A.judge(rs, obj)
            bad = sorted(f for f, o in J.families().items() if not o["met"])
            hard = sorted({c for v in J.views for c in v.hard})
            secs = round(sum(r.get("seconds") or 0 for r in rs) / max(len(rs), 1))
            print(f"- {rs[0].get('origin')}: {rs[0]['delta'] or 'defaults'} {rs[0]['patches'] or ''} | seeds "
                  f"{J.seeds} | hard {J.hard_violations} {hard} | excess {J.excess:.2f} | failures {J.failures:.1f}"
                  f" | {secs}s/run | not met: {bad}")
        base = [r for r in h.records if not r["delta"] and not r["patches"] and r["seed"] in obj.seeds_search]
        if base:
            J = A.judge(base, obj)
            print("baseline families (failed per seed / null quantile q / checks):")
            for f, o in sorted(J.families().items(), key=lambda kv: -sum(kv[1]["failed"])):
                if sum(o["failed"]):
                    print(f"  {f}: {o['failed']} q={o['q']} n={o['n_checks']} mu={o['mu']}")
            print("baseline systematic checks:")
            for cid, d in sorted(A.systematic(base).items()):
                vals = [c.get("value") for r in base for c in r["checks"] if c["id"] == cid]
                lim = next((c.get("limit") for r in base for c in r["checks"] if c["id"] == cid), None)
                print(f"  [{family(cid)}] {cid}: {d['fails']}/{d['n']} values {vals} limit {lim}")
            print("baseline domain values (real, synthetic):")
            for c in base[0]["checks"]:
                if c["id"].startswith("domain:"):
                    syn = [x.get("value") for r in base for x in r["checks"] if x["id"] == c["id"]]
                    print(f"  {c['id']}: real {c.get('real')} syn {syn}")
    rep = out / "report.json"
    if rep.exists():
        r = json.loads(rep.read_text(encoding="utf-8"))
        print("\n===== report.json =====")
        print(json.dumps({k: r.get(k) for k in ("met", "confirmed", "delta", "patches", "stopped", "runs", "holdout")},
                         default=str)[:4000])
        an = r.get("analysis") or {}
        for f, o in sorted((an.get("families") or {}).items()):
            if not o.get("met"):
                print(f"family {f}: " + json.dumps({k: v for k, v in o.items() if k != "targeted"}, default=str)[:1500])
        for c, o in sorted((an.get("constraints") or {}).items()):
            if not o.get("met"):
                print(f"constraint {c}: " + json.dumps(o, default=str)[:1500])
        print("conflicts: " + json.dumps(an.get("conflicts"), default=str)[:3000])
        print("trajectory:")
        for t in r.get("trajectory") or []:
            print("  " + json.dumps(t, default=str)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
