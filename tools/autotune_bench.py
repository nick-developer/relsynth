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
    a = ap.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
