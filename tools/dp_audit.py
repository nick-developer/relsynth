# SPDX-License-Identifier: Apache-2.0
"""Attack audit of DP mode against its epsilon bound (roadmap T07, T07b).

    python tools/dp_audit.py --db contoso_sqlite --eps 1 --measure aim --out audit.jsonl

Runs `relsynth privacy` (relsynth.privacy.assess) with the DP pipeline as the
generator: 16 + 16 shadow databases of 1,000 people, each a full DP run.  The
record holds every context's verdict, its DP bounds and every check, so the
membership-inference TPR at 1 % FPR can be read against e^eps * 0.01 + delta.
The bench data must be in the cache first (`python -m bench fetch --datasets ...`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def summary(path: str) -> None:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    print("| database | measure | epsilon | context | verdict | DOMIAS TPR at 1 % FPR | bound | failed checks |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        for name, c in r["contexts"].items():
            tpr = next((x["value"] for x in c["checks"] if x["attack"] == "domias_tpr_at_fpr_0.01"), None)
            bound = c["bound"].get("dp_bound_tpr_at_fpr_0.01")
            fails = [x["attack"] for x in c["checks"] if x["status"] == "fail"]
            print(f"| {r['audit']} | {r['measure']} | {r['eps']:g} | {name} | {c['verdict']} | "
                  f"{'' if tpr is None else f'{tpr:.4f}'} | {'' if bound is None else f'{bound:.4f}'} | "
                  f"{', '.join(fails) or 'none'} |")


def main() -> int:
    from dp_curves import UNITS, _source
    from relsynth import privacy as P
    from relsynth.spec import read_spec
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="contoso_sqlite")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--measure", default="copula")
    ap.add_argument("--shadows", type=int, default=16)
    ap.add_argument("--out", default="dp-audit.jsonl")
    ap.add_argument("--summary", metavar="JSONL")
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
        return 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    s = read_spec(ROOT / "bench" / "specs" / f"{a.db}.yaml")
    s["source"] = _source(a.db)
    s.setdefault("synthesis", {}).setdefault("privacy", {})["dp"] = {
        "unit": UNITS[a.db], "epsilon": a.eps, "delta": 1e-9, "public_tables": "all_unowned",
        "spec_is_public": True, "measure": a.measure}
    cfg = P.load_config()
    cfg["unit"] = UNITS[a.db]
    cfg["attacks"]["membership"]["shadow"].update(
        {"n_train": a.shadows, "n_test": a.shadows, "n_test_gated": a.shadows, "shadow_size": 1000,
         "targets_vulnerable": 5, "targets_random": 5})
    t = time.time()
    rep = P.assess(s, cfg, seed=7, progress=lambda m: print(m, flush=True))
    out = {"audit": a.db, "eps": a.eps, "measure": a.measure, "secs": round(time.time() - t),
           "release": rep.get("release"),
           "contexts": {k: {"verdict": v["verdict"],
                            "bound": {kk: vv for kk, vv in v.items() if kk.startswith("dp_bound")},
                            "checks": [{f: c.get(f) for f in ("attack", "value", "limit", "status")}
                                       for c in v["checks"]]}
                        for k, v in rep["contexts"].items()}}
    print(json.dumps(out), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
