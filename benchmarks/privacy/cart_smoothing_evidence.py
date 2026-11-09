# SPDX-License-Identifier: Apache-2.0
"""Evidence for cart's default smoothing (G5): the privacy report of the ACS
benchmark database released by cart, without and with smoothing.

    python benchmarks/privacy/cart_smoothing_evidence.py --out docs/assurance/evidence/G5/cart_acs.json

Needs the ACS database (`python -m bench fetch acs`).  Only aggregates are
written: each internal check's value, interval and status, and the verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(smoothing: str, seed: int) -> dict:
    """The internal-context checks of `relsynth privacy` on ACS, every table on cart."""
    from bench.assess import load_spec, privacy_config, with_generator
    from bench.datasets import DATASETS
    from relsynth.privacy import assess
    spec = with_generator(load_spec("acs", DATASETS["acs"].location(), []), "cart")
    for ts in spec["tables"].values():
        ts["generator"] = {"name": "cart", "options": {"smoothing": smoothing}}
    t0 = time.time()
    rep = assess(spec, privacy_config("acs"), seed=seed)
    ctx = rep["contexts"]["internal"]
    return {"verdict": ctx["verdict"], "seconds": round(time.time() - t0, 1),
            "checks": {f"{c['criterion']}|{c['attack']}": {"value": round(float(c["value"]), 4),
                                                         "ci": [round(float(x), 4) for x in c["ci"]],
                                                         "status": c["status"]} for c in ctx["checks"]}}


def main(argv=None) -> int:
    """Write both runs to --out."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    os.environ.setdefault("RELSYNTH_QUIET", "1")
    doc = {"what": "cart on ACS: privacy report without and with smoothing (G5)", "seed": a.seed,
           "machine": {"cpus": os.cpu_count(), "python": platform.python_version()},
           "smoothing_none": run("none", a.seed), "smoothing_density": run("density", a.seed)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    for k in ("smoothing_none", "smoothing_density"):
        so = doc[k]["checks"].get("record_isolation|singling_out_univariate", {})
        print(k, doc[k]["verdict"], "univariate singling out", so.get("value"), so.get("ci"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
