# SPDX-License-Identifier: Apache-2.0
"""The 2.0 quality report and the privacy attacks of one benchmark run, as
harness checks.

    python -m bench.assess quality NAME LOCATION SYN_DIR OUT.json [--set KEY=VALUE ...]
    python -m bench.assess privacy NAME LOCATION SEED OUT.json [--set KEY=VALUE ...]

Run by bench.runner in a subprocess per dataset and seed (so that no run's
memory carries into the next).  Each writes {"checks": [...], "seconds": ...}:

- quality: every calibrated 2.0 check (`quality:<id>`), the three area
  verdicts of the quality vector (`area:<area>`), the roadmap acceptance
  metrics that the report defines (`goal:<metric>`, docs/acceptance-metrics.md)
  and the SyntheRela summaries and area pass rates as tracked values
  (`metric:<name>`, pass None, with a direction for the drift rule R4).
- privacy: the attacks of relsynth.privacy on a release fitted to 40% of the
  people (T06), judged against the internal context's limits
  (`privacy:<criterion>|<attack>`, passing when the upper confidence bound is
  under the limit), the verdict of each context (`goal:privacy_<context>`,
  where an external context must stay blocked without DP) and the attack values
  as tracked values.

Only aggregates leave this module: values, limits, verdicts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "benchmarks" / "targets.yaml"
PRIVACY = ROOT / "benchmarks" / "privacy.yaml"


def load_spec(name: str, location: str, overrides: list[str]) -> dict:
    from .datasets import DATASETS
    spec = yaml.safe_load(Path(DATASETS[name].spec()).read_text(encoding="utf-8"))
    kind, path = location.split(":", 1)
    spec["source"] = {**(spec.get("source") or {}), "kind": kind, "path": path}
    for o in overrides:
        k, v = o.split("=", 1)
        d = spec
        *head, last = k.split(".")
        for p in head:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(v)
    return spec


def read_synthetic(spec: dict, folder: Path) -> dict:
    import pandas as pd
    out = {}
    for t in spec["tables"]:
        p = folder / f"{t}.csv"
        if p.exists():
            out[t] = pd.read_csv(p, low_memory=False)
    return out


def _r(x, n=6):
    try:
        return None if x is None else round(float(x), n)
    except (TypeError, ValueError):
        return None


def tracked(name: str, value, direction: str) -> dict:
    """A value recorded for the drift rule R4, without a verdict of its own."""
    return {"id": f"metric:{name}", "pass": None, "value": _r(value), "direction": direction}


def quality_checks(rep: dict) -> list[dict]:
    out = []
    for c in rep["checks"]:
        if c.get("source") != "quality-2.0":      # the 1.0 items are already scorecard: checks
            continue
        out.append({"id": f"quality:{c['id']}", "pass": None if c["value"] is None else bool(c["pass"]),
                    "value": c["value"], "limit": c["limit"], **({"alpha": c["alpha"]} if c.get("alpha") else {})})
    vec = rep["vector"]
    for a in ("fidelity", "utility", "integrity"):
        v = vec.get(a) or {}
        if v.get("checks"):
            # an area verdict already allows for its expected false failures (Prop. 111): an exact check
            out.append({"id": f"area:{a}", "pass": bool(v["pass"]), "value": v["pass_rate"], "limit": 0})
            out.append(tracked(f"{a}_pass_rate", v["pass_rate"], "higher"))
    by_metric: dict = {}
    for c in rep["checks"]:
        by_metric.setdefault(c["metric"], []).append(c)
    # roadmap acceptance metrics (docs/acceptance-metrics.md)
    fk = [c["value"] for c in by_metric.get("fk_integrity", []) if c["value"] is not None]
    if fk:
        out.append({"id": "goal:referential_integrity", "pass": min(fk) >= 1.0, "value": min(fk), "real": 1.0})
    hard = [c for c in by_metric.get("rule_violation_rate", []) if c["value"] is not None]
    if hard:
        # each rule's synthetic violation rate within max(0, the real data's own rate)
        worst = max(c["value"] for c in hard)
        out.append({"id": "goal:business_rule_preservation", "pass": all(c["pass"] for c in hard),
                    "value": _r(1 - worst), "limit": 0.99})
    fid = vec.get("fidelity") or {}
    if fid.get("checks"):
        out.append({"id": "goal:distribution_match", "pass": bool(fid["pass"]) and fid["pass_rate"] >= 0.95,
                    "value": fid["pass_rate"], "limit": 0.95})
    gaps = [c["value"] for m in ("tstr_gap", "mt_tstr_gap") for c in by_metric.get(m, []) if c["value"] is not None]
    if gaps:
        out.append({"id": "goal:utility_gap", "pass": max(gaps) < 0.05, "value": _r(max(gaps)), "limit": 0.05})
        out.append(tracked("utility_gap_max", max(gaps), "lower"))
    sr = (rep.get("benchmark") or {}).get("syntherela") or {}
    if sr.get("c2st_agg_accuracy_max") is not None:
        out.append(tracked("c2st_agg_accuracy_max", sr["c2st_agg_accuracy_max"], "lower"))
    card = [v for v in (sr.get("cardinality_similarity") or {}).values() if v is not None]
    if card:
        out.append(tracked("cardinality_similarity_mean", sum(card) / len(card), "higher"))
        out.append(tracked("cardinality_similarity_min", min(card), "higher"))
    for hop, v in (sr.get("k_hop_trend") or {}).items():
        if v is not None:
            out.append(tracked(f"k_hop_trend_{hop}", v, "higher"))
    return out


def run_quality(name, location, syn_dir, overrides) -> list[dict]:
    import relsynth as rs
    from .datasets import DATASETS
    spec = load_spec(name, location, overrides)
    targets = yaml.safe_load(TARGETS.read_text(encoding="utf-8")) or {}
    cfg = {**(targets.get(name) or {}), **DATASETS[name].quality}
    syn = read_synthetic(spec, Path(syn_dir))
    rep = rs.assess_quality(spec, syn, config=cfg, privacy=False).to_dict()
    return quality_checks(rep)


def privacy_checks(rep: dict) -> list[dict]:
    out = []
    for ctx_name, ctx in rep["contexts"].items():
        if ctx_name == "internal":
            for c in ctx["checks"]:
                cid = f"privacy:{c['criterion']}|{c['attack']}"
                ci = c.get("ci") or [None, None]
                out.append({"id": cid, "pass": None if c["status"] == "info" else c["status"] == "pass",
                            "value": _r(c["value"]), "limit": _r(c["limit"]), "upper": _r(ci[1]),
                            "status": c["status"]})
                out.append(tracked(f"privacy_{c['criterion']}_{c['attack']}", c["value"], "lower"))
            out.append({"id": "goal:privacy_internal", "pass": ctx["verdict"] != "fail", "value": None,
                        "status": ctx["verdict"]})
        elif ctx.get("external"):
            # without DP an external release must be refused (user decision of 2026-10-26)
            dp = (rep.get("release") or {}).get("dp")
            out.append({"id": f"goal:privacy_{ctx_name}_needs_dp", "pass": bool(dp) or ctx["verdict"] == "blocked",
                        "value": None, "status": ctx["verdict"]})
    g = (rep.get("guards") or {}).get("dcr")
    if isinstance(g, dict):
        for k, v in g.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(tracked(f"privacy_guard_dcr_{k}", v, "lower" if "match" in k else "higher"))
    return out


def run_privacy(name, location, seed, overrides) -> list[dict]:
    from relsynth.privacy import assess
    from .datasets import DATASETS
    spec = load_spec(name, location, overrides)
    allc = yaml.safe_load(PRIVACY.read_text(encoding="utf-8")) or {}
    cfg = json.loads(json.dumps({**(allc.get("defaults") or {})}))
    for k, v in (allc.get(name) or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    cfg.setdefault("attacks", {}).setdefault("membership", {})
    shadow = DATASETS[name].privacy.get("shadow")
    if shadow is not None:
        cfg["attacks"]["membership"]["shadow"] = {**cfg["attacks"]["membership"].get("shadow", {}), **shadow}
    rep = assess(spec, cfg, seed=int(seed))
    return privacy_checks(rep)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.assess")
    ap.add_argument("suite", choices=["quality", "privacy"])
    ap.add_argument("name")
    ap.add_argument("location")
    ap.add_argument("arg", help="synthetic folder (quality) or seed (privacy)")
    ap.add_argument("out")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    a = ap.parse_args(argv)
    import os
    os.environ.setdefault("RELSYNTH_QUIET", "1")
    t0 = time.time()
    fn = run_quality if a.suite == "quality" else run_privacy
    checks = fn(a.name, a.location, a.arg, a.overrides)
    Path(a.out).write_text(json.dumps({"checks": checks, "seconds": round(time.time() - t0, 1)}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
