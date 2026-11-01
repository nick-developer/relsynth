# SPDX-License-Identifier: Apache-2.0
"""Evaluation of the review agent on the golden databases (roadmap T16).

    python -m bench.agents                                   # Chinook, AdventureWorks DW, Contoso (SQLite)
    python -m bench.agents --datasets chinook --backend openai --model qwen2.5:1.5b-instruct
    RELSYNTH_BENCH_BI_DIR=... python -m bench.agents --datasets bi_corep_lr

For each database the draft is made in benchmark mode (the catalog's keys and
links hidden), the agent answers its review items, and every answer about a
key or a link is compared with the ground truth: the declared catalog, or for
bi/COREP (CSV, no catalog) the hand-reviewed spec in RELSYNTH_BENCH_BI_DIR.
The results are aggregates and schema names only; for a private dataset the
item texts are left out.  `--min-accuracy` makes the command a gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from .datasets import DATASETS, available, is_private, say

DEFAULT = ["chinook", "adventureworks", "contoso_sqlite"]


def run_one(name: str, backend_cfg, scope: str = "aggregates") -> dict:
    import relsynth as rs
    from relsynth.agents.backends import make_backend
    from relsynth.agents.evaluation import evaluate, truth_from_spec
    from relsynth.agents.review import review
    ds = DATASETS[name]
    loc = ds.location()
    t0 = time.perf_counter()
    draft = rs.introspect(loc, declared=False).raw
    t_draft = time.perf_counter() - t0
    if name == "bi_corep_lr":
        truth_spec = yaml.safe_load(Path(ds.spec()).read_text(encoding="utf-8"))
        truth_kind = "curated spec"
    else:
        truth_spec = rs.introspect(loc).raw
        truth_kind = "declared catalog"
    backend = make_backend(backend_cfg)
    res = review(draft, source=loc, backend=backend, scope=scope)
    ev = evaluate(res, truth_from_spec(truth_spec))
    ev.update(dataset=name, truth=truth_kind, draft_seconds=round(t_draft, 1), backend=res.backend, scope=res.scope,
              summary=res.summary(), model_calls=getattr(backend, "calls", 0) if not isinstance(
                  getattr(backend, "calls", 0), list) else len(backend.calls))
    if is_private(name):
        for w in ev["wrong_answers"]:
            w.pop("line", None)
        ev["links_missed_by_draft"] = len(ev["links_missed_by_draft"])
        ev["approval"]["hurried_reviewer_would_accept_wrong"] = len(ev["approval"]["hurried_reviewer_would_accept_wrong"])
    return ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.agents")
    ap.add_argument("--datasets", default=",".join(DEFAULT))
    ap.add_argument("--backend", default="rules", choices=["rules", "openai"])
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--scope", default="aggregates")
    ap.add_argument("--min-accuracy", type=float)
    ap.add_argument("--max-unflagged-wrong", type=int, help="gate: wrong answers not flagged for attention")
    ap.add_argument("--out-dir", default="agents-out")
    a = ap.parse_args(argv)
    cfg: dict | str = "rules"
    if a.backend == "openai":
        cfg = {"kind": "openai", **({"base_url": a.base_url} if a.base_url else {}),
               **({"model": a.model} if a.model else {}), "timeout_s": 300}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, rc = [], 0
    for name in [x.strip() for x in a.datasets.split(",") if x.strip()]:
        ok, why = available(name)
        if not ok:
            say(f"{name}: skipped ({why})")
            rows.append({"dataset": name, "skipped": why})
            continue
        say(f"{name}: drafting in benchmark mode and reviewing ({a.backend})")
        ev = run_one(name, cfg, a.scope)
        (out / f"{name}-{a.backend}.json").write_text(json.dumps(ev, indent=1, default=str), encoding="utf-8")
        rows.append(ev)
        say(f"{name}: {ev['correct']} correct, {ev['wrong']} wrong, {ev['abstained']} left to people of "
            f"{ev['judged_items']}; accuracy {ev['accuracy']:.3f}; wrong unflagged "
            f"{ev['approval']['wrong_unflagged']}; {ev['time']['agent_seconds']} s")
        if a.min_accuracy is not None and not ev["accuracy"] >= a.min_accuracy:
            rc = 1
        if a.max_unflagged_wrong is not None and ev["approval"]["wrong_unflagged"] > a.max_unflagged_wrong:
            rc = 1
    tot = [r for r in rows if "correct" in r]
    if tot:
        c, w, ab = (sum(r[k] for r in tot) for k in ("correct", "wrong", "abstained"))
        from relsynth.agents.evaluation import wilson
        lo, hi = wilson(c, c + w)
        summary = {"datasets": [r["dataset"] for r in tot], "correct": c, "wrong": w, "abstained": ab,
                   "accuracy": c / (c + w) if c + w else None, "accuracy_ci95": [lo, hi],
                   "coverage": (c + w) / (c + w + ab) if c + w + ab else None,
                   "wrong_unflagged": sum(r["approval"]["wrong_unflagged"] for r in tot),
                   "oracle_wrong_changes_applied": sum(r["approval"]["oracle_wrong_changes_applied"] for r in tot),
                   "backend": a.backend, "model": a.model}
        (out / f"summary-{a.backend}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        say(json.dumps(summary))
    return rc


if __name__ == "__main__":
    sys.exit(main())
