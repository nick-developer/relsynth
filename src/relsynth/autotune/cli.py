# SPDX-License-Identifier: Apache-2.0
"""`relsynth autotune spec.yaml [--objectives objectives.yaml]`: tune the
engine settings toward the objectives, keep a history of every run, and write
the tuned spec and the constraints report.

Exit codes: 0 every objective met (and confirmed on the held-out seeds), 1 not
met or not confirmed (the report says why), 2 invalid spec or objectives."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..errors import EXIT_GATE, EXIT_OK


def cmd_autotune(a, out):
    from .loop import autotune
    from .objectives import load
    obj = load(a.objectives)
    if a.seeds:
        obj.seeds_search = [int(x) for x in a.seeds.split(",") if x.strip()]
    if a.holdout is not None:
        obj.seeds_holdout = [int(x) for x in a.holdout.split(",") if x.strip()]
    stem = Path(a.spec).with_suffix("")
    hist = a.history or f"{stem}.tune"
    spec = a.spec
    if a.set or a.settings:
        from ..cli import _spec
        spec = _spec(a)                     # --set / --settings: the starting point of the search
    res = autotune(spec, obj, history_dir=hist, runs=a.runs, progress=out.line if not out.as_json else None,
                   resume=not a.fresh, workers=max(1, int(a.workers or 1)))
    dest = Path(a.out or f"{stem}.tuned.yaml")
    dest.write_text(yaml.safe_dump(json.loads(json.dumps(res.spec, default=str)), sort_keys=False, width=110),
                    encoding="utf-8")
    rep = Path(a.report or Path(hist) / "report.md")
    rep.write_text(res.report_markdown, encoding="utf-8")
    Path(rep.with_suffix(".json")).write_text(json.dumps(res.to_dict(), indent=1, default=str), encoding="utf-8")
    out.put(met=res.met, confirmed=res.confirmed, delta=res.delta, patches=res.patches, stopped=res.stopped,
            runs=res.runs, runs_reused=res.runs_reused, tuned_spec=str(dest), report=str(rep),
            judgement={k: v for k, v in res.judgement.items() if k != "families"})
    out.line(f"{'all objectives met' if res.met else 'not every objective met'}; stopped: {res.stopped}")
    out.line(f"tuned settings: {res.delta or 'defaults'}" + (f"; spec fixes: {res.patches}" if res.patches else ""))
    if res.confirmed is not None:
        out.line(f"held-out seeds: {'confirmed' if res.confirmed else 'NOT confirmed'}")
    out.line(f"wrote {dest} and {rep}")
    return EXIT_OK if res.met and res.confirmed is not False else EXIT_GATE


def add_parser(sub, common):
    s = sub.add_parser("autotune", help="tune engine settings toward your objectives over several seeds; "
                                        "write the tuned spec and a constraints report")
    s.add_argument("spec", help="the YAML specification")
    s.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a spec value before tuning, e.g. --set source.path=db.sqlite (repeatable)")
    s.add_argument("--settings", metavar="FILE", help="an engine settings file (or preset:fast | preset:thorough) "
                   "as the starting point")
    s.add_argument("--objectives", help="objectives.yaml (default: every hard constraint and every statistical family)")
    s.add_argument("--runs", type=int, help="budget: fits (one candidate on one seed is one run)")
    s.add_argument("--seeds", help="search seeds, e.g. 1,2,3 (overrides the objectives)")
    s.add_argument("--holdout", help="held-out seeds, e.g. 11,12 ('' for none)")
    s.add_argument("--history", help="history folder (default: <spec>.tune)")
    s.add_argument("--workers", type=int, default=1,
                   help="run the seeds of a candidate in this many parallel processes (memory: one fit each)")
    s.add_argument("--fresh", action="store_true", help="forget this spec's earlier runs")
    s.add_argument("--out", help="tuned spec (default: <spec>.tuned.yaml)")
    s.add_argument("--report", help="report file (default: <history>/report.md, plus report.json)")
    s.set_defaults(fn=cmd_autotune)
