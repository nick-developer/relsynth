# SPDX-License-Identifier: Apache-2.0
"""`relsynth text ...`: the licence register, PII recall, leakage audit and metrics.

    relsynth text models [--use lm|pii|embedding]
    relsynth text pii-eval [--labelled file.jsonl] [--recognizer regex --recognizer gliner=/models/gliner]
    relsynth text audit spec.yaml --table ticket --column body [--canaries 64 --repeats 4]
    relsynth text metrics real.csv synthetic.csv --column body --condition category
"""
from __future__ import annotations

import json
from pathlib import Path

from ..errors import EXIT_OK


def _registry() -> dict:
    import importlib.resources as ir

    import yaml
    return yaml.safe_load((ir.files("relsynth") / "data" / "text_models.yaml").read_text(encoding="utf-8"))


def _rec(arg: str):
    if "=" in arg:
        name, model = arg.split("=", 1)
        return {"name": name, "model": model}
    return arg


def cmd_text(a, out):
    if a.action == "models":
        reg = _registry()
        rows = [m for m in reg["models"] if not a.use or a.use in m["use"]]
        out.put(models=rows, libraries=reg["libraries"])
        for m in rows:
            out.line(f"{m['id']:<42} {m['use']:<22} {m['licence']:<30} {m['status']}")
        return EXIT_OK
    if a.action == "pii-eval":
        from .labelled import labelled_set
        from .pii import PiiPipeline, evaluate
        if a.labelled:
            lab = [json.loads(x) for x in Path(a.labelled).read_text(encoding="utf-8").splitlines() if x.strip()]
        else:
            lab = labelled_set(a.n, seed=a.seed)
        res = evaluate(PiiPipeline([_rec(r) for r in (a.recognizer or ["regex"])]), lab)
        out.put(pii_eval=res)
        o = res["overall"]
        out.line(f"recognizers {', '.join(res['recognizers'])}: recall {o['recall']:.3f} "
                 f"[{o['recall_ci'][0]:.3f}, {o['recall_ci'][1]:.3f}] on {o['n']} entities, precision {o['precision']:.3f}")
        for t, d in res["types"].items():
            out.line(f"  {t:<9} n={d['n']:<5} recall {d['recall']:.3f} [{d['recall_ci'][0]:.3f}, {d['recall_ci'][1]:.3f}]")
        return EXIT_OK
    if a.action == "audit":
        from ..data import load_tables
        from ..spec import load_spec
        from .audit import run_audit
        spec = load_spec(a.target)
        cs = spec["tables"][a.table]["columns"][a.column]
        df = load_tables(spec)[a.table]
        res = run_audit(df, a.column, cs.get("text") or {}, canaries=a.canaries, repeats=a.repeats, words=a.words,
                        samples=a.samples, owner_column=a.owner)
        out.put(audit=res)
        e = res.get("exposure") or {}
        if e:
            out.line(f"exposure of {res['canaries']} canaries (x{res['repeats']}): mean {e['mean']:.2f}, max {e['max']:.2f} "
                     f"of {e['max_possible']:.2f} bits")
            for t, b in (e.get("bound") or {}).items():
                out.line(f"  P(exposure >= {t}) = {b['observed']:.3f}, DP bound {b['bound']:.3f}, p = {b['p_value']:.3g}")
        d = res["duplicates"]
        out.line(f"generated {d['n']}: exact copies {d['exact_rate']:.3f}, near duplicates {d['near_rate']:.3f}; "
                 f"extraction: {res['extraction']['extracted']} rare training texts in the top {res['extraction']['top']}")
        return EXIT_OK
    if a.action == "metrics":
        import pandas as pd

        from .metrics import report
        res = report(pd.read_csv(a.target), pd.read_csv(a.synthetic), a.column, a.condition or [])
        out.put(metrics=res)
        out.line(json.dumps(res, indent=2, default=str))
        return EXIT_OK
    raise ValueError(a.action)


def add_parser(sub, common=None):
    s = sub.add_parser("text", help="synthetic text: model licences, PII recall, leakage audit, metrics")
    s.add_argument("action", choices=["models", "pii-eval", "audit", "metrics"])
    s.add_argument("target", nargs="?", help="audit: spec; metrics: real CSV")
    s.add_argument("synthetic", nargs="?", help="metrics: synthetic CSV")
    s.add_argument("--use", help="models: lm, pii, embedding or mauve-features")
    s.add_argument("--labelled", help="pii-eval: JSON lines {text, spans: [[start, end, type]]}")
    s.add_argument("--recognizer", action="append", help="pii-eval: regex, presidio, spacy=<model>, gliner=<folder>")
    s.add_argument("--n", type=int, default=1000, help="pii-eval: size of the generated labelled set")
    s.add_argument("--seed", type=int, default=1506)
    s.add_argument("--table")
    s.add_argument("--column")
    s.add_argument("--owner", help="audit: column of the person (default: text.dp.unit_column, else the row)")
    s.add_argument("--canaries", type=int, default=64)
    s.add_argument("--repeats", type=int, default=4)
    s.add_argument("--words", type=int, default=16)
    s.add_argument("--samples", type=int, default=2000)
    s.add_argument("--condition", action="append", help="metrics: conditioning column (repeatable)")
    s.set_defaults(fn=cmd_text)
