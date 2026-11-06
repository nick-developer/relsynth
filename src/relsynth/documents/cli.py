# SPDX-License-Identifier: Apache-2.0
"""`relsynth docs ...`: discover, synthesise and check document collections.

    relsynth docs discover jsonl:./dump --out schema.ir.yaml        (inferred schema with counts, as IR 2.0)
    relsynth docs synth jsonl:./dump --to jsonl:./synthetic [--scale 2] [--report report.json]
    relsynth docs synth mongodb://host/shop --to jsonl:./out --collections orders customers
    relsynth docs synth dynamodb:app-table --to jsonl:./out          (single-table design: item types)
    relsynth docs check jsonl:./dump jsonl:./synthetic [--report report.json]

Sources and targets: ``jsonl:<folder>``, ``mongodb://...`` (or a YAML file with a
document-store section: kind mongodb | cosmos | dynamodb | couchbase | jsonl,
credentials from the environment: ``url_env``, ``key_env``).
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..errors import EXIT_GATE, EXIT_OK


def _section(s: str) -> dict:
    p = Path(s)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return yaml.safe_load(p.read_text())
    from .connectors import parse_location
    return parse_location(s)


def _read(sec: dict, collections=None, limit=None):
    from . import dynamo
    from .connectors import DynamoDB, connect_documents
    con = connect_documents(sec)
    if collections:
        con.include = collections
    if isinstance(con, DynamoDB) and sec.get("table"):
        ks = con.key_schema(sec["table"])
        items = list(con.read(sec["table"], limit))
        cols, design = dynamo.split(items, partition_key=ks["partition_key"], sort_key=ks["sort_key"],
                                    indexes=ks["indexes"], discriminator=sec.get("discriminator"))
        return cols, design
    return con.read_all(limit), None


def cmd_discover(a, out):
    from .infer import find_references, profile_collection
    from .ir import to_ir_document
    sec = _section(a.source)
    cols, design = _read(sec, a.collections, a.limit)
    profs = {c: profile_collection(c, d) for c, d in cols.items()}
    refs = find_references(profs)
    doc = to_ir_document(profs, refs, name=a.name or "documents", system=sec.get("kind", "mongodb"))
    if design is not None:
        from .dynamo import to_ir
        cont, phys = to_ir(design, sec["table"])
        doc["containers"] = {sec["table"]: cont}
        for e, p in phys.items():
            doc["entities"][e]["physical"] = p
    text = yaml.safe_dump(json.loads(json.dumps(doc, default=str)), sort_keys=False, width=110)
    for r in refs:
        if r.get("candidate"):
            out.line(f"candidate reference (not taken): {r['collection']}.{r['path']} -> {r['to']}.{r['to_path']} "
                     f"coverage {r.get('coverage')}" + (f", {r['target_duplicates']} duplicate targets"
                                                         if r.get("target_duplicates") else ""))
    if a.out:
        Path(a.out).write_text(text)
        out.line(f"wrote {a.out} ({len(profs)} collections, {sum(1 for r in refs if not r.get('candidate'))} "
                 f"references)")
    else:
        out.line(text.rstrip("\n"))
    out.put(collections={c: p.documents for c, p in profs.items()}, references=refs)
    return EXIT_OK


def cmd_synth(a, out):
    from . import dynamo
    from .connectors import connect_documents
    from .pipeline import DocumentSynthesizer
    sec = _section(a.source)
    cols, design = _read(sec, a.collections, a.limit)
    refs = yaml.safe_load(Path(a.references).read_text()) if a.references else None
    ds = DocumentSynthesizer(cols, references=refs, seed=a.seed, workdir=a.workdir)
    syn = ds.synthesize(scale=a.scale)
    tgt = _section(a.to)
    if design is not None and tgt.get("kind") == "dynamodb":
        items = dynamo.merge(syn, design)
        connect_documents(tgt).write(tgt.get("table") or sec["table"], items, a.if_exists)
        out.line(f"wrote {len(items)} items: {dynamo.check(items, design)}")
    else:
        if design is not None:
            syn = {sec["table"]: dynamo.merge(syn, design)}
        n = connect_documents(tgt).write_all(syn, a.if_exists)
        out.line("wrote " + ", ".join(f"{c} {k:,}" for c, k in n.items()))
    rc = EXIT_OK
    if a.report:
        rep = ds.report(syn if design is None else ds_split(syn, design, ds), splits=a.splits)
        Path(a.report).write_text(json.dumps(rep, indent=1, default=str))
        out.line(f"wrote {a.report}: {'pass' if rep['pass'] else 'FAIL'}")
        rc = EXIT_OK if rep["pass"] else EXIT_GATE
    return rc


def ds_split(syn, design, ds):
    """The synthetic collections of a single-table design (before merging)."""
    if len(syn) == 1:
        from .dynamo import split
        items = next(iter(syn.values()))
        cols, _ = split(items, partition_key=design["partition_key"], sort_key=design["sort_key"],
                        indexes=design["indexes"], discriminator=design["discriminator"])
        return cols
    return syn


def cmd_check(a, out):
    from .connectors import connect_documents
    from .infer import find_references, profile_collection
    from .metrics import reference_integrity, shape_checks
    from .validate import validate_collection
    real = connect_documents(_section(a.real)).read_all()
    syn = connect_documents(_section(a.synthetic)).read_all()
    profs = {c: profile_collection(c, d) for c, d in real.items()}
    refs = find_references(profs)
    rep = {"collections": {}, "references": reference_integrity(syn, refs)}
    ok = all(r["share"] == 1.0 for r in rep["references"])
    for c, docs in syn.items():
        if c not in profs:
            continue
        v = validate_collection(profs[c], docs)
        sh = shape_checks(real[c], docs, c, splits=a.splits)
        rep["collections"][c] = {"validity": v, "shape": sh}
        ok = ok and v["share"] == 1.0 and all(m["pass"] is not False for m in sh)
        bad = [m["metric"] for m in sh if m["pass"] is False]
        out.line(f"{c}: {v['valid']}/{v['documents']} valid; shape " + ("pass" if not bad else "FAIL " + ", ".join(bad)))
    for r in rep["references"]:
        out.line(f"{r['reference']}: {r['resolved']}/{r['values']} resolve")
    rep["pass"] = ok
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1, default=str))
    out.put(**{"pass": ok})
    return EXIT_OK if ok else EXIT_GATE


def add_parser(sub, common):
    p = sub.add_parser("docs", help="document collections (MongoDB, Cosmos DB, DynamoDB, Couchbase, JSON): "
                                    "discover, synthesise, check")
    s2 = p.add_subparsers(dest="docs_cmd", required=True, metavar="DOCS_COMMAND")

    def src(s):
        s.add_argument("source", help="jsonl:<folder>, mongodb://..., dynamodb:<table>, or a YAML store section")
        s.add_argument("--collections", nargs="*", help="only these collections")
        s.add_argument("--limit", type=int, help="read at most this many documents per collection")

    s = s2.add_parser("discover", help="infer the schema with counts (IR 2.0 with x-relsynth-counts)")
    src(s)
    s.add_argument("--name", help="estate name")
    s.add_argument("--out", help="output file (default: stdout)")
    s.set_defaults(fn=cmd_discover)
    s = s2.add_parser("synth", help="synthesise the collections and write them")
    src(s)
    s.add_argument("--to", required=True, help="target (jsonl:<folder>, mongodb://..., a YAML store section)")
    s.add_argument("--scale", type=float, help="documents per source document (default 1)")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--references", help="YAML list of declared references {collection, path, to, to_path}")
    s.add_argument("--if-exists", default="replace", choices=["replace", "append", "fail"])
    s.add_argument("--workdir", help="keep the shredded database and the spec here")
    s.add_argument("--report", help="write the quality report (JSON) here")
    s.add_argument("--splits", type=int, default=10, help="real-versus-real splits for the limits")
    s.set_defaults(fn=cmd_synth)
    s = s2.add_parser("check", help="validate synthetic collections and score their shape against the real ones")
    s.add_argument("real")
    s.add_argument("synthetic")
    s.add_argument("--report")
    s.add_argument("--splits", type=int, default=10)
    s.set_defaults(fn=cmd_check)
    for sp in s2.choices.values():
        common(sp)
    return p
