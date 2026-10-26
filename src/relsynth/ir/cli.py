"""`relsynth ir ...`: convert, validate, migrate and export IR documents.

    relsynth ir migrate spec.yaml --out spec.ir.yaml [--privacy-unit Customer]
    relsynth ir validate estate.yaml
    relsynth ir convert estate.yaml --to v1 --out spec.yaml
    relsynth ir privacy estate.yaml [--unit Customer]
    relsynth ir odcs estate.yaml --out contract.yaml          (ODCS v3.2.0 export)
    relsynth ir from-odcs contract.yaml --out estate.yaml     (ODCS v3 import)
    relsynth ir openlineage estate.yaml --namespace sqlite://shop [--inputs-namespace ...]
    relsynth ir schema --out relsynth-ir-2.0.schema.json
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..errors import EXIT_GATE, EXIT_OK, EXIT_SPEC


def _write(text: str, dest, out, what):
    if dest:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text(text, encoding="utf-8")
        out.line(f"wrote {dest} ({what})")
        out.put(written=str(dest))
    else:
        out.line(text.rstrip("\n"))


def _dump(d: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(d, indent=2, default=str) + "\n"
    return yaml.safe_dump(json.loads(json.dumps(d, default=str)), sort_keys=False, width=110, allow_unicode=True)


def _fmt(dest, given):
    return given or ("json" if dest and str(dest).endswith(".json") else "yaml")


def cmd_migrate(a, out):
    from .migrate import migrate
    m = migrate(a.spec, privacy_unit=a.privacy_unit, public=a.public)
    from . import dumps
    fmt = _fmt(a.out, a.format)
    text = (m.header(Path(a.spec).name) if fmt == "yaml" else "") + dumps(m.estate, fmt)
    out.put(identical=m.identical, differences=m.differences, notes=m.notes, source_format=m.source_format)
    if not m.identical:
        out.line("the conversion is NOT identical:\n  - " + "\n  - ".join(m.differences))
        return EXIT_SPEC
    _write(text, a.out, out, f"IR {m.estate.ir_version}; the 1.0 spec it lowers to is identical to the original")
    return EXIT_OK


def cmd_validate(a, out):
    from . import load
    from .validate import check
    est = load(a.doc)
    errs, warns = check(est)
    schema_errs = []
    try:
        from ..spec import read_yaml
        from .jsonschema import validate_json
        raw = read_yaml(a.doc)
        if "ir_version" in raw:
            schema_errs = validate_json(raw)
    except ImportError:
        out.line("(jsonschema not installed: JSON Schema check skipped; pip install 'relsynth[ir]')")
    errs = [f"(JSON Schema) {e}" for e in schema_errs] + errs
    for w in warns:
        out.line(f"warning: {w}")
    out.put(errors=errs, warnings=warns, entities=len(est.entities), relationships=len(est.relationships))
    if errs:
        out.line("invalid:\n  - " + "\n  - ".join(errs))
        return EXIT_SPEC
    out.line(f"valid: {len(est.entities)} entities, {len(est.relationships)} relationships, {len(est.rules)} rules")
    return EXIT_OK


def cmd_convert(a, out):
    from . import dumps, load
    from .v1 import lower
    est = load(a.doc)
    if a.to == "v1":
        spec, losses = lower(est, strict=a.strict)
        for x in losses:
            out.line(f"not in the 1.0 spec: {x}")
        out.put(losses=losses)
        _write(_dump(spec, _fmt(a.out, a.format)), a.out, out, "relsynth 1.0 spec")
        return EXIT_OK
    _write(dumps(est, _fmt(a.out, a.format)), a.out, out, "IR 2.0")
    return EXIT_OK


def cmd_privacy(a, out):
    from . import load
    from .privacy import trace
    est = load(a.doc)
    tr = trace(est, a.unit)
    out.put(trace=tr.to_dict())
    out.line(tr.to_markdown(est).rstrip("\n"))
    return EXIT_OK


def cmd_odcs(a, out):
    from . import load
    from .odcs import to_odcs
    c = to_odcs(load(a.doc), contract_id=a.id, version=a.contract_version)
    _write(_dump(c, _fmt(a.out, a.format)), a.out, out, "ODCS v3.2.0 data contract")
    return EXIT_OK


def cmd_from_odcs(a, out):
    from ..spec import read_yaml
    from . import dumps
    from .odcs import from_odcs
    from .validate import check
    est = from_odcs(read_yaml(a.contract))
    errs, warns = check(est, relational=False)
    out.put(errors=errs, warnings=warns)
    _write(dumps(est, _fmt(a.out, a.format)), a.out, out, "IR 2.0 from an ODCS contract")
    if errs:
        out.line("the contract converts, but the IR has errors to fix:\n  - " + "\n  - ".join(errs))
        return EXIT_GATE
    return EXIT_OK


def cmd_openlineage(a, out):
    from . import load
    from .openlineage import datasets
    ds = datasets(load(a.doc), a.namespace, a.inputs_namespace)
    _write(json.dumps(ds, indent=2, default=str) + "\n", a.out, out, f"{len(ds)} OpenLineage datasets")
    return EXIT_OK


def cmd_schema(a, out):
    from .jsonschema import schema_text
    _write(schema_text(), a.out, out, "JSON Schema of the IR")
    return EXIT_OK


def add_parser(sub, common):
    p = sub.add_parser("ir", help="IR 2.0 documents: migrate, validate, convert, privacy paths, ODCS, OpenLineage")
    s2 = p.add_subparsers(dest="ir_cmd", required=True, metavar="IR_COMMAND")

    def fmt(s):
        s.add_argument("--out", help="output file (default: stdout)")
        s.add_argument("--format", choices=["yaml", "json"], help="output format (default: from --out, else yaml)")

    s = s2.add_parser("migrate", help="convert a 1.0 spec to IR 2.0 and prove the conversion")
    s.add_argument("spec")
    s.add_argument("--privacy-unit", help="entity whose records are the privacy units (e.g. Customer)")
    s.add_argument("--public", nargs="*", help="entities that hold no data of the unit (code lists, calendars)")
    fmt(s)
    s.set_defaults(fn=cmd_migrate)
    s = s2.add_parser("validate", help="validate an IR document (or a 1.0 spec read as IR)")
    s.add_argument("doc")
    s.set_defaults(fn=cmd_validate)
    s = s2.add_parser("convert", help="convert between IR 2.0 and the 1.0 spec")
    s.add_argument("doc")
    s.add_argument("--to", choices=["ir", "v1"], default="ir")
    s.add_argument("--strict", action="store_true", help="--to v1: fail when something cannot be carried")
    fmt(s)
    s.set_defaults(fn=cmd_convert)
    s = s2.add_parser("privacy", help="the privacy unit and the reference paths that carry its data")
    s.add_argument("doc")
    s.add_argument("--unit", help="entity of the privacy unit (default: privacy.unit.entity)")
    s.set_defaults(fn=cmd_privacy)
    s = s2.add_parser("odcs", help="export an Open Data Contract Standard v3.2.0 contract")
    s.add_argument("doc")
    s.add_argument("--id", help="contract id (default: the estate's name)")
    s.add_argument("--contract-version", default="1.0.0", help="the contract's own version")
    fmt(s)
    s.set_defaults(fn=cmd_odcs)
    s = s2.add_parser("from-odcs", help="import an ODCS v3 contract as IR 2.0")
    s.add_argument("contract")
    fmt(s)
    s.set_defaults(fn=cmd_from_odcs)
    s = s2.add_parser("openlineage", help="OpenLineage dataset facets of every entity (JSON)")
    s.add_argument("doc")
    s.add_argument("--namespace", required=True, help="OpenLineage namespace of the datasets (e.g. sqlite://shop)")
    s.add_argument("--inputs-namespace", help="for synthetic outputs: namespace of the real inputs (adds column "
                                              "lineage)")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_openlineage)
    s = s2.add_parser("schema", help="the JSON Schema of the IR")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_schema)
    for sp in s2.choices.values():
        common(sp)
    return p
