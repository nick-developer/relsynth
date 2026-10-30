# SPDX-License-Identifier: Apache-2.0
"""The JSON Schema (draft 2020-12) of the IR and its Markdown reference, both
generated from the object model (relsynth.ir.model), so neither can drift from
what the code reads.

    python -m relsynth.ir.jsonschema schema    > src/relsynth/ir/schema/relsynth-ir-2.0.schema.json
    python -m relsynth.ir.jsonschema reference > docs/ir-reference.md

A test compares the committed files with the generated ones.
"""
from __future__ import annotations

import json
import sys
import types
import typing
from pathlib import Path
from typing import Any, Union

from . import model as M


def _class_doc(cls, first_paragraph: bool = False) -> str:
    """A class docstring as one line, the same on every Python version (3.13 dedents
    docstrings; a dataclass without one gets a signature whose text varies)."""
    doc = cls.__doc__ or ""
    if doc.startswith(cls.__name__ + "("):
        return ""
    lines = [ln.strip() for ln in doc.strip().splitlines()]
    if first_paragraph and "" in lines:
        lines = lines[:lines.index("")]
    return " ".join(" ".join(lines).split())

SCHEMA_ID = ("https://raw.githubusercontent.com/nick-developer/relsynth/main/src/relsynth/ir/schema/"
             f"relsynth-ir-{M.IR_VERSION}.schema.json")
SCHEMA_FILE = Path(__file__).parent / "schema" / f"relsynth-ir-{M.IR_VERSION}.schema.json"

# sections kept in the form of a 1.0 spec: their keys come from the 1.0 key schema
_V1_SECTIONS = {("Estate", "source"): "source", ("Estate", "target"): "target",
                ("AttributeSynthesis", "distribution"): "distribution", ("AttributeSynthesis", "text"): "text",
                ("AttributeSynthesis", "pii"): "pii"}


def _v1_section(kind: str) -> dict:
    from ..schema import S
    section = typing.cast(dict, S[kind])
    return {"type": "object", "additionalProperties": False,
            "properties": {k: {"description": doc} for k, (_, _, doc) in section.items()}}


def _type_schema(tp) -> dict:
    if typing.get_origin(tp) in (Union, types.UnionType):
        alts = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(alts) == 1:
            return _type_schema(alts[0])
        return {"anyOf": [_type_schema(a) for a in alts]}
    if tp is Any:
        return {}
    o = typing.get_origin(tp)
    if o is list or tp is list:
        args = typing.get_args(tp)
        return {"type": "array", "items": _type_schema(args[0])} if args else {"type": "array"}
    if o is dict or tp is dict:
        args = typing.get_args(tp)
        if args:
            return {"type": "object", "additionalProperties": _type_schema(args[1])}
        return {"type": "object"}
    if isinstance(tp, type) and issubclass(tp, M.Node):
        return {"$ref": f"#/$defs/{tp.__name__}"}
    return {"type": {str: "string", int: "integer", float: "number", bool: "boolean"}[tp]}


def _node_schema(cls) -> dict:
    hints = typing.get_type_hints(cls)
    props, req = {}, []
    for f in cls.node_fields():
        key = f.metadata.get("key") or f.name
        if (cls.__name__, f.name) in _V1_SECTIONS:
            s = _v1_section(_V1_SECTIONS[(cls.__name__, f.name)])
        elif f.metadata.get("schema") is not None:
            s = dict(f.metadata["schema"])
        else:
            s = _type_schema(hints[f.name])
        if f.metadata.get("enum"):
            if s.get("type") == "array":
                s["items"] = {**s.get("items", {}), "enum": list(f.metadata["enum"])}
            else:
                s["enum"] = list(f.metadata["enum"])
        s["description"] = f.metadata["doc"]
        if f.default is not None and f.default is not M.dataclasses.MISSING and not isinstance(f.default, M.Node):
            s["default"] = f.default
        props[key] = s
        if f.metadata.get("required"):
            req.append(key)
    out = {"type": "object", "description": _class_doc(cls, first_paragraph=True),
           "properties": props, "patternProperties": {"^x-": {}}, "additionalProperties": False}
    if req:
        out["required"] = req
    return out


def json_schema() -> dict:
    defs = {c.__name__: _node_schema(c) for c in M.NODE_TYPES}
    root = defs.pop("Estate")
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_ID,
            "title": f"relsynth IR {M.IR_VERSION}", **root, "$defs": defs}


def schema_text() -> str:
    return json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n"


def validate_json(doc: dict) -> list[str]:
    """Errors of a parsed document against the JSON Schema (needs the jsonschema
    package: pip install "relsynth[ir]"); [] when it is valid."""
    import jsonschema
    v = jsonschema.Draft202012Validator(json_schema())
    return [f"{'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}"
            for e in sorted(v.iter_errors(doc), key=lambda e: list(map(str, e.absolute_path)))]


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------

def _type_text(tp, f) -> str:
    if f.metadata.get("enum"):
        return " \\| ".join(f"`{x}`" for x in f.metadata["enum"])
    if typing.get_origin(tp) in (Union, types.UnionType):
        return " or ".join(_type_text(a, _NoEnum) for a in typing.get_args(tp) if a is not type(None))
    if tp is Any:
        return "any"
    o = typing.get_origin(tp)
    if o is list or tp is list:
        args = typing.get_args(tp)
        return f"list of {_type_text(args[0], _NoEnum)}" if args else "list"
    if o is dict or tp is dict:
        args = typing.get_args(tp)
        return f"map of {_type_text(args[1], _NoEnum)}" if args else "mapping"
    if isinstance(tp, type) and issubclass(tp, M.Node):
        return f"[{tp.__name__}](#{tp.__name__.lower()})"
    return {str: "string", int: "integer", float: "number", bool: "boolean"}.get(tp, str(tp))


class _NoEnum:
    metadata: dict = {}


def reference() -> str:
    out = [f"# IR {M.IR_VERSION} reference", "",
           "Generated from `src/relsynth/ir/model.py` by `python -m relsynth.ir.jsonschema reference`; do not edit.",
           "The JSON Schema is `src/relsynth/ir/schema/" + SCHEMA_FILE.name + "`. The guide is "
           "[ir-guide.md](ir-guide.md) and the design record [adr/0001-canonical-ir.md](adr/0001-canonical-ir.md).",
           "", "Every node also accepts vendor extensions: keys starting with `x-`, kept and never interpreted.", ""]
    for cls in M.NODE_TYPES:
        hints = typing.get_type_hints(cls)
        doc = _class_doc(cls)
        out += [f"## {cls.__name__}", "", doc, "", "| Key | Type | Default | Meaning |", "|---|---|---|---|"]
        for f in cls.node_fields():
            key = f.metadata.get("key") or f.name
            req = " **(required)**" if f.metadata.get("required") else ""
            d = "" if f.default in (None, M.dataclasses.MISSING) else f"`{json.dumps(f.default)}`"
            out.append(f"| `{key}`{req} | {_type_text(hints[f.name], f)} | {d} | {f.metadata['doc']} |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    what = argv[0] if argv else "schema"
    sys.stdout.write(schema_text() if what == "schema" else reference())


if __name__ == "__main__":
    main()
