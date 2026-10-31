# SPDX-License-Identifier: Apache-2.0
"""Validation of documents against an inferred schema.

The inferred schema (`infer.Node` tree) admits a document when

* every field it holds is a field of the schema (a map admits any name),
* every field present in every source object is present (within a variant:
  every field of the case that the discriminator names, and no field of
  another case),
* every value has a type the field took in the source (null included),
* every array and map has a length within the source's [min, max],
* every length field equals the length of its array.

These are the structural facts that hold on every source document by
construction of the counts (Proposition 1400), so a synthetic document that
passes has the source's shape.  Value ranges are not part of it: they are
statistical, and the quality metrics measure them.
"""
from __future__ import annotations

from typing import Any

from .infer import CollectionProfile, Node
from .values import type_name


def violations(node: Node, v: Any, path: str = "", out: list | None = None, limit: int = 20) -> list[str]:
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    t = type_name(v)
    if node.types.get(t, 0) == 0:
        out.append(f"{path or '$'}: type {t} not in {sorted(node.types)}")
        return out
    if t == "object" and node.is_map:
        lo, hi = node.length_bounds()
        if not lo <= len(v) <= hi:
            out.append(f"{path}: {len(v)} entries outside [{lo}, {hi}]")
        for k, x in v.items():
            violations(node.items, x, f"{path}.{k}", out, limit)
    elif t == "object":
        nobj = node.types["object"]
        var = node.variants
        case = None
        if var:
            case = set(var["cases"].get(v.get(var["discriminator"]), []))
            if v.get(var["discriminator"]) not in var["cases"]:
                out.append(f"{path}.{var['discriminator']}: unknown variant {v.get(var['discriminator'])!r}")
        for k, x in v.items():
            f = node.fields.get(k)
            if f is None:
                out.append(f"{path}.{k}: field not in the schema")
                continue
            if var and k in var["fields"] and k not in case:
                out.append(f"{path}.{k}: not a field of variant {v.get(var['discriminator'])!r}")
            violations(f, x, f"{path}.{k}" if path else k, out, limit)
            if f.length_of is not None:
                arr = v.get(f.length_of)
                if not isinstance(arr, list) or x != len(arr):
                    out.append(f"{path}.{k}: {x!r} is not the length of {f.length_of}")
        for k, f in node.fields.items():
            if k in v:
                continue
            required = f.present == nobj or (var and k in var["fields"] and k in case)
            if required:
                out.append(f"{path}.{k}: required field missing" if path else f"{k}: required field missing")
    elif t == "array":
        lo, hi = node.length_bounds()
        if not lo <= len(v) <= hi:
            out.append(f"{path}: length {len(v)} outside [{lo}, {hi}]")
        for i, x in enumerate(v):
            violations(node.items, x, f"{path}[{i}]", out, limit)
    return out


def validate_collection(prof: CollectionProfile, docs, limit: int = 20) -> dict:
    """{documents, valid, share, examples} for documents against a collection's schema."""
    n = bad = 0
    examples = []
    for d in docs:
        n += 1
        v = violations(prof.root, d, "", None, 5)
        if v:
            bad += 1
            if len(examples) < limit:
                examples.extend(v[:2])
    return {"documents": n, "valid": n - bad, "share": (n - bad) / n if n else 1.0, "examples": examples[:limit]}
