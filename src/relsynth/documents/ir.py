# SPDX-License-Identifier: Apache-2.0
"""The inferred schema of document collections as an IR 2.0 estate.

Every collection is an entity of shape ``document``; nested objects are
attributes of type ``object`` with ``fields``, arrays are ``array`` with
``items`` and ``length: {min, max}``, maps are ``map`` with ``items``, a
discriminated sub-document carries ``variants``, and references are
relationships.  What the IR has no field for (the counts of each type of a
union, presence rates, length histograms) is kept in ``x-relsynth-counts``
extensions, which every IR reader carries along unchanged; a map's value schema is in
``x-relsynth-map`` (RFC-T13-01 in roadmap/rfcs/).
"""
from __future__ import annotations

from .infer import CollectionProfile, Node

_IR_TYPE = {"string": "string", "objectId": "string", "int": "integer", "long": "integer", "double": "number",
            "decimal": "number", "bool": "boolean", "date": "timestamp", "binary": "binary", "object": "object",
            "array": "array", "null": "string"}
_PHYSICAL = {"objectId": "ObjectId", "long": "int64", "int": "int32", "decimal": "decimal128", "double": "double"}


def _attr(nd: Node, *, key: bool = False) -> dict:
    vt = nd.value_types or ["null"]
    t = vt[0]
    a: dict = {"type": "map" if nd.is_map else _IR_TYPE.get(t, "string")}
    if t in _PHYSICAL:
        a["physical_type"] = _PHYSICAL[t]
    if not nd.nullable:
        a["nullable"] = False
    if nd.optional:
        a["optional"] = True
    if key:
        a["kind"] = "id"
    counts = {"present": nd.present, "of": nd.n, "types": dict(nd.types)}
    if nd.is_map or nd.is_array:
        counts["lengths"] = {str(k): v for k, v in sorted(nd.lengths.items())}
    if len(vt) > 1:
        counts["union"] = vt
    if nd.is_map:
        # IR 2.0 gives `items` and `length` to arrays only (RFC-T13-01 proposes
        # them for maps): the value schema and entry counts go in an extension
        lo, hi = nd.length_bounds()
        m = {"length": {"min": lo, "max": hi}}
        if nd.items is not None:
            m["values"] = _attr(nd.items)
        if nd.key_field:
            m["key_field"] = nd.key_field
        a["x-relsynth-map"] = m
    elif t == "object":
        fields = {k: _attr(f) for k, f in nd.fields.items()}
        if nd.variants:
            var = nd.variants
            cases = {}
            for val, names in var["cases"].items():
                cases[str(val)] = {n: fields[n] for n in names if n in fields}
            for n in var["fields"]:
                fields.pop(n, None)
            a["variants"] = {"discriminator": var["discriminator"], "cases": cases}
        a["fields"] = fields
    elif t == "array":
        lo, hi = nd.length_bounds()
        a["length"] = {"min": lo, "max": hi}
        if nd.items is not None:
            a["items"] = _attr(nd.items)
        if nd.distinct_items and nd.items is not None and nd.items.scalar_types:
            counts["distinct_items"] = True
    else:
        if nd.vmin is not None and t in ("int", "long", "double", "decimal"):
            a["min"], a["max"] = nd.vmin, nd.vmax
        if t in ("double", "decimal") and 0 < nd.decimals < 15:
            a["decimals"] = nd.decimals
    if nd.length_of:
        counts["length_of"] = nd.length_of
    a["x-relsynth-counts"] = counts
    return a


def to_ir_document(profiles: dict[str, CollectionProfile], references: list[dict], *, name: str = "documents",
                   system: str = "mongodb", database: str | None = None) -> dict:
    db = database or name
    doc = {"ir_version": "2.0", "name": name,
           "description": "Inferred by relsynth.documents (parametric schema inference with counts).",
           "containers": {db: {"kind": "collection", "system": system, "name": db}},
           "entities": {}, "relationships": []}
    for cname, prof in profiles.items():
        attrs = {k: _attr(f, key=(k == "_id")) for k, f in prof.root.fields.items()}
        ent = {"shape": "document", "attributes": attrs, "physical": {"container": db, "name": cname},
               "x-relsynth-counts": {"documents": prof.documents, "depth": prof.root.depth()}}
        if "_id" in prof.root.fields:
            ent["keys"] = [{"attributes": ["_id"], "primary": True, "kind": "surrogate"}]
        doc["entities"][cname] = ent
    for r in references:
        if r.get("candidate"):
            continue
        doc["relationships"].append({"name": f"{r['collection']}_{r['path']}".replace(".", "_").replace("[]", ""),
                                     "from": {"entity": r["collection"], "attributes": [r["path"]]},
                                     "to": {"entity": r["to"], "attributes": [r["to_path"]]}})
    return doc


def to_ir(profiles, references, *, name="documents", **kw):
    """The estate (relsynth.ir.Estate) of the inferred schema."""
    from ..ir import from_document
    return from_document(to_ir_document(profiles, references, name=name, **kw))
