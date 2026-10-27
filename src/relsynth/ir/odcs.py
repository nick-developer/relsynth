# SPDX-License-Identifier: Apache-2.0
"""Import and export of Open Data Contract Standard (ODCS) v3 contracts.

ODCS (Bitol, Linux Foundation AI & Data; https://bitol.io) describes datasets
as schema objects with typed properties, keys, foreign-key relationships,
partitions, classifications and quality rules.  `to_odcs` writes one contract
per estate (apiVersion v3.2.0): each entity becomes a schema object, each
attribute a property (nested objects and arrays included), keys and
relationships use the native fields, and rules become `custom` quality checks
run by relsynth.  What ODCS cannot say (roles, time, variants, privacy units,
synthesis settings) travels in custom properties with vendor `relsynth`, so
that `from_odcs(to_odcs(est))` gives the estate back.  `from_odcs` also reads
contracts written by other tools (v3.0.0 to v3.2.0) from their native fields.
"""
from __future__ import annotations

import copy
import re

from .model import (Attribute, AttributePrivacy, End, Entity, Estate, IRError, Key, PartitionField, Physical,
                    Relationship, Rule)

API_VERSION = "v3.2.0"
READS = ("v3.0.0", "v3.0.1", "v3.0.2", "v3.1.0", "v3.2.0")
VENDOR = "relsynth"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*$")
_SHAPE_PHYSICAL = {"record": "table", "document": "collection", "node": "node", "edge": "edge", "event": "topic"}
_ODCS_TYPES = ("string", "date", "timestamp", "time", "number", "integer", "object", "array", "boolean", "map")


def _cp(prop, value):
    return {"property": prop, "value": value, "vendor": VENDOR}


def _get_cp(node, prop):
    for c in node.get("customProperties") or []:
        if c.get("property") == prop:
            return c.get("value")
    return None


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def to_odcs(est: Estate, contract_id: str | None = None, version: str = "1.0.0", status: str = "draft") -> dict:
    """An ODCS v3.2.0 contract of the estate.  Passwords in source and target URLs
    are masked."""
    from ..data import mask_spec_urls
    c = {"apiVersion": API_VERSION, "kind": "DataContract", "id": contract_id or _stable_id(est.name or "estate"),
         "version": version, "status": status}
    if est.name:
        c["name"] = est.name
    if est.description:
        c["description"] = {"purpose": est.description}
    schema = []
    for n, e in est.entities.items():
        schema.append(_object(n, e, est))
    c["schema"] = schema
    rest = Estate(name=est.name, description=est.description, containers=est.containers, privacy=est.privacy,
                  synthesis=est.synthesis, review=est.review,
                  relationships=[r for r in est.relationships if not _native_rel(r, est)])
    masked = mask_spec_urls({k: v for k, v in (("source", est.source), ("target", est.target)) if v is not None})
    rest.source, rest.target = masked.get("source"), masked.get("target")
    payload = rest.to_dict()
    payload.pop("entities", None)
    payload["x-relationship-order"] = [r.name for r in est.relationships]
    c["customProperties"] = [_cp("relsynthEstate", payload)]
    return c


def _stable_id(s):
    return re.sub(r"[\s.#/\\@!%&^]+", "-", str(s)) or "estate"


def _native_rel(r: Relationship, est: Estate) -> bool:
    """Whether a relationship is written as a native ODCS foreign key."""
    if r.cardinality == "many_to_many" or not r.from_.attributes:
        return False
    names = [r.from_.entity, r.to.entity] + list(r.from_.attributes) + list(r.to.attributes or [])
    return all(_NAME.match(str(x)) for x in names) and not any("." in a for a in r.from_.attributes)


def _object(n: str, e: Entity, est: Estate) -> dict:
    o = {"name": n, "logicalType": "object",
         "physicalName": (e.physical.name if e.physical and e.physical.name else n),
         "physicalType": "view" if e.role == "view" else _SHAPE_PHYSICAL[e.shape]}
    if e.description:
        o["description"] = e.description
    pk = e.primary_key
    pk_pos = {a: i + 1 for i, a in enumerate(pk.attributes)} if pk else {}
    single_unique = {k.attributes[0] for k in e.keys if not k.primary and k.where is None and len(k.attributes) == 1}
    parts = {p.source: i + 1 for i, p in enumerate(e.physical.partitioning)
             if p.transform == "identity"} if e.physical else {}
    o["properties"] = [_property(a, attr, pk_pos.get(a), a in single_unique, parts.get(a))
                       for a, attr in e.attributes.items()]
    rels = []
    for r in est.relationships:
        if r.from_.entity != n or not _native_rel(r, est):
            continue
        ta = r.to.attributes or (est.entities[r.to.entity].primary_key.attributes
                                 if r.to.entity in est.entities and est.entities[r.to.entity].primary_key else [])
        if not ta:
            continue
        fr = [f"{n}.{a}" for a in r.from_.attributes]
        to = [f"{r.to.entity}.{a}" for a in ta]
        rd = {"type": "foreignKey", "from": fr[0] if len(fr) == 1 else fr, "to": to[0] if len(to) == 1 else to,
              "customProperties": [_cp("relsynthRelationship", r.to_dict())]}
        rels.append(rd)
    if rels:
        o["relationships"] = rels
    q = [{"type": "custom", "engine": VENDOR, "name": r.name or f"{r.type} rule",
          "description": r.description or r.expr or r.type, "implementation": r.to_dict(),
          "customProperties": [_cp("relsynthRuleIndex", i)]}
         for i, r in enumerate(est.rules) if r.entity == n]
    if q:
        o["quality"] = q
    payload = e.to_dict()
    payload.pop("attributes", None)
    payload.pop("description", None)
    o["customProperties"] = [_cp("relsynthEntity", payload)]
    return o


def _logical(a: Attribute) -> str:
    t = a.effective_type
    if t in _ODCS_TYPES:
        return t
    return "string"           # binary and untyped values (id, categorical, lookup, constant)


def _property(name, a: Attribute, pk_pos=None, unique=False, part_pos=None, item=False) -> dict:
    p = {} if item else {"name": name}
    p["logicalType"] = _logical(a)
    if a.physical_type:
        p["physicalType"] = a.physical_type
    if a.physical_name and not item:
        p["physicalName"] = a.physical_name
    if a.description:
        p["description"] = a.description
    if not item:
        p["required"] = not a.nullable and not a.optional
    if pk_pos:
        p["primaryKey"] = True
        p["primaryKeyPosition"] = pk_pos
    if unique:
        p["unique"] = True
    if part_pos:
        p["partitioned"] = True
        p["partitionKeyPosition"] = part_pos
    if a.privacy is not None and (a.privacy.classification or a.privacy.role):
        p["classification"] = a.privacy.classification or a.privacy.role
    opts = {}
    if p["logicalType"] in ("number", "integer"):
        if isinstance(a.min, (int, float)) and not isinstance(a.min, bool):
            opts["minimum"] = a.min
        if isinstance(a.max, (int, float)) and not isinstance(a.max, bool):
            opts["maximum"] = a.max
    if a.effective_type == "binary":
        opts["format"] = "binary"
    if opts:
        p["logicalTypeOptions"] = opts
    if a.fields:
        p["properties"] = [_property(k, f) for k, f in a.fields.items()]
    if a.items is not None:
        p["items"] = _property("items", a.items, item=True)
    if p["logicalType"] == "map":
        p["map"] = {"key": {"logicalType": "string"}, "value": {"logicalType": "string"}}
    payload = a.to_dict()
    for k in ("fields", "items", "description"):
        payload.pop(k, None)
    p["customProperties"] = [_cp("relsynthAttribute", payload)]
    return p


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

def from_odcs(contract: dict) -> Estate:
    """The estate a contract describes.  Contracts written by to_odcs come back
    whole; others are read from their native fields (objects, properties, keys,
    relationships, partitions, classifications, sql and custom quality rules)."""
    if not isinstance(contract, dict) or contract.get("kind", "DataContract") != "DataContract":
        raise IRError("not an ODCS data contract (kind DataContract)")
    if contract.get("apiVersion") not in READS:
        raise IRError(f"ODCS apiVersion {contract.get('apiVersion')!r} is not supported (reads {', '.join(READS)})")
    payload = _get_cp(contract, "relsynthEstate")
    order = None
    if payload is not None:
        payload = copy.deepcopy(payload)
        order = payload.pop("x-relationship-order", None)
        est = Estate.from_dict({"ir_version": "2.0", **payload, "entities": {}})
    else:
        desc = contract.get("description")
        est = Estate(name=contract.get("name") or contract.get("id"),
                     description=desc.get("purpose") or desc.get("usage") if isinstance(desc, dict) else None)
    native_rels, rules = [], []
    for o in contract.get("schema") or []:
        n = o.get("name")
        ep = _get_cp(o, "relsynthEntity")
        attrs = {p.get("name"): _attr_from(p) for p in o.get("properties") or []}
        if ep is not None:
            e = Entity.from_dict({**copy.deepcopy(ep), "attributes": {}}, f"entities.{n}")
            if o.get("description"):
                e.description = o["description"]
            e.attributes = attrs
        else:
            e = _entity_from(o, attrs)
        est.entities[n] = e
        for rd in o.get("relationships") or []:
            native_rels.append(_rel_from(n, rd))
        for p in o.get("properties") or []:
            for rd in p.get("relationships") or []:
                native_rels.append(_rel_from(n, {**rd, "from": f"{n}.{p.get('name')}"}))
        for q in o.get("quality") or []:
            r = _rule_from(n, q)
            if r is not None:
                rules.append((_get_cp(q, "relsynthRuleIndex"), len(rules), r))
    if rules:
        est.rules = [r for _, _, r in sorted(rules, key=lambda x: (x[0] is None, x[0] or 0, x[1]))]
    # relationships in the estate's order: native ones were written per object, the
    # rest in the estate payload; the payload of each native one carries its name
    all_rels = native_rels + list(est.relationships)
    if order:
        all_rels.sort(key=lambda r: order.index(r.name) if r.name in order else len(order))
    est.relationships = all_rels
    return est


def _attr_from(p: dict) -> Attribute:
    ap = _get_cp(p, "relsynthAttribute")
    nested = {q.get("name"): _attr_from(q) for q in p.get("properties") or []}
    items = _attr_from(p["items"]) if isinstance(p.get("items"), dict) else None
    if ap is not None:
        a = Attribute.from_dict(copy.deepcopy(ap), "attribute")
        a.fields, a.items = nested, items
        if p.get("description"):
            a.description = p["description"]
        return a
    lt = p.get("logicalType")
    opts = p.get("logicalTypeOptions") or {}
    a = Attribute(type=("binary" if opts.get("format") == "binary" else lt) if lt in _ODCS_TYPES else None,
                  nullable=not p.get("required", False), description=p.get("description"),
                  physical_type=p.get("physicalType"), physical_name=p.get("physicalName"),
                  fields=nested, items=items)
    if a.type is None:
        a.kind = "categorical"
    if p.get("primaryKey") and lt in ("integer", "string"):
        a.kind = "id"
    for k, f in (("minimum", "min"), ("maximum", "max")):
        if k in opts:
            setattr(a, f, opts[k])
    cl = p.get("classification")
    if cl:
        role = cl if cl in ("direct_identifier", "quasi_identifier", "sensitive", "non_sensitive") else None
        a.privacy = AttributePrivacy(role=role, classification=None if role else cl)
    return a


def _entity_from(o: dict, attrs: dict) -> Entity:
    pt = (o.get("physicalType") or "table").lower()
    shape = {"collection": "document", "node": "node", "edge": "edge", "topic": "event", "stream": "event"}.get(pt,
                                                                                                             "record")
    if shape == "record" and any(a.fields or a.items for a in attrs.values()):
        shape = "document"
    e = Entity(shape=shape, role="view" if pt == "view" else "entity", description=o.get("description"),
               attributes=attrs)
    props = o.get("properties") or []
    pk = sorted((p for p in props if p.get("primaryKey")), key=lambda p: p.get("primaryKeyPosition", -1))
    if pk:
        e.keys.append(Key(attributes=[p["name"] for p in pk], primary=True))
    for p in props:
        if p.get("unique") and not p.get("primaryKey"):
            e.keys.append(Key(attributes=[p["name"]]))
    parts = sorted((p for p in props if p.get("partitioned")), key=lambda p: p.get("partitionKeyPosition", -1))
    if parts or (o.get("physicalName") and o.get("physicalName") != o.get("name")):
        e.physical = Physical(name=o.get("physicalName") if o.get("physicalName") != o.get("name") else None,
                              partitioning=[PartitionField(source=p["name"]) for p in parts])
    if e.role == "view":
        e.role = "entity"        # a view needs view.of, which ODCS does not state: read as an entity
    return e


def _split_ref(ref):
    refs = ref if isinstance(ref, list) else [ref]
    out = []
    for x in refs:
        x = str(x)
        if "/" in x:
            raise IRError(f"fully qualified ODCS references ({x}) are not supported yet; use table.column")
        t, _, c = x.partition(".")
        out.append((t, c))
    return out


def _rel_from(n: str, rd: dict) -> Relationship:
    rp = _get_cp(rd, "relsynthRelationship")
    if rp is not None:
        return Relationship.from_dict(copy.deepcopy(rp), "relationship")
    fr, to = _split_ref(rd["from"]), _split_ref(rd["to"])
    ft, tt = fr[0][0], to[0][0]
    if any(t != ft for t, _ in fr) or any(t != tt for t, _ in to):
        raise IRError(f"{n}: a relationship's columns must come from one object on each side")
    fa, ta = [c for _, c in fr], [c for _, c in to]
    name = rd.get("id") or (f"{ft}.{fa[0]}->{tt}" if len(fa) == 1 else f"{ft}.({'+'.join(fa)})->{tt}")
    return Relationship(name=name, from_=End(entity=ft, attributes=fa), to=End(entity=tt, attributes=ta),
                        kind="hierarchy" if ft == tt else "reference")


def _rule_from(n: str, q: dict):
    if q.get("type") == "custom" and q.get("engine") == VENDOR and isinstance(q.get("implementation"), dict):
        return Rule.from_dict(q["implementation"], "rule")
    return None          # text, library and sql checks have no relsynth rule form; kept in the contract
