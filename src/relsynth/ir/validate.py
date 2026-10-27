# SPDX-License-Identifier: Apache-2.0
"""Semantic validation of an IR document: every name resolves, every structure is
used where it means something, and the relational part passes the 1.0 checks.

`check(est)` returns (errors, warnings).  Structure (keys, types) is checked when
the document is read (`Estate.from_dict`); this module checks meaning.
"""
from __future__ import annotations

import re
import string

from .model import Attribute, Entity, Estate

_PLACEHOLDER = string.Formatter()


def check(est: Estate, relational: bool = True) -> tuple[list[str], list[str]]:
    """Errors and warnings of a document.  relational: also run the 1.0 spec
    validator on the relational part when the whole document is relational."""
    errs, warns = [], []
    ents = est.entities
    if not ents:
        errs.append("entities: no entity declared")
    rels = {}
    for r in est.relationships:
        if r.name in rels:
            errs.append(f"relationships: name {r.name!r} is used twice")
        rels[r.name] = r
    for n, e in ents.items():
        _entity(n, e, est, rels, errs, warns)
    for r in est.relationships:
        _relationship(r, est, rels, errs)
    for i, r in enumerate(est.rules):
        w = f"rules[{i}]"
        if r.entity not in ents:
            errs.append(f"{w}: unknown entity {r.entity!r}")
            continue
        if r.type in ("check", "derive") and not r.expr:
            errs.append(f"{w}: a {r.type} rule needs expr")
        if r.type == "aggregate":
            if r.source not in ents:
                errs.append(f"{w}: aggregate needs source: <child entity>")
            if r.attribute not in ents[r.entity].attributes:
                errs.append(f"{w}: aggregate attribute {r.attribute!r} is not an attribute of {r.entity}")
        if r.via is not None and r.via not in rels:
            errs.append(f"{w}: via {r.via!r} is not a relationship")
        if r.holds is not None and not 0 < r.holds <= 1:
            errs.append(f"{w}: holds must be in (0, 1]")
    for cn, c in est.containers.items():
        users = [n for n, e in ents.items() if e.physical and e.physical.container == cn]
        keys = [(k, getattr(c, k)) for k in ("partition_key", "sort_key")]
        keys += [(f"indexes.{ix.name}.{k}", getattr(ix, k)) for ix in c.indexes for k in ("partition_key", "sort_key")]
        for k, v in keys:
            if v is None:
                continue
            for u in users:
                if not _has(ents[u], v) and v not in ents[u].physical.key_templates:
                    if k.startswith("indexes."):
                        continue      # an entity need not appear in every secondary index (sparse index)
                    errs.append(f"containers.{cn}.{k}: {u} stores no {v!r} (an attribute or a key template)")
        if len(users) > 1 and c.kind != "graph":     # a graph tells node labels and edge types apart
            for u in users:
                if ents[u].physical.discriminator is None:
                    warns.append(f"containers.{cn}: {u} shares it without a discriminator")
    _privacy(est, rels, errs, warns)
    if relational and not errs:
        from .v1 import _lower
        spec, unsupported = _lower(est)
        if not unsupported:
            from ..spec import SpecError, normalise, validate
            try:
                errs += [f"(as relsynth 1.0 spec) {e}" for e in validate(normalise(spec))]
            except SpecError as e:
                errs.append(f"(as relsynth 1.0 spec) {e}")
    return errs, warns


def _attrs(where, attrs: dict[str, Attribute], errs):
    for an, a in attrs.items():
        w = f"{where}.{an}"
        t = a.type
        if a.fields and t not in (None, "object"):
            errs.append(f"{w}: fields need type object")
        if a.items is not None and t != "array":
            errs.append(f"{w}: items need type array")
        if t == "array" and a.items is None:
            errs.append(f"{w}: an array needs items")
        if a.length is not None:
            if t != "array":
                errs.append(f"{w}: length is for arrays")
            lo, hi = a.length.get("min"), a.length.get("max")
            if lo is not None and hi is not None and lo > hi:
                errs.append(f"{w}: length min {lo} > max {hi}")
        if a.type is None and a.kind is None:
            errs.append(f"{w}: needs a type or a kind")
        if a.kind == "sequence" and a.sequence is None:
            errs.append(f"{w}: kind sequence needs sequence.within")
        if a.variants is not None:
            _variants(w, a.variants, a.fields, errs)
        _attrs(f"{w}.fields", a.fields, errs)
        if a.items is not None:
            _attrs(f"{w}", {"items": a.items}, errs)


def _variants(where, v, attrs, errs):
    if v.discriminator not in attrs:
        errs.append(f"{where}.variants: discriminator {v.discriminator!r} is not an attribute")
    for val, case in v.cases.items():
        clash = set(case) & set(attrs)
        if clash:
            errs.append(f"{where}.variants.cases.{val}: {sorted(clash)} are already common attributes")
        _attrs(f"{where}.variants.cases.{val}", case, errs)


def resolve(e: Entity, path: str) -> Attribute | None:
    """The attribute a name or a nested path (customer.id, items.sku) denotes, or None."""
    if path in e.attributes:
        return e.attributes[path]
    cur, attrs = None, e.attributes
    for part in str(path).split("."):
        cur = attrs.get(part)
        if cur is None:
            return None
        while cur.type == "array" and cur.items is not None:
            cur = cur.items
        attrs = cur.fields
    return cur


def _has(e: Entity, a: str) -> bool:
    return resolve(e, a) is not None


def _entity(n, e: Entity, est: Estate, rels, errs, warns):
    w = f"entities.{n}"
    ents = est.entities
    _attrs(f"{w}.attributes", e.attributes, errs)
    pks = [k for k in e.keys if k.primary]
    if len(pks) > 1:
        errs.append(f"{w}: more than one primary key")
    for i, k in enumerate(e.keys):
        if not k.attributes:
            errs.append(f"{w}.keys[{i}]: no attributes")
        for a in k.attributes:
            if not _has(e, a):
                errs.append(f"{w}.keys[{i}]: {a!r} is not an attribute")
        if k.primary and k.where:
            errs.append(f"{w}.keys[{i}]: a primary key cannot be filtered (where)")
    for i, d in enumerate(e.dependencies):
        for a in [d.determinant] + d.dependents:
            if not _has(e, a):
                errs.append(f"{w}.dependencies[{i}]: {a!r} is not an attribute")
    if e.variants is not None:
        _variants(w, e.variants, e.attributes, errs)
    if (e.role == "view") != (e.view is not None):
        errs.append(f"{w}: role view and view.of go together")
    if e.view is not None and (e.view.of not in ents or ents[e.view.of].role == "view"):
        errs.append(f"{w}.view.of: {e.view.of!r} must be an entity that is not a view")
    t = e.time
    if t is not None:
        for k in ("event_time", "ingestion_time"):
            if getattr(t, k) is not None and not _has(e, getattr(t, k)):
                errs.append(f"{w}.time.{k}: {getattr(t, k)!r} is not an attribute")
        if t.valid_time is not None:
            for a in (t.valid_time.from_, t.valid_time.to):
                if a is not None and not _has(e, a):
                    errs.append(f"{w}.time.valid_time: {a!r} is not an attribute")
        if t.versioning is not None:
            if t.valid_time is None:
                errs.append(f"{w}.time.versioning needs valid_time")
            if not _has(e, t.versioning.natural_key):
                errs.append(f"{w}.time.versioning.natural_key: {t.versioning.natural_key!r} is not an attribute")
    if e.allocation is not None:
        for a in (e.allocation.group, e.allocation.weight):
            if not _has(e, a):
                errs.append(f"{w}.allocation: {a!r} is not an attribute")
    if (e.shape == "edge") != (e.edge is not None):
        errs.append(f"{w}: shape edge and edge go together")
    if e.edge is not None:
        for side in ("source", "target"):
            rn = getattr(e.edge, side)
            r = rels.get(rn)
            if r is None:
                errs.append(f"{w}.edge.{side}: {rn!r} is not a relationship")
            elif r.from_.entity != n:
                errs.append(f"{w}.edge.{side}: relationship {rn} does not start at {n}")
            elif r.to.entity in ents and ents[r.to.entity].shape != "node":
                warns.append(f"{w}.edge.{side}: endpoint {r.to.entity} is not a node entity")
    if e.stream is not None:
        if e.shape != "event":
            errs.append(f"{w}: stream is for shape event")
        for a in e.stream.key + ([e.stream.order_by] if e.stream.order_by else []):
            if not _has(e, a):
                errs.append(f"{w}.stream: {a!r} is not an attribute")
        if e.stream.max_lateness is not None and not re.fullmatch(r"P(?!$)(\d+D)?(T(?=\d)(\d+H)?(\d+M)?(\d+(\.\d+)?S)?)?",
                                                                    e.stream.max_lateness):
            errs.append(f"{w}.stream.max_lateness: {e.stream.max_lateness!r} is not an ISO 8601 duration (PT5M)")
    p = e.physical
    if p is not None:
        if p.container is not None and p.container not in est.containers:
            errs.append(f"{w}.physical.container: {p.container!r} is not in containers")
        for i, pf in enumerate(p.partitioning):
            if not _has(e, pf.source):
                errs.append(f"{w}.physical.partitioning[{i}]: {pf.source!r} is not an attribute")
            if pf.transform in ("bucket", "truncate") and not (isinstance(pf.param, int) and pf.param > 0):
                errs.append(f"{w}.physical.partitioning[{i}]: {pf.transform} needs param > 0")
            src = e.attributes.get(pf.source)
            if src is not None and pf.transform in ("year", "month", "day", "hour") \
                    and src.effective_type not in ("date", "timestamp", None):
                errs.append(f"{w}.physical.partitioning[{i}]: {pf.transform} needs a date or timestamp")
            if src is not None and pf.transform == "hour" and src.effective_type == "date":
                errs.append(f"{w}.physical.partitioning[{i}]: hour needs a timestamp")
        for i, sf in enumerate(p.sort_order):
            if not _has(e, sf.source):
                errs.append(f"{w}.physical.sort_order[{i}]: {sf.source!r} is not an attribute")
        for a in p.clustering:
            if not _has(e, a):
                errs.append(f"{w}.physical.clustering: {a!r} is not an attribute")
        if p.discriminator is not None and not _has(e, p.discriminator.attribute):
            errs.append(f"{w}.physical.discriminator: {p.discriminator.attribute!r} is not an attribute")
        for k, tpl in p.key_templates.items():
            try:
                names = [f for _, f, _, _ in _PLACEHOLDER.parse(tpl) if f]
            except ValueError as ex:
                errs.append(f"{w}.physical.key_templates.{k}: {ex}")
                continue
            for a in names:
                if not _has(e, a):
                    errs.append(f"{w}.physical.key_templates.{k}: {{{a}}} is not an attribute")
    s = e.synthesis
    if s is not None and s.fact is not None:
        for k in ("anchor", "partner"):
            v = getattr(s.fact, k)
            if v is not None and v not in ents and v not in rels and not _has(e, v):
                errs.append(f"{w}.synthesis.fact.{k}: {v!r} is not an entity, a relationship or a key attribute")


def _relationship(r, est: Estate, rels, errs):
    w = f"relationships.{r.name}"
    ents = est.entities
    f, t = r.from_, r.to
    if f.entity not in ents or t.entity not in ents:
        errs.append(f"{w}: unknown entity ({f.entity!r} -> {t.entity!r})")
        return
    fe, te = ents[f.entity], ents[t.entity]
    if r.cardinality == "many_to_many":
        if r.via is None:
            errs.append(f"{w}: many_to_many needs via (the bridge entity and its two references)")
        else:
            for side, want in (("from_", f.entity), ("to", t.entity)):
                vr = rels.get(getattr(r.via, side))
                if vr is None or vr.from_.entity != r.via.entity or vr.to.entity != want:
                    errs.append(f"{w}.via: {getattr(r.via, side)!r} must be a relationship from {r.via.entity} "
                                f"to {want}")
        return
    fa = f.attributes or []
    if not fa:
        errs.append(f"{w}: from.attributes are needed")
    for a in fa:
        if "." in a and fe.shape != "document":
            # a 1.0 composite key may name a parent's column through a foreign key (fk.col)
            if len(fa) == 1:
                errs.append(f"{w}: fk.col paths are allowed in composite references only")
        elif not _has(fe, a):
            errs.append(f"{w}: {a!r} is not an attribute of {f.entity}")
    ta = t.attributes or (te.primary_key.attributes if te.primary_key else None)
    if ta is None:
        errs.append(f"{w}: {t.entity} has no primary key; name to.attributes")
    else:
        for a in ta:
            if a not in te.attributes:
                errs.append(f"{w}: {a!r} is not an attribute of {t.entity}")
        if len(ta) != len(fa):
            errs.append(f"{w}: from and to have {len(fa)} and {len(ta)} attributes")
        if not any(sorted(k.attributes) == sorted(ta) and k.where is None for k in te.keys) and te.keys:
            errs.append(f"{w}: to.attributes {ta} are not a key of {t.entity}")
    if r.kind == "hierarchy" and f.entity != t.entity:
        errs.append(f"{w}: a hierarchy references its own entity")
    if r.cardinality == "one_to_one" and not any(sorted(k.attributes) == sorted(fa) and k.where is None
                                                 for k in fe.keys):
        errs.append(f"{w}: one_to_one needs from.attributes {fa} to be a key of {f.entity}")
    if r.offset is not None:
        base = next((x for x in est.relationships if x.from_.entity == f.entity and x.from_.attributes ==
                     [r.offset.from_] and x.to.entity == t.entity), None)
        if base is None:
            errs.append(f"{w}.offset.from: {r.offset.from_!r} is not another reference of {f.entity} to {t.entity}")
        if r.offset.order_by not in te.attributes:
            errs.append(f"{w}.offset.order_by: {r.offset.order_by!r} is not an attribute of {t.entity}")


def _privacy(est: Estate, rels, errs, warns):
    pv = est.privacy
    for n, e in est.entities.items():
        for an, a in e.attributes.items():
            if a.privacy is not None and a.privacy.role == "direct_identifier" and e.privacy and e.privacy.public:
                errs.append(f"entities.{n}: marked public but {an} is a direct identifier")
    if pv is None:
        return
    for rn in pv.exclude_paths:
        if rn not in rels:
            errs.append(f"privacy.exclude_paths: {rn!r} is not a relationship")
    if pv.unit is None:
        return
    if pv.unit.entity not in est.entities:
        errs.append(f"privacy.unit.entity: {pv.unit.entity!r} is not an entity")
        return
    from .privacy import trace
    tr = trace(est)
    for n, info in tr.entities.items():
        e = est.entities[n]
        if e.privacy is not None and e.privacy.public and info.status in ("unit", "owned"):
            errs.append(f"entities.{n}: marked public but holds data of the privacy unit "
                        f"({' -> '.join(info.paths[0]) if info.paths else 'it is the unit'})")
    warns += tr.warnings
