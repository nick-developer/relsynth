# SPDX-License-Identifier: Apache-2.0
"""Conversion between a relsynth 1.0 YAML spec and the IR.

`lift(raw)` reads a 1.0 spec (as written, or normalised) into an `Estate`;
`lower(estate)` writes the 1.0 spec the engine reads.  The pair is loss-free on
1.0 specs: `normalise(lower(lift(s))) == normalise(s)` for every spec the 1.0
key schema accepts (Prop. 106), so the engine sees the same spec and, for the
same seed, writes the same synthetic data.  An IR document that uses what a 1.0
spec cannot say (documents, graphs, streams, partitions, privacy annotations)
lowers to its relational part; `lower` reports what was left out, and
`strict=True` refuses instead.

Field correspondence (1.0 -> IR):

    tables.T                       entities.T  (shape record)
      role, description            role, description
      mode, key_order              synthesis.mode, synthesis.key_order
      primary_key                  keys[] {attributes: [pk], primary: true}
      unique / partial_unique      keys[] {attributes} / {attributes, where}
      columns.C                    attributes.C (type -> kind; see Attribute)
      fact (without time_column)   synthesis.fact
      fact.time_column             time.event_time
      scd                          time.valid_time + time.versioning
      dimensions                   dependencies
      view_of                      view.of
      bridge                       allocation
      group_effects, link_affinity synthesis.group_effects, synthesis.link_affinity
      unknown_members              synthesis.unknown_members
      generator                    synthesis.generator {name, version, options}
      sequence, series             synthesis.sequence, synthesis.series (options unchanged)
    relationships[]                relationships[] (child -> from, parent -> to)
    rules[]                        rules[] (table -> entity, column -> attribute, from -> source)
    version: 1                     ir_version: "2.0"
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any

from .model import (KIND_TYPE, TYPE_KIND, Allocation, Attribute, AttributeSynthesis, Dependency, End, Entity,
                    EntitySynthesis, Estate, FactModel, GeneratorChoice, GroupEffect, Header, IRError, Key,
                    KeyedLookup, LinkAffinity, Node, Offset, Relationship, Rule, Sequence, Time, ValidTime, Versioning,
                    View, ends)

TOP_KEYS = ("version", "name", "description", "source", "target", "tables", "relationships", "rules", "synthesis",
            "review")
_ATTR_SYNTH = ("distribution", "text", "pii", "offset_from", "ratio_of", "ratio_same_as")
_FACT_KEYS = ("anchor", "partner", "pairs", "multiplicity", "count_model", "group_fallback", "time_bucket",
              "condition_on_parents")
_REL_KIND = {"many_to_one": "reference", "self_reference": "hierarchy", "derived": "derived"}
_REL_KIND_BACK = {v: k for k, v in _REL_KIND.items()}
_RULE_RENAME = {"table": "entity", "column": "attribute", "from": "source"}
_RULE_BACK = {v: k for k, v in _RULE_RENAME.items()}


def is_ir(doc) -> bool:
    """Whether a parsed document is an IR document (it has `ir_version`)."""
    return isinstance(doc, dict) and "ir_version" in doc


# ---------------------------------------------------------------------------
# 1.0 -> IR
# ---------------------------------------------------------------------------

def lift(raw: dict) -> Estate:
    """The IR of a 1.0 spec (raw as written, or normalised).  Raises IRError on
    keys a 1.0 spec cannot have; the spec's own validation is left to the 1.0
    validator (a spec with errors still converts)."""
    from ..schema import check_keys
    if not isinstance(raw, dict):
        raise IRError("a 1.0 spec must be a mapping")
    raw = copy.deepcopy({k: v for k, v in raw.items() if not str(k).startswith("_")})
    bad = [e for e in check_keys(raw) if "unknown key" in e]
    if bad:
        raise IRError(bad)
    tables = raw.get("tables") or {}
    rels = raw.get("relationships") or []
    est = Estate(name=raw.get("name"), description=raw.get("description"), source=raw.get("source"),
                 target=raw.get("target"), synthesis=raw.get("synthesis") or {}, review=list(raw.get("review") or []))
    pks = {t: (T or {}).get("primary_key") for t, T in tables.items() if isinstance(T, dict)}
    rel_objs = [_lift_rel(r, pks) for r in rels]
    for tname, T in tables.items():
        est.entities[tname] = _lift_table(tname, T or {}, rel_objs)
    for r in rel_objs:
        src, _ = ends(r)
        if r.cardinality == "many_to_one" and src.entity in est.entities and r.kind == "reference":
            ent = est.entities[src.entity]
            if any(sorted(k.attributes) == sorted(src.attributes or []) and k.where is None for k in ent.keys):
                r.cardinality = "one_to_one"
    est.relationships = rel_objs
    est.rules = [_lift_rule(r) for r in raw.get("rules") or []]
    return est


def _default_rel_name(r):
    ck = r.get("child_key")
    if isinstance(ck, (list, tuple)):
        return f"{r.get('child')}.({'+'.join(ck)})->{r.get('parent')}"
    return f"{r.get('child')}.{ck}->{r.get('parent')}"


def _lift_rel(r: dict, pks: dict) -> Relationship:
    ck: Any = r.get("child_key")
    composite = isinstance(ck, (list, tuple))
    pk: Any
    to_attrs: list | None
    if composite:
        pk = r.get("parent_key", list(ck))
        to_attrs = list(pk) if isinstance(pk, (list, tuple)) else [pk]
    else:
        pk = r.get("parent_key", pks.get(r.get("parent")))
        to_attrs = [pk] if pk is not None else None
    kind = r.get("kind") or ("self_reference" if r.get("child") == r.get("parent") else "many_to_one")
    rel = Relationship(name=r.get("name") or _default_rel_name(r),
                       from_=End(entity=r.get("child"), attributes=list(ck) if composite else [ck]),
                       to=End(entity=r.get("parent"), attributes=to_attrs),
                       kind=_REL_KIND.get(kind, kind))
    if r.get("offset_from") is not None or r.get("order_by") is not None:
        rel.offset = Offset(from_=r.get("offset_from"), order_by=r.get("order_by"))
    return rel


def _lift_table(tname: str, T: dict, rels) -> Entity:
    e = Entity(role=T.get("role", "entity"), description=T.get("description"))
    for c, cs in (T.get("columns") or {}).items():
        e.attributes[c] = _lift_column(cs)
    if T.get("primary_key") is not None:
        e.keys.append(Key(attributes=[T["primary_key"]], primary=True))
    u = T.get("unique") or []
    if u and all(isinstance(x, str) for x in u):
        u = [u]
    partial = list(T.get("partial_unique") or []) + [x for x in u if isinstance(x, dict) and x.get("where")]
    for x in u:
        if isinstance(x, dict) and x.get("where"):
            continue
        e.keys.append(Key(attributes=list(x["columns"] if isinstance(x, dict) else x)))
    for x in partial:
        e.keys.append(Key(attributes=list(x["columns"]), where=str(x["where"])))
    for d in T.get("dimensions") or []:
        e.dependencies.append(Dependency(determinant=d.get("key"), dependents=list(d.get("columns") or []),
                                         anchor=d.get("anchor") or None))
    if T.get("view_of") is not None:
        e.view = View(of=T["view_of"])
    f = T.get("fact")
    tm = Time()
    if isinstance(f, dict) and f.get("time_column") is not None:
        tm.event_time = f["time_column"]
    sc = T.get("scd")
    if isinstance(sc, dict):
        tm.valid_time = ValidTime(from_=sc.get("valid_from"), to=sc.get("valid_to"))
        tm.versioning = Versioning(natural_key=sc.get("natural_key"), scd_type=sc.get("type"),
                                   stable=sc.get("stable"), current=sc.get("current"))
    if tm.event_time or tm.valid_time:
        e.time = tm
    br = T.get("bridge")
    if isinstance(br, dict):
        e.allocation = Allocation(group=br.get("group"), weight=br.get("weight"), total=br.get("total"),
                                  decimals=br.get("decimals"))
    syn = EntitySynthesis(mode=T.get("mode"), key_order=T.get("key_order"))
    if isinstance(f, dict):
        fm = FactModel(**{k: f[k] for k in _FACT_KEYS if k in f})
        if isinstance(f.get("header"), dict):
            fm.header = Header(**{k: f["header"][k] for k in ("key", "columns", "line_number") if k in f["header"]})
        for k in ("anchor", "partner"):
            # a normalised spec stores a role-playing anchor as (table, key column)
            if f.get(f"{k}_key") is not None:
                r = next((r for r in rels if r.from_.entity == tname and r.to.entity == f.get(k)
                          and r.from_.attributes == [f[f"{k}_key"]]), None)
                setattr(fm, k, r.name if r is not None else f[f"{k}_key"])
        syn.fact = fm
    for g in T.get("group_effects") or []:
        by = g.get("by")
        syn.group_effects.append(GroupEffect(attribute=g.get("column"),
                                             by=[by] if isinstance(by, str) else list(by or [])))
    for la in T.get("link_affinity") or []:
        syn.link_affinity.append(LinkAffinity(match=list(la.get("match") or []),
                                              share=float(la["share"]) if la.get("share") is not None else None))
    um = T.get("unknown_members")
    if um is not None:
        syn.unknown_members = um if isinstance(um, list) else [um]
    g = T.get("generator")
    if isinstance(g, str):
        syn.generator = GeneratorChoice(name=g)
    elif isinstance(g, dict):
        syn.generator = GeneratorChoice(name=g.get("name"), version=g.get("version"),
                                        options=copy.deepcopy(g["options"]) if g.get("options") is not None else None)
    for k in ("sequence", "series"):
        if T.get(k) is not None:
            setattr(syn, k, copy.deepcopy(T[k]))
    if syn != EntitySynthesis():
        e.synthesis = syn
    return e


def _lift_column(cs) -> Attribute:
    if isinstance(cs, str):
        if cs.startswith("lookup:"):
            cs = {"type": "lookup", "lookup": cs[len("lookup:"):].strip()}
        else:
            cs = {"type": cs}
    if not isinstance(cs, dict):
        raise IRError(f"a column is a type name or a mapping, not {cs!r}")
    a = Attribute(kind=cs.get("type"), nullable=cs.get("nullable", True), description=cs.get("description"),
                  min=cs.get("min"), max=cs.get("max"), decimals=cs.get("decimals"), map=cs.get("map"),
                  default=cs.get("default"), base_column=cs.get("base_column"))
    lk = cs.get("lookup")
    a.lookup = KeyedLookup(table=lk.get("table"), match=lk.get("match"), value=lk.get("value")) \
        if isinstance(lk, dict) else lk
    if cs.get("within") is not None or cs.get("start") is not None:
        a.sequence = Sequence(within=cs.get("within"), start=cs.get("start"))
    syn = AttributeSynthesis(**{k: cs[k] for k in _ATTR_SYNTH if cs.get(k) is not None})
    if syn != AttributeSynthesis():
        a.synthesis = syn
    if a.kind == "pii":
        from .model import AttributePrivacy
        a.privacy = AttributePrivacy(role="direct_identifier")
    return a


def _lift_rule(r: dict) -> Rule:
    d = {_RULE_RENAME.get(k, k): v for k, v in r.items()}
    if d.get("type") == "soft":
        d["type"] = "derive"
    return Rule.from_dict(d, "rule")


# ---------------------------------------------------------------------------
# IR -> 1.0
# ---------------------------------------------------------------------------

def lower(est: Estate | dict, strict: bool = False) -> tuple[dict, list[str]]:
    """The 1.0 spec of an IR document and the list of what it could not carry.

    The list names, by IR path, every value that a 1.0 spec cannot hold: the
    engine of relsynth 1.0 ignores it (an annotation such as a privacy role or a
    partition spec) or cannot synthesise it (a shape other than record, a nested
    attribute).  With strict=True a non-empty list raises IRError.
    """
    doc: Estate = Estate.from_dict(est) if isinstance(est, dict) else est
    out, unsupported = _lower(doc)
    losses = list(unsupported)
    try:
        back = canonical(lift(out))
        losses += [p for p in diff(canonical(doc), back)
                   if not any(p.startswith(u.split(":")[0]) for u in unsupported)]
    except IRError as e:
        if not unsupported:
            losses += [f"(the 1.0 spec does not convert back: {x})" for x in e.errors]
    if strict and losses:
        raise IRError(["not representable in a 1.0 spec: " + x for x in losses])
    return out, losses


def lower_for_engine(est: Estate | dict) -> dict:
    """The 1.0 spec the engine runs.  Annotations it does not read are left out;
    structures it cannot synthesise raise IRError."""
    doc: Estate = Estate.from_dict(est) if isinstance(est, dict) else est
    out, unsupported = _lower(doc)
    if unsupported:
        raise IRError(["not synthesised by this release (relational records only): " + u for u in unsupported])
    return out


def update(est: Estate | dict, v1: dict) -> Estate:
    """est with the changes of a 1.0 spec written from it (doctor --apply, tune).

    A three-way merge whose base is est's 1.0 projection, lift(lower(est)): a
    value the 1.0 spec changed, added or removed takes the 1.0 value; every other
    value keeps est's, so annotations a 1.0 spec cannot carry (privacy roles,
    shapes, extensions) survive the round trip."""
    doc: Estate = Estate.from_dict(est) if isinstance(est, dict) else est
    return _merge3(doc, lift(_lower(doc)[0]), lift(v1))


def _merge3(o, b, n):
    if n == b:
        return copy.deepcopy(o)
    if isinstance(n, Node) and type(o) is type(b) is type(n):
        return dataclasses.replace(n, **{f.name: _merge3(getattr(o, f.name), getattr(b, f.name), getattr(n, f.name))
                                         for f in dataclasses.fields(n)})
    if isinstance(o, dict) and isinstance(b, dict) and isinstance(n, dict):
        out = {}
        for k in list(o) + [k for k in n if k not in o]:
            if k in n:
                out[k] = _merge3(o[k], b[k], n[k]) if k in o and k in b else copy.deepcopy(n[k])
            elif k not in b:            # not in the 1.0 projection: the 1.0 spec could not remove it
                out[k] = copy.deepcopy(o[k])
        return out
    return copy.deepcopy(n)


def _lower(est: Estate) -> tuple[dict, list[str]]:
    unsupported: list[str] = []
    out: dict[str, Any] = {"version": 1}
    for k in ("name", "description", "source", "target"):
        if getattr(est, k) is not None:
            out[k] = copy.deepcopy(getattr(est, k))
    out["tables"] = {n: _lower_entity(n, e, est, unsupported) for n, e in est.entities.items()}
    rels = []
    for r in est.relationships:
        if r.cardinality == "many_to_many":
            unsupported.append(f"relationships.{r.name}: many_to_many through {r.via.entity if r.via else '?'} "
                               "(1.0 keeps its two references only)")
            continue
        rels.append(_lower_rel(r, est))
    if rels:
        out["relationships"] = rels
    if est.rules:
        out["rules"] = [_lower_rule(r) for r in est.rules]
    if est.synthesis:
        out["synthesis"] = copy.deepcopy(est.synthesis)
    if est.review:
        out["review"] = list(est.review)
    return out, unsupported


def _lower_rel(r: Relationship, est: Estate) -> dict:
    src, dst = ends(r)
    fa = list(src.attributes or [])
    ta = dst.attributes
    d: dict[str, Any] = {"child": src.entity}
    parent = est.entities.get(dst.entity) if dst.entity is not None else None
    ppk = parent.primary_key.attributes if parent is not None and parent.primary_key else None
    if len(fa) == 1 and (ta is None or (ppk is not None and list(ta) == list(ppk)) or (ppk is None and len(ta) == 1)):
        d["child_key"] = fa[0]
        if ta is not None and len(ta) == 1:
            d["parent_key"] = ta[0]
    else:
        d["child_key"] = fa
        d["parent_key"] = list(ta) if ta is not None else fa
    d["parent"] = dst.entity
    d["kind"] = _REL_KIND_BACK.get(r.kind, r.kind)
    d["name"] = r.name
    if r.offset is not None:
        d["offset_from"] = r.offset.from_
        d["order_by"] = r.offset.order_by
    return d


def _lower_entity(name: str, e: Entity, est: Estate, unsupported: list) -> dict:
    where = f"entities.{name}"
    if e.shape != "record":
        unsupported.append(f"{where}.shape: {e.shape} (relsynth 1.0 synthesises records only)")
    for k in ("variants", "edge", "stream"):
        if getattr(e, k) is not None:
            unsupported.append(f"{where}.{k}: no 1.0 equivalent")
    T: dict[str, Any] = {}
    if e.role != "entity":
        T["role"] = e.role
    if e.description is not None:
        T["description"] = e.description
    s = e.synthesis or EntitySynthesis()
    if s.mode is not None:
        T["mode"] = s.mode
    pk = e.primary_key
    if pk is not None:
        if len(pk.attributes) != 1:
            unsupported.append(f"{where}.keys: composite primary key {pk.attributes} (1.0: one column; "
                               "declare it as a unique key)")
            T["unique"] = [list(pk.attributes)]
        else:
            T["primary_key"] = pk.attributes[0]
    T["columns"] = {c: _lower_attr(a, f"{where}.attributes.{c}", unsupported) for c, a in e.attributes.items()}
    uniq = [list(k.attributes) for k in e.keys if not k.primary and k.where is None]
    part = [{"columns": list(k.attributes), "where": k.where} for k in e.keys if not k.primary and k.where]
    if uniq:
        T["unique"] = T.get("unique", []) + uniq
    if part:
        T["partial_unique"] = part
    fact: dict[str, Any] | None = None
    if s.fact is not None:
        fact = {k: getattr(s.fact, k) for k in _FACT_KEYS if getattr(s.fact, k) is not None}
        if s.fact.header is not None:
            fact["header"] = s.fact.header.to_dict()
    tm = e.time
    if tm is not None and tm.event_time is not None:
        fact = fact if fact is not None else {}
        fact["time_column"] = tm.event_time
    if fact is not None:
        T["fact"] = fact
    if e.dependencies:
        T["dimensions"] = [{"key": d.determinant, "columns": list(d.dependents),
                            **({"anchor": d.anchor} if d.anchor else {})} for d in e.dependencies]
    if s.key_order is not None:
        T["key_order"] = s.key_order
    if e.view is not None:
        T["view_of"] = e.view.of
    if s.group_effects:
        T["group_effects"] = [{"column": g.attribute, "by": list(g.by)} for g in s.group_effects]
    if s.link_affinity:
        T["link_affinity"] = [{"match": list(la.match), **({"share": la.share} if la.share is not None else {})}
                              for la in s.link_affinity]
    if tm is not None and tm.versioning is not None:
        v = tm.versioning
        sc: dict[str, Any] = {}
        if v.scd_type is not None:
            sc["type"] = v.scd_type
        if v.natural_key is not None:
            sc["natural_key"] = v.natural_key
        if tm.valid_time is not None and tm.valid_time.from_ is not None:
            sc["valid_from"] = tm.valid_time.from_
        if tm.valid_time is not None and tm.valid_time.to is not None:
            sc["valid_to"] = tm.valid_time.to
        if v.stable is not None:
            sc["stable"] = v.stable
        if v.current is not None:
            sc["current"] = v.current
        T["scd"] = sc
    if s.unknown_members is not None:
        T["unknown_members"] = list(s.unknown_members)
    if s.generator is not None:
        g = s.generator
        # a bare name stays a name: the form a 1.0 spec writes most often
        T["generator"] = g.name if g.version is None and g.options is None else {
            "name": g.name, **({"version": g.version} if g.version is not None else {}),
            **({"options": copy.deepcopy(g.options)} if g.options is not None else {})}
    for k in ("sequence", "series"):
        if getattr(s, k) is not None:
            T[k] = copy.deepcopy(getattr(s, k))
    if e.allocation is not None:
        T["bridge"] = {k: getattr(e.allocation, k) for k in ("group", "weight", "total", "decimals")
                       if getattr(e.allocation, k) is not None}
    return T


def _lower_attr(a: Attribute, where: str, unsupported: list):
    kind = a.effective_kind
    if a.type in ("object", "array", "map") or a.fields or a.items is not None or a.variants is not None:
        unsupported.append(f"{where}: nested {a.type or 'object'} (relsynth 1.0 has flat columns)")
    elif kind is None:
        unsupported.append(f"{where}: no 1.0 column type for {a.type!r}")
    c = {"type": kind}
    if not a.nullable:
        c["nullable"] = False
    for k in ("min", "max", "decimals", "map", "default", "base_column", "description"):
        if getattr(a, k) is not None:
            c[k] = copy.deepcopy(getattr(a, k))
    if a.lookup is not None:
        c["lookup"] = a.lookup.to_dict() if isinstance(a.lookup, KeyedLookup) else a.lookup
    if a.sequence is not None:
        c["within"] = a.sequence.within
        if a.sequence.start is not None:
            c["start"] = a.sequence.start
    if a.synthesis is not None:
        for k in _ATTR_SYNTH:
            if getattr(a.synthesis, k) is not None:
                c[k] = copy.deepcopy(getattr(a.synthesis, k))
    if list(c) == ["type"]:
        return kind
    return c


def _lower_rule(r: Rule) -> dict:
    d = r.to_dict()
    d = {_RULE_BACK.get(k, k): v for k, v in d.items() if not k.startswith("x-")}
    d.pop("name", None)
    return d


# ---------------------------------------------------------------------------
# canonical form and differences
# ---------------------------------------------------------------------------

def canonical(est: Estate) -> Estate:
    """The canonical form of a document: values that follow from others are made
    explicit (an attribute's kind from its type, a reference's target attributes
    from the parent's primary key) and redundant ones dropped (a type that its
    kind implies), so that two documents that say the same are equal."""
    est = copy.deepcopy(est)

    def attr(a: Attribute):
        if a.kind is None and a.type in TYPE_KIND:
            a.kind = TYPE_KIND[a.type]
        if a.type is not None and a.kind is not None and KIND_TYPE.get(a.kind) == a.type:
            a.type = None
        if a.kind == "pii" and a.privacy is None:
            from .model import AttributePrivacy
            a.privacy = AttributePrivacy(role="direct_identifier")
        for f in a.fields.values():
            attr(f)
        if a.items is not None:
            attr(a.items)
        if a.variants is not None:
            for case in a.variants.cases.values():
                for f in case.values():
                    attr(f)
    for e in est.entities.values():
        for a in e.attributes.values():
            attr(a)
        if e.variants is not None:
            for case in e.variants.cases.values():
                for f in case.values():
                    attr(f)
    for r in est.relationships:
        p = est.entities.get(r.to.entity) if r.to and r.to.entity is not None else None
        if r.to is not None and r.to.attributes is None and p is not None and p.primary_key is not None:
            r.to.attributes = list(p.primary_key.attributes)
    return est


def diff(a, b, path: str = "") -> list[str]:
    """IR paths where two documents (or nodes) differ."""
    da = a.to_dict() if hasattr(a, "to_dict") else a
    db = b.to_dict() if hasattr(b, "to_dict") else b
    out: list[str] = []
    _diff(da, db, path, out)
    return out


def _diff(x, y, path, out):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in list(x) + [k for k in y if k not in x]:
            p = f"{path}.{k}" if path else str(k)
            if k not in y:
                out.append(f"{p}: {_short(x[k])} (dropped)")
            elif k not in x:
                out.append(f"{p}: {_short(y[k])} (added)")
            else:
                _diff(x[k], y[k], p, out)
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            _diff(u, v, f"{path}[{i}]", out)
    elif x != y:
        out.append(f"{path}: {_short(x)} -> {_short(y)}")


def _short(v, n=80):
    s = repr(v)
    return s if len(s) <= n else s[:n - 3] + "..."
