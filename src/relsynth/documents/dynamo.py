# SPDX-License-Identifier: Apache-2.0
"""DynamoDB single-table designs: item types as logical collections, and back.

In a single-table design (AWS, "Best practices for DynamoDB table design";
Rick Houlihan, re:Invent 2018 DAT401; Alex DeBrie, "The DynamoDB Book") several
entity types share one table.  An item's partition key and sort key (and the
keys of the global secondary indexes) are strings built from the entity's own
attributes, for example ``PK = CUSTOMER#{customer_id}``, ``SK =
ORDER#{ordered_at}#{order_id}``, and a type attribute (``entity_type``) or the
key prefix tells the types apart.

`split` finds the item types and, for every key attribute of every type, the
*key template* that rebuilds it from the other attributes: each ``#``-separated
part of the key is either the same literal on every item of the type or equal
to one attribute on every item.  The key attributes are then removed from the
logical documents (they carry no information beyond the template), the types
are synthesised as collections with references between them, and `merge`
rebuilds every key from its template, so the access patterns of the source
(items of a customer under one partition, an order's items under the GSI)
hold exactly in the synthetic table (Proposition 1404).  A key part that no
attribute explains stays an ordinary attribute and is modelled.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

_TYPE_ATTRS = ("entity_type", "entityType", "type", "_type", "__typename", "Type", "EntityType", "item_type")
SEP = "#"


def _type_attribute(items: list[dict], key_attrs: list[str]) -> str | None:
    for a in _TYPE_ATTRS:
        if a in key_attrs:
            continue
        vals = [it.get(a) for it in items]
        if all(isinstance(v, str) for v in vals) and 1 < len(set(vals)) <= 64:
            return a
    return None


def _prefix(v) -> str:
    return str(v).split(SEP, 1)[0] if v is not None else ""


def _template(items: list[dict], key: str, exclude: set[str]) -> str | None:
    """The template of one key attribute over the items of one type."""
    vals = [it.get(key) for it in items]
    if any(not isinstance(v, str) for v in vals):
        return None
    parts = [str(v).split(SEP) for v in vals]
    if len({len(p) for p in parts}) != 1:
        return None
    out = []
    attrs = [a for a in items[0] if a not in exclude and a != key]
    for i in range(len(parts[0])):
        col = [p[i] for p in parts]
        if len(set(col)) == 1:          # the same on every item: a literal (PROFILE, ORDER)
            out.append(col[0].replace("{", "{{").replace("}", "}}"))
            continue
        hit = next((a for a in attrs if all(_fmt(it.get(a)) == c for it, c in zip(items, col))), None)
        if hit is None:
            return None
        out.append("{" + hit + "}")
    return SEP.join(out)


def _fmt(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def split(items: list[dict], *, partition_key: str = "PK", sort_key: str | None = "SK",
          indexes: list[dict] | None = None, discriminator: str | None = None) -> tuple[dict, dict]:
    """(collections {type: [documents]}, design) of the items of one table."""
    key_attrs: list[Any] = [partition_key] + ([sort_key] if sort_key else [])
    for g in indexes or []:
        key_attrs += [g.get("HASH") or g.get("partition_key"), g.get("RANGE") or g.get("sort_key")]
    key_attrs = [k for k in dict.fromkeys(key_attrs) if k]
    disc = discriminator or _type_attribute(items, key_attrs)
    groups: dict[str, list[dict]] = {}
    for it in items:
        t = str(it.get(disc)) if disc else (_prefix(it.get(sort_key)) if sort_key else _prefix(it.get(partition_key)))
        groups.setdefault(t, []).append(it)
    design: dict[str, Any] = {"partition_key": partition_key, "sort_key": sort_key, "indexes": indexes or [],
              "discriminator": disc, "entities": {}}
    cols = {}
    for t, its in groups.items():
        present = [k for k in key_attrs if all(k in it for it in its)]
        partial = [k for k in key_attrs if k not in present and any(k in it for it in its)]
        templates = {}
        for k in present:
            tp = _template(its, k, set(key_attrs))
            if tp is not None:
                templates[k] = tp
        name = _entity_name(t)
        design["entities"][name] = {"value": t, "templates": templates, "partial_keys": partial,
                                    "items": len(its)}
        drop = set(templates) | ({disc} if disc else set())
        cols[name] = [{k: v for k, v in it.items() if k not in drop} for it in its]
    return cols, design


def _entity_name(t: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]+", "_", t).strip("_") or "items"
    return s[:1].upper() + s[1:].lower() if s.isupper() else s


def render(template: str, doc: dict) -> str:
    def rep(m):
        v = doc.get(m.group(1))
        return "" if v is None else _fmt(v)
    out = re.sub(r"(?<!\{)\{([^{}]+)\}(?!\})", rep, template)
    return out.replace("{{", "{").replace("}}", "}")


def merge(collections: dict[str, list[dict]], design: dict) -> list[dict]:
    """The items of the table: every key rebuilt from its template, the type
    attribute restored."""
    out = []
    disc = design.get("discriminator")
    for name, docs in collections.items():
        ent = design["entities"][name]
        for d in docs:
            it = {}
            for k, tp in ent["templates"].items():
                it[k] = render(tp, d)
            if disc:
                it[disc] = ent["value"]
            for k, v in d.items():
                if k == "_id":
                    continue
                it[k] = v
            out.append(it)
    return out


def check(items: list[dict], design: dict) -> dict:
    """Primary keys unique; every templated key equals its template on every item."""
    pk, sk = design["partition_key"], design["sort_key"]
    keys = Counter((it.get(pk), it.get(sk) if sk else None) for it in items)
    dup = sum(c - 1 for c in keys.values() if c > 1)
    bad = 0
    disc = design.get("discriminator")
    by_value = {e["value"]: e for e in design["entities"].values()}
    for it in items:
        ent = by_value.get(str(it.get(disc))) if disc else None
        if ent is None:
            continue
        for k, tp in ent["templates"].items():
            bad += it.get(k) != render(tp, it)
    return {"items": len(items), "duplicate_keys": dup, "template_violations": bad,
            "pass": dup == 0 and bad == 0}


def to_ir(design: dict, table: str):
    """The physical part of the IR of the table (container and key templates)."""
    cont = {"kind": "key_value_table", "system": "dynamodb", "name": table,
            "partition_key": design["partition_key"]}
    if design.get("sort_key"):
        cont["sort_key"] = design["sort_key"]
    idx = []
    for g in design.get("indexes") or []:
        e = {"name": g.get("name")}
        if g.get("HASH") or g.get("partition_key"):
            e["partition_key"] = g.get("HASH") or g.get("partition_key")
        if g.get("RANGE") or g.get("sort_key"):
            e["sort_key"] = g.get("RANGE") or g.get("sort_key")
        idx.append(e)
    if idx:
        cont["indexes"] = idx
    phys = {}
    for name, e in design["entities"].items():
        p: dict[str, Any] = {"container": table, "key_templates": dict(e["templates"])}
        if design.get("discriminator"):
            p["discriminator"] = {"attribute": design["discriminator"], "value": e["value"]}
        phys[name] = p
    return cont, phys
