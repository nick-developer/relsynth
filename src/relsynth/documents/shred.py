# SPDX-License-Identifier: Apache-2.0
"""Shredding: document collections as relational tables, and back.

Every collection becomes one *root table*; every array (and every map, an
object keyed by data) becomes a *child table* whose rows are the elements, with
a foreign key to the row that holds the array (``_parent``) and, for a map,
the entry's key (``_key``), and its position ``_pos`` (0, 1, ... in the
source).  The engine models ``_pos`` as an ordinary integer attribute of the
element, jointly with its content (the first item of an order is more often
the expensive one); a synthetic array is ordered by its elements' ``_pos``,
ties broken by row id.  Row ids are given in document order, so the round trip
keeps the source order exactly.
Nested arrays give grandchild tables, so the tables form a tree per collection
(the classic "shredding" of XML and JSON into relations: Shanmugasundaram et
al., VLDB 1999; Florescu and Kossmann, IEEE Data Eng. Bull. 1999; Chasseur et
al., Argo, WebDB 2013).  Inside one table the nested objects are flattened:
the field ``customer.age`` is the column ``customer__age``.

A path whose state varies gets a *tag* column ``<path>__t`` with the values
``absent``, ``null`` or a type name, so that missing and null stay apart and a
field with several types (a union) keeps its type mix; each type of a union
has its own value column ``<path>__<type>``.  A field equal to the length of a
sibling array is not stored: it is recomputed from the array when the
documents are rebuilt.  A reference to another collection is a foreign key to
that collection's unique field, so the engine keeps it resolving.

`Layout` is the mapping; `shred` applies it to documents and `nest` inverts it
(Proposition 1401: `nest(shred(D)) == D` for every source document set D, up
to the order of documents and the fresh identifiers of rows).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import pandas as pd

from .infer import FULL_PRECISION, CollectionProfile, Node
from .values import NUMERIC, ObjectId, from_cell, object_id, to_cell, type_name

SEP = "__"
ROW_ID = "_rid"
PARENT = "_parent"
POS = "_pos"
KEY = "_key"


@dataclass
class Column:
    name: str
    path: list[str]            # keys from the table's row context ([] = the element itself)
    role: str                  # id, parent, pos, key, value, typed, tag, ref
    type: str | None = None    # the document type of the value (value, typed, ref)
    decimals: int | None = None
    ref: dict | None = None    # {table, column} for role ref
    fmt: str | None = None     # strings that are all dates in one format (written back in it)


@dataclass
class TableLayout:
    name: str
    collection: str
    kind: str                         # root, array, map, object (an optional sub-document)
    context: str                      # the path whose objects (or elements) are the rows
    parent: str | None = None
    holder: list[str] = field(default_factory=list)   # path of the array/map in the parent row
    columns: list[Column] = field(default_factory=list)
    id_column: str = ROW_ID           # _rid, _id, or a unique top-level field another collection references
    id_type: str = "int"
    field_order: list[str] = field(default_factory=list)   # root: the source's top-level field order
    fresh: dict = field(default_factory=dict)  # root fields that are fresh identifiers (path -> type)
    key_mode: str | None = None       # map keys: categorical, field (a field of the value) or generated
    key_format: dict | None = None

    def column(self, name) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)


@dataclass
class Layout:
    tables: dict[str, TableLayout] = field(default_factory=dict)
    # per table, path (tuple joined with '.') -> shape info used when rebuilding
    shapes: dict[str, dict] = field(default_factory=dict)
    references: list[dict] = field(default_factory=list)
    order: list[str] = field(default_factory=list)       # collections in input order

    def to_dict(self) -> dict:
        return {"tables": {k: asdict(v) for k, v in self.tables.items()}, "shapes": self.shapes,
                "references": self.references, "order": self.order}

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        tabs = {}
        for k, v in d["tables"].items():
            v = dict(v)
            v["columns"] = [Column(**c) for c in v["columns"]]
            tabs[k] = TableLayout(**v)
        return cls(tabs, d["shapes"], d.get("references", []), d.get("order", []))

    def children(self, table: str) -> list[TableLayout]:
        return [t for t in self.tables.values() if t.parent == table]

    def roots(self) -> list[TableLayout]:
        return [self.tables[c] for c in self.order]


_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                 "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M")


def _date_format(f: Node) -> str | None:
    """The format of a string field whose every (kept) value is a date in it."""
    import datetime as dt
    if f.overflow or not f.distinct or f.scalar_types != ["string"]:
        return None
    vals = [v for v in list(f.distinct)[:200] if isinstance(v, str)]
    for fmt in _DATE_FORMATS:
        try:
            if all(dt.datetime.strptime(v, fmt).strftime(fmt) == v for v in vals):
                return fmt
        except ValueError:
            continue
    return None


def _decimals(f: Node) -> int | None:
    return f.decimals if 0 < f.decimals < FULL_PRECISION else (0 if f.decimals == 0 else None)


def _col(path: list[str]) -> str:
    return SEP.join(p.replace(".", "_") for p in path)


def _states(nd: Node) -> list[str]:
    """The states a path takes: absent, null, then its value types."""
    st = (["absent"] if nd.optional else []) + (["null"] if nd.nullable else [])
    return st + nd.value_types


def _needs_tag(nd: Node) -> bool:
    vt = nd.value_types
    if len(vt) == 1 and vt[0] not in ("object", "array"):
        return nd.optional and nd.nullable          # absent and null both occur: NULL alone cannot tell
    return len(_states(nd)) > 1


def build_layout(profiles: dict[str, CollectionProfile], references: list[dict] | None = None) -> Layout:
    """The tables of a set of collections (references: the taken ones from
    `infer.find_references` and the declared ones)."""
    lay = Layout(order=list(profiles))
    refs = [r for r in (references or []) if not r.get("candidate")]
    lay.references = refs
    targets = {(r["to"], r["to_path"]) for r in refs}
    for cname, prof in profiles.items():
        root = prof.root
        tl = TableLayout(cname, cname, "root", "", field_order=list(root.fields))

        def unique(nd):
            return (nd is not None and not nd.optional and not nd.nullable and len(nd.scalar_types) == 1
                    and not nd.overflow and len(nd.distinct) == prof.documents)

        idn = root.fields.get("_id")
        natural = next((r["to_path"] for r in refs if r["to"] == cname and r["to_path"] != "_id"
                        and "." not in r["to_path"] and unique(root.fields.get(r["to_path"]))), None)
        if natural is not None:
            # referenced by a natural key: that key is the row id, so the engine
            # draws the references among the synthetic keys
            tl.id_column, tl.id_type = natural, root.fields[natural].scalar_types[0]
            if unique(idn):
                tl.fresh["_id"] = idn.scalar_types[0]
        elif unique(idn):
            tl.id_column, tl.id_type = "_id", idn.scalar_types[0]
        else:
            tl.columns.append(Column(ROW_ID, [], "id", "int"))
        lay.tables[cname] = tl
        _walk_object(lay, prof, tl, root, [], refs, targets)
    return lay


def _ref_of(refs, cname, path):
    return next((r for r in refs if r["collection"] == cname and r["path"] == path), None)


def _walk_object(lay: Layout, prof: CollectionProfile, tl: TableLayout, nd: Node, prefix: list[str], refs, targets):
    """Columns of the object `nd` (fields under `prefix` of the row context)."""
    shapes = lay.shapes.setdefault(tl.name, {})
    for name, f in nd.fields.items():
        if not prefix and tl.kind == "root" and (name == tl.id_column or name in tl.fresh):
            continue
        path = prefix + [name]
        _walk_value(lay, prof, tl, f, path, refs, targets)
    if nd.variants:
        shapes.setdefault(".".join(prefix), {})["variants"] = nd.variants


def _abs_path(tl: TableLayout, path: list[str]) -> str:
    rel = ".".join(path)
    if not tl.context:
        return rel
    return tl.context + ("." + rel if rel else "")


def _walk_value(lay, prof, tl, f: Node, path: list[str], refs, targets):
    shapes = lay.shapes.setdefault(tl.name, {})
    key = ".".join(path)
    info = {"types": f.value_types, "optional": f.optional, "nullable": f.nullable}
    if f.length_of:
        info["length_of"] = f.length_of
        info["type"] = f.value_types[0]
        shapes[key] = info
        return
    cname = _col(path) if path else "value"
    if _needs_tag(f):
        tag = (cname + SEP + "t") if path else "value__t"
        tl.columns.append(Column(tag, path, "tag"))
        info["tag"] = tag
    scal = f.scalar_types
    if f.is_object and not f.is_map and f.value_types == ["object"] and (f.optional or f.nullable) and f.fields:
        # an optional sub-document: a table of 0 or 1 row per holder, so the
        # presence of its fields is modelled given that it is there
        info["object"] = True
        info["object_table"] = _child_table(lay, prof, tl, f, path, "object", refs, targets)
    elif f.is_object and not f.is_map:
        info["object"] = True
        shapes[key] = info
        _walk_object(lay, prof, tl, f, path, refs, targets)
    elif f.is_object and f.is_map:
        info["object"] = True
        info["map"] = _child_table(lay, prof, tl, f, path, "map", refs, targets)
        shapes[key] = info
    if f.is_array:
        info["array"] = _child_table(lay, prof, tl, f, path, "array", refs, targets)
        shapes[key] = info
    if scal:
        ref = _ref_of(refs, tl.collection, _abs_path(tl, path)) if len(scal) == 1 else None
        if len(scal) == 1 and len(f.value_types) == 1:
            c = Column(cname, path, "ref" if ref else "value", scal[0], _decimals(f))
            if scal[0] == "string":
                c.fmt = _date_format(f)
            if ref:
                c.ref = {"table": ref["to"], "path": ref["to_path"]}
            tl.columns.append(c)
            info["column"] = cname
        else:
            # a union: one column per type, the numeric types sharing one column
            # (the tag says int or double, the value is modelled once)
            info["typed"] = {}
            nums = [t for t in scal if t in NUMERIC]
            for t in scal:
                if t in nums and len(nums) > 1:
                    if t != nums[0]:
                        info["typed"][t] = cname + SEP + "num"
                        continue
                    c = Column(cname + SEP + "num", path, "typed", "num", _decimals(f))
                else:
                    c = Column(cname + SEP + t, path, "typed", t, _decimals(f))
                if ref and t == scal[0]:
                    c.role, c.ref = "ref", {"table": ref["to"], "path": ref["to_path"]}
                tl.columns.append(c)
                info["typed"][t] = c.name
    shapes[key] = info


def _child_table(lay, prof, tl, f: Node, path, kind, refs, targets) -> str:
    name = tl.name + SEP + (_col(path) if path else "items")
    ctx = _abs_path(tl, path) + {"map": "{}", "array": "[]", "object": ""}[kind]
    ct = TableLayout(name, tl.collection, kind, ctx, parent=tl.name, holder=list(path))
    ct.columns.append(Column(ROW_ID, [], "id", "int"))
    ct.columns.append(Column(PARENT, [], "parent", None))
    if kind != "object":
        ct.columns.append(Column(POS, [], "pos", "int"))
    if kind == "map":
        entries = sum(k * c for k, c in f.lengths.items())
        sample = list(f.map_keys)
        if f.key_field:
            ct.key_mode = "field"
            ct.key_format = {"field": f.key_field}
        elif sample and len(sample) < 1000 and len(sample) <= 0.5 * max(entries, 1):
            ct.key_mode = "categorical"
            ct.columns.append(Column(KEY, [], "key", "string"))
        else:
            ct.key_mode = "generated"
            ct.key_format = _key_format(sample[:50])
    lay.tables[name] = ct
    if kind == "object":
        lay.shapes.setdefault(name, {}).setdefault("", {})["object"] = True
        _walk_object(lay, prof, ct, f, [], refs, targets)
        return name
    it = f.items
    if it is not None:
        _walk_value(lay, prof, ct, it, [], refs, targets)
    return name


def _key_format(sample: list[str]) -> dict:
    import re
    if sample and all(re.fullmatch(r"[0-9a-f]+", k) for k in sample):
        return {"kind": "hex", "length": max(len(k) for k in sample)}
    if sample and all(re.fullmatch(r"\d+", k) for k in sample):
        return {"kind": "digits", "length": max(len(k) for k in sample)}
    return {"kind": "hex", "length": 24}


# ---------------------------------------------------------------- shred
class _Rows:
    def __init__(self):
        self.rows: dict[str, list[dict]] = {}
        self.next_id: dict[str, int] = {}

    def add(self, table, row):
        self.rows.setdefault(table, []).append(row)

    def new_id(self, table):
        i = self.next_id.get(table, 0)
        self.next_id[table] = i + 1
        return i


def shred(lay: Layout, collections: dict[str, Iterable[dict]]) -> dict[str, pd.DataFrame]:
    """The tables of the documents (one DataFrame per table, columns in layout order)."""
    out = _Rows()
    keymaps: dict[tuple, dict] = {}
    for cname in lay.order:
        tl = lay.tables[cname]
        for doc in collections[cname]:
            row = {}
            rid = doc.get(tl.id_column) if tl.id_column != ROW_ID else out.new_id(cname)
            row[tl.id_column] = to_cell(rid, tl.id_type)
            _fill(lay, tl, row, doc, [], out, rid if tl.id_column == ROW_ID else row[tl.id_column])
            out.add(cname, row)
    frames = {}
    for name, tl in lay.tables.items():
        cols = [tl.id_column] if tl.id_column != ROW_ID else []
        cols += [c.name for c in tl.columns]
        frames[name] = pd.DataFrame(out.rows.get(name, []), columns=cols)
    del keymaps
    return frames


def _get(d, path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return _ABSENT
        d = d[p]
    return d


class _Absent:
    def __repr__(self):
        return "<absent>"


_ABSENT = _Absent()


def _state(v) -> str:
    return "absent" if v is _ABSENT else type_name(v)


def _fill(lay, tl: TableLayout, row: dict, ctx_value, prefix, out: _Rows, row_id):
    """Cells of one row: ctx_value is the row's object (or element)."""
    shapes = lay.shapes.get(tl.name, {})
    for c in tl.columns:
        if c.role in ("id", "parent", "pos", "key"):
            continue
        v = _get(ctx_value, c.path)
        st = _state(v)
        if c.role == "tag":
            row[c.name] = st
        elif c.role in ("value", "ref"):
            row[c.name] = to_cell(v, c.type) if st == c.type else None
        elif c.role == "typed":
            if c.type == "num":
                row[c.name] = to_cell(v, "double") if st in NUMERIC else None
            else:
                row[c.name] = to_cell(v, c.type) if st == c.type else None
    for key, info in shapes.items():
        if "object_table" in info:
            v = _get(ctx_value, key.split(".") if key else [])
            if isinstance(v, dict):
                _child_row(lay, lay.tables[info["object_table"]], v, out, row_id)
        for kind in ("array", "map"):
            if kind not in info:
                continue
            v = _get(ctx_value, key.split(".") if key else [])
            child = lay.tables[info[kind]]
            if kind == "array" and isinstance(v, list):
                for i, x in enumerate(v):
                    _child_row(lay, child, x, out, row_id, pos=i)
            elif kind == "map" and isinstance(v, dict):
                for i, (k, x) in enumerate(v.items()):
                    _child_row(lay, child, x, out, row_id, pos=i, key=k)


def _child_row(lay, ct: TableLayout, x, out: _Rows, parent_id, pos=None, key=None):
    rid = out.new_id(ct.name)
    row = {ROW_ID: rid, PARENT: parent_id}
    if ct.kind != "object":
        row[POS] = pos
    if ct.key_mode == "categorical":
        row[KEY] = key
    _fill(lay, ct, row, x, [], out, rid)
    out.add(ct.name, row)


# ---------------------------------------------------------------- nest
def nest(lay: Layout, frames: dict[str, pd.DataFrame], *, seed: Any = 0, keep_ids: bool = False
         ) -> dict[str, list[dict]]:
    """Documents of the tables: the inverse of `shred`.  Identifiers of root rows
    that were ObjectIds become fresh ObjectIds that are a function of (seed,
    collection, row id), and every reference to them is mapped the same way
    (`keep_ids` keeps the cells as they are: the round trip of source tables)."""
    idmap: dict[str, dict] = {}
    for cname in lay.order:
        tl = lay.tables[cname]
        df = frames[cname]
        if tl.id_column != ROW_ID and tl.id_type == "objectId" and not keep_ids:
            idmap[cname] = {k: object_id(seed, cname, k) for k in df[tl.id_column].tolist()}
    # elements of child tables, bottom-up: table -> parent id -> [(order, value)]
    built: dict[str, dict] = {}
    for name in _bottom_up(lay):
        tl = lay.tables[name]
        if tl.kind == "root":
            continue
        df = frames[name]
        recs = df.to_dict("records")
        groups: dict[Any, list] = {}
        for r in recs:
            val = _build(lay, tl, r, built, idmap, frames, seed)
            order = r.get(POS)
            groups.setdefault(r[PARENT], []).append((order, r[ROW_ID], val))
        for g in groups.values():
            g.sort(key=lambda t: (_sort_key(t[0]), _sort_key(t[1])))
        built[name] = groups
    docs = {}
    for cname in lay.order:
        tl = lay.tables[cname]
        out = []
        for r in frames[cname].to_dict("records"):
            d = _build(lay, tl, r, built, idmap, frames, seed)
            if tl.id_column != ROW_ID:
                rid = r[tl.id_column]
                d[tl.id_column] = idmap[cname][rid] if cname in idmap else from_cell(rid, tl.id_type)
                for fname, ftype in tl.fresh.items():
                    d[fname] = (object_id(seed, cname, fname, rid) if ftype == "objectId"
                                else from_cell(str(object_id(seed, cname, fname, rid)), ftype)
                                if ftype == "string" else from_cell(len(out), ftype))
                order = {k: i for i, k in enumerate(tl.field_order)}
                d = dict(sorted(d.items(), key=lambda kv: order.get(kv[0], len(order))))
            out.append(d)
        docs[cname] = out
    return docs


def _sort_key(v):
    if v is None or (isinstance(v, float) and v != v):
        return (1, "")
    return (0, float(v)) if isinstance(v, (int, float)) else (0, str(v))


def _bottom_up(lay: Layout) -> list[str]:
    depth = {}
    for name, tl in lay.tables.items():
        d, p = 0, tl
        while p.parent:
            d += 1
            p = lay.tables[p.parent]
        depth[name] = d
    return sorted(lay.tables, key=lambda n: -depth[n])


def _build(lay, tl: TableLayout, r: dict, built, idmap, frames, seed):
    """The value of one row: its object (root, object elements) or its element."""
    shapes = lay.shapes.get(tl.name, {})
    rid = r[tl.id_column] if tl.id_column in r else r.get(ROW_ID)
    cols = {c.name: c for c in tl.columns}

    def value_at(path: list[str]):
        key = ".".join(path)
        info = shapes.get(key)
        if info is None:
            return _ABSENT
        if "object_table" in info:
            rows = built.get(info["object_table"], {}).get(rid)
            if rows:
                return rows[0][2]
            st = r.get(info["tag"]) if "tag" in info else None
            return None if st == "null" or (not info["optional"] and info["nullable"]) else _ABSENT
        state = None
        if "tag" in info:
            state = r.get(info["tag"])
            if state is None or (isinstance(state, float) and state != state):
                state = "absent" if info["optional"] else (info["types"][0] if info["types"] else "null")
            if state == "absent":
                return _ABSENT
            if state == "null":
                return None
        if "length_of" in info:
            return ("__length_of__", info["length_of"], info["type"])
        if state is None:
            state = info["types"][0] if info["types"] else "null"
        if state == "object" and "map" in info:
            return _map_value(lay, info["map"], built, rid, seed)
        if state == "object":
            return obj_at(path)
        if state == "array":
            items = built.get(info["array"], {}).get(rid, [])
            return [v for _, _, v in items]
        if "column" in info:
            c = cols[info["column"]]
            v = r.get(c.name)
            if _isnull(v):
                return _ABSENT if (info["optional"] and not info["nullable"]) else None
            return _restore(c, v, idmap, seed)
        if "typed" in info and state in info["typed"]:
            c = cols[info["typed"][state]]
            v = r.get(c.name)
            if _isnull(v):
                # the type's column is empty in this synthetic row: cast the
                # value another type's column holds, when it converts
                for other in dict.fromkeys(info["typed"].values()):
                    w = r.get(other)
                    if other != c.name and not _isnull(w):
                        try:
                            return from_cell(w, state, c.decimals)
                        except (TypeError, ValueError, ArithmeticError):
                            continue
                return None
            if c.type == "num":
                return from_cell(v, state, c.decimals if state in ("double", "decimal") else None)
            return _restore(c, v, idmap, seed)
        return None

    def obj_at(path: list[str]):
        prefix = ".".join(path)
        names = [k for k in shapes if k and k != prefix and _parent_of(k) == prefix]
        var = shapes.get(prefix, {}).get("variants")
        keep = None
        if var:
            disc = value_at(path + [var["discriminator"]])
            keep = set(var["cases"].get(disc, []))
        o = {}
        for k in names:
            fname = k[len(prefix) + 1:] if prefix else k
            if var and fname in var["fields"] and fname not in keep:
                continue
            v = value_at(path + [fname])
            if v is _ABSENT:
                continue
            o[fname] = v
        for k, v in list(o.items()):
            if isinstance(v, tuple) and v and v[0] == "__length_of__":
                arr = o.get(v[1])
                o[k] = from_cell(len(arr) if isinstance(arr, list) else 0, v[2])
        return o

    top = shapes.get("")
    if tl.kind != "root" and top is not None and "object" not in top:
        v = value_at([])
        return None if v is _ABSENT else v
    return obj_at([])


def _parent_of(k: str) -> str:
    return k.rsplit(".", 1)[0] if "." in k else ""


def _isnull(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _restore(c: Column, v, idmap, seed):
    if c.role == "ref" and c.ref and c.ref["table"] in idmap:
        m = idmap[c.ref["table"]]
        key = v
        if key not in m:
            key = next((k for k in m if str(k) == str(v)), None)
        return m.get(key, ObjectId(str(v)))
    if c.fmt and c.type == "string":
        try:
            return pd.Timestamp(v).strftime(c.fmt)
        except (TypeError, ValueError):
            return str(v)
    return from_cell(v, c.type, c.decimals)


def _map_value(lay, table, built, rid, seed):
    tl = lay.tables[table]
    items = built.get(table, {}).get(rid, [])
    o = {}
    for k, row_id, v in items:
        if tl.key_mode == "generated":
            k = _gen_key(tl.key_format, seed, table, row_id)
        elif tl.key_mode == "field":
            k = v.get(tl.key_format["field"]) if isinstance(v, dict) else None
            if k is None:
                k = _gen_key({"kind": "hex", "length": 24}, seed, table, row_id)
        if k in o:
            continue
        o[str(k)] = v
    return o


def _gen_key(fmt, seed, table, row_id) -> str:
    h = str(object_id(seed, table, "key", row_id)) * 3
    if fmt and fmt.get("kind") == "digits":
        return str(int(h, 16))[: fmt["length"]]
    return h[: (fmt or {}).get("length", 24)]
