# SPDX-License-Identifier: Apache-2.0
"""The shredded collections as a relational database the engine reads.

`write_sqlite` stores the tables of a `Layout` in a SQLite file with every key
declared (primary key per table, ``_parent`` and every reference as a foreign
key), so that `relsynth.introspect` drafts the 1.0 spec from declared keys
rather than guessing them.  `spec_for` then adds what only the layout knows:

* the tag columns (``<path>__t``) and map keys are categorical;
* an element table is a fact anchored on the table that holds the array, so
  the number of elements per document follows the source's length law, and a
  root table that is referenced is an entity;
* nothing of the document is copied as reference data unless the user says so
  (``mode: copy`` is reserved for small code lists, and a document's own
  arrays are never code lists).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .shred import KEY, PARENT, POS, ROW_ID, Layout, _col

_SQL_TYPE = {"num": "REAL", "int": "INTEGER", "long": "INTEGER", "double": "REAL", "decimal": "REAL", "bool": "INTEGER",
             "date": "DATETIME", "string": "TEXT", "objectId": "TEXT", "binary": "TEXT", None: "TEXT"}


def target_column(lay: Layout, table: str, path: str) -> str:
    tl = lay.tables[table]
    if path == tl.id_column:
        return path
    return _col(path.split("."))


def write_sqlite(lay: Layout, frames: dict[str, pd.DataFrame], path) -> Path:
    """Write the shredded tables to a new SQLite file with every key declared.

    Names come from the documents, so every identifier is quoted with `_q`; values
    are bound by pandas.  An existing file at `path` is replaced.
    """
    path = Path(path)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        for name in _parents_first(lay):
            tl = lay.tables[name]
            df = frames[name]
            cols, fks = [], []
            for c in df.columns:
                col = tl.column(c)
                if c == tl.id_column:
                    ty = _SQL_TYPE.get(tl.id_type, "TEXT")
                elif c == PARENT and tl.parent is not None:
                    p = lay.tables[tl.parent]
                    ty = _SQL_TYPE.get(p.id_type, "TEXT") if p.id_column != ROW_ID else "INTEGER"
                    fks.append(f"FOREIGN KEY ({_q(c)}) REFERENCES {_q(tl.parent)} ({_q(p.id_column)})")
                elif col is not None and col.role == "ref" and col.ref is not None:
                    ty = _SQL_TYPE.get(col.type, "TEXT")
                    tcol = target_column(lay, col.ref["table"], col.ref["path"])
                    fks.append(f"FOREIGN KEY ({_q(c)}) REFERENCES {_q(col.ref['table'])} ({_q(tcol)})")
                elif col is not None:
                    ty = _SQL_TYPE.get(col.type, "TEXT") if col.role not in ("tag", "key") else "TEXT"
                else:
                    ty = "INTEGER"
                cols.append(f"{_q(c)} {ty}" + (" PRIMARY KEY" if c == tl.id_column else ""))
            uniq = [f"UNIQUE ({_q(target_column(lay, name, r['to_path']))})" for r in lay.references
                    if r["to"] == name and target_column(lay, name, r["to_path"]) != tl.id_column]
            # every name is quoted by _q: field paths come from the documents, so a name can hold a quote
            con.execute(f"CREATE TABLE {_q(name)} ({', '.join(cols + sorted(set(uniq)) + fks)})")  # nosemgrep
            out = df.copy()
            for c in out.columns:
                if pd.api.types.is_bool_dtype(out[c]) or out[c].map(lambda v: isinstance(v, bool)).any():
                    out[c] = out[c].map(lambda v: None if v is None else int(v))
                elif pd.api.types.is_datetime64_any_dtype(out[c]):
                    out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            out.to_sql(name, con, if_exists="append", index=False)
        con.commit()
    finally:
        con.close()
    return path


def _q(name) -> str:
    """A quoted SQLite identifier (embedded double quotes doubled)."""
    return '"' + str(name).replace('"', '""') + '"'


def _parents_first(lay: Layout) -> list[str]:
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        tl = lay.tables[n]
        if tl.parent:
            visit(tl.parent)
        for c in tl.columns:
            if c.role == "ref" and c.ref["table"] != n:
                visit(c.ref["table"])
        order.append(n)

    for n in lay.tables:
        visit(n)
    return order


def _structural(item) -> bool:
    """A drafted group effect or affinity that involves the row bookkeeping columns."""
    txt = repr(item)
    return any(c in txt for c in (ROW_ID, KEY + "'"))


def spec_for(lay: Layout, db_path, *, name: str = "documents", seed: int = 7, sample_rows: int = 20000,
             sequences: list[dict] | None = None) -> dict:
    """The 1.0 spec of the shredded database (introspected, then completed from
    the layout)."""
    from ..api import introspect
    raw = introspect({"kind": "sqlite", "path": str(db_path)}, name=name, sample_rows=sample_rows).raw
    tables = raw["tables"]
    for tname, tl in lay.tables.items():
        t = tables[tname]
        cols = t.setdefault("columns", {})
        if t.get("mode") == "copy" and tl.kind != "root":
            t.pop("mode", None)
        if tl.kind != "root" and t.get("mode") == "copy":
            t.pop("mode")
        if POS in cols:
            cols[POS] = {"type": "integer", "min": 0}
        for cn, cd in list(cols.items()):
            # a drafted ratio or offset to a holder's column (``_parent.x``) is a
            # coincidence of sparse fields more often than a business rule
            for k in ("ratio_of", "offset_from"):
                if isinstance(cd, dict) and "." in str(cd.get(k, "")):
                    cols[cn] = {kk: vv for kk, vv in cd.items() if kk not in (k, "ratio_same_as")}
                    if cols[cn].get("type") in (None, "lookup"):
                        cols[cn]["type"] = "float"
        for name_, (lo, hi) in tl.length_bounds.items():
            if name_ in cols:
                cols[name_] = {"type": "integer", "min": lo, "max": hi}
        for c in tl.columns:
            if c.role == "tag" and c.name in cols:
                cols[c.name] = {"type": "categorical"}
            elif c.role == "key" and c.name in cols:
                cols[c.name] = {"type": "categorical"}
            elif c.role == "multihot" and c.name in cols:
                cols[c.name] = {"type": "integer", "min": 0}
        if tl.kind != "root":
            t.pop("mode", None)
            if lay.children(tname):
                # an element that holds arrays itself: the engine generates facts
                # only under entities, so it is an entity linked to its holder
                t["role"] = "entity"
                t.pop("fact", None)
            else:
                t["role"] = "fact"
                fact = t.setdefault("fact", {})
                fact["anchor"] = tl.parent
                # elements per document: the Dirichlet-process count law keeps the
                # source's length support (Prop. 1401 of counts.py's DP choice)
                fact.setdefault("count_model", "dp")
                if fact.get("partner") == tl.parent:
                    fact.pop("partner")
                # copied dimensions of functional dependencies are an entity
                # feature; in an element table the columns stay modelled jointly
                t.pop("dimensions", None)
        if tl.kind != "root" and "unique" in t:
            # a drafted grain (_parent, value) would turn the element counts into
            # pair occupancies; set semantics are kept by conform instead
            t["unique"] = [u for u in t["unique"] if PARENT not in u]
            if not t["unique"]:
                t.pop("unique")
        for k in ("group_effects", "link_affinity"):
            if k in t:
                t[k] = [g for g in t[k] if not _structural(g)]
                if not t[k]:
                    t.pop(k)
        t["primary_key"] = tl.id_column
    for sq in sequences or []:
        t = tables.get(sq["collection"])
        if t is not None and t.get("role") == "fact" and sq["path"] in t.get("columns", {}):
            t["columns"][sq["path"]] = {"type": "sequence", "within": _col(sq["within"].split(".")),
                                       "start": sq["start"]}
            for k in ("group_effects", "link_affinity", "dimensions"):
                if k in t:
                    t[k] = [g for g in t[k] if sq["path"] not in repr(g)]
                    if not t[k]:
                        t.pop(k)
    # a drafted count over an element table is the length column itself
    # (structural, matched in nest): it must not overwrite the modelled length
    lens = {(n, c) for n, tl in lay.tables.items() for c in tl.length_bounds}
    raw["rules"] = [r for r in raw.get("rules", []) if not (r.get("type") == "aggregate" and r.get("agg") == "count"
                                                          and (r.get("table"), r.get("column")) in lens)]
    # a multi-hot cell holds a position, so "cell <= length" holds in the source by
    # construction; drafted as a check it rejects short arrays and inflates the
    # length law (mflix genres: 49 % of length 3 against 40 %); nesting already
    # matches the cells to the length (Prop. 1785)
    hot = {c.name for tl in lay.tables.values() for c in tl.columns if c.role == "multihot"}
    if hot:
        import re as _re
        raw["rules"] = [r for r in raw.get("rules", [])
                        if not hot & set(_re.findall(r"[A-Za-z_][A-Za-z0-9_]*", repr(r)))]
    # elements summed into a holder's attribute by an aggregate rule stay with
    # the holder the engine drew them for (moving them would break the rule)
    for r in raw.get("rules", []):
        if r.get("type") == "aggregate" and r.get("from") in lay.tables:
            lay.tables[r["from"]].match_lengths = False
    raw.setdefault("synthesis", {})["seed"] = seed
    raw["source"] = {"kind": "sqlite", "path": str(db_path)}
    return raw


__all__ = ["write_sqlite", "spec_for", "target_column", "ROW_ID", "PARENT", "POS", "KEY"]
