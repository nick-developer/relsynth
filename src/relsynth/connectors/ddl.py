# SPDX-License-Identifier: Apache-2.0
"""Target table structure: cloned from the source's DDL or built from the
specification, created parents first, and loaded so that foreign keys hold.

Cloning keeps, per table: column names and order, types (the source's own types
on the same engine; their generic equivalents across engines, e.g. Oracle
NUMBER(10) -> INTEGER on PostgreSQL), nullability, the primary key, unique
constraints, foreign keys, unique and plain indexes, and identity columns
(see the connector of each engine for how explicit key values are loaded into
them).  Server defaults and check constraints are dialect text and are not
copied across engines.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import sqlalchemy as sa

log = logging.getLogger("relsynth.connectors")

# longest VARCHAR before a text type is used (Oracle VARCHAR2 limit in bytes for
# the default MAX_STRING_SIZE; SQL Server NVARCHAR(n) up to 4000)
MAX_VARCHAR = {"oracle": 4000, "mssql": 4000}


def load_order(spec: dict, names: list[str]) -> list[str]:
    """Tables in an order where every parent comes before its children (links
    within the written tables only)."""
    from ..spec import topo_order
    edges = [(r["child"], r["parent"]) for r in spec.get("relationships", [])
             if r.get("kind") == "many_to_one" and r["child"] in names and r["parent"] in names
             and r["child"] != r["parent"]]
    return topo_order(list(names), edges)


def rows_parents_first(spec: dict, table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Rows of a self-referencing table (an employee's manager, a category tree)
    ordered by depth, so that a self foreign key holds after every insert."""
    rel = next((r for r in spec.get("relationships", []) if r.get("kind") == "self_reference"
                and r["child"] == table), None)
    if rel is None or not len(df):
        return df
    pk = spec["tables"][table].get("primary_key")
    if pk not in df.columns or rel["child_key"] not in df.columns:
        return df
    from ..prepare import _keystr
    keys = _keystr(df[pk]).to_numpy(dtype=object)
    par = _keystr(df[rel["child_key"]]).to_numpy(dtype=object)
    pos = pd.Index(keys).get_indexer(par)
    depth = np.full(len(df), -1)
    depth[pos < 0] = 0
    for d in range(1, len(df) + 1):
        todo = depth < 0
        if not todo.any():
            break
        ready = todo & (depth[np.maximum(pos, 0)] == d - 1) & (pos >= 0)
        if not ready.any():
            depth[todo] = d             # a cycle: keep the remaining rows in their order
            break
        depth[ready] = d
    return df.iloc[np.argsort(depth, kind="stable")].reset_index(drop=True)


def _generic(ty, dialect: str):
    """A type that the target engine can create: the source type itself on the
    same engine, its generic equivalent otherwise."""
    try:
        g = ty.as_generic()
    except NotImplementedError:
        g = sa.String() if not isinstance(ty, sa.types.NullType) else sa.String()
    if isinstance(g, sa.Text) and dialect == "mssql":
        return sa.UnicodeText()                 # NVARCHAR(max): TEXT and VARCHAR(max) lose non-Latin text
    if isinstance(g, sa.String) and not isinstance(g, (sa.Text, sa.Enum)):
        n = getattr(g, "length", None)
        lim = MAX_VARCHAR.get(dialect)
        if n is None and dialect in ("oracle", "mssql", "mysql", "mariadb"):
            return sa.Text() if dialect != "mssql" else sa.UnicodeText()
        if lim and n and n > lim:
            return sa.Text() if dialect != "mssql" else sa.UnicodeText()
        return sa.Unicode(n) if dialect == "mssql" else sa.String(n)
    if isinstance(g, sa.Numeric) and not isinstance(g, (sa.Float, sa.Integer)):
        # NUMBER without precision (Oracle) has no generic precision: keep a float
        if getattr(g, "precision", None) is None:
            return sa.Float()
        if getattr(g, "scale", None) in (0, None) and g.precision <= 18:
            return sa.BigInteger() if g.precision > 9 else sa.Integer()
    if isinstance(g, sa.LargeBinary) and dialect == "oracle":
        return sa.LargeBinary()
    return g


def _sqlite_inline_uniques(source, table: str) -> set[tuple[str, ...]]:
    """UNIQUE written inline on a SQLite column lives only in an automatic index,
    which SQLAlchemy's reflection does not report."""
    out = set()
    q = sa.text("SELECT name, \"unique\", origin FROM pragma_index_list(:t)")
    with source.engine.connect() as con:
        for name, uniq, origin in con.execute(q, {"t": table}).fetchall():
            if uniq and origin == "u":
                cols = con.execute(sa.text("SELECT name FROM pragma_index_info(:i) ORDER BY seqno"),
                                   {"i": name}).scalars().all()
                if cols:
                    out.add(tuple(cols))
    return out


def clone_metadata(source, names: list[str], dialect: str, schema: str | None) -> sa.MetaData:
    """The source tables' structure, for creation in `schema` of a `dialect` database."""
    src_md = sa.MetaData(schema=source.schema)
    for t in names:
        sa.Table(t, src_md, autoload_with=source.engine)
    same = source.dialect == dialect
    md = sa.MetaData(schema=schema)
    for t in names:
        st = src_md.tables[f"{source.schema}.{t}" if source.schema else t]
        cols = []
        for c in st.columns:
            ty = c.type if same else _generic(c.type, dialect)
            kw = {"nullable": c.nullable, "primary_key": c.primary_key, "autoincrement": False}
            args = []
            if c.identity is not None:
                ident = c.identity
                # explicit key values must be accepted on load (PostgreSQL COPY overrides
                # ALWAYS; SQL Server uses IDENTITY_INSERT; Oracle needs BY DEFAULT)
                always = bool(ident.always) and dialect != "oracle"
                args.append(sa.Identity(always=always, start=ident.start, increment=ident.increment))
            cols.append(sa.Column(c.name, ty, *args, **kw))
        tbl = sa.Table(t, md, *cols)
        uniques = {tuple(c.name for c in uc.columns) for uc in st.constraints
                   if isinstance(uc, sa.UniqueConstraint) and uc.columns}
        if source.dialect == "sqlite":
            uniques |= _sqlite_inline_uniques(source, t)
        for cols_u in sorted(uniques):
            tbl.append_constraint(sa.UniqueConstraint(*cols_u))
        for fk in st.foreign_key_constraints:
            ref = fk.referred_table.name
            if ref not in names:
                continue
            tbl.append_constraint(sa.ForeignKeyConstraint(
                [c.name for c in fk.columns],
                [f"{schema}.{ref}.{e.column.name}" if schema else f"{ref}.{e.column.name}" for e in fk.elements]))
        for ix in st.indexes:
            if all(c.name in tbl.c for c in ix.columns) and ix.columns:
                sa.Index(None, *[tbl.c[c.name] for c in ix.columns], unique=ix.unique)
    return md


def _type_for(cs: dict, s: pd.Series, dialect: str):
    t = cs.get("type")
    if t == "date":
        return sa.Date()
    if t == "datetime":
        return sa.DateTime()
    if t == "boolean":
        return sa.Boolean()
    if t in ("integer", "id", "sequence") and pd.api.types.is_integer_dtype(s):
        return sa.BigInteger()
    if t in ("integer", "float") and pd.api.types.is_numeric_dtype(s):
        return sa.Float()
    if t == "binary":
        return sa.LargeBinary()
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return sa.BigInteger()
    if pd.api.types.is_float_dtype(s):
        return sa.Float()
    if pd.api.types.is_bool_dtype(s):
        return sa.Boolean()
    if pd.api.types.is_datetime64_any_dtype(s):
        return sa.DateTime()
    v = s.dropna().astype(str)
    n = int(v.str.len().max()) if len(v) else 1
    # room to grow: the next multiple of 50 above twice the longest value
    n = int(min(max(50, -(-2 * n // 50) * 50), 10**6))
    lim = MAX_VARCHAR.get(dialect, 10**6)
    if n > lim:
        return sa.UnicodeText() if dialect == "mssql" else sa.Text()
    return sa.Unicode(n) if dialect == "mssql" else sa.String(n)


def metadata_from_spec(spec: dict, tables: dict[str, pd.DataFrame], dialect: str, schema: str | None) -> sa.MetaData:
    """Tables from the specification: its columns (in the data's order), the column
    types of the data, the primary key, unique keys and the foreign keys among the
    written tables."""
    md = sa.MetaData(schema=schema)
    names = list(tables)
    for t, df in tables.items():
        ts = spec["tables"].get(t, {"columns": {}})
        pk = ts.get("primary_key")
        cols = []
        for c in df.columns:
            if str(c).startswith("__"):
                continue
            cs = ts.get("columns", {}).get(c, {})
            cols.append(sa.Column(c, _type_for(cs, df[c], dialect), primary_key=(c == pk and df[c].notna().all()
                                                                               and not df[c].duplicated().any()),
                                  autoincrement=False))
        tbl = sa.Table(t, md, *cols)
        for u in ts.get("unique", []) or []:
            if all(c in tbl.c for c in u) and not df.duplicated(list(u)).any():
                tbl.append_constraint(sa.UniqueConstraint(*u))
    for r in spec.get("relationships", []):
        if r.get("kind") not in ("many_to_one", "self_reference") or r.get("composite"):
            continue
        c, p = r["child"], r["parent"]
        if c not in names or p not in names:
            continue
        pk = spec["tables"][p].get("primary_key")
        ck = r.get("child_key")
        if not isinstance(pk, str) or not isinstance(ck, str):
            continue                       # composite or missing keys: no single-column constraint
        ct, pt = md.tables[f"{schema}.{c}" if schema else c], md.tables[f"{schema}.{p}" if schema else p]
        if ck in ct.c and pk in pt.c and pt.c[pk].primary_key:
            ct.append_constraint(sa.ForeignKeyConstraint([ck], [pt.c[pk]]))
    return md


def create_tables(engine, md: sa.MetaData, names: list[str], replace: bool, schema: str | None) -> None:
    """Create the tables of md (dropping the existing ones first when replace)."""
    tbls = [md.tables[f"{schema}.{t}" if schema else t] for t in names]
    with engine.begin() as con:
        if replace:
            existing = sa.MetaData(schema=schema)
            have = [t for t in names if sa.inspect(con).has_table(t, schema=schema)]
            for t in have:
                sa.Table(t, existing, autoload_with=con)
            # children first; foreign keys from tables outside this set block the drop
            existing.drop_all(con, tables=[existing.tables[f"{schema}.{t}" if schema else t] for t in have])
        md.create_all(con, tables=tbls)
