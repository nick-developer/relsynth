# SPDX-License-Identifier: Apache-2.0
"""Read access to a source database for profiling (introspect).

One interface over SQL databases (any SQLAlchemy URL, SQLite files) and folders
of CSV or Parquet files (every read is normalised by data.normalise_frame):

* the catalog: column types, primary keys, foreign keys and unique constraints
  as declared, with composite keys kept together (one entry per constraint);
* row counts;
* a uniform random sample of rows per table (ORDER BY the dialect's random
  function; the whole table when it is small), so that every inference is made
  on a sample whose error rates are known, not on the first rows in key order;
* the complete values of one column (for inclusion tests and key sets), or of
  several columns read together (composite keys, self-references);
* every row of chosen groups (cluster samples), for dependencies within groups:
  an order's lines, a product's history, a parent's children.

Binary columns are never read: they are reported as binary and left empty.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data import FILE_KINDS, normalise_frame, read_table_file, table_files

# up to this many rows the primary key is read to draw a seeded sample; beyond it the
# dialect's random order is used (not reproducible, reported in the log)
KEY_SAMPLE_MAX_ROWS = 5_000_000
BINARY_SQL = r"binary|blob|image|bytea|(^|\W)raw|varbinary|largebinary|_binary"


def is_binary_sqltype(t: str) -> bool:
    return bool(re.search(BINARY_SQL, str(t).lower()))


def _match(name, include, exclude):
    if include and not any(re.fullmatch(p, name, flags=re.I) for p in include):
        return False
    if exclude and any(re.fullmatch(p, name, flags=re.I) for p in exclude):
        return False
    return True


class Source:
    def __init__(self, source: dict, include=None, exclude=None, seed=7):
        self.source = source
        self.kind = source.get("kind", "csv")
        self.seed = seed
        self.meta = {}
        self.random_order = True
        self._col_cache = {}
        self.unseeded = []
        if self.kind in FILE_KINDS:
            self.kind = "csv"                     # one code path for every file format
            self._files = {t: p for t, p in table_files(source["path"]).items() if _match(t, include, exclude)}
            self._frames = {}
            for t in self._files:
                df = self._frame(t)
                self.meta[t] = {"pk": [], "fks": [], "uniques": [], "types": {c: "" for c in df.columns},
                                "binary": []}
            self.names = list(self._files)
            return
        from sqlalchemy import MetaData, Table, func, inspect, literal_column

        from .connectors import from_spec
        self.conn = from_spec(source, source.get("_base_dir"))
        self.eng = self.conn.engine
        self.schema = self.conn.schema
        insp = inspect(self.eng)
        names = [t for t in insp.get_table_names(schema=self.schema) if _match(t, include, exclude)]
        self.names = names
        self._tables = {}
        md = MetaData()
        for t in names:
            cols = insp.get_columns(t, schema=self.schema)
            types = {str(c["name"]): str(c["type"]).lower() for c in cols}
            pk = [str(c) for c in (insp.get_pk_constraint(t, schema=self.schema) or {}).get("constrained_columns") or []]
            fks = []
            for fk in insp.get_foreign_keys(t, schema=self.schema):
                if fk.get("referred_table") is None:
                    continue
                fks.append({"cols": [str(c) for c in fk["constrained_columns"]], "parent": str(fk["referred_table"]),
                            "pcols": [str(c) for c in fk["referred_columns"]]})
            uniques = []
            try:
                uniques += [[str(c) for c in u["column_names"]] for u in insp.get_unique_constraints(t, schema=self.schema)]
            except NotImplementedError:
                pass
            try:
                uniques += [[str(c) for c in ix["column_names"]] for ix in insp.get_indexes(t, schema=self.schema)
                            if ix.get("unique") and all(ix.get("column_names") or [None])]
            except NotImplementedError:
                pass
            uq = []
            for u in uniques:
                if u and u != pk and u not in uq:
                    uq.append(u)
            self.meta[t] = {"pk": pk, "fks": fks, "uniques": uq, "types": types,
                            "binary": [c for c, ty in types.items() if is_binary_sqltype(ty)]}
            self._tables[t] = Table(t, md, autoload_with=self.eng, schema=self.schema)
        d = self.eng.dialect.name
        self._rnd = {"sqlite": func.random(), "postgresql": func.random(), "mssql": func.newid(),
                     "mysql": func.rand(), "mariadb": func.rand(), "databricks": func.rand(), "snowflake": func.random(),
                     "bigquery": func.rand(),
                     "oracle": literal_column("dbms_random.value")}.get(d)
        self.random_order = self._rnd is not None
        self._counts = {}

    # ---------------------------------------------------------------- csv
    def _frame(self, t):
        if t not in self._frames:
            self._frames[t] = read_table_file(self._files[t])
        return self._frames[t]

    # ---------------------------------------------------------------- api
    def count(self, t) -> int:
        if self.kind == "csv":
            return len(self._frame(t))
        if t not in self._counts:
            from sqlalchemy import func, select
            with self.eng.connect() as con:
                self._counts[t] = int(con.execute(select(func.count()).select_from(self._tables[t])).scalar())
        return self._counts[t]

    def _cols(self, t):
        if self.kind == "csv":
            return list(self._frame(t).columns)
        return [c for c in self._tables[t].columns.keys() if c not in self.meta[t]["binary"]]

    def _finish(self, t, df):
        df.columns = [str(c) for c in df.columns]
        df = normalise_frame(df)
        for c in self.meta[t]["binary"]:
            df[c] = None
        allc = list(self.meta[t]["types"])
        return df[[c for c in allc if c in df.columns]]

    def sample(self, t, n):
        """(rows, is_whole_table)"""
        total = self.count(t)
        if self.kind == "csv":
            df = self._frame(t)
            if total <= n:
                return df.copy(), True
            return df.sample(n, random_state=self.seed).reset_index(drop=True), False
        from sqlalchemy import select
        tbl = self._tables[t]
        q = select(*[tbl.c[c] for c in self._cols(t)])
        whole = total <= n
        if not whole:
            pk = self.meta[t]["pk"] or next(iter(self.meta[t]["uniques"]), None)
            if pk and total <= KEY_SAMPLE_MAX_ROWS:
                return self._finish(t, self._key_sample(t, pk, n)), False
            q = (q.order_by(self._rnd) if self._rnd is not None else q).limit(n)
            self.unseeded.append(t)
        with self.eng.connect() as con:
            df = pd.read_sql(q, con)
        return self._finish(t, df), whole

    def _key_sample(self, t, pk, n):
        """Seeded simple random sample: read the primary key columns in key order,
        choose n of them with the seeded generator, fetch those rows.  The same
        database and seed give the same sample (ORDER BY random() does not), so the
        drafted spec is reproducible."""
        from sqlalchemy import select
        tbl = self._tables[t]
        with self.eng.connect() as con:
            keys = pd.read_sql(select(*[tbl.c[c] for c in pk]).order_by(*[tbl.c[c] for c in pk]), con)
        keys.columns = [str(c) for c in keys.columns]
        rng = np.random.default_rng([self.seed, sum(map(ord, t))])
        pick = keys.iloc[np.sort(rng.choice(len(keys), size=n, replace=False))].reset_index(drop=True)
        # filter on the most selective key column, then keep exactly the chosen keys
        col = max(pk, key=lambda c: pick[c].nunique())
        df = self.rows_in(t, col, pick[col].tolist())
        if len(pk) > 1:
            df = df.merge(pick.drop_duplicates(), on=pk, how="inner")
        return df.sort_values(pk, kind="stable").reset_index(drop=True)

    def column(self, t, c) -> pd.Series:
        """Every value of one column (cached)."""
        if (t, c) not in self._col_cache:
            if self.kind == "csv":
                s = self._frame(t)[c]
            else:
                from sqlalchemy import select
                with self.eng.connect() as con:
                    s = normalise_frame(pd.read_sql(select(self._tables[t].c[c]), con))[c]
            self._col_cache[(t, c)] = s.reset_index(drop=True)
        return self._col_cache[(t, c)]

    def columns(self, t, cols) -> pd.DataFrame:
        """Every row of several columns, read together in one query (aligned row by
        row), for tests on the whole table: a composite key, a self-reference."""
        cols = list(cols)
        if self.kind == "csv":
            return self._frame(t)[cols].reset_index(drop=True)
        from sqlalchemy import select
        with self.eng.connect() as con:
            df = pd.read_sql(select(*[self._tables[t].c[c] for c in cols]), con)
        df.columns = [str(c) for c in df.columns]
        return normalise_frame(df)[cols].reset_index(drop=True)

    def rows_in(self, t, col, keys, cols=None):
        """All rows of t whose column col takes one of the given values."""
        keys = [k for k in pd.unique(pd.Series(list(keys), dtype=object)) if k is not None and not
                (isinstance(k, float) and np.isnan(k))]
        if self.kind == "csv":
            df = self._frame(t)
            ks = set(map(str, keys))
            out = df[df[col].astype(str).isin(ks)]
            return (out[cols] if cols else out).reset_index(drop=True)
        from sqlalchemy import select
        tbl = self._tables[t]
        use = [c for c in (cols or self._cols(t)) if c not in self.meta[t]["binary"]]
        parts = []
        with self.eng.connect() as con:
            for i in range(0, len(keys), 500):
                chunk = [k.item() if hasattr(k, "item") else k for k in keys[i:i + 500]]
                q = select(*[tbl.c[c] for c in use]).where(tbl.c[col].in_(chunk))
                parts.append(pd.read_sql(q, con))
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=use)
        df.columns = [str(c) for c in df.columns]
        return df
