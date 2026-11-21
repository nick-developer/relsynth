# SPDX-License-Identifier: Apache-2.0
"""Read access to a source database for profiling (introspect).

One interface over SQL databases (any SQLAlchemy URL, SQLite files) and folders
of CSV or Parquet files (every read is normalised by data.normalise_frame):

* the catalog: column types, primary keys, foreign keys and unique constraints
  as declared, with composite keys kept together (one entry per constraint);
* row counts;
* a uniform random sample of rows per table (seeded: through the primary key, or
  for a table without one by a keyed hash of each row's content, so the same
  database gives the same sample and the same draft; ORDER BY the dialect's random
  function only past KEY_SAMPLE_MAX_ROWS; the whole table when it is small), so
  that every inference is made on a sample whose error rates are known, not on the
  first rows in key order;
* the complete values of one column (for inclusion tests and key sets), or of
  several columns read together (composite keys, self-references);
* every row of chosen groups (cluster samples), for dependencies within groups:
  an order's lines, a product's history, a parent's children.

Binary columns are never read: they are reported as binary and left empty.

A Parquet file with more than LAZY_PARQUET_ROWS rows or LAZY_PARQUET_BYTES
(uncompressed; pandas needs several times that) is not held in memory: counts
come from its footer, samples are drawn while streaming its row groups, the
distinct values of an integer column are collected group by group, and columns
or chosen rows are read on demand (a 60-million-row TPC-H lineitem or a table
of long texts would not fit on a 16 GB machine otherwise).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _LEADING_ZERO, FILE_KINDS, normalise_frame, read_table_file, table_files

# up to this many rows a seeded sample is drawn (through the primary key, or for a table
# without one through a hash of each row's content); beyond it the dialect's random
# order is used (not reproducible, reported in the log)
KEY_SAMPLE_MAX_ROWS = 5_000_000
LAZY_PARQUET_ROWS = 5_000_000
LAZY_PARQUET_BYTES = 256 << 20
BINARY_SQL = r"binary|blob|image|bytea|(^|\W)raw|varbinary|largebinary|_binary"


def is_binary_sqltype(t: str) -> bool:
    return bool(re.search(BINARY_SQL, str(t).lower()))


def _mix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser (Steele et al. 2014) on uint64, wrapping: a bijection that
    spreads (hash, copy) pairs uniformly, so the copies of a duplicated row get
    independent priorities."""
    x = np.asarray(x, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def _large_parquet(path) -> bool:
    """A Parquet file over LAZY_PARQUET_ROWS rows or LAZY_PARQUET_BYTES uncompressed (from its footer)."""
    if not str(path).lower().endswith(".parquet"):
        return False
    import pyarrow.parquet as pq
    md = pq.ParquetFile(path).metadata
    return md.num_rows > LAZY_PARQUET_ROWS or \
        sum(md.row_group(i).total_byte_size for i in range(md.num_row_groups)) > LAZY_PARQUET_BYTES


def _key_text(s) -> pd.Series:
    """Keys as text for matching a file table's rows: a key column with NULLs comes back from Parquet as
    floats (2450815.0), and integer-valued keys must match the integer column of the parent (2450815).
    Each value is mapped on its own (not by the rest of its column, as canonical_keys decides), so the
    requested keys and every batch of the source column agree: whole numbers and whole-number text
    without a leading zero become the integer's digits, any other value its own text."""
    s = pd.Series(s).reset_index(drop=True)
    na = s.isna().to_numpy()
    if pd.api.types.is_bool_dtype(s):
        out = s.astype(str).to_numpy(dtype=object)
    elif pd.api.types.is_integer_dtype(s):
        out = s.astype("Int64").astype(str).to_numpy(dtype=object)
    elif pd.api.types.is_numeric_dtype(s):
        f = s.to_numpy(dtype=float, na_value=np.nan)
        with np.errstate(invalid="ignore"):
            whole = np.isfinite(f) & (f == np.round(f)) & (f >= -(2.0**63)) & (f < 2.0**63)
        out = s.astype(str).to_numpy(dtype=object)
        out[whole] = [str(x) for x in f[whole].astype(np.int64)]
    else:
        out = s.astype(str).to_numpy(dtype=object)
        t = pd.Series(out, dtype=object).str.strip()
        m = ~na & t.str.fullmatch(r"[+-]?\d+(\.0*)?").fillna(False).to_numpy(dtype=bool) \
            & ~t.str.match(_LEADING_ZERO).fillna(False).to_numpy(dtype=bool)
        out[m] = [str(int(x.split(".")[0])) for x in t[m]]
    out[na] = "<NA>"
    return pd.Series(out, dtype=object)


def _match(name, include, exclude):
    if include and not any(re.fullmatch(p, name, flags=re.I) for p in include):
        return False
    if exclude and any(re.fullmatch(p, name, flags=re.I) for p in exclude):
        return False
    return True


class Source:
    def __init__(self, source: dict, include=None, exclude=None, seed=7):
        """A data source from its spec section, limited to the included tables."""
        self.source = source
        self.kind = source.get("kind", "csv")
        self.seed = seed
        self.meta = {}
        self.random_order = True
        self._col_cache: dict = {}
        self.unseeded: list = []
        self._lazy: dict = {}                 # table -> pyarrow ParquetFile read on demand
        if self.kind in ("iceberg", "delta"):
            # lakehouse tables: read like files (whole, in memory), keys as recorded
            from .connectors import from_spec
            self.lake = from_spec(source, source.get("_base_dir"))
            self.kind = "csv"
            self._files: dict[str, Path | None] = {t: None for t in self.lake.tables(include, exclude)}
            self._frames: dict = {}
            for t in self._files:
                info = self.lake.describe(t)
                fks = [{"cols": list(f.columns), "parent": f.ref_table, "pcols": list(f.ref_columns)}
                       for f in info.foreign_keys]
                self.meta[t] = {"pk": list(info.primary_key), "fks": fks, "uniques": [list(u) for u in info.unique],
                                "types": {c.name: c.sql_type for c in info.columns}, "binary": []}
            self.names = list(self._files)
            return
        if self.kind in FILE_KINDS:
            self.kind = "csv"                     # one code path for every file format
            self._files = {t: p for t, p in table_files(source["path"]).items() if _match(t, include, exclude)}
            self._frames = {}
            for t in self._files:
                if _large_parquet(self._files[t]):
                    import pyarrow.parquet as pq
                    self._lazy[t] = pq.ParquetFile(self._files[t])
                    names = list(self._lazy[t].schema_arrow.names)
                else:
                    names = list(self._frame(t).columns)
                self.meta[t] = {"pk": [], "fks": [], "uniques": [], "types": {c: "" for c in names},
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
                pass                    # the dialect cannot list them
            try:
                uniques += [[str(c) for c in ix["column_names"]] for ix in insp.get_indexes(t, schema=self.schema)
                            if ix.get("unique") and all(ix.get("column_names") or [None])]
            except NotImplementedError:
                pass                    # the dialect cannot list them
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
        self._counts: dict = {}

    # ---------------------------------------------------------------- csv
    def _frame(self, t):
        if t not in self._frames:
            self._frames[t] = self.lake.read_table(t) if self._files[t] is None else read_table_file(self._files[t])
        return self._frames[t]

    # ---------------------------------------------------------------- api
    def count(self, t) -> int:
        if t in self._lazy:
            return int(self._lazy[t].metadata.num_rows)
        if self.kind == "csv":
            return len(self._frame(t))
        if t not in self._counts:
            from sqlalchemy import func, select
            with self.eng.connect() as con:
                self._counts[t] = int(con.execute(select(func.count()).select_from(self._tables[t])).scalar())
        return self._counts[t]

    def _cols(self, t):
        if t in self._lazy:
            return list(self._lazy[t].schema_arrow.names)
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
        if t in self._lazy:
            return self._lazy_sample(t, n, total)
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
            if total <= KEY_SAMPLE_MAX_ROWS:
                return self._finish(t, self._content_sample(t, n)), False
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

    def _row_hashes(self, rows) -> np.ndarray:
        """64-bit keyed BLAKE2b hashes of whole rows (the repr of the driver's values),
        so a row's hash depends on its content and the seed only, never on its position."""
        import hashlib
        key = f"relsynth-sample-{self.seed}".encode()
        return np.fromiter((int.from_bytes(hashlib.blake2b(repr(tuple(r)).encode("utf-8", "surrogatepass"),
                                                           digest_size=8, key=key).digest(), "little")
                            for r in rows), dtype=np.uint64, count=len(rows))

    def _content_sample(self, t, n, chunk=50_000):
        """Seeded simple random sample of a table with no key: bottom-n by a keyed hash
        of each row's content (order sampling with uniform priorities, Rosen 1997; the
        bottom-k sketch of Cohen and Kaplan 2007).  ORDER BY random() gives another
        sample on every run, and so another drafted spec; this sample depends only on
        the table's rows and the seed, not on the order the database returns them in.
        Identical rows share a content hash, so copy j of a row gets the priority of
        (hash, j): every row, duplicated or not, has the same chance n / rows of being
        chosen.  Two streamed passes: the hashes (8 bytes a row), then the chosen rows."""
        from sqlalchemy import select
        tbl = self._tables[t]
        cols = self._cols(t)
        q = select(*[tbl.c[c] for c in cols])

        def stream():
            with self.eng.connect() as con:
                res = con.execution_options(stream_results=True).execute(q)
                while True:
                    rows = res.fetchmany(chunk)
                    if not rows:
                        return
                    yield rows
        h = np.concatenate([self._row_hashes(rows) for rows in stream()] or [np.zeros(0, np.uint64)])
        order = np.argsort(h, kind="stable")
        hs = h[order]
        start = np.r_[0, np.flatnonzero(hs[1:] != hs[:-1]) + 1]
        rank = np.arange(len(hs)) - np.repeat(start, np.diff(np.r_[start, len(hs)]))
        pri = _mix64(hs ^ (rank.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)))
        pick = np.argsort(pri, kind="stable")[:n] if n < len(pri) else np.arange(len(pri))
        need: dict[int, int] = {}
        for x in hs[pick].tolist():
            need[x] = need.get(x, 0) + 1
        prio = {}                                 # hash -> priorities of its chosen copies, in rank order
        for x, p, _r in sorted(zip(hs[pick].tolist(), pri[pick].tolist(), rank[pick].tolist()), key=lambda z: z[2]):
            prio.setdefault(x, []).append(p)
        kept, keys = [], []
        for rows in stream():
            for r, x in zip(rows, self._row_hashes(rows).tolist()):
                left = need.get(x, 0)
                if left:
                    need[x] = left - 1
                    kept.append(tuple(r))
                    keys.append(prio[x][len(prio[x]) - left])
        df = pd.DataFrame.from_records([kept[i] for i in np.argsort(np.asarray(keys, dtype=np.uint64), kind="stable")],
                                       columns=cols) if kept else pd.DataFrame(columns=cols)
        return df

    def column(self, t, c) -> pd.Series:
        """Every value of one column (cached)."""
        if (t, c) not in self._col_cache:
            if t in self._lazy:                   # read again when asked: a large table's columns are not kept
                return self._lazy_read(t, [c])[c]
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
        if t in self._lazy:
            return self._lazy_read(t, cols)
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
        if t in self._lazy:
            return self._lazy_rows_in(t, col, keys, cols)
        if self.kind == "csv":
            df = self._frame(t)
            ks = set(_key_text(keys))
            out = df[_key_text(df[col]).isin(ks).to_numpy()]
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

    # ---------------------------------------------------------------- large Parquet files
    def _lazy_read(self, t, cols) -> pd.DataFrame:
        return normalise_frame(self._lazy[t].read(columns=list(cols)).to_pandas())[list(cols)].reset_index(drop=True)

    def _lazy_sample(self, t, n, total):
        """A seeded Bernoulli(n / rows) sample drawn while streaming the row groups."""
        pf = self._lazy[t]
        if total <= n:
            return normalise_frame(pf.read().to_pandas()), True
        rng = np.random.default_rng([self.seed, sum(map(ord, t))])
        q = n / total
        parts = []
        for b in pf.iter_batches(batch_size=65_536):
            keep = np.flatnonzero(rng.random(b.num_rows) < q)
            if len(keep):
                parts.append(b.take(keep).to_pandas())
        df = pd.concat(parts, ignore_index=True) if parts else pf.schema_arrow.empty_table().to_pandas()
        return normalise_frame(df), False

    def _lazy_rows_in(self, t, col, keys, cols=None):
        """Rows whose col is one of keys, compared as canonical key text (as for a CSV table), streamed by row
        group."""
        pf = self._lazy[t]
        use = list(cols) if cols else list(pf.schema_arrow.names)
        read = use if col in use else [*use, col]
        import pyarrow as pa
        ks = set(_key_text(keys))
        parts = []
        for b in pf.iter_batches(batch_size=65_536, columns=read):
            # only the key column becomes pandas; the other columns (long texts) only for the matching rows
            k = normalise_frame(b.select([col]).to_pandas())[col]
            hit = _key_text(k).isin(ks).to_numpy()
            if hit.any():
                parts.append(normalise_frame(b.filter(pa.array(hit)).to_pandas())[use])
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=use)
        return out[use].reset_index(drop=True)

    def nulls_and_repeats(self, t, cols):
        """(any NULL in cols, any repeated combination among the rows without NULLs) of a large Parquet
        table, counted by DuckDB in bounded memory (pandas would hold every row of a 133-million-row table
        and hash it); None for any other table."""
        if t not in self._lazy:
            return None
        import tempfile

        import duckdb
        from duckdb import ColumnExpression, FunctionExpression
        notnull = None
        for c in cols:
            e = ColumnExpression(str(c)).isnotnull()
            notnull = e if notnull is None else notnull & e
        with tempfile.TemporaryDirectory() as tmp:
            con = duckdb.connect(config={"memory_limit": "2GB", "temp_directory": tmp})
            try:
                con.execute("SET enable_progress_bar = false")
                rel = con.read_parquet(str(self._files[t]))
                nulls = bool(rel.filter(~notnull).limit(1).fetchall())
                keys = [ColumnExpression(str(c)) for c in cols]
                full = rel.filter(notnull).project(*keys)
                n = FunctionExpression("count_star")
                reps = full.aggregate([n]).fetchall()[0][0] != full.distinct().aggregate([n]).fetchall()[0][0]
            finally:
                con.close()
        return bool(nulls), bool(reps)

    def unique_where_set(self, t, cols):
        """(rows where every column of cols is set, [is each column unique on those rows])
        over the whole table, counted in the store (SQL COUNT(DISTINCT), DuckDB for a
        large Parquet table) so no column is held in memory.  Values compare as the
        store compares them (a case-insensitive collation counts 'a' and 'A' once)."""
        cols = [str(c) for c in cols]
        if t in self._lazy:
            import tempfile

            import duckdb
            from duckdb import ColumnExpression, FunctionExpression
            notnull = None
            for c in cols:
                e = ColumnExpression(c).isnotnull()
                notnull = e if notnull is None else notnull & e
            n = FunctionExpression("count_star")
            with tempfile.TemporaryDirectory() as tmp:
                con = duckdb.connect(config={"memory_limit": "2GB", "temp_directory": tmp})
                try:
                    con.execute("SET enable_progress_bar = false")
                    full = con.read_parquet(str(self._files[t])).filter(notnull)
                    rows = int(full.aggregate([n]).fetchall()[0][0])
                    dist = [int(full.project(ColumnExpression(c)).distinct().aggregate([n]).fetchall()[0][0])
                            for c in cols]
                finally:
                    con.close()
        elif self.kind == "csv":
            sub = self._frame(t)[cols].dropna()
            rows, dist = len(sub), [int(sub[c].nunique()) for c in cols]
        else:
            from sqlalchemy import and_, distinct, func, select
            tbl = self._tables[t]
            q = select(func.count(), *[func.count(distinct(tbl.c[c])) for c in cols]) \
                .where(and_(*[tbl.c[c].isnot(None) for c in cols]))
            with self.eng.connect() as con:
                rows, *dist = (int(x) for x in con.execute(q).one())
        return rows, [d == rows for d in dist]

    def int_distinct(self, t, c):
        """(sorted distinct values, non-null count) of an integer or float column of a large Parquet table,
        streamed by row group (a 133-million-row column never becomes Python objects); None for any other."""
        if t not in self._lazy:
            return None
        import pyarrow as pa
        import pyarrow.compute as pc
        pf = self._lazy[t]
        ty = pf.schema_arrow.field(c).type
        dec = pa.types.is_decimal(ty)             # read as floats (normalise_frame does the same)
        if not (pa.types.is_integer(ty) or pa.types.is_floating(ty) or dec):
            return None
        parts, n = [], 0
        for b in pf.iter_batches(batch_size=1 << 20, columns=[c]):
            col = b.column(0).drop_null()
            if dec:
                col = col.cast(pa.float64())
            if dec or pa.types.is_floating(ty):
                col = col.filter(pc.invert(pc.is_nan(col)))
            n += len(col)
            parts.append(np.asarray(pc.unique(col)))
        if not parts:
            return np.empty(0, dtype=np.int64), 0
        u = np.unique(np.concatenate(parts))
        if u.dtype.kind == "f":
            # as canonical_keys and keyfind do on a whole column: whole numbers become integer keys; other
            # values are cut to integers the way the object path's int64 conversion cuts them
            if not (np.isfinite(u).all() and (u >= -(2.0**63)).all() and (u < 2.0**63).all()):
                return None
            whole = bool((u == np.round(u)).all())
            u = np.unique(u.astype(np.int64)) if whole else np.sort(u.astype(np.int64))
        return u.astype(np.int64), n
