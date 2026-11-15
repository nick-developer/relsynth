# SPDX-License-Identifier: Apache-2.0
"""The rows stage 1 of DP mode reads: in memory, or in the source database
(gap plan G2; docs/dp.md section 9, Props. 1760-1763).

Stage 1 reads each owned table only through a few kinds of question about its
*bounded* rows (the rows left after contribution bounding, `bound`):

* `groups(cols)`: the distinct tuples of cols and how many rows hold each one.
  Every histogram, level count and row count of stage 1 is a sum over rows of a
  vector that depends on one row's values, so it is the same sum over the
  distinct tuples, each weighted by its count (Prop. 1761);
* `degrees(child, ...)`: each row's number of rows in a child table, as a
  frequency table;
* `group_sizes(...)`: rows per (parent row, group) as a frequency table;
* `joint(...)`: the distinct tuples of a row's modelled columns, its capped
  child counts and its parent row's columns, with their counts: the input of
  the correlation sums and of the two-way tables (Prop. 1762);
* `bound(child, ...)`: the child's rows whose parent row is kept, at most D per
  parent row, chosen by a keyed priority of the row's own values (Prop. 1760).

`FrameRows` answers them from frames in memory (the reference; DP runs with
tables given, or on sources without SQL).  `SqlRows` answers them with
GROUP BY, window functions and joins in the source database, through the same
SQL layer as the non-DP pushdown (SQLite, PostgreSQL, SQL Server, Oracle,
MySQL/MariaDB, DuckDB files, the warehouses).  Only frequency tables leave the
database, read in chunks of `CHUNK` tuples, so stage 1 holds no table in memory.
"""
from __future__ import annotations

import abc
import itertools
import secrets
from collections.abc import Iterator

import numpy as np
import pandas as pd

from ..errors import PrivacyError

CHUNK = 200_000                  # tuples per chunk read from the database
_alias = itertools.count()


def bincount(idx, weights, minlength: int) -> np.ndarray:
    """Integer counts of idx, each occurrence counted `weights` times (None: once)."""
    idx = np.asarray(idx, dtype=np.int64)
    if weights is None:
        return np.bincount(idx, minlength=minlength).astype(np.int64)
    return np.rint(np.bincount(idx, weights=np.asarray(weights, dtype=float), minlength=minlength)).astype(np.int64)


def _keys(k):
    return list(k) if isinstance(k, (list, tuple)) else [k]


class Rows(abc.ABC):
    """The bounded rows of one owned table (see the module docstring)."""

    table: str = ""

    @property
    @abc.abstractmethod
    def n(self) -> int:
        """Number of rows (exact; used only for control flow, never released)."""

    @abc.abstractmethod
    def groups(self, cols: list) -> Iterator[tuple[pd.DataFrame, np.ndarray | None]]:
        """Chunks of (distinct tuples of cols, rows holding each tuple); None means
        one row per frame row."""

    @abc.abstractmethod
    def degrees(self, child: str, pk: list, ck: list) -> tuple[np.ndarray, np.ndarray | None]:
        """(number of rows of the whole child table referencing a row, rows with that number)."""

    @abc.abstractmethod
    def group_sizes(self, ck: list, g: str) -> tuple[np.ndarray, np.ndarray | None]:
        """(rows of a group (parent key, value of g; null a group of its own per parent),
        groups of that size)."""

    @abc.abstractmethod
    def bound(self, child: str, pk: list, ck: list, D: int) -> Rows:
        """The child's rows whose parent row is among these rows, at most D per parent row."""

    @abc.abstractmethod
    def set_degree(self, name: str, child: str, pk: list, ck: list, D: int) -> None:
        """Make min(rows of child per row, D) the derived column `name` of the joint tuples."""

    @abc.abstractmethod
    def joint(self, cols: list, degrees: list | tuple = (), parent: tuple | None = None
              ) -> Iterator[tuple[pd.DataFrame, np.ndarray | None]]:
        """Chunks of distinct tuples of (cols, derived degree columns, the parent
        row's columns as '^c') with their row counts.  parent: (parent Rows, ck, pk,
        parent columns)."""


class RowSource(abc.ABC):
    """Where stage 1 reads the real data: the unit tables and the public tables."""

    pushdown: bool = False

    @abc.abstractmethod
    def unit(self, name: str) -> Rows:
        """Every row of a unit's own table (one row per person)."""

    @abc.abstractmethod
    def public(self, names: list) -> dict:
        """The public tables (released as given), loaded and coerced."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


# ------------------------------------------------------------------ in memory
class FrameSource(RowSource):
    """The real tables in memory (prepared frames, as `load_tables` gives them)."""

    def __init__(self, real: dict, key: str | None = None):
        self.real = real
        self.key = key or secrets.token_hex(8)

    def unit(self, name):
        return FrameRows(self, name, self.real[name].reset_index(drop=True))

    def public(self, names):
        return {t: self.real[t] for t in names if t in self.real}


class FrameRows(Rows):
    def __init__(self, src: FrameSource, table: str, df: pd.DataFrame):
        self.src, self.table, self.df = src, table, df.reset_index(drop=True)
        self._deg: dict = {}

    @property
    def n(self):
        return len(self.df)

    def groups(self, cols):
        yield self.df[list(cols)], None

    def _child_counts(self, child, pk, ck):
        key = (child, tuple(pk), tuple(ck))
        if key not in self._deg:
            from .relational import _degrees
            self._deg[key] = _degrees(self.df, self.src.real[child], pk, ck)
        return self._deg[key]

    def degrees(self, child, pk, ck):
        return self._child_counts(child, pk, ck), None

    def group_sizes(self, ck, g):
        from .relational import _group_sizes
        return _group_sizes(self.df, ck, g), None

    def bound(self, child, pk, ck, D):
        from .relational import _truncate
        return FrameRows(self.src, child, _truncate(self.src.real[child], self.df, pk, ck, D, self.src.key))

    def set_degree(self, name, child, pk, ck, D):
        self.df[name] = np.minimum(self._child_counts(child, pk, ck), D)

    def joint(self, cols, degrees=(), parent=None):
        out = self.df[list(cols) + list(degrees)].copy()
        if parent is not None:
            from .relational import _parent_index
            prows, ck, pk, pcols = parent
            idx = _parent_index(self.df, prows.df, ck, pk)
            for c in pcols:
                out[f"^{c}"] = prows.df[c].to_numpy()[idx]
        yield out, None


# ------------------------------------------------------------------ in the database
def sql_connector(conn):
    """The SQL connector behind a source connector, or None (files without DuckDB)."""
    from ..connectors.sql import SqlConnector
    if isinstance(conn, SqlConnector):
        return conn
    duck = getattr(conn, "duck", None) if getattr(conn, "kind", "") in ("csv", "parquet") else None
    return duck if isinstance(duck, SqlConnector) else None


class SqlSource(RowSource):
    """Stage 1 inside the source database.  `seed` is the run's secret key of the
    truncation priorities (drawn per run; Prop. 1760 does not need it secret)."""

    pushdown = True

    def __init__(self, spec: dict, conn, owner=None, seed: int | None = None):
        sq = sql_connector(conn)
        if sq is None:
            raise PrivacyError(f"DP pushdown needs a database source; {getattr(conn, 'kind', '?')} sources are "
                               "read into memory (synthesis.privacy.dp.pushdown: off)")
        self.spec, self.conn, self.sql, self.owner = spec, conn, sq, owner
        self.seed = int(seed if seed is not None else secrets.randbits(31))
        self.dialect = sq.dialect

    def close(self):
        if self.owner is not None:
            self.owner.close()

    def _columns(self, table: str) -> list:
        """The spec's columns of a table that the database stores (binary columns are never read)."""
        from ..connectors.sql import is_binary_type
        info = self.sql.describe(table)
        stored = {c.name: c for c in info.columns}
        return [c for c in self.spec["tables"][table]["columns"]
                if c in stored and not is_binary_type(stored[c].sql_type)]

    def unit(self, name):
        import sqlalchemy as sa
        tbl = self.sql.table(name)
        sel = sa.select(*[tbl.c[c] for c in self._columns(name)]).subquery(f"rs_dp{next(_alias)}")
        return SqlRows(self, name, sel)

    def public(self, names):
        from ..data import coerce
        from ..prepare import computed_columns
        from ..pushdown.loader import canonical_order
        comp = computed_columns(self.spec)
        out = {}
        for t in names:
            if self.conn.has_table(t):
                T = self.spec["tables"][t]
                raw = canonical_order(self.conn.read_table(t), T.get("primary_key"), T.get("unique"))
                out[t] = coerce(raw, T, optional=set(comp.get(t, ())), table=t)
        return out

    def hash_of(self, table: str, tbl):
        """The truncation priority of each row of table: the seeded hash of its key
        (primary key, else a unique key, else every column), or refuse."""
        cols = self.sql._sample_key(table)
        exprs = [tbl.c[c] for c in cols if c in tbl.c]
        h = self.sql._row_hash(exprs, self.seed)
        if h is None:
            raise PrivacyError(f"DP pushdown: no seeded hash function on {self.dialect} to bound the rows of {table}; "
                               "set synthesis.privacy.dp.pushdown: off")
        return h, exprs

    def stream(self, q, names: list) -> Iterator[pd.DataFrame]:
        """The result of q in chunks, columns renamed to names."""
        from ..data import normalise_frame
        with self.sql.engine.connect().execution_options(stream_results=True, max_row_buffer=CHUNK) as con:
            for chunk in pd.read_sql(q, con, chunksize=CHUNK):
                chunk.columns = list(names)
                yield normalise_frame(chunk)


def _coerce(df: pd.DataFrame, colspecs: dict) -> pd.DataFrame:
    """The columns of df named in colspecs cast as `load_tables` casts them (the
    in-memory path reads the same values)."""
    from ..data import coerce
    have = {c: cs for c, cs in colspecs.items() if c in df.columns}
    if not have:
        return df
    got = coerce(df[list(have)], {"columns": have})
    for c in have:
        df[c] = got[c].to_numpy() if len(df) else got[c]
    return df


class SqlRows(Rows):
    def __init__(self, src: SqlSource, table: str, sel):
        self.src, self.table, self.sel = src, table, sel
        self.tspec = src.spec["tables"][table]
        self._n: int | None = None
        self._deg: dict = {}

    @property
    def n(self):
        if self._n is None:
            import sqlalchemy as sa
            with self.src.sql.engine.connect() as con:
                self._n = int(con.execute(sa.select(sa.func.count()).select_from(self.sel)).scalar() or 0)
        return self._n

    def _freq(self, exprs: list, names: list):
        """SELECT exprs, COUNT(*) ... GROUP BY exprs, from a subquery so every
        dialect groups by plain columns."""
        import sqlalchemy as sa
        inner = sa.select(*[e.label(f"g{i}") for i, e in enumerate(exprs)]).subquery(f"rs_dp{next(_alias)}")
        cols = [inner.c[f"g{i}"] for i in range(len(exprs))]
        q = sa.select(*cols, sa.func.count().label("rs_w")).group_by(*cols)
        for chunk in self.src.stream(q, list(names) + ["rs_w"]):
            w = pd.to_numeric(chunk.pop("rs_w")).to_numpy(dtype=float)
            yield chunk, w

    def groups(self, cols):
        cols = list(cols)
        for chunk, w in self._freq([self.sel.c[c] for c in cols], cols):
            yield _coerce(chunk, {c: self.tspec["columns"][c] for c in cols}), w

    def _counts_of(self, child: str, ck: list):
        """rows of the whole child table per non-null key: (subquery, key columns, count column)."""
        import sqlalchemy as sa
        ct = self.src.sql.table(child)
        keys = [ct.c[c].label(f"k{i}") for i, c in enumerate(ck)]
        q = sa.select(*keys, sa.func.count().label("rs_n")).where(sa.and_(*[ct.c[c].is_not(None) for c in ck])) \
            .group_by(*[ct.c[c] for c in ck]).subquery(f"rs_dp{next(_alias)}")
        return q, [q.c[f"k{i}"] for i in range(len(ck))], q.c["rs_n"]

    def _degree_expr(self, frm, child, pk, ck, D=None):
        import sqlalchemy as sa
        cnt, keys, n = self._counts_of(child, ck)
        frm = frm.outerjoin(cnt, sa.and_(*[self.sel.c[p] == k for p, k in zip(pk, keys)]))
        d = sa.func.coalesce(n, 0)
        if D is not None:
            d = sa.case((d > int(D), int(D)), else_=d)
        return frm, d

    def degrees(self, child, pk, ck):
        import sqlalchemy as sa
        frm, d = self._degree_expr(self.sel, child, pk, ck)
        inner = sa.select(d.label("d")).select_from(frm).subquery(f"rs_dp{next(_alias)}")
        q = sa.select(inner.c["d"], sa.func.count().label("rs_w")).group_by(inner.c["d"])
        vals, ws = [], []
        for chunk in self.src.stream(q, ["d", "rs_w"]):
            vals.append(pd.to_numeric(chunk["d"]).to_numpy(dtype=np.int64))
            ws.append(pd.to_numeric(chunk["rs_w"]).to_numpy(dtype=float))
        if not vals:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        return np.concatenate(vals), np.concatenate(ws)

    def group_sizes(self, ck, g):
        import sqlalchemy as sa
        by = [self.sel.c[c] for c in ck] + [self.sel.c[g]]
        g1 = sa.select(*by, sa.func.count().label("s")).group_by(*by).subquery(f"rs_dp{next(_alias)}")
        q = sa.select(g1.c["s"], sa.func.count().label("rs_w")).group_by(g1.c["s"])
        vals, ws = [], []
        for chunk in self.src.stream(q, ["s", "rs_w"]):
            vals.append(pd.to_numeric(chunk["s"]).to_numpy(dtype=np.int64))
            ws.append(pd.to_numeric(chunk["rs_w"]).to_numpy(dtype=float))
        if not vals:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        return np.concatenate(vals), np.concatenate(ws)

    def bound(self, child, pk, ck, D):
        import sqlalchemy as sa
        ct = self.src.sql.table(child)
        cols = self.src._columns(child)
        h, hcols = self.src.hash_of(child, ct)
        # the parent row is kept (a correlated EXISTS works for composite keys on every dialect)
        par = sa.select(sa.literal(1)).select_from(self.sel) \
            .where(sa.and_(*[self.sel.c[p] == ct.c[c] for p, c in zip(pk, ck)]))
        rn = sa.func.row_number().over(partition_by=[ct.c[c] for c in ck], order_by=[h, *hcols]).label("rs_rn")
        inner = sa.select(*[ct.c[c] for c in cols], rn).where(sa.exists(par)).subquery(f"rs_dp{next(_alias)}")
        kept = sa.select(*[inner.c[c] for c in cols]).where(inner.c["rs_rn"] <= int(D)) \
            .subquery(f"rs_dp{next(_alias)}")
        return SqlRows(self.src, child, kept)

    def set_degree(self, name, child, pk, ck, D):
        self._deg[name] = (child, list(pk), list(ck), int(D))

    def joint(self, cols, degrees=(), parent=None):
        import sqlalchemy as sa
        cols, degrees = list(cols), list(degrees)
        frm = self.sel
        exprs, names = [self.sel.c[c] for c in cols], list(cols)
        for name in degrees:
            child, pk, ck, D = self._deg[name]
            frm, d = self._degree_expr(frm, child, pk, ck, D)
            exprs.append(d)
            names.append(name)
        pspec: dict = {}
        if parent is not None:
            prows, ck, pk, pcols = parent
            frm = frm.join(prows.sel, sa.and_(*[self.sel.c[c] == prows.sel.c[p] for c, p in zip(ck, pk)]))
            for c in pcols:
                exprs.append(prows.sel.c[c])
                names.append(f"^{c}")
                pspec[f"^{c}"] = prows.tspec["columns"][c]
        inner = sa.select(*[e.label(f"g{i}") for i, e in enumerate(exprs)]).select_from(frm) \
            .subquery(f"rs_dp{next(_alias)}")
        gcols = [inner.c[f"g{i}"] for i in range(len(exprs))]
        q = sa.select(*gcols, sa.func.count().label("rs_w")).group_by(*gcols)
        own = {c: self.tspec["columns"][c] for c in cols}
        for chunk in self.src.stream(q, names + ["rs_w"]):
            w = pd.to_numeric(chunk.pop("rs_w")).to_numpy(dtype=float)
            chunk = _coerce(chunk, own)
            chunk = _coerce(chunk, pspec)
            for name in degrees:
                chunk[name] = pd.to_numeric(chunk[name]).fillna(0).astype(np.int64)
            yield chunk, w


def source_for(spec: dict, mode: str, tables: dict | None = None):
    """The RowSource of a DP run: tables given -> in memory; mode 'on' -> in the
    database (refused when the source has no SQL); 'auto' -> in the database when
    it can be; 'off' -> loaded.  Returns (source, why) with why the reason a run
    with mode auto reads into memory (None otherwise)."""
    if tables is not None:
        from ..data import coerce
        real = {t: coerce(df, spec["tables"][t], optional=set()) if t in spec["tables"] else df
                for t, df in tables.items()}
        return FrameSource(real), "tables given in memory"
    if mode != "off":
        from ..data import source_connector
        conn = source_connector(spec)
        if sql_connector(conn) is not None:
            return SqlSource(spec, conn, owner=conn), None
        conn.close()
        if mode == "on":
            raise PrivacyError(f"synthesis.privacy.dp.pushdown: on needs a database source (or files with DuckDB); "
                               f"the source is {spec.get('source', {}).get('kind', 'csv')}; nothing was read")
        why = "the source has no SQL engine"
    else:
        why = "synthesis.privacy.dp.pushdown: off"
    from ..pushdown import load_for_fit
    return FrameSource(load_for_fit(spec, mode="off").tables), why
