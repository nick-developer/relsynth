# SPDX-License-Identifier: Apache-2.0
"""SQL databases through SQLAlchemy: SQLite, PostgreSQL, Microsoft SQL Server,
Oracle (and, untested, any other SQLAlchemy dialect).

`SqlConnector` holds one engine (a connection pool) and the reflected tables.
Dialect-specific behaviour lives in small subclasses (sqlite.py, postgres.py,
mssql.py, oracle.py): bulk loads, sampling and identity columns.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Iterator

import numpy as np
import pandas as pd
import sqlalchemy as sa

from ..data import mask_url, normalise_frame
from ..pushdown.stats import SCORE_PREFIX, SQUARE_PREFIX, Require, TableStats
from .base import WRITE_MODES, ColumnInfo, Connector, ForeignKey, TableInfo, match_table
from .sqlfuncs import ndtri, row_hash, to_float

log = logging.getLogger("relsynth.connectors")

#: tables with more rows than this are read in chunks (bounded driver memory)
STREAM_ROWS = 500_000
CHUNK_ROWS = 100_000
#: below this many rows `sample` reads the whole table
WHOLE_TABLE_ROWS = 0


def is_binary_type(t) -> bool:
    """Binary SQL types (never read: reported as binary and left empty)."""
    from ..source import is_binary_sqltype
    return is_binary_sqltype(str(t))


def resolve_url(url: str | None = None, url_env: str | None = None, password_env: str | None = None) -> str:
    """The database URL from the specification and the environment.

    * url_env names an environment variable holding the complete URL (used when
      the URL in the file is missing or masked with ***);
    * password_env names a variable holding only the password, inserted into the URL.
    Shorthand driver names are completed: postgresql:// -> postgresql+psycopg,
    mssql:// -> mssql+pyodbc (or +pymssql when pyodbc or its ODBC driver is missing),
    oracle:// -> oracle+oracledb, mysql:// and mariadb:// -> +mysqldb when
    mysqlclient is installed, else +pymysql (with charset=utf8mb4)."""
    if url_env:
        v = os.environ.get(url_env)
        if v:
            url = v
        elif not url or "***" in str(url):
            raise ValueError(f"set the environment variable {url_env} to the database URL (with its password); "
                             "the URL in the specification is not complete")
    if not url:
        raise ValueError("no database URL given")
    u = sa.engine.make_url(str(url))
    if password_env:
        pw = os.environ.get(password_env)
        if pw is None:
            raise ValueError(f"set the environment variable {password_env} to the database password")
        u = u.set(password=pw)
    u = u.set(drivername=_driver(u.drivername))
    if u.get_backend_name() in ("mysql", "mariadb") and "charset" not in u.query:
        u = u.update_query_dict({"charset": "utf8mb4"})   # the server default may be latin1 or 3-byte utf8
    return u.render_as_string(hide_password=False)


def _has(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def _driver(name: str) -> str:
    if "+" in name:
        return name
    if name in ("postgresql", "postgres"):
        return "postgresql+psycopg" if _has("psycopg") else "postgresql+psycopg2"
    if name == "mssql":
        if _has("pyodbc"):
            try:
                import pyodbc
                if any("SQL Server" in d for d in pyodbc.drivers()):
                    return "mssql+pyodbc"
            except Exception as e:  # noqa: BLE001 - no ODBC driver manager
                log.debug("pyodbc unavailable (%s); trying pymssql", e)
        return "mssql+pymssql" if _has("pymssql") else "mssql+pyodbc"
    if name == "oracle":
        return "oracle+oracledb"
    if name in ("mysql", "mariadb"):
        return f"{name}+mysqldb" if _has("MySQLdb") else f"{name}+pymysql"
    return name


def py_value(v):
    """A pandas / numpy value as the plain Python value a DBAPI driver binds."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, np.generic):
        v = v.item()
        return None if isinstance(v, float) and v != v else v
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def frame_records(df: pd.DataFrame, cols: list[str]) -> list[tuple]:
    """Rows of df as tuples of Python values (nulls as None)."""
    parts = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            arr = np.array([None if pd.isna(x) else x.to_pydatetime() for x in s], dtype=object)
        elif pd.api.types.is_bool_dtype(s) and not pd.api.types.is_object_dtype(s):
            arr = np.array([None if pd.isna(x) else bool(x) for x in s], dtype=object)
        elif pd.api.types.is_integer_dtype(s):
            arr = np.array([None if pd.isna(x) else int(x) for x in s], dtype=object)
        elif pd.api.types.is_float_dtype(s):
            v = s.to_numpy(dtype=float)
            arr = np.array([None if x != x else float(x) for x in v], dtype=object)
        else:
            arr = np.array([py_value(x) for x in s.to_numpy(dtype=object)], dtype=object)
        parts.append(arr)
    return list(zip(*parts)) if parts else [()] * len(df)


class SqlConnector(Connector):
    """A schema of a SQL database."""

    kind = "sql"
    pushdown = True

    def __init__(self, url: str | None = None, schema: str | None = None, *, engine: sa.Engine | None = None,
                 url_env: str | None = None, password_env: str | None = None,
                 connect_args: dict | None = None, engine_options: dict | None = None):
        self.schema = schema or None
        if engine is None:
            full = resolve_url(url, url_env, password_env)
            opts = {"pool_pre_ping": True, **self._engine_options(full), **(engine_options or {})}
            if connect_args:
                opts["connect_args"] = {**opts.get("connect_args", {}), **connect_args}
            engine = sa.create_engine(full, **opts)
            self._owns_engine = True
        else:
            self._owns_engine = False
        self.engine = engine
        self.dialect = engine.dialect.name
        self._setup_engine(engine)
        self._md = sa.MetaData(schema=self.schema)
        self._tables: dict[str, sa.Table] = {}
        self._counts: dict[str, int] = {}
        self._insp = None

    # ------------------------------------------------------------ hooks
    def _engine_options(self, url: str) -> dict:
        return {}

    def _setup_engine(self, engine) -> None:
        """Dialect hooks (event listeners) on a new engine."""

    def location(self) -> str:
        loc = mask_url(self.engine.url.render_as_string(hide_password=False))
        return f"{loc} schema={self.schema}" if self.schema else loc

    def close(self) -> None:
        if self._owns_engine:
            self.engine.dispose()

    # ------------------------------------------------------------ catalog
    @property
    def inspector(self):
        if self._insp is None:
            self._insp = sa.inspect(self.engine)
        return self._insp

    def refresh(self) -> None:
        """Forget cached reflection (after the database changed)."""
        self._insp = None
        self._md = sa.MetaData(schema=self.schema)
        self._tables.clear()
        self._counts.clear()

    def tables(self, include=None, exclude=None):
        names = self.inspector.get_table_names(schema=self.schema)
        return [str(t) for t in names if match_table(str(t), include, exclude)]

    def has_table(self, table):
        return sa.inspect(self.engine).has_table(table, schema=self.schema)

    def table(self, name: str) -> sa.Table:
        """The reflected table (quoted by the dialect in every statement)."""
        if name not in self._tables:
            self._tables[name] = sa.Table(name, self._md, autoload_with=self.engine)
        return self._tables[name]

    def describe(self, table):
        insp = self.inspector
        cols = []
        for c in insp.get_columns(table, schema=self.schema):
            ty = c["type"]
            cols.append(ColumnInfo(name=str(c["name"]), sql_type=str(ty).lower(), nullable=bool(c.get("nullable", True)),
                                   identity=bool(c.get("identity")),
                                   length=getattr(ty, "length", None)))
        pk = [str(c) for c in (insp.get_pk_constraint(table, schema=self.schema) or {}).get("constrained_columns") or []]
        fks = []
        for fk in insp.get_foreign_keys(table, schema=self.schema):
            if fk.get("referred_table") is None:
                continue
            fks.append(ForeignKey([str(c) for c in fk["constrained_columns"]], str(fk["referred_table"]),
                                  [str(c) for c in fk["referred_columns"]], fk.get("name")))
        uniques = []
        for get in (insp.get_unique_constraints, insp.get_indexes):
            try:
                for u in get(table, schema=self.schema):
                    cn = [str(c) for c in (u.get("column_names") or []) if c is not None]
                    if get is insp.get_indexes and not u.get("unique"):
                        continue
                    if cn and cn != pk and cn not in uniques and len(cn) == len(u.get("column_names") or []):
                        uniques.append(cn)
            except NotImplementedError:
                pass
        return TableInfo(name=table, columns=cols, primary_key=pk, foreign_keys=fks, unique=uniques)

    def count(self, table):
        if table not in self._counts:
            with self.engine.connect() as con:
                self._counts[table] = int(con.execute(sa.select(sa.func.count()).select_from(self.table(table))).scalar())
        return self._counts[table]

    def binary_columns(self, table) -> list[str]:
        return [c.name for c in self.table(table).columns if is_binary_type(c.type)]

    # ------------------------------------------------------------ reads
    def _select(self, table, columns=None, binary=True):
        tbl = self.table(table)
        cols = list(tbl.columns) if columns is None else [tbl.c[c] for c in columns]
        if not binary:
            cols = [c for c in cols if not is_binary_type(c.type)]
        return sa.select(*cols)

    def read_table(self, table, columns=None):
        """SELECT the columns (all by default), normalised.  Large tables are
        streamed in chunks so the driver never holds the whole result twice."""
        q = self._select(table, columns)
        if self.count(table) <= STREAM_ROWS:
            with self.engine.connect() as con:
                df = pd.read_sql(q, con)
        else:
            parts = list(self.iter_table(table, columns))
            df = pd.concat(parts, ignore_index=True) if parts else pd.read_sql(q.limit(0), self.engine)
            df = df.infer_objects()
        df.columns = [str(c) for c in df.columns]
        return normalise_frame(df)

    def iter_table(self, table, columns=None, chunksize=CHUNK_ROWS) -> Iterator[pd.DataFrame]:
        q = self._select(table, columns)
        with self.engine.connect().execution_options(stream_results=True, max_row_buffer=chunksize) as con:
            for chunk in pd.read_sql(q, con, chunksize=chunksize):
                chunk.columns = [str(c) for c in chunk.columns]
                yield chunk

    def _sample_key(self, table, key=None) -> list[str]:
        info = self.describe(table)
        if key:
            return list(key)
        if info.primary_key:
            return info.primary_key
        if info.unique:
            return info.unique[0]
        return [c.name for c in info.columns if not is_binary_type(c.sql_type)]

    def sample(self, table, n, seed=7, key=None):
        """A Bernoulli sample of about n rows: a row is kept when a seeded hash of
        its key falls below n / rows (reproducible for the same data and seed;
        every row has the same chance).  The whole table when it has at most n rows."""
        total = self.count(table)
        if total <= max(n, WHOLE_TABLE_ROWS):
            return self.read_table(table), True
        tbl = self.table(table)
        cols = [tbl.c[c] for c in self._sample_key(table, key)]
        h = row_hash(cols, seed, self.dialect)
        q = self._select(table)
        if h is None:
            log.warning("%s: no seeded hash function for dialect %s; sampling in random order (not reproducible)",
                        table, self.dialect)
            q = q.order_by(sa.func.random()).limit(n)
        else:
            from .sqlfuncs import HASH_RANGE
            q = q.where(h < int(round(HASH_RANGE * n / total)))
        df = self.frame(q)
        df.columns = [str(c) for c in df.columns]
        if h is not None:
            # the sample size is Binomial(total, n / total): far outside it means the
            # hash is not uniform on this backend (a driver or collation problem)
            p = min(n / total, 1.0)
            sd = (total * p * (1 - p)) ** 0.5
            if abs(len(df) - n) > 8 * sd + 10:
                raise RuntimeError(f"{table}: sampling by hash returned {len(df):,} rows where about {n:,} "
                                   f"were expected; the seeded hash is not uniform on {self.dialect}")
        return normalise_frame(df), False

    # ------------------------------------------------------------ statistics
    def stats(self, table):
        return SqlStats(self, table)

    # ------------------------------------------------------------ writes
    def create_schema(self, schema: str | None = None) -> None:
        """CREATE SCHEMA when it does not exist (PostgreSQL, SQL Server)."""
        schema = schema or self.schema
        if not schema:
            return
        insp = sa.inspect(self.engine)
        if schema in insp.get_schema_names():
            return
        with self.engine.begin() as con:
            con.execute(sa.schema.CreateSchema(schema))

    def write_table(self, table, df, mode="replace", tspec=None):
        """Write one table: replace (drop and create from the data's types and
        tspec's keys), append or truncate (into the existing table, created when
        missing), fail (stop when it exists)."""
        if mode not in WRITE_MODES:
            raise ValueError(f"mode must be one of {', '.join(WRITE_MODES)}")
        ts = {"columns": {}, **(tspec or {})}
        self.write_tables({"tables": {table: ts}, "relationships": []}, {table: df}, mode=mode)
        return len(df)

    def bulk_insert(self, con, tbl: sa.Table, df: pd.DataFrame) -> int:
        """Insert the rows of df, converted to the target column types."""
        from .ddl_types import prepare_frame
        return self._insert_rows(con, tbl, prepare_frame(df, tbl))

    def _insert_rows(self, con, tbl: sa.Table, df: pd.DataFrame, chunk: int = 20_000) -> int:
        """executemany in batches (subclasses use the vendor's bulk path)."""
        cols = list(df.columns)
        stmt = tbl.insert()
        n = 0
        for i in range(0, len(df), chunk):
            part = df.iloc[i:i + chunk]
            recs = [dict(zip(cols, r)) for r in frame_records(part, cols)]
            if recs:
                con.execute(stmt, recs)
                n += len(recs)
        return n

    def before_load(self, con, tbl: sa.Table) -> None:
        """Hook run in the load transaction before rows are inserted (identity columns)."""

    def after_load(self, con, tbl: sa.Table) -> None:
        """Hook run after rows are inserted (identity sequences restarted after the keys)."""

    def write_tables(self, spec, tables, mode="replace", structure="infer", source=None):
        """Write the synthetic tables, parents first, in one transaction per table.

        structure:
          * clone: the target tables are created with the source's DDL (types,
            nullability, primary, unique and foreign keys, identity columns), when
            the source is a SQL database; otherwise as infer;
          * infer: created from the specification (key and link constraints) and
            the column types of the data;
          * existing: the tables exist already (mode append or truncate).
        mode: replace (drop and create), append, truncate (delete rows, children
        first, then insert), fail (stop when a table exists)."""
        from . import ddl
        if mode not in WRITE_MODES:
            raise ValueError(f"mode must be one of {', '.join(WRITE_MODES)}")
        names = list(tables)
        order = ddl.load_order(spec, names)
        exists = {t for t in names if self.has_table(t)}
        if mode == "fail" and exists:
            raise FileExistsError(f"target tables exist: {', '.join(sorted(exists))} (if_exists: fail)")
        if structure == "existing" and mode == "replace":
            mode = "truncate"
        if mode == "replace" or not exists:
            if structure == "clone" and isinstance(source, SqlConnector):
                md = ddl.clone_metadata(source, names, self.dialect, self.schema)
            else:
                md = ddl.metadata_from_spec(spec, tables, self.dialect, self.schema)
            self.create_schema()
            ddl.create_tables(self.engine, md, names, replace=(mode == "replace"), schema=self.schema)
            self.refresh()
        elif mode == "truncate":
            with self.engine.begin() as con:
                for t in reversed(order):
                    if t in exists:
                        con.execute(sa.delete(sa.Table(t, sa.MetaData(schema=self.schema), autoload_with=con)))
        md = sa.MetaData(schema=self.schema)
        for t in order:
            tbl = sa.Table(t, md, autoload_with=self.engine)
            df = ddl.rows_parents_first(spec, t, tables[t])
            cols = [c for c in df.columns if c in tbl.c]
            missing = [c for c in df.columns if c not in tbl.c and not c.startswith("__")]
            if missing:
                log.warning("%s: columns not in the target table, not written: %s", t, ", ".join(missing))
            with self.engine.begin() as con:
                self.before_load(con, tbl)
                self.bulk_insert(con, tbl, df[cols])
                self.after_load(con, tbl)
            log.info("wrote %s: %s rows", t, f"{len(df):,}")
        self.refresh()
        return self.location()

    def frame(self, q) -> pd.DataFrame:
        """The result of a SELECT as a frame (subclasses use a faster fetch path)."""
        with self.engine.connect() as con:
            return pd.read_sql(q, con)

    def ping(self) -> dict[str, Any]:
        out = super().ping()
        if out["ok"]:
            out["server"] = ".".join(map(str, self.engine.dialect.server_version_info or ()))
            out["dialect"] = self.dialect
        return out


def _plain_values(values) -> list:
    """Distinct non-null driver values of a key list."""
    vals = (py_value(v) for v in pd.unique(pd.Series(list(values), dtype=object)))
    return [v for v in vals if v is not None]


class SqlStats(TableStats):
    """TableStats computed in the database (GROUP BY and window functions)."""

    def __init__(self, conn: SqlConnector, table: str):
        self.conn = conn
        self.table = table
        self.tbl = conn.table(table)
        self.n_rows = conn.count(table)

    def col(self, name):
        """The column as compared, grouped and filtered (dialects with
        case-insensitive default collations compare text exactly instead)."""
        return self.tbl.c[name]

    def _scalar(self, q):
        with self.conn.engine.connect() as con:
            return con.execute(q).scalar()

    def notnull(self, col):
        return int(self._scalar(sa.select(sa.func.count(self.tbl.c[col]))))

    def distinct(self, col):
        x = self.col(col)
        q = sa.select(x.label(col)).where(x.is_not(None)).distinct()
        df = self.conn.frame(q)
        df.columns = [col]
        return normalise_frame(df)[col].reset_index(drop=True)

    def _score(self, x):
        """Normal score of x among the column's non-null values (0 for null): the
        average rank r = min rank + (ties - 1) / 2 within the non-null partition,
        Phi^-1(r / (N + 1))."""
        isnull = sa.case((x.is_(None), 1), else_=0)
        rmin = sa.func.rank().over(partition_by=isnull, order_by=x)
        ties = sa.func.count(sa.literal(1)).over(partition_by=x)
        nn = sa.func.count(x).over()
        p = (to_float(rmin) + (to_float(ties) - 1.0) / 2.0) / (to_float(nn) + 1.0)
        return isnull, p

    # parent key lists longer than this are applied after the query, on the grouped rows
    MAX_IN_LIST = 2_000

    def _require(self, col, rq: Require):
        """SQL filter of rows whose col is one of the parent's keys, or None when the
        filter must be applied after the query (a long list of allowed values)."""
        x = self.col(col)
        conds = [x.is_not(None)]
        if rq.table is not None and self.conn.has_table(rq.table):
            par = self.conn.table(rq.table)
            conds.append(x.in_(sa.select(par.c[rq.column]).where(par.c[rq.column].is_not(None))))
        elif rq.values is not None:
            vals = _plain_values(rq.values)
            if len(vals) > self.MAX_IN_LIST:
                return None
            if not vals:
                return sa.false()
            conds.append(x.in_(vals))
        ex = _plain_values(rq.exclude)
        if ex:
            if len(ex) > self.MAX_IN_LIST:
                return None
            conds.append(x.not_in(ex))
        return sa.and_(*conds)

    def minmax(self, col):
        c = self.tbl.c[col]
        with self.conn.engine.connect() as con:
            lo, hi = con.execute(sa.select(sa.func.min(c), sa.func.max(c))).one()
        return py_value(lo), py_value(hi)

    def _row_order(self, tbl) -> list:
        """Columns that order a key's rows after `order`: the primary key."""
        info = self.conn.describe(self.table)
        return [tbl.c[c] for c in (info.primary_key or []) if c in tbl.c]

    def first_rows(self, key, cols, order=None, differences=True):
        keys = [key] if isinstance(key, str) else list(key)
        cols = [c for c in cols if c not in keys]
        tbl = self.tbl
        # nulls last, written portably (SQL Server has no NULLS LAST): the null flag, then the value
        ob = [x for c in (order or []) for x in (sa.case((tbl.c[c].is_(None), 1), else_=0), tbl.c[c].asc())]
        ob += self._row_order(tbl) or [tbl.c[k] for k in keys]
        rn = sa.func.row_number().over(partition_by=[self.col(k) for k in keys], order_by=ob).label("rs_rn")
        inner = sa.select(*[tbl.c[k] for k in keys], *[tbl.c[c] for c in cols], rn).subquery("rs_h")
        q = sa.select(*[inner.c[c] for c in keys + cols]).where(inner.c["rs_rn"] == 1) \
            .order_by(*[inner.c[k] for k in keys])
        H = self.conn.frame(q)
        H.columns = keys + cols
        H = normalise_frame(H)
        diff = None
        if cols and differences:
            # cells that differ from the key's first row (null-safe)
            h = q.subquery("rs_first")
            terms = [sa.case((sa.or_(self.col(c) == h.c[c], sa.and_(tbl.c[c].is_(None), h.c[c].is_(None))), 0),
                             else_=1) for c in cols]
            on = sa.and_(*[self.col(k) == h.c[k] for k in keys])
            dq = sa.select(sa.func.sum(sum(terms[1:], terms[0]))).select_from(tbl.join(h, on))
            try:
                with self.conn.engine.connect() as con:
                    diff = int(con.execute(dq).scalar() or 0)
            except sa.exc.DBAPIError as e:      # e.g. LOB columns cannot be compared
                log.warning("%s: header consistency not computed in the database (%s)", self.table,
                            str(e).splitlines()[0][:200])
        elif not cols:
            diff = 0
        return H, diff

    def group_counts(self, cols, scores=(), require=None):
        cols, scores = list(cols), list(scores)
        tbl = self.tbl
        inner_cols = [self.col(c).label(f"k{i}") for i, c in enumerate(cols)]
        pcols = []
        for j, s in enumerate(scores):
            isnull, p = self._score(tbl.c[s])
            pcols += [isnull.label(f"n{j}"), p.label(f"p{j}")]
        keep, late = [], {}
        for c, rq in (require or {}).items():
            cond = self._require(c, rq)
            if cond is None:
                late[c] = rq
            else:
                keep.append(cond)
        if late:
            # group by the filtered columns too, filter the groups, then sum them up
            extra = [c for c in late if c not in cols]
            res = self.group_counts(cols + extra, scores,
                                    {c: rq for c, rq in (require or {}).items() if c not in late})
            from ..prepare import _keystr
            ok = np.ones(len(res), bool)
            for c, rq in late.items():
                allowed = set(_keystr(pd.Series(list(rq.values), dtype=object)).dropna()) \
                    - set(_keystr(pd.Series(list(rq.exclude), dtype=object)).dropna())
                ok &= _keystr(res[c]).isin(allowed).to_numpy()
            res = res[ok]
            if not extra:
                return res.reset_index(drop=True)
            out = res.drop(columns=extra).groupby(cols, dropna=False, sort=False).sum(numeric_only=True).reset_index()
            out["n"] = out["n"].astype(np.int64)
            return out
        if keep:
            # ranks are taken over every row (as in memory), the filter applies after
            flag = sa.case((sa.and_(*keep), 1), else_=0).label("rs_keep")
            inner = sa.select(*inner_cols, *pcols, flag).subquery("rs_p")
        else:
            inner = sa.select(*inner_cols, *pcols).subquery("rs_p")
        zcols = []
        for j in range(len(scores)):
            z = sa.case((inner.c[f"n{j}"] == 1, sa.literal(0.0, sa.Float)),
                        else_=ndtri(inner.c[f"p{j}"], self.conn.dialect))
            zcols.append(z.label(f"z{j}"))
        mid_q = sa.select(*[inner.c[f"k{i}"] for i in range(len(cols))], *zcols)
        if keep:
            mid_q = mid_q.where(inner.c["rs_keep"] == 1)
        mid = mid_q.subquery("rs_z")
        keys = [mid.c[f"k{i}"] for i in range(len(cols))]
        aggs = [sa.func.count(sa.literal(1)).label("n")]
        for j in range(len(scores)):
            z = mid.c[f"z{j}"]
            aggs += [sa.func.sum(z).label(f"{SCORE_PREFIX}{j}"), sa.func.sum(z * z).label(f"{SQUARE_PREFIX}{j}")]
        q = sa.select(*keys, *aggs).group_by(*keys)
        df = self.conn.frame(q)
        df.columns = cols + [c for c in df.columns[len(cols):]]
        df = normalise_frame(df)
        df["n"] = df["n"].astype(np.int64)
        for c in df.columns[len(cols) + 1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        return df
