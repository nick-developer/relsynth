# SPDX-License-Identifier: Apache-2.0
"""DuckDB as the statistics engine of a folder of CSV and Parquet files.

A folder source loads its entities whole, but a pushed-down fact should not be
loaded at all (docs/pushdown.md).  Without a database, 1.0 computed a fact's
statistics in pandas, one column group at a time, re-reading the file for each
statistic.  `DuckDBFiles` instead opens an in-process DuckDB database (Raasveldt
and Mühleisen 2019) whose tables are views over the files, so the same SQL as
on the database servers (`SqlStats`: GROUP BY, window-function rank sums, first
rows, a seeded hash sample) runs over the files with DuckDB's streaming,
multi-threaded, out-of-core operators, reading only the columns a query needs.

Each view gives the values the pandas reader gives:

* Parquet: the stored types (decimals are read as decimals and normalised to
  floats, as in pandas);
* CSV: every field is read as text, with only the source's null tokens as nulls
  (an empty field by default), and a column becomes a number exactly when the
  pandas reader makes it one: every non-null value parses as a number and none
  has a leading zero (relsynth.data._parse_text_column).  Integers are BIGINT,
  other numbers DOUBLE; everything else stays text.

First rows follow the file's row order, as in memory: Parquet's
file_row_number, and for CSV the scan order, which DuckDB keeps
(preserve_insertion_order, its default).  The sample hash is the first 32 bits
of SHA-256('seed|row'), as on MySQL (relsynth.connectors.sqlfuncs).  Normal scores
use the exact quantile (scipy's ndtri as a vectorised DuckDB function), so the
statistics equal the pandas ones up to floating-point summation order.
"""
from __future__ import annotations

from typing import Any

import logging
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from ..data import normalise_frame
from .base import ColumnInfo, TableInfo, match_table
from .sql import SqlConnector, SqlStats

log = logging.getLogger("relsynth.connectors")

#: the DuckDB schema holding, per table, a view with the file row number
ROWS_SCHEMA = "rs_rows"
ROW_COLUMN = "__rs_row"
# the pandas reader's rules (relsynth.data): a number with a leading zero is a code
_LEADING_ZERO = r"^[+-]?0[0-9]"
_INTEGER = r"^[+-]?[0-9]+$"


def sa_type(duck_type: str):
    """The SQLAlchemy type of a DuckDB column type (tables are declared from
    DESCRIBE, not reflected: duckdb-engine's reflection reads PostgreSQL catalog
    tables that DuckDB does not have)."""
    t = duck_type.upper()
    if t.startswith("DECIMAL"):
        return sa.Numeric()
    if t in ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"):
        return sa.BigInteger()
    if t in ("FLOAT", "DOUBLE", "REAL"):
        return sa.Float()
    if t == "BOOLEAN":
        return sa.Boolean()
    if t == "DATE":
        return sa.Date()
    if t.startswith("TIMESTAMP"):
        return sa.DateTime(timezone="TIME ZONE" in t)
    if t == "BLOB":
        return sa.LargeBinary()
    if t == "VARCHAR":
        return sa.String()
    return sa.types.NullType()


def register_duckdb_functions(con) -> None:
    """rs_ndtri: the exact normal quantile (scipy), vectorised over Arrow batches, so
    normal scores over files equal the in-memory ones (no Acklam approximation)."""
    import duckdb
    import numpy as np
    import pyarrow as pa
    from scipy.special import ndtri

    def _ndtri(p):
        return pa.array(ndtri(np.asarray(p.to_numpy(zero_copy_only=False), dtype=float)))
    types: Any = getattr(duckdb, "sqltypes", None) or getattr(duckdb, "typing", None)  # renamed in DuckDB 1.4
    con.create_function("rs_ndtri", _ndtri, [types.DOUBLE], types.DOUBLE, type="arrow", side_effects=False)


def available() -> bool:
    import importlib.util
    return importlib.util.find_spec("duckdb") is not None and importlib.util.find_spec("duckdb_engine") is not None


def _lit(text: str) -> str:
    """A SQL string literal (DuckDB table functions take no bound parameters in a
    view).  Every statement built here (the lines marked S608 / B608) pastes only
    such literals and identifiers quoted by the dialect, never raw text."""
    return "'" + str(text).replace("'", "''") + "'"


class DuckDBFiles(SqlConnector):
    """The files of a FileConnector as views of an in-memory DuckDB database.

    settings: memory_limit ('4GB'), threads, temp_directory (spill folder for
    larger-than-memory sorts and groupings)."""

    kind = "duckdb"
    pushdown = True

    def __init__(self, files, settings: dict | None = None):
        self.files_conn = files
        self.settings = dict(settings or {})
        # an in-memory database lives as long as its connection: the statements that
        # build it are kept and replayed on every new connection (after close())
        self._ddl: list[str] = ["SET preserve_insertion_order = true",
                                f"CREATE SCHEMA IF NOT EXISTS {ROWS_SCHEMA}"]
        for k in ("memory_limit", "threads", "temp_directory"):
            if self.settings.get(k) is not None:
                v = self.settings[k]
                self._ddl.insert(0, f"SET {k} = {int(v) if k == 'threads' else _lit(v)}")
        engine = sa.create_engine("duckdb:///:memory:", poolclass=sa.pool.StaticPool)
        super().__init__(engine=engine)
        self._owns_engine = True
        self._views: dict[str, list[str]] = {}

    def _setup_engine(self, engine):
        @sa.event.listens_for(engine, "connect")
        def _on_connect(dbapi_con, _rec):
            register_duckdb_functions(dbapi_con)
            cur = dbapi_con.cursor()
            for stmt in self._ddl:
                cur.execute(stmt)
            cur.close()

    def _run(self, con, stmt: str) -> None:
        """Run a statement that builds the database, and keep it for replays."""
        con.exec_driver_sql(stmt)
        self._ddl.append(stmt)

    def location(self):
        return f"duckdb over {self.files_conn.location()}"

    # ------------------------------------------------------------ views
    def _q(self, name: str) -> str:
        return self.engine.dialect.identifier_preparer.quote_identifier(name)

    def _ensure(self, table: str) -> None:
        if table in self._views:
            return
        f = self.files_conn.file(table)
        q = self._q(table)
        with self.engine.begin() as con:
            if self.files_conn.is_parquet(f):
                src = f"read_parquet({_lit(f)}, file_row_number = true)"
                cols = [r[0] for r in con.exec_driver_sql(f"DESCRIBE SELECT * FROM read_parquet({_lit(f)})").fetchall()]  # noqa: S608  # nosec B608
                body = ", ".join(self._q(c) for c in cols)
                self._run(con, f"CREATE OR REPLACE VIEW {q} AS SELECT {body} FROM {src}")  # noqa: S608  # nosec B608
                self._run(con, f"CREATE OR REPLACE VIEW {ROWS_SCHEMA}.{q} AS SELECT {body}, "  # noqa: S608  # nosec B608
                               f"file_row_number AS {ROW_COLUMN} FROM {src}")
            else:
                cols = self._csv_view(con, table, f)
                self._run(con, f"CREATE OR REPLACE VIEW {ROWS_SCHEMA}.{q} AS SELECT *, "  # noqa: S608  # nosec B608
                               f"row_number() OVER () - 1 AS {ROW_COLUMN} FROM main.{q}")
        self._views[table] = cols

    def _csv_view(self, con, table: str, f: Path) -> list[str]:
        nv = [""] if self.files_conn.null_values is None else [str(x) for x in self.files_conn.null_values]
        # the header as pandas reads it (DuckDB's sniffer would rename or retype columns)
        cols = [str(c) for c in pd.read_csv(f, nrows=0).columns]
        spec = ", ".join(f"{_lit(c)}: 'VARCHAR'" for c in cols)
        raw = (f"read_csv({_lit(str(f))}, header = true, columns = {{{spec}}}, delim = ',', quote = '\"', "
               f"escape = '\"', nullstr = [{', '.join(_lit(v) for v in nv)}], auto_detect = false, "
               f"null_padding = true)")
        raw_cols = cols
        probes = []
        for i, rc in enumerate(raw_cols):
            x = f"trim({self._q(rc)})"
            probes += [f"count({self._q(rc)}) AS n{i}",
                       f"count(TRY_CAST({x} AS DOUBLE)) AS d{i}",
                       f"count(*) FILTER (WHERE regexp_full_match({x}, {_lit(_INTEGER)})) AS i{i}",
                       f"count(*) FILTER (WHERE regexp_matches({x}, {_lit(_LEADING_ZERO)})) AS z{i}"]
        row = con.exec_driver_sql(f"SELECT {', '.join(probes)} FROM {raw}").one()  # noqa: S608  # nosec B608
        sel = []
        for i, (rc, c) in enumerate(zip(raw_cols, cols)):
            n, d, ints, lead = row[4 * i:4 * i + 4]
            x = f"trim({self._q(rc)})"
            if n and d == n and not lead:
                ty = "BIGINT" if ints == n else "DOUBLE"
                sel.append(f"TRY_CAST({x} AS {ty}) AS {self._q(c)}")
            else:
                sel.append(f"{self._q(rc)} AS {self._q(c)}")
        self._run(con, f"CREATE OR REPLACE VIEW {self._q(table)} AS SELECT {', '.join(sel)} FROM {raw}")  # noqa: S608  # nosec B608
        return cols

    # ------------------------------------------------------------ catalog
    def tables(self, include=None, exclude=None):
        return [t for t in self.files_conn.tables() if match_table(t, include, exclude)]

    def has_table(self, table):
        return self.files_conn.has_table(table)

    def _columns(self, table: str, schema: str | None = None) -> list[tuple[str, str]]:
        self._ensure(table)
        name = f"{ROWS_SCHEMA}.{self._q(table)}" if schema else self._q(table)
        with self.engine.connect() as con:
            return [(str(r[0]), str(r[1])) for r in con.exec_driver_sql(f"DESCRIBE {name}").fetchall()]

    def table(self, name):
        if name not in self._tables:
            self._tables[name] = sa.Table(name, sa.MetaData(),
                                          *[sa.Column(c, sa_type(t)) for c, t in self._columns(name)])
        return self._tables[name]

    def rows_table(self, name) -> sa.Table:
        """The view with the file row number (column __rs_row)."""
        key = f"{ROWS_SCHEMA}.{name}"
        if key not in self._tables:
            self._tables[key] = sa.Table(name, sa.MetaData(schema=ROWS_SCHEMA),
                                         *[sa.Column(c, sa_type(t)) for c, t in self._columns(name, ROWS_SCHEMA)])
        return self._tables[key]

    def describe(self, table):
        return TableInfo(name=table, columns=[ColumnInfo(name=c, sql_type=t.lower()) for c, t in self._columns(table)])

    def count(self, table):
        if table not in self._counts:
            f = self.files_conn.file(table)
            if self.files_conn.is_parquet(f):
                self._counts[table] = self.files_conn.count(table)        # Parquet metadata
            else:
                self._ensure(table)
                with self.engine.connect() as con:
                    self._counts[table] = int(con.exec_driver_sql(f"SELECT count(*) FROM {self._q(table)}").scalar())  # noqa: S608  # nosec B608
        return self._counts[table]

    def _sample_key(self, table, key=None):
        return list(key) if key else [c.name for c in self.describe(table).columns]

    def sample(self, table, n, seed=7, key=None):
        df, whole = super().sample(table, n, seed=seed, key=key)
        if whole:          # the whole table: the pandas reader's exact values and order
            return self.files_conn.read_table(table), True
        return df, whole

    def read_table(self, table, columns=None):
        return self.files_conn.read_table(table, columns)

    def stats(self, table):
        return DuckDBStats(self, table)

    def frame(self, q):
        """Fetch through DuckDB's columnar result (Arrow / NumPy) instead of Python
        row tuples: grouped results of a large fact have millions of rows."""
        c = q.compile(dialect=self.engine.dialect, compile_kwargs={"render_postcompile": True})
        params = [c.params[k] for k in (c.positiontup or [])]
        with self.engine.connect() as con:
            # duckdb_engine's wrapper passes execute to the DuckDB connection
            return con.connection.driver_connection.execute(str(c), params).df()

    # ------------------------------------------------------------ writes
    def write_table(self, table, df, mode="replace", tspec=None):
        raise NotImplementedError("DuckDBFiles reads files; write through the folder's FileConnector")

    def write_tables(self, spec, tables, mode="replace", structure="infer", source=None):
        raise NotImplementedError("DuckDBFiles reads files; write through the folder's FileConnector")

    def refresh(self):
        super().refresh()
        self._views = {}


class DuckDBStats(SqlStats):
    """SqlStats over a file view; first rows in file order."""

    def first_rows(self, key, cols, order=None, differences=True):
        base = self.tbl
        self.tbl = self.conn.rows_table(self.table)
        try:
            return super().first_rows(key, cols, order, differences)
        finally:
            self.tbl = base

    def _row_order(self, tbl):
        return [tbl.c[ROW_COLUMN]] if ROW_COLUMN in tbl.c else super()._row_order(tbl)

    def distinct(self, col):
        out = super().distinct(col)
        return normalise_frame(pd.DataFrame({col: out}))[col]
