# SPDX-License-Identifier: Apache-2.0
"""PostgreSQL: bulk loads with COPY (psycopg 3), identity sequences restarted
after the loaded keys; TimescaleDB hypertables read as tables and created in
targets (target.timescale)."""
from __future__ import annotations

import logging

import sqlalchemy as sa

from .sql import SqlConnector, frame_records

log = logging.getLogger("relsynth.connectors")


class PostgresConnector(SqlConnector):
    kind = "postgresql"

    def __init__(self, *a, timescale: dict | None = None, **kw):
        super().__init__(*a, **kw)
        self.timescale = timescale

    def _insert_rows(self, con, tbl, df, chunk=50_000):
        raw = con.connection.driver_connection
        mod = type(raw).__module__
        if not mod.startswith("psycopg") or mod.startswith("psycopg2"):
            return super()._insert_rows(con, tbl, df)
        from psycopg import sql
        cols = list(df.columns)
        name = sql.Identifier(*(([tbl.schema] if tbl.schema else []) + [tbl.name]))
        stmt = sql.SQL("COPY {} ({}) FROM STDIN").format(name, sql.SQL(", ").join(sql.Identifier(c) for c in cols))
        n = 0
        with raw.cursor() as cur:
            with cur.copy(stmt) as cp:
                for i in range(0, len(df), chunk):
                    for rec in frame_records(df.iloc[i:i + chunk], cols):
                        cp.write_row(rec)
                        n += 1
        return n

    def after_load(self, con, tbl):
        """Identity and serial columns continue after the largest loaded key."""
        for c in tbl.columns:
            if c.identity is None and not (c.autoincrement is True and c.server_default is not None):
                continue
            qual = con.dialect.identifier_preparer.format_table(tbl)
            seq = con.execute(sa.select(sa.func.pg_get_serial_sequence(qual, c.name))).scalar()
            if not seq:
                continue
            mx = con.execute(sa.select(sa.func.max(c))).scalar()
            if mx is not None:
                con.execute(sa.select(sa.func.setval(seq, int(mx), True)))

    # ------------------------------------------------------------ TimescaleDB
    # TimescaleDB is a PostgreSQL extension: its hypertables are ordinary tables
    # for reads (compressed chunks included), and a target table becomes a
    # hypertable when create_hypertable runs on it while it is still empty.
    def timescale_version(self) -> str | None:
        """The installed TimescaleDB extension version, or None."""
        if not hasattr(self, "_tsv"):
            try:
                with self.engine.connect() as con:
                    self._tsv = con.execute(sa.text(
                        "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")).scalar()
            except sa.exc.SQLAlchemyError:
                self._tsv = None
        return self._tsv

    def hypertables(self) -> dict:
        """{table: {time_column, chunk_interval, dimensions, compression}} of the
        schema's hypertables (empty without TimescaleDB)."""
        if not self.timescale_version():
            return {}
        q = sa.text("""SELECT h.hypertable_name, d.column_name, d.dimension_type,
                              d.time_interval::text AS interval, d.num_partitions, h.compression_enabled
                       FROM timescaledb_information.hypertables h
                       JOIN timescaledb_information.dimensions d
                         ON d.hypertable_schema = h.hypertable_schema AND d.hypertable_name = h.hypertable_name
                       WHERE h.hypertable_schema = :s
                       ORDER BY h.hypertable_name, d.dimension_number""")
        out: dict = {}
        with self.engine.connect() as con:
            for name, col, kind, interval, parts, comp in con.execute(q, {"s": self.schema or "public"}):
                h = out.setdefault(name, {"time_column": None, "chunk_interval": None, "dimensions": [],
                                          "compression": bool(comp)})
                h["dimensions"].append({"column": col, "type": str(kind).lower(),
                                        "interval": interval, "partitions": parts})
                if h["time_column"] is None and str(kind).lower() == "time":
                    h["time_column"], h["chunk_interval"] = col, interval
        return out

    def plan_hypertables(self, spec, names, source=None) -> dict:
        """{table: time column} to create as hypertables in this target:
        target.timescale.hypertables (auto: the source's hypertables when it is
        TimescaleDB, and tables whose series, sequence or fact time column is a
        datetime or date; none; or an explicit {table: column} map)."""
        opt = getattr(self, "timescale", None) or {}
        want = opt.get("hypertables", "auto")
        if want in (None, "none", False) or not self.timescale_version():
            return {}
        if isinstance(want, dict):
            return {t: c for t, c in want.items() if t in names}
        plan = {}
        if isinstance(source, PostgresConnector):
            for t, h in source.hypertables().items():
                if t in names and h["time_column"]:
                    plan[t] = h["time_column"]
        for t in names:
            T = spec["tables"].get(t, {})
            col = (T.get("series") or {}).get("time") or (T.get("sequence") or {}).get("time") \
                or (T.get("fact") or {}).get("time_column")
            if col and (T.get("columns", {}).get(col) or {}).get("type") in ("date", "datetime"):
                plan.setdefault(t, col)
        return plan

    def write_tables(self, spec, tables, mode="replace", structure="infer", source=None):
        self._hyper = self.plan_hypertables(spec, list(tables), source)
        return super().write_tables(spec, tables, mode=mode, structure=structure, source=source)

    def before_load(self, con, tbl):
        col = getattr(self, "_hyper", {}).get(tbl.name)
        if not col:
            return
        # a primary or unique key that leaves out the time column cannot exist on a
        # hypertable (each chunk enforces it alone): such tables stay plain tables
        keys = [set(c.name for c in k.columns) for k in tbl.constraints
                if isinstance(k, (sa.PrimaryKeyConstraint, sa.UniqueConstraint)) and len(k.columns)]
        if any(col not in k for k in keys):
            log.warning("%s: not a hypertable (a key without %s)", tbl.name, col)
            return
        qual = con.dialect.identifier_preparer.format_table(tbl)
        args = {"t": qual, "c": col}
        interval = (getattr(self, "timescale", None) or {}).get("chunk_interval")
        stmt = "SELECT create_hypertable(CAST(:t AS regclass), by_range(:c" + \
            (", CAST(:i AS interval)" if interval else "") + "), if_not_exists => TRUE)"
        if interval:
            args["i"] = str(interval)
        con.execute(sa.text(stmt), args)
        log.info("%s: TimescaleDB hypertable on %s", tbl.name, col)
