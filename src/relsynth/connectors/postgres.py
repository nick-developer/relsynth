# SPDX-License-Identifier: Apache-2.0
"""PostgreSQL: bulk loads with COPY (psycopg 3), identity sequences restarted
after the loaded keys."""
from __future__ import annotations

import logging

import sqlalchemy as sa

from .sql import SqlConnector, frame_records

log = logging.getLogger("relsynth.connectors")


class PostgresConnector(SqlConnector):
    kind = "postgresql"

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
