# SPDX-License-Identifier: Apache-2.0
"""Microsoft SQL Server and Azure SQL.

Loads use ODBC array binding (pyodbc fast_executemany) with the Microsoft ODBC
driver, the recommended setup; with pymssql they use multi-row INSERT batches
(about 5-10x slower).  Explicit values are loaded into identity columns with
IDENTITY_INSERT, and SQL Server continues the identity after the largest one."""
from __future__ import annotations

import sqlalchemy as sa

from .sql import SqlConnector, frame_records


class MSSQLConnector(SqlConnector):
    kind = "mssql"

    def _engine_options(self, url):
        if sa.engine.make_url(url).get_driver_name() == "pyodbc":
            # one round trip per batch of parameter arrays (ODBC array binding)
            return {"fast_executemany": True, "use_insertmanyvalues": False}
        return {}

    def _insert_rows(self, con, tbl, df, chunk=20_000):
        cols = list(df.columns)
        if self.engine.dialect.driver == "pyodbc" or not cols:
            return super()._insert_rows(con, tbl, df, chunk)
        # one INSERT ... VALUES (...), (...) per batch: at most 1000 rows and 2100
        # bound parameters per statement
        per = max(1, min(1000, 2000 // len(cols)))
        n = 0
        for i in range(0, len(df), per):
            recs = [dict(zip(cols, r)) for r in frame_records(df.iloc[i:i + per], cols)]
            if recs:
                con.execute(tbl.insert().values(recs))
                n += len(recs)
        return n

    def _identity(self, tbl):
        return any(c.identity is not None for c in tbl.columns)

    def before_load(self, con, tbl):
        if self._identity(tbl):
            con.exec_driver_sql(f"SET IDENTITY_INSERT {con.dialect.identifier_preparer.format_table(tbl)} ON")

    def after_load(self, con, tbl):
        if self._identity(tbl):
            con.exec_driver_sql(f"SET IDENTITY_INSERT {con.dialect.identifier_preparer.format_table(tbl)} OFF")
