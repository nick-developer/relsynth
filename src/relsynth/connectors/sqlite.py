"""SQLite files."""
from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa

from .sql import SqlConnector
from .sqlfuncs import register_sqlite_functions


class SQLiteConnector(SqlConnector):
    """A SQLite database file.  Foreign keys are enforced on every connection, so a
    load that would break one fails; the statistics functions (normal scores, row
    hashes) are registered as SQL functions."""

    kind = "sqlite"

    def __init__(self, path: str | None = None, *, url: str | None = None, **kw):
        if url is None:
            if path is None:
                raise ValueError("sqlite needs a file path")
            url = f"sqlite:///{Path(path)}"
        kw.pop("schema", None)
        super().__init__(url, schema=None, **kw)

    def _setup_engine(self, engine):
        @sa.event.listens_for(engine, "connect")
        def _on_connect(dbapi_con, _rec):
            register_sqlite_functions(dbapi_con)
            cur = dbapi_con.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    def create_schema(self, schema=None):
        return None
