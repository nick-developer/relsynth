"""MySQL 8 and MariaDB 10.6+ (PyMySQL, or mysqlclient when installed).

A MySQL "schema" is a database: `schema` names the database to read or write,
and a missing target database is created (CREATE SCHEMA is a synonym of
CREATE DATABASE).

Three MySQL defaults would change the data or the statistics, so the connector
sets them explicitly:

* text comparison: the default collations (utf8mb4_0900_ai_ci on MySQL 8,
  utf8mb4_uca1400_ai_ci on MariaDB 11) ignore case, accents and, on MariaDB,
  trailing spaces, so GROUP BY, DISTINCT and the parent-key filters would merge
  'abc', 'ABC' and 'abc '.  Statistics compare text with the binary no-pad
  collation of the column's character set, and tables relsynth creates use
  that collation, so unique keys hold on the exact values;
* FLOAT is single precision in MySQL: created float columns are DOUBLE;
* DATETIME keeps whole seconds by default: created columns are DATETIME(6).

Sessions run with time_zone '+00:00' (TIMESTAMP values are read as stored) and
the client character set utf8mb4.  Loads use the driver's executemany, which
sends multi-row INSERT statements; LOAD DATA LOCAL INFILE is not used because
servers disable it by default (a client-side file read the server requests).
AUTO_INCREMENT columns continue after the largest loaded key on their own.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa

from .sql import SqlConnector, SqlStats

log = logging.getLogger("relsynth.connectors")

#: binary collations without trailing-space padding, by server family
NOPAD_BIN = {"mysql": "utf8mb4_0900_bin", "mariadb": "utf8mb4_nopad_bin"}


def is_mysql_family(dialect: str) -> bool:
    return dialect in ("mysql", "mariadb")


class MySQLConnector(SqlConnector):
    """A database of a MySQL 8 or MariaDB server."""

    kind = "mysql"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # mysql+pymysql:// on a MariaDB server reports the dialect name "mysql"; the
        # server tells which it is on the first connection
        if self.engine.dialect.name != "mariadb":
            try:
                with self.engine.connect():
                    pass
            except sa.exc.DBAPIError as e:      # unreachable now: ping and reads report it
                log.debug("mysql: no connection yet (%s)", type(e).__name__)
        self.dialect = "mariadb" if getattr(self.engine.dialect, "is_mariadb", False) else "mysql"
        self.kind = self.dialect
        self._collations: dict[str, dict[str, str]] = {}

    def _engine_options(self, url):
        # servers close idle connections after wait_timeout (8 hours by default)
        return {"pool_recycle": 3600}

    def _setup_engine(self, engine):
        @sa.event.listens_for(engine, "connect")
        def _on_connect(dbapi_con, _rec):
            cur = dbapi_con.cursor()
            cur.execute("SET time_zone = '+00:00'")
            cur.close()

    @property
    def text_collation(self) -> str:
        """The binary no-pad collation used for created tables and for comparisons."""
        return NOPAD_BIN[self.dialect]

    def create_schema(self, schema=None):
        schema = schema or self.schema
        if not schema:
            return
        if schema in sa.inspect(self.engine).get_schema_names():
            return
        prep = self.engine.dialect.identifier_preparer
        with self.engine.begin() as con:
            # the identifier is quoted by the dialect; the collation is one of two constants
            con.exec_driver_sql(f"CREATE DATABASE {prep.quote_identifier(schema)} "
                                f"CHARACTER SET utf8mb4 COLLATE {self.text_collation}")

    def table(self, name):
        tbl = super().table(name)
        for c in tbl.columns:
            # DOUBLE and FLOAT reflect as asdecimal=True, which rounds to 10 decimals
            if isinstance(c.type, sa.Float) and getattr(c.type, "asdecimal", False):
                c.type.asdecimal = False
        return tbl

    def column_collations(self, table: str) -> dict[str, str]:
        """Collation of every text column of a table (information_schema)."""
        if table not in self._collations:
            q = sa.text("SELECT COLUMN_NAME, COLLATION_NAME FROM information_schema.COLUMNS "
                        "WHERE TABLE_SCHEMA = COALESCE(:s, DATABASE()) AND TABLE_NAME = :t "
                        "AND COLLATION_NAME IS NOT NULL")
            with self.engine.connect() as con:
                rows = con.execute(q, {"s": self.schema, "t": table}).fetchall()
            self._collations[table] = {str(c): str(k) for c, k in rows}
        return self._collations[table]

    def refresh(self):
        super().refresh()
        self._collations = {}

    def stats(self, table):
        return MySQLStats(self, table)


def binary_collation(collation: str, dialect: str) -> str | None:
    """The binary collation of a collation's character set, or None when the
    collation compares exactly already."""
    cs = collation.split("_", 1)[0]
    if collation.endswith("_bin") and ("nopad" in collation or "0900" in collation):
        return None
    if cs == "utf8mb4":
        return NOPAD_BIN[dialect]
    if cs == "binary":
        return None
    return f"{cs}_bin"


class MySQLStats(SqlStats):
    """SqlStats with text compared exactly (binary collation), whatever the
    column's own collation."""

    def __init__(self, conn: MySQLConnector, table: str):
        super().__init__(conn, table)
        exact = {}
        for c, coll in conn.column_collations(table).items():
            b = binary_collation(coll, conn.dialect)
            if b is not None and c in self.tbl.c:
                exact[c] = b
        self._exact = exact

    def col(self, name):
        x = self.tbl.c[name]
        b = self._exact.get(name)
        return x.collate(b) if b else x
