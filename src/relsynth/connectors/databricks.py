# SPDX-License-Identifier: Apache-2.0
"""Databricks SQL warehouses and clusters (databricks-sqlalchemy over the
Databricks SQL connector).

    databricks://<workspace host>?http_path=/sql/1.0/warehouses/<id>&catalog=main&schema=sales

`schema` in the section overrides the URL's; the catalog comes from the URL
(the session starts in it).  Credentials never go in the URL or the
specification: `auth` picks the method (see warehouse.py),

| method      | needs                                      | driver path |
|-------------|--------------------------------------------|-------------|
| token       | token_env (default DATABRICKS_TOKEN)       | personal access token |
| oauth-m2m   | client_id_env, client_secret_env           | Databricks service principal, OAuth client credentials |
| azure-sp    | client_id_env, client_secret_env, tenant_id_env | Microsoft Entra ID service principal |
| oauth-u2m   | (browser sign-in)                          | Databricks OAuth, user to machine |
| azure-u2m   | (browser sign-in)                          | Entra ID OAuth, user to machine |
| default     | the Databricks SDK's credential chain      | env variables, ~/.databrickscfg, Azure CLI, managed identity |

Without `auth`, a token in DATABRICKS_TOKEN is used when it is set, else the
SDK's chain.

Engine differences handled here:

* FLOAT is single precision: created float columns are DOUBLE, and statistics
  cast with CAST(x AS DOUBLE);
* table names are stored in lower case and found without regard to case;
* text compares byte-wise (UTF8_BINARY), so no collation is needed;
* column names with spaces, commas, parentheses or '=' need Delta column
  mapping, which is switched on for the tables that have such names;
* constraints are informational and exist only in Unity Catalog: primary and
  foreign keys are declared (NOT ENFORCED, as Databricks always does) on a
  Unity Catalog schema, and left out in hive_metastore and on open-source
  Spark; UNIQUE constraints and indexes do not exist and are never created;
* the sample hash is SHA-256 of 'seed|key' (sha2), the same 32 bits as MySQL
  and DuckDB, so a sample can be checked from Python;
* loads: multi-row INSERT statements with typed literals by default (one Delta
  commit per batch of about 4 MB), or staged Parquet files and COPY INTO
  (`bulk: {method: volume, volume: /Volumes/<catalog>/<schema>/<volume>}`, the
  files are uploaded with the connector's PUT and removed afterwards; or
  `bulk: {method: path, path: <location the warehouse reads>}`).
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import os
import uuid

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import CreateTable

from .sql import frame_records
from .warehouse import WarehouseConnector

log = logging.getLogger("relsynth.connectors")

#: catalogs without Unity Catalog constraints
NO_CONSTRAINT_CATALOGS = ("hive_metastore", "spark_catalog")
#: a multi-row INSERT carries at most this much SQL text
INSERT_BYTES = 4_000_000


def sql_literal(v, kind: str) -> str:
    """A Python value as a typed Databricks SQL literal (strings escaped with
    backslashes, the Databricks rule; doubles in exponent form, which Databricks
    reads as DOUBLE, not DECIMAL)."""
    if v is None:
        return "NULL"
    if kind == "bool" or isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if kind == "int" and not isinstance(v, float):
        return str(int(v))
    if isinstance(v, (float, np.floating)) or kind == "float":
        f = float(v)
        if math.isnan(f):
            return "double('NaN')"
        if math.isinf(f):
            return "double('Infinity')" if f > 0 else "double('-Infinity')"
        r = repr(f)
        return r if ("e" in r or "E" in r) else r + "E0"
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        return f"TIMESTAMP_NTZ'{pd.Timestamp(v).tz_localize(None) if pd.Timestamp(v).tzinfo else pd.Timestamp(v)}'"
    if isinstance(v, dt.date):
        return f"DATE'{v.isoformat()}'"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (bytes, bytearray)):
        return f"X'{bytes(v).hex()}'"
    s = str(v).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"


#: characters Delta allows in column names only with column mapping
DELTA_SPECIAL = set(" ,;{}()\n\t=")
COLUMN_MAPPING = ("'delta.columnMapping.mode' = 'name', 'delta.minReaderVersion' = '2', "
                  "'delta.minWriterVersion' = '5', ")


@compiles(CreateTable, "databricks")
def _create_table(element, compiler, **kw):
    """CREATE TABLE with Delta column mapping when a column name needs it (spaces,
    commas, parentheses, ...): Databricks then stores the name as written."""
    text = compiler.visit_create_table(element, **kw)
    if element.element.info.get("rs_column_mapping"):
        text = text.replace("TBLPROPERTIES(", "TBLPROPERTIES(" + COLUMN_MAPPING, 1)
    return text


def _kind(ty) -> str:
    from .ddl_types import _kind as k
    return k(ty)


class DatabricksConnector(WarehouseConnector):
    """A schema of a Databricks SQL warehouse (or of open-source Spark with Delta
    behind a Thrift server, the local stand-in used by the tests)."""

    kind = "databricks"
    AUTH_METHODS = {"token": ("token",), "oauth-m2m": ("client_id", "client_secret"),
                    "azure-sp": ("client_id", "client_secret", "tenant_id"),
                    "oauth-u2m": (), "azure-u2m": (), "default": ()}

    def __init__(self, url=None, schema=None, **kw):
        super().__init__(url, schema, **kw)
        # the URL's catalog (not `catalog`, which is Connector.catalog())
        self.catalog_name = self.engine.dialect.__dict__.get("catalog") or sa.engine.make_url(
            str(self.engine.url)).query.get("catalog")
        if self.schema is None:
            self.schema = sa.engine.make_url(str(self.engine.url)).query.get("schema") or None
            self._md = sa.MetaData(schema=self.schema)
        self._platform: str | None = None

    # ------------------------------------------------------------ connection
    def auth_connect_args(self, url):
        from .warehouse import resolve_auth
        auth = self._auth
        if auth is None:
            if url.password:
                return {}
            # names the variable, not a secret
            token = {"method": "token", "token_env": "DATABRICKS_TOKEN"}  # nosec B105
            auth = token if os.environ.get("DATABRICKS_TOKEN") else {"method": "default"}
        a = resolve_auth(auth, self.AUTH_METHODS)
        m = a["method"]
        host = url.host
        if m == "token":
            return {"access_token": a["token"]}
        if m == "oauth-m2m":
            from databricks.sdk.core import Config, oauth_service_principal

            def provider():
                return oauth_service_principal(Config(host=f"https://{host}", client_id=a["client_id"],
                                                      client_secret=a["client_secret"]))
            return {"credentials_provider": provider}
        if m == "azure-sp":
            return {"auth_type": "azure-sp-m2m", "azure_client_id": a["client_id"],
                    "azure_client_secret": a["client_secret"], "azure_tenant_id": a["tenant_id"]}
        if m == "oauth-u2m":
            return {"auth_type": "databricks-oauth"}
        if m == "azure-u2m":
            return {"auth_type": "azure-oauth"}
        # the Databricks SDK's unified credential chain (environment, config profile, Azure CLI, MSI)
        from databricks.sdk.core import Config

        def default_provider():
            cfg = Config(host=f"https://{host}", **{k: v for k, v in a.items() if k in ("profile", "auth_type")})
            return cfg.authenticate
        return {"credentials_provider": default_provider}

    def _engine_options(self, url):
        return {"pool_recycle": 3600}

    def ping(self):
        """As SqlConnector.ping, giving up after 30 s: the driver retries for up to
        15 minutes by default (a stopped warehouse takes minutes to start)."""
        quick = {**self._cargs, "_retry_stop_after_attempts_count": 2, "_retry_stop_after_attempts_duration": 30}
        main, self.engine = self.engine, sa.create_engine(self.engine.url, connect_args=quick, poolclass=sa.pool.NullPool)
        self._setup_engine(self.engine)
        self._insp = None
        try:
            out = super().ping()
        finally:
            self.engine.dispose()
            self.engine, self._insp = main, None
        return out

    @property
    def platform(self) -> str:
        """'databricks' on a Databricks SQL warehouse or cluster, 'spark' on
        open-source Spark (current_version() exists only on Databricks)."""
        if self._platform is None:
            try:
                with self.engine.connect() as con:
                    con.exec_driver_sql("SELECT current_version()").fetchall()
                self._platform = "databricks"
            except sa.exc.DBAPIError:
                self._platform = "spark"
        return self._platform

    @property
    def constraints(self) -> bool:
        """Whether primary and foreign keys can be declared (Unity Catalog)."""
        return self.platform == "databricks" and (self.catalog_name or "").lower() not in NO_CONSTRAINT_CATALOGS

    # ------------------------------------------------------------ catalog
    def tables(self, include=None, exclude=None):
        try:
            return super().tables(include, exclude)
        except sa.exc.DBAPIError as e:
            if "SCHEMA_NOT_FOUND" in str(e) or "NoSuchDatabaseException" in str(e):
                return []                       # a target schema not created yet
            raise

    def stored_name(self, table):
        """Databricks keeps table names in lower case and looks them up without
        regard to case: a spec's 'Album' is the table 'album'."""
        names = {t.lower(): t for t in self.tables()}
        return names.get(str(table).lower(), table)

    def has_table(self, table):
        return str(table).lower() in {t.lower() for t in self.tables()}

    def _qualified(self, table: str) -> str:
        prep = self.engine.dialect.identifier_preparer
        parts = ([self.schema] if self.schema else []) + [self.stored_name(table)]
        return ".".join(prep.quote_identifier(p) for p in parts)

    def create_schema(self, schema=None):
        schema = schema or self.schema
        if not schema:
            return
        prep = self.engine.dialect.identifier_preparer
        with self.engine.begin() as con:
            con.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {prep.quote_identifier(schema)}")


    def _table_bytes(self, table):
        with self.engine.connect() as con:
            row = con.exec_driver_sql(f"DESCRIBE DETAIL {self._qualified(table)}").mappings().first()
        return int(row["sizeInBytes"]) if row and row.get("sizeInBytes") is not None else None

    def cost_extra(self):
        return {"billing": "compute time of the SQL warehouse (DBU per hour by size); bytes scanned "
                           "are the Delta size of the tables read"}

    # ------------------------------------------------------------ writes
    def adapt_metadata(self, md):
        """Only what Databricks can declare: primary and foreign keys (to a primary
        key) on Unity Catalog, identity columns on Databricks; never UNIQUE or
        indexes.  Key columns stay NOT NULL either way."""
        keys = self.constraints
        identity = self.platform == "databricks"
        out = sa.MetaData(schema=md.schema)
        for tbl in md.sorted_tables:
            cols = []
            for c in tbl.columns:
                args = []
                if identity and c.identity is not None:
                    args.append(sa.Identity(always=False))
                kw = {} if args else {"autoincrement": False}
                cols.append(sa.Column(c.name, c.type, *args, nullable=c.nullable and not c.primary_key,
                                      primary_key=keys and c.primary_key, **kw))
            new = sa.Table(tbl.name, out, *cols)
            if any(set(c.name) & DELTA_SPECIAL for c in tbl.columns):
                new.info["rs_column_mapping"] = True
            if not keys:
                continue
            for fk in tbl.foreign_key_constraints:
                ref = fk.referred_table
                to = [e.column.name for e in fk.elements]
                if sorted(to) != sorted(c.name for c in ref.primary_key.columns):
                    continue                    # Databricks keys reference a primary key only
                new.append_constraint(sa.ForeignKeyConstraint(
                    [c.name for c in fk.columns], [f"{ref.fullname}.{n}" for n in to]))
        return out

    def _insert_rows(self, con, tbl, df, chunk=20_000):
        method = self.bulk.get("method", "insert")
        if method in ("volume", "path") and len(df):
            return self._copy_into(con, tbl, df, method)
        return self._insert_literals(con, tbl, df)

    def _insert_literals(self, con, tbl, df) -> int:
        """INSERT INTO t (cols) VALUES (...), (...) with typed literals, in batches
        of at most INSERT_BYTES of SQL."""
        cols = list(df.columns)
        if not len(df):
            return 0
        prep = con.dialect.identifier_preparer
        # identifiers quoted by the dialect, values as escaped typed literals (sql_literal)
        head = f"INSERT INTO {prep.format_table(tbl)} ({', '.join(prep.quote(c) for c in cols)}) VALUES "  # noqa: S608  # nosec B608
        kinds = [_kind(tbl.c[c].type) for c in cols]
        batch: list[str] = []
        n, size = 0, len(head)
        for rec in frame_records(df, cols):
            row = "(" + ", ".join(sql_literal(v, k) for v, k in zip(rec, kinds)) + ")"
            if batch and size + len(row) + 2 > INSERT_BYTES:
                con.exec_driver_sql(head + ", ".join(batch))
                n += len(batch)
                batch, size = [], len(head)
            batch.append(row)
            size += len(row) + 2
        if batch:
            con.exec_driver_sql(head + ", ".join(batch))
            n += len(batch)
        return n

    def _copy_into(self, con, tbl, df, method) -> int:
        """Stage the rows as one Parquet file and load it with COPY INTO."""
        import tempfile

        import pyarrow as pa
        import pyarrow.parquet as pq
        prep = con.dialect.identifier_preparer
        name = f"relsynth_{tbl.name}_{uuid.uuid4().hex[:12]}.parquet".replace("/", "_")
        local = os.path.join(tempfile.mkdtemp(prefix="relsynth_"), name)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), local)
        if method == "volume":
            vol = str(self.bulk.get("volume", "")).rstrip("/")
            if not vol.startswith("/Volumes/"):
                raise ValueError("bulk.volume must be a Unity Catalog volume path /Volumes/<catalog>/<schema>/<volume>")
            remote = f"{vol}/{name}"
            raw = con.connection.dbapi_connection
            cur = raw.cursor()
            try:
                cur.execute(f"PUT {sql_literal(local, 'str')} INTO {sql_literal(remote, 'str')} OVERWRITE")
            finally:
                cur.close()
        else:
            base = str(self.bulk.get("path", "")).rstrip("/")
            if not base:
                raise ValueError("bulk.path must name a location the warehouse reads")
            remote = f"{base}/{name}"
            from pyarrow import fs
            filesystem, path = fs.FileSystem.from_uri(remote) if "://" in remote else (fs.LocalFileSystem(), remote)
            fs.copy_files(local, path, destination_filesystem=filesystem)
        cols = ", ".join(f"CAST({prep.quote(c)} AS {tbl.c[c].type.compile(dialect=con.dialect)}) AS {prep.quote(c)}"
                         for c in df.columns)
        loc = sql_literal(remote, "str")
        con.exec_driver_sql(f"COPY INTO {prep.format_table(tbl)} FROM (SELECT {cols} FROM {loc}) "  # noqa: S608  # nosec B608
                            "FILEFORMAT = PARQUET COPY_OPTIONS ('force' = 'true')")
        if method == "volume" and not self.bulk.get("keep_files"):
            raw = con.connection.dbapi_connection
            cur = raw.cursor()
            try:
                cur.execute(f"REMOVE {sql_literal(remote, 'str')}")
            finally:
                cur.close()
        os.remove(local)
        return len(df)
