# SPDX-License-Identifier: Apache-2.0
"""Microsoft Fabric: the Warehouse, and the SQL analytics endpoint of a Lakehouse
(read only), over TDS with the Microsoft ODBC Driver 18.

    fabric://<id>.datawarehouse.fabric.microsoft.com/<warehouse>

The URL becomes mssql+pyodbc with the ODBC Driver 18 and encryption on (query
options in the URL are kept).  Fabric signs in with Microsoft Entra ID only;
`auth.method` picks how (see warehouse.py):

* default: azure-identity's DefaultAzureCredential (AZURE_* variables, workload
  identity, managed identity, Azure CLI, ...);
* azure-sp: a service principal (client_id_env, client_secret_env, tenant_id_env);
* azure-cli: the account of `az login` on this machine;
* managed-identity: the Azure resource's managed identity (optional client_id_env);
* interactive: browser sign-in by the ODBC driver (ActiveDirectoryInteractive);
* sql-login: user and password (password_env), for SQL Server stand-ins only.

A token from azure-identity is passed to the driver per connection
(SQL_COPT_SS_ACCESS_TOKEN), so it is refreshed when it expires and never
appears in a connection string.

Engine differences handled here (the same on a SQL Server stand-in, apart from
constraints):

* types: Fabric has no NVARCHAR, NCHAR, TEXT, DATETIME, TINYINT or MONEY.  Text
  is VARCHAR in the warehouse's UTF-8 collation, sized in bytes (4 per
  character; VARCHAR(MAX) beyond 8,000), date-times DATETIME2(6), times
  TIME(6), floats FLOAT(53), TINYINT SMALLINT, MONEY DECIMAL(19, 4);
* text compares in Latin1_General_100_BIN2_UTF8 (binary) when the warehouse was
  created case-insensitive, so grouping and keys see the exact values;
* constraints: primary keys (NONCLUSTERED), unique keys (NONCLUSTERED) and
  foreign keys are declared NOT ENFORCED, the only form Fabric has; indexes,
  identity columns and defaults are not created.  On SQL Server, which has no
  NOT ENFORCED form, the keys are ordinary constraints;
* the sample hash is the first 32 bits of SHA-256 of 'seed|key' in UTF-8
  (HASHBYTES over VARCHAR, whose bytes are UTF-8 in every Fabric collation),
  the same as Python's hashlib.sha256 and the MySQL, DuckDB and Databricks hash;
* loads: parameter arrays (pyodbc fast_executemany) by default, or Parquet
  staged in Azure Data Lake Storage / OneLake and COPY INTO
  (`bulk: {method: copy, stage: https://<account>.dfs.core.windows.net/<container>/<dir>}`),
  read by the warehouse with the caller's Entra ID identity.
"""
from __future__ import annotations

import io
import logging
import struct
import uuid

import sqlalchemy as sa
from sqlalchemy.dialects import mssql

from .mssql import MSSQLConnector
from .sql import SqlStats
from .sqlfuncs import row_hash
from .warehouse import WarehouseConnector, resolve_auth

log = logging.getLogger("relsynth.connectors")

UTF8_BIN = "Latin1_General_100_BIN2_UTF8"
SCOPE = "https://database.windows.net/.default"
SQL_COPT_SS_ACCESS_TOKEN = 1256
ODBC_DRIVER = "ODBC Driver 18 for SQL Server"
#: SERVERPROPERTY('EngineEdition') of Fabric Warehouse and SQL analytics endpoints
FABRIC_EDITIONS = (11, 12)
#: bytes per character of UTF-8 (VARCHAR lengths count bytes) and the longest VARCHAR(n)
UTF8_CHAR_BYTES, MAX_VARCHAR_BYTES = 4, 8000
#: longest index key in bytes (a clustered SQL Server key; Fabric keys are not indexed but keep the limit)
KEY_BYTES = 900


def token_struct(token: str) -> bytes:
    """An Entra ID access token as the ODBC driver takes it (UTF-16-LE, length prefixed)."""
    raw = token.encode("utf-16-le")
    return struct.pack(f"<I{len(raw)}s", len(raw), raw)


def fabric_type(ty):
    """The Fabric type of a column type (see the module docstring)."""
    if isinstance(ty, sa.Enum):
        ty = sa.String(max(len(str(v)) for v in ty.enums) if ty.enums else None)
    if isinstance(ty, (sa.Text, sa.UnicodeText)) or (isinstance(ty, sa.String) and not getattr(ty, "length", None)):
        return mssql.VARCHAR(None)                      # VARCHAR(MAX)
    if isinstance(ty, sa.String):
        n = ty.length * UTF8_CHAR_BYTES
        return mssql.VARCHAR(n if n <= MAX_VARCHAR_BYTES else None)
    if isinstance(ty, (mssql.MONEY, mssql.SMALLMONEY)):
        return sa.Numeric(19, 4)
    if isinstance(ty, mssql.TINYINT):
        return sa.SmallInteger()
    if isinstance(ty, sa.DateTime):
        return mssql.DATETIME2(precision=6)
    if isinstance(ty, sa.Time):
        return mssql.TIME(precision=6)
    if isinstance(ty, sa.Float) and (getattr(ty, "precision", None) or 53) > 24:
        return mssql.FLOAT(53)
    if isinstance(ty, sa.LargeBinary):
        return mssql.VARBINARY(None)
    return ty


def key_widths(tbl: sa.Table) -> dict[str, int]:
    """Bytes a text column of a key may take: an index key holds at most
    KEY_BYTES, shared by the key's columns."""
    groups = [list(tbl.primary_key.columns)]
    groups += [list(c.columns) for c in tbl.constraints if isinstance(c, sa.UniqueConstraint)]
    groups += [list(fk.columns) for fk in tbl.foreign_key_constraints]
    width: dict[str, int] = {}
    for g in groups:
        for c in g:
            width[c.name] = min(width.get(c.name, KEY_BYTES), KEY_BYTES // max(len(g), 1))
    return width


def key_type(ty, width: int | None):
    """A key column's VARCHAR within the index key limit (VARCHAR(MAX) cannot be a key)."""
    if width and isinstance(ty, mssql.VARCHAR) and (ty.length is None or ty.length > width):
        return mssql.VARCHAR(width)
    return ty


class FabricConnector(WarehouseConnector, MSSQLConnector):
    """A Fabric Warehouse or Lakehouse SQL endpoint (or SQL Server 2022 standing in for one)."""

    kind = "fabric"
    AUTH_METHODS = {"default": (), "azure-sp": ("client_id", "client_secret", "tenant_id"), "azure-cli": (),
                    "managed-identity": (), "interactive": (), "sql-login": ()}

    def __init__(self, url=None, schema=None, **kw):
        self._credential = None
        self._platform: str | None = None
        self._collation: str | None = None
        super().__init__(url, schema, **kw)

    # ------------------------------------------------------------ connection
    def _method(self, url: sa.URL) -> str:
        return resolve_auth(self._auth, self.AUTH_METHODS, default="sql-login" if url.password else "default")["method"]

    def _engine_url(self, url: str) -> str:
        u = sa.engine.make_url(url)
        q = {k.lower(): v for k, v in u.query.items()}
        add = {}
        if "driver" not in q:
            add["driver"] = ODBC_DRIVER
        if "encrypt" not in q:
            add["Encrypt"] = "yes"
        if self._method(u) == "interactive" and "authentication" not in q:
            add["Authentication"] = "ActiveDirectoryInteractive"
        return u.update_query_dict(add).render_as_string(hide_password=False) if add else url

    def auth_connect_args(self, url):
        a = resolve_auth(self._auth, self.AUTH_METHODS, default="sql-login" if url.password else "default")
        m = a["method"]
        if m in ("sql-login", "interactive"):
            return {}
        from azure import identity
        if m == "azure-sp":
            self._credential = identity.ClientSecretCredential(a["tenant_id"], a["client_id"], a["client_secret"])
        elif m == "azure-cli":
            self._credential = identity.AzureCliCredential()
        elif m == "managed-identity":
            self._credential = identity.ManagedIdentityCredential(client_id=a.get("client_id"))
        else:
            self._credential = identity.DefaultAzureCredential()
        return {}

    def _setup_engine(self, engine):
        super()._setup_engine(engine)
        if self._credential is None:
            return
        cred = self._credential

        @sa.event.listens_for(engine, "do_connect")
        def _token(dialect, conn_rec, cargs, cparams):
            # a fresh token per connection; the connection string carries no user
            cargs[0] = cargs[0].replace(";Trusted_Connection=Yes", "")
            tok = cred.get_token(SCOPE).token
            cparams["attrs_before"] = {**cparams.get("attrs_before", {}), SQL_COPT_SS_ACCESS_TOKEN: token_struct(tok)}

    @property
    def platform(self) -> str:
        """'fabric' on Fabric, 'sqlserver' on a SQL Server stand-in."""
        if self._platform is None:
            with self.engine.connect() as con:
                ed = con.exec_driver_sql("SELECT CAST(SERVERPROPERTY('EngineEdition') AS INT)").scalar()
            self._platform = "fabric" if int(ed or 0) in FABRIC_EDITIONS else "sqlserver"
        return self._platform

    @property
    def collation(self) -> str:
        """The database's collation (text columns and literals take it)."""
        if self._collation is None:
            with self.engine.connect() as con:
                self._collation = str(con.exec_driver_sql(
                    "SELECT CAST(DATABASEPROPERTYEX(DB_NAME(), 'Collation') AS VARCHAR(128))").scalar() or "")
        return self._collation

    def cost_extra(self):
        return {"billing": "capacity units (CU) of the Fabric capacity; Fabric reports no stored table size, "
                           "so max_bytes cannot be checked before a query"}

    def _row_hash(self, cols, seed):
        if "UTF8" not in self.collation.upper():
            raise ValueError(f"the database collation {self.collation} is not UTF-8: the Fabric sample hash "
                             "needs one (Fabric warehouses always have one; create a stand-in with "
                             "fabric.stand_in_database)")
        return row_hash(cols, seed, "fabric")

    def stats(self, table):
        self.guard_scan(table, what="in-database statistics")
        return FabricStats(self, table)

    # ------------------------------------------------------------ writes
    def adapt_metadata(self, md):
        """Fabric types (both platforms); keys NOT ENFORCED on Fabric (added after
        the table, the form Fabric accepts) and ordinary constraints on SQL
        Server; no indexes, identity columns or defaults."""
        fabric = self.platform == "fabric"
        out = sa.MetaData(schema=md.schema)
        prep = mssql.dialect().identifier_preparer

        def q(names):
            return ", ".join(prep.quote(n) for n in names)

        def full(t):
            return ".".join(prep.quote(p) for p in ([t.schema] if t.schema else []) + [t.name])
        for tbl in md.sorted_tables:
            pk = [c.name for c in tbl.primary_key.columns]
            width = key_widths(tbl)
            cols = [sa.Column(c.name, key_type(fabric_type(c.type), width.get(c.name)),
                              nullable=c.nullable and c.name not in pk,
                              primary_key=(not fabric and c.name in pk), autoincrement=False)
                    for c in tbl.columns]
            uniques = [[c.name for c in uc.columns] for uc in tbl.constraints if isinstance(uc, sa.UniqueConstraint)]
            new = sa.Table(tbl.name, out, *cols)
            if not fabric:
                for u in uniques:
                    new.append_constraint(sa.UniqueConstraint(*u))
                continue
            ddl = []
            if pk:
                ddl.append(f"ALTER TABLE {full(tbl)} ADD CONSTRAINT {prep.quote('pk_' + tbl.name)} "
                           f"PRIMARY KEY NONCLUSTERED ({q(pk)}) NOT ENFORCED")
            for i, u in enumerate(uniques):
                ddl.append(f"ALTER TABLE {full(tbl)} ADD CONSTRAINT {prep.quote(f'uq_{tbl.name}_{i}')} "
                           f"UNIQUE NONCLUSTERED ({q(u)}) NOT ENFORCED")
            for i, fk in enumerate(tbl.foreign_key_constraints):
                ddl.append(f"ALTER TABLE {full(tbl)} ADD CONSTRAINT {prep.quote(f'fk_{tbl.name}_{i}')} "
                           f"FOREIGN KEY ({q([c.name for c in fk.columns])}) REFERENCES {full(fk.referred_table)} "
                           f"({q([e.column.name for e in fk.elements])}) NOT ENFORCED")
            new.info["rs_constraints"] = ddl          # run after CREATE TABLE
            for stmt in ddl:
                sa.event.listen(new, "after_create", sa.DDL(stmt.replace("%", "%%")))
        if not fabric:              # foreign keys once every table exists in out
            for tbl in md.sorted_tables:
                new = out.tables[tbl.key]
                for fk in tbl.foreign_key_constraints:
                    ref = out.tables[fk.referred_table.key]
                    new.append_constraint(sa.ForeignKeyConstraint([c.name for c in fk.columns],
                                                                  [ref.c[e.column.name] for e in fk.elements]))
        return out

    def _insert_rows(self, con, tbl, df, chunk=20_000):
        if self.bulk.get("method") == "copy" and len(df):
            return self._copy_into(con, tbl, df)
        return super()._insert_rows(con, tbl, df, chunk)

    def _copy_into(self, con, tbl, df) -> int:
        """Stage the rows as one Parquet file in ADLS Gen2 / OneLake and COPY INTO."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        stage = str(self.bulk.get("stage", "")).rstrip("/")
        if not stage.startswith("https://"):
            raise ValueError("bulk.stage must be an https:// URL of an ADLS Gen2 or OneLake folder")
        name = f"relsynth_{tbl.name}_{uuid.uuid4().hex[:12]}.parquet".replace("/", "_")
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf)
        from urllib.parse import urlsplit

        from azure.storage.filedatalake import DataLakeFileClient
        u = urlsplit(stage)
        fs, _, folder = u.path.strip("/").partition("/")       # container (OneLake: workspace), then the folder
        if not fs:
            raise ValueError("bulk.stage must name a container (or OneLake workspace) and a folder")
        client = DataLakeFileClient(f"{u.scheme}://{u.netloc}", fs, f"{folder}/{name}" if folder else name,
                                    credential=self._credential)
        client.upload_data(buf.getvalue(), overwrite=True)
        prep = con.dialect.identifier_preparer
        cols = ", ".join(prep.quote(c) for c in df.columns)
        url = f"{stage}/{name}".replace("'", "''")
        con.exec_driver_sql(f"COPY INTO {prep.format_table(tbl)} ({cols}) FROM '{url}' "  # noqa: S608  # nosec B608
                            "WITH (FILE_TYPE = 'PARQUET')")
        if not self.bulk.get("keep_files"):
            client.delete_file()
        return len(df)


class FabricStats(SqlStats):
    """SqlStats with text compared in the binary UTF-8 collation when the
    database's own collation ignores case or accents."""

    def __init__(self, conn, table):
        super().__init__(conn, table)
        binary = "BIN" in conn.collation.upper()
        self._text = set() if binary else {c.name for c in self.tbl.columns if isinstance(c.type, sa.String)}

    def col(self, name):
        x = self.tbl.c[name]
        return x.collate(UTF8_BIN) if name in self._text else x


def stand_in_database(url: str, name: str) -> None:
    """Create a database in the UTF-8 binary collation on a SQL Server stand-in (tests)."""
    eng = sa.create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with eng.connect() as con:
            if not con.exec_driver_sql("SELECT 1 FROM sys.databases WHERE name = ?", (name,)).first():
                con.exec_driver_sql(f"CREATE DATABASE [{name.replace(']', ']]')}] COLLATE {UTF8_BIN}")
    finally:
        eng.dispose()


__all__ = ["FabricConnector", "FabricStats", "fabric_type", "key_type", "key_widths", "token_struct", "stand_in_database"]
