# SPDX-License-Identifier: Apache-2.0
"""Connectors: where data is read from and written to.

    from relsynth.connectors import connect
    src = connect("postgresql://user@host/db", schema="sales", password_env="PGPASSWORD")
    src = connect("mssql+pyodbc://user@host/db?driver=ODBC+Driver+18+for+SQL+Server")
    src = connect("oracle://user@host:1521/?service_name=ORCLPDB1")
    src = connect("mysql://user@host:3306/sales")      # or mariadb://
    src = connect("databricks://adb-1.azuredatabricks.net?http_path=/sql/1.0/warehouses/ab&catalog=main",
                  schema="sales", auth={"method": "oauth-m2m", "client_id_env": "DATABRICKS_CLIENT_ID",
                                        "client_secret_env": "DATABRICKS_CLIENT_SECRET"})
    src = connect("fabric://abc.datawarehouse.fabric.microsoft.com/sales_wh", schema="dbo",
                  auth={"method": "azure-cli"})
    src = connect("sqlite:shop.db")          # or sqlite:///shop.db
    src = connect("csv:./data")              # a folder of CSV files
    src = connect("parquet:./lake/sales")    # a folder of Parquet files

A specification's `source` and `target` sections are connected with
`from_spec(section, base_dir)`.

| kind       | backend                      | driver (pip extra)                 |
|------------|------------------------------|------------------------------------|
| postgresql | PostgreSQL 12+               | psycopg 3 (relsynth[postgres])     |
| mssql      | SQL Server 2016+, Azure SQL  | pyodbc + ODBC Driver 18, or pymssql (relsynth[mssql]) |
| oracle     | Oracle Database 19c+         | python-oracledb thin (relsynth[oracle]) |
| mysql      | MySQL 8.0+, MariaDB 10.6+    | PyMySQL (relsynth[mysql]); mysqlclient when installed |
| databricks | Databricks SQL warehouses, clusters | databricks-sqlalchemy (relsynth[databricks]) |
| fabric     | Microsoft Fabric Warehouse, Lakehouse SQL endpoint | pyodbc + ODBC Driver 18, azure-identity (relsynth[fabric]) |
| sqlite     | SQLite 3.25+ files           | standard library                   |
| csv        | folder of .csv / .csv.gz     | pandas                             |
| parquet    | folder of .parquet           | pyarrow (relsynth[parquet])        |

Folders compute the statistics of pushed-down facts with DuckDB (SQL over the
files, see connectors/duckfiles.py); `engine: pandas` in the source section
keeps the 1.0 in-memory path.
"""
from __future__ import annotations

from pathlib import Path

from .base import ColumnInfo, Connector, ForeignKey, TableInfo  # noqa: F401
from .files import FileConnector

__all__ = ["connect", "from_spec", "Connector", "TableInfo", "ColumnInfo", "ForeignKey", "FileConnector",
           "KINDS"]

KINDS = ("csv", "files", "parquet", "sqlite", "postgresql", "mssql", "oracle", "mysql", "mariadb", "databricks", "fabric",
         "sqlalchemy")
_FILE_PREFIXES = {"csv:": "csv", "files:": "csv", "parquet:": "parquet"}


def _sql_class(dialect: str):
    if dialect == "sqlite":
        from .sqlite import SQLiteConnector
        return SQLiteConnector
    if dialect == "postgresql":
        from .postgres import PostgresConnector
        return PostgresConnector
    if dialect == "mssql":
        from .mssql import MSSQLConnector
        return MSSQLConnector
    if dialect == "oracle":
        from .oracle import OracleConnector
        return OracleConnector
    if dialect in ("mysql", "mariadb"):
        from .mysql import MySQLConnector
        return MySQLConnector
    if dialect == "databricks":
        from .databricks import DatabricksConnector
        return DatabricksConnector
    if dialect == "fabric":
        from .fabric import FabricConnector
        return FabricConnector
    from .sql import SqlConnector
    return SqlConnector


def connect(source, *, schema: str | None = None, url_env: str | None = None, password_env: str | None = None,
            null_values=None, fmt: str | None = None, create: bool = False, connect_args: dict | None = None,
            engine_options: dict | None = None, base_dir: str | Path | None = None,
            auth: dict | None = None, guards: dict | None = None, bulk: dict | None = None) -> Connector:
    """A connector for a data location.

    source: a URL (any SQLAlchemy URL; postgresql://, mssql://, oracle:// pick the
    recommended driver), 'sqlite:FILE', 'csv:DIR', 'files:DIR', 'parquet:DIR', a
    spec section ({kind, path or url, schema, ...}) or a Connector (returned as is).
    Credentials: url_env names a variable holding the whole URL, password_env one
    holding only the password; neither is written to logs or bundles.
    Warehouses (databricks, fabric) also take auth (credentials from environment
    variables or the platform's credential chain), guards (cost guards) and bulk
    (the load path); see connectors/warehouse.py."""
    if isinstance(source, Connector):
        return source
    if isinstance(source, dict):
        return from_spec(source, base_dir)
    s = str(source)
    base = Path(base_dir) if base_dir else None

    def _p(x):
        p = Path(x)
        return p if p.is_absolute() or base is None else base / p
    for pre, kind in _FILE_PREFIXES.items():
        if s.startswith(pre):
            return FileConnector(_p(s[len(pre):]), fmt=fmt or kind, null_values=null_values, create=create)
    if s.startswith("sqlite:") and not s.startswith("sqlite://"):
        from .sqlite import SQLiteConnector
        return SQLiteConnector(str(_p(s[len("sqlite:"):])), connect_args=connect_args, engine_options=engine_options)
    if "://" not in s:
        p = _p(s)
        if p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
            from .sqlite import SQLiteConnector
            return SQLiteConnector(str(p))
        return FileConnector(p, fmt=fmt or "csv", null_values=null_values, create=create)
    import sqlalchemy as sa
    dialect = sa.engine.make_url(s if "***" not in s else s.replace("***", "x")).get_backend_name()
    if dialect == "postgres":
        dialect = "postgresql"
    cls = _sql_class(dialect)
    if dialect == "sqlite":
        return cls(url=s, connect_args=connect_args, engine_options=engine_options)
    extra = _warehouse_options(cls, auth=auth, guards=guards, bulk=bulk)
    return cls(s, schema=schema, url_env=url_env, password_env=password_env, connect_args=connect_args,
               engine_options=engine_options, **extra)


def _warehouse_options(cls, **opts) -> dict:
    """auth, guards and bulk for a warehouse connector (refused elsewhere)."""
    from .warehouse import WarehouseConnector
    given = {k: v for k, v in opts.items() if v is not None}
    if given and not issubclass(cls, WarehouseConnector):
        raise ValueError(f"{', '.join(given)}: only for cloud warehouses (databricks, fabric)")
    return given


def from_spec(section: dict, base_dir: str | Path | None = None, create: bool = False) -> Connector:
    """The connector of a spec's source or target section."""
    sec = dict(section or {})
    kind = sec.get("kind", "csv")
    base = Path(base_dir) if base_dir else Path(".")

    def _p(x):
        p = Path(x)
        return p if p.is_absolute() else base / p
    common = dict(connect_args=sec.get("connect_args"), engine_options=sec.get("engine_options"))
    if kind in ("csv", "files", "parquet"):
        return FileConnector(_p(sec.get("path", ".")), fmt=sec.get("format") or ("parquet" if kind == "parquet" else "csv"),
                             null_values=sec.get("null_values"), create=create, engine=sec.get("engine", "auto"),
                             duckdb=sec.get("duckdb"))
    if kind == "sqlite" and (sec.get("path") and not sec.get("url_env")):
        from .sqlite import SQLiteConnector
        return SQLiteConnector(str(_p(sec["path"])), **common)
    url = sec.get("url")
    env = sec.get("url_env")
    if kind == "sqlite" and not url and not env:
        raise ValueError("sqlite needs `path` (or `url`)")
    import os
    import sqlalchemy as sa
    probe = os.environ.get(env) if env and os.environ.get(env) else url
    if not probe:
        raise ValueError(f"{kind}: set `url`, or `url_env` naming an environment variable with the URL")
    dialect = sa.engine.make_url(str(probe).replace("***", "x")).get_backend_name()
    if dialect == "postgres":
        dialect = "postgresql"
    cls = _sql_class(dialect)
    if dialect == "sqlite":
        from .sql import resolve_url
        return cls(url=resolve_url(url, env), **common)
    extra = _warehouse_options(cls, auth=sec.get("auth"), guards=sec.get("guards"), bulk=sec.get("bulk"))
    return cls(url, schema=sec.get("schema"), url_env=env, password_env=sec.get("password_env"), **common, **extra)
