# SPDX-License-Identifier: Apache-2.0
"""Snowflake (snowflake-sqlalchemy over the Snowflake Connector for Python).

    snowflake://<user>@<account identifier>/<database>/<schema>?warehouse=<warehouse>&role=<role>

`schema` in the section overrides the URL's.  Credentials never go in the URL or
the specification: `auth` picks the method (see warehouse.py),

* key-pair: private_key_file (a PKCS#8 PEM file) and, when it is encrypted,
  private_key_passphrase_env; the recommended method for service users;
* pat: token_env, a programmatic access token;
* oauth: token_env, an OAuth access token from your identity provider;
* password: password_env (Snowflake is phasing out single-factor passwords);
* sso: browser sign-in through your identity provider (externalbrowser);
* workload-identity: the cloud platform's identity of the machine relsynth runs
  on (AWS, Azure, Google Cloud), with optional `provider`;
* default: a connection from ~/.snowflake/connections.toml (`connection_name`).

Without `auth`, a password in the URL (or password_env) is used.

Engine differences handled here:

* FLOAT is double precision and VARCHAR holds up to 16 MB, so the generic types
  keep the data; date-times are TIMESTAMP_NTZ, stored as given;
* unquoted names are stored in upper case; the dialect maps lower-case names
  to them and quotes mixed-case names, so 'Album' stays 'Album';
* primary, unique and foreign keys are declared but not enforced (NOT NULL is);
  indexes and identity columns are not created;
* the sample hash is SHA-256 of 'seed|key' (SHA2), the same 32 bits as MySQL,
  DuckDB, Databricks and Fabric.  The first 8 hex digits are read with
  CHARINDEX arithmetic rather than TO_NUMBER(..., 'XXXXXXXX'), so the same
  statement runs on the fakesnow stand-in;
* the query tag is the session's QUERY_TAG, which Snowflake keeps in its query
  history (QUERY_HISTORY, ACCOUNT_USAGE);
* table sizes come from INFORMATION_SCHEMA.TABLES.BYTES (compressed storage);
* loads: multi-row INSERT statements by default, or
  `bulk: {method: stage}`: the rows are written to Parquet, PUT in a temporary
  stage of the session and loaded with COPY INTO (write_pandas).
"""
from __future__ import annotations

import logging

import sqlalchemy as sa

from .sql import frame_records
from .sqlfuncs import row_hash
from .warehouse import WarehouseConnector, resolve_auth

log = logging.getLogger("relsynth.connectors")


def snowflake_type(ty):
    """The Snowflake type of a column type (see the module docstring)."""
    from snowflake.sqlalchemy import TIMESTAMP_NTZ
    if isinstance(ty, sa.DateTime) and not isinstance(ty, TIMESTAMP_NTZ):
        return TIMESTAMP_NTZ()
    if isinstance(ty, sa.Float):
        return sa.Float()                       # FLOAT is always double precision
    if isinstance(ty, sa.Enum):
        return sa.String()
    return ty


class SnowflakeConnector(WarehouseConnector):
    """A schema of a Snowflake database (or of the fakesnow stand-in)."""

    kind = "snowflake"
    AUTH_METHODS = {"key-pair": ("private_key_file",), "pat": ("token",), "oauth": ("token",),
                    "password": ("password",), "sso": (), "workload-identity": (), "default": ()}

    def __init__(self, url=None, schema=None, **kw):
        self._platform: str | None = None
        super().__init__(url, schema, **kw)
        u = sa.engine.make_url(str(self.engine.url))
        parts = (u.database or "").split("/")
        self.database = parts[0] or None
        if self.schema is None and len(parts) > 1 and parts[1]:
            self.schema = parts[1]
            self._md = sa.MetaData(schema=self.schema)

    # ------------------------------------------------------------ connection
    def auth_connect_args(self, url):
        tag = {"session_parameters": {"QUERY_TAG": self.query_tag}}
        if self._auth is None:
            return tag
        a = resolve_auth(self._auth, self.AUTH_METHODS)
        m = a["method"]
        if m == "key-pair":
            out = {"private_key_file": a["private_key_file"]}
            if a.get("private_key_passphrase"):
                out["private_key_file_pwd"] = a["private_key_passphrase"]
        elif m == "pat":
            out = {"authenticator": "PROGRAMMATIC_ACCESS_TOKEN", "token": a["token"]}
        elif m == "oauth":
            out = {"authenticator": "oauth", "token": a["token"]}
        elif m == "password":
            out = {"password": a["password"]}
        elif m == "sso":
            out = {"authenticator": "externalbrowser"}
        elif m == "workload-identity":
            out = {"authenticator": "WORKLOAD_IDENTITY"}
            if a.get("provider"):
                out["workload_identity_provider"] = str(a["provider"]).upper()
        else:
            out = {"connection_name": a["connection_name"]} if a.get("connection_name") else {}
        return {**out, **tag}

    def tag_statements(self, engine):
        """The tag is the session's QUERY_TAG (set when connecting), not a comment."""

    def _setup_engine(self, engine):
        super()._setup_engine(engine)
        dialect = engine.dialect
        one, multi = dialect.get_foreign_keys, dialect.get_multi_foreign_keys

        # a key that names no referenced table cannot be followed (the fakesnow
        # stand-in reports primary keys this way); Snowflake itself never does
        def get_foreign_keys(connection, table_name, schema=None, **kw):
            return [f for f in one(connection, table_name, schema=schema, **kw) if f.get("referred_table")]

        def get_multi_foreign_keys(connection, **kw):
            return [(k, [f for f in fks if f.get("referred_table")]) for k, fks in multi(connection, **kw)]
        dialect.get_foreign_keys = get_foreign_keys
        dialect.get_multi_foreign_keys = get_multi_foreign_keys

    @property
    def platform(self) -> str:
        """'snowflake', or 'fakesnow' on the local stand-in (which has no CURRENT_ACCOUNT())."""
        if self._platform is None:
            try:
                with self.engine.connect() as con:
                    con.exec_driver_sql("SELECT CURRENT_ACCOUNT()").fetchall()
                self._platform = "snowflake"
            except sa.exc.DBAPIError:
                self._platform = "fakesnow"
        return self._platform

    @property
    def constraints(self) -> bool:
        """Whether declared keys can be read back (not on the stand-in)."""
        return self.platform == "snowflake"

    def _engine_options(self, url):
        return {"pool_recycle": 3600}

    # ------------------------------------------------------------ catalog
    def create_schema(self, schema=None):
        schema = schema or self.schema
        if not schema:
            return
        prep = self.engine.dialect.identifier_preparer
        with self.engine.begin() as con:
            con.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {prep.quote(schema)}")

    def _stored(self, name: str) -> str:
        """A name as Snowflake stores it (upper case unless it needed quoting)."""
        return self.engine.dialect.denormalize_name(name)

    def _table_bytes(self, table):
        info = (f"{self.engine.dialect.identifier_preparer.quote(self.database)}.INFORMATION_SCHEMA.TABLES"
                if self.database else "INFORMATION_SCHEMA.TABLES")
        q = f"SELECT BYTES FROM {info} WHERE TABLE_SCHEMA = %(s)s AND TABLE_NAME = %(t)s"  # noqa: S608  # nosec B608
        with self.engine.connect() as con:
            row = con.exec_driver_sql(q, {"s": self._stored(self.schema or "PUBLIC"),
                                          "t": self._stored(self.stored_name(table))}).first()
        return int(row[0]) if row and row[0] is not None else None

    def cost_extra(self):
        return {"billing": "credits of the virtual warehouse per second it runs (by size); bytes are the "
                           "compressed storage of the tables read (INFORMATION_SCHEMA.TABLES.BYTES)"}

    def _row_hash(self, cols, seed):
        return row_hash(cols, seed, "snowflake")

    # ------------------------------------------------------------ writes
    def adapt_metadata(self, md):
        """Snowflake types; keys declared (informational; foreign keys not on the
        stand-in); no indexes, identity columns or defaults."""
        out = sa.MetaData(schema=md.schema)
        for tbl in md.sorted_tables:
            cols = [sa.Column(c.name, snowflake_type(c.type), nullable=c.nullable and not c.primary_key,
                              primary_key=c.primary_key, autoincrement=False) for c in tbl.columns]
            new = sa.Table(tbl.name, out, *cols)
            for uc in tbl.constraints:
                if isinstance(uc, sa.UniqueConstraint):
                    new.append_constraint(sa.UniqueConstraint(*[c.name for c in uc.columns]))
        if not self.constraints:
            return out              # the stand-in (DuckDB) would enforce them; Snowflake never does
        for tbl in md.sorted_tables:
            new = out.tables[tbl.key]
            for fk in tbl.foreign_key_constraints:
                ref = out.tables[fk.referred_table.key]
                new.append_constraint(sa.ForeignKeyConstraint([c.name for c in fk.columns],
                                                              [ref.c[e.column.name] for e in fk.elements]))
        return out

    def _insert_rows(self, con, tbl, df, chunk=20_000):
        if self.bulk.get("method", "insert") == "stage" and len(df):
            return self._write_pandas(con, tbl, df)
        # parameters named p0, p1, ...: the driver rewrites executemany into one
        # multi-row INSERT and cannot parse parameters named after unusual columns
        cols = list(df.columns)
        stmt = tbl.insert().values({c: sa.bindparam(f"p{j}", type_=tbl.c[c].type) for j, c in enumerate(cols)})
        n = 0
        for i in range(0, len(df), chunk):
            recs = [{f"p{j}": v for j, v in enumerate(r)} for r in frame_records(df.iloc[i:i + chunk], cols)]
            if recs:
                con.execute(stmt, recs)
                n += len(recs)
        return n

    def _write_pandas(self, con, tbl, df) -> int:
        """PUT the rows as Parquet in a temporary stage and COPY INTO (write_pandas)."""
        from snowflake.connector.pandas_tools import write_pandas
        raw = con.connection.dbapi_connection
        df = df.rename(columns=self._stored)            # names as stored: quote_identifiers keeps them
        db = self._stored(self.database) if self.database else None
        ok, _, rows, _ = write_pandas(raw, df, self._stored(tbl.name), database=db,
                                      schema=self._stored(tbl.schema) if tbl.schema else None,
                                      quote_identifiers=True, auto_create_table=False, use_logical_type=True,
                                      chunk_size=int(self.bulk.get("chunk_rows", 1_000_000)))
        if not ok:
            raise RuntimeError(f"{tbl.name}: COPY INTO from the stage did not load every row")
        return int(rows)


__all__ = ["SnowflakeConnector", "snowflake_type"]
