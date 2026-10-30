# SPDX-License-Identifier: Apache-2.0
"""Cloud warehouse connectors without a warehouse: compiled SQL, created types,
literals, credentials from the environment and cost guards."""
from __future__ import annotations

import datetime as dt
import hashlib

import pandas as pd
import pytest
import sqlalchemy as sa

pytest.importorskip("databricks.sqlalchemy")

from relsynth.connectors import connect  # noqa: E402
from relsynth.connectors.sqlfuncs import py_row_hash_sha256, row_hash, to_float  # noqa: E402
from relsynth.connectors.warehouse import CostGuardError, parse_bytes, resolve_auth  # noqa: E402

DBX = "databricks://adb-1.azuredatabricks.net?http_path=/sql/1.0/warehouses/ab&catalog=main&schema=sales"


def _sql(expr, engine):
    return str(expr.compile(engine, compile_kwargs={"literal_binds": True}))


@pytest.fixture
def dbx(monkeypatch):
    monkeypatch.setenv("RS_TEST_DBX_TOKEN", "dapi-secret-123")
    c = connect(DBX, auth={"method": "token", "token_env": "RS_TEST_DBX_TOKEN"})
    yield c
    c.close()


def test_parse_bytes():
    assert parse_bytes("100GB") == 100 * 10**9 and parse_bytes("1.5 TiB") == int(1.5 * 2**40)
    assert parse_bytes(10) == 10 and parse_bytes(None) is None
    with pytest.raises(ValueError):
        parse_bytes("lots")


def test_auth_comes_from_the_environment_never_the_spec(monkeypatch):
    methods = {"token": ("token",), "oauth-m2m": ("client_id", "client_secret")}
    monkeypatch.setenv("RS_A", "s3cr3t")
    assert resolve_auth({"method": "token", "token_env": "RS_A"}, methods)["token"] == "s3cr3t"
    with pytest.raises(ValueError, match="secrets are never written"):
        resolve_auth({"method": "token", "token": "s3cr3t"}, methods)
    monkeypatch.delenv("RS_B", raising=False)
    with pytest.raises(ValueError, match="RS_B") as e:
        resolve_auth({"method": "oauth-m2m", "client_id_env": "RS_A", "client_secret_env": "RS_B"}, methods)
    assert "s3cr3t" not in str(e.value)
    with pytest.raises(ValueError, match="needs client_secret_env"):
        resolve_auth({"method": "oauth-m2m", "client_id_env": "RS_A"}, methods)
    with pytest.raises(ValueError, match="auth.method"):
        resolve_auth({"method": "password"}, methods)


def test_databricks_token_reaches_the_driver_only(dbx):
    assert dbx.kind == "databricks" and dbx.schema == "sales" and dbx.catalog_name == "main" and callable(dbx.catalog)
    assert "dapi-secret-123" not in dbx.location() and "dapi-secret-123" not in repr(dbx.engine.url)
    assert not hasattr(dbx, "token")


def test_databricks_methods_map_to_driver_arguments(monkeypatch):
    from relsynth.connectors.databricks import DatabricksConnector
    monkeypatch.setenv("RS_ID", "id")
    monkeypatch.setenv("RS_SECRET", "sec")
    monkeypatch.setenv("RS_TENANT", "ten")
    c = DatabricksConnector.__new__(DatabricksConnector)
    url = sa.engine.make_url(DBX)
    c._auth = {"method": "azure-sp", "client_id_env": "RS_ID", "client_secret_env": "RS_SECRET",
               "tenant_id_env": "RS_TENANT"}
    assert c.auth_connect_args(url) == {"auth_type": "azure-sp-m2m", "azure_client_id": "id",
                                        "azure_client_secret": "sec", "azure_tenant_id": "ten"}
    c._auth = {"method": "oauth-u2m"}
    assert c.auth_connect_args(url) == {"auth_type": "databricks-oauth"}
    c._auth = {"method": "oauth-m2m", "client_id_env": "RS_ID", "client_secret_env": "RS_SECRET"}
    assert callable(c.auth_connect_args(url)["credentials_provider"])
    c._auth = None
    monkeypatch.setenv("DATABRICKS_TOKEN", "tok")
    assert c.auth_connect_args(url) == {"access_token": "tok"}
    monkeypatch.delenv("DATABRICKS_TOKEN")
    assert callable(c.auth_connect_args(url)["credentials_provider"])      # the SDK's chain


def test_databricks_sql_functions(dbx):
    t = sa.table("t", sa.column("id"), sa.column("x"))
    assert _sql(to_float(t.c.x), dbx.engine) == "CAST(t.x AS DOUBLE)"
    h = _sql(row_hash([t.c.id], 7, "databricks"), dbx.engine)
    assert "sha2(concat('7|'," in h and "conv(left(" in h and "16, 10" in h
    # the Python twin of the in-warehouse hash (the first 32 bits of SHA-256)
    assert py_row_hash_sha256(7, "42") == int(hashlib.sha256(b"7|42").hexdigest()[:8], 16)


def test_databricks_literals_round_trip_exactly():
    from relsynth.connectors.databricks import sql_literal
    assert sql_literal(0.1, "float") == "0.1E0"                 # DOUBLE, not DECIMAL
    assert sql_literal(1e-300, "float") == "1e-300"
    assert sql_literal(float("nan"), "float") == "double('NaN')"
    assert sql_literal(3, "int") == "3" and sql_literal(None, "str") == "NULL"
    assert sql_literal("it's a \\ path", "str") == "'it\\'s a \\\\ path'"
    assert sql_literal(dt.date(2024, 2, 29), "date") == "DATE'2024-02-29'"
    assert sql_literal(pd.Timestamp("2024-01-02 03:04:05.123456"), "datetime") == \
        "TIMESTAMP_NTZ'2024-01-02 03:04:05.123456'"
    assert sql_literal(True, "bool") == "TRUE"


def test_databricks_created_types_keep_values():
    from relsynth.connectors.ddl import adapt_types
    md = sa.MetaData()
    t = sa.Table("t", md, sa.Column("x", sa.Float()), sa.Column("s", sa.String(20)), sa.Column("u", sa.Unicode(5)),
                 sa.Column("ts", sa.DateTime()), sa.Column("small", sa.Float(precision=10)))
    adapt_types(md, "databricks")
    assert isinstance(t.c.x.type, sa.Double) and type(t.c.small.type) is sa.Float
    assert t.c.s.type.length is None and t.c.u.type.length is None


def test_databricks_declares_only_what_it_supports(dbx):
    from sqlalchemy.schema import CreateTable
    md = sa.MetaData(schema="sales")
    sa.Table("p", md, sa.Column("id", sa.BigInteger, primary_key=True), sa.Column("code", sa.String(9)),
             sa.UniqueConstraint("code"))
    sa.Table("c", md, sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
             sa.Column("pid", sa.BigInteger, sa.ForeignKey("sales.p.id")),
             sa.Column("code", sa.String(9), sa.ForeignKey("sales.p.code")), sa.Index("ix", "pid"))
    dbx._platform = "databricks"                               # Unity Catalog: keys declared
    out = dbx.adapt_metadata(md)
    ddl = str(CreateTable(out.tables["sales.c"]).compile(dbx.engine))
    assert "PRIMARY KEY (id)" in ddl and "REFERENCES sales.p (id)" in ddl and "(code)" not in ddl
    assert "GENERATED BY DEFAULT AS IDENTITY" in ddl and not out.tables["sales.c"].indexes
    assert "UNIQUE" not in str(CreateTable(out.tables["sales.p"]).compile(dbx.engine))
    dbx._platform = "spark"                                    # no Unity Catalog: no keys, NOT NULL kept
    ddl = str(CreateTable(dbx.adapt_metadata(md).tables["sales.c"]).compile(dbx.engine))
    assert "PRIMARY KEY" not in ddl and "REFERENCES" not in ddl and "IDENTITY" not in ddl
    assert "id BIGINT NOT NULL" in ddl
    dbx._platform = "databricks"
    dbx.catalog_name = "hive_metastore"
    assert not dbx.constraints


def test_cost_guards(dbx, monkeypatch):
    c = connect(DBX.replace("adb-1", "adb-2"), auth={"method": "token", "token_env": "RS_TEST_DBX_TOKEN"},
                guards={"max_rows": 100, "max_bytes": "1MB", "query_tag": "team-x"})
    try:
        assert c.query_tag == "team-x" and c.max_bytes == 10**6
        monkeypatch.setattr(c, "count", lambda t: 101)
        with pytest.raises(CostGuardError, match="max_rows"):
            c.read_table("t")
        monkeypatch.setattr(c, "_table_bytes", lambda t: 5 * 10**6)
        with pytest.raises(CostGuardError, match="max_bytes"):
            c.stats("t")
        rep = c.cost_report(["t", "u"], pushed={"t": 10**9})
        assert rep["tables"]["t"]["scanned"] == 12 * 5 * 10**6 and rep["scanned"] == 13 * 5 * 10**6
    finally:
        c.close()
    with pytest.raises(ValueError, match="unknown keys"):
        connect(DBX, auth={"method": "token", "token_env": "RS_TEST_DBX_TOKEN"}, guards={"max_cost": 1})


def test_query_tag_is_a_comment_on_every_statement(dbx):
    seen = []
    sa.event.listen(dbx.engine, "before_cursor_execute", lambda *a: seen.append(a[2]))
    hooks = dbx.engine.dispatch.before_cursor_execute
    stmt = "SELECT 1"
    for fn in hooks:
        out = fn(None, None, stmt, None, None, False)
        if isinstance(out, tuple):
            stmt = out[0]
    assert stmt.startswith("/* relsynth */ ")


def test_warehouse_options_are_refused_elsewhere(tmp_path):
    with pytest.raises(ValueError, match="only for cloud warehouses"):
        connect("postgresql://u@h/db", auth={"method": "token"})
    from relsynth.spec import _validate_io
    errs = _validate_io({"source": {"kind": "postgresql", "url": "postgresql://u@h/db", "guards": {}}})
    assert any("cloud warehouses" in e for e in errs)
    errs = _validate_io({"source": {"kind": "databricks", "url": DBX, "auth": {"method": "default"}}})
    assert not [e for e in errs if e.startswith("source")]
