# SPDX-License-Identifier: Apache-2.0
"""The Snowflake connector without an account: credentials to driver arguments,
the session query tag, created types and keys, and the sample hash's SQL."""
from __future__ import annotations

import hashlib

import pytest
import sqlalchemy as sa

pytest.importorskip("snowflake.sqlalchemy")

from snowflake.sqlalchemy import TIMESTAMP_NTZ  # noqa: E402

from relsynth.connectors import connect  # noqa: E402
from relsynth.connectors import snowflake as S  # noqa: E402
from relsynth.connectors.sqlfuncs import py_row_hash_sha256, row_hash  # noqa: E402

URL = "snowflake://svc_rs@myorg-acct1/SALES_DB/PUBLIC?warehouse=WH_XS"


def _args(monkeypatch, auth, **kw):
    c = connect(URL, auth=auth, **kw)
    try:
        return c._cargs, c
    finally:
        c.close()


def test_methods_map_to_driver_arguments(monkeypatch, tmp_path):
    key = tmp_path / "rs_key.p8"
    key.write_text("-----BEGIN PRIVATE KEY-----\n")
    monkeypatch.setenv("RS_SF_PASS", "key-pass-123")
    monkeypatch.setenv("RS_SF_TOKEN", "pat-token-456")
    a, _ = _args(monkeypatch, {"method": "key-pair", "private_key_file": str(key),
                               "private_key_passphrase_env": "RS_SF_PASS"})
    assert a["private_key_file"] == str(key) and a["private_key_file_pwd"] == "key-pass-123"
    a, _ = _args(monkeypatch, {"method": "pat", "token_env": "RS_SF_TOKEN"})
    assert a["authenticator"] == "PROGRAMMATIC_ACCESS_TOKEN" and a["token"] == "pat-token-456"
    a, _ = _args(monkeypatch, {"method": "oauth", "token_env": "RS_SF_TOKEN"})
    assert a["authenticator"] == "oauth"
    a, _ = _args(monkeypatch, {"method": "sso"})
    assert a["authenticator"] == "externalbrowser"
    a, _ = _args(monkeypatch, {"method": "workload-identity", "provider": "azure"})
    assert a["authenticator"] == "WORKLOAD_IDENTITY" and a["workload_identity_provider"] == "AZURE"
    with pytest.raises(ValueError, match="private_key_file"):
        connect(URL, auth={"method": "key-pair"})
    with pytest.raises(ValueError, match="never written"):
        connect(URL, auth={"method": "pat", "token": "pat-token-456"})


def test_secrets_stay_out_of_location_and_errors(monkeypatch):
    monkeypatch.setenv("RS_SF_TOKEN", "pat-token-456")
    c = connect(URL, auth={"method": "pat", "token_env": "RS_SF_TOKEN"})
    assert "pat-token-456" not in c.location() and "pat-token-456" not in repr(c)
    assert c.database == "SALES_DB" and c.schema == "PUBLIC"
    c.close()


def test_query_tag_is_a_session_parameter():
    c = connect(URL, guards={"query_tag": "rs-team-a"})
    assert c._cargs["session_parameters"] == {"QUERY_TAG": "rs-team-a"}
    assert not c.engine.dispatch.before_cursor_execute       # no comment is put in front of statements
    c.close()


def test_types_and_declared_keys():
    assert isinstance(S.snowflake_type(sa.DateTime()), TIMESTAMP_NTZ)
    assert type(S.snowflake_type(sa.Float(precision=24))) is sa.Float
    c = connect(URL)
    c._platform = "snowflake"           # no account to ask
    md = sa.MetaData(schema="PUBLIC")
    p = sa.Table("Parent", md, sa.Column("id", sa.Integer, primary_key=True), sa.Column("code", sa.String(5)),
                 sa.UniqueConstraint("code"))
    sa.Table("child", md, sa.Column("id", sa.Integer, primary_key=True), sa.Column("pid", sa.Integer),
             sa.Column("t", sa.DateTime), sa.ForeignKeyConstraint(["pid"], [p.c.id]), sa.Index("ix", "pid"))
    out = c.adapt_metadata(md)
    ddl = {t.name: str(sa.schema.CreateTable(t).compile(dialect=c.engine.dialect)) for t in out.sorted_tables}
    assert "AUTOINCREMENT" not in ddl["child"] and "PRIMARY KEY (id)" in ddl["child"]
    assert 'REFERENCES "PUBLIC"."Parent" (id)' in ddl["child"] and "TIMESTAMP_NTZ" in ddl["child"]
    assert "UNIQUE (code)" in ddl["Parent"] and not out.tables["PUBLIC.child"].indexes
    c.close()


def test_sample_hash_sql_reads_the_first_eight_hex_digits():
    t = sa.table("t", sa.column("k"))
    from snowflake.sqlalchemy import snowdialect
    sql = str(row_hash([t.c.k], 5, "snowflake").compile(dialect=snowdialect.dialect(),
                                                       compile_kwargs={"literal_binds": True}))
    assert "sha2(" in sql and sql.count("charindex(") == 8 and "268435456" in sql
    # the same arithmetic in Python gives the reference hash
    h = hashlib.sha256(b"5|abc").hexdigest()
    assert sum(("0123456789abcdef".index(h[i])) * 16 ** (7 - i) for i in range(8)) == py_row_hash_sha256(5, "abc")


def test_insert_statements_stay_small():
    rows_per_statement = S.rows_per_statement
    recs = [(i, "x" * 100, 1.5) for i in range(10)]
    n = rows_per_statement(recs, 20_000, limit=10_000)
    assert 1 <= n < 100 and n * max(sum(len(repr(v)) + 2 for v in r) for r in recs) <= 10_000
    assert rows_per_statement([], 500) == 500
    assert rows_per_statement([("y" * 50_000,)], 500, limit=1_000) == 1
