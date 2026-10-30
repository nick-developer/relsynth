# SPDX-License-Identifier: Apache-2.0
"""The Fabric connector without a warehouse: the driver URL, Entra ID tokens,
Fabric column types, NOT ENFORCED constraints and the sample hash's SQL."""
from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mssql

pytest.importorskip("pyodbc", exc_type=ImportError)  # also when libodbc is missing
pytest.importorskip("azure.identity")

from relsynth.connectors import connect  # noqa: E402
from relsynth.connectors import fabric as F  # noqa: E402
from relsynth.connectors.sqlfuncs import row_hash  # noqa: E402

URL = "fabric://abc-123.datawarehouse.fabric.microsoft.com/sales_wh"


class _Cred:
    made: list = []

    def __init__(self, *args, **kw):
        _Cred.made.append(args)

    def get_token(self, scope):
        assert scope == F.SCOPE
        return type("T", (), {"token": "eyJ-entra-token"})()


@pytest.fixture
def fab(monkeypatch):
    from azure import identity
    monkeypatch.setattr(identity, "ClientSecretCredential", _Cred)
    for k, v in {"RS_T_ID": "app-id", "RS_T_SECRET": "s3cret-value", "RS_T_TENANT": "tenant-id"}.items():
        monkeypatch.setenv(k, v)
    c = connect(URL, schema="dbo", auth={"method": "azure-sp", "client_id_env": "RS_T_ID",
                                         "client_secret_env": "RS_T_SECRET", "tenant_id_env": "RS_T_TENANT"})
    yield c
    c.close()


def test_url_gets_the_odbc_driver_and_encryption(fab):
    u = fab.engine.url
    assert u.drivername == "mssql+pyodbc" and u.query["driver"] == F.ODBC_DRIVER and u.query["Encrypt"] == "yes"
    assert "s3cret" not in fab.location() and "s3cret" not in str(fab.ping())
    c = connect(URL, auth={"method": "interactive"})
    assert c.engine.url.query["Authentication"] == "ActiveDirectoryInteractive"
    c.close()


def test_entra_token_reaches_the_driver_per_connection(fab):
    assert _Cred.made[-1] == ("tenant-id", "app-id", "s3cret-value")
    cargs, cparams = ["DRIVER={x};SERVER=h;Trusted_Connection=Yes"], {}
    for fn in fab.engine.dialect.dispatch.do_connect:
        fn(fab.engine.dialect, None, cargs, cparams)
    assert "Trusted_Connection" not in cargs[0]
    assert cparams["attrs_before"][F.SQL_COPT_SS_ACCESS_TOKEN] == F.token_struct("eyJ-entra-token")
    assert F.token_struct("ab") == b"\x04\x00\x00\x00a\x00b\x00"


def test_fabric_types():
    assert isinstance(F.fabric_type(sa.Unicode(10)), mssql.VARCHAR) and F.fabric_type(sa.Unicode(10)).length == 40
    assert F.fabric_type(sa.Unicode(3000)).length is None and F.fabric_type(sa.UnicodeText()).length is None
    assert F.fabric_type(sa.DateTime()).precision == 6 and F.fabric_type(sa.Time()).precision == 6
    assert isinstance(F.fabric_type(sa.Float()), mssql.FLOAT) and F.fabric_type(sa.Float()).precision == 53
    assert isinstance(F.fabric_type(mssql.TINYINT()), sa.SmallInteger)
    assert F.fabric_type(mssql.MONEY()).scale == 4 and isinstance(F.fabric_type(sa.Integer()), sa.Integer)


def test_constraints_are_not_enforced_on_fabric(fab):
    fab._platform = "fabric"
    md = sa.MetaData(schema="dbo")
    p = sa.Table("Parent", md, sa.Column("id", sa.Integer, primary_key=True), sa.Column("code", sa.Unicode(5)),
                 sa.UniqueConstraint("code"))
    sa.Table("child", md, sa.Column("id", sa.Integer, primary_key=True), sa.Column("pid", sa.Integer),
             sa.ForeignKeyConstraint(["pid"], [p.c.id]), sa.Index("ix", "pid"))
    out = fab.adapt_metadata(md)
    eng = sa.create_mock_engine("mssql+pyodbc://", lambda *a, **k: None)
    for t in out.sorted_tables:
        create = str(sa.schema.CreateTable(t).compile(dialect=eng.dialect))
        assert "PRIMARY KEY" not in create and "FOREIGN KEY" not in create and "IDENTITY" not in create
        assert not t.indexes
    stmts = [d for t in out.sorted_tables for d in t.info["rs_constraints"]]
    assert any("PRIMARY KEY NONCLUSTERED (id) NOT ENFORCED" in s for s in stmts)
    assert any("UNIQUE NONCLUSTERED (code) NOT ENFORCED" in s for s in stmts)
    assert any("FOREIGN KEY (pid) REFERENCES dbo.[Parent] (id) NOT ENFORCED" in s for s in stmts)
    assert out.tables["dbo.Parent"].c.code.type.length == 20


def test_text_keys_fit_the_index_limit(fab):
    fab._platform = "sqlserver"
    md = sa.MetaData()
    sa.Table("pair", md, sa.Column("a", sa.Text, primary_key=True), sa.Column("b", sa.String, primary_key=True),
             sa.Column("note", sa.Text))
    out = fab.adapt_metadata(md)
    c = out.tables["pair"].c
    assert c.a.type.length == F.KEY_BYTES // 2 and c.b.type.length == F.KEY_BYTES // 2 and c.note.type.length is None


def test_sample_hash_sql_and_collation_check(fab):
    t = sa.table("t", sa.column("k"))
    sql = str(row_hash([t.c.k], 5, "fabric").compile(dialect=mssql.dialect()))
    assert "hashbytes('SHA2_256'" in sql and "AS VARCHAR(max)" in sql and "'5|'" in sql
    fab._collation = "SQL_Latin1_General_CP1_CI_AS"
    with pytest.raises(ValueError, match="not UTF-8"):
        fab._row_hash([t.c.k], 5)
    fab._collation = "Latin1_General_100_BIN2_UTF8"
    assert fab._row_hash([t.c.k], 5) is not None


def test_copy_into_stages_parquet_in_adls(fab, monkeypatch):
    import pandas as pd
    from azure.storage import filedatalake
    made = []

    class FakeFile:
        def __init__(self, account_url, fs, path, credential=None):
            made.append((account_url, fs, path))

        def upload_data(self, data, overwrite=False):
            assert data[:4] == b"PAR1" and overwrite

        def delete_file(self):
            made.append("deleted")
    monkeypatch.setattr(filedatalake, "DataLakeFileClient", FakeFile)
    sent = []

    class Con:
        dialect = mssql.dialect()

        def exec_driver_sql(self, sql):
            sent.append(sql)
    fab.bulk = {"method": "copy", "stage": "https://acct.dfs.core.windows.net/landing/relsynth/run1"}
    tbl = sa.Table("t", sa.MetaData(schema="dbo"), sa.Column("id", sa.Integer))
    assert fab._insert_rows(Con(), tbl, pd.DataFrame({"id": [1, 2]})) == 2
    acct, fs, path = made[0]
    assert acct == "https://acct.dfs.core.windows.net" and fs == "landing" and path.startswith("relsynth/run1/relsynth_t_")
    assert sent[0].startswith("COPY INTO dbo.t (id) FROM 'https://acct.dfs.core.windows.net/landing/relsynth/run1/relsynth_t_")
    assert "FILE_TYPE = 'PARQUET'" in sent[0] and made[-1] == "deleted"
