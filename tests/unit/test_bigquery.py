# SPDX-License-Identifier: Apache-2.0
"""The BigQuery connector without a project: credentials, job labels and the
byte limit, dry-run cost guards, created types and NOT ENFORCED keys, the sample
hash's SQL and load jobs.  No BigQuery emulator could be run here, so nothing
in this file reaches a server; see docs/warehouses.md."""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest
import sqlalchemy as sa

pytest.importorskip("sqlalchemy_bigquery")

import google.auth  # noqa: E402
from google.auth.credentials import AnonymousCredentials  # noqa: E402

from relsynth.connectors import connect  # noqa: E402
from relsynth.connectors import bigquery as B  # noqa: E402
from relsynth.connectors.sqlfuncs import py_row_hash_sha256, row_hash  # noqa: E402
from relsynth.connectors.warehouse import CostGuardError  # noqa: E402

URL = "bigquery://rs-project/sales?location=EU"


@pytest.fixture
def anon(monkeypatch):
    monkeypatch.setattr(google.auth, "default", lambda scopes=None: (AnonymousCredentials(), "rs-project"))


@pytest.fixture
def bq(anon):
    c = connect(URL, guards={"query_tag": "Team A/test", "max_bytes": "10GB"})
    yield c
    c.close()


def test_url_project_dataset_and_own_client(bq):
    assert bq.project == "rs-project" and bq.schema == "sales" and bq.location == "EU"
    assert bq.engine.url.query["user_supplied_client"] == "true"
    cl = bq.client()
    assert cl.project == "rs-project" and cl.location == "EU"
    cfg = cl._default_query_job_config
    assert cfg.labels == {"relsynth": "team_a_test"} and cfg.maximum_bytes_billed == 10 * 10**9
    assert str(cfg.default_dataset) == "rs-project.sales" or cfg.default_dataset.dataset_id == "sales"


def test_auth_methods(anon, monkeypatch, tmp_path):
    from google.oauth2 import service_account
    key = tmp_path / "sa.json"
    key.write_text("{}")
    seen = {}
    monkeypatch.setattr(service_account.Credentials, "from_service_account_file",
                        classmethod(lambda cls, path, scopes=None: seen.setdefault("key", path) and AnonymousCredentials()))
    c = connect(URL, auth={"method": "service-account", "key_file": str(key)})
    c.credentials()
    assert seen["key"] == str(key)
    c.close()
    c = connect(URL, auth={"method": "impersonate", "service_account": "rs@p.iam.gserviceaccount.com"})
    from google.auth import impersonated_credentials
    assert isinstance(c.credentials(), impersonated_credentials.Credentials)
    c.close()
    with pytest.raises(ValueError, match="key_file"):
        connect(URL, auth={"method": "service-account"}).credentials()
    with pytest.raises(ValueError, match="never written"):
        connect(URL, auth={"method": "default", "private_key": "-----BEGIN"}).credentials()


def test_label_values():
    assert B.label_value("relsynth") == "relsynth" and B.label_value("A b.C") == "a_b_c"
    assert len(B.label_value("x" * 100)) == 63


def test_dry_run_guard(bq, monkeypatch):
    class Job:
        total_bytes_processed = 12 * 10**9
    calls = []

    def query(sql, job_config=None):
        calls.append((sql, job_config))
        return Job()
    monkeypatch.setattr(bq.client(), "query", query)
    tbl = sa.Table("fact", sa.MetaData(schema="sales"), sa.Column("id", sa.Integer), sa.Column("v", sa.Float))
    monkeypatch.setattr(bq, "table", lambda name: tbl)
    with pytest.raises(CostGuardError, match="max_bytes"):
        bq.guard_scan("fact", what="read")
    sql, cfg = calls[-1]
    assert cfg.dry_run and not cfg.use_query_cache and "`sales`.`fact`" in sql
    rep = bq.cost_extra()
    assert rep["job_label"] == "relsynth=team_a_test"


def test_types_and_not_enforced_keys(bq):
    md = sa.MetaData(schema="sales")
    p = sa.Table("parent", md, sa.Column("id", sa.Integer, primary_key=True), sa.Column("code", sa.String(5)),
                 sa.UniqueConstraint("code"))
    sa.Table("child", md, sa.Column("id", sa.Integer, primary_key=True), sa.Column("pid", sa.Integer),
             sa.Column("t", sa.DateTime), sa.Column("x", sa.Float), sa.ForeignKeyConstraint(["pid"], [p.c.id]))
    out = bq.adapt_metadata(md)
    ddl = {t.name: str(sa.schema.CreateTable(t).compile(dialect=bq.engine.dialect)) for t in out.sorted_tables}
    assert "PRIMARY KEY" not in ddl["child"] and "UNIQUE" not in ddl["parent"]
    assert "`code` STRING" in ddl["parent"] and "STRING(5)" not in ddl["parent"]
    assert "`t` DATETIME" in ddl["child"] and "`x` FLOAT64" in ddl["child"] and "`id` INT64 NOT NULL" in ddl["child"]
    stmts = [d for t in out.sorted_tables for d in t.info["rs_constraints"]]
    assert "ALTER TABLE `rs-project.sales.parent` ADD PRIMARY KEY (`id`) NOT ENFORCED" in stmts
    assert any("ADD FOREIGN KEY (`pid`) REFERENCES `rs-project.sales.parent` (`id`) NOT ENFORCED" in s for s in stmts)


def test_sample_hash_sql():
    from sqlalchemy_bigquery import BigQueryDialect
    t = sa.table("t", sa.column("k"))
    sql = str(row_hash([t.c.k], 5, "bigquery").compile(dialect=BigQueryDialect(), compile_kwargs={"literal_binds": True}))
    assert "sha256(" in sql.lower() and "to_hex(" in sql.lower() and "'0x'" in sql and "AS INT64" in sql
    h = hashlib.sha256(b"5|abc").hexdigest()
    assert int("0x" + h[:8], 16) == py_row_hash_sha256(5, "abc")


def test_loads_use_load_jobs(bq, monkeypatch):
    got = {}

    class Job:
        output_rows = 3

        def result(self):
            return self

    def load(df, dest, job_config=None):
        got.update(df=df, dest=dest, cfg=job_config)
        return Job()
    monkeypatch.setattr(bq.client(), "load_table_from_dataframe", load)
    tbl = sa.Table("t", sa.MetaData(schema="sales"), sa.Column("id", sa.Integer))
    assert bq._insert_rows(None, tbl, pd.DataFrame({"id": [1, 2, 3]})) == 3
    assert got["dest"] == "rs-project.sales.t" and got["cfg"].write_disposition == "WRITE_APPEND"
    assert got["cfg"].labels == {"relsynth": "team_a_test"}
