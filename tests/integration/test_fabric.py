# SPDX-License-Identifier: Apache-2.0
"""Microsoft Fabric specifics on a live warehouse or the SQL Server 2022 stand-in
(skipped unless RELSYNTH_TEST_FABRIC_URL is set; see tests/emulators/fabric):
the Fabric column types, values that survive them exactly, the UTF-8 SHA-256
sample hash against Python, exact text statistics in a case-insensitive
collation, and the query tag."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from conftest import drop_schema, server_target

from relsynth.connectors import connect, from_spec
from relsynth.connectors.sqlfuncs import HASH_RANGE, py_row_hash_sha256

pytestmark = pytest.mark.fabric


@pytest.fixture
def db(unique_schema):
    tgt = server_target("fabric", unique_schema)
    drop_schema("fabric", unique_schema)
    conn = from_spec(tgt, create=True)
    conn.create_schema()
    yield conn, tgt
    conn.close()
    drop_schema("fabric", unique_schema)


def _types(conn, table):
    q = ("SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, DATETIME_PRECISION, NUMERIC_PRECISION "
         "FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?")
    with conn.engine.connect() as con:
        return {r[0]: r[1:] for r in con.exec_driver_sql(q, (conn.schema, table)).fetchall()}


def test_fabric_types_and_exact_values(db):
    conn, _ = db
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "x": [0.1, 1e-300, -2.5e17, np.nan],
        "s": ["it's", "ünïcödé ✓ 𝄞", "x" * 3000, None],
        "ts": pd.to_datetime(["2024-02-29 12:34:56.123456", "1900-01-01", "2099-12-31 23:59:59.999999", None],
                             format="ISO8601"),
        "b": [True, False, None, True],
    })
    conn.write_table("vals", df, tspec={"primary_key": "id", "columns": {"b": {"type": "boolean"}}})
    t = _types(conn, "vals")
    assert t["s"][0] == "varchar" and t["s"][1] == -1                 # VARCHAR(MAX): 3000 chars > 8000 / 4
    assert t["ts"][0] == "datetime2" and t["ts"][2] == 6
    assert t["x"][0] == "float" and t["x"][3] == 53
    assert not any(v[0] in ("nvarchar", "nchar", "ntext", "text", "datetime", "tinyint", "money") for v in t.values())
    back = conn.read_table("vals").sort_values("id").reset_index(drop=True)
    assert back["x"].tolist()[:3] == [0.1, 1e-300, -2.5e17] and np.isnan(back["x"].iloc[3])
    assert back["s"].tolist()[:3] == df["s"].tolist()[:3] and back["s"].isna().iloc[3]
    assert back["ts"].iloc[0] == pd.Timestamp("2024-02-29 12:34:56.123456") and pd.isna(back["ts"].iloc[3])


def test_sample_hash_is_sha256_of_the_utf8_key(db):
    conn, _ = db
    keys = [f"k{i}" for i in range(150)] + ["ünï", "✓", "𝄞", "a|b", "日本"]
    conn.write_table("h", pd.DataFrame({"k": keys}), tspec={"primary_key": "k"})
    tbl = conn.table("h")
    got = conn.frame(sa.select(tbl.c.k, conn._row_hash([tbl.c.k], 11).label("h")))
    want = [py_row_hash_sha256(11, k) for k in got["k"]]
    assert got["h"].astype(np.int64).tolist() == want
    assert all(0 <= h < HASH_RANGE for h in want)


def test_statistics_are_exact_in_a_case_insensitive_collation(unique_schema):
    """A Fabric warehouse may be created case-insensitive
    (Latin1_General_100_CI_AS_KS_WS_SC_UTF8): grouping still sees 'a' and 'A'."""
    url = os.environ.get("RELSYNTH_TEST_FABRIC_URL")
    if not url:
        pytest.skip("RELSYNTH_TEST_FABRIC_URL is not set")
    if "localhost" not in url and "127.0.0.1" not in url:
        pytest.skip("creates a database: stand-in only")
    master = sa.engine.make_url(url).set(drivername="mssql+pyodbc", database="master").update_query_dict(
        {"driver": "ODBC Driver 18 for SQL Server"}).render_as_string(hide_password=False)
    eng = sa.create_engine(master, isolation_level="AUTOCOMMIT")
    with eng.connect() as con:
        if not con.exec_driver_sql("SELECT 1 FROM sys.databases WHERE name = 'rs_fabric_ci'").first():
            con.exec_driver_sql("CREATE DATABASE rs_fabric_ci COLLATE Latin1_General_100_CI_AS_KS_WS_SC_UTF8")
    eng.dispose()
    ci = sa.engine.make_url(url).set(database="rs_fabric_ci").render_as_string(hide_password=False)
    conn = connect(ci, schema=unique_schema)
    try:
        conn.create_schema()
        conn.write_table("t", pd.DataFrame({"id": [1, 2, 3, 4], "s": ["a", "A", "á", "a"]}), tspec={"primary_key": "id"})
        assert "CI" in conn.collation
        assert sorted(conn.stats("t").distinct("s").tolist()) == ["A", "a", "á"]
        tbl = conn.table("t")
        assert conn.frame(sa.select(conn._row_hash([tbl.c.s], 3).label("h")).where(tbl.c.id == 2))["h"].iloc[0] \
            == py_row_hash_sha256(3, "A")
    finally:
        with conn.engine.begin() as con:
            con.exec_driver_sql(f"DROP TABLE IF EXISTS [{unique_schema}].[t]")
            con.exec_driver_sql(f"DROP SCHEMA IF EXISTS [{unique_schema}]")
        conn.close()


def test_query_tag_reaches_the_server(db):
    _, tgt = db
    tagged = from_spec({**tgt, "guards": {"query_tag": "rs-tag-check"}})
    seen = []
    sa.event.listen(tagged.engine, "after_cursor_execute", lambda c, cur, stmt, *a: seen.append(stmt))
    try:
        tagged.tables()
        assert seen and all(s.startswith("/* rs-tag-check */") for s in seen)
    finally:
        tagged.close()
