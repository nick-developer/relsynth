# SPDX-License-Identifier: Apache-2.0
"""Snowflake specifics on a live account or the fakesnow stand-in (skipped unless
RELSYNTH_TEST_SNOWFLAKE_URL is set; see tests/emulators/snowflake): values that
survive the created types exactly, the SHA-256 sample hash against Python,
mixed-case names, staged loads and the query tag."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from conftest import drop_schema, server_target

from relsynth.connectors import from_spec
from relsynth.connectors.sqlfuncs import HASH_RANGE, py_row_hash_sha256

pytestmark = pytest.mark.snowflake


@pytest.fixture
def db(unique_schema):
    tgt = server_target("snowflake", unique_schema)
    drop_schema("snowflake", unique_schema)
    conn = from_spec(tgt, create=True)
    conn.create_schema()
    yield conn, tgt
    conn.close()
    drop_schema("snowflake", unique_schema)


VALS = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "x": [0.1, 1e-300, -2.5e17, np.nan],
    "s": ["it's", "back\\slash", "ünïcödé ✓ 𝄞", None],
    "ts": pd.to_datetime(["2024-02-29 12:34:56.123456", "1900-01-01", "2099-12-31 23:59:59.999999", None],
                         format="ISO8601"),
    "b": [True, False, None, True],
})


def _check_vals(back):
    back = back.sort_values("id").reset_index(drop=True)
    assert back["x"].tolist()[:3] == [0.1, 1e-300, -2.5e17] and np.isnan(back["x"].iloc[3])
    assert back["s"].tolist()[:3] == VALS["s"].tolist()[:3] and back["s"].isna().iloc[3]
    assert back["ts"].iloc[0] == pd.Timestamp("2024-02-29 12:34:56.123456") and pd.isna(back["ts"].iloc[3])
    assert back["ts"].iloc[2] == pd.Timestamp("2099-12-31 23:59:59.999999")


def test_values_round_trip_exactly(db):
    conn, _ = db
    conn.write_table("vals", VALS.copy(), tspec={"primary_key": "id", "columns": {"b": {"type": "boolean"}}})
    _check_vals(conn.read_table("vals"))


def test_staged_load(unique_schema):
    """bulk method stage: Parquet PUT in the table stage, then COPY INTO (write_pandas)."""
    tgt = {**server_target("snowflake", unique_schema), "bulk": {"method": "stage"}}
    drop_schema("snowflake", unique_schema)
    conn = from_spec(tgt, create=True)
    if conn.platform != "snowflake":
        conn.close()
        pytest.skip("the stand-in has no PUT to an internal stage")
    try:
        conn.create_schema()
        conn.write_table("vals", VALS.copy(), tspec={"primary_key": "id", "columns": {"b": {"type": "boolean"}}})
        _check_vals(conn.read_table("vals"))
        df = pd.DataFrame({"id": range(1, 2001), "v": np.linspace(0, 1, 2000)})
        conn.write_table("Many", df, tspec={"primary_key": "id"})
        assert conn.count("Many") == 2000
    finally:
        conn.close()
        drop_schema("snowflake", unique_schema)


def test_sample_hash_is_sha256_of_the_key(db):
    conn, _ = db
    keys = [f"k{i}" for i in range(150)] + ["ünï", "✓", "𝄞", "a|b"]
    conn.write_table("h", pd.DataFrame({"k": keys}), tspec={"primary_key": "k"})
    tbl = conn.table("h")
    got = conn.frame(sa.select(tbl.c.k, conn._row_hash([tbl.c.k], 11).label("h")))
    assert got["h"].astype(np.int64).tolist() == [py_row_hash_sha256(11, k) for k in got["k"]]
    assert all(0 <= h < HASH_RANGE for h in got["h"])


def test_mixed_case_names_keep_their_case(db):
    conn, _ = db
    conn.write_table("Album", pd.DataFrame({"AlbumId": [1, 2], "Title": ["a", "b"]}), tspec={"primary_key": "AlbumId"})
    conn.refresh()
    assert "Album" in conn.tables() and conn.count("Album") == 2
    assert conn.read_table("Album").columns.tolist() == ["AlbumId", "Title"]


def test_query_tag_is_the_session_tag(db):
    _, tgt = db
    tagged = from_spec({**tgt, "guards": {"query_tag": "rs-tag-check"}})
    try:
        assert tagged._cargs["session_parameters"]["QUERY_TAG"] == "rs-tag-check"
        if tagged.platform == "snowflake":          # the stand-in keeps no session parameters
            with tagged.engine.connect() as con:
                rows = con.exec_driver_sql("SHOW PARAMETERS LIKE 'QUERY_TAG' IN SESSION").fetchall()
            assert rows and rows[0][1] == "rs-tag-check"
    finally:
        tagged.close()
