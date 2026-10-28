"""Databricks SQL specifics on a live warehouse or the local stand-in (skipped
unless RELSYNTH_TEST_DATABRICKS_URL is set; see tests/emulators/databricks):
values that survive the INSERT literals and Delta types exactly, the SHA-256
sample hash against Python, table names without regard to case, staged
Parquet loads, and the cost guards' table sizes."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from conftest import drop_schema, server_target

from relsynth.connectors import from_spec
from relsynth.connectors.sqlfuncs import HASH_RANGE, py_row_hash_sha256, row_hash
from relsynth.connectors.warehouse import CostGuardError

pytestmark = pytest.mark.databricks


@pytest.fixture
def db(unique_schema):
    tgt = server_target("databricks", unique_schema)
    drop_schema("databricks", unique_schema)
    conn = from_spec(tgt, create=True)
    conn.create_schema()
    yield conn, tgt
    conn.close()
    drop_schema("databricks", unique_schema)


def test_values_round_trip_exactly(db):
    conn, _ = db
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "x": [0.1, 1e-300, -2.5e17, np.nan],
        "s": ["it's", "back\\slash", "ünïcödé ✓ 𝄞", None],
        "ts": pd.to_datetime(["2024-02-29 12:34:56.123456", "1900-01-01", "2099-12-31 23:59:59.999999", None], format="ISO8601"),
        "d": pd.to_datetime(["2024-02-29", "1900-01-01", "2099-12-31", "2000-01-01"]),
        "b": [True, False, None, True],
    })
    conn.write_table("vals", df, tspec={"primary_key": "id",
                                        "columns": {"d": {"type": "date"}, "b": {"type": "boolean"}}})
    back = conn.read_table("vals").sort_values("id").reset_index(drop=True)
    assert back["x"].tolist()[:3] == [0.1, 1e-300, -2.5e17] and np.isnan(back["x"].iloc[3])
    assert back["s"].tolist()[:3] == ["it's", "back\\slash", "ünïcödé ✓ 𝄞"] and back["s"].isna().iloc[3]
    assert back["ts"].iloc[0] == pd.Timestamp("2024-02-29 12:34:56.123456") and pd.isna(back["ts"].iloc[3])
    assert pd.Timestamp(back["d"].iloc[0]) == pd.Timestamp("2024-02-29")
    with conn.engine.connect() as con:
        rows = con.exec_driver_sql(f"DESCRIBE TABLE {conn._qualified('vals')}").fetchall()
    types = {r[0]: r[1].lower() for r in rows if r[0] and not r[0].startswith("#")}
    assert types["x"] == "double" and types["ts"] == "timestamp_ntz" and types["s"] == "string"


def test_sample_hash_is_sha256_of_the_key(db):
    conn, _ = db
    conn.write_table("h", pd.DataFrame({"id": range(1, 201)}), tspec={"primary_key": "id"})
    tbl = conn.table("h")
    import sqlalchemy as sa
    q = sa.select(tbl.c.id, row_hash([tbl.c.id], 11, "databricks").label("h")).order_by(tbl.c.id)
    got = conn.frame(q)
    want = [py_row_hash_sha256(11, str(i)) for i in got["id"]]
    assert got["h"].astype(np.int64).tolist() == want
    assert all(0 <= h < HASH_RANGE for h in want)


def test_table_names_are_found_without_regard_to_case(db):
    conn, _ = db
    conn.write_table("Album", pd.DataFrame({"AlbumId": [1, 2], "Title": ["a", "b"]}), tspec={"primary_key": "AlbumId"})
    conn.refresh()
    assert conn.has_table("Album") and conn.has_table("album")
    assert conn.count("Album") == 2 and conn.read_table("Album").columns.tolist() == ["AlbumId", "Title"]
    assert conn.describe("Album").name == "Album"


def test_staged_parquet_load(unique_schema, tmp_path):
    """bulk method path: one Parquet file per table, loaded with COPY INTO (the
    stand-in turns it into INSERT ... SELECT).  The location must be one the
    warehouse reads: RELSYNTH_TEST_DATABRICKS_STAGE, or a local folder on the stand-in."""
    stage = os.environ.get("RELSYNTH_TEST_DATABRICKS_STAGE") or (
        str(tmp_path) if "localhost" in os.environ.get("RELSYNTH_TEST_DATABRICKS_URL", "") else None)
    if not stage:
        pytest.skip("set RELSYNTH_TEST_DATABRICKS_STAGE to a location the warehouse reads")
    tgt = {**server_target("databricks", unique_schema), "bulk": {"method": "path", "path": stage}}
    drop_schema("databricks", unique_schema)
    conn = from_spec(tgt, create=True)
    try:
        conn.create_schema()
        df = pd.DataFrame({"id": range(1, 1001), "x": np.linspace(0, 1, 1000), "s": [f"v{i}" for i in range(1000)],
                           "t": pd.date_range("2020-01-01", periods=1000, freq="h")})
        conn.write_table("staged", df, tspec={"primary_key": "id"})
        back = conn.read_table("staged").sort_values("id").reset_index(drop=True)
        pd.testing.assert_frame_equal(back, df, check_dtype=False)
    finally:
        conn.close()
        drop_schema("databricks", unique_schema)


def test_cost_guards_use_the_delta_size(db, unique_schema):
    conn, tgt = db
    conn.write_table("big", pd.DataFrame({"id": range(1, 501), "v": ["x" * 50] * 500}), tspec={"primary_key": "id"})
    size = conn.table_bytes("big")
    assert size and size > 0
    rep = conn.cost_report(["big"], pushed={"big": 500})
    assert rep["tables"]["big"]["bytes"] == size and rep["scanned"] == 12 * size
    guarded = from_spec({**tgt, "guards": {"max_bytes": size - 1, "max_rows": 400}})
    try:
        with pytest.raises(CostGuardError, match="max_rows"):
            guarded.read_table("big")
        with pytest.raises(CostGuardError, match="max_bytes"):
            guarded.stats("big")
    finally:
        guarded.close()


def test_query_tag_reaches_the_warehouse(db):
    conn, tgt = db
    tagged = from_spec({**tgt, "guards": {"query_tag": "rs-tag-check"}})
    seen = []
    import sqlalchemy as sa
    sa.event.listen(tagged.engine, "after_cursor_execute", lambda c, cur, stmt, *a: seen.append(stmt))
    try:
        tagged.tables()
        assert seen and all(s.startswith("/* rs-tag-check */") for s in seen)
    finally:
        tagged.close()

