"""MySQL 8 and MariaDB specifics on live servers (skipped unless
RELSYNTH_TEST_MYSQL_URL / RELSYNTH_TEST_MARIADB_URL is set): exact text
comparison under case-insensitive collations, the SHA-256 sample hash, types that
keep the data (DOUBLE, DATETIME(6), 4-byte UTF-8), MySQL's index and row-size
limits when cloning, and a MariaDB server reached through a mysql:// URL."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from conftest import drop_schema, server_target, server_url

import relsynth as rs
from relsynth.connectors import connect, from_spec
from relsynth.connectors.sqlfuncs import HASH_RANGE, py_row_hash_sha256, row_hash
from relsynth.pushdown.stats import FrameStats, Require

KINDS = [pytest.param("mysql", marks=pytest.mark.mysql), pytest.param("mariadb", marks=pytest.mark.mariadb)]


@pytest.fixture(params=KINDS)
def db(request, unique_schema):
    kind = request.param
    tgt = server_target(kind, unique_schema)
    drop_schema(kind, unique_schema)
    conn = from_spec(tgt, create=True)
    conn.create_schema()
    yield kind, conn, unique_schema
    conn.close()
    drop_schema(kind, unique_schema)


def _exec(conn, sql, **params):
    with conn.engine.begin() as con:
        con.execute(sa.text(sql), params)


def test_text_is_compared_exactly_under_a_case_insensitive_collation(db):
    """A source table in the server's default collation (case- and
    accent-insensitive) keeps 'abc', 'ABC', 'abc ' and 'àbc' apart in every
    statistic, as the in-memory reference does."""
    kind, conn, schema = db
    coll = "utf8mb4_0900_ai_ci" if kind == "mysql" else "utf8mb4_uca1400_ai_ci"
    q = conn.engine.dialect.identifier_preparer.quote_identifier
    _exec(conn, f"CREATE TABLE {q(schema)}.t_ci (id INT PRIMARY KEY, k VARCHAR(20) COLLATE {coll}, "
                f"x DOUBLE) COLLATE {coll}")
    keys = ["abc", "ABC", "abc ", "àbc", "abc", None, "xyz"]
    df = pd.DataFrame({"id": range(1, 8), "k": keys, "x": [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
    with conn.engine.begin() as con:
        con.execute(sa.text(f"INSERT INTO {q(schema)}.t_ci VALUES (:id, :k, :x)"),
                    df.astype(object).where(df.notna(), None).to_dict("records"))
    sq, fs = conn.stats("t_ci"), FrameStats(conn.read_table("t_ci"))
    assert sorted(sq.distinct("k")) == sorted(fs.distinct("k")) == ["ABC", "abc", "abc ", "xyz", "àbc"]
    a = sq.group_counts(["k"], ["x"]).sort_values("k", na_position="first").reset_index(drop=True)
    b = fs.group_counts(["k"], ["x"]).sort_values("k", na_position="first").reset_index(drop=True)
    assert a["k"].fillna("<null>").tolist() == b["k"].fillna("<null>").tolist()
    assert a["n"].tolist() == b["n"].tolist()
    assert np.allclose(a["s0"].to_numpy(float), b["s0"].to_numpy(float), atol=1e-8)
    req = {"k": Require(values=["abc", "xyz"])}
    assert sq.group_counts(["k"], require=req)["n"].sum() == fs.group_counts(["k"], require=req)["n"].sum() == 3
    h1, d1 = sq.first_rows("k", ["x"])
    h2, d2 = fs.first_rows("k", ["x"])
    assert len(h1) == len(h2) == 6 and d1 == d2


def test_sample_hash_is_sha256_of_the_key(db):
    """The server's row hash equals py_row_hash_sha256, so samples can be checked
    outside the database, and it is uniform (Proposition 103)."""
    kind, conn, schema = db
    n = 5000
    conn.write_table("t_hash", pd.DataFrame({"id": np.arange(1, n + 1), "v": np.arange(n) % 7}),
                     tspec={"primary_key": "id"})
    t = conn.table("t_hash")
    with conn.engine.connect() as con:
        got = con.execute(sa.select(t.c.id, row_hash([t.c.id], 11, conn.dialect)).order_by(t.c.id)).fetchall()
    assert [int(h) for _, h in got] == [py_row_hash_sha256(11, str(i)) for i, _ in got]
    from scipy.stats import kstest
    assert kstest(np.array([h for _, h in got], float) / HASH_RANGE, "uniform").pvalue > 1e-4
    s1, whole = conn.sample("t_hash", 1000, seed=3)
    s2, _ = conn.sample("t_hash", 1000, seed=3)
    assert not whole and s1["id"].tolist() == s2["id"].tolist()
    expect = [i for i in range(1, n + 1) if py_row_hash_sha256(3, str(i)) < round(HASH_RANGE * 1000 / n)]
    assert sorted(s1["id"].tolist()) == expect


def test_types_keep_the_values(db):
    """Doubles keep 17 digits, timestamps keep microseconds, text keeps 4-byte
    UTF-8 and exact case; a unique key on text that differs only in case loads."""
    kind, conn, _ = db
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "x": [0.1 + 0.2, 1e-300, 123456789.123456789],
        "ts": pd.to_datetime(["2024-01-02 03:04:05.123456", "1999-12-31 23:59:59.999999", "2030-06-01 00:00:00.000000"]),
        "code": ["abc", "ABC", "abc "],
        "txt": ["😀 émoji", "日本語", "x" * 3000],
        "flag": [True, False, True]})
    conn.write_table("t_types", df, tspec={"primary_key": "id", "unique": [["code"]],
                                           "columns": {"flag": {"type": "boolean"}}})
    back = conn.read_table("t_types").sort_values("id").reset_index(drop=True)
    assert back["x"].tolist() == df["x"].tolist()
    assert (pd.to_datetime(back["ts"]) == df["ts"]).all()
    assert back["code"].tolist() == df["code"].tolist() and back["txt"].tolist() == df["txt"].tolist()
    assert [bool(v) for v in back["flag"]] == df["flag"].tolist()


def test_clone_text_keys_and_wide_tables_from_sqlite(db, tmp_path):
    """SQLite TEXT keys become VARCHAR within MySQL's 3072-byte index limit, and a
    wide table of long VARCHARs fits the 65,535-byte row limit."""
    kind, conn, schema = db
    lite = connect(f"sqlite:{tmp_path / 'k.db'}")
    wide = ", ".join(f"c{i} VARCHAR(2000)" for i in range(40))
    with lite.engine.begin() as con:
        con.execute(sa.text("CREATE TABLE p (code TEXT PRIMARY KEY, name TEXT)"))
        con.execute(sa.text("CREATE TABLE c (id INTEGER PRIMARY KEY, a TEXT, b TEXT, "
                            "code TEXT REFERENCES p(code), UNIQUE (a, b))"))
        con.execute(sa.text(f"CREATE TABLE w (id INTEGER PRIMARY KEY, {wide})"))
        con.execute(sa.text("INSERT INTO p VALUES ('K1', 'one'), ('k1', 'other case')"))
        con.execute(sa.text("INSERT INTO c VALUES (1, 'a', 'b', 'K1'), (2, 'A', 'b', 'k1')"))
    made = rs.clone_schema(f"sqlite:{tmp_path / 'k.db'}", server_target(kind, schema))
    assert set(made) == {"p", "c", "w"}
    conn.refresh()
    info = conn.describe("c")
    assert info.foreign_keys and info.foreign_keys[0].ref_table == "p"
    data = {"p": lite.read_table("p"), "c": lite.read_table("c")}
    spec = {"tables": {"p": {"primary_key": "code", "columns": {}}, "c": {"primary_key": "id", "columns": {}}},
            "relationships": []}
    conn.write_tables(spec, data, mode="truncate", structure="existing")
    assert conn.count("p") == 2 and conn.count("c") == 2
    lite.close()


def test_mysql_url_on_a_mariadb_server_is_mariadb(db):
    kind, conn, schema = db
    if kind != "mariadb":
        pytest.skip("MariaDB only")
    url = sa.engine.make_url(server_url("mariadb")).set(drivername="mysql")
    c = connect(url.render_as_string(hide_password=False), schema=schema)
    try:
        assert c.kind == "mariadb" and c.text_collation == "utf8mb4_nopad_bin"
        assert c.ping()["ok"]
    finally:
        c.close()


def test_time_zone_is_utc(db):
    _, conn, _ = db
    with conn.engine.connect() as con:
        assert con.exec_driver_sql("SELECT @@session.time_zone").scalar() == "+00:00"
    assert isinstance(dt.timezone.utc, dt.tzinfo)
