# SPDX-License-Identifier: Apache-2.0
"""Live PostgreSQL, SQL Server, Oracle, MySQL and MariaDB (skipped unless
RELSYNTH_TEST_*_URL is set; docker-compose.test.yml starts all five).

Each test works in a fresh schema (Oracle: the connecting user's schema, with
the tables dropped afterwards)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from conftest import drop_schema, scenario, server_target, write_scenario

import relsynth as rs
from relsynth.errors import SourceError
from relsynth.connectors import from_spec
from relsynth.pushdown.stats import FrameStats, Require

KINDS = [pytest.param("postgresql", marks=pytest.mark.postgres), pytest.param("mssql", marks=pytest.mark.mssql),
         pytest.param("oracle", marks=pytest.mark.oracle), pytest.param("mysql", marks=pytest.mark.mysql),
         pytest.param("mariadb", marks=pytest.mark.mariadb)]
STAR = ("customer", "product", "sales")


@pytest.fixture(params=KINDS)
def server(request, unique_schema):
    kind = request.param
    tgt = server_target(kind, unique_schema)
    drop_schema(kind, unique_schema, tables=STAR + ("t_stats", "evil"))
    yield kind, tgt, unique_schema
    drop_schema(kind, unique_schema, tables=STAR + ("t_stats", "evil"))


@pytest.fixture
def star_on_server(server):
    kind, tgt, schema = server
    spec = write_scenario("S2_star", tgt)
    return kind, spec, schema


def test_write_read_and_catalog(star_on_server):
    kind, spec, _ = star_on_server
    t, _, _ = scenario("S2_star")
    conn = from_spec(spec["source"])
    try:
        assert {n.lower() for n in conn.tables()} >= set(STAR)
        for name, df in t.items():
            assert conn.count(name) == len(df)
        info = conn.describe("sales")
        assert {f.ref_table.lower() for f in info.foreign_keys} == {"customer", "product"}
        back = conn.read_table("customer")
        assert len(back) == len(t["customer"])
        assert conn.ping()["ok"]
    finally:
        conn.close()


def test_introspect_fit_and_pushdown_run(star_on_server, tmp_path):
    kind, spec, _ = star_on_server
    draft = rs.introspect(spec["source"], include=["^(customer|product|sales)$"])
    assert {t.lower() for t in draft.tables} == set(STAR)
    s = copy.deepcopy(spec)
    s["synthesis"] = {"pushdown": {"mode": "on", "sample_rows": 4000, "min_rows": 0}, "seed": 3}
    model = rs.fit(s)
    assert model.pushdown["pushed"] == {"sales": len(scenario("S2_star")[0]["sales"])}
    rep = model.evaluate(model.sample(dataset=0), privacy=False)
    assert rep.score >= 0.95, rep.failed
    # reproducible: the database's hash gives the same sample again
    again = rs.fit(s).sample(dataset=0)
    pd.testing.assert_frame_equal(model.sample(dataset=0)["sales"].reset_index(drop=True),
                                  again["sales"].reset_index(drop=True), check_dtype=False)


def test_full_sample_pushdown_matches_sqlite_in_memory(star_on_server, star_sqlite):
    kind, spec, _ = star_on_server
    s = copy.deepcopy(spec)
    s["synthesis"] = {"pushdown": {"mode": "on", "sample_rows": 10**7, "min_rows": 0}, "seed": 8}
    lite = copy.deepcopy(star_sqlite[0])
    lite["synthesis"] = {"pushdown": {"mode": "off"}, "seed": 8}
    a, b = rs.fit(s).sample(dataset=0), rs.fit(lite).sample(dataset=0)
    # rows are put in a canonical order on load, so the server and the file give one model
    for t in STAR:
        pd.testing.assert_frame_equal(a[t].reset_index(drop=True), b[t].reset_index(drop=True),
                                      check_dtype=False, obj=t)


def test_write_synthetic_to_server_in_every_mode(server, star_sqlite):
    kind, tgt, schema = server
    spec, _ = star_sqlite
    data = rs.fit(spec, seed=1).sample(dataset=0)
    data.write(tgt)
    conn = from_spec(tgt)
    try:
        assert conn.count("sales") == len(data["sales"])
    finally:
        conn.close()
    data.write({**tgt, "if_exists": "truncate", "structure": "existing"})
    with pytest.raises(SourceError, match="if_exists: fail"):
        data.write({**tgt, "if_exists": "fail"})
    conn = from_spec(tgt)
    try:
        assert conn.count("sales") == len(data["sales"])
    finally:
        conn.close()


def test_clone_schema_from_sqlite_and_back(server, tmp_path, star_sqlite):
    kind, tgt, schema = server
    _, db = star_sqlite
    made = rs.clone_schema(f"sqlite:{db}", tgt)
    assert made.index("customer") < made.index("sales")
    conn = from_spec(tgt)
    try:
        assert all(conn.count(t) == 0 for t in STAR)
        assert {f.ref_table.lower() for f in conn.describe("sales").foreign_keys} == {"customer", "product"}
    finally:
        conn.close()
    back = rs.clone_schema(tgt, f"sqlite:{tmp_path / 'back.db'}", tables=list(STAR))
    assert {t.lower() for t in back} == set(STAR)


def _frame(n=2000, seed=4):
    g = np.random.default_rng(seed)
    return pd.DataFrame({"id": np.arange(1, n + 1), "a": g.integers(1, 30, n),
                         "b": np.where(g.random(n) < 0.1, np.nan, g.integers(1, 9, n)),
                         "x": np.where(g.random(n) < 0.05, np.nan, g.gamma(2.0, 3.0, n).round(2)),
                         "y": g.normal(size=n).round(1), "k": g.integers(0, 40, n)})


def test_statistics_match_the_in_memory_reference(server):
    """Proposition P1 on a live server: GROUP BY counts exact, normal-score sums
    within the Acklam bound (P3)."""
    kind, tgt, schema = server
    df = _frame()
    conn = from_spec(tgt, create=True)
    try:
        if hasattr(conn, "create_schema"):
            conn.create_schema(schema if kind != "oracle" else None)
        conn.write_table("t_stats", df, mode="replace", tspec={"primary_key": "id"})
        sq, fs = conn.stats("t_stats"), FrameStats(df)
        assert sq.n_rows == fs.n_rows and sq.notnull("x") == fs.notnull("x")
        a = sq.group_counts(["a", "b"], ["x", "y"]).sort_values(["a", "b"], na_position="first").reset_index(drop=True)
        b = fs.group_counts(["a", "b"], ["x", "y"]).sort_values(["a", "b"], na_position="first").reset_index(drop=True)
        assert (a["n"].to_numpy() == b["n"].to_numpy()).all()
        for c in a.columns[3:]:
            assert np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)).max() < 1e-6, c
        req = {"b": Require(values=list(range(1, 9)), exclude=[3])}
        assert sq.group_counts(["a"], require=req)["n"].sum() == fs.group_counts(["a"], require=req)["n"].sum()
        h1, d1 = sq.first_rows("k", ["a"])
        h2, d2 = fs.first_rows("k", ["a"])
        assert d1 == d2 and len(h1) == len(h2)
        s1, _ = conn.sample("t_stats", 500, seed=2)
        s2, _ = conn.sample("t_stats", 500, seed=2)
        assert s1.sort_values("id").reset_index(drop=True).equals(s2.sort_values("id").reset_index(drop=True))
        assert abs(len(s1) - 500) < 6 * np.sqrt(2000 * 0.25 * 0.75)
    finally:
        conn.close()


def test_hostile_identifiers_on_server(server):
    kind, tgt, schema = server
    conn = from_spec(tgt, create=True)
    evil_col = "v'); DROP TABLE customer; --"
    try:
        if hasattr(conn, "create_schema"):
            conn.create_schema(schema if kind != "oracle" else None)
        conn.write_table("evil", pd.DataFrame({"id": [1, 2], evil_col: [3, 4]}))
        back = conn.read_table("evil")
        assert back.columns.tolist()[1].lower() == evil_col.lower() and back.iloc[:, 1].tolist() == [3, 4]
        st = conn.stats("evil")
        assert st.n_rows == 2
    finally:
        conn.close()


def test_profile_and_masked_errors(server):
    kind, tgt, schema = server
    url = tgt["url"]
    pw = sa.engine.make_url(url).password
    bad = sa.engine.make_url(url).set(database="no_such_db_rs", port=1)
    res = rs.ping(bad.render_as_string(hide_password=False))
    assert not res["ok"] and (not pw or pw not in str(res))
