# SPDX-License-Identifier: Apache-2.0
"""In-database statistics: the SQL functions (normal quantile, seeded hash) and
SqlStats against the in-memory reference FrameStats (Proposition P1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtri

from relsynth.connectors import connect
from relsynth.connectors.sql import SqlStats
from relsynth.connectors.sqlfuncs import HASH_RANGE, acklam_py, py_row_hash
from relsynth.pushdown.stats import FrameStats, Require, normal_scores


# ---------------------------------------------------------------- functions
def test_acklam_error_bound():
    """Acklam's rational approximation: relative error below 1.15e-9 on (0, 1)
    (Proposition P3) - what server engines use for Phi^-1."""
    p = np.concatenate([np.linspace(1e-12, 1e-6, 2000), np.linspace(1e-6, 1 - 1e-6, 200001),
                        1 - np.linspace(1e-12, 1e-6, 2000)])
    x = np.array([acklam_py(v) for v in p[::50]])
    ref = ndtri(p[::50])
    rel = np.abs(x - ref) / np.maximum(np.abs(ref), 1e-300)
    assert np.nanmax(rel[np.abs(ref) > 1e-3]) < 1.15e-9
    assert np.max(np.abs(x - ref)) < 1e-8


def test_python_hash_is_uniform_and_seeded():
    v = np.array([py_row_hash(7, str(i)) for i in range(20000)])
    assert v.min() >= 0 and v.max() < HASH_RANGE
    # uniform on [0, 2^32): KS against the uniform law
    from scipy.stats import kstest
    assert kstest(v / HASH_RANGE, "uniform").pvalue > 1e-4
    assert py_row_hash(7, "a") != py_row_hash(8, "a")
    assert py_row_hash(7, "a") == py_row_hash(7, "a")


def test_sqlite_hash_matches_python(tmp_path):
    c = connect(f"sqlite:{tmp_path / 'h.db'}")
    with c.engine.begin() as con:
        con.execute(sa.text("CREATE TABLE t (k INTEGER PRIMARY KEY)"))
        con.execute(sa.text("INSERT INTO t VALUES (1), (2), (300)"))
    from relsynth.connectors.sqlfuncs import row_hash
    t = c.table("t")
    with c.engine.connect() as con:
        got = con.execute(sa.select(t.c.k, row_hash([t.c.k], 11, "sqlite")).order_by(t.c.k)).fetchall()
    assert [h for _, h in got] == [py_row_hash(11, str(k)) for k, _ in got]


@settings(max_examples=60, deadline=None)
@given(st.lists(st.one_of(st.none(), st.integers(-50, 50)), min_size=1, max_size=60))
def test_normal_scores_are_rank_based(vals):
    s = pd.Series(vals, dtype="Float64")
    z = normal_scores(s)
    assert np.all(z[s.isna().to_numpy()] == 0)
    # any increasing transform gives the same scores
    z2 = normal_scores(s * 3 + 1000)
    assert np.allclose(z, z2)
    # without ties the scores are symmetric about 0 (ranks 1..N map to +-pairs)
    u = pd.Series(pd.unique(s.dropna()), dtype="Float64")
    assert abs(normal_scores(u).sum()) < 1e-9


# ---------------------------------------------------------------- parity
def _frame(n=3000, seed=1):
    g = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "a": g.integers(1, 40, n).astype(float),
        "b": np.where(g.random(n) < 0.1, np.nan, g.integers(1, 12, n)),
        "x": np.where(g.random(n) < 0.05, np.nan, g.gamma(2.0, 3.0, n).round(2)),
        "y": g.normal(size=n).round(1),           # ties
        "d": pd.to_datetime("2020-01-01") + pd.to_timedelta(g.integers(0, 900, n), unit="D"),
        "k": g.integers(0, 50, n),
    })


@pytest.fixture(scope="module")
def sqlite_stats(tmp_path_factory):
    df = _frame()
    db = tmp_path_factory.mktemp("st") / "s.db"
    eng = sa.create_engine(f"sqlite:///{db}")
    with eng.begin() as con:     # a primary key: "first row" is first in key order, as in the frame
        con.execute(sa.text("CREATE TABLE t (id INTEGER PRIMARY KEY, a REAL, b REAL, x REAL, y REAL, d TIMESTAMP, k INTEGER)"))
    df.to_sql("t", eng, index=False, if_exists="append")
    pd.DataFrame({"pk": np.arange(1, 30)}).to_sql("par", eng, index=False)
    eng.dispose()
    c = connect(f"sqlite:{db}")
    return df, c.stats("t"), FrameStats(df)


def _same(a, b, cols, tol=1e-6):
    a = a.sort_values(cols, na_position="first").reset_index(drop=True)
    b = b.sort_values(cols, na_position="first").reset_index(drop=True)
    assert len(a) == len(b)
    assert (a["n"].to_numpy() == b["n"].to_numpy()).all()
    for c in a.columns[len(cols) + 1:]:
        assert np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)).max() < tol, c


def test_counts_and_scores_match(sqlite_stats):
    df, sq, fs = sqlite_stats
    assert sq.n_rows == fs.n_rows == len(df)
    for c in ("a", "b", "x"):
        assert sq.notnull(c) == fs.notnull(c)
        assert sorted(sq.distinct(c).astype(float)) == sorted(fs.distinct(c).astype(float))
    _same(sq.group_counts(["a", "b"], ["x", "y", "d"]), fs.group_counts(["a", "b"], ["x", "y", "d"]), ["a", "b"],
          tol=1e-9)


def test_require_semi_join_values_and_late_filter(sqlite_stats):
    df, sq, fs = sqlite_stats
    ref = fs.group_counts(["a"], ["x"], require={"b": Require(values=list(range(1, 30)), exclude=[3])})
    _same(sq.group_counts(["a"], ["x"], require={"b": Require(table="par", column="pk", exclude=[3])}), ref, ["a"])
    _same(sq.group_counts(["a"], ["x"], require={"b": Require(values=list(range(1, 30)), exclude=[3])}), ref, ["a"])
    old = SqlStats.MAX_IN_LIST
    SqlStats.MAX_IN_LIST = 5               # long key lists are applied after the query
    try:
        _same(sq.group_counts(["a"], ["x"], require={"b": Require(values=list(range(1, 30)), exclude=[3])}), ref, ["a"])
        _same(sq.group_counts(["a", "b"], ["x"], require={"b": Require(values=list(range(1, 30)), exclude=[3])}),
              fs.group_counts(["a", "b"], ["x"], require={"b": Require(values=list(range(1, 30)), exclude=[3])}),
              ["a", "b"])
    finally:
        SqlStats.MAX_IN_LIST = old
    empty = sq.group_counts(["a"], require={"b": Require(values=[])})
    assert len(empty) == 0


def test_first_rows_and_minmax(sqlite_stats):
    df, sq, fs = sqlite_stats
    h1, d1 = sq.first_rows("k", ["a", "b"])
    h2, d2 = fs.first_rows("k", ["a", "b"])
    assert d1 == d2 and len(h1) == len(h2) == df["k"].nunique()
    m = h1.sort_values("k").reset_index(drop=True)
    first = df.sort_values("id").groupby("k", sort=True).first().reset_index()
    assert (m["a"].astype(float).to_numpy() == first["a"].to_numpy()).all()
    # ordered by a column: the row with the earliest date per k
    h3, _ = sq.first_rows(["k"], ["d"], order=["d"], differences=False)
    want = df.groupby("k")["d"].min().sort_index()
    got = pd.to_datetime(h3.sort_values("k")["d"]).reset_index(drop=True)
    assert (got.to_numpy() == want.to_numpy()).all()
    h4, _ = fs.first_rows(["k"], ["d"], order=["d"], differences=False)
    assert (pd.to_datetime(h4.sort_values("k")["d"]).to_numpy() == want.to_numpy()).all()
    lo, hi = sq.minmax("x")
    assert float(lo) == df["x"].min() and float(hi) == df["x"].max()
    assert fs.minmax("x") == (df["x"].min(), df["x"].max())


def test_sample_is_uniform_reproducible_and_sized(sqlite_stats, tmp_path):
    df, _, _ = sqlite_stats
    db = tmp_path / "big.db"
    eng = sa.create_engine(f"sqlite:///{db}")
    pd.DataFrame({"id": np.arange(20000), "v": np.arange(20000) % 7}).to_sql("t", eng, index=False)
    eng.dispose()
    c = connect(f"sqlite:{db}")
    s1, whole = c.sample("t", 5000, seed=3)
    s2, _ = c.sample("t", 5000, seed=3)
    s3, _ = c.sample("t", 5000, seed=4)
    assert not whole and s1.equals(s2) and not s1.equals(s3)
    assert abs(len(s1) - 5000) < 6 * np.sqrt(20000 * 0.25 * 0.75)
    # uniform: the share of each residue class is about 1/7
    share = s1["v"].value_counts(normalize=True)
    assert share.max() - share.min() < 0.04
    full, whole = c.sample("t", 50000)
    assert whole and len(full) == 20000


def test_lazy_frame_stats_loads_only_needed_columns():
    calls = []
    df = _frame(200)

    def loader(cols):
        calls.append(list(cols))
        return df[cols]
    fs = FrameStats.lazy(loader, n_rows=len(df))
    fs.notnull("a")
    fs.group_counts(["a", "b"])
    assert calls == [["a"], ["b"]]
