"""DuckDB over CSV and Parquet folders: the statistics, first rows and samples of a
pushed-down fact computed in SQL over the files equal the in-memory reference
(Propositions 100 and 101), with the CSV typing of the pandas reader."""
from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import pytest
from conftest import scenario, write_scenario

import relsynth as rs
from relsynth.connectors import from_spec
from relsynth.connectors.files import FileConnector
from relsynth.pushdown.stats import FrameStats, Require

pytest.importorskip("duckdb")
pytest.importorskip("duckdb_engine")


def _frame(n=3000, seed=5):
    g = np.random.default_rng(seed)
    return pd.DataFrame({"id": np.arange(1, n + 1), "a": g.integers(1, 30, n),
                         "b": np.where(g.random(n) < 0.1, np.nan, g.integers(1, 9, n)),
                         "x": np.where(g.random(n) < 0.05, np.nan, g.gamma(2.0, 3.0, n).round(2)),
                         "y": g.normal(size=n).round(1), "k": g.integers(0, 60, n),
                         "code": [f"{v:04d}" for v in g.integers(0, 50, n)],           # leading zeros: text
                         "day": pd.Series(pd.date_range("2024-01-01", periods=n, freq="h")).dt.strftime("%Y-%m-%d")})


@pytest.fixture(params=["csv", "parquet"])
def folder(request, tmp_path):
    df = _frame()
    FileConnector(tmp_path, fmt=request.param, create=True).write_table("t", df)
    return request.param, tmp_path


def _sorted(df, cols):
    return df.sort_values(cols, na_position="first", kind="stable").reset_index(drop=True)


def test_statistics_equal_the_in_memory_reference(folder):
    fmt, path = folder
    duck = FileConnector(path, fmt=fmt, engine="duckdb")
    mem = FileConnector(path, fmt=fmt, engine="pandas")
    sq, fs = duck.stats("t"), FrameStats(mem.read_table("t"))
    assert type(sq).__name__ == "DuckDBStats" and sq.n_rows == fs.n_rows == 3000
    for c in ("b", "x", "code"):
        assert sq.notnull(c) == fs.notnull(c)
        assert sorted(sq.distinct(c).tolist()) == sorted(fs.distinct(c).tolist())
    for cols, scores in ((["a", "b"], ["x", "y"]), (["code"], ["day"]), (["k"], [])):
        a, b = _sorted(sq.group_counts(cols, scores), cols), _sorted(fs.group_counts(cols, scores), cols)
        assert a[cols].astype(str).equals(b[cols].astype(str)) and (a["n"] == b["n"]).all()
        for c in a.columns[len(cols) + 1:]:
            assert np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)).max() < 1e-9, c
    req = {"b": Require(values=list(range(1, 9)), exclude=[3])}
    assert (_sorted(sq.group_counts(["a"], require=req), ["a"])["n"].tolist()
            == _sorted(fs.group_counts(["a"], require=req), ["a"])["n"].tolist())
    assert sq.minmax("x") == fs.minmax("x")


def test_first_rows_follow_file_order(folder):
    """The first row of each key is the first in the file (in-memory semantics),
    also with an ordering column that has nulls (nulls last)."""
    fmt, path = folder
    sq = FileConnector(path, fmt=fmt, engine="duckdb").stats("t")
    fs = FileConnector(path, fmt=fmt, engine="pandas").stats("t")
    for order in (None, ["x"], ["b", "y"]):
        h1, d1 = sq.first_rows("k", ["a", "x"], order=order)
        h2, d2 = fs.first_rows("k", ["a", "x"], order=order)
        h1, h2 = _sorted(h1, ["k"]), _sorted(h2, ["k"])
        assert d1 == d2
        pd.testing.assert_frame_equal(h1, h2, check_dtype=False)


def test_csv_typing_matches_the_pandas_reader(tmp_path):
    txt = ("id,code,num,mixed,na,flt,neg\n"
           "1,0101,10,a,NA,1.5,-3\n"
           "2,0202,,2,,2,4\n"
           "3,,30,3,None,1e3,+5\n")
    (tmp_path / "t.csv").write_text(txt)
    duck = FileConnector(tmp_path, engine="duckdb").duck
    duck.describe("t")                                  # views are created on first use
    with duck.engine.connect() as con:
        got = pd.read_sql("SELECT * FROM t ORDER BY id", con)
    mem = FileConnector(tmp_path, engine="pandas").read_table("t")
    assert got["code"].fillna("<null>").tolist() == ["0101", "0202", "<null>"] == mem["code"].fillna("<null>").tolist()
    assert got["na"].tolist()[0] == "NA" and got["na"].tolist()[2] == "None"     # only '' is null by default
    for c in ("num", "flt", "neg"):
        assert np.allclose(got[c].to_numpy(float), mem[c].to_numpy(float), equal_nan=True), c
    assert got["mixed"].tolist() == ["a", "2", "3"]
    nv = FileConnector(tmp_path, engine="duckdb", null_values=["", "NA", "None"]).duck
    nv.describe("t")
    with nv.engine.connect() as con:
        assert pd.read_sql("SELECT na FROM t", con)["na"].isna().all()


def test_samples_are_seeded_sized_and_survive_close(folder):
    fmt, path = folder
    c = FileConnector(path, fmt=fmt, engine="duckdb")
    s1, whole = c.sample("t", 600, seed=4)
    assert not whole and abs(len(s1) - 600) < 6 * np.sqrt(3000 * 0.2 * 0.8)
    assert "__rs_row" not in s1.columns and set(s1.columns) == set(_frame().columns)
    st = c.stats("t")
    c.close()                                           # an in-memory database is rebuilt on reconnect
    s2, _ = c.sample("t", 600, seed=4)
    assert sorted(s1["id"]) == sorted(s2["id"])
    assert st.group_counts(["a"])["n"].sum() == 3000
    s3, _ = c.sample("t", 600, seed=5)
    assert sorted(s1["id"]) != sorted(s3["id"])
    full, whole = c.sample("t", 5000)
    assert whole and len(full) == 3000


def test_full_sample_pushdown_over_files_equals_in_memory(star_csv, star_parquet):
    """Proposition 101 over files: DuckDB pushdown with a sample of every row gives
    the in-memory synthetic database."""
    for spec, _ in (star_csv, star_parquet):
        s_on = copy.deepcopy(spec)
        s_on["source"] = {**spec["source"], "engine": "duckdb"}
        s_on["synthesis"] = {"pushdown": {"mode": "on", "sample_rows": 10**7, "min_rows": 0}, "seed": 6}
        s_off = copy.deepcopy(spec)
        s_off["synthesis"] = {"pushdown": {"mode": "off"}, "seed": 6}
        on, off = rs.fit(s_on), rs.fit(s_off)
        assert on.pushdown["pushed"] == {"sales": len(scenario("S2_star")[0]["sales"])}
        a, b = on.sample(dataset=0), off.sample(dataset=0)
        for t in ("customer", "product", "sales"):
            pd.testing.assert_frame_equal(a[t].reset_index(drop=True), b[t].reset_index(drop=True),
                                          check_dtype=False, obj=t)


def test_engine_setting(tmp_path, star_csv):
    spec, path = star_csv
    assert from_spec({"kind": "csv", "path": str(path), "engine": "pandas"}).duck is None
    c = from_spec({"kind": "csv", "path": str(path), "duckdb": {"threads": 2, "memory_limit": "512MB"}})
    assert c.duck is not None and type(c.stats("sales")).__name__ == "DuckDBStats"
    with c.duck.engine.connect() as con:
        assert con.exec_driver_sql("SELECT current_setting('threads')").scalar() == 2
    with pytest.raises(ValueError):
        FileConnector(path, engine="spark")
    bad = copy.deepcopy(spec)
    bad["source"] = {**spec["source"], "engine": "spark", "duckdb": {"cores": 3}}
    errs = rs.validate_spec(bad)
    assert any("engine" in e for e in errs) and any("duckdb" in e for e in errs)


def test_hostile_file_and_folder_names(tmp_path):
    """Folder paths and table names with quotes are literals and identifiers, never SQL."""
    dq = "" if os.name == "nt" else '"'                 # Windows file names cannot hold a double quote
    d = tmp_path / f"it's {dq}here{dq}; DROP"
    c = FileConnector(d, create=True)
    name = f"o'k{dq}; SELECT 1; --"
    c.write_table(name, pd.DataFrame({"id": [1, 2, 3], "v'\"": [4, 5, 6]}))
    duck = FileConnector(d, engine="duckdb")
    st = duck.stats(name)
    assert st.n_rows == 3 and st.group_counts(["v'\""])["n"].sum() == 3
    assert duck.sample(name, 1, seed=1)[0].shape[1] == 2


def test_write_scenario_to_parquet_and_fit_with_duckdb(tmp_path):
    spec = write_scenario("S10_order_dw", {"kind": "parquet", "path": str(tmp_path / "src")})
    spec["source"]["engine"] = "duckdb"
    off = copy.deepcopy(spec)
    off["synthesis"] = {**spec.get("synthesis", {}), "pushdown": {"mode": "off"}}
    on = copy.deepcopy(spec)
    on["synthesis"] = {**spec.get("synthesis", {}), "pushdown": {"mode": "on", "min_rows": 0, "sample_rows": 10**7}}
    a, b = rs.fit(off, seed=11), rs.fit(on, seed=11)
    assert b.pushdown["pushed"]
    x, y = a.sample(), b.sample()
    for t in x:
        pd.testing.assert_frame_equal(x[t].reset_index(drop=True), y[t].reset_index(drop=True), check_dtype=False,
                                      obj=t)
