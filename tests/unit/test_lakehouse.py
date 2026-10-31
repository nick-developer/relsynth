# SPDX-License-Identifier: Apache-2.0
"""Iceberg and Delta Lake tables on a local disk: the connector contract
(docs/connector-sdk.md), versions (snapshot, as_of, history), keys kept in the
table metadata, the write modes, DuckDB statistics over the version read
(Props. 100, 101) and the decisive check: a source copied into a lakehouse gives
the same synthetic database as the in-memory source."""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import pytest
from conftest import scenario, write_scenario

import relsynth as rs
from relsynth.connectors import connect, from_spec
from relsynth.connectors.lakehouse import arrow_table, parse_partition, storage_options
from relsynth.pushdown.stats import FrameStats

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb_engine")

KINDS = []
try:
    import pyiceberg  # noqa: F401
    KINDS.append("iceberg")
except ImportError:          # pragma: no cover - the iceberg extra is not installed
    pass
try:
    import deltalake  # noqa: F401
    KINDS.append("delta")
except ImportError:          # pragma: no cover
    pass
if not KINDS:                # pragma: no cover
    pytest.skip("neither pyiceberg nor deltalake installed", allow_module_level=True)


def _frame(n=3000, seed=5):
    g = np.random.default_rng(seed)
    return pd.DataFrame({"id": np.arange(1, n + 1), "a": g.integers(1, 30, n),
                         "b": np.where(g.random(n) < 0.1, np.nan, g.integers(1, 9, n)),
                         "x": np.where(g.random(n) < 0.05, np.nan, g.gamma(2.0, 3.0, n).round(2)),
                         "k": g.integers(0, 60, n),
                         "code": [f"{v:04d}" for v in g.integers(0, 50, n)],
                         "ts": pd.Timestamp("2024-01-01") + pd.to_timedelta(g.integers(0, 10**6, n), unit="s"),
                         "day": pd.Series(pd.date_range("2024-01-01", periods=n, freq="h")).dt.strftime("%Y-%m-%d")})


def _section(kind, path, **kw):
    return {"kind": kind, "path": str(path), "schema": "test", **kw}


SPEC = {"tables": {"t": {"primary_key": "id", "unique": [["code", "day"]], "columns": {}},
                   "p": {"primary_key": "k", "columns": {}}},
        "relationships": [{"child": "t", "parent": "p", "kind": "many_to_one", "child_key": "k", "parent_key": "k"}]}


@pytest.fixture(params=KINDS)
def lake(request, tmp_path):
    sec = _section(request.param, tmp_path / "lake")
    c = from_spec(sec)
    c.write_tables(SPEC, {"t": _frame(), "p": pd.DataFrame({"k": np.arange(60), "name": [f"p{i}" for i in range(60)]})})
    return sec


def _sorted(df, cols):
    return df.sort_values(cols, na_position="first", kind="stable").reset_index(drop=True)


def test_write_read_and_catalog(lake):
    c = from_spec(lake)
    assert sorted(c.tables()) == ["p", "t"] and c.has_table("t") and not c.has_table("nope")
    assert c.count("t") == 3000 and c.count("p") == 60
    info = c.describe("t")
    assert info.primary_key == ["id"] and info.unique == [["code", "day"]]
    assert [(f.columns, f.ref_table, f.ref_columns) for f in info.foreign_keys] == [(["k"], "p", ["k"])]
    got = c.read_table("t")
    want = _frame()
    pd.testing.assert_frame_equal(got, want, check_dtype=False)
    assert got["code"].iloc[0] == want["code"].iloc[0]           # leading zeros kept (text)
    assert c.read_table("t", ["x", "a"]).columns.tolist() == ["x", "a"]
    assert sum(len(ch) for ch in c.iter_table("t", chunksize=700)) == 3000
    assert c.ping()["ok"] and c.ping()["tables"] == 2
    assert set(c.versions()) == {"t", "p"}                       # what was read, per table


def test_statistics_equal_the_in_memory_reference(lake):
    c = from_spec(lake)
    sq = c.stats("t")
    fs = FrameStats(from_spec({**lake, "engine": "pandas"}).read_table("t"))
    assert type(sq).__name__ == "DuckDBStats" and sq.n_rows == fs.n_rows == 3000
    for col in ("b", "x", "code"):
        assert sq.notnull(col) == fs.notnull(col)
        assert sorted(sq.distinct(col).tolist()) == sorted(fs.distinct(col).tolist())
    for cols, scores in ((["a", "b"], ["x"]), (["code"], ["day"]), (["k"], [])):
        a, b = _sorted(sq.group_counts(cols, scores), cols), _sorted(fs.group_counts(cols, scores), cols)
        assert a[cols].astype(str).equals(b[cols].astype(str)) and (a["n"] == b["n"]).all()
        for col in a.columns[len(cols) + 1:]:
            assert np.abs(a[col].to_numpy(float) - b[col].to_numpy(float)).max() < 1e-9
    for order in (None, ["x"]):
        h1, d1 = sq.first_rows("k", ["a", "x"], order=order)
        h2, d2 = fs.first_rows("k", ["a", "x"], order=order)
        assert d1 == d2
        pd.testing.assert_frame_equal(_sorted(h1, ["k"]), _sorted(h2, ["k"]), check_dtype=False)


def test_samples_are_seeded_and_survive_close(lake):
    c = from_spec(lake)
    s1, whole = c.sample("t", 600, seed=4)
    assert not whole and abs(len(s1) - 600) < 6 * np.sqrt(3000 * 0.2 * 0.8)
    c.close()
    s2, _ = c.sample("t", 600, seed=4)
    assert sorted(s1["id"]) == sorted(s2["id"])
    assert sorted(c.sample("t", 600, seed=5)[0]["id"]) != sorted(s1["id"])
    full, whole = c.sample("t", 5000)
    assert whole and len(full) == 3000
    p, whole = from_spec({**lake, "engine": "pandas"}).sample("t", 600, seed=4)
    assert not whole and 400 < len(p) < 800


def test_versions_pin_and_time_travel(lake):
    c = from_spec(lake)
    v0 = c.versions().get("t") if c.versions() else None
    c.count("t")
    v0 = c.versions()["t"]
    time.sleep(0.05)
    t_mid = pd.Timestamp.now(tz="UTC")
    time.sleep(0.05)
    extra = _frame(10, seed=9).assign(id=np.arange(5001, 5011))
    c.write_tables(SPEC, {"t": extra}, mode="append")
    assert from_spec(lake).count("t") == 3010
    assert from_spec({**lake, "snapshot": v0}).count("t") == 3000
    assert from_spec({**lake, "snapshot": {"t": v0}}).count("t") == 3000
    assert from_spec({**lake, "as_of": t_mid.isoformat()}).count("t") == 3000
    h = c.history("t")
    assert len(h) == 2 and h[0]["relsynth"].get("relsynth.version")
    with pytest.raises(ValueError):
        from_spec({**lake, "snapshot": v0, "as_of": t_mid.isoformat()})


@pytest.mark.parametrize("kind", KINDS)
def test_write_modes(kind, tmp_path):
    sec = _section(kind, tmp_path / "m")
    c = from_spec(sec)
    df = pd.DataFrame({"id": [1, 2, 3], "v": ["a", "b", None]})
    spec = {"tables": {"x": {"primary_key": "id", "columns": {}}}, "relationships": []}
    c.write_tables(spec, {"x": df})
    c.write_tables(spec, {"x": df}, mode="append")
    assert from_spec(sec).count("x") == 6
    c.write_tables(spec, {"x": df.head(2)}, mode="truncate")
    assert from_spec(sec).count("x") == 2
    c.write_tables(spec, {"x": df.assign(w=1.5)}, mode="replace")                 # a new column: recreated
    got = from_spec(sec).read_table("x")
    assert got.columns.tolist() == ["id", "v", "w"] and len(got) == 3
    with pytest.raises(FileExistsError):
        c.write_tables(spec, {"x": df}, mode="fail")
    with pytest.raises(ValueError):
        c.write_tables(spec, {"x": df}, mode="upsert")


def test_partitioned_targets(tmp_path):
    df = _frame(500)
    if "iceberg" in KINDS:
        sec = _section("iceberg", tmp_path / "i", partition_by={"t": ["month(ts)", "bucket(4, k)", "a"]})
        from_spec(sec).write_tables({"tables": {"t": {}}, "relationships": []}, {"t": df})
        c = from_spec(sec)
        tbl = c._table("t")
        assert [str(f.transform) for f in tbl.spec().fields] == ["month", "bucket[4]", "identity"]
        pd.testing.assert_frame_equal(_sorted(c.read_table("t"), ["id"]), df, check_dtype=False)
        assert c.stats("t").n_rows == 500
    if "delta" in KINDS:
        sec = _section("delta", tmp_path / "d", partition_by={"t": ["a"]})
        from_spec(sec).write_tables({"tables": {"t": {}}, "relationships": []}, {"t": df})
        c = from_spec(sec)
        assert c._dt("t").metadata().partition_columns == ["a"]
        got = _sorted(c.read_table("t"), ["id"])
        pd.testing.assert_frame_equal(got[df.columns], df, check_dtype=False)
        with pytest.raises(ValueError, match="column values only"):
            from_spec(_section("delta", tmp_path / "d2", partition_by={"t": ["month(ts)"]})).write_tables(
                {"tables": {"t": {}}, "relationships": []}, {"t": df})


def test_helpers_and_validation(tmp_path, monkeypatch):
    assert parse_partition("Region") == ("identity", "Region", None)
    assert parse_partition("month(OrderDate)") == ("month", "OrderDate", None)
    assert parse_partition("bucket(16, CustomerKey)") == ("bucket", "CustomerKey", 16)
    for bad in ("median(x)", "bucket(x)", "month(a, b)"):
        with pytest.raises(ValueError):
            parse_partition(bad)
    monkeypatch.setenv("RS_TEST_KEY", "s3cr3t")
    assert storage_options({"AWS_REGION": "eu-west-1"}, {"AWS_SECRET_ACCESS_KEY": "RS_TEST_KEY"}) == \
        {"AWS_REGION": "eu-west-1", "AWS_SECRET_ACCESS_KEY": "s3cr3t"}
    with pytest.raises(ValueError, match="not set"):
        storage_options({}, {"X": "RS_TEST_MISSING"})
    t = arrow_table(pd.DataFrame({"a": [1, None], "b": [None, None], "c": [1, "x"], "__h": [1, 2],
                                  "d": pd.to_datetime(["2024-01-01", None])}))
    assert t.schema.names == ["a", "b", "c", "d"] and str(t.schema.field("d").type) == "timestamp[us]"
    assert t.column("c").to_pylist() == ["1", "x"]
    with pytest.raises(ValueError, match="catalog"):
        from_spec({"kind": "iceberg", "path": "s3://bucket/x"})
    good = copy.deepcopy(scenario("S2_star")[1])
    good["source"] = {"kind": "delta", "path": "x", "storage": {"AWS_REGION": "eu-west-1"}, "as_of": "2024-01-01"}
    assert not [e for e in rs.validate_spec(good) if "source" in e]
    bad = copy.deepcopy(good)
    bad["source"] = {"kind": "csv", "path": "x", "catalog": {"type": "rest"}, "storage": "s3"}
    bad["target"] = {"kind": "delta", "snapshot": 3}
    errs = rs.validate_spec(bad)
    assert any("only for the lakehouse kinds" in e for e in errs)
    assert any("source.storage must be a mapping" in e for e in errs)
    assert any("target: kind delta needs `path`" in e for e in errs)
    assert any("target.snapshot" in e for e in errs)


def test_hostile_table_names(tmp_path):
    for kind in KINDS:
        c = from_spec(_section(kind, tmp_path / kind))
        name = "o'k; SELECT 1; --"
        c.write_tables({"tables": {name: {}}, "relationships": []}, {name: pd.DataFrame({"id": [1, 2], "v'\"": [3, 4]})})
        st = from_spec(_section(kind, tmp_path / kind)).stats(name)
        assert st.n_rows == 2 and st.group_counts(["v'\""])["n"].sum() == 2
        if kind == "delta":
            with pytest.raises(ValueError):
                c.write_tables({"tables": {"../x": {}}, "relationships": []}, {"../x": pd.DataFrame({"id": [1]})})


@pytest.mark.parametrize("kind", KINDS)
def test_scenario_in_a_lakehouse_gives_the_same_synthetic_database(kind, tmp_path):
    """The decisive check (docs/connector-sdk.md): the same data gives the same
    synthetic database, in memory and with full-sample pushdown over DuckDB."""
    lite = write_scenario("S10_order_dw", {"kind": "sqlite", "path": str(tmp_path / "ref.db")})
    lake = write_scenario("S10_order_dw", _section(kind, tmp_path / "src"))
    lite["synthesis"] = {**lite.get("synthesis", {}), "pushdown": {"mode": "off"}}
    ref = rs.fit(lite, seed=11).sample(dataset=0)
    for pushdown in ({"mode": "off"}, {"mode": "on", "min_rows": 0, "sample_rows": 10**7}):
        spec = copy.deepcopy(lake)
        spec["synthesis"] = {**spec.get("synthesis", {}), "pushdown": pushdown}
        m = rs.fit(spec, seed=11)
        if pushdown["mode"] == "on":
            assert m.pushdown["pushed"]
        got = m.sample(dataset=0)
        for name in ref:
            pd.testing.assert_frame_equal(ref[name].reset_index(drop=True), got[name].reset_index(drop=True),
                                          check_dtype=False, obj=f"{kind} {pushdown['mode']} {name}")


@pytest.mark.parametrize("kind", KINDS)
def test_synthetic_lake_introspects_with_its_keys(kind, tmp_path):
    lake = write_scenario("S2_star", _section(kind, tmp_path / "src"))
    spec = rs.introspect(lake["source"]).raw
    assert set(spec["tables"]) == {"customer", "product", "sales"}
    assert any(r["child"] == "sales" for r in spec["relationships"])


def test_connect_shorthands(tmp_path):
    for kind in KINDS:
        c = connect(f"{kind}:{tmp_path / kind}")
        assert c.kind == kind and c.tables() == []
    from relsynth.pipeline import parse_source
    assert parse_source("delta:s3://b/x") == {"kind": "delta", "path": "s3://b/x"}
