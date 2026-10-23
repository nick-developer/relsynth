"""Pushdown: the planner's decisions, and parity with in-memory fitting.

Proposition P1 (sufficiency) says the in-database statistics are the ones the
in-memory fit computes; with a sample that holds every row, a pushed-down fit
must therefore produce the very same synthetic database, bit for bit, for every
shape a pushed fact can have (plain, nested, header-split, view base, running
balance)."""
from __future__ import annotations

import copy

import pandas as pd
import pytest
from conftest import scenario, write_scenario

import relsynth as rs
from relsynth.pushdown.planner import ineligible_reason, plan
from relsynth.spec import load_spec


def _with_pushdown(spec: dict, mode: str, sample_rows: int, min_rows: int = 0) -> dict:
    s = copy.deepcopy(spec)
    s.setdefault("synthesis", {})["pushdown"] = {"mode": mode, "sample_rows": sample_rows, "min_rows": min_rows}
    return s


# ---------------------------------------------------------------- planner
def test_ineligible_reasons():
    _, raw, _ = scenario("S9_small_shop")
    s = load_spec(raw)
    facts = [t for t, ts in s["tables"].items() if ts["role"] == "fact"]
    reasons = {t: ineligible_reason(s, t) for t in s["tables"]}
    assert all("not a fact" in reasons[t] for t in s["tables"] if t not in facts)
    referenced = {r["parent"] for r in s["relationships"]}
    for f in facts:
        if f in referenced:
            assert reasons[f] == "other tables reference it"
    s2 = copy.deepcopy(s)
    leaf = next(f for f in facts if f not in referenced)
    s2["tables"][leaf]["scd"] = {"key": "x"}
    assert ineligible_reason(s2, leaf) == "type 2 history"
    s2["tables"][leaf]["mode"] = "copy"
    assert ineligible_reason(s2, leaf) == "copied table"


def test_aggregate_source_not_stored_blocks_pushdown():
    s = {"tables": {"h": {"role": "entity"}, "l": {"role": "fact"}}, "relationships": [],
         "rules": [{"type": "aggregate", "table": "h", "column": "total", "from": "l"}]}
    assert ineligible_reason(s, "l", stored={"h": {"id"}}).startswith("the source does not store h.total")
    assert ineligible_reason(s, "l", stored={"h": {"id", "total"}}) is None


def test_plan_modes(star_sqlite):
    spec, _ = star_sqlite
    conn = rs.connect(spec["source"]["path"])
    try:
        s = load_spec(_with_pushdown(spec, "auto", 1000, min_rows=10**9))
        assert plan(s, conn).pushed == {}                              # below min_rows
        s = load_spec(_with_pushdown(spec, "auto", 1000, min_rows=10))
        p = plan(s, conn)
        assert set(p.pushed) == {"sales"} and set(p.loaded) == {"customer", "product"}
        assert p.pushed["sales"] == conn.count("sales")
        s = load_spec(_with_pushdown(spec, "off", 1000))
        assert plan(s, conn).pushed == {}
        assert plan(load_spec(_with_pushdown(spec, "on", 1000, min_rows=10**9)), conn).pushed  # on ignores min_rows
        assert p.to_dict()["mode"] == "auto"
    finally:
        conn.close()


# ---------------------------------------------------------------- parity
def _tables_equal(a, b):
    assert set(a) == set(b)
    for t in a:
        x, y = a[t].reset_index(drop=True), b[t].reset_index(drop=True)
        pd.testing.assert_frame_equal(x, y, check_dtype=False, obj=t)


SHAPES = [("S2_star", "plain fact"),
          pytest.param("S8_reporting", "nested fact", marks=pytest.mark.slow),
          ("S10_order_dw", "header-split fact with running balance"),
          pytest.param("S11_retail_dw", "view base", marks=pytest.mark.slow)]


@pytest.mark.parametrize("name, shape", SHAPES)
def test_full_sample_pushdown_equals_in_memory(tmp_path, name, shape):
    spec = write_scenario(name, {"kind": "sqlite", "path": str(tmp_path / "src.db")})
    off = rs.fit(_with_pushdown(spec, "off", 10**7), seed=11)
    on = rs.fit(_with_pushdown(spec, "on", 10**7), seed=11)
    assert on.pushdown["pushed"], f"{shape}: nothing was pushed down ({on.pushdown['reasons']})"
    _tables_equal(off.sample(), on.sample())


def test_pushdown_from_files_and_from_sqlite_agree(star_sqlite, star_parquet):
    a = rs.fit(_with_pushdown(star_sqlite[0], "on", 10**7), seed=2).sample()
    b = rs.fit(_with_pushdown(star_parquet[0], "on", 10**7), seed=2).sample()
    _tables_equal(a, b)


def test_sampled_pushdown_keeps_quality_and_memory_bound(star_sqlite):
    spec, _ = star_sqlite
    m = rs.fit(_with_pushdown(spec, "on", 3000), seed=4)
    n_src = m.pushdown["pushed"]["sales"]
    assert len(m.fit_data.tables["sales"]) < 0.4 * n_src                # a sample, not the table
    d = m.sample()
    assert abs(len(d["sales"]) / n_src - 1) < 0.1                       # sized from the exact count
    rep = m.evaluate(d)
    assert rep.score >= 0.95, rep.failed
    # the real side of the scorecard uses the exact row count, not the sample's
    t = rep.raw["tables"]["sales"]
    assert t["rows_real"] == n_src and t["rows_real_sample"] == len(m.fit_data.tables["sales"])


def test_sampled_pushdown_is_reproducible(star_sqlite):
    spec, _ = star_sqlite
    a = rs.fit(_with_pushdown(spec, "on", 3000), seed=9).sample()
    b = rs.fit(_with_pushdown(spec, "on", 3000), seed=9).sample()
    _tables_equal(a, b)


def test_header_split_without_unique_key_agrees(tmp_path):
    """Order headers come out in key order on every path, so a fact without a
    declared unique key (rows in hash order) still fits the same pushed or loaded."""
    spec = write_scenario("S10_order_dw", {"kind": "sqlite", "path": str(tmp_path / "src.db")})
    spec["tables"]["fact_sales"]["unique"] = []
    off = rs.fit(_with_pushdown(spec, "off", 10**7), seed=11)
    on = rs.fit(_with_pushdown(spec, "on", 10**7), seed=11)
    assert "fact_sales" in on.pushdown["pushed"]
    _tables_equal(off.sample(), on.sample())
