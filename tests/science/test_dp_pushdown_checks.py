# SPDX-License-Identifier: Apache-2.0
"""Checks C1760-C1764: DP stage 1 inside the source database (gap plan G2,
Props. 1760-1763 in docs/proofs, section v2.9; guide docs/dp.md section 9).

* C1760: the database's contribution bounding keeps at most D rows per parent
  row and is stable: removing one unit leaves every other unit's kept rows
  unchanged (Prop. 1760).
* C1761: with nothing truncated, every exact statistic stage 1 asks a mechanism
  to perturb is the same integer vector in the database as in memory, for
  copula, mst and aim, on a two-level chain, in SQLite and in Parquet files
  read through DuckDB (Props. 1761, 1762).
* C1762: the same holds when the database's frequency tables arrive in chunks
  of 7 tuples (Prop. 1763).
* C1763: sensitivity replay on the pushdown path: removing one customer from
  the database moves each released statistic by at most its charged L2
  sensitivity (the C902 check, on SQLite).
* C1764: a DP run with pushdown keeps the guarantee and the release shape: same
  plan, budget within the stated epsilon, valid links, stage 1 read in the
  database.

    pytest -m science tests/science/test_dp_pushdown_checks.py
"""
from __future__ import annotations

import copy
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.science]
opendp = pytest.importorskip("opendp")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fakedata"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import write_scenario  # noqa: E402

import relsynth.dp as dp  # noqa: E402
from relsynth.dp import mechanisms as M  # noqa: E402
from relsynth.dp import relational as R  # noqa: E402
from relsynth.dp import rows as RW  # noqa: E402
from relsynth.dp import select as SEL  # noqa: E402
from relsynth.pushdown import load_for_fit  # noqa: E402
from relsynth.spec import load_spec  # noqa: E402


class _Recorder:
    """Stands in for mechanisms.gaussian: records each exact statistic and its
    sensitivity; returns it unchanged (first run) or the first run's noisy
    output for the same mechanism (replay)."""

    def __init__(self, replay=None):
        self.calls, self.replay = [], replay

    def __call__(self, values, sensitivity, rho, acct, stage, mechanism):
        v = np.asarray(values, dtype=np.int64).ravel()
        acct.charge(M.Charge(stage, mechanism, rho))
        i = len(self.calls)
        self.calls.append((mechanism, v.copy(), float(sensitivity)))
        if self.replay is None:
            return v
        name, out, _ = self.replay.calls[i]
        assert name == mechanism and len(out) == len(v), (name, mechanism, len(out), len(v))
        return out


def _argmax(scores, sensitivity, rho, acct, stage, mechanism):
    acct.charge(M.Charge(stage, mechanism, rho))
    return int(np.argmax(scores))


def _db(tmp_path, name):
    return write_scenario(name, {"kind": "sqlite", "path": str(tmp_path / f"{name}.db")})


def _dp(spec, **kw):
    s = copy.deepcopy(spec)
    s.setdefault("synthesis", {}).setdefault("privacy", {})["dp"] = {
        "unit": "customer", "epsilon": 2.0, "delta": 1e-9, "public_tables": "all_unowned", "spec_is_public": True,
        **kw}
    return load_spec(s)


def _no_truncation(spec):
    """max_degree at every child's largest real degree: nothing is truncated."""
    cfg = dp.parse(spec)
    pl = R.make_plan(spec, cfg)
    real = load_for_fit(spec, mode="off").tables
    mx = {}
    for t in pl.order:
        for r in pl.tables[t].children:
            ck = R._keys(r["child_key"])
            mx[r["child"]] = int(real[r["child"]].groupby(ck).size().max())
    return mx


def _stats(spec, mode, monkeypatch, replay=None, seed=11):
    rec = _Recorder(replay)
    monkeypatch.setattr(R, "gaussian", rec)
    monkeypatch.setattr(SEL, "noisy_max", _argmax)
    monkeypatch.setattr(R, "_hash_key", lambda: "0123456789abcdef")
    cfg = dp.parse(spec)
    pl = R.make_plan(spec, cfg)
    if mode == "on":
        from relsynth.data import source_connector
        conn = source_connector(spec)
        src = RW.SqlSource(spec, conn, owner=conn, seed=seed)
    else:
        src, _ = RW.source_for(spec, "off")
    try:
        R.fit(spec, src, cfg, M.Accountant(cfg.rho, cfg.delta), pl)
    finally:
        src.close()
    return rec


def _assert_same(a, b):
    assert [m for m, _, _ in a.calls] == [m for m, _, _ in b.calls]
    for (m, va, sa), (_, vb, sb) in zip(a.calls, b.calls):
        assert sa == sb, m
        np.testing.assert_array_equal(va, vb, err_msg=m)


# --------------------------------------------------------------------- C1760
def test_c1760_database_truncation_is_bounded_and_stable(tmp_path):
    """Prop. 1760: at most D kept sales per customer, and removing the busiest
    customer from the database leaves every other customer's kept sales the same."""
    spec = _dp(_db(tmp_path, "S2_star"))
    db = spec["source"]["path"]
    D = 3

    def kept(path):
        s = copy.deepcopy(spec)
        s["source"] = {"kind": "sqlite", "path": path}
        from relsynth.data import source_connector
        conn = source_connector(s)
        src = RW.SqlSource(s, conn, owner=conn, seed=12345)
        try:
            rows = src.unit("customer").bound("sales", ["customer_id"], ["customer_id"], D)
            parts = [c for c, _ in rows.groups(["sale_id", "customer_id"])]
            return pd.concat(parts).sort_values("sale_id").reset_index(drop=True)
        finally:
            src.close()
    k1 = kept(db)
    assert k1.groupby("customer_id").size().max() <= D
    with sqlite3.connect(db) as con:
        busiest = con.execute("SELECT customer_id FROM sales GROUP BY customer_id ORDER BY COUNT(*) DESC LIMIT 1"
                              ).fetchone()[0]
    db2 = str(tmp_path / "without.db")
    with sqlite3.connect(db) as src_con, sqlite3.connect(db2) as dst:
        src_con.backup(dst)
        dst.execute("DELETE FROM sales WHERE customer_id = ?", (busiest,))
        dst.execute("DELETE FROM customer WHERE customer_id = ?", (busiest,))
    k2 = kept(db2)
    a = k1[k1["customer_id"].astype(str) != str(busiest)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, k2, check_dtype=False)


# --------------------------------------------------------------------- C1761
@pytest.mark.parametrize("measure", ["copula", "mst", "aim"])
def test_c1761_database_statistics_equal_the_in_memory_ones(tmp_path, monkeypatch, measure):
    """Props. 1761, 1762: S9 (customer -> invoice -> invoice_line, a public link
    and a lookup), nothing truncated: every exact statistic is the same vector
    with the same sensitivity in the database and in memory."""
    base = _dp(_db(tmp_path, "S9_small_shop"), measure=measure, aim_rounds=10)
    spec = _dp(base, measure=measure, aim_rounds=10, max_degree=_no_truncation(base))
    mem = _stats(spec, "off", monkeypatch)
    sql = _stats(spec, "on", monkeypatch)
    assert any("correlations" in m or "pair" in m for m, _, _ in mem.calls)
    assert any(m.startswith("invoice_line") for m, _, _ in mem.calls)
    _assert_same(mem, sql)


def test_c1761_duckdb_file_statistics_equal_the_in_memory_ones(tmp_path, monkeypatch):
    """Props. 1761, 1762 on Parquet files read through DuckDB (no database server)."""
    pytest.importorskip("duckdb")
    pytest.importorskip("duckdb_engine")
    raw = write_scenario("S9_small_shop", {"kind": "parquet", "path": str(tmp_path / "pq")})
    raw["source"]["engine"] = "duckdb"
    base = _dp(raw)
    spec = _dp(base, max_degree=_no_truncation(base))
    _assert_same(_stats(spec, "off", monkeypatch), _stats(spec, "on", monkeypatch))


# --------------------------------------------------------------------- C1762
def test_c1762_chunked_frequency_tables_give_the_same_sums(tmp_path, monkeypatch):
    """Prop. 1763: frequency tables read 7 tuples at a time give the same statistics."""
    base = _dp(_db(tmp_path, "S9_small_shop"))
    spec = _dp(base, max_degree=_no_truncation(base))
    whole = _stats(spec, "on", monkeypatch)
    monkeypatch.setattr(RW, "CHUNK", 7)
    _assert_same(whole, _stats(spec, "on", monkeypatch))


# --------------------------------------------------------------------- C1763
def test_c1763_pushdown_statistics_respect_their_sensitivity(tmp_path, monkeypatch):
    """C902 on the pushdown path: ranges and levels public, the degree cap fixed;
    removing any one customer (and their sales) from the SQLite database moves
    every released statistic by at most its charged L2 sensitivity, replaying
    the full database's noisy outputs (adaptive composition).  Includes the
    customer with the most sales, where truncation binds."""
    raw = _db(tmp_path, "S2_star")
    db = raw["source"]["path"]
    real = load_for_fit(load_spec(copy.deepcopy(raw)), mode="off").tables
    pub = {"customer.region": {"levels": sorted(real["customer"]["region"].astype(str).unique())},
           "sales.channel": {"levels": sorted(real["sales"]["channel"].astype(str).unique())}}
    for t, c in (("customer", "age"), ("customer", "income"), ("sales", "quantity"), ("sales", "amount")):
        x = real[t][c]
        pub[f"{t}.{c}"] = {"min": float(x.min()), "max": float(x.max())}
    spec = _dp(raw, public=pub, max_degree={"sales": 4}, public_tables=["product"])
    base = _stats(spec, "on", monkeypatch)
    counts = real["sales"]["customer_id"].value_counts()
    ids = [counts.index[0]] + list(real["customer"]["customer_id"].sample(8, random_state=3))
    worst: dict = {}
    for i, cid in enumerate(ids):
        p = str(tmp_path / f"n{i}.db")
        with sqlite3.connect(db) as src_con, sqlite3.connect(p) as dst:
            src_con.backup(dst)
            dst.execute("DELETE FROM sales WHERE customer_id = ?", (cid,))
            dst.execute("DELETE FROM customer WHERE customer_id = ?", (cid,))
        s2 = copy.deepcopy(spec)
        s2["source"] = {"kind": "sqlite", "path": p}
        rep = _stats(s2, "on", monkeypatch, replay=base)
        for (name, v0, sens), (_, v1, _) in zip(base.calls, rep.calls):
            d = float(np.linalg.norm(v0 - v1))
            worst[name] = max(worst.get(name, 0.0), d / sens)
            assert d <= sens + 1e-9, (cid, name, d, sens)
    assert worst["sales: marginals"] > 0.3 and worst["customer: marginals"] > 0.3, worst


# --------------------------------------------------------------------- C1764
@pytest.mark.parametrize("measure", ["copula", "aim"])
def test_c1764_pushdown_run_keeps_the_guarantee(tmp_path, measure):
    """A whole DP run reads stage 1 in the database, spends within its budget,
    and releases valid links and the public table as given."""
    spec = _dp(_db(tmp_path, "S2_star"), epsilon=1.0, measure=measure, public_tables=["product"])
    res = dp.synthesize(spec, write=False, evaluate=False)
    assert res.report["stage1"]["source"] == "database"
    assert res.real is None                                  # nothing was loaded
    assert res.epsilon <= 1.0 + 1e-9 and res.rho <= res.config.rho * (1 + 1e-9)
    d = res.data
    assert d["sales"]["customer_id"].isin(d["customer"]["customer_id"]).all()
    assert len(d["product"]) == len(load_for_fit(spec, mode="off").tables["product"])
    off = dp.synthesize(_dp(spec, epsilon=1.0, measure=measure, public_tables=["product"], pushdown="off"),
                        write=False, evaluate=False)
    assert off.report["stage1"]["source"] == "memory"
    if measure == "copula":            # a fixed set of statistics; aim's rounds depend on the noise
        assert [c.mechanism for c in off.charges] == [c.mechanism for c in res.charges]
        assert [round(c.rho, 12) for c in off.charges] == [round(c.rho, 12) for c in res.charges]
    assert abs(off.rho - res.rho) <= 1e-9 * res.config.rho


def test_c1764_pushdown_on_refuses_a_file_source_before_reading(tmp_path):
    """pushdown: on with CSV files and no DuckDB is refused before the touch."""
    from relsynth.errors import PrivacyError
    spec = _dp(_db(tmp_path, "S2_star"), pushdown="on")
    spec["source"] = {"kind": "csv", "path": str(tmp_path / "nowhere"), "engine": "pandas"}
    touched = []
    with pytest.raises(PrivacyError, match="pushdown: on"):
        dp.synthesize(spec, write=False, evaluate=False, before_read=lambda: touched.append(1))
    assert not touched
