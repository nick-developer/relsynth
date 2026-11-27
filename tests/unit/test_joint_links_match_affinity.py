# SPDX-License-Identifier: Apache-2.0
"""Two generic engine cases of facts with several links (Props 1870, 1871).

1. Role-playing extra links (two links of a fact to the same parent table: payment and receiving
   currency, origin and destination airport, ship-from and ship-to warehouse) are drawn jointly
   over their observed combinations, with combinations rarer than the small-count floor left out.
   Drawn one by one, they agree only by chance.
2. A declared link-affinity match is a hidden attribute of the row, drawn with the other
   attributes, and the link is chosen to agree with it.  Classes: own-account events (a
   reinvestment is a self-loop), regional shipments (truck only within a region), currency pairs
   (cross-currency payments are wires).
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

import relsynth as rs
from relsynth.engine import _log_bf_joint
from relsynth.spec import SpecError, load_spec


def _transfers(tmp_path, n=6000, seed=1, reinvest=False, rare_pair=False):
    """Transfers between 300 accounts in 12 currencies: 97 % in one currency; a cross-currency
    payment is a wire.  reinvest: 8 % reinvestments, each a self-loop, and 2 % other self-loops.
    rare_pair: one currency pair held by 3 rows only."""
    rng = np.random.default_rng(seed)
    cur = pd.DataFrame({"cur": [f"C{i:02d}" for i in range(12)], "rate": np.round(rng.uniform(0.5, 5, 12), 3)})
    acc = pd.DataFrame({"acc": np.arange(1, 301), "kind": rng.choice(["p", "b"], 300)})
    pc = rng.choice(12, n, p=np.r_[0.4, 0.2, 0.1, np.full(9, 0.3 / 9)])
    cross = rng.random(n) < 0.03
    rc = np.where(cross, (pc + 1 + rng.integers(0, 2, n)) % 12, pc)
    fmt = np.where(cross, "Wire", rng.choice(["ACH", "Card", "Cash", "Wire"], n, p=[.4, .3, .2, .1])).astype(object)
    fr, to = rng.integers(1, 301, n), rng.integers(1, 301, n)
    if rare_pair:
        same = np.flatnonzero(~cross)[:3]
        pc[same], rc[same], fmt[same] = 0, 11, "Wire"            # C00 -> C11: 3 rows
        rc[(pc == 0) & (rc == 11) & ~np.isin(np.arange(n), same)] = 1
    if reinvest:
        r = (rng.random(n) < 0.08) & ~cross
        fmt[r], to[r] = "Reinv", fr[r]
        o = ~r & (rng.random(n) < 0.02)
        to[o] = fr[o]
    t = pd.DataFrame({"tid": np.arange(1, n + 1), "fr": fr, "to": to, "pay": cur.cur.values[pc],
                      "recv": cur.cur.values[rc], "fmt": fmt, "amt": np.round(rng.lognormal(4, 1, n), 2)})
    db = tmp_path / "transfers.db"
    with sqlite3.connect(db) as con:
        cur.to_sql("cur", con, index=False)
        acc.to_sql("acc", con, index=False)
        t.to_sql("t", con, index=False)
    spec = {"version": 1, "name": "transfers", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {
                "cur": {"role": "entity", "mode": "copy", "primary_key": "cur",
                        "columns": {"cur": "id", "rate": "float"}},
                "acc": {"role": "entity", "primary_key": "acc", "columns": {"acc": "id", "kind": "categorical"}},
                "t": {"role": "fact", "primary_key": "tid", "fact": {"anchor": "fr", "partner": "to"},
                      "columns": {"tid": "id", "fr": "id", "to": "id", "pay": "id", "recv": "id",
                                  "fmt": "categorical", "amt": "float"}}},
            "relationships": [{"child": "t", "child_key": "fr", "parent": "acc", "parent_key": "acc"},
                              {"child": "t", "child_key": "to", "parent": "acc", "parent_key": "acc"},
                              {"child": "t", "child_key": "pay", "parent": "cur", "parent_key": "cur"},
                              {"child": "t", "child_key": "recv", "parent": "cur", "parent_key": "cur"}],
            "rules": [{"table": "t", "type": "check", "expr": "pay == recv or fmt == 'Wire'"}]}
    if reinvest:
        spec["rules"].append({"table": "t", "type": "check", "expr": "fmt != 'Reinv' or fr == to"})
    return spec, t


def _shipments(tmp_path, n=6000, seed=2):
    """Shipments between 120 warehouses in 6 regions (synthesised, not copied): 90 % within the
    sender's region; a truck runs only within a region."""
    rng = np.random.default_rng(seed)
    wh = pd.DataFrame({"wh": np.arange(1, 121), "region": np.repeat(list("NSEWCX"), 20),
                       "cap": rng.gamma(3, 100, 120).round(1)})
    cu = pd.DataFrame({"cu": np.arange(1, 301), "seg": rng.choice(["a", "b", "c"], 300)})
    pr = pd.DataFrame({"pr": np.arange(1, 101), "cat": rng.choice(["x", "y"], 100)})
    f = rng.integers(1, 121, n)
    same = rng.random(n) < 0.9
    to = np.where(same, (f - 1) // 20 * 20 + 1 + rng.integers(0, 20, n), rng.integers(1, 121, n))
    intra = (f - 1) // 20 == (to - 1) // 20
    mode = np.where(intra, rng.choice(["truck", "rail"], n, p=[.7, .3]), rng.choice(["rail", "air"], n))
    s = pd.DataFrame({"sid": np.arange(1, n + 1), "cu": rng.integers(1, 301, n), "pr": rng.integers(1, 101, n),
                      "wf": f, "wt": to, "mode": mode})
    db = tmp_path / "ship.db"
    with sqlite3.connect(db) as con:
        for name, d in (("wh", wh), ("cu", cu), ("pr", pr), ("s", s)):
            d.to_sql(name, con, index=False)
    spec = {"version": 1, "name": "ship", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {
                "wh": {"role": "entity", "primary_key": "wh",
                       "columns": {"wh": "id", "region": "categorical", "cap": "float"}},
                "cu": {"role": "entity", "primary_key": "cu", "columns": {"cu": "id", "seg": "categorical"}},
                "pr": {"role": "entity", "primary_key": "pr", "columns": {"pr": "id", "cat": "categorical"}},
                "s": {"role": "fact", "primary_key": "sid", "fact": {"anchor": "cu", "partner": "pr"},
                      "link_affinity": [{"match": ["wf.region", "wt.region"]}],
                      "columns": {"sid": "id", "cu": "id", "pr": "id", "wf": "id", "wt": "id",
                                  "mode": "categorical"}}},
            "relationships": [{"child": "s", "child_key": "cu", "parent": "cu", "parent_key": "cu"},
                              {"child": "s", "child_key": "pr", "parent": "pr", "parent_key": "pr"},
                              {"child": "s", "child_key": "wf", "parent": "wh", "parent_key": "wh"},
                              {"child": "s", "child_key": "wt", "parent": "wh", "parent_key": "wh"}],
            "rules": [{"table": "s", "type": "check", "expr": "mode != 'truck' or wf.region == wt.region"}]}
    return spec, s, wh


def _tvd(a: pd.Series, b: pd.Series) -> float:
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    return 0.5 * float((pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum())


def _run(spec, **synthesis):
    spec = {**spec, "synthesis": {**spec.get("synthesis", {}), **synthesis}}
    out = rs.run(spec, write=False, privacy=False, seed=1)
    assert out.ok, out.errors
    return out.data, out.model.synthesizer.log["decisions"]


def test_log_bf_joint_detects_dependence_only():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 6, 3000)
    indep = np.bincount(a * 6 + rng.integers(0, 6, 3000), minlength=36)
    dep = np.bincount(a * 6 + np.where(rng.random(3000) < 0.9, a, rng.integers(0, 6, 3000)), minlength=36)
    assert _log_bf_joint(indep, (6, 6)) < np.log(100)
    assert _log_bf_joint(dep, (6, 6)) > 100


def test_role_playing_links_drawn_jointly(tmp_path):
    spec, src = _transfers(tmp_path)
    real_pairs = set(zip(src.pay, src.recv))
    off, _ = _run(spec, facts={"joint_links": "off"})
    on, dec = _run(spec)
    s_off, s_on = off["t"], on["t"]
    same_real = float((src.pay == src.recv).mean())
    # one by one, the two currencies agree only by chance and most combinations are new
    assert (s_off.pay == s_off.recv).mean() < same_real - 0.2
    assert "drawn jointly" in dec["t"]["joint links pay + recv"]
    assert abs((s_on.pay == s_on.recv).mean() - same_real) < 0.04
    assert np.mean([p not in real_pairs for p in zip(s_on.pay, s_on.recv)]) == 0.0
    assert _tvd(s_on.fmt, src.fmt) < 0.05 < _tvd(s_off.fmt, src.fmt)


def test_rare_combination_left_out(tmp_path):
    spec, src = _transfers(tmp_path, rare_pair=True)
    assert ((src.pay == "C00") & (src.recv == "C11")).sum() == 3
    on, dec = _run(spec)
    assert "fewer than 5 rows" in dec["t"]["joint links pay + recv"]
    assert not ((on["t"].pay == "C00") & (on["t"].recv == "C11")).any()


def test_joint_links_off_per_fact(tmp_path):
    spec, _ = _transfers(tmp_path)
    spec["tables"]["t"]["fact"]["joint_links"] = "off"
    _, dec = _run(spec)
    assert not any(k.startswith("joint links") for k in dec["t"])


def test_self_loop_match_follows_attributes(tmp_path):
    spec, src = _transfers(tmp_path, reinvest=True)
    spec["tables"]["t"]["link_affinity"] = [{"match": ["fr.acc", "to.acc"]}]
    share, _ = _run(spec, structure={"link_affinity": "share"})
    rows, dec = _run(spec)
    real_reinv = float((src.fmt == "Reinv").mean())
    # the share model places self-loops regardless of the format: reinvestments are rejected
    assert (share["t"].fmt == "Reinv").mean() < real_reinv / 2
    assert "hidden attribute" in dec["t"]["link affinity fr.acc = to.acc"]
    s = rows["t"]
    assert abs((s.fmt == "Reinv").mean() - real_reinv) < 0.015
    assert ((s.fmt != "Reinv") | (s.fr == s.to)).all()
    other = (s.fr == s.to)[s.fmt != "Reinv"].mean()
    assert abs(other - (src.fr == src.to)[src.fmt != "Reinv"].mean()) < 0.015
    assert "__match_0" not in s.columns


def test_match_on_role_playing_links_exact(tmp_path):
    spec, src = _transfers(tmp_path)
    spec["tables"]["t"]["link_affinity"] = [{"match": ["pay", "recv.cur"]}]
    on, _ = _run(spec)
    s = on["t"]
    real_pairs = set(zip(src.pay, src.recv))
    assert abs((s.pay == s.recv).mean() - (src.pay == src.recv).mean()) < 0.01
    assert np.mean([p not in real_pairs for p in zip(s.pay, s.recv)]) == 0.0
    assert _tvd(s.fmt, src.fmt) < 0.03


def test_regional_match_follows_attributes(tmp_path):
    spec, src, wh = _shipments(tmp_path)
    rw = wh.set_index("wh").region
    intra = src.wf.map(rw).to_numpy() == src.wt.map(rw).to_numpy()
    share, _ = _run(spec, structure={"link_affinity": "share"})
    rows, _ = _run(spec)
    for out in (share, rows):
        w = out["wh"].set_index("wh").region
        s = out["s"]
        assert abs((s.wf.map(w).to_numpy() == s.wt.map(w).to_numpy()).mean() - intra.mean()) < 0.03
    # trucks within a region: the share model moves links after the rows are drawn, so trucks
    # drawn on cross-region rows are rejected and the mode shares shift
    assert _tvd(share["s"]["mode"], src["mode"]) > 0.1
    assert _tvd(rows["s"]["mode"], src["mode"]) < 0.03


def test_link_affinity_model_validated(tmp_path):
    spec, _ = _transfers(tmp_path)
    spec["tables"]["t"]["link_affinity"] = [{"match": ["fr.acc", "to.acc"], "model": "joint"}]
    with pytest.raises(SpecError, match="model must be rows or share"):
        load_spec(spec)
