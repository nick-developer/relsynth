# SPDX-License-Identifier: Apache-2.0
"""An exact derive that reads a soft rule's column sees the value the soft rule set (Engine._after_soft).

Two classes of data: a key set by a soft rule and read through as a lookup (payments in the format's own
currency on most rows, the received amount converted at that currency's rate), and a plain column set by
a soft rule and read directly (a standard discount on most rows, the net price from it).
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

import relsynth as rs


def _payments_db(tmp_path, n=6000, seed=3):
    rng = np.random.default_rng(seed)
    cur = pd.DataFrame({"code": ["USD", "EUR", "BTC"], "rate": [1.0, 0.9, 0.00005]})
    fmt = rng.choice(["Card", "Wire", "Crypto"], n, p=[0.6, 0.3, 0.1])
    pay = np.where(fmt == "Crypto", "BTC", rng.choice(["USD", "EUR"], n))
    flip = (fmt == "Crypto") & (rng.random(n) < 0.02)          # the soft rule's exceptions
    pay = np.where(flip, "USD", pay)
    rec = pay.copy()
    paid = np.round(rng.gamma(2.0, 200.0, n), 2)
    rate = dict(zip(cur.code, cur.rate))
    received = paid * np.array([rate[r] for r in rec]) / np.array([rate[p] for p in pay])
    tx = pd.DataFrame({"tid": np.arange(1, n + 1), "fmt": fmt, "pay": pay, "rec": rec, "paid": paid,
                       "received": received})
    db = tmp_path / "pay.db"
    with sqlite3.connect(db) as con:
        cur.to_sql("currency", con, index=False)
        tx.to_sql("tx", con, index=False)
    return db


def _payments_spec(db):
    return {"version": 1, "name": "pay", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {
                "currency": {"role": "entity", "mode": "copy", "primary_key": "code",
                             "columns": {"code": {"type": "id"}, "rate": {"type": "float"}}},
                "tx": {"role": "entity", "primary_key": "tid",
                       "columns": {"tid": {"type": "id"}, "fmt": {"type": "categorical"},
                                   "pay": {"type": "id"}, "rec": {"type": "id"},
                                   "paid": {"type": "float"}, "received": {"type": "float"}}}},
            "relationships": [{"child": "tx", "child_key": "pay", "parent": "currency", "parent_key": "code"},
                              {"child": "tx", "child_key": "rec", "parent": "currency", "parent_key": "code"}],
            "rules": [{"table": "tx", "type": "derive", "expr": "pay = where(fmt == 'Crypto', 'BTC', pay)",
                       "holds": 0.98, "violation": "model"},
                      {"table": "tx", "type": "derive", "expr": "received = paid * rec.rate / pay.rate",
                       "tolerance": 0.01}],
            "synthesis": {"seed": 5}}


def test_lookup_through_a_soft_key_reads_the_final_key(tmp_path):
    spec = _payments_spec(_payments_db(tmp_path))
    tx = rs.run(spec, write=False, seed=1).data["tx"]
    rate = {"USD": 1.0, "EUR": 0.9, "BTC": 0.00005}
    expect = tx["paid"] * tx["rec"].map(rate) / tx["pay"].map(rate)
    ok = (tx["received"] - expect).abs() <= (0.002 * expect.abs()).clip(lower=0.011)
    assert ok.mean() > 0.999
    assert ((tx["fmt"] == "Crypto") & (tx["pay"] == "BTC")).sum() > 0


def _discount_db(tmp_path, n=5000, seed=4):
    rng = np.random.default_rng(seed)
    seg = rng.choice(["retail", "trade"], n, p=[0.7, 0.3])
    disc = np.where(seg == "trade", 0.15, rng.choice([0.0, 0.05], n))
    disc = np.where((seg == "trade") & (rng.random(n) < 0.03), 0.10, disc)
    gross = np.round(rng.gamma(3.0, 40.0, n), 2)
    df = pd.DataFrame({"oid": np.arange(1, n + 1), "seg": seg, "disc": disc, "gross": gross,
                       "net": np.round(gross * (1 - disc), 2)})
    db = tmp_path / "disc.db"
    with sqlite3.connect(db) as con:
        df.to_sql("orders", con, index=False)
    return db


def test_derive_reading_a_soft_column_directly(tmp_path):
    db = _discount_db(tmp_path)
    spec = {"version": 1, "name": "disc", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {"orders": {"role": "entity", "primary_key": "oid",
                                  "columns": {"oid": {"type": "id"}, "seg": {"type": "categorical"},
                                              "disc": {"type": "float"}, "gross": {"type": "float"},
                                              "net": {"type": "float"}}}},
            "relationships": [],
            "rules": [{"table": "orders", "type": "derive", "expr": "disc = where(seg == 'trade', 0.15, disc)",
                       "holds": 0.97, "violation": "model"},
                      {"table": "orders", "type": "derive", "expr": "net = gross * (1 - disc)", "tolerance": 0.01}],
            "synthesis": {"seed": 5}}
    o = rs.run(spec, write=False, seed=2).data["orders"]
    assert ((o["net"] - o["gross"] * (1 - o["disc"])).abs() <= 0.011).mean() > 0.999


def test_running_balance_of_a_column_derived_from_a_soft_one(tmp_path):
    """A cumulative rule whose step reads an exact derive of a soft column (running spend per customer of
    net prices) follows the soft rule's final values."""
    rng = np.random.default_rng(6)
    n = 4000
    cust = rng.integers(1, 200, n)
    seg = np.where(cust % 3 == 0, "trade", "retail")
    disc = np.where(seg == "trade", 0.15, rng.choice([0.0, 0.05], n))
    disc = np.where((seg == "trade") & (rng.random(n) < 0.03), 0.10, disc)
    gross = np.round(rng.gamma(3.0, 40.0, n), 2)
    df = pd.DataFrame({"oid": np.arange(1, n + 1), "cust": cust, "seg": seg, "disc": disc, "gross": gross,
                       "net": gross * (1 - disc)})
    df["spend"] = df.sort_values("oid").groupby("cust")["net"].cumsum()
    db = tmp_path / "spend.db"
    with sqlite3.connect(db) as con:
        df.to_sql("orders", con, index=False)
    spec = {"version": 1, "name": "spend", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {"orders": {"role": "entity", "primary_key": "oid",
                                  "columns": {"oid": {"type": "id"}, "cust": {"type": "categorical"},
                                              "seg": {"type": "categorical"}, "disc": {"type": "float"},
                                              "gross": {"type": "float"}, "net": {"type": "float"},
                                              "spend": {"type": "float"}}}},
            "relationships": [],
            "rules": [{"table": "orders", "type": "derive", "expr": "disc = where(seg == 'trade', 0.15, disc)",
                       "holds": 0.97, "violation": "model"},
                      {"table": "orders", "type": "derive", "expr": "net = gross * (1 - disc)", "tolerance": 0.01},
                      {"table": "orders", "type": "cumulative", "column": "spend", "step": "net", "by": "cust",
                       "order": "oid", "opening": 0}],
            "synthesis": {"seed": 5}}
    o = rs.run(spec, write=False, seed=2).data["orders"]
    o = o.assign(k=pd.to_numeric(o["oid"])).sort_values("k")
    expect = o.groupby("cust")["net"].cumsum()
    assert ((o["spend"] - expect).abs() <= 0.02 + 1e-6 * expect.abs()).mean() > 0.999
