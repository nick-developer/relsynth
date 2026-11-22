# SPDX-License-Identifier: Apache-2.0
"""Dates of a line offset from a date of its header (`offset_from: fk.column`) and offsets that
chain (receipt = ship + gap, ship = order date + gap): the class of header/line tables with an
accumulating snapshot (order lines, claim lines, encounter events).  The gap is modelled from the
source rows, so the rules on it hold on every synthetic row without rejection or removal."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

import relsynth as rs
from relsynth.spec import SpecError, load_spec, validate


def _db(tmp_path, n_orders=400, seed=5):
    rng = np.random.default_rng(seed)
    od = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 700, n_orders), unit="D")
    orders = pd.DataFrame({"o_id": np.arange(1, n_orders + 1), "o_date": od.strftime("%Y-%m-%d"),
                           "o_prio": rng.choice(["1-URGENT", "2-HIGH", "3-LOW"], n_orders)})
    rows = []
    lid = 1
    for o, d in zip(orders["o_id"], od):
        for _ in range(int(rng.integers(1, 6))):
            ship = d + pd.Timedelta(days=int(rng.integers(1, 31)))
            commit = d + pd.Timedelta(days=int(rng.integers(10, 41)))
            receipt = ship + pd.Timedelta(days=int(rng.integers(1, 8)))
            rows.append((lid, o, ship.strftime("%Y-%m-%d"), commit.strftime("%Y-%m-%d"),
                         receipt.strftime("%Y-%m-%d"), float(rng.gamma(2.0, 50.0))))
            lid += 1
    lines = pd.DataFrame(rows, columns=["l_id", "l_order", "l_ship", "l_commit", "l_receipt", "l_price"])
    db = tmp_path / "ol.db"
    with sqlite3.connect(db) as con:
        orders.to_sql("orders", con, index=False)
        lines.to_sql("lines", con, index=False)
    return db


def _spec(db, ship_base="l_order.o_date", receipt_base="l_ship"):
    return {"version": 1, "name": "ol", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {
                "orders": {"role": "entity", "primary_key": "o_id",
                           "columns": {"o_id": {"type": "id"}, "o_date": {"type": "date"},
                                       "o_prio": {"type": "categorical"}}},
                "lines": {"role": "fact", "primary_key": "l_id", "fact": {"anchor": "l_order"},
                          "columns": {"l_id": {"type": "id"}, "l_order": {"type": "id"},
                                      "l_ship": {"type": "date", "offset_from": ship_base},
                                      "l_commit": {"type": "date", "offset_from": "l_order.o_date"},
                                      "l_receipt": {"type": "date", "offset_from": receipt_base},
                                      "l_price": {"type": "float"}}}},
            "relationships": [{"child": "lines", "child_key": "l_order", "parent": "orders",
                               "parent_key": "o_id"}],
            "rules": [
                {"table": "lines", "type": "check", "expr": "between(days(l_ship) - days(l_order.o_date), 1, 30)"},
                {"table": "lines", "type": "check",
                 "expr": "between(days(l_commit) - days(l_order.o_date), 10, 40)"},
                {"table": "lines", "type": "check", "expr": "between(days(l_receipt) - days(l_ship), 1, 7)"}]}


def _gaps(out):
    o = out["orders"].set_index("o_id")["o_date"]
    li = out["lines"]
    od = pd.to_datetime(li["l_order"].map(o))
    return ((pd.to_datetime(li["l_ship"]) - od).dt.days, (pd.to_datetime(li["l_commit"]) - od).dt.days,
            (pd.to_datetime(li["l_receipt"]) - pd.to_datetime(li["l_ship"])).dt.days)


def test_offsets_through_a_key_and_chained_hold_on_every_row(tmp_path):
    db = _db(tmp_path)
    r = rs.run(_spec(db), write=False, privacy=False, seed=1)
    ship, commit, receipt = _gaps(r.data)
    assert len(r.data["lines"]) > 0
    assert ship.between(1, 30).all() and commit.between(10, 40).all() and receipt.between(1, 7).all()
    log_ = r.data.sample_log
    assert "rows_removed_by_rules" in log_ and not any(log_["rows_removed_by_rules"].values())
    # valid on the first draw: no row needed a rejection redraw
    assert (log_.get("redrawn_rows") or {}).get("lines", {}).get("invalid_first_draw", 0) == 0
    # the gap's spread is kept, not collapsed onto one value
    assert ship.nunique() > 10 and receipt.nunique() >= 5


def test_validation_of_offset_paths(tmp_path):
    db = _db(tmp_path, n_orders=5)
    assert not validate(load_spec(_spec(db)))
    for base, msg in (("l_order.o_missing", "o_missing"), ("l_price.o_date", "not a foreign key"),
                      ("l_order.o_prio", "offset_from l_order.o_prio")):
        with pytest.raises(SpecError, match=msg):
            load_spec(_spec(db, ship_base=base))


def test_offset_cycle_is_refused(tmp_path):
    db = _db(tmp_path, n_orders=5)
    with pytest.raises(SpecError, match="cycle"):
        load_spec(_spec(db, ship_base="l_receipt"))
