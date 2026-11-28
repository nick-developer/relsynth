# SPDX-License-Identifier: Apache-2.0
"""Two generic engine cases.

1. A line's key offset from a key of its order header (`offset_from` on a relationship whose base
   key is a `fact.header` column): the class of header/line facts whose ship, due or paid date key
   follows the order's date key (retail order lines, insurance claim lines).  The base key moves to
   the internal header table, so the offset is applied after the header columns are read back,
   and lookups through the offset key (the ship date's year) are read after it.
2. SCD2 version dates: versions of a natural key keep strictly increasing starts and positive
   lengths drawn from the source, also when many versions share a few release dates.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

import relsynth as rs
from relsynth.spec import SpecError, load_spec


def _date_dim(start="2020-01-01", days=900):
    d = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({"d_sk": np.arange(1, days + 1), "d_date": d.strftime("%Y-%m-%d"), "d_year": d.year})


def _header_db(tmp_path, names, lo, hi, n_orders=300, seed=3):
    """Lines of n_orders orders; the order's date key is a header column and the line's
    second date key = order date key + lo..hi days."""
    o_key, ln_no, base, late, amt = names
    rng = np.random.default_rng(seed)
    dd = _date_dim()
    rows, lid = [], 1
    for o in range(1, n_orders + 1):
        b = int(rng.integers(1, 700))
        for ln in range(1, int(rng.integers(1, 5)) + 1):
            rows.append((lid, o, ln, b, b + int(rng.integers(lo, hi + 1)), float(rng.gamma(2.0, 30.0))))
            lid += 1
    lines = pd.DataFrame(rows, columns=["line_id", o_key, ln_no, base, late, amt])
    db = tmp_path / f"{o_key}.db"
    with sqlite3.connect(db) as con:
        dd.to_sql("date_dim", con, index=False)
        lines.to_sql("lines", con, index=False)
    return db


def _header_spec(db, names, lo, hi, offset_from=None):
    o_key, ln_no, base, late, amt = names
    return {"version": 1, "name": "hk", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {
                "date_dim": {"role": "entity", "primary_key": "d_sk", "mode": "copy",
                             "columns": {"d_sk": {"type": "id"}, "d_date": {"type": "date"},
                                         "d_year": {"type": "integer"}}},
                "lines": {"role": "fact", "primary_key": "line_id",
                          "fact": {"header": {"key": o_key, "columns": [base], "line_number": ln_no}},
                          "columns": {"line_id": {"type": "id"}, o_key: {"type": "id"},
                                      ln_no: {"type": "integer"}, base: {"type": "id"}, late: {"type": "id"},
                                      "late_year": {"type": "lookup", "lookup": f"{late}.d_year"},
                                      amt: {"type": "float"}}}},
            "relationships": [
                {"child": "lines", "child_key": base, "parent": "date_dim", "parent_key": "d_sk"},
                {"child": "lines", "child_key": late, "parent": "date_dim", "parent_key": "d_sk",
                 "offset_from": offset_from or base, "order_by": "d_date"}],
            "rules": [{"table": "lines", "type": "check",
                       "expr": f"between(days({late}.d_date) - days({base}.d_date), {lo}, {hi})"}]}


@pytest.mark.parametrize("names,lo,hi", [
    (("order_no", "line_no", "sold_date_sk", "ship_date_sk", "sales_price"), 2, 90),   # retail order lines
    (("claim_id", "claim_line", "service_date_sk", "paid_date_sk", "paid_amount"), 0, 30),  # claim lines
])
def test_line_key_offset_from_header_key(tmp_path, names, lo, hi):
    o_key, _, base, late, _ = names
    db = _header_db(tmp_path, names, lo, hi)
    out = rs.run(_header_spec(db, names, lo, hi), write=False, privacy=False, seed=1).data
    li, dd = out["lines"], out["date_dim"].set_index("d_sk")
    gap = (pd.to_datetime(li[late].map(dd["d_date"])) - pd.to_datetime(li[base].map(dd["d_date"]))).dt.days
    # a header date drawn near the end of the date dimension can have no row lo..hi days later
    # (the key is then null, as for any offset key); every key that is set keeps the gap
    assert len(li) > 0 and gap.notna().mean() > 0.97 and gap.dropna().between(lo, hi).all()
    assert gap.nunique() > 5                                    # the gap's spread is kept
    assert li.groupby(o_key)[base].nunique().max() == 1         # one header date per order
    # the lookup through the offset key reads the offset key, not a stale value
    has = li[late].notna()
    assert (li.loc[has, "late_year"] == li.loc[has, late].map(dd["d_year"])).all()
    log_ = out.sample_log
    assert not any((log_.get("rows_removed_by_rules") or {}).values())


def test_offset_from_a_column_that_is_no_key_is_refused(tmp_path):
    names = ("order_no", "line_no", "sold_date_sk", "ship_date_sk", "sales_price")
    db = _header_db(tmp_path, names, 1, 5, n_orders=5)
    with pytest.raises(SpecError, match="offset_from"):
        load_spec(_header_spec(db, names, 1, 5, offset_from="sales_price"))


def _scd_db(tmp_path, shape, n_keys=250, seed=11):
    """Versions of n_keys natural keys. 'releases': starts are one of four release dates (many
    keys share each start, as catalogue items do); 'continuous': starts anywhere (staff history)."""
    rng = np.random.default_rng(seed)
    rel = pd.to_datetime(["2017-10-27", "2019-10-28", "2020-10-27", "2021-10-27"])
    rows, sk = [], 1
    for k in range(n_keys):
        nv = int(rng.integers(1, 4))
        if shape == "releases":
            starts = rel[np.sort(rng.choice(4, nv, replace=False))]
        else:
            starts = pd.Timestamp("2015-01-01") + pd.to_timedelta(np.sort(rng.choice(2500, nv, replace=False)),
                                                                   unit="D")
        for i, s in enumerate(starts):
            end = starts[i + 1] - pd.Timedelta(days=1) if i + 1 < nv else (
                pd.NaT if rng.random() < 0.7 else s + pd.Timedelta(days=int(rng.integers(100, 400))))
            rows.append((sk, f"K{k:05d}", s.strftime("%Y-%m-%d"),
                         None if pd.isna(end) else end.strftime("%Y-%m-%d"),
                         rng.choice(["red", "blue", "green"]), float(rng.gamma(3.0, 10.0))))
            sk += 1
    df = pd.DataFrame(rows, columns=["sk", "nk", "start_date", "end_date", "colour", "price"])
    db = tmp_path / f"scd_{shape}.db"
    with sqlite3.connect(db) as con:
        df.to_sql("product", con, index=False)
    return db


@pytest.mark.parametrize("shape", ["releases", "continuous"])
def test_scd2_versions_keep_increasing_starts_and_positive_lengths(tmp_path, shape):
    db = _scd_db(tmp_path, shape)
    spec = {"version": 1, "name": "scd", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {"product": {"role": "entity", "primary_key": "sk",
                                   "scd": {"type": 2, "natural_key": "nk", "valid_from": "start_date",
                                           "valid_to": "end_date"},
                                   "columns": {"sk": {"type": "id"}, "nk": {"type": "id"},
                                               "start_date": {"type": "date"}, "end_date": {"type": "date"},
                                               "colour": {"type": "categorical"}, "price": {"type": "float"}}}},
            "relationships": [],
            "rules": [{"table": "product", "type": "check",
                       "expr": "isnull(end_date) or days(start_date) < days(end_date)"}]}
    p = rs.run(spec, write=False, privacy=False, seed=4).data["product"]
    s, e = pd.to_datetime(p["start_date"]), pd.to_datetime(p["end_date"])
    assert len(p) > 0 and p["nk"].duplicated().any()             # versions are generated
    assert (e.isna() | (s < e)).all()                            # every closed version has a length
    srt = p.assign(s=s).sort_values(["nk", "s"])
    assert (srt.groupby("nk")["s"].diff().dropna() > pd.Timedelta(0)).all()   # strictly increasing
    if shape == "releases":     # moved starts stay on the release dates, never past the last one
        rel = {"2017-10-27", "2019-10-28", "2020-10-27", "2021-10-27"}
        assert s.dt.strftime("%Y-%m-%d").isin(rel).mean() > 0.95 and s.max() <= pd.Timestamp("2021-10-27")
    ln = (e - s).dt.days.dropna()
    assert (ln <= 1).mean() < 0.05                               # not collapsed to one-day versions


def test_scd2_first_starts_after_the_last_reversioning_are_kept(tmp_path):
    """Entities that appear after the source's last later-version start (new customers, new products) keep
    their first start; only later versions move onto the source's later-version starts."""
    rng = np.random.default_rng(5)
    rows, sk = [], 1
    for k in range(400):
        if k < 250:     # older keys, re-versioned up to 2019
            starts = pd.Timestamp("2015-01-01") + pd.to_timedelta(np.sort(rng.choice(1500, 2, replace=False)), unit="D")
        else:           # new keys, one version, arriving 2020-2021
            starts = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 700)))]
        for i, s in enumerate(starts):
            end = starts[i + 1] - pd.Timedelta(days=1) if i + 1 < len(starts) else pd.NaT
            rows.append((sk, f"K{k:05d}", s.strftime("%Y-%m-%d"), None if pd.isna(end) else end.strftime("%Y-%m-%d"),
                         float(rng.gamma(3.0, 10.0))))
            sk += 1
    df = pd.DataFrame(rows, columns=["sk", "nk", "start_date", "end_date", "price"])
    db = tmp_path / "scd_new.db"
    with sqlite3.connect(db) as con:
        df.to_sql("customer", con, index=False)
    spec = {"version": 1, "name": "scd", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {"customer": {"role": "entity", "primary_key": "sk",
                                    "scd": {"type": 2, "natural_key": "nk", "valid_from": "start_date",
                                            "valid_to": "end_date"},
                                    "columns": {"sk": {"type": "id"}, "nk": {"type": "id"},
                                                "start_date": {"type": "date"}, "end_date": {"type": "date"},
                                                "price": {"type": "float"}}}},
            "relationships": []}
    c = rs.run(spec, write=False, privacy=False, seed=3).data["customer"]
    first = pd.to_datetime(c["start_date"]).groupby(c["nk"]).min()
    real_first = pd.to_datetime(df["start_date"]).groupby(df["nk"]).min()
    top = pd.Timestamp("2019-02-09")    # after every later-version start of the source
    assert (real_first > top).mean() > 0.3
    # before the fix every first start moved onto a source start (0 % past top); rows are grouped into
    # keys independently of their start, so late rows that land in two-version keys still move (the
    # share stays below the source's 37 %)
    assert (first > top).mean() > 0.1
    assert first.nunique() > 50          # first starts not collapsed onto one date


@pytest.mark.parametrize("shape", ["releases", "continuous"])
def test_scd2_source_whose_versions_end_before_they_start_keeps_its_end_dates(tmp_path, shape):
    """Class: SCD sources whose closed versions end on or before their start (end dates loaded
    from an earlier system, placeholder ends; AdventureWorksDW products end in 2007/2008 after
    starting in 2011/2012). Their end dates are kept as the source has them, not replaced by
    start + 1 day; a well-formed source (the tests above) still gets positive lengths."""
    rng = np.random.default_rng(7)
    rows = []
    for k in range(500):
        if shape == "releases":
            s = pd.Timestamp(str(rng.choice(["2011-07-01", "2012-07-01", "2013-07-01"])))
            e = {"2011-07-01": pd.Timestamp("2007-12-28"), "2012-07-01": pd.Timestamp("2008-12-27")}.get(
                s.strftime("%Y-%m-%d"), pd.NaT)
        else:
            s = pd.Timestamp("2016-01-01") + pd.Timedelta(days=int(rng.integers(0, 1500)))
            e = s - pd.Timedelta(days=int(rng.integers(200, 900))) if rng.random() < 0.5 else pd.NaT
        rows.append((k + 1, f"K{k:05d}", s.strftime("%Y-%m-%d"), None if pd.isna(e) else e.strftime("%Y-%m-%d"),
                     float(rng.gamma(3.0, 10.0))))
    df = pd.DataFrame(rows, columns=["sk", "nk", "start_date", "end_date", "price"])
    db = tmp_path / "scd_ill.db"
    with sqlite3.connect(db) as con:
        df.to_sql("product", con, index=False)
    spec = {"version": 1, "name": "scd", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {"product": {"role": "entity", "primary_key": "sk",
                                   "scd": {"type": 2, "natural_key": "nk", "valid_from": "start_date",
                                           "valid_to": "end_date"},
                                   "columns": {"sk": {"type": "id"}, "nk": {"type": "id"},
                                               "start_date": {"type": "date"}, "end_date": {"type": "date"},
                                               "price": {"type": "float"}}}},
            "relationships": []}
    p = rs.run(spec, write=False, privacy=False, seed=2).data["product"]
    s, e = pd.to_datetime(p["start_date"]), pd.to_datetime(p["end_date"])
    closed = e.notna()
    assert closed.mean() > 0.3
    assert ((e - s).dt.days[closed] == 1).mean() < 0.05          # before: every closed version was start + 1 day
    assert (e[closed] < s[closed]).mean() > 0.8                  # as in the source (100 %)
    if shape == "releases":                                      # the source's end dates
        assert e[closed].dt.strftime("%Y-%m-%d").isin({"2007-12-28", "2008-12-27"}).mean() > 0.8
