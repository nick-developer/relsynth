# SPDX-License-Identifier: Apache-2.0
"""Reproducible drafting: seeded samples of tables without a key, and the effect-size
test of drafted ratio columns (bench finding: MIMIC-IV demo drafts differed per run)."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from relsynth.discovery import _entropy, _step_rng
from relsynth.pipeline import parse_source
from relsynth.source import Source, _mix64


def _keyless_db(path, df, shuffle_seed=None):
    """A table with no primary or unique key, rows written in a given order."""
    if shuffle_seed is not None:
        df = df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE ev (a INTEGER, b REAL, c TEXT)")
    con.executemany("INSERT INTO ev VALUES (?, ?, ?)", df[["a", "b", "c"]].itertuples(index=False, name=None))
    con.commit()
    con.close()
    return f"sqlite:{path}"


def _frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.integers(0, 50, n), "b": np.round(rng.gamma(2, 3, n), 2),
                         "c": rng.choice(["x", "y", "z", None], n)})


def _rows(df):
    return sorted(map(repr, df.astype(object).where(df.notna(), None).itertuples(index=False, name=None)))


def test_keyless_sample_is_seeded_and_ignores_row_order(tmp_path):
    df = _frame()
    s1 = Source(parse_source(_keyless_db(tmp_path / "a.db", df)))
    s2 = Source(parse_source(_keyless_db(tmp_path / "b.db", df, shuffle_seed=3)))
    a, whole = s1.sample("ev", 400)
    b, _ = s1.sample("ev", 400)
    c, _ = s2.sample("ev", 400)
    assert not whole and len(a) == 400
    assert a.equals(b)                                 # the same database twice: the same sample
    assert _rows(a) == _rows(c)                        # another physical order: the same rows
    assert s1.unseeded == [] and s2.unseeded == []     # no ORDER BY random() any more
    # another seed gives another sample
    d, _ = Source(parse_source(f"sqlite:{tmp_path / 'a.db'}"), seed=8).sample("ev", 400)
    assert _rows(a) != _rows(d)


def test_keyless_sample_is_uniform_with_duplicate_rows(tmp_path):
    # 200 distinct rows, one of them copied 200 times: every row (copies too) must have
    # the same inclusion chance n / N, so the copies' share in the sample stays near
    # their share in the table (a sample by content hash alone would take all or none)
    base = pd.DataFrame({"a": np.arange(200), "b": np.arange(200) / 10.0, "c": ["u"] * 200})
    dup = pd.DataFrame({"a": [-1] * 200, "b": [0.5] * 200, "c": ["dup"] * 200})
    df = pd.concat([base, dup], ignore_index=True)
    path = _keyless_db(tmp_path / "d.db", df)
    shares = []
    for seed in range(40):
        s, _ = Source(parse_source(path), seed=seed).sample("ev", 100)
        assert len(s) == 100
        shares.append(float((s["c"] == "dup").mean()))
    # table share 0.5; hypergeometric sd of one share ~0.043, of the mean of 40 ~0.007
    assert abs(np.mean(shares) - 0.5) < 0.03
    assert 0.3 < min(shares) and max(shares) < 0.7


def test_mix64_is_a_bijection_on_a_range():
    x = np.arange(100_000, dtype=np.uint64)
    assert len(np.unique(_mix64(x))) == len(x)


def test_step_rng_is_keyed():
    a = _step_rng("ratio", "t", "y", "x").permutation(50)
    b = _step_rng("ratio", "t", "y", "x").permutation(50)
    c = _step_rng("ratio", "t", "y", "z").permutation(50)
    assert (a == b).all() and not (a == c).all()


def test_entropy():
    assert _entropy([1, 1, 1]) == pytest.approx(0.0)
    assert _entropy([1, 2, 3, 4]) == pytest.approx(np.log(4))


def _ratio_db(path, n=20000, seed=0):
    """A parent with a price and a small integer; a child with (i) a true ratio
    (price times one of three discounts) and (ii) a hierarchical code with a few
    values (1.1, 1.2, 2.1, ...) that depends only weakly on the parent's dose count:
    some factor list reproduces it on every row, under random pairing on all but a
    few tenths of a percent, which 20,000 rows make "significant"."""
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE item (item_id INTEGER PRIMARY KEY, price REAL, doses INTEGER)")
    con.execute("CREATE TABLE line (line_id INTEGER PRIMARY KEY, item_id INTEGER REFERENCES item(item_id), "
                "qty INTEGER, unit_price REAL, ordinal REAL)")
    items = [(i, float(np.round(rng.gamma(3, 40) + 5, 2)), int(rng.integers(1, 7))) for i in range(300)]
    con.executemany("INSERT INTO item VALUES (?, ?, ?)", items)
    price = {i: p for i, p, _ in items}
    doses = {i: d for i, _, d in items}
    one = (np.array([1.1, 1.2, 1.3, 1.4, 1.5]), np.array([0.84, 0.10, 0.03, 0.02, 0.01]))
    two = (np.array([1.1, 1.2, 1.3, 2.1, 2.2]), np.array([0.80, 0.10, 0.04, 0.04, 0.02]))
    rows = []
    for j in range(n):
        i = int(rng.integers(0, 300))
        disc = rng.choice([1.0, 0.9, 0.8], p=[0.7, 0.2, 0.1])
        codes, probs = two if doses[i] >= 5 else one       # a second step only for frequent doses
        rows.append((j, i, int(rng.integers(1, 5)), float(np.round(price[i] * disc, 2)),
                     float(rng.choice(codes, p=probs))))
    con.executemany("INSERT INTO line VALUES (?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    return f"sqlite:{path}"


def test_drafted_ratio_needs_a_simpler_factor(tmp_path):
    import relsynth as rs
    spec = rs.introspect(_ratio_db(tmp_path / "r.db"), name="r")
    cols = spec.raw["tables"]["line"]["columns"]
    up, od = cols["unit_price"], cols["ordinal"]
    assert isinstance(up, dict) and up.get("ratio_of") == "item_id.price"   # three discounts: a real ratio
    assert not (isinstance(od, dict) and od.get("ratio_of"))                 # a code: no ratio of doses


def _header_detail_db(path, parents=600, seed=0):
    """A child with one header row per parent (fields set on that row only) and
    detail rows; and a true composite grain (order, line) on another child."""
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE parent (pid INTEGER PRIMARY KEY, kind TEXT)")
    con.execute("CREATE TABLE detail (pid INTEGER REFERENCES parent(pid), status TEXT, amount REAL)")
    con.execute("CREATE TABLE line (pid INTEGER REFERENCES parent(pid), slot TEXT, qty INTEGER)")
    con.executemany("INSERT INTO parent VALUES (?, ?)", [(i, str(rng.choice(["a", "b"]))) for i in range(parents)])
    det, lines = [], []
    for i in range(parents):
        det.append((i, str(rng.choice(["given", "held", "refused"])), None))         # the header row
        det += [(i, None, float(np.round(rng.gamma(2, 5), 1))) for _ in range(int(rng.integers(1, 6)))]
        for s in rng.choice(["x", "y", "z", "w"], size=int(rng.integers(2, 5)), replace=False):
            lines.append((i, str(s), int(rng.integers(1, 9))))
    con.executemany("INSERT INTO detail VALUES (?, ?, ?)", det)
    con.executemany("INSERT INTO line VALUES (?, ?, ?)", lines)
    con.commit()
    con.close()
    return f"sqlite:{path}"


def test_composite_key_must_be_minimal(tmp_path):
    import relsynth as rs
    spec = rs.introspect(_header_detail_db(tmp_path / "h.db"), name="h")
    tables = spec.raw["tables"]
    assert ["pid", "status"] not in (tables["detail"].get("unique") or [])     # pid alone is unique where status is set
    assert ["pid", "slot"] in (tables["line"].get("unique") or [])             # a real grain is kept
