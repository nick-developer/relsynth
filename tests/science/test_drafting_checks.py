# SPDX-License-Identifier: Apache-2.0
"""Checks C1840-C1844: reproducible drafting (docs/proofs/sec_v72_drafting.tex),
tested on the shipped code.

    pytest -m science tests/science/test_drafting_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

pytestmark = [pytest.mark.science]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unit.test_introspect_stability import (  # noqa: E402
    _header_detail_db,
    _keyless_db,
    _optional_links_db,
    _ratio_db,
)

import relsynth as rs  # noqa: E402
from relsynth.discovery import _entropy  # noqa: E402
from relsynth.pipeline import parse_source  # noqa: E402
from relsynth.source import Source  # noqa: E402


def test_c1840_keyless_sample_is_seeded_uniform_and_order_free(tmp_path):
    """Prop. 1840: (a) the sample is a function of the rows and the seed only;
    (b) on distinct rows every row has inclusion chance n / N."""
    N, n, seeds = 300, 60, 400
    df = pd.DataFrame({"a": np.arange(N), "b": np.arange(N) / 7.0, "c": ["k"] * N})
    p1 = _keyless_db(tmp_path / "a.db", df)
    p2 = _keyless_db(tmp_path / "b.db", df, shuffle_seed=11)
    hits = np.zeros(N)
    for s in range(seeds):
        a, _ = Source(parse_source(p1), seed=s).sample("ev", n)
        if s < 5:   # (a): another physical order, the same rows
            b, _ = Source(parse_source(p2), seed=s).sample("ev", n)
            assert sorted(a["a"]) == sorted(b["a"])
        hits[a["a"].to_numpy()] += 1
    # (b): each row's count is Binomial(seeds, n/N); chi-square over the N rows
    exp = seeds * n / N
    chi2 = float(((hits - exp) ** 2 / (exp * (1 - n / N))).sum())
    assert stats.chi2.sf(chi2, N - 1) > 1e-3
    # copies of one content: their share of the sample stays at their share of the table
    dup = pd.concat([df, pd.DataFrame({"a": [-1] * N, "b": [0.0] * N, "c": ["d"] * N})], ignore_index=True)
    p3 = _keyless_db(tmp_path / "c.db", dup)
    share = [float((Source(parse_source(p3), seed=s).sample("ev", n)[0]["c"] == "d").mean()) for s in range(100)]
    assert abs(np.mean(share) - 0.5) < 3 * 0.5 / np.sqrt(n * 100)


def test_c1841_ratio_factor_must_be_simpler(tmp_path):
    """Prop. 1841: (a) plug-in entropies obey H(y) = H(x, f) > H(f) for an injective
    y = x f with x varying within a factor level; (b) a code column independent of x
    gives a factor with H(f) >= H(y); drafts keep true ratios and drop false ones."""
    rng = np.random.default_rng(0)
    x = rng.choice(np.round(rng.gamma(3, 40, 200) + 5, 2), 5000)
    f = rng.choice([1.0, 0.9, 0.8], 5000, p=[0.7, 0.2, 0.1])
    y = np.round(x * f, 6)
    joint = np.array([f"{a}|{b}" for a, b in zip(x, f)])
    assert _entropy(y) <= _entropy(joint) + 1e-12 and _entropy(f) < _entropy(y)
    pairs = pd.DataFrame({"y": y, "j": joint}).drop_duplicates()
    inj = ~np.isin(y, pairs.loc[pairs["y"].duplicated(), "y"])     # rows whose y has one (x, f)
    assert inj.mean() > 0.9
    assert _entropy(y[inj]) == pytest.approx(_entropy(joint[inj]))
    code = rng.choice([1.1, 1.2, 1.3, 2.1], 5000, p=[0.7, 0.1, 0.1, 0.1])
    fhat = np.round(code / x, 6)
    assert _entropy(fhat) >= _entropy(code)
    for seed in range(3):
        cols = rs.introspect(_ratio_db(tmp_path / f"r{seed}.db", seed=seed), name="r").raw["tables"]["line"]["columns"]
        up, od = cols["unit_price"], cols["ordinal"]
        assert isinstance(up, dict) and up.get("ratio_of") == "item_id.price"
        assert not (isinstance(od, dict) and od.get("ratio_of"))


def test_c1842_drafted_composite_keys_are_minimal(tmp_path):
    """Prop. 1842: no drafted composite key has a column that is unique on its own
    where both are set; true grains are kept."""
    for seed in range(3):
        tables = rs.introspect(_header_detail_db(tmp_path / f"h{seed}.db", seed=seed), name="h").raw["tables"]
        con = Source(parse_source(f"sqlite:{tmp_path / f'h{seed}.db'}"))
        for t, spec in tables.items():
            for key in spec.get("unique") or []:
                if len(key) != 2:
                    continue
                whole = con.columns(t, key).dropna()
                assert not (whole[key[0]].is_unique or whole[key[1]].is_unique), (t, key)
        assert ["pid", "slot"] in (tables["line"].get("unique") or [])


def test_c1843_rows_without_a_partner(tmp_path):
    """Prop. 1843: rows with a NULL partner key are modelled (share, rows per anchor,
    attributes and extra link given no partner), in memory and with push-down;
    facts.null_partner: drop keeps the 2.2 model (no such rows)."""
    import sqlite3

    from unit.test_null_partner import _header_detail_db, _shares
    for seed in range(3):
        src = _header_detail_db(tmp_path / f"n{seed}.db", seed=seed)
        real = _shares(pd.read_sql("SELECT * FROM detail", sqlite3.connect(tmp_path / f"n{seed}.db")))
        base = rs.introspect(src, name="n")
        for pushdown in ("off", "on"):
            spec = base.set("synthesis.pushdown.mode", pushdown).set("synthesis.pushdown.sample_rows", 500)
            s = _shares(rs.fit(spec).sample()["detail"])
            assert abs(s["null_partner"] - real["null_partner"]) < 0.03, (seed, pushdown, s)
            assert abs(s["header_per_anchor"] - 1.0) < 0.02
            assert s["status_with_partner"] + s["dose_without_partner"] <= 0.005 * s["rows"]
            assert s["junior_on_header"] < 0.05
        s = _shares(rs.fit(base.set("synthesis.facts.null_partner", "drop")).sample()["detail"])
        assert s["null_partner"] == 0


def test_c1844_a_fact_needs_an_anchor_that_is_set(tmp_path):
    """Prop. 1844: a keyed table whose links are all mostly NULL is drafted as an entity
    (every row kept); a fact anchors on a link that is set on nearly every row."""
    import sqlite3
    for seed in range(3):
        src = _optional_links_db(tmp_path / f"o{seed}.db", seed=seed)
        spec = rs.introspect(src, name="o")
        T = spec.raw["tables"]
        assert T["Tags"]["role"] == "entity"
        assert T["Votes"]["fact"]["anchor"] == "Posts"
        syn = rs.fit(spec).sample()
        real = pd.read_sql("SELECT * FROM Tags", sqlite3.connect(tmp_path / f"o{seed}.db"))
        assert len(syn["Tags"]) == len(real)
        assert abs(syn["Tags"]["WikiPostId"].isna().mean() - real["WikiPostId"].isna().mean()) < 0.1
