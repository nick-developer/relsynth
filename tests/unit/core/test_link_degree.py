# SPDX-License-Identifier: Apache-2.0
"""Degree law of the keys of link tables (synthesis.facts.link_degree).

The pooled NB size against simulated data, the Dirichlet-multinomial identity
the NB law samples by, the degree-preserving repair of repeated pairs and
self-links, and a small link table fitted end to end: with `auto` the second
key's degrees follow the source, with the default `segment` nothing changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import relsynth as rs
from relsynth import counts as C
from relsynth.engine import Synthesizer
from relsynth.spec import SpecError


def test_nb_size_pooled_recovers_the_shared_size():
    rng = np.random.default_rng(0)
    r = 0.8
    groups = [rng.negative_binomial(r, r / (r + m), size=4000) for m in (1.5, 6.0, 20.0)]
    assert C.nb_size_pooled(groups) == pytest.approx(r, rel=0.08)
    # Poisson counts: a size far beyond the data's (inf when not over-dispersed at all)
    assert C.nb_size_pooled([rng.poisson(3.0, size=4000), rng.poisson(8.0, size=4000)]) > 100
    assert C.nb_size_pooled([np.array([3, 3, 3, 4]), np.array([1, 1, 2])]) == np.inf
    assert C.nb_size_pooled([np.array([2])]) == np.inf


def test_nb_given_its_total_is_dirichlet_multinomial():
    """i.i.d. NB(r, p) counts conditioned on their sum R are Dirichlet-
    multinomial(R; r, ..., r), whatever p: the law _link_degrees samples."""
    rng = np.random.default_rng(1)
    r, M, R = 0.7, 4, 6
    x = rng.negative_binomial(r, 0.3, size=(400_000, M))
    first = x[x.sum(1) == R][:, 0]
    w = rng.gamma(r, size=(len(first), M))
    dm = np.array([rng.multinomial(R, wi / wi.sum())[0] for wi in w[:20_000]])
    exact = stats.betabinom.pmf(np.arange(R + 1), R, r, (M - 1) * r)
    for sample in (first, dm):
        emp = np.bincount(sample, minlength=R + 1) / len(sample)
        assert np.abs(emp - exact).max() < 0.012


def _synth():
    return Synthesizer.__new__(Synthesizer)


def test_repair_pairs_keeps_degrees_and_blocks():
    s = _synth()
    s.rng = np.random.default_rng(2)
    rng = np.random.default_rng(3)
    n, K1, K2 = 300, 2, 3
    z1, z2 = rng.integers(0, K1, size=n), rng.integers(0, K2, size=n)
    u = rng.integers(0, n, size=2000)
    v = rng.integers(0, n, size=2000)
    u[:50], v[:50] = 7, 7                                  # repeated self-links
    blk = z1[u] * K2 + z2[v]
    u2, v2, b2, moved, dropped = s._repair_pairs(u, v, blk, n, K2, unique=True, no_loops=True)
    assert moved == dropped == 0 and len(u2) == len(u)
    assert not (u2 == v2).any()
    assert not pd.Series(u2 * n + v2).duplicated().any()
    assert (b2 == blk).all() and (z1[u2] * K2 + z2[v2] == b2).all()     # blocks match the segments
    assert (np.bincount(u2, minlength=n) == np.bincount(u, minlength=n)).all()
    assert (np.bincount(v2, minlength=n) == np.bincount(v, minlength=n)).all()


def test_repair_pairs_moves_or_drops_what_a_dense_block_cannot_hold():
    s = _synth()
    s.rng = np.random.default_rng(4)
    # row 0 links to partner 0 three times; partners 0 and 1 exist in the block
    u = np.array([0, 0, 0, 1])
    v = np.array([0, 0, 1, 1])
    u2, v2, _, moved, dropped = s._repair_pairs(u, v, np.zeros(4, np.int64), 2, 1, unique=True, no_loops=False)
    assert dropped == 1 and len(u2) == 3                  # row 0 can hold only partners 0 and 1
    assert not pd.Series(u2 * 2 + v2).duplicated().any()
    # a free partner exists but no swap can reach it: the row moves to it
    u = np.array([0, 0, 1, 1])
    v = np.array([0, 0, 0, 2])
    u2, v2, _, moved, dropped = s._repair_pairs(u, v, np.zeros(4, np.int64), 3, 1, unique=True, no_loops=False)
    assert (moved, dropped) == (1, 0)
    assert sorted(zip(u2.tolist(), v2.tolist())) == [(0, 0), (0, 2), (1, 0), (1, 2)]


def test_cap_degrees_keeps_the_total():
    s = _synth()
    s.rng = np.random.default_rng(5)
    d = np.array([50, 3, 0, 1, 6])
    c = s._cap_degrees(d, 15)
    assert c.sum() == d.sum() and c.max() <= 15 and c[0] == 15
    assert (s._cap_degrees(d, 11) == d).all()             # 60 links cannot fit under 11 x 5


def _link_db(seed=0):
    """Molecules, atoms and bonds: every atom is the second atom of exactly one
    bond (or none), while the first atom's degree varies; one type column."""
    rng = np.random.default_rng(seed)
    n_mol, per = 60, 25
    atom = pd.DataFrame({"atom_id": [f"a{i}" for i in range(n_mol * per)],
                         "molecule_id": np.repeat([f"m{i}" for i in range(n_mol)], per),
                         "element": rng.choice(["c", "h", "o"], size=n_mol * per, p=[0.5, 0.4, 0.1])})
    mol = pd.DataFrame({"molecule_id": [f"m{i}" for i in range(n_mol)], "activity": rng.normal(size=n_mol)})
    rows = []
    for m in range(n_mol):
        ids = np.arange(m * per, (m + 1) * per)
        second = rng.permutation(ids)[: per - 2]            # 23 of 25 atoms are a second atom once
        first = rng.choice(ids, size=len(second))
        rows += [(f"a{a}", f"a{b}", int(rng.integers(1, 3))) for a, b in zip(first, second) if a != b]
    bond = pd.DataFrame(rows, columns=["atom_id", "atom_id2", "type"])
    raw = {"name": "bonds", "tables": {
        "molecule": {"columns": {"molecule_id": "id", "activity": "float"}, "primary_key": "molecule_id"},
        "atom": {"columns": {"atom_id": "id", "molecule_id": "id", "element": "categorical"}, "primary_key": "atom_id"},
        "bond": {"columns": {"atom_id": "id", "atom_id2": "id", "type": "categorical"}, "role": "fact"}},
        "relationships": [{"child": "atom", "child_key": "molecule_id", "parent": "molecule"},
                          {"child": "bond", "child_key": "atom_id", "parent": "atom"},
                          {"child": "bond", "child_key": "atom_id2", "parent": "atom"}]}
    return raw, {"molecule": mol, "atom": atom, "bond": bond}


def _second_degrees(tables):
    cnt = tables["bond"]["atom_id2"].value_counts()
    return tables["atom"]["atom_id"].map(cnt).fillna(0).to_numpy()


@pytest.mark.parametrize("setting", ["segment", "auto"])
def test_link_table_second_key_degrees(setting):
    from relsynth.data import coerce
    raw, frames = _link_db()
    spec = rs.Spec(raw).resolved
    spec["synthesis"]["seed"] = 3
    spec["synthesis"]["facts"]["link_degree"] = setting
    real = {t: coerce(df, spec["tables"][t]) for t, df in frames.items()}
    s = Synthesizer(spec).fit(real)
    syn = s.sample(scale=spec["synthesis"]["scale"], index=0)
    ks = stats.ks_2samp(_second_degrees(real), _second_degrees(syn)).statistic
    dec = s.log["decisions"]["bond"]
    if setting == "segment":
        assert s.fact_info["bond"]["link_degree"] == {}
        assert not any("degree_model" in k for k in dec)
        assert ks > 0.2                                     # uniform within segments: Poisson-like degrees
    else:
        assert dec["atom_id2 degree_model"] == "dp"
        assert ks < 0.05
        b = syn["bond"]
        assert not (b["atom_id"] == b["atom_id2"]).any()   # the source has no self-bonds


def test_link_degree_spec_values():
    raw, _ = _link_db()
    raw["tables"]["bond"]["fact"] = {"link_degree": "sometimes"}
    with pytest.raises(SpecError, match="link_degree"):
        rs.Spec(raw).resolved


def test_complete_grid_stays_complete():
    """Every product on every date (AdventureWorks inventory): complete blocks are
    structural, so the degree law leaves them as the occupancy model draws them."""
    from relsynth.data import coerce
    prods = pd.DataFrame({"product_id": [f"p{i}" for i in range(40)],
                          "colour": np.random.default_rng(0).choice(["red", "blue"], size=40)})
    dates = pd.DataFrame({"date_id": [f"d{i}" for i in range(30)]})
    inv = pd.DataFrame([(p, d) for p in prods["product_id"] for d in dates["date_id"]],
                       columns=["product_id", "date_id"])
    inv["units"] = np.random.default_rng(1).poisson(5, size=len(inv))
    raw = {"name": "grid", "tables": {
        "product": {"columns": {"product_id": "id", "colour": "categorical"}, "primary_key": "product_id"},
        "date": {"columns": {"date_id": "id"}, "primary_key": "date_id"},
        "inventory": {"columns": {"product_id": "id", "date_id": "id", "units": "integer"}, "role": "fact",
                      "unique": [["product_id", "date_id"]]}},
        "relationships": [{"child": "inventory", "child_key": "product_id", "parent": "product"},
                          {"child": "inventory", "child_key": "date_id", "parent": "date"}]}
    out = {}
    for setting in ("segment", "auto"):
        spec = rs.Spec(raw).resolved
        spec["synthesis"]["facts"]["link_degree"] = setting
        real = {t: coerce(df, spec["tables"][t]) for t, df in {"product": prods, "date": dates, "inventory": inv}.items()}
        syn = Synthesizer(spec).fit(real).sample(scale=spec["synthesis"]["scale"], index=0)
        g = syn["inventory"]
        assert len(g) == len(syn["product"]) * len(syn["date"])
        assert not g.duplicated(["product_id", "date_id"]).any()
        out[setting] = g
    pd.testing.assert_frame_equal(out["segment"], out["auto"])          # complete blocks untouched
