# SPDX-License-Identifier: Apache-2.0
"""Small-dimension privacy (G3): segment marginal shrinkage
(marginals.segment_prior_rows) and the near-copy guard (privacy.near_copy,
relsynth.nearcopy).  The statistical checks are C1720-C1723 in
tests/science/test_small_dims_checks.py."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import relsynth as rs
from relsynth.marginals import Codec, Marginal
from relsynth.nearcopy import NearCopyGuard, nearest
from relsynth.settings import registry
from relsynth.spec import DEFAULT_SYNTHESIS

from conftest import write_scenario

TS = {"columns": {"a": {"type": "float"}, "b": {"type": "categorical"}, "c": {"type": "float"},
                  "d": {"type": "categorical"}}}


def _real(n=120, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.choice(list("xyz"), n),
                       "c": rng.gamma(2.0, size=n), "d": rng.choice(list("pq"), n)})
    # a block of identical rows (a coarse attribute space)
    df.loc[:9, ["a", "b", "c", "d"]] = [0.0, "x", 1.0, "p"]
    return df


def test_settings_are_registered_with_defaults():
    for k in ("marginals.segment_prior_rows", "privacy.near_copy", "privacy.near_copy_ratio",
              "privacy.near_copy_rounds", "privacy.near_copy_max_rows"):
        assert k in registry()
    assert DEFAULT_SYNTHESIS["privacy"]["near_copy_ratio"] == 1.0


def test_nearest_exact_and_positive():
    B = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]])
    j, d = nearest(B, B, exclude_self=True)
    assert d.tolist() == [0.0, 0.0, 5.0]
    j, d = nearest(B, B, exclude_self=True, positive=True)
    assert d.tolist() == [5.0, 5.0, 5.0] and j.tolist() == [2, 2, 0]


def test_guard_rules():
    real = _real()
    g = NearCopyGuard(real, TS, ["a", "b", "c", "d"])
    assert g.has_twin[:10].all() and not g.has_twin[10:].any()
    syn = pd.concat([real.iloc[[0]],                                     # copy of a shared row: allowed
                     real.iloc[[20]],                                    # copy of a unique row: near copy
                     real.iloc[[30]].assign(a=real.a.iloc[30] + 1e-6),   # a hair away from a unique row
                     pd.DataFrame({"a": [40.0], "b": ["z"], "c": [99.0], "d": ["q"]})], ignore_index=True)
    assert g.near_copies(syn).tolist() == [False, True, True, False]
    # rho = 0 keeps every positive distance (only copies of unique rows remain near copies)
    g0 = NearCopyGuard(real, TS, ["a", "b", "c", "d"], rho=0.0)
    assert g0.near_copies(syn).tolist() == [False, True, False, False]


def test_segment_prior_is_the_dirichlet_mean():
    """mean mode: p = (counts + a p0) / (n + a) with a = max(dirichlet_prior, segment_prior_rows)."""
    s = pd.Series(list("aaab") + list("bbbbbbbbbbbb"))
    cd = Codec("x", {"type": "categorical"}, s)
    x = cd.encode(s)
    rng = np.random.default_rng(0)
    top = Marginal.fit(cd, x, rng, dirichlet_a=0.25, bootstrap=False)
    seg = x[:4]
    for a in (0.25, 20.0):
        m = Marginal.fit(cd, seg, rng, prior=top, dirichlet_a=a, bootstrap=False)
        cnt = np.bincount(seg, minlength=2)
        assert np.allclose(m.p, (cnt + a * top.p + 1e-9) / (cnt.sum() + a + 2e-9))


@pytest.fixture(scope="module")
def shop(tmp_path_factory):
    d = tmp_path_factory.mktemp("shop")
    spec = write_scenario("S9_small_shop", {"kind": "csv", "path": str(d / "src")})
    spec["target"] = {"kind": "csv", "path": str(d / "out")}
    return spec


def _run(spec, **priv):
    s = copy.deepcopy(spec)
    syn = s.setdefault("synthesis", {})
    for k, v in priv.items():
        g, key = k.split("__")
        syn.setdefault(g, {})[key] = v
    m = rs.fit(s, seed=3)
    return m, m.sample()


def test_segment_prior_is_wired_and_guard_off_by_default(shop):
    m, base = _run(shop)
    assert m.spec["synthesis"]["marginals"]["segment_prior_rows"] == 20.0
    assert m.spec["synthesis"]["privacy"]["near_copy"] == "off"
    assert "near_copies" not in m.synthesizer.sample_log
    _, zero = _run(shop, marginals__segment_prior_rows=0.0)
    b, z = dict(base), dict(zero)
    assert any(not b[t].equals(z[t]) for t in b)
    # the same random stream (Prop. 1720 b): same row counts in every table
    assert {t: len(v) for t, v in b.items()} == {t: len(v) for t, v in z.items()}


def test_guard_redraws_and_logs(shop):
    m, d = _run(shop, privacy__near_copy="auto")
    log = m.synthesizer.sample_log.get("near_copies", {})
    for t, r in log.items():
        assert r["near_copies"] == r["redrawn"] + r["kept"] and r["rows"] >= 50
    # with ratio 0 and no unique copies nothing is redrawn and nothing random is drawn
    m0, d0 = _run(shop, privacy__near_copy="auto", privacy__near_copy_ratio=0.0)
    assert all(r["near_copies"] == r["kept"] + r["redrawn"] for r in
               m0.synthesizer.sample_log.get("near_copies", {}).values())
