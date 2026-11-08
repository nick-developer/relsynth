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
    assert DEFAULT_SYNTHESIS["privacy"]["near_copy_max_rows"] == 20_000


def test_nearest_exact_and_positive():
    B = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]])
    j, d = nearest(B, B, exclude_self=True)
    assert d.tolist() == [0.0, 0.0, 5.0]
    j, d = nearest(B, B, exclude_self=True, positive=True)
    assert d.tolist() == [5.0, 5.0, 5.0] and j.tolist() == [2, 2, 0]


def test_nearest_matches_brute_force_in_small_blocks():
    """The block search over distinct rows gives the brute-force distances, never
    picks the row itself under exclude_self, and stays exact with blocks of a few
    cells (memory bound)."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        B = rng.integers(0, 3, (rng.integers(1, 40), 3)).astype(float)
        A = rng.integers(0, 3, (rng.integers(1, 30), 3)).astype(float)
        for ex, pos, AA in [(False, False, A), (False, True, A), (True, False, B), (True, True, B)]:
            j, d = nearest(AA, B, exclude_self=ex, positive=pos, max_cells=7)
            D = np.sqrt(((AA[:, None] - B[None]) ** 2).sum(2))
            if ex:
                np.fill_diagonal(D, np.inf)
            if pos:
                D[D == 0] = np.inf
            np.testing.assert_allclose(d, D.min(1))
            fin = np.isfinite(d)
            np.testing.assert_allclose(np.sqrt(((AA[fin] - B[j[fin]]) ** 2).sum(1)), d[fin])
            if ex:
                assert (j[fin] != np.arange(len(AA))[fin]).all()


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
    """Shape-mode shrinkage (40 rows) is on by default, the guard is off, and
    shrinkage 0 gives other values on the same random stream."""
    m, base = _run(shop)
    assert m.spec["synthesis"]["marginals"]["segment_prior_rows"] == 40.0
    assert m.spec["synthesis"]["marginals"]["segment_prior_mode"] == "shape"
    assert m.spec["synthesis"]["privacy"]["near_copy"] == "off"
    assert "near_copies" not in m.synthesizer.sample_log
    _, zero = _run(shop, marginals__segment_prior_rows=0.0)
    b, z = dict(base), dict(zero)
    assert any(not b[t].equals(z[t]) for t in b)
    # the same random stream (Prop. 1720 b): same row counts in every table
    assert {t: len(v) for t, v in b.items()} == {t: len(v) for t, v in z.items()}


def _catalogue(d):
    """A product list in families (30 models x 10 variants sharing price, cost and line)."""
    rng = np.random.default_rng(12)
    rows = []
    for m in range(30):
        price = float(np.round(rng.lognormal(5, 1), 2))
        line = rng.choice(list("MRTS"))
        for v in range(10):
            rows.append({"id": len(rows) + 1, "price": price, "cost": round(price * 0.6, 2),
                         "weight": float(np.round(rng.normal(10 + m, 0.3), 2)),
                         "color": rng.choice(["red", "black", "silver", "blue"]), "size": str(38 + 2 * (v % 5)),
                         "line": line})
    (d / "src").mkdir()
    pd.DataFrame(rows).to_csv(d / "src" / "product.csv", index=False)
    cols = {"id": "id", "price": "float", "cost": "float", "weight": "float", "color": "categorical",
            "size": "categorical", "line": "categorical"}
    return {"version": 1, "name": "catalogue", "source": {"kind": "csv", "path": str(d / "src")},
            "target": {"kind": "csv", "path": str(d / "out")},
            "tables": {"product": {"role": "entity", "primary_key": "id",
                                   "columns": {c: {"type": t} for c, t in cols.items()}}},
            "relationships": [], "rules": []}


def test_guard_redraws_and_logs(tmp_path):
    spec = _catalogue(tmp_path)
    m, d = _run(spec, privacy__near_copy="auto", marginals__segment_prior_rows=0.0)
    log = m.synthesizer.sample_log.get("near_copies", {})
    assert set(log) == {"product"}
    r = log["product"]
    assert r["near_copies"] > 0 and r["near_copies"] == r["redrawn"] + r["kept"] and r["rows"] == 300
    # dtypes survive the redraw
    _, off = _run(spec, privacy__near_copy="off", marginals__segment_prior_rows=0.0)
    assert dict(d)["product"].dtypes.to_dict() == dict(off)["product"].dtypes.to_dict()


def test_constant_real_table_flags_every_other_row():
    """Every real row identical: no positive neighbour distance (nn = inf), so a
    synthetic row at any positive distance is a near copy and an exact copy is not
    (the real row has twins).  The engine never guards such a table (fewer than 3
    distinct columns), but the rule is defined."""
    real = pd.DataFrame({"a": [1.0] * 60, "b": ["x"] * 60, "c": [2.0] * 60, "d": ["p"] * 60})
    g = NearCopyGuard(real, TS, ["a", "b", "c", "d"])
    assert np.isinf(g.nn).all() and g.has_twin.all()
    syn = pd.DataFrame({"a": [1.0, 5.0], "b": ["x", "y"], "c": [2.0, 2.0], "d": ["p", "p"]})
    assert g.near_copies(syn).tolist() == [False, True]


def test_rule_redraws_are_guarded_and_kept_counts_emitted_rows(tmp_path):
    """Rows redrawn later for an attribute rule join the near-copy rejection, and
    the log's `kept` is the number of near copies among the rows emitted."""
    spec = _catalogue(tmp_path)
    spec["rules"] = [{"table": "product", "type": "check", "expr": "weight < 30"}]
    m, d = _run(spec, privacy__near_copy="auto", marginals__segment_prior_rows=0.0)
    syn = m.synthesizer
    r = syn.sample_log["near_copies"]["product"]
    assert r.get("redrawn_by_rules", 0) > 0
    g = syn._near_copy_guard("product")
    assert r["kept"] == int(g.near_copies(dict(d)["product"]).sum())


def test_near_copy_redraw_never_trades_a_valid_row_for_a_violation(tmp_path, monkeypatch):
    """With a short rule budget (2 rounds, near-copy budget 1), a row redrawn only
    for being a near copy and then drawn into a rule violation gets its rule-valid
    values back: the guard adds no rule violations."""
    import relsynth.nearcopy as nc
    spec = _catalogue(tmp_path)
    spec["rules"] = [{"table": "product", "type": "check", "expr": "weight < 30"}]
    # every redrawn row counts as a near copy, so rule-valid rows of round 1 are redrawn
    monkeypatch.setattr(nc.NearCopyGuard, "near_copies", lambda self, df: np.ones(len(df), bool))
    m, d = _run(spec, privacy__near_copy="auto", privacy__near_copy_rounds=1, marginals__segment_prior_rows=0.0,
                rules__max_rejection_rounds=2)
    log = m.synthesizer.sample_log
    assert log["near_copies"]["product"].get("restored_after_rules", 0) > 0
    # restored rows keep their hidden draws, so they pass the rules and keys again
    assert log["near_copies"]["product"]["restored_still_bad"] == 0
    out = dict(d)["product"]
    assert log["rule_violations"]["product"]["weight < 30"] == int((out["weight"] >= 30).sum())


def test_restored_rows_keep_their_hidden_draws(tmp_path, monkeypatch):
    """A rule on a column with its own distribution (a hidden draw): restored rows
    get back the draw they passed with, not a fresh one, so none fails again."""
    import relsynth.nearcopy as nc
    spec = _catalogue(tmp_path)
    src = tmp_path / "src" / "product.csv"
    df = pd.read_csv(src)
    df["promo"] = np.random.default_rng(0).uniform(0, 1, len(df)).round(3)
    df.to_csv(src, index=False)
    spec["tables"]["product"]["columns"]["promo"] = {"type": "float",
                                                     "distribution": {"kind": "uniform", "min": 0, "max": 1}}
    spec["rules"] = [{"table": "product", "type": "check", "expr": "promo < 0.5"}]
    monkeypatch.setattr(nc.NearCopyGuard, "near_copies", lambda self, df: np.ones(len(df), bool))
    m, d = _run(spec, privacy__near_copy="auto", privacy__near_copy_rounds=1, marginals__segment_prior_rows=0.0,
                rules__max_rejection_rounds=2)
    log = m.synthesizer.sample_log["near_copies"]["product"]
    assert log.get("restored_after_rules", 0) > 0
    assert log["restored_still_bad"] == 0


def test_kept_counts_emitted_rows_after_a_group_effect(tmp_path):
    """No rule redraw, but a group effect (one price per line) rewrites guarded
    columns after the guard: `kept` is the near-copy count of the emitted rows."""
    spec = _catalogue(tmp_path)
    spec["tables"]["product"]["group_effects"] = [{"column": "price", "by": "line"}]
    m, d = _run(spec, privacy__near_copy="auto", marginals__segment_prior_rows=0.0)
    out = dict(d)["product"]
    assert out.groupby("line")["price"].nunique().max() == 1          # the group effect applied
    g = m.synthesizer._near_copy_guard("product")
    log = m.synthesizer.sample_log.get("near_copies", {}).get("product", {"kept": 0})
    assert log["kept"] == int(g.near_copies(out).sum())


@pytest.mark.parametrize("nc_rounds,max_rounds,expect", [(1, 10, 1), (3, 4, 2), (5, 20, 5)])
def test_near_copy_rejection_respects_both_round_limits(tmp_path, monkeypatch, nc_rounds, max_rounds, expect):
    """Every row counts as a near copy, so the near-copy rejection would run in every
    rule round: it runs in min(near_copy_rounds, max_rejection_rounds // 2) rounds."""
    import relsynth.nearcopy as nc
    spec = _catalogue(tmp_path)
    spec["rules"] = [{"table": "product", "type": "check", "expr": "weight < 10.5"}]
    monkeypatch.setattr(nc.NearCopyGuard, "near_copies", lambda self, df: np.ones(len(df), bool))
    m, _ = _run(spec, privacy__near_copy="auto", privacy__near_copy_rounds=nc_rounds,
                marginals__segment_prior_rows=0.0, rules__max_rejection_rounds=max_rounds)
    assert m.synthesizer.sample_log["near_copies"]["product"]["near_copy_rule_rounds"] == expect


def test_group_effect_rows_join_the_near_copy_recheck(tmp_path):
    """Rule redraws change a group's shared price, so rows that were not redrawn
    change too; those rows are part of the near-copy recheck (changed_by_rules)."""
    spec = _catalogue(tmp_path)
    spec["tables"]["product"]["group_effects"] = [{"column": "price", "by": "line"}]
    spec["rules"] = [{"table": "product", "type": "check", "expr": "weight < 10.5"}]
    m, _ = _run(spec, privacy__near_copy="auto", marginals__segment_prior_rows=0.0)
    log = m.synthesizer.sample_log["near_copies"]["product"]
    assert log["changed_by_rules"] > log["redrawn_by_rules"] > 0


def test_guard_encoding_is_fitted_on_real_rows_only(tmp_path):
    """The guard ranks numbers and picks levels among the real rows; the scorecard's
    design ranks real and synthetic rows together, so the two encodings differ."""
    from relsynth.evaluate import _design
    from relsynth.nearcopy import NearCopyGuard
    tspec = {"columns": {"x": {"type": "float"}, "c": {"type": "categorical"}}}
    real = pd.DataFrame({"x": np.arange(10.0), "c": list("ab") * 5})
    syn = pd.DataFrame({"x": np.arange(100.0, 110.0), "c": ["z"] * 10})
    g = NearCopyGuard(real, tspec, ["x", "c"])
    eg = g.encode(syn)
    assert np.unique(eg, axis=0).shape[0] == 1          # all above every real value, level z unknown
    assert not eg[:, 3:].any()                            # z is not one of the real levels
    es = _design([real, syn], tspec, ["x", "c"])[10:]
    assert np.unique(es, axis=0).shape[0] == 10          # ranked among real and synthetic rows
    np.testing.assert_array_equal(g.encode(real), NearCopyGuard(real, tspec, ["x", "c"]).X)


def test_guard_skips_small_or_narrow_tables(tmp_path):
    """The guard is not built for a table of fewer than 50 rows, nor for one with
    fewer than three sampled distance columns (claim PRV-32)."""
    spec = _catalogue(tmp_path)
    m, _ = _run(spec, privacy__near_copy="auto", marginals__segment_prior_rows=0.0)
    assert m.synthesizer._near_copy_guard("product") is not None
    # two sampled distance columns only
    narrow = copy.deepcopy(spec)
    cols = narrow["tables"]["product"]["columns"]
    for c in ("cost", "weight", "size", "line"):
        cols.pop(c)
    m2, _ = _run(narrow, privacy__near_copy="auto", marginals__segment_prior_rows=0.0)
    assert m2.synthesizer._near_copy_guard("product") is None
    # fewer than 50 rows
    src = tmp_path / "small"
    (src / "src").mkdir(parents=True)
    pd.read_csv(tmp_path / "src" / "product.csv").head(40).to_csv(src / "src" / "product.csv", index=False)
    small = copy.deepcopy(spec)
    small["source"]["path"] = str(src / "src")
    m3, _ = _run(small, privacy__near_copy="auto", marginals__segment_prior_rows=0.0)
    assert m3.synthesizer._near_copy_guard("product") is None
