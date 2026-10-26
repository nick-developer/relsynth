# SPDX-License-Identifier: Apache-2.0
"""Evaluator (relsynth.evaluate): metric primitives against closed forms, and the
scorecard on data whose answer is known (identical, or broken in one way)."""
from __future__ import annotations

import copy
import importlib
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from relsynth import load_spec

E = importlib.import_module("relsynth.evaluate")       # relsynth.evaluate is also a function name


def test_canonical_spelling_of_values():
    assert E._canon(10000) == E._canon(10000.0) == E._canon(Decimal("10000.00")) == "10000"
    assert E._canon(0.1) == repr(0.1)
    assert E._canon(None) == E._canon(float("nan")) == E._canon(pd.NA) == E._canon(pd.NaT) == "<null>"
    assert E._canon(True) == "True" and E._canon(np.bool_(False)) == "False"
    assert E._canon("x") == "x" and E._canon(float("inf")) == "inf"
    assert E._canon(2.0 ** 60) == repr(round(2.0 ** 60, 12))


def test_canon_counts_equals_the_per_row_definition():
    vals = pd.Series([1, 1.0, Decimal("1"), "a", None, np.nan, 2, "a", "b", 2.5, 2.5, 2.5], dtype=object)
    ref = vals.map(E._canon).value_counts(normalize=True)
    pd.testing.assert_series_equal(E._canon_counts(vals), ref, check_names=False, check_index_type=False)
    mixed = pd.Series([True, 1, "1", False, 0], dtype=object)       # bool and int hash alike
    pd.testing.assert_series_equal(E._canon_counts(mixed), mixed.map(E._canon).value_counts(normalize=True),
                                   check_names=False)
    assert len(E._canon_counts(pd.Series([], dtype=object))) == 0


def test_tvd_and_ks():
    a = pd.Series(["x"] * 6 + ["y"] * 4)
    b = pd.Series(["x"] * 3 + ["z"] * 7)
    assert E.tvd(a, b) == pytest.approx(0.5 * (abs(0.6 - 0.3) + 0.4 + 0.7))
    assert E.tvd(a, a) == 0.0
    x = np.r_[np.linspace(0, 1, 50), np.nan]
    y = np.linspace(0.5, 1.5, 40)
    assert E.ks(x, y) == pytest.approx(stats.ks_2samp(x[:-1], y).statistic)
    assert np.isnan(E.ks(np.array([1.0]), y))


def test_null_limits():
    assert E._eff(99, 50) == pytest.approx(1 / 100 + 1 / 50)
    assert E._eff(0, 0) == pytest.approx(1.5)
    # Sidak: the largest of k |N(0,1)| exceeds z with probability alpha
    z = E._sidak_z(0.01, 20)
    assert 1 - (1 - 2 * stats.norm.sf(z)) ** 20 == pytest.approx(0.01)
    z1 = E._sidak_z(0.05, 1, two_sided=False)
    assert z1 == pytest.approx(stats.norm.ppf(0.95))
    assert E._sidak_z(0.05, 0) == E._sidak_z(0.05, 1)
    # KS: c(alpha) sqrt(v), conservative and calibrated factors
    assert E.ks_crit(100, 200, alpha=0.05) == pytest.approx(np.sqrt(-np.log(0.025) / 2) * np.sqrt(2 / 100 + 1 / 200))
    assert E.ks_crit(100, 200, alpha=0.05, pp=True) == pytest.approx(np.sqrt(-np.log(0.025) / 2) * np.sqrt(E._eff(100, 200)))
    assert E._ks_alpha(E.ks_crit(100, 200, alpha=0.05, pp=True), 100, 200) == pytest.approx(0.05)
    assert E._ks_alpha(None, 1, 1) is None and E._ks_alpha(float("inf"), 1, 1) is None
    p = np.array([0.5, 0.3, 0.2])
    v = 2 / 50 + 1 / 80
    assert E.tvd_crit(p, 50, 80) == pytest.approx(2.5 * 0.5 * np.sum(np.sqrt(2 * p * (1 - p) / np.pi * v)))
    mu, sd = E.tvd_null(p, 50, 80)
    s = np.sqrt(p * (1 - p) * E._eff(50, 80))
    assert mu == pytest.approx(0.5 * np.sqrt(2 / np.pi) * s.sum())
    assert sd == pytest.approx(0.5 * np.sqrt(((1 - 2 / np.pi) * s * s).sum()))
    assert E._z_alpha(2.0, 1.0) == pytest.approx(2 * stats.norm.sf(2.0))
    assert E._z_alpha(2.0, 1.0, two_sided=False) == pytest.approx(stats.norm.sf(2.0))
    assert E._z_alpha(2.0, 0.0) is None and E._z_alpha(None, 1.0) is None


def test_auc():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert E._auc(np.array([1, 2, 3, 4, 5, 6.0]), y) == 1.0
    assert E._auc(np.array([6, 5, 4, 3, 2, 1.0]), y) == 0.0
    assert E._auc(np.array([1, 4, 2, 3, 5, 6.0]), y) == pytest.approx(stats.mannwhitneyu([3, 5, 6], [1, 4, 2]).statistic / 9)
    assert E.auc_null_sd(59) == pytest.approx(np.sqrt(60 / (3 * 59 ** 2)))
    assert E.auc_null_sd(1) == E.auc_null_sd(2)


def test_cardinality_and_fk_integrity():
    parent = pd.DataFrame({"id": [1, 2, 3, 4]})
    real = pd.DataFrame({"p": [1, 1, 2, 2, 2, 3]})
    syn = pd.DataFrame({"p": [1, 2, 2, 3, 3, 4, 4, 4]})
    c = E.cardinality(real, syn, "p", parent, parent, "id")
    assert c["mean_real"] == 1.5 and c["mean_syn"] == 2.0
    assert c["zero_share_real"] == 0.25 and c["zero_share_syn"] == 0.0
    assert c["mean_rel_err"] == pytest.approx(2.0 / 1.5 - 1)
    assert c["ks_shape"] == pytest.approx(stats.ks_2samp([2, 3, 1, 0], [1, 2, 2, 3]).statistic)
    assert c["dispersion_real"] == pytest.approx(np.var([2, 3, 1, 0]) / 1.5)
    # exact counts from statistics replace the (sampled) real child
    rc = pd.DataFrame({"p": [1, 2, 3, None], "n": [2, 3, 1, 5]})
    assert E.cardinality(real.iloc[:1], syn, "p", parent, parent, "id", real_counts=rc)["mean_real"] == 1.5
    assert E.fk_integrity(syn, "p", parent, "id") == 1.0
    assert E.fk_integrity(pd.DataFrame({"p": [1, 9, None, 2.0]}), "p", parent, "id") == pytest.approx(2 / 3)
    assert E.fk_integrity(pd.DataFrame({"p": [None]}), "p", parent, "id") == 1.0


def test_ks_shape_thins_the_larger_side():
    rng = np.random.default_rng(0)
    rc = rng.poisson(4.0, 3000).astype(float)
    sc = rng.poisson(8.0, 3000).astype(float)
    assert E._ks_shape(rc, sc, 2.0) < 0.05 < E._ks_shape(rc, sc, 1.0)
    assert E._ks_shape(sc, rc, 0.5) < 0.05


@pytest.fixture(scope="module")
def star():
    from conftest import scenario
    t, s, _ = scenario("S2_star")
    return load_spec(s), t


def _failed(spec, real, syn, privacy=False):
    sc = E.scorecard(E.evaluate(spec, real, syn, privacy=privacy))
    return {(i["area"], i["metric"]) for i in sc["items"] if not i["pass"]}, sc


def test_scorecard_passes_identical_data_and_flags_copies(star):
    spec, real = star
    failed, sc = _failed(spec, real, copy.deepcopy(real))
    assert failed == set() and sc["passed"] == sc["total"] >= 30
    failed, _ = _failed(spec, real, copy.deepcopy(real), privacy=True)
    assert ("customer", "exact_match_rate") in failed and ("customer", "dcr_ratio") in failed


def test_scorecard_flags_each_kind_of_damage(star):
    spec, real = star
    rng = np.random.default_rng(0)
    syn = copy.deepcopy(real)
    syn["sales"]["amount"] = syn["sales"]["amount"] * 1.5                                 # a shifted column
    assert ("sales.amount", "marginal_ks") in _failed(spec, real, syn)[0]
    syn = copy.deepcopy(real)
    syn["customer"]["region"] = rng.choice(["A", "B"], len(syn["customer"]))              # a wrong marginal
    assert ("customer.region", "marginal_tvd") in _failed(spec, real, syn)[0]
    syn = copy.deepcopy(real)
    few = syn["customer"]["customer_id"].to_numpy()[:200]
    syn["sales"]["customer_id"] = rng.choice(few, len(syn["sales"]))                      # wrong counts per parent
    assert ("sales.customer_id->customer", "cardinality_ks") in _failed(spec, real, syn)[0]
    syn = copy.deepcopy(real)
    syn["sales"].loc[syn["sales"].index[:500], "product_id"] = -1                        # dangling keys
    assert any(m == "fk_integrity" for _, m in _failed(spec, real, syn)[0])
    syn = copy.deepcopy(real)
    for c in ("income", "age"):                                                          # broken dependence
        syn["customer"][c] = rng.permutation(syn["customer"][c].to_numpy())
    assert ("customer", "association_max_diff") in _failed(spec, real, syn)[0]
