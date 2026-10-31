# SPDX-License-Identifier: Apache-2.0
"""Checks C900-C906: the differential privacy propositions of docs/proofs
(section v2.5, DP engine), tested on the shipped code (`relsynth.dp`).

C902 is the central one: for neighbouring databases (one customer and all
their sales removed) every statistic the engine releases moves by at most the
L2 sensitivity it is charged for.  The engine is adaptive (later statistics
use earlier noisy ones), so the check replays the noisy outputs of the run on
the full database into the run on its neighbour and compares the exact
statistics each mechanism was asked to perturb.  About a minute:

    pytest -m science tests/science/test_dp_checks.py
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

pytestmark = [pytest.mark.science]
opendp = pytest.importorskip("opendp")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fakedata"))
import scenarios as S  # noqa: E402

import relsynth.dp as dp  # noqa: E402
from relsynth.dp import mechanisms as M  # noqa: E402
from relsynth.dp import relational as R  # noqa: E402
from relsynth.errors import PrivacyError  # noqa: E402
from relsynth.spec import load_spec  # noqa: E402


def _dgauss_logpmf(sigma, lo, hi):
    k = np.arange(lo, hi + 1)
    lp = -k.astype(float) ** 2 / (2 * sigma ** 2)
    return k, lp - logsumexp(lp)


# --------------------------------------------------------------------- C900
def test_c900_discrete_gaussian_is_rho_zcdp_and_charged_as_such():
    """Prop. 900: discrete Gaussian noise of scale sigma on an integer query of
    L2 sensitivity D is D^2/(2 sigma^2)-zCDP.  The Renyi divergence of every
    order alpha between the shifted and unshifted distributions, computed on
    the support, is at most alpha * rho, and the accountant charges rho."""
    for D, rho in ((1, 0.5), (3, 0.05), (10, 0.01)):
        sigma = M.sigma_for(D, rho)
        lim = int(40 * sigma) + 10 * D
        k, lp = _dgauss_logpmf(sigma, -lim, lim)
        lq = np.interp(k - D, k, lp, left=-np.inf, right=-np.inf)        # pmf of noise + D
        ok = np.isfinite(lq)
        for a in (1.5, 2, 4, 8, 16):
            ren = logsumexp(a * lq[ok] + (1 - a) * lp[ok]) / (a - 1)
            assert ren <= a * rho * (1 + 1e-6) + 1e-12, (D, rho, a, ren, a * rho)
        acct = M.Accountant(rho)
        out = M.gaussian(np.zeros(2000, dtype=int), D, rho, acct, "fit", "test")
        assert out.dtype == np.int64
        assert math.isclose(acct.spent[0], rho, rel_tol=1e-6)
        # the noise has variance at most sigma^2 (Canonne, Kamath and Steinke 2020, Prop. 19)
        assert np.var(out) < sigma ** 2 * 1.15


def test_c900_accountant_refuses_beyond_the_plan():
    acct = M.Accountant(0.1)
    M.gaussian([1, 2], 1, 0.06, acct, "fit", "a")
    with pytest.raises(PrivacyError):
        M.gaussian([1, 2], 1, 0.05, acct, "fit", "b")
    assert len(acct.charges) == 1


# --------------------------------------------------------------------- C903
def test_c903_partition_selection_threshold_bounds_delta_exactly():
    """Prop. 903: a level held only by the removed unit has true count at most
    c; it is released when count + noise > tau.  Summed exactly over the
    discrete Gaussian, the probability is at most delta / k."""
    for c, rho, delta, k in ((1, 0.01, 1e-9, 1), (5, 0.02, 1e-10, 5), (40, 0.1, 5e-10, 40)):
        sigma = M.sigma_for(c, rho)
        tau = M.threshold_for(c, sigma, delta, k)
        lim = int(tau + 60 * sigma)
        kk, lp = _dgauss_logpmf(sigma, -lim, lim)
        p = math.exp(logsumexp(lp[c + kk > tau]))
        assert p <= delta / k * (1 + 1e-9), (c, rho, delta, k, p, delta / k)


# ------------------------------------------------------- neighbouring databases
def _star(n_cust=150):
    t, s, _ = S.SCENARIOS["S2_star"]()
    keep = set(t["customer"]["customer_id"].head(n_cust))
    t["customer"] = t["customer"][t["customer"]["customer_id"].isin(keep)].reset_index(drop=True)
    t["sales"] = t["sales"][t["sales"]["customer_id"].isin(keep)].reset_index(drop=True)
    return t, s


def _dp_spec(s, **kw):
    s = copy.deepcopy(s)
    s.setdefault("synthesis", {}).setdefault("privacy", {})["dp"] = {
        "unit": "customer", "epsilon": 2.0, "delta": 1e-9, "public_tables": ["product"], **kw}
    return load_spec(s)


class _Recorder:
    """Stands in for mechanisms.gaussian: records the exact statistic and its
    sensitivity; returns the exact statistic (first run) or the first run's
    output for the same mechanism (replay)."""

    def __init__(self, replay=None):
        self.calls, self.replay = [], replay

    def __call__(self, values, sensitivity, rho, acct, stage, mechanism):
        v = np.asarray(values, dtype=np.int64).ravel()
        acct.charge(M.Charge(stage, mechanism, rho))
        i = len(self.calls)
        self.calls.append((mechanism, v.copy(), float(sensitivity)))
        if self.replay is None:
            return v
        name, out, _ = self.replay.calls[i]
        assert name == mechanism and len(out) == len(v), (name, mechanism, len(out), len(v))
        return out


def _run(spec, tables, rec, monkeypatch):
    monkeypatch.setattr(R, "gaussian", rec)
    monkeypatch.setattr(R, "_hash_key", lambda: "0123456789abcdef")
    cfg = dp.parse(spec)
    pl = R.make_plan(spec, cfg)
    acct = M.Accountant(cfg.rho, cfg.delta)
    real = {t: df.reset_index(drop=True) for t, df in tables.items()}
    return R.fit(spec, real, cfg, acct, pl)


def _without(tables, cid):
    t = {k: v.copy() for k, v in tables.items()}
    t["customer"] = t["customer"][t["customer"]["customer_id"] != cid].reset_index(drop=True)
    t["sales"] = t["sales"][t["sales"]["customer_id"] != cid].reset_index(drop=True)
    return t


def test_c902_every_released_statistic_respects_its_sensitivity(monkeypatch):
    """Props. 901, 902, 904: with the degree cap fixed and the ranges and levels
    public (so the shapes of the statistics do not depend on the data), removing
    any one customer and their sales moves each statistic by at most its charged
    L2 sensitivity, given the earlier noisy outputs (adaptive composition).
    Includes the customer with the most sales, where truncation binds."""
    tables, s = _star()
    pub = {"customer.region": {"levels": sorted(tables["customer"]["region"].astype(str).unique())},
           "sales.channel": {"levels": sorted(tables["sales"]["channel"].astype(str).unique())}}
    for t, c in (("customer", "age"), ("customer", "income"), ("sales", "quantity"), ("sales", "amount")):
        x = tables[t][c]
        pub[f"{t}.{c}"] = {"min": float(x.min()), "max": float(x.max())}
    spec = _dp_spec(s, public=pub, max_degree={"sales": 4})
    base = _Recorder()
    _run(spec, tables, base, monkeypatch)
    assert {m for m, _, _ in base.calls} >= {"customer: marginals", "customer: correlations", "sales: marginals",
                                             "sales: correlations", "sales: rows per customer row"}
    counts = tables["sales"]["customer_id"].value_counts()
    ids = [counts.index[0]] + list(tables["customer"]["customer_id"].sample(12, random_state=3))
    worst = {}
    for cid in ids:
        rep = _Recorder(replay=base)
        _run(spec, _without(tables, cid), rep, monkeypatch)
        for (name, v0, sens), (_, v1, _) in zip(base.calls, rep.calls):
            d = float(np.linalg.norm(v0 - v1))
            worst[name] = max(worst.get(name, 0.0), d / sens)
            assert d <= sens + 1e-9, (cid, name, d, sens)
    # the bound is not vacuous: removing the busiest customer comes near it for the row-level statistics
    assert worst["sales: marginals"] > 0.3 and worst["customer: marginals"] > 0.3, worst


def test_c901_truncation_is_stable(monkeypatch):
    """Prop. 901: with the run's hash key fixed, removing one customer leaves the
    kept sales of every other customer unchanged, and no customer keeps more
    than D sales."""
    tables, s = _star()
    key, D = "fedcba9876543210", 3
    kept = R._truncate(tables["sales"], tables["customer"], ["customer_id"], ["customer_id"], D, key)
    assert kept.groupby("customer_id").size().max() <= D
    cid = tables["sales"]["customer_id"].value_counts().index[0]
    t2 = _without(tables, cid)
    kept2 = R._truncate(t2["sales"], t2["customer"], ["customer_id"], ["customer_id"], D, key)
    a = kept[kept["customer_id"] != cid].sort_values("sale_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, kept2.sort_values("sale_id").reset_index(drop=True))


# --------------------------------------------------------------- C904-C906
def test_c904_discovery_spends_only_within_the_plan_and_selects_levels():
    """Props. 903, 905: with nothing public, discovery releases the ranges and
    levels within the plan, rare levels are suppressed, and the whole run
    (discovery, fit, decode, stage 2) spends at most the stated budget."""
    tables, s = _star()
    tables["customer"].loc[0, "region"] = "a-region-of-one"
    spec = _dp_spec(s, epsilon=1.0)
    res = dp.synthesize(spec, tables=tables, write=False, evaluate=False)
    assert res.epsilon <= 1.0 + 1e-9 and res.delta <= res.config.delta * res.config.levels_delta_share * (1 + 1e-9)
    assert res.rho <= res.config.rho * (1 + 1e-9)
    assert "a-region-of-one" not in set(res.data["customer"]["region"].astype(str))
    st = res.statement
    assert st["schema"] == "relsynth.dp-statement/1" and st["unit"] == "customer"
    assert set(st["mechanisms"]) == {"discovery", "fit"}


def test_c905_post_processing_keeps_keys_valid_and_public_tables_as_given():
    """Prop. 905: stage 2 only post-processes: links are valid by construction,
    and the public table is released as given."""
    tables, s = _star()
    res = dp.synthesize(_dp_spec(s), tables=tables, write=False, evaluate=False)
    d = res.data
    assert d["sales"]["customer_id"].isin(d["customer"]["customer_id"]).all()
    assert d["sales"]["product_id"].isin(tables["product"]["product_id"]).all()
    assert d["customer"]["customer_id"].is_unique


def test_c906_refusals_happen_before_any_read():
    """Prop. 906 (scope): a unit that is not a table, a missing budget and a delta
    above 1e-5 are refused by the plan, which reads no data."""
    _, s = _star()
    with pytest.raises(PrivacyError, match="no\\s+default epsilon"):
        dp.plan(_dp_spec(s, epsilon=None))
    with pytest.raises(PrivacyError, match="delta"):
        dp.plan(_dp_spec(s, delta=1e-3))
    with pytest.raises(PrivacyError, match="not a table"):
        dp.plan(_dp_spec(s, unit="nobody"))
