"""Checks C600-C606: the privacy propositions of docs/proofs (section v2.5),
tested by simulation with known answers.  Each check names its proposition.
About ten minutes in total:

    pytest -m science tests/science/test_privacy_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fakedata"))
import scenarios as S  # noqa: E402

from relsynth.privacy import criteria, mia, stats, unit as U  # noqa: E402
from relsynth.privacy.attribute import attribute_inference  # noqa: E402
from relsynth.spec import load_spec  # noqa: E402

pytestmark = [pytest.mark.science, pytest.mark.slow]


def _people(n, seed):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n)
    return pd.DataFrame({"age": age, "income": np.round(rng.lognormal(10, 1, n), 2),
                         "region": rng.choice(list("ABCDEFG"), n), "score": rng.normal(0, 1, n).round(2),
                         "smoker": np.where(rng.random(n) < 0.15 + 0.4 * (age > 60), "y", "n")})


def _shop():
    t, s, _ = S.SCENARIOS["S9_small_shop"]()
    return t, load_spec(s)


def test_C600_person_split_keeps_rows_and_keys():
    """Prop. 600: a split by person loses and shares no owned row; every owned
    foreign key resolves inside the subset; shared tables stay whole."""
    t, s = _shop()
    u = U.resolve(s, "customer")
    assert set(u.owned) == {"invoice", "invoice_line"}
    a, b = U.split(u, t, 0.5, seed=3)
    A, B = U.subset(u, t, a), U.subset(u, t, b)
    for x in ("invoice", "invoice_line"):
        assert len(A[x]) + len(B[x]) == len(t[x])
    for D in (A, B):
        assert D["invoice"]["customer_id"].isin(D["customer"]["customer_id"]).all()
        assert D["invoice_line"]["invoice_id"].isin(D["invoice"]["invoice_id"]).all()
        assert len(D["track"]) == len(t["track"]) and len(D["staff"]) == len(t["staff"])


def test_C601_record_isolation_is_centred_under_the_null():
    """Prop. 601: with equal training and control sizes (or the subsample) and a
    release that never saw the people, the risk interval covers 0 at about the
    nominal rate; Anonymeter's fitted correction with unequal sizes does not."""
    cover, cover_anon = [], []
    for rep in range(20):
        pop = _people(9000, rep)
        tr, ct, syn = pop.iloc[:3000], pop.iloc[3000:6000], pop.iloc[6000:]
        r = criteria.record_isolation(tr, syn, ct, mode="multivariate", n_attacks=200, seed=rep)
        cover.append(r.risk.ci[0] == 0.0)
        if rep < 6:
            r2 = criteria.record_isolation(tr, syn, ct.iloc[:1200], mode="multivariate", n_attacks=200, seed=rep,
                                           size_correction="anonymeter")
            cover_anon.append(r2.risk.ci[0] == 0.0)
    assert np.mean(cover) >= 0.85
    assert np.mean(cover_anon) < np.mean(cover)


def test_C602_shared_shadows_null_auc_and_independent_labels():
    """Prop. 602: against a generator that ignores individuals, the pooled AUC
    interval of the shared-shadow attack covers 0.5, and per-target AUCs centre
    on 0.5."""
    covers, per_target = [], []
    for rep in range(6):
        pop = _people(5000, 100 + rep)
        pop.index = [f"p{i}" for i in range(len(pop))]
        ref = pop.iloc[:1500]
        tids = list(pop.index[1500:1510])
        kinds = pd.Series({t: "random" for t in tids})
        r = mia.shadow_attack(list(pop.index[1510:]), pop.loc[tids], kinds,
                              lambda ids, sd: ref.sample(len(ids), random_state=sd).reset_index(drop=True),
                              shadow_size=300, n_train=24, n_test=24, seed=rep, ref_profiles=ref)
        lo, hi = r.pooled["classifier"]["auc_ci"]
        covers.append(lo <= 0.5 <= hi)
        per_target += r.targets["classifier_auc"].dropna().tolist()
    assert np.mean(covers) >= 0.8
    assert abs(np.mean(per_target) - 0.5) < 0.05


def test_C603_dp_bound_holds_for_laplace_counts():
    """Prop. 603: against an epsilon-DP release (a person count plus Laplace
    noise of scale 1/epsilon), no threshold test beats e^eps * FPR (Kairouz et
    al. 2015): exactly on a grid of thresholds, and on simulated releases with
    the threshold set on independent draws."""
    from scipy.stats import laplace
    rng = np.random.default_rng(0)
    for eps in (0.5, 1.0, 2.0):
        t = np.linspace(95, 110, 301)
        fpr, tpr = laplace.sf(t, 100, 1 / eps), laplace.sf(t, 101, 1 / eps)
        assert np.all(tpr <= np.array([stats.dp_tpr_bound(f, eps) for f in fpr]) + 1e-12)
        thr = np.quantile(100 + rng.laplace(0, 1 / eps, 200_000), 0.99)
        out = 100 + rng.laplace(0, 1 / eps, 400_000)
        inn = 101 + rng.laplace(0, 1 / eps, 400_000)
        f, p = (out > thr).mean(), (inn > thr).mean()
        se = np.sqrt(p * (1 - p) / len(inn) + (np.exp(eps) ** 2) * f * (1 - f) / len(out))
        assert p <= stats.dp_tpr_bound(f, eps) + 3 * se


def test_C604_tpr_interval_coverage():
    """Prop. 604: thresholds from calibration scores, Clopper-Pearson intervals
    on test scores cover the true TPR and FPR at the threshold."""
    rng = np.random.default_rng(1)
    hit_t, hit_f, fprs = [], [], []
    for _ in range(400):
        yc, yt = rng.random(2000) < 0.5, rng.random(2000) < 0.5
        sc, st = rng.normal(size=2000) + 1.2 * yc, rng.normal(size=2000) + 1.2 * yt
        e = stats.tpr_at_fpr(sc, yc, st, yt, 0.01)
        tau = e["threshold"]
        true_tpr, true_fpr = 1 - norm.cdf(tau - 1.2), 1 - norm.cdf(tau)
        hit_t.append(e["tpr_ci"][0] <= true_tpr <= e["tpr_ci"][1])
        hit_f.append(e["fpr_ci"][0] <= true_fpr <= e["fpr_ci"][1])
        fprs.append(true_fpr)
    assert np.mean(hit_t) >= 0.93 and np.mean(hit_f) >= 0.93
    assert np.mean(fprs) <= 0.01 + 1 / 1000 + 0.001


def test_C605_profiles_invariant_on_a_multi_table_shop():
    """Prop. 605: profiles ignore key values and row order (two-level chain,
    parents two hops away)."""
    t, s = _shop()
    u = U.resolve(s, "customer")
    p1 = U.profiles(s, u, t)
    rng = np.random.default_rng(5)
    maps = {tb: {k: f"K{tb}{i}" for i, k in enumerate(rng.permutation(t[tb][s["tables"][tb]["primary_key"]]))}
            for tb in ("customer", "invoice", "track", "album", "genre", "staff")}
    t2 = {k: v.copy() for k, v in t.items()}
    for tb, m in maps.items():
        pk = s["tables"][tb]["primary_key"]
        t2[tb][pk] = t2[tb][pk].map(m)
    for r in s["relationships"]:
        if r["parent"] in maps and not r.get("composite"):
            t2[r["child"]][r["child_key"]] = t2[r["child"]][r["child_key"]].map(maps[r["parent"]])
    t2 = {k: v.sample(frac=1, random_state=9) for k, v in t2.items()}
    p2 = U.profiles(s, u, t2)
    p2.index = p2.index.map({v: k for k, v in maps["customer"].items()})
    pd.testing.assert_frame_equal(p1.sort_index(), p2.sort_index(), check_dtype=False, atol=1e-9)


def test_C606_specific_leakage_centred_for_population_generator():
    """Prop. 606: a release drawn from the population (not the members) gives
    specific attribute leakage whose interval covers 0 at about the nominal rate."""
    cover = []
    for rep in range(20):
        pop = _people(9000, 200 + rep)
        tr, ct, ref, syn = pop.iloc[:2000], pop.iloc[2000:4000], pop.iloc[4000:6000], pop.iloc[6000:]
        a = attribute_inference(tr, ct, syn, ref, secret="smoker", known=["age", "region", "income", "score"],
                                seed=rep)
        lo, hi = a["specific_leakage"]["ci"]
        cover.append(lo <= 0 <= hi)
    assert np.mean(cover) >= 0.85
