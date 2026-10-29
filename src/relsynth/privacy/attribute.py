# SPDX-License-Identifier: Apache-2.0
"""Attribute inference for declared sensitive attributes, against a baseline
attacker that sees only the population.

EDPB Guidelines 02/2026 (para. 67) count an inference as a failure of the
"no inference" criterion only when it is *specific* (about one person) and
*meaningful*, that is not available from general knowledge or population-level
data.  So the attacker that uses the synthetic data is compared with one that
uses population data instead (people who are in neither the training nor the
evaluation sets), with the same model:

    advantage(members)     = acc_syn - acc_pop   on training people
    advantage(non-members) = acc_syn - acc_pop   on control people
    specific leakage       = advantage(members) - advantage(non-members)

A positive advantage on non-members is what the synthetic data teaches about
everyone (utility); only the excess on members is about the people in the
training data (Prop. 606).  Numeric attributes are guessed as their decile band
of the population, so success is "the right band".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import paired_difference, z_of


def _design(frames, cols, ref):
    parts = []
    for c in cols:
        s = ref[c]
        if s.dtype.kind in "biuf" and s.nunique() > 10:
            lo, sd = float(np.nanmean(s.astype(float))), float(np.nanstd(s.astype(float))) or 1.0
            parts.append([np.column_stack([np.nan_to_num((pd.to_numeric(f[c], errors="coerce").to_numpy(float) - lo)
                                                         / sd), f[c].isna().to_numpy(float)]) for f in frames])
        else:
            lv = s.astype(str).value_counts().index[:30]
            parts.append([np.column_stack([(f[c].astype(str).to_numpy() == x).astype(float) for x in lv])
                          for f in frames])
    return [np.column_stack([p[i] for p in parts] + [np.ones(len(f))]) for i, f in enumerate(frames)]


def _softmax_fit(X, y, K, lam=1.0):
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    d = X.shape[1]
    Y = np.eye(K)[y]

    def f(w):
        W = w.reshape(d, K)
        Z = X @ W
        lse = logsumexp(Z, axis=1)
        P = np.exp(Z - lse[:, None])
        return (lse - (Z * Y).sum(1)).sum() + lam * (W[:-1] ** 2).sum(), \
            (X.T @ (P - Y) + 2 * lam * np.vstack([W[:-1], np.zeros((1, K))])).ravel()
    w = minimize(f, np.zeros(d * K), jac=True, method="L-BFGS-B", options={"maxiter": 300}).x
    return w.reshape(d, K)


def _labels(s: pd.Series, ref: pd.Series):
    if ref.dtype.kind in "biuf" and ref.nunique() > 10:
        edges = np.unique(np.nanquantile(ref.astype(float), np.linspace(0, 1, 11)[1:-1]))
        return np.searchsorted(edges, pd.to_numeric(s, errors="coerce").fillna(-np.inf), "right"), len(edges) + 1, True
    lv = {x: i for i, x in enumerate(ref.astype(str).value_counts().index)}
    other = len(lv)
    return s.astype(str).map(lv).fillna(other).astype(int).to_numpy(), other + 1, False


def attribute_inference(members: pd.DataFrame, non_members: pd.DataFrame, syn: pd.DataFrame,
                        population: pd.DataFrame, *, attribute: str, known: list[str], max_rows: int = 20000,
                        seed: int = 0, level: float = 0.95) -> dict:
    """Advantage of the synthetic-data attacker over the population attacker,
    on members and non-members, with intervals."""
    rng = np.random.default_rng(seed)
    known = [c for c in known if c != attribute and c in syn.columns]
    take = lambda df: df.sample(min(len(df), max_rows), random_state=int(rng.integers(2**31)))  # noqa: E731
    syn_s, pop_s = take(syn), take(population)
    Xs, Xp, Xm, Xn = _design([syn_s, pop_s, members, non_members], known, population)
    ys, K, banded = _labels(syn_s[attribute], population[attribute])
    yp, _, _ = _labels(pop_s[attribute], population[attribute])
    ym, _, _ = _labels(members[attribute], population[attribute])
    yn, _, _ = _labels(non_members[attribute], population[attribute])
    Ws, Wp = _softmax_fit(Xs, ys, K), _softmax_fit(Xp, yp, K)
    cs_m, cp_m = (Xm @ Ws).argmax(1) == ym, (Xm @ Wp).argmax(1) == ym
    cs_n, cp_n = (Xn @ Ws).argmax(1) == yn, (Xn @ Wp).argmax(1) == yn
    adv_m = paired_difference(cs_m, cp_m, level)
    adv_n = paired_difference(cs_n, cp_n, level)
    se = np.hypot((adv_m["ci"][1] - adv_m["value"]), (adv_n["ci"][1] - adv_n["value"])) / z_of(level)
    spec = adv_m["value"] - adv_n["value"]
    z = z_of(level)
    correct_ids = list(members.index[cs_m & ~cp_m])
    return {"criterion": "inference", "attack": "attribute_inference_vs_population", "attribute": attribute,
            "known": known, "banded": banded,
            "accuracy_syn_members": float(cs_m.mean()), "accuracy_population_members": float(cp_m.mean()),
            "accuracy_syn_non_members": float(cs_n.mean()), "accuracy_population_non_members": float(cp_n.mean()),
            "advantage_members": adv_m, "advantage_non_members": adv_n,
            "specific_leakage": {"value": float(spec), "ci": [float(spec - z * se), float(spec + z * se)]},
            "_gained_ids": correct_ids}
