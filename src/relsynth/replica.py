#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone sampler of an exported relsynth model.

    python replicate.py model.json --out synthetic/ [--draw 0|mean] [--seed 7]
                        [--scale 1.0] [--reference DIR]

Needs only numpy, pandas and scipy, and relsynth_expr.py (the engine's rule
language, written next to this script by the export).  model.json is written by
`relsynth export-model`; it holds the structure of the model and one or more
posterior draws theta^(m) of every parameter block (plus their posterior mean).
A synthetic database is a draw from the posterior predictive distribution:

    theta ~ p(theta | data)          (pick one exported draw, or the mean)
    D*    ~ p(D* | theta)            (this script)

The generative process (model.md / model.pdf give it in full):

  1. tables in topological order (parents first);
  2. an entity table of n rows: segment sizes n*pi rounded by largest remainder,
     one latent segment per row; copied reference tables are read from
     --reference (they are copied, not modelled); a self-referencing table grows
     as a multi-type Galton-Watson forest with the source's level x segment counts;
  3. a row's modelled columns given its segment: a Gaussian copula with the
     segment's correlation matrix and the segment's marginals (Dirichlet
     categorical laws, quantile functions for numbers and dates);
  4. a many-to-one link of a row in segment k: null with probability p_null[k],
     else parent segment l ~ phi[k, .], then a parent row of segment l by its degree
     weight (Gamma frailty), or by the DP count law of that link;
  5. a fact: rows per anchor row (Poisson, 1 + Poisson or DP counts), partner
     segment by block rate and partner row by weight (Poisson block model), or
     active anchors, present pairs and multiplicities (occupancy model); further
     dimension keys by the block's segment law; the fact's columns given its block
     (and the parent attributes it is conditioned on); the time column from the
     anchor segment's period profile;
  6. lookups, derived columns and aggregates (expressions of the spec, evaluated
     with the engine's own evaluator, so a rule gives the same values here).

What this script leaves out is listed in model.json under "replica_scope" (rules
that condition rows by rejection, slowly changing versions, participation and
affinity moves, group effects, text models, nested facts, ...).  The engine
(`relsynth run`) applies them on top of the same process; model.md states for
each its mathematical form and the proposition that proves it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, ndtr, ndtri

try:                                     # shipped next to this script by the export
    from relsynth_expr import Evaluator, derive_values
except ImportError:                      # inside the relsynth package
    from relsynth.expr import Evaluator, derive_values

OTHER = "__OTHER__"
DAY_NS = 86400 * 10**9


# ====================================================================== helpers
def nearest_corr(C):
    C = (np.asarray(C, float) + np.asarray(C, float).T) / 2
    w, V = np.linalg.eigh(C)
    C = (V * np.maximum(w, 1e-8)) @ V.T
    d = np.sqrt(np.diag(C))
    return C / d[:, None] / d[None, :]


def allocate(p, n, rng):
    """Segment labels with counts n p rounded by largest remainder, random order."""
    p = np.asarray(p, float)
    p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / len(p))
    raw = p * n
    cnt = np.floor(raw).astype(int)
    rem = n - cnt.sum()
    if rem > 0:
        cnt[np.argsort(-(raw - cnt) + 1e-9 * rng.random(len(p)))[:rem]] += 1
    return rng.permutation(np.repeat(np.arange(len(p)), cnt))


def keystr(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def dateish(s):
    """Dates or datetimes (or ISO date strings) in at least 95% of the first 200 values."""
    import datetime as _dt
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    head = s.dropna().head(200)
    if not len(head):
        return False
    if head.map(lambda x: isinstance(x, (_dt.date, _dt.datetime, pd.Timestamp))).mean() > 0.95:
        return True
    if head.map(lambda x: isinstance(x, str)).mean() > 0.95 and head.astype(str).str.len().median() >= 8 and \
            head.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.95:
        return pd.to_datetime(head, errors="coerce", format="mixed").notna().mean() > 0.95
    return False


# ====================================================================== codecs and marginals
def from_uniform(m, u):
    """Inverse of one marginal: discrete codes (-1 = null) or values (nan = null)."""
    u = np.asarray(u, float)
    pn = m["p_null"]
    if m["kind"] == "discrete":
        p = np.asarray(m["p"], float)
        cum = pn + (1 - pn) * np.cumsum(p)
        idx = np.minimum(np.searchsorted(cum, u, side="left"), len(p) - 1)
        return np.where(u < pn, -1, idx)
    uu = np.clip((u - pn) / max(1 - pn, 1e-12), 0, 1)
    v = np.interp(uu, m["knots_c"], m["knots_v"])
    return np.where(u < pn, np.nan, v)


def latent_bounds(m, x):
    """(lo, hi] on the N(0, 1) scale of observed model values (for conditioning)."""
    from scipy.special import ndtri
    pn = m["p_null"]
    if m["kind"] == "discrete":
        x = np.asarray(x)
        p = np.asarray(m["p"], float)
        cum = np.clip(np.concatenate([[0.0, pn], pn + (1 - pn) * np.cumsum(p)]), 0, 1)
        cum[-1] = 1.0
        idx = np.where(x < 0, 0, x + 1)
        lo, hi = cum[idx], cum[idx + 1]
    else:
        kv, kc = np.asarray(m["knots_v"], float), np.asarray(m["knots_c"], float)
        uv = np.unique(kv)
        c = np.array([kc[kv == q].max() for q in uv])
        cum = pn + (1 - pn) * c
        cum[-1] = 1.0
        obs = ~np.isnan(x)
        q = np.minimum(np.searchsorted(uv, np.where(obs, x, uv[0]), side="left"), len(uv) - 1)
        lo_q = np.where(q == 0, pn, cum[np.maximum(q - 1, 0)])
        hi_q = cum[q]
        if not cum[0] > pn + 1e-9 and len(uv) > 1:
            lo_q = np.where(q == 0, cum[0], lo_q)
            hi_q = np.where(q == 0, cum[1], hi_q)
        lo = np.where(obs, lo_q, 0.0)
        hi = np.where(obs, hi_q, pn)
    lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    hi = np.where(hi <= lo, np.minimum(lo + 1e-9, 1.0), hi)
    return ndtri(lo), ndtri(hi)


def encode(cd, values):
    """Source values -> model values of one codec (for parent attributes given to a fact)."""
    s = pd.Series(values).reset_index(drop=True)
    if cd["kind"] == "discrete":
        lv = cd["levels"]
        idx = {keystr(v) if cd["ctype"] in ("integer", "float") else str(v): i for i, v in enumerate(lv)}
        other = idx.get(OTHER)
        out = np.full(len(s), -1, dtype=np.int64)
        for i, v in enumerate(s.to_numpy(dtype=object)):
            if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT:
                continue
            k = keystr(v) if cd["ctype"] in ("integer", "float") else str(v)
            j = idx.get(k, other)
            out[i] = -1 if j is None else j
        return out
    if cd["ctype"] == "date":
        return pd.to_datetime(s, errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / DAY_NS \
            * np.where(pd.to_datetime(s, errors="coerce").isna(), np.nan, 1.0)
    if cd["ctype"] == "datetime":
        return pd.to_datetime(s, errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / 1e9 \
            * np.where(pd.to_datetime(s, errors="coerce").isna(), np.nan, 1.0)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def decode(cd, arr, rng, tail_w=None):
    if cd.get("parts"):
        lv = [tuple(x) for x in cd["levels"]]
        out = []
        for q, part in enumerate(cd["parts"]):
            codes = np.array([lv[i][q] if i >= 0 else -1 for i in np.asarray(arr)], dtype=np.int64)
            out.append(decode(part, codes, rng))
        return out
    if cd["kind"] == "discrete":
        arr = np.asarray(arr)
        lv = np.array(list(cd["levels"]) + [None], dtype=object)
        out = lv[np.where(arr < 0, len(cd["levels"]), arr)]
        if OTHER in cd["levels"]:
            oth = np.flatnonzero(arr == cd["levels"].index(OTHER))
            if len(oth):
                tail = cd.get("tail")
                if tail:
                    vals, cnt = tail
                    w = np.asarray(tail_w if tail_w is not None else np.asarray(cnt, float) / np.sum(cnt))
                    out[oth] = np.array(vals, dtype=object)[rng.choice(len(vals), size=len(oth), p=w / w.sum())]
                else:
                    kept = [v for v in cd["levels"] if v != OTHER]
                    out[oth] = np.array(kept, dtype=object)[rng.integers(0, len(kept), size=len(oth))] if kept else None
        if cd["ctype"] == "integer":
            return pd.array([None if v is None else int(v) for v in out], dtype="Int64")
        if cd["ctype"] == "float":
            return np.array([np.nan if v is None else float(v) for v in out], dtype=float)
        if cd["ctype"] == "boolean":
            return np.array([None if v is None else bool(v) for v in out], dtype=object)
        return out
    arr = np.asarray(arr, float)
    if cd.get("min") is not None:
        arr = np.where(np.isnan(arr), arr, np.maximum(arr, cd["min"]))
    if cd.get("max") is not None:
        arr = np.where(np.isnan(arr), arr, np.minimum(arr, cd["max"]))
    if cd["ctype"] == "date":
        ns = np.where(np.isnan(arr), np.nan, np.round(arr) * DAY_NS)
        return pd.to_datetime(pd.Series(ns), errors="coerce").dt.normalize().to_numpy()
    if cd["ctype"] == "datetime":
        return pd.to_datetime(pd.Series(np.where(np.isnan(arr), np.nan, arr * 1e9)), errors="coerce").to_numpy()
    if cd["ctype"] == "integer":
        return pd.array([None if np.isnan(v) else int(round(v)) for v in arr], dtype="Int64")
    if cd.get("decimals") is not None:
        arr = np.round(arr, cd["decimals"])
    return arr


def sample_attributes(tm, th, groups, rng, given=None):
    """The table model: per group a Gaussian copula with that group's marginals
    (falling back to the table's pooled law); given parent attributes enter as
    observed coordinates of the copula (conditional normal)."""
    groups = np.asarray(groups)
    M, p = len(groups), len(tm["codecs"])
    out = np.zeros((M, p))
    names = [c["name"] for c in tm["codecs"]]
    ai = [names.index(a) for a in tm["aux"]] if (tm["aux"] and given is not None) else []
    ri = [j for j in range(p) if j not in ai]
    G = [encode(tm["codecs"][j], given[names[j]]) for j in ai] if ai else []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        ms = th["groups"].get(str(int(g)), th["global"])
        C = nearest_corr(th["copulas"].get(str(int(g)), th["pooled"])) if p > 1 else np.eye(p)
        if ai:
            # a given column's latent value is uniform on its bin: Z ~ N(0, 1) truncated
            # to (lo, hi], by the inverse cdf of a uniform on (Phi(lo), Phi(hi)]
            za = []
            for q, j in enumerate(ai):
                lo, hi = latent_bounds(ms[j], G[q][idx])
                a, b = ndtr(lo), ndtr(hi)
                zq = ndtri(np.clip(a + (b - a) * rng.random(len(a)), 1e-12, 1 - 1e-12))
                za.append(np.clip(np.nan_to_num(zq, nan=0.0, posinf=6, neginf=-6), -6, 6))
            za = np.column_stack(za)
            Caa, Cra, Crr = C[np.ix_(ai, ai)], C[np.ix_(ri, ai)], C[np.ix_(ri, ri)]
            B = np.linalg.solve(Caa, Cra.T).T
            S = Crr - B @ Cra.T
            L = np.linalg.cholesky((S + S.T) / 2 + 1e-9 * np.eye(len(ri)))
            Z = np.zeros((len(idx), p))
            Z[:, ri] = za @ B.T + rng.standard_normal((len(idx), len(ri))) @ L.T
        else:
            Z = rng.standard_normal((len(idx), p)) @ np.linalg.cholesky(C).T if p else np.zeros((len(idx), 0))
        U = ndtr(Z)
        for j in range(p):
            if j in ai:
                continue
            out[idx, j] = from_uniform(ms[j], U[:, j])
    cols = {}
    for j, cd in enumerate(tm["codecs"]):
        if cd["name"] in tm["aux"]:
            continue
        v = out[:, j].astype(np.int64) if cd["kind"] == "discrete" else out[:, j]
        dec = decode(cd, v, rng, th.get("tail_w", {}).get(cd["name"]))
        if cd.get("parts"):
            for part, vals in zip(cd["parts"], dec):
                cols[part["name"]] = vals
        else:
            cols[cd["name"]] = dec
    return pd.DataFrame({c: cols[c] for c in tm["out_cols"] if c in cols})


# ====================================================================== links
def row_weights(alpha, z, rng, obs=None, copied=False):
    """Degree weights of parent rows, rescaled to sum to their segment's size:
    Gamma(alpha_k) frailties (1 when alpha_k is infinite); rows of a copied parent
    get a Gamma(observed degree + a_k) draw of their own share (degree-corrected
    block model)."""
    z = np.asarray(z)
    K = int(z.max()) + 1 if len(z) else 0
    K = max(K, len(alpha))
    if copied and obs is not None and len(obs) == len(z):
        obs = np.asarray(obs, float)
        m = np.bincount(z, minlength=K).astype(float)
        mu = np.bincount(z, weights=obs, minlength=K) / np.maximum(m, 1)
        ex2 = np.bincount(z, weights=obs * obs, minlength=K) / np.maximum(m, 1)
        var = (ex2 - mu ** 2) * m / np.maximum(m - 1, 1)
        a = np.where((var > mu * (1 + 1e-9)) & (mu > 0), mu ** 2 / np.maximum(var - mu, 1e-12), 1e6)
        a = np.maximum(a, 1.0 / np.maximum(m, 1.0))
        w = rng.gamma(obs + a[z])
        tot = np.bincount(z, weights=w, minlength=K)
        return w * (m[z] / np.maximum(tot[z], 1e-300))
    al = np.array([np.inf if a is None else a for a in alpha] + [np.inf] * (K - len(alpha)), float)[z]
    fin = np.isfinite(al)
    w = np.ones(len(z))
    w[fin] = rng.gamma(al[fin], 1 / al[fin])
    tot = np.bincount(z, weights=w, minlength=K)
    cnt = np.bincount(z, minlength=K)
    return w * (cnt[z] / np.maximum(tot[z], 1e-12))


def dp_counts(dp, z, lam, rng, bb):
    """Children per row from the DP posterior predictive of its segment's counts."""
    out = np.zeros(len(z), dtype=np.int64)
    for k in np.unique(z):
        idx = np.flatnonzero(z == k)
        obs = np.asarray(dp["counts"][k] if k < len(dp["counts"]) else [], dtype=np.int64)
        n, a = len(obs), dp["a"]
        from_obs = rng.random(len(idx)) < n / (n + a) if n else np.zeros(len(idx), bool)
        if n:
            w = bb(("dp", id(dp), int(k)), n)
            out[idx[from_obs]] = obs[rng.choice(n, size=from_obs.sum(), p=w)]
        mn = dp.get("min", 0)
        out[idx[~from_obs]] = mn + rng.poisson(max(lam[k] - mn, 0.0), size=(~from_obs).sum())
    return out


def conditional_counts(R, M, obs, a, rng, bb, key, mn=0, sweeps=30):
    """M counts i.i.d. from the DP posterior predictive, conditioned on their sum R
    (pairwise Gibbs on splits, which keeps the conditional law invariant)."""
    if M == 0:
        return np.zeros(0, np.int64)
    obs = np.asarray(obs, np.int64)
    n = len(obs)
    lam = obs.mean() if n else R / M
    cmax = int(max(R, obs.max(initial=0))) + 1
    x = np.arange(cmax + 1)
    lb = max(lam - mn, 1e-12)
    xm = np.maximum(x - mn, 0)
    f = np.where(x >= mn, (a / (n + a)) * np.exp(xm * np.log(lb) - lb - gammaln(xm + 1)), 0.0)
    if n:
        np.add.at(f, np.minimum(obs, cmax), (n / (n + a)) * bb(key, n))
    lf = np.log(np.maximum(f, 1e-300))
    N = np.full(M, R // M, dtype=np.int64)
    N[rng.permutation(M)[: R - N.sum()]] += 1
    if M < 2:
        return N
    for _ in range(sweeps):
        perm = rng.permutation(M)
        i, j = perm[0:M - 1:2], perm[1:M:2]
        s = N[i] + N[j]
        xs = np.arange(s.max() + 1)
        L = lf[np.minimum(xs[None, :], cmax)] + lf[np.minimum(np.maximum(s[:, None] - xs[None, :], 0), cmax)]
        L = np.where(xs[None, :] <= s[:, None], L, -np.inf)
        pick = np.argmax(L - np.log(-np.log(rng.random(L.shape))), axis=1)
        N[i], N[j] = pick, s - pick
    return N


def choose_parents(lk, th, zc, zp, pkeys, rng, bb, w):
    phi = np.asarray(th["phi"], float)
    Kp = phi.shape[1]
    members = [np.flatnonzero(zp == t) for t in range(Kp)]
    avail = np.array([len(m) > 0 for m in members])
    res = np.empty(len(zc), dtype=object)
    isnull = rng.random(len(zc)) < np.asarray(th["p_null"], float)[zc]
    seg_of = np.full(len(zc), -1)
    for k in np.unique(zc):
        rows = np.flatnonzero((zc == k) & ~isnull)
        pk = phi[k] * avail
        if pk.sum() == 0:
            pk = avail.astype(float)
        seg_of[rows] = rng.choice(Kp, size=len(rows), p=pk / pk.sum())
    dp = lk.get("count_dp")
    for t in np.unique(seg_of[seg_of >= 0]):
        rr = np.flatnonzero(seg_of == t)
        m = members[t]
        if dp and len(rr) == 1:
            res[rr] = pkeys[m[rng.integers(len(m))]]
        elif dp:
            obs = dp["counts"][t] if t < len(dp["counts"]) else []
            cnt = conditional_counts(len(rr), len(m), obs, dp["a"], rng, bb, ("dp", lk["name"], int(t)), mn=dp.get("min", 0))
            res[rng.permutation(rr)] = pkeys[np.repeat(m, cnt)]
        elif lk.get("min_one"):
            rr = rng.permutation(rr)
            k1 = min(len(rr), len(m))
            res[rr[:k1]] = pkeys[rng.permutation(m)[:k1]]
            if len(rr) > k1:
                res[rr[k1:]] = pkeys[m[rng.choice(len(m), size=len(rr) - k1, p=w[m] / w[m].sum())]]
        else:
            res[rr] = pkeys[m[rng.choice(len(m), size=len(rr), p=w[m] / w[m].sum())]]
    res[isnull] = None
    return res


def grow_tree(tr, th, s, rng):
    """Forest with round(s n_kd) nodes of segment k at depth d; each node of depth d
    attached to a parent of depth d-1 with probability proportional to
    w_i lam[k_i, d-1] psi[k_i, d-1, k] (Poisson counts conditioned on totals)."""
    cnt = np.asarray(tr["level_counts"], float)          # depth x K
    lam_d, psi_d = np.asarray(th["lam_d"], float), np.asarray(th["psi_d"], float)
    alpha = np.array([np.inf if a is None else a for a in th["frailty"]], float)
    target = np.round(s * cnt).astype(np.int64)
    if target[0].sum() == 0:
        target[0, int(np.argmax(cnt[0]))] = 1
    z = np.repeat(np.arange(cnt.shape[1]), target[0])
    par = np.full(len(z), -1, dtype=np.int64)
    prev = np.arange(len(z))
    for d in range(1, cnt.shape[0]):
        if not len(prev) or not target[d].sum():
            break
        zp = z[prev]
        w = np.ones(len(prev))
        fin = np.isfinite(alpha[zp])
        w[fin] = rng.gamma(alpha[zp[fin]], 1 / alpha[zp[fin]])
        base = w * lam_d[zp, d - 1]
        nz, npar = [], []
        for k2 in np.flatnonzero(target[d]):
            pr = base * psi_d[zp, d - 1, k2]
            if not pr.sum() > 0:
                pr = base if base.sum() > 0 else np.ones(len(prev))
            npar.append(prev[rng.choice(len(prev), size=target[d, k2], p=pr / pr.sum())])
            nz.append(np.full(target[d, k2], k2))
        start = len(z)
        z, par = np.concatenate([z, *nz]), np.concatenate([par, *npar])
        prev = np.arange(start, len(z))
    return z.astype(np.int64), par


# ====================================================================== facts
def bernoulli_grid(A, J, p, rng):
    """All pairs (i, j) of A x J, each present with probability p (geometric skips)."""
    N = len(A) * len(J)
    if N == 0 or p <= 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    k = rng.binomial(N, p)
    pos = np.sort(rng.choice(N, size=k, replace=False))
    return np.asarray(A)[pos // len(J)], np.asarray(J)[pos % len(J)]


def sample_pairs(fk, th, z1, z2, w1, w2, m_real, rng, bb):
    """Poisson block model: N_kl ~ Poisson(eta*_kl W1_k W2_l), anchors and partners
    by weight; with a count law on the anchor, rows per anchor first, then each
    row's partner segment by rate."""
    eta = np.asarray(th["eta"], float)
    K1, K2 = eta.shape
    m_syn = np.bincount(z2, minlength=K2).astype(float)
    m_real = np.asarray(m_real, float)
    mem1 = [np.flatnonzero(z1 == k) for k in range(K1)]
    mem2 = [np.flatnonzero(z2 == l) for l in range(K2)]
    if fk.get("count_dp") or fk.get("min_one"):
        W2 = np.array([w2[m].sum() for m in mem2])
        ok2 = (W2 > 0) & (m_real > 0)
        rate = eta * np.where(ok2, m_real / np.maximum(m_syn, 1), 0.0)[None, :] * W2[None, :]
        lam1 = (eta * m_real[None, :]).sum(1)
        n_u = dp_counts(fk["count_dp"], z1, lam1, rng, bb) if fk.get("count_dp") else \
            1 + rng.poisson(np.maximum(lam1[z1] - 1, 0))
        u = np.repeat(np.arange(len(z1)), n_u)
        v = np.zeros(len(u), np.int64)
        blk = np.zeros(len(u), np.int64)
        for k in np.unique(z1[u]):
            rows = np.flatnonzero(z1[u] == k)
            pk = rate[k] if rate[k].sum() > 0 else ok2.astype(float)
            ls = rng.choice(K2, size=len(rows), p=pk / pk.sum())
            for l in np.unique(ls):
                rr = rows[ls == l]
                pv = w2[mem2[l]] / W2[l]
                v[rr] = mem2[l][rng.choice(len(pv), size=len(rr), p=pv)]
                blk[rr] = k * K2 + l
        perm = rng.permutation(len(u))
        return u[perm], v[perm], blk[perm]
    us, vs, bl = [], [], []
    for k in range(K1):
        if not len(mem1[k]):
            continue
        W1 = w1[mem1[k]].sum()
        for l in range(K2):
            if not len(mem2[l]) or m_real[l] == 0:
                continue
            N = rng.poisson(eta[k, l] * m_real[l] / m_syn[l] * W1 * w2[mem2[l]].sum())
            if N:
                us.append(mem1[k][rng.choice(len(mem1[k]), size=N, p=w1[mem1[k]] / W1)])
                vs.append(mem2[l][rng.choice(len(mem2[l]), size=N, p=w2[mem2[l]] / w2[mem2[l]].sum())])
                bl.append(np.full(N, k * K2 + l))
    u = np.concatenate(us) if us else np.zeros(0, np.int64)
    v = np.concatenate(vs) if vs else np.zeros(0, np.int64)
    blk = np.concatenate(bl) if bl else np.zeros(0, np.int64)
    perm = rng.permutation(len(u))
    return u[perm], v[perm], blk[perm]


def sample_occupancy(fk, th, z1, z2, m_real, rng):
    """Occupancy model: anchor i active ~ Bernoulli(pi_k); pair (i, j) present ~
    Bernoulli(p*_kl), p* = min(1, p m_l / m*_l) for 0 < p < 1; rows of a present
    pair 1 + Poisson(mu); with mandatory participation, empty active anchors are
    redrawn (rejection, exact for the zero-truncated law)."""
    pi, P = np.asarray(th["pi"], float), np.asarray(th["p"], float)
    K1, K2 = P.shape
    m_real = np.asarray(m_real, float)
    mem1 = [np.flatnonzero(z1 == k) for k in range(K1)]
    mem2 = [np.flatnonzero(z2 == l) for l in range(K2)]
    m_syn = np.array([len(J) for J in mem2], float)
    ra = fk.get("row_active")
    active = np.asarray(ra, bool) if ra is not None and len(ra) == len(z1) else rng.random(len(z1)) < pi[z1]
    us, vs, bl = [], [], []

    def draw(sel):
        for k in range(K1):
            A = mem1[k][sel[mem1[k]]]
            if not len(A):
                continue
            for l in range(K2):
                p = P[k, l]
                if not len(mem2[l]) or p <= 0 or m_real[l] == 0:
                    continue
                if p < 1:
                    p = min(1.0, p * m_real[l] / m_syn[l])
                if p >= 1:
                    ii, jj = np.repeat(A, len(mem2[l])), np.tile(mem2[l], len(A))
                else:
                    ii, jj = bernoulli_grid(A, mem2[l], p, rng)
                if len(ii):
                    us.append(ii)
                    vs.append(jj)
                    bl.append(np.full(len(ii), k * K2 + l))
    draw(active)
    if fk.get("occ_min_one"):
        for _ in range(40):
            has = np.zeros(len(z1), bool)
            for ii in us:
                has[ii] = True
            need = active & ~has
            if not need.any():
                break
            draw(need)
    u = np.concatenate(us) if us else np.zeros(0, np.int64)
    v = np.concatenate(vs) if vs else np.zeros(0, np.int64)
    blk = np.concatenate(bl) if bl else np.zeros(0, np.int64)
    mb = fk.get("mult_by", "none")
    if mb != "none" and len(u):
        mu = np.asarray(th["mu"], float)
        mu = mu[blk % K2] if mb == "partner" else (mu[blk // K2] if mb == "anchor" else mu.reshape(-1)[blk])
        rep = 1 + rng.poisson(mu)
        u, v, blk = np.repeat(u, rep), np.repeat(v, rep), np.repeat(blk, rep)
    perm = rng.permutation(len(u))
    return u[perm], v[perm], blk[perm]


# ====================================================================== expressions
def resolver(df, out, tables, rels, t):
    """resolve('fk.col' or 'fk.fk2.col') -> the parent values aligned with the rows
    of df (table t), following the links rels {table: {key column: {"parent": table}}} through
    the generated tables `out`; a key without a parent row reads null."""
    cache = {}

    def resolve(name):
        if name in cache:
            return cache[name]
        parts = name.split(".")
        if parts[0] not in df.columns:
            raise KeyError(name)
        cur, vals = t, df[parts[0]].to_numpy(dtype=object)
        for q, part in enumerate(parts[:-1]):
            parent = rels.get(cur, {}).get(part, {}).get("parent")
            nxt = parts[q + 1]
            if parent is None or parent not in out or nxt not in out[parent].columns:
                raise KeyError(name)
            p = out[parent]
            m = dict(zip(p[tables[parent]["primary_key"]].map(keystr), p[nxt].to_numpy(dtype=object)))
            vals = np.array([m.get(keystr(k)) for k in vals], dtype=object)
            cur = parent
        cache[name] = pd.Series(vals, index=df.index)
        return cache[name]
    return resolve


def evaluate(expr, df, out, tables, rels, t=None):
    """Values of an expression of the spec over the rows of df (table t)."""
    return Evaluator(df, resolver(df, out, tables, rels, t)).eval(expr)


# ====================================================================== main
AGGREGATES = ("count", "sum", "min", "max", "mean", "exists", "count_distinct")


def table_path(root, t):
    """root/<t>.csv, refused when the table name leads outside root."""
    root = Path(root).resolve()
    p = (root / f"{t}.csv").resolve()
    if p.parent != root:
        raise SystemExit(f"table name {t!r} points outside the folder {root}")
    return p


def generate(model, draw="mean", seed=7, scale=1.0, reference=None):
    S = model["structure"]
    th = model["mean"] if draw == "mean" else model["draws"][int(draw)]
    rng = np.random.default_rng(seed)
    T = S["tables"]
    bbc = {}

    def bb(key, n):
        if key not in bbc or len(bbc[key]) != n:
            bbc[key] = np.full(n, 1.0 / n) if draw == "mean" else rng.dirichlet(np.ones(n))
        return bbc[key]
    out, zs, keys = {}, {}, {}
    notes = []
    # key paths of lookups and expressions: every single-column link of a table
    rels = {t: {c: {"parent": p} for c, p in S.get("paths", {}).get(t, {}).items()} for t in T}
    for r in S["links"].values():
        rels[r["child"]][r["child_key"]] = r
    for f, fk in S["facts"].items():
        rels[f][fk["anchor_key"]] = {"parent": fk["anchor"]}
        if fk.get("partner"):
            rels[f][fk["partner_key"]] = {"parent": fk["partner"]}
        for x in fk.get("extras", []):
            rels[f][x["child_key"]] = {"parent": x["parent"]}
    for e, E in S["entities"].items():
        for c, p in (E.get("strata") or {}).get("by_parent", {}).items():
            rels[e][c] = {"parent": p}
    for e in S["entity_order"]:
        E = S["entities"][e]
        s = scale * E.get("scale", 1.0)
        pk = T[e]["primary_key"]
        if E.get("copied"):
            if isinstance(reference, dict) and e in reference:
                df = reference[e].reset_index(drop=True)
            elif reference is not None and not isinstance(reference, dict) and table_path(reference, e).exists():
                df = pd.read_csv(table_path(reference, e))
            else:
                raise SystemExit(f"{e} is a copied reference table: pass --reference DIR with {e}.csv "
                                 "(its rows in the source's order)")
            if len(df) != len(E["z"]):
                raise SystemExit(f"{e}: the reference has {len(df)} rows, the model {len(E['z'])}")
            out[e], zs[e], keys[e] = df, np.asarray(E["z"], np.int64), df[pk].to_numpy(dtype=object)
            continue
        if E.get("unsupported"):
            notes.append(f"{e}: {E['unsupported']} is not generated by the replica")
            continue
        tree = S["trees"].get(e)
        if tree:
            z, par = grow_tree(tree, th["trees"][e], s, rng)
        elif E.get("fixed_counts") is not None:
            # a tiny stratified entity: its segments' sizes as observed (scaled)
            z = rng.permutation(np.repeat(np.arange(len(E["fixed_counts"])),
                                          np.round(s * np.asarray(E["fixed_counts"], float)).astype(np.int64)))
            par = None
        else:
            z = allocate(th["entities"][e]["pi"], int(round(s * E["rows"])), rng)
            par = None
        df = sample_attributes(S["attributes"][e], th["attributes"][e], z, rng)
        k = np.arange(1, len(z) + 1)
        if E.get("key_prefix") is not None:
            k = np.array([f"{E['key_prefix']}{i:0{E['key_width']}d}" for i in k], dtype=object)
        keys[e] = k
        df.insert(0, pk, k)
        if par is not None:
            df[tree["child_key"]] = pd.array([k[q] if q >= 0 else None for q in par], dtype=object)
        st = E.get("strata")
        if st:
            ss = np.asarray(st["seg_stratum"])[z]
            if st.get("attr"):
                df[st["attr"]] = np.array([l[0] for l in st["labels"]], dtype=object)[ss]
            for pos, colname in enumerate(st["by"]):
                # a stratum is a row of a copied parent: its label is that row's key
                p = st["by_parent"][colname]
                back = {keystr(k): k for k in keys[p]}
                lab = np.array([back.get(l[pos]) for l in st["labels"]], dtype=object)
                df[colname] = lab[ss]
        for name, lk in S["links"].items():
            if lk["child"] != e or lk["child_key"] in df.columns:
                continue
            p = lk["parent"]
            if p not in zs:
                df[lk["child_key"]] = None
                continue
            w = row_weights(th["links"][name]["parent_frailty"], zs[p], rng, lk.get("obs"), lk.get("parent_copied"))
            df[lk["child_key"]] = choose_parents(lk, th["links"][name], z, zs[p], np.asarray(keys[p], dtype=object),
                                                 rng, bb, w)
        out[e], zs[e] = df, z
    for f in S["fact_order"]:
        fk = S["facts"][f]
        tf = th["facts"][f]
        a = fk["anchor"]
        z1 = zs[a]
        w1 = row_weights(tf["frailty1"], z1, rng, fk.get("obs1"), fk.get("anchor_copied"))
        cols = {}
        if fk["model"] == "single":
            lam = np.asarray(tf["lam"], float)
            if fk.get("count_dp"):
                n_rows = dp_counts(fk["count_dp"], z1, lam, rng, bb)
            elif fk.get("min_one"):
                n_rows = 1 + rng.poisson(np.maximum(lam[z1] - 1, 0) * w1)
            else:
                n_rows = rng.poisson(lam[z1] * w1)
            u = np.repeat(np.arange(len(z1)), n_rows)
            v, blk, groups = None, z1[u], z1[u]
        else:
            p2 = fk["partner"]
            z2 = zs[p2]
            w2 = row_weights(tf["frailty2"], z2, rng, fk.get("obs2"), fk.get("partner_copied"))
            if fk["model"] == "occupancy":
                u, v, blk = sample_occupancy(fk, tf, z1, z2, fk["m_real"], rng)
            else:
                u, v, blk = sample_pairs(fk, tf, z1, z2, w1, w2, fk["m_real"], rng, bb)
            cols[fk["partner_key"]] = np.asarray(keys[p2], dtype=object)[v]
            K2 = len(fk["m_real"])
            OFF1, OFF2 = 10**6, 2 * 10**6
            coarse = (OFF1 + blk // K2) if fk["fallback"] == "anchor" else (OFF2 + blk % K2)
            groups = np.where(np.isin(blk, fk["fitted_groups"]), blk, coarse)
            for x, tx in zip(fk["extras"], tf["extras"]):
                zx, kx = zs[x["parent"]], np.asarray(keys[x["parent"]], dtype=object)
                P = np.asarray(tx["P"], float)
                Kx = P.shape[1]
                wx = row_weights(tx["frailty"], zx, rng, x.get("obs"), x.get("parent_copied"))
                memx = [np.flatnonzero(zx == s_) for s_ in range(Kx)]
                avail = np.array([len(m) > 0 and wx[m].sum() > 0 for m in memx], float)
                if x.get("null"):
                    avail[-1] = 1.0
                res = np.empty(len(u), dtype=object)
                for b in np.unique(blk):
                    rows = np.flatnonzero(blk == b)
                    pb = P[b] * avail
                    pb = pb / pb.sum() if pb.sum() > 0 else avail / avail.sum()
                    seg = rng.choice(Kx, size=len(rows), p=pb)
                    for s_ in np.unique(seg):
                        rr, m = rows[seg == s_], memx[s_]
                        if len(m) == 0 or wx[m].sum() <= 0:
                            res[rr] = None
                        else:
                            res[rr] = kx[m[rng.choice(len(m), size=len(rr), p=wx[m] / wx[m].sum())]]
                cols[x["child_key"]] = res
        cols[fk["anchor_key"]] = np.asarray(keys[a], dtype=object)[u]
        given = None
        if fk.get("aux_map") and len(u):
            given = pd.DataFrame({name: out[p].set_index(T[p]["primary_key"])[c].reindex(cols[k_]).to_numpy()
                                  for name, k_, p, c in fk["aux_map"] if k_ in cols})
            if given.shape[1] != len(fk["aux_map"]):
                given = None
        df = sample_attributes(S["attributes"][f], th["attributes"][f], groups, rng, given) if len(u) else \
            pd.DataFrame(columns=S["attributes"][f]["out_cols"])
        for c, vals in cols.items():
            df[c] = vals
        if fk.get("time"):
            ti, tt = fk["time"], tf["time"]
            P = np.asarray(tt["P"], float)
            zrow = z1[u]
            uu = rng.random(len(zrow))
            b = np.zeros(len(zrow), np.int64)
            for j in np.unique(zrow):
                cdf = (P[j] / P[j].sum()).cumsum()
                cdf /= cdf[-1]
                rows = np.flatnonzero(zrow == j)
                b[rows] = cdf.searchsorted(uu[rows], side="right")
            starts = pd.to_datetime(ti["starts"]).to_numpy()
            ends = pd.to_datetime(ti["ends"]).to_numpy()
            span = (ends[b] - starts[b]).astype("timedelta64[s]").astype(np.int64)
            ts = pd.Series(starts[b] + (rng.random(len(b)) * span).astype(np.int64).astype("timedelta64[s]"))
            if not ti["is_dt"]:
                ts = ts.dt.normalize()
            ts = ts.clip(pd.Timestamp(ti["tmin"]), pd.Timestamp(ti["tmax"]))
            ts[rng.random(len(ts)) < tt["null_rate"]] = pd.NaT
            df[ti["col"]] = ts.to_numpy()
        pkf = T[f].get("primary_key")
        if pkf and pkf not in df.columns:
            df.insert(0, pkf, np.arange(1, len(df) + 1))
        out[f] = df
    # text, identifier and sequence columns: placeholders; constants as observed
    for t, cols in S.get("placeholders", {}).items():
        if t in out:
            n = len(out[t])
            for c in cols:
                if c not in out[t].columns:
                    out[t][c] = [f"{c[:1].upper()}{i:07d}" for i in range(1, n + 1)]
    for t, cons in S.get("constants", {}).items():
        if t in out:
            for c, v in cons.items():
                out[t][c] = v

    def offset_keys(df, o):
        """Key of the parent row whose order value is the base row's plus the offset."""
        p = out[o["parent"]]
        pk = T[o["parent"]]["primary_key"]
        if o["unit"] == "day":
            num = (pd.to_datetime(p[o["order_by"]], errors="coerce") - pd.Timestamp(0)) / pd.Timedelta(days=1)
        else:
            num = pd.to_numeric(p[o["order_by"]], errors="coerce")
        num = np.round(num.to_numpy(dtype=float), 6)
        by_key = dict(zip(p[pk].map(keystr), num))
        by_num = {}
        for k, v in zip(p[pk].to_numpy(dtype=object), num):
            if np.isfinite(v):
                by_num.setdefault(v, k)
        base = np.array([by_key.get(keystr(k), np.nan) for k in df[o["offset_from"]]], dtype=float)
        tv = np.round(base + pd.to_numeric(df[o["offset_column"]], errors="coerce").to_numpy(dtype=float), 6)
        return np.array([by_num.get(v) if np.isfinite(v) else None for v in tv], dtype=object)

    def computed(final=True):
        """Offset keys, lookups and derived columns, parents first.  A column that
        reads an aggregate is computed on the pass after the aggregates (final)."""
        for t in S["table_order"]:
            if t not in out or S["tables"][t].get("copied"):
                continue
            df = out[t]
            for q in S.get("sequences", {}).get(t, []):     # line numbers start.. within their key
                if q["within"] in df.columns:
                    key = df[q["within"]].map(keystr).fillna("\x00null").to_numpy(dtype=object)
                    df[q["column"]] = pd.Series(key).groupby(key, sort=False).cumcount().to_numpy() + q["start"]
            for o in S.get("offsets", {}).get(t, []):
                if o["offset_from"] in df.columns and o["offset_column"] in df.columns:
                    df[o["child_key"]] = offset_keys(df, o)
            res = resolver(df, out, T, rels, t)
            for lk in S["lookups"].get(t, []):
                try:
                    v = res(lk["path"]).to_numpy(dtype=object)
                    if lk.get("map") is not None:          # a code mapped through a fixed table
                        m = {str(k): x for k, x in lk["map"].items()}
                        v = np.array([None if keystr(k) is None else m.get(keystr(k), lk.get("default"))
                                      for k in v], dtype=object)
                    df[lk["column"]] = v
                except Exception as e:  # noqa: BLE001
                    if final:
                        notes.append(f"{t}.{lk['column']}: lookup {lk['path']} not evaluated ({e!r})")
            for dv in S["derive"].get(t, []):
                try:
                    _, v = derive_values(df, f"{dv['column']} = {dv['expr']}", res, dv.get("type"))
                    if dv.get("decimals") is not None and dv.get("type") == "float":
                        v = np.round(pd.to_numeric(pd.Series(np.asarray(v, dtype=object)), errors="coerce")
                                     .to_numpy(dtype=float), dv["decimals"])
                    df[dv["column"]] = v
                except Exception as e:  # noqa: BLE001
                    if final:
                        notes.append(f"{t}.{dv['column']}: expression {dv['expr']} not evaluated ({e!r})")
            out[t] = df
    computed(final=not S["aggregates"])
    for ag in S["aggregates"]:
        p, child, col, agg, of = ag["table"], ag["from"], ag["column"], ag["agg"], ag.get("of")
        if agg not in AGGREGATES:          # names reach getattr below: only the known ones
            raise SystemExit(f"{p}.{col}: unknown aggregate {agg!r} (one of {', '.join(AGGREGATES)})")
        if p not in out or child not in out:
            continue
        ch = out[child]
        ck = ch[ag["child_key"]].map(keystr)
        x = None if of is None else (ch[of] if of in ch.columns else
                                     pd.Series(evaluate(of, ch, out, T, rels, child), index=ch.index))
        dates = None
        if x is not None and agg in ("min", "max") and dateish(pd.Series(np.asarray(x, dtype=object))):
            dates = pd.to_datetime(pd.Series(np.asarray(x, dtype=object)), errors="coerce", format="mixed")
        if agg == "count_distinct":
            v = pd.Series(np.asarray(x, dtype=object), index=ck.to_numpy()).groupby(level=0).nunique()
        elif dates is not None:                  # first / last dates, compared as dates
            v = getattr(pd.Series(dates.to_numpy(), index=ck.to_numpy()).groupby(level=0), agg)()
        else:
            vals = np.ones(len(ch)) if x is None else pd.to_numeric(pd.Series(np.asarray(x, dtype=object)),
                                                                   errors="coerce").to_numpy(dtype=float)
            g = pd.Series(vals, index=ck.to_numpy()).groupby(level=0)
            v = g.count() if agg == "count" else g.size().gt(0) if agg == "exists" else getattr(g, agg)()
        pkk = out[p][T[p]["primary_key"]].map(keystr)
        arr = pkk.map(v)
        if agg in ("count", "count_distinct", "sum"):
            arr = arr.fillna(0)
        elif agg == "exists":
            arr = arr.fillna(False)
        if dates is not None:
            dv = pd.to_datetime(arr)
            arr = (dv.dt.strftime("%Y-%m-%d") if T[p]["types"].get(col) == "date" else dv).astype(object) \
                .where(dv.notna(), None)
        out[p][col] = arr.to_numpy()
    if S["aggregates"]:
        computed()                       # columns defined from an aggregate
    hidden = set(S.get("hidden", []))
    for t, df in out.items():
        miss = [c for c in T.get(t, {}).get("columns", []) if c not in df.columns]
        if miss and t not in hidden:
            notes.append(f"{t}: {', '.join(miss)} not generated by the replica (left empty)")
            for c in miss:
                df[c] = None
    generate.notes = sorted(set(notes))
    return {t: df[[c for c in T[t]["columns"] if c in df.columns]] for t, df in out.items()
            if t in T and t not in hidden}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--out", default="replica")
    ap.add_argument("--draw", default="mean", help="index of the posterior draw, or 'mean'")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--reference", help="folder with the copied reference tables as CSV")
    a = ap.parse_args(argv)
    model = json.loads(Path(a.model).read_text())
    tables = generate(model, a.draw, a.seed, a.scale, a.reference)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for t, df in tables.items():
        df.to_csv(table_path(a.out, t), index=False)
    print(f"wrote {len(tables)} tables to {a.out}: " + ", ".join(f"{t} {len(df):,}" for t, df in tables.items()))
    for n in getattr(generate, "notes", []):
        print(f"  note: {n}")


if __name__ == "__main__":
    main()
