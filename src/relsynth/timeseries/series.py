# SPDX-License-Identifier: Apache-2.0
"""Regular series: a value per series and time step (meters, sensors, stores).

A table declares ``series: {entity: <foreign key>, time: <datetime>, values: [..]}``:
each entity row (a meter) owns one series on a regular grid.  Synthetic
entities get new series, not copies of real ones: every real series is
summarised by a parameter vector, the population of parameter vectors is
modelled, and a synthetic series is simulated from a drawn vector.

Model of one value column, on the transformed scale x = log(1 + y) for
non-negative values (else y), for series i at grid step t (Durbin and Koopman
2012, ch. 3; West and Harrison 1997, ch. 8):

    x_it = level_i + day_i[phase(t)] + week_i[weekday(t)] + year_i[month(t)] + e_it
    e_it = sigma_i (lambda_i f_t + sqrt(1 - lambda_i^2) eps_it)
    f_t   ~ AR(p) with unit variance (one common factor: weather, holidays)
    eps_i ~ AR(p) with unit variance, coefficients phi_i (stationary)

before its start s_i the series is 0 (a meter not yet installed).

Per-series parameters theta_i = (level, log sigma, atanh lambda, partial
autocorrelations of eps (atanh), logit start, scores of the day, week and year
profiles on their principal components) are modelled by a Gaussian mixture
with a Normal-inverse-Wishart prior per component (number of components by
BIC, at most 5); a dataset draws the component weights, means and covariances
from their posterior and each synthetic series' theta from its component
(Prop. 1205).  The common factor of a dataset is drawn from its fitted AR law,
so cross-series correlation comes from the loadings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import invwishart

from .calendar import to_days

MAX_COMPONENTS = 5


def infer_step(t_days):
    """Grid step (days) of sorted times: the most common positive difference."""
    d = np.diff(np.unique(t_days))
    d = d[d > 0]
    if not len(d):
        return 1.0
    v, c = np.unique(np.round(d * 86400).astype(np.int64), return_counts=True)
    return float(v[c.argmax()]) / 86400.0


def _pacf_to_ar(r):
    """AR coefficients from partial autocorrelations in (-1, 1) (Durbin-Levinson;
    any such vector gives a stationary AR, Barndorff-Nielsen and Schou 1973)."""
    phi = np.zeros(0)
    for k, rk in enumerate(r):
        phi = np.concatenate([phi - rk * phi[::-1], [rk]]) if k else np.array([rk])
    return phi


def _ar_to_pacf(phi):
    phi = np.asarray(phi, float).copy()
    p = len(phi)
    r = np.zeros(p)
    for k in range(p - 1, -1, -1):
        rk = phi[k]
        r[k] = rk
        if k and abs(rk) < 1:
            phi = (phi[:k] + rk * phi[:k][::-1]) / (1 - rk * rk)
    return np.clip(r, -0.999, 0.999)


def _yule_walker(x, p):
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = len(x)
    if n <= p + 2:
        return np.zeros(p)
    g = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(p + 1)])
    if g[0] <= 0:
        return np.zeros(p)
    R = np.array([[g[abs(i - j)] for j in range(p)] for i in range(p)])
    try:
        return np.linalg.solve(R, g[1:])
    except np.linalg.LinAlgError:
        return np.zeros(p)


def _simulate_ar(phi, T, m, rng, burn=200):
    """m independent unit-variance AR(phi) paths of length T (rows)."""
    p = phi.shape[1] if phi.ndim == 2 else len(phi)
    phi = np.broadcast_to(phi, (m, p))
    # innovation variance for unit marginal variance: 1 - sum phi_k rho_k
    var = np.ones(m)
    for i in range(m):
        var[i] = max(_innov_var(phi[i]), 1e-6)
    e = rng.standard_normal((m, T + burn)) * np.sqrt(var)[:, None]
    x = np.zeros((m, T + burn))
    for t in range(T + burn):
        acc = e[:, t].copy()
        for k in range(1, p + 1):
            if t - k >= 0:
                acc += phi[:, k - 1] * x[:, t - k]
        x[:, t] = acc
    return x[:, burn:]


def _innov_var(phi):
    """Innovation variance of a unit-variance AR(phi): prod (1 - r_k^2) over its
    partial autocorrelations."""
    return float(np.prod(1 - _ar_to_pacf(phi) ** 2))


class _PCA:
    def __init__(self, X, var=0.95, max_k=6):
        self.mean = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        ev = S ** 2 / max(np.sum(S ** 2), 1e-12)
        k = int(np.searchsorted(np.cumsum(ev), var) + 1) if ev.sum() > 0 else 0
        self.k = int(min(k, max_k, X.shape[1] - 1, max(X.shape[0] - 2, 0)))
        self.V = Vt[: self.k]

    def scores(self, X):
        return (X - self.mean) @ self.V.T

    def back(self, Z):
        return self.mean + Z @ self.V


class _Mixture:
    """Gaussian mixture of parameter vectors with NIW posteriors per component."""

    def fit(self, X, rng, max_k=MAX_COMPONENTS):
        n, d = X.shape
        self.d = d
        best = None
        for K in range(1, max(1, min(max_k, n // max(4 * d, 1))) + 1):
            z, ll = self._em(X, K, rng)
            nparam = K * (d + d * (d + 1) / 2) + K - 1
            bic = -2 * ll + nparam * np.log(n)
            if best is None or bic < best[0]:
                best = (bic, K, z)
        _, self.K, z = best
        self.z = z
        self.X = X
        self._posterior()
        self.draw(None)
        return self

    def _em(self, X, K, rng, iters=60):
        n, d = X.shape
        z = rng.integers(0, K, n) if K > 1 else np.zeros(n, np.int64)
        reg = 1e-3 * np.eye(d) * max(np.var(X, 0).mean(), 1e-6)
        ll = -np.inf
        for _ in range(iters):
            w, mu, cov = [], [], []
            for k in range(K):
                m = z == k
                if m.sum() < d + 2:
                    m = np.ones(n, bool)
                w.append(m.mean())
                mu.append(X[m].mean(0))
                cov.append(np.cov(X[m].T).reshape(d, d) + reg)
            w = np.array(w) / np.sum(w)
            L = np.stack([np.log(w[k]) + _mvn_logpdf(X, mu[k], cov[k]) for k in range(K)], 1)
            ll = float(np.logaddexp.reduce(L, 1).sum())
            z_new = L.argmax(1)
            if np.array_equal(z_new, z):
                break
            z = z_new
        return z, ll

    def _posterior(self):
        X, d = self.X, self.d
        m0 = X.mean(0)
        S0 = np.cov(X.T).reshape(d, d) + 1e-6 * np.eye(d)
        self.post = []
        for k in range(self.K):
            Xk = X[self.z == k]
            nk = len(Xk)
            k0, nu0 = 1.0, d + 2.0
            xb = Xk.mean(0)
            Sk = (Xk - xb).T @ (Xk - xb)
            kn, nun = k0 + nk, nu0 + nk
            mn = (k0 * m0 + nk * xb) / kn
            Psi = S0 * (nu0 - d - 1) + Sk + (k0 * nk / kn) * np.outer(xb - m0, xb - m0)
            self.post.append((mn, kn, nun, Psi, nk))
        self.counts = np.array([p[4] for p in self.post], float)

    def draw(self, rng):
        self.w = (self.counts + 1) / (self.counts + 1).sum() if rng is None else rng.dirichlet(self.counts + 1)
        self.mu, self.cov = [], []
        for mn, kn, nun, Psi, _ in self.post:
            if rng is None:
                S = Psi / max(nun - self.d - 1, 1)
                m = mn
            else:
                S = np.atleast_2d(invwishart.rvs(df=nun, scale=Psi, random_state=_seed(rng)))
                m = rng.multivariate_normal(mn, S / kn)
            self.mu.append(m)
            self.cov.append(S)
        return self

    def sample(self, n, rng, q=0.99):
        """n draws, each from its component's Normal truncated to the central q
        ellipsoid (Mahalanobis distance below the chi-square q quantile), so a
        draw never combines the extremes of several parameters."""
        from scipy.stats import chi2
        lim = chi2.ppf(q, self.d)
        c = rng.choice(self.K, size=n, p=self.w)
        out = np.empty((n, self.d))
        for k in range(self.K):
            todo = np.flatnonzero(c == k)
            P = np.linalg.inv(self.cov[k])
            for _ in range(100):
                if not len(todo):
                    break
                x = rng.multivariate_normal(self.mu[k], self.cov[k], size=len(todo))
                dx = x - self.mu[k]
                ok = np.einsum("ij,jk,ik->i", dx, P, dx) <= lim
                out[todo[ok]] = x[ok]
                todo = todo[~ok]
            if len(todo):
                out[todo] = self.mu[k]
        return out


def _seed(rng):
    return int(rng.integers(0, 2 ** 31 - 1))


def _mvn_logpdf(X, mu, cov):
    d = len(mu)
    L = np.linalg.cholesky(cov)
    y = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (y * y).sum(0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)


class SeriesModel:
    """Model of the regular series of one table (one or more value columns)."""

    def __init__(self, ar=2, factor_ar=2, day=True, week=True, year="auto"):
        self.p = int(ar)
        self.pf = int(factor_ar)
        self.use_day, self.use_week, self.use_year = day, week, year

    # ------------------------------------------------------------ fitting
    def fit(self, frame, entity, time, values, rng):
        """frame: the real table; entity, time: its key and time columns;
        values: value columns."""
        t = to_days(frame[time])
        self.values = list(values)
        self.step = infer_step(t[np.isfinite(t)])
        self.t0 = float(np.nanmin(t))
        T = int(round((np.nanmax(t) - self.t0) / self.step)) + 1
        self.T = T
        idx = np.round((t - self.t0) / self.step).astype(np.int64)
        ids, ent = np.unique(frame[entity].astype(str).to_numpy(), return_inverse=True)
        self.n = len(ids)
        steps_per_day = max(int(round(1 / self.step)), 1)
        self.P1 = steps_per_day if (self.use_day and steps_per_day > 1) else 1
        tt = self.t0 + np.arange(T) * self.step
        self.phase = np.floor((tt - np.floor(tt)) * self.P1 + 1e-9).astype(np.int64) % max(self.P1, 1)
        self.weekday = ((np.floor(tt).astype(np.int64) + 3) % 7)
        self.month = np.asarray(pd.to_datetime(np.floor(tt).astype("int64"), unit="D").month) - 1
        span_days = T * self.step
        self.has_week = bool(self.use_week and span_days >= 21)
        self.has_year = bool(span_days >= 400) if self.use_year == "auto" else bool(self.use_year)
        self.cols = {}
        F_all = []
        for c in self.values:
            Y = np.full((self.n, T), np.nan)
            Y[ent, idx] = pd.to_numeric(frame[c], errors="coerce").to_numpy(float)
            self.cols[c] = self._fit_column(Y)
            F_all.append(self.cols[c]["f"])
        self._fit_params(rng)
        return self

    def _fit_column(self, Y):
        n, T = Y.shape
        finite = Y[np.isfinite(Y)]
        nonneg = bool(len(finite) and finite.min() >= 0)
        dec = next((d for d in range(7) if np.allclose(finite[:5000], np.round(finite[:5000], d))), 6) \
            if len(finite) else 6
        X = np.log1p(np.maximum(Y, 0)) if nonneg else Y.copy()
        # start: first step with a nonzero value (leading zeros = not yet active)
        nz = np.isfinite(Y) & (Y != 0)
        start = np.where(nz.any(1), nz.argmax(1), T - 1)
        active = np.arange(T)[None, :] >= start[:, None]
        X = np.where(active, X, np.nan)
        level = np.nanmean(X, 1)
        R = X - level[:, None]
        prof = {}

        def profile(key, P):
            M = np.full((n, P), 0.0)
            for j in range(P):
                v = R[:, key == j]
                with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                    __import__("warnings").simplefilter("ignore", RuntimeWarning)
                    M[:, j] = np.nanmean(v, 1)
            M = np.nan_to_num(M - np.nanmean(M, 1, keepdims=True))
            return M
        if self.P1 > 1:
            prof["day"] = profile(self.phase, self.P1)
            R = R - prof["day"][:, self.phase]
        if self.has_week:
            prof["week"] = profile(self.weekday, 7)
            R = R - prof["week"][:, self.weekday]
        if self.has_year:
            prof["year"] = profile(self.month, 12)
            R = R - prof["year"][:, self.month]
        sigma = np.sqrt(np.nanmean(R ** 2, 1))
        # a floor of 1% (log scale: +-1% of the level) keeps all-zero or constant
        # series from dragging the scale of the parameter model
        sigma = np.where(np.isfinite(sigma), np.maximum(sigma, 0.01), 0.01)
        Z = R / sigma[:, None]
        # common factor: first principal component of the standardised residuals
        Zf = np.nan_to_num(Z)
        U, S, Vt = np.linalg.svd(Zf, full_matrices=False)
        f = Vt[0] * S[0] / np.sqrt(n)
        f = (f - f.mean()) / max(f.std(), 1e-9)
        with np.errstate(all="ignore"):
            lam = np.array([np.nanmean(Z[i] * f) / max(np.nanmean(Z[i] ** 2), 1e-12) ** 0.5 for i in range(n)])
        lam = np.clip(np.nan_to_num(lam), -0.98, 0.98)
        E = (Z - lam[:, None] * f[None, :]) / np.sqrt(1 - lam ** 2)[:, None]
        phi = np.stack([_yule_walker(E[i], self.p) for i in range(n)]) if self.p else np.zeros((n, 0))
        pacf = np.stack([_ar_to_pacf(ph) for ph in phi]) if self.p else np.zeros((n, 0))
        phi_f = _yule_walker(f, self.pf) if self.pf else np.zeros(0)
        return {"nonneg": nonneg, "decimals": dec, "start": start, "level": level, "sigma": sigma, "lam": lam,
                "pacf": pacf, "prof": prof, "f": f, "phi_f": _pacf_to_ar(_ar_to_pacf(phi_f))}

    def _theta(self):
        """Parameter matrix (series x features) and the layout to read it back."""
        blocks, self.layout = [], []
        T = self.T
        for c, m in self.cols.items():
            st = np.clip((m["start"] + 0.5) / T, 1e-4, 1 - 1e-4)
            parts = [("level", m["level"][:, None]), ("log_sigma", np.log(m["sigma"])[:, None]),
                     ("lam", np.arctanh(m["lam"])[:, None]), ("pacf", np.arctanh(m["pacf"])),
                     ("start", np.log(st / (1 - st))[:, None])]
            m["pca"] = {}
            for k, M in m["prof"].items():
                pca = _PCA(M)
                m["pca"][k] = pca
                if pca.k:
                    parts.append((k, pca.scores(M)))
            for name, arr in parts:
                self.layout.append((c, name, arr.shape[1]))
                blocks.append(arr)
        return np.nan_to_num(np.concatenate(blocks, 1))

    def _fit_params(self, rng):
        th = self._theta()
        self.theta_mean = th.mean(0)
        self.theta_sd = np.maximum(th.std(0), 1e-9)
        # synthetic parameters stay inside the range of the real ones (widened by
        # a twentieth of it): a Gaussian tail would give meters with absurd loads
        lo, hi = th.min(0), th.max(0)
        pad = 0.05 * (hi - lo)
        self.theta_lo, self.theta_hi = lo - pad, hi + pad
        self.mix = _Mixture().fit((th - self.theta_mean) / self.theta_sd, rng)

    def redraw(self, rng=None):
        self.mix.draw(rng)
        return self

    # ----------------------------------------------------------- sampling
    def sample(self, n, rng, keys=None):
        """A frame of n synthetic series on the real grid: columns entity index
        (0..n-1, or keys), time (days) and the value columns."""
        th = self.mix.sample(n, rng) * self.theta_sd + self.theta_mean
        th = np.clip(th, self.theta_lo, self.theta_hi)
        T = self.T
        vals = {}
        pos = 0
        parts = {}
        for c, name, w in self.layout:
            parts[(c, name)] = th[:, pos: pos + w]
            pos += w
        tt = self.t0 + np.arange(T) * self.step
        for c, m in self.cols.items():
            level = parts[(c, "level")][:, 0]
            sigma = np.exp(parts[(c, "log_sigma")][:, 0])
            lam = np.tanh(parts[(c, "lam")][:, 0])
            pacf = np.tanh(parts[(c, "pacf")]) if self.p else np.zeros((n, 0))
            st = 1 / (1 + np.exp(-parts[(c, "start")][:, 0]))
            start = np.clip(np.floor(st * T), 0, T - 1).astype(np.int64)
            X = np.repeat(level[:, None], T, 1)
            for k, key in (("day", self.phase), ("week", self.weekday), ("year", self.month)):
                if k in m["prof"]:
                    pca = m["pca"][k]
                    M = pca.back(parts[(c, k)]) if pca.k else np.repeat(pca.mean[None, :], n, 0)
                    X = X + M[:, key]
            f = _simulate_ar(m["phi_f"], T, 1, rng)[0] if len(m["phi_f"]) else rng.standard_normal(T)
            if self.p:
                phi = np.stack([_pacf_to_ar(r) for r in pacf])
                eps = _simulate_ar(phi, T, n, rng)
            else:
                eps = rng.standard_normal((n, T))
            X = X + sigma[:, None] * (lam[:, None] * f[None, :] + np.sqrt(1 - lam ** 2)[:, None] * eps)
            Y = np.expm1(X).clip(0, None) if m["nonneg"] else X
            Y = np.where(np.arange(T)[None, :] >= start[:, None], Y, 0.0)
            vals[c] = np.round(Y, m["decimals"])
        ent = np.repeat(np.arange(n) if keys is None else np.asarray(keys, dtype=object), T)
        out = pd.DataFrame({"__entity__": ent, "__time__": np.tile(tt, n)})
        for c in self.values:
            out[c] = vals[c].reshape(-1)
        return out

    def summary(self):
        return {"series": self.n, "steps": self.T, "step_hours": round(self.step * 24, 4), "day_phases": self.P1,
                "weekly": self.has_week, "yearly": self.has_year, "ar": self.p, "components": self.mix.K,
                "parameters": int(sum(w for *_, w in self.layout))}
