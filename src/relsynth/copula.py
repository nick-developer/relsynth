"""Two-stage Bayesian Gaussian copula (layer 1 algorithm 2).

Stage 1 (marginals.py) fixes latent bin bounds from full-data marginals.
Stage 2 (here) samples the latent correlation C from a row sample:
    z_i ~ N(0, C) truncated to its bins,  C ~ LKJ(eta)
by data augmentation: truncated-normal updates of Z, then random-walk
Metropolis on the C-vine canonical partial correlations (atanh scale).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr, ndtri


def vine_chol(w, p):
    """Cholesky factor of a correlation matrix from C-vine CPCs (LKJ 2009)."""
    L = np.zeros((p, p))
    L[0, 0] = 1.0
    k = 0
    W = np.zeros((p, p))
    for j in range(p - 1):
        for i in range(j + 1, p):
            W[i, j] = w[k]
            k += 1
    for i in range(1, p):
        ss = 0.0
        for j in range(i):
            L[i, j] = W[i, j] * np.sqrt(max(1.0 - ss, 0.0))
            ss += L[i, j] ** 2
        L[i, i] = np.sqrt(max(1.0 - ss, 1e-300))
    return L


def chol_to_cpc(C):
    """Inverse of vine_chol: CPCs w from a correlation matrix."""
    p = C.shape[0]
    L = np.linalg.cholesky(C)
    w = []
    for j in range(p - 1):
        for i in range(j + 1, p):
            ss = (L[i, :j] ** 2).sum()
            w.append(L[i, j] / np.sqrt(max(1 - ss, 1e-12)))
    return np.clip(np.array(w), -0.999, 0.999)


def rtruncnorm(mu, sigma, lo, hi, rng):
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    flip = a > 0
    a2 = np.where(flip, -b, a)
    b2 = np.where(flip, -a, b)
    pa, pb = ndtr(a2), ndtr(b2)
    u = pa + (pb - pa) * rng.random(mu.shape)
    t = ndtri(np.clip(u, 1e-300, 1 - 1e-16))
    t = np.clip(t, a2, b2)
    t = np.where(np.isfinite(t), t, np.where(np.isfinite(a2), a2, b2))
    t = np.where(flip, -t, t)
    return mu + sigma * t


def nearest_corr(C):
    C = (C + C.T) / 2
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-4)
    C = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def split_rhat(trace):
    x = np.asarray(trace, dtype=float)
    if len(x) < 8:
        return float("nan")
    h = len(x) // 2
    chains = np.stack([x[:h], x[h:2 * h]])
    m = chains.mean(axis=1)
    W = chains.var(axis=1, ddof=1).mean()
    B = h * m.var(ddof=1)
    if W <= 0:
        return 1.0
    var = (h - 1) / h * W + B / h
    return float(np.sqrt(var / W))


def batch_mcse(x):
    """Monte Carlo standard error of the mean of a chain by non-overlapping batch
    means with batch size floor(sqrt(N)) (consistent under polynomial ergodicity;
    Jones, Haran, Caffo & Neath 2006; Flegal, Haran & Jones 2008)."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    b = int(np.sqrt(N))
    a = N // b if b else 0
    if a < 2:
        return float("nan")
    m = x[: a * b].reshape(a, b).mean(axis=1)
    return float(np.sqrt(b * m.var(ddof=1) / N))


MCSE_Z = 0.01          # fixed-width tolerance on the Fisher-z scale of a correlation
MCSE_MIN_DRAWS = 100   # the batch-means estimate needs about 10 batches of 10
RARE_SHARE = 0.01      # a column kept independent within a sample changes its joint law by <= 2%


def chain_converged(draws, trace_ld, rhat_target, eps=MCSE_Z):
    """Run-length rule for a chain of correlation matrices.  A monitored scalar
    passes when its split R-hat is at most rhat_target (Gelman et al., BDA3
    sec. 11.4) or when its Monte Carlo standard error is below a fixed width
    (the fixed-width rule of Jones et al. 2006): eps on the Fisher-z scale for
    each correlation (atanh, so that near +-1 eps is a relative change of about
    2 eps in the conditional variance 1 - rho^2), and 2 p eps for log det C
    (the sum of p log conditional variances).  The second branch is what lets a
    nearly deterministic pair (rho = -0.999, posterior sd 1e-4) pass: its R-hat
    is scale-free and large for a drift of 1e-4, which no synthetic value can
    show.  The standard-error branch needs at least MCSE_MIN_DRAWS kept draws, so
    that the batch means are not computed on a chain shorter than its
    autocorrelation.  Returns (ok, diagnostics)."""
    if len(draws) < 8:
        return True, {}
    D = np.stack(draws)
    p = D.shape[1]
    iu = np.triu_indices(p, 1)
    R = D[:, iu[0], iu[1]]
    Zt = np.arctanh(np.clip(R, -1 + 1e-12, 1 - 1e-12))
    rh = np.array([split_rhat(R[:, j]) for j in range(R.shape[1])])
    mz = np.array([batch_mcse(Zt[:, j]) for j in range(R.shape[1])])
    rld, mld = split_rhat(trace_ld), batch_mcse(trace_ld)
    long = len(D) >= MCSE_MIN_DRAWS
    bad = ~((rh <= rhat_target) | (long & (mz <= eps)))
    bad_ld = not (rld <= rhat_target or (long and mld <= 2 * p * eps))
    worst = np.argsort(-np.nan_to_num(rh * bad, nan=0.0))[: min(3, int(bad.sum()))]
    return (not bad.any() and not bad_ld), {"mcse_z_max": float(np.nanmax(mz)) if len(mz) else 0.0,
                                            "unconverged_entries": int(bad.sum()) + int(bad_ld),
                                            "unconverged_pairs": [[int(iu[0][k]), int(iu[1][k])] for k in worst]}


class GaussianCopula:
    def __init__(self, p):
        self.p = p
        self.draws = [np.eye(p)]
        self.info = {}

    def fit(self, lo, hi, rng, eta=2.0, iters=120, burn_in=60, mh_steps=15, mode="bayes",
            max_iters=None, rhat_target=1.1, min_dep_rows=0):
        """max_iters: if the split R-hat of the kept draws exceeds rhat_target after
        `iters` iterations, the chain is continued in blocks of (iters - burn_in)
        up to max_iters; the first half of all post-burn-in draws is then discarded
        as further burn-in (run-length control by convergence diagnostics, as in
        Gelman et al., BDA3 sec. 11.4).  Every kept draw is still a state of the
        same Markov chain, so the target is unchanged.  min_dep_rows: a column whose
        non-modal bins hold fewer rows of this sample, and at most RARE_SHARE of it,
        is kept independent (the rule of marginals.min_dependence_rows applied to the
        sample the copula is fitted on: within a segment a column can be nearly
        constant although it is not in the whole table; the joint law changes by at
        most 2 RARE_SHARE in total variation)."""
        n, p = lo.shape
        self.n = n
        if p < 2 or n < 3:                         # nothing to correlate
            self.draws = [np.eye(p)]
            self.info = {"mode": "trivial", "n": n, "p": p}
            return self
        # columns without information in this sample (a single bin, e.g. a level that
        # is constant within a segment) have unidentified correlations: keep them
        # independent instead of letting the chain wander under the prior
        live = ~np.all(np.isneginf(lo) & np.isposinf(hi), axis=0) if n else np.zeros(p, bool)
        live &= np.array([np.unique(np.round(lo[:, j], 9)).size > 1 or np.unique(np.round(hi[:, j], 9)).size > 1
                          for j in range(p)]) if n else live
        if min_dep_rows and n:
            for j in range(p):
                key = np.round(np.where(np.isfinite(lo[:, j]), lo[:, j], -50), 9) * 1e3 \
                    + np.round(np.where(np.isfinite(hi[:, j]), hi[:, j], 50), 9)
                cnt = np.unique(key, return_counts=True)[1]
                rare = n - cnt.max()
                if rare < min_dep_rows and rare <= RARE_SHARE * n:
                    live[j] = False
        if p >= 2 and live.sum() < p:
            sub = GaussianCopula(int(live.sum())).fit(lo[:, live], hi[:, live], rng, eta, iters, burn_in, mh_steps, mode,
                                                      max_iters, rhat_target, min_dep_rows) \
                if live.sum() >= 2 else None
            self.draws = []
            for D in (sub.draws if sub else [np.eye(1)]):
                C = np.eye(p)
                if sub:
                    C[np.ix_(live, live)] = D
                self.draws.append(C)
            self.info = dict(sub.info if sub else {"mode": "trivial"}, n=n, p=p, dropped_constant=int((~live).sum()))
            return self
        plo, phi = ndtr(lo), ndtr(hi)
        Z = ndtri(np.clip((plo + phi) / 2, 1e-12, 1 - 1e-12))
        Z = np.clip(Z, lo, hi)
        # point estimate from conditional means under independence
        dens = np.maximum(phi - plo, 1e-300)
        Zm = (np.exp(-0.5 * np.where(np.isfinite(lo), lo, -40) ** 2)
              - np.exp(-0.5 * np.where(np.isfinite(hi), hi, 40) ** 2)) / np.sqrt(2 * np.pi) / dens
        C0 = nearest_corr(np.corrcoef(Zm, rowvar=False)) if np.all(Zm.std(0) > 0) else np.eye(p)
        if mode == "fast":
            self.draws = [C0]
            self.info = {"mode": "fast", "n": n, "p": p}
            return self
        if mode == "rank":
            return self._fit_rank(lo, hi, Z, rng, iters, burn_in, max_iters, rhat_target)
        d = p * (p - 1) // 2
        levels = np.concatenate([[j] * (p - 1 - j) for j in range(p - 1)])
        bshape = eta + (p - 1 - (levels + 1)) / 2.0
        y = np.arctanh(chol_to_cpc(C0))

        def logpost(yv, R):
            w = np.tanh(yv)
            L = vine_chol(w, p)
            logdet = 2 * np.log(np.diag(L)).sum()
            A = solve_triangular(L, R, lower=True)
            ll = -0.5 * n * logdet - 0.5 * (A ** 2).sum()
            lp = ((bshape - 1) * (np.log1p(w) + np.log1p(-w))).sum()
            jac = np.log1p(-w ** 2).sum()
            return ll + lp + jac

        h = 2.38 / np.sqrt(d * max(n, 1))
        prop_L = np.eye(d)            # adaptive Metropolis (Haario et al. 2001): after half of
        ys = []                       # the burn-in the proposal follows the draws' covariance
        draws, acc_hist, trace_ld = [], [], []
        max_iters = max(iters, max_iters or iters)
        t, block = 0, max(iters - burn_in, 8)
        iu = np.triu_indices(p, 1)

        def converged():
            return chain_converged(draws, trace_ld, rhat_target)[0]
        all_d, all_ld = [], []
        while True:
            if t >= iters and (t - iters) % block == 0:
                # after an extension the first half of all post-burn-in draws is
                # discarded as extra burn-in and the second half is diagnosed and kept
                k = len(all_d) // 2 if t > iters else 0
                draws, trace_ld = all_d[k:], all_ld[k:]
                if t >= max_iters or converged():
                    break
            t += 1
            L = vine_chol(np.tanh(y), p)
            C = L @ L.T
            Om = np.linalg.inv(C)
            for j in range(p):
                mu = -(Z @ Om[:, j] - Z[:, j] * Om[j, j]) / Om[j, j]
                Z[:, j] = rtruncnorm(mu, 1 / np.sqrt(Om[j, j]), lo[:, j], hi[:, j], rng)
            S = Z.T @ Z + 1e-8 * np.eye(p)
            R = np.linalg.cholesky(S)
            cur = logpost(y, R)
            acc = 0
            for _ in range(mh_steps):
                y2 = y + h * (prop_L @ rng.standard_normal(d))
                new = logpost(y2, R)
                if np.log(rng.random()) < new - cur:
                    y, cur = y2, new
                    acc += 1
            a = acc / mh_steps
            acc_hist.append(a)
            if t <= burn_in:
                ys.append(y.copy())
                if t == burn_in // 2 + 1 and len(ys) > d + 2:
                    cov = np.cov(np.array(ys[len(ys) // 3:]).T) + 1e-10 * np.eye(d)
                    sc = np.sqrt(np.mean(np.diag(cov)))
                    prop_L = np.linalg.cholesky(cov / sc ** 2)
                    h = 2.38 / np.sqrt(d) * sc
                h *= np.exp(a - 0.3)
            else:
                L = vine_chol(np.tanh(y), p)
                Cd = L @ L.T
                all_d.append(Cd)
                all_ld.append(np.linalg.slogdet(Cd)[1])
        self.draws = draws or [C0]
        self.iterations = t
        R = np.stack(self.draws)[:, iu[0], iu[1]] if len(self.draws) > 8 else None
        self.info = {"mode": "bayes", "n": n, "p": p, "iterations": t, "kept_draws": len(self.draws),
                     "mh_acceptance": float(np.mean(acc_hist[burn_in:])) if iters > burn_in else None,
                     "rhat_logdet": split_rhat(trace_ld),
                     "rhat_max_entry": float(max(split_rhat(R[:, j]) for j in range(R.shape[1]))) if R is not None else None,
                     # practical stability: largest change of a correlation between chain halves
                     "drift_max": (float(np.abs(R[: len(R) // 2].mean(0) - R[len(R) // 2:].mean(0)).max())
                                   if R is not None else None),
                     "posterior_sd_max": float(np.std(np.stack(self.draws), axis=0).max())}
        ok, diag = chain_converged(draws, trace_ld, rhat_target)
        self.info.update(diag, converged=bool(ok))
        return self

    def _fit_rank(self, lo, hi, Z, rng, iters, burn_in, max_iters, rhat_target):
        """Extended rank likelihood with parameter expansion (Hoff 2007, sbgcop).

        Only the order of the latent bins is used: rows in a lower bin of column j
        must have a lower z_j.  This likelihood is invariant to increasing maps of
        each latent coordinate, in particular to rescaling, so the model can be
        expanded to an unrestricted covariance Sigma = D C D with the conjugate
        prior Sigma ~ inverse-Wishart(p + 1, I), which induces uniform marginal
        priors on every correlation (Barnard, McCulloch & Meng 2000).  The Gibbs
        cycle is
            z_ij | z_-ij, Sigma ~ N(conditional mean, conditional var) truncated to
                                  (max z of lower bins, min z of higher bins),
            Sigma | Z ~ inverse-Wishart(p + 1 + n, I + Z'Z),
        and the draws are reported as C = D^-1 Sigma D^-1.  Rows of alternate bins
        are updated as blocks (they are conditionally independent given the rest),
        and the free scale of each coordinate (the expansion) is what removes the
        slow mixing of the fixed-threshold sampler on coarse discrete columns."""
        n, p = lo.shape
        # bin index of every row (bins are ordered; nulls are the lowest bin)
        lev = np.zeros((n, p), dtype=np.int64)
        nlev = np.zeros(p, dtype=np.int64)
        for j in range(p):
            key = np.round(np.where(np.isfinite(lo[:, j]), lo[:, j], -50), 9) * 1e3 \
                + np.round(np.where(np.isfinite(hi[:, j]), hi[:, j], 50), 9)
            _, lev[:, j] = np.unique(key, return_inverse=True)
            nlev[j] = lev[:, j].max() + 1
        # start from the normal scores of the bins' mid-ranks, so that the whole
        # chain is a function of the bin order alone (exact invariance to any
        # increasing transformation of a column, not only in distribution)
        Z = np.empty((n, p))
        for j in range(p):
            cnt = np.bincount(lev[:, j], minlength=nlev[j]).astype(float)
            mid = (np.cumsum(cnt) - cnt / 2) / n
            Z[:, j] = ndtri(np.clip(mid, 1e-12, 1 - 1e-12))[lev[:, j]]
        nu0 = p + 1
        S0 = np.eye(p)
        Sig = np.eye(p)
        max_iters = max(iters, max_iters or iters)
        block = max(iters - burn_in, 8)
        draws, trace_ld = [], []
        iu = np.triu_indices(p, 1)

        def converged():
            return chain_converged(draws, trace_ld, rhat_target)[0]

        t = 0
        all_d, all_ld = [], []
        while True:
            if t >= iters and (t - iters) % block == 0:
                k = len(all_d) // 2 if t > iters else 0       # as in the bayes mode above
                draws, trace_ld = all_d[k:], all_ld[k:]
                if t >= max_iters or converged():
                    break
            t += 1
            Om = np.linalg.inv(Sig)
            for j in range(p):
                if nlev[j] < 2:
                    continue
                sd = 1 / np.sqrt(Om[j, j])
                for parity in (0, 1):
                    rows = np.flatnonzero(lev[:, j] % 2 == parity)
                    if not len(rows):
                        continue
                    zj = Z[:, j]
                    L_ = lev[:, j]
                    # bounds from the neighbouring bins (which are not updated in this block)
                    mx = np.full(nlev[j], -np.inf)
                    mn = np.full(nlev[j], np.inf)
                    np.maximum.at(mx, L_, zj)
                    np.minimum.at(mn, L_, zj)
                    lower = np.concatenate([[-np.inf], mx[:-1]])[L_[rows]]
                    upper = np.concatenate([mn[1:], [np.inf]])[L_[rows]]
                    mu = -(Z[rows] @ Om[:, j] - Z[rows, j] * Om[j, j]) / Om[j, j]
                    Z[rows, j] = rtruncnorm(mu, np.full(len(rows), sd), lower, upper, rng)
            S = S0 + Z.T @ Z
            # inverse-Wishart(nu0 + n, S) draw via Bartlett decomposition of its inverse
            Lw = np.linalg.cholesky(np.linalg.inv(S))
            df = nu0 + n
            A = np.zeros((p, p))
            A[np.diag_indices(p)] = np.sqrt(rng.chisquare(df - np.arange(p)))
            A[np.tril_indices(p, -1)] = rng.standard_normal(p * (p - 1) // 2)
            W = Lw @ A
            Sig = np.linalg.inv(W @ W.T)
            if t > burn_in:
                dg = np.sqrt(np.diag(Sig))
                Cd = Sig / np.outer(dg, dg)
                all_d.append(Cd)
                all_ld.append(np.linalg.slogdet(Cd)[1])
        self.draws = draws or [np.eye(p)]
        R = np.stack(self.draws)[:, iu[0], iu[1]] if len(self.draws) > 8 else None
        self.info = {"mode": "rank", "n": n, "p": p, "iterations": t, "kept_draws": len(self.draws),
                     "mh_acceptance": None,
                     "rhat_logdet": split_rhat(trace_ld),
                     "rhat_max_entry": float(max(split_rhat(R[:, j]) for j in range(R.shape[1]))) if R is not None else None,
                     "drift_max": (float(np.abs(R[: len(R) // 2].mean(0) - R[len(R) // 2:].mean(0)).max())
                                   if R is not None else None),
                     "posterior_sd_max": float(np.std(np.stack(self.draws), axis=0).max())}
        ok, diag = chain_converged(draws, trace_ld, rhat_target)
        self.info.update(diag, converged=bool(ok))
        return self

    def mean(self):
        return nearest_corr(np.mean(np.stack(self.draws), axis=0))

    def draw(self, rng):
        cur = getattr(self, "current", None)
        if cur is not None:
            return cur
        return self.draws[rng.integers(len(self.draws))]

    def set_current(self, rng, mode="draw"):
        """Fix the correlation matrix of one dataset: one posterior draw, or the
        posterior mean in mean mode (every row of the dataset uses it)."""
        self.current = self.mean() if mode == "mean" else self.draws[rng.integers(len(self.draws))]

    def sample_conditional(self, za, ai, ri, rng, C=None):
        """Draw the latent coordinates ri given observed latent values za at ai."""
        C = nearest_corr(self.draw(rng) if C is None else C)
        Caa = C[np.ix_(ai, ai)]
        Cra = C[np.ix_(ri, ai)]
        Crr = C[np.ix_(ri, ri)]
        B = np.linalg.solve(Caa, Cra.T).T
        S = Crr - B @ Cra.T
        L = np.linalg.cholesky((S + S.T) / 2 + 1e-9 * np.eye(len(ri)))
        return za @ B.T + rng.standard_normal((len(za), len(ri))) @ L.T

    def sample(self, M, rng, C=None):
        C = self.draw(rng) if C is None else C
        L = np.linalg.cholesky(nearest_corr(C) if self.p > 1 else np.eye(self.p))
        return rng.standard_normal((M, self.p)) @ L.T
