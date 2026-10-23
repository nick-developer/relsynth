"""Compiled kernels (v0.8).

Each kernel computes exactly what the numpy expression it replaces computes:
the same IEEE operations on each element in the same order, so results are
bit-identical (checked in checks C69).  Numba is optional; without it the numpy
expression itself is used.  No kernel draws random numbers.
"""
from __future__ import annotations

import numpy as np

try:                                    # pragma: no cover - depends on the installation
    import numba
    HAVE_NUMBA = True
except Exception:                       # noqa: BLE001
    HAVE_NUMBA = False


def _gauss_ll_numpy(L, X, mu, s2, a):
    for j in range(X.shape[1]):
        x = X[:, j][:, None]
        L += a[:, j] - 0.5 * (x - mu[:, j]) ** 2 / s2[:, j]
    return L


if HAVE_NUMBA:
    @numba.njit(cache=False, nogil=True)
    def _gauss_ll_nb(L, X, mu, s2, a):
        n, K = L.shape
        for j in range(X.shape[1]):
            for i in range(n):
                x = X[i, j]
                for k in range(K):
                    d = x - mu[k, j]
                    L[i, k] = L[i, k] + (a[k, j] - (0.5 * (d * d)) / s2[k, j])
        return L


def gauss_ll(L, X, mu, s2):
    """L[i, k] += sum_j log N(X[i, j]; mu[k, j], s2[k, j]), column by column, as
    L += -0.5 log(2 pi s2_j) - 0.5 (x - mu_j)^2 / s2_j."""
    if X.shape[1] == 0 or L.size == 0:
        return L
    # the log term exactly as the numpy expression forms it (one column at a time)
    a = np.stack([-0.5 * np.log(2 * np.pi * s2[:, j]) for j in range(X.shape[1])], 1)
    if HAVE_NUMBA and L.dtype == np.float64:
        return _gauss_ll_nb(L, np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(mu),
                            np.ascontiguousarray(s2), a)
    return _gauss_ll_numpy(L, X, mu, s2, a)


def count_table(z, x, K, V):
    """cnt[k, v] = #{i: z_i = k, x_i = v} as floats (np.add.at(cnt, (z, x), 1))."""
    return np.bincount(np.asarray(z, np.int64) * V + np.asarray(x, np.int64), minlength=K * V)[:K * V] \
        .reshape(K, V).astype(float)


def dirichlet_rows(rng, alpha):
    """np.stack([rng.dirichlet(a) for a in alpha]) with one call: numpy's
    Dirichlet sampler (every concentration >= 0.1) draws standard gammas in order
    and multiplies by the reciprocal of their running sum, which is reproduced
    exactly, so the draws and the random stream are identical."""
    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim != 2 or alpha.shape[0] == 0 or not np.isfinite(alpha).all() \
            or (alpha.max(axis=1) < 0.1).any() or (alpha <= 0).any():
        return np.stack([rng.dirichlet(a) for a in alpha]) if len(alpha) else np.zeros(alpha.shape)
    g = rng.standard_gamma(alpha)
    acc = np.cumsum(g, axis=1)[:, -1]
    return g * (1.0 / acc)[:, None]
