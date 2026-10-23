"""One posterior draw (or the posterior mean) of every parameter, per synthetic dataset.

A synthetic dataset is a draw from the posterior predictive distribution

    theta ~ p(theta | data),    D* ~ p(D* | theta),

so every row of one dataset must use the same theta (Rubin 1993; Raghunathan,
Reiter & Rubin 2003).  The engine registers every parameter block it fits; at
the start of each `sample` all blocks are drawn once from a parameter stream,
and the rows are then drawn from a separate data stream.  Draws that the
samplers used to repeat per call (copula matrix, Bayesian-bootstrap weights of
count laws and of pooled tail levels) are fixed for the run.

`mode: mean` replaces each draw by the posterior mean (Gamma, Beta, Dirichlet
means; the Bayesian bootstrap by the empirical distribution; the copula by its
posterior mean correlation) and each spike-and-slab indicator by its posterior
median, i.e. the structural zero is kept exactly when its posterior probability
exceeds 1/2.  This is the plug-in synthesis of Reiter & Kinney (2012): stable
from run to run, for test and development data.
"""
from __future__ import annotations

import numpy as np

MODES = ("draw", "mean")


class MeanRNG:
    """Stands in for a numpy Generator inside parameter-draw code: the
    distributions that parameters are drawn from return their means, and uniform
    draws return 1/2 (so a mixture indicator takes its median component).  Any
    other call goes to the wrapped generator."""

    def __init__(self, rng):
        self._rng = rng

    def __getattr__(self, name):
        return getattr(self._rng, name)

    @staticmethod
    def _shape(size, *arrs):
        if size is not None:
            return size
        b = np.broadcast(*[np.asarray(a) for a in arrs]) if arrs else None
        return b.shape if b is not None else ()

    def _out(self, val, size, *arrs):
        shape = self._shape(size, *arrs)
        v = np.broadcast_to(np.asarray(val, dtype=float), shape).copy() if shape != () else float(np.asarray(val))
        return v

    def gamma(self, shape, scale=1.0, size=None):
        return self._out(np.asarray(shape, float) * np.asarray(scale, float), size, shape, scale)

    def standard_gamma(self, shape, size=None):
        return self._out(np.asarray(shape, float), size, shape)

    def beta(self, a, b, size=None):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return self._out(a / (a + b), size, a, b)

    def dirichlet(self, alpha, size=None):
        alpha = np.asarray(alpha, float)
        p = alpha / alpha.sum()
        return np.broadcast_to(p, (size, len(p))).copy() if size else p

    def chisquare(self, df, size=None):
        return self._out(df, size, df)

    def standard_normal(self, size=None):
        return np.zeros(size) if size is not None else 0.0

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._out(loc, size, loc, scale)

    def random(self, size=None):
        return np.full(size, 0.5) if size is not None else 0.5

    def choice(self, a, size=None, replace=True, p=None):
        # the median of a discrete law (used to pick a grid cell of a posterior)
        if p is None or size is not None:
            return self._rng.choice(a, size=size, replace=replace, p=p)
        n = a if np.isscalar(a) else len(a)
        i = int(np.searchsorted(np.cumsum(p), 0.5))
        i = min(i, n - 1)
        return i if np.isscalar(a) else np.asarray(a)[i]


def param_rng(rng, mode):
    return MeanRNG(rng) if mode == "mean" else rng


def run_streams(seed, sample_seed, index):
    """(parameter stream, data stream) of dataset `index` (0-based).  Without a
    sample seed the streams derive from the synthesis seed, so a spec gives the
    same datasets on every run; dataset i uses its own independent streams."""
    base = seed if sample_seed is None else sample_seed
    ss = np.random.SeedSequence([int(base), 7919, int(index)])
    a, b = ss.spawn(2)
    return np.random.default_rng(a), np.random.default_rng(b)


class RunCache:
    """Per-dataset cache of draws that samplers reuse (Bayesian-bootstrap weights
    of count laws and pooled levels): drawn at first use, fixed for the run."""

    def __init__(self, mode="draw"):
        self.mode = mode
        self.store = {}

    def bb(self, key, n, rng):
        """Bayesian-bootstrap weights over n observations (Dirichlet(1,...,1));
        the empirical weights 1/n in mean mode."""
        w = self.store.get(key)
        if w is None or len(w) != n:
            w = np.full(n, 1.0 / n) if self.mode == "mean" else rng.dirichlet(np.ones(n))
            self.store[key] = w
        return w
