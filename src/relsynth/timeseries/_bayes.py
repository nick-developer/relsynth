# SPDX-License-Identifier: Apache-2.0
"""Maximum a posteriori fits with a Laplace approximation of the posterior,
shared by the event-process models (relsynth.timeseries.hawkes, .states).

A parameter block theta of a segment has a Normal prior N(mean, diag(sd^2))
(centred on the pooled fit for a segment, Prop. 1202) and a log likelihood
that is smooth in theta.  The posterior is summarised by its mode and the
inverse Hessian of the negative log posterior at the mode, so a dataset draws
theta ~ N(mode, H^-1) (Bernstein-von Mises: the posterior is asymptotically
this Normal; van der Vaart 1998, Thm 10.1).  The curvature is bounded below by
the prior's, so a flat likelihood falls back to the prior and never to an
improper draw.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def map_laplace(loglik, mean, sd, x0=None, step=1e-3):
    """(mode, covariance, converged) of the posterior loglik(theta) + log N(theta;
    mean, diag(sd^2)).  loglik may return -inf or NaN outside its support."""
    mean = np.asarray(mean, dtype=float)
    sd = np.broadcast_to(np.asarray(sd, dtype=float), mean.shape).copy()
    p = len(mean)

    def nlp(th):
        v = -loglik(th) + 0.5 * np.sum(((th - mean) / sd) ** 2)
        return float(v) if np.isfinite(v) else 1e300
    res = minimize(nlp, mean.copy() if x0 is None else np.asarray(x0, float), method="L-BFGS-B")
    th = res.x
    H = np.zeros((p, p))
    e = step
    f0 = nlp(th)
    for i in range(p):
        ei = np.zeros(p)
        ei[i] = e
        H[i, i] = (nlp(th + ei) - 2 * f0 + nlp(th - ei)) / e ** 2
        for j in range(i + 1, p):
            ej = np.zeros(p)
            ej[j] = e
            H[i, j] = H[j, i] = (nlp(th + ei + ej) - nlp(th + ei - ej) - nlp(th - ei + ej)
                                 + nlp(th - ei - ej)) / (4 * e ** 2)
    H = np.where(np.isfinite(H), H, 0.0)
    w, V = np.linalg.eigh((H + H.T) / 2)
    w = np.maximum(w, 1.0 / float(sd.max()) ** 2)          # the prior bounds the curvature from below
    cov = (V / w) @ V.T
    return th, cov, bool(res.success)


def draw(mode, cov, rng):
    """One posterior draw (rng given) or the mode (rng None)."""
    if rng is None:
        return np.asarray(mode, dtype=float).copy()
    return rng.multivariate_normal(mode, cov)
