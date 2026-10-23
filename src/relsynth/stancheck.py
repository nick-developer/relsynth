"""Independent verification of the engine's posterior blocks with Stan (v0.8).

Every parameter block of the engine is drawn from a posterior that is either a
closed-form conjugate update or a Markov chain (the copula).  Here each block is
written as a Stan program (the model in its textbook form: prior and likelihood,
nothing else) and sampled with Stan's NUTS sampler; the draws are compared with
the engine's own posterior draws for the same data.  Agreement of the posterior
mean and standard deviation of every parameter, within Monte Carlo error, checks
the engine's algebra and samplers against a general-purpose implementation that
shares no code with it.

Blocks verified:
  rate       Poisson block rate eta_kl: C ~ Poisson(eta E), eta ~ Gamma(a, b)
             (relational.PairRel.draw_params; Gamma(a + C, b + E))
  occupancy  presence probability: S ~ Binomial(T, p), p ~ Beta(a, b)
             (relational.OccRel.draw_params; Beta(a + S, b + T - S))
  levels     category probabilities: counts ~ Multinomial(theta), theta ~ Dirichlet(alpha)
             (marginals.Marginal.fit; Dirichlet(alpha + counts))
  nig        segment attribute model: x ~ N(mu, s2), mu | s2 ~ N(0, s2 / k0), s2 ~ IG(a0, b0)
             (relational.Entity.draw_attr_params)
  copula     latent correlation: z_i ~ N(0, R), R ~ LKJ(eta)
             (copula.GaussianCopula.fit, MH-within-Gibbs on the canonical partial correlations)

Stan is optional (pip install pystan); without it `verify` reports that it was
skipped.  Stan is used for verification only: for the latent-segment model a
Stan fit would have to sum the discrete segments out of every row, and it is
orders of magnitude slower than the conjugate Gibbs sampler (measured in the
v0.8 report).
"""
from __future__ import annotations

import time

import numpy as np

PROGRAMS = {
    "rate": """
data { int<lower=0> C; real<lower=0> E; real<lower=0> a; real<lower=0> b; }
parameters { real<lower=0> eta; }
model { eta ~ gamma(a, b); C ~ poisson(eta * E); }
""",
    "occupancy": """
data { int<lower=0> T; int<lower=0, upper=T> S; real<lower=0> a; real<lower=0> b; }
parameters { real<lower=0, upper=1> p; }
model { p ~ beta(a, b); S ~ binomial(T, p); }
""",
    "levels": """
data { int<lower=1> V; array[V] int<lower=0> y; vector<lower=0>[V] alpha; }
parameters { simplex[V] theta; }
model { theta ~ dirichlet(alpha); y ~ multinomial(theta); }
""",
    "nig": """
data { int<lower=1> N; vector[N] x; real<lower=0> k0; real<lower=0> a0; real<lower=0> b0; }
parameters { real mu; real<lower=0> s2; }
model { s2 ~ inv_gamma(a0, b0); mu ~ normal(0, sqrt(s2 / k0)); x ~ normal(mu, sqrt(s2)); }
""",
    "copula": """
data { int<lower=1> N; int<lower=2> P; array[N] vector[P] z; real<lower=0> eta; }
parameters { cholesky_factor_corr[P] L; }
model { L ~ lkj_corr_cholesky(eta); z ~ multi_normal_cholesky(rep_vector(0, P), L); }
generated quantities { matrix[P, P] R = multiply_lower_tri_self_transpose(L); }
""",
}


def _stan_draws(name, data, seed, n=4000):
    import stan
    post = stan.build(PROGRAMS[name], data=data, random_seed=seed)
    fit = post.sample(num_chains=4, num_samples=n // 4, num_warmup=500)
    return fit


def _compare(label, stan_draws, eng_draws, ess=None):
    """Posterior means and sds of two sets of draws; z = difference of means over
    its Monte Carlo standard error (independent draws; Stan's from 4 chains are
    thinned to its effective sample size when given)."""
    s, e = np.asarray(stan_draws, float), np.asarray(eng_draws, float)
    ns = ess or len(s)
    se = np.sqrt(s.var() / ns + e.var() / len(e))
    z = float(abs(s.mean() - e.mean()) / max(se, 1e-300))
    sd_ratio = float(e.std() / max(s.std(), 1e-300))
    return {"parameter": label, "stan_mean": float(s.mean()), "engine_mean": float(e.mean()),
            "stan_sd": float(s.std()), "engine_sd": float(e.std()), "z_mean": round(z, 2),
            "sd_ratio": round(sd_ratio, 3), "ok": bool(z < 4 and 0.85 < sd_ratio < 1.15)}


def verify(seed=11, n=4000):
    """Run every block; returns {"available": bool, "blocks": [...], "seconds": ...}."""
    try:
        import stan  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"Stan not installed ({e!r})", "blocks": []}
    from .copula import GaussianCopula
    from .marginals import Codec, Marginal
    from .relational import NIG, Entity, OccRel, PairRel
    import pandas as pd
    rng = np.random.default_rng(seed)
    out, t0 = [], time.time()

    # ---- rate: the engine's own PairRel.draw_params on a one-block relation
    class _E:
        def __init__(self, n_, K=1):
            self.n, self.K, self.z = n_, K, np.zeros(n_, np.int64)

        def counts(self):
            return np.bincount(self.z, minlength=self.K).astype(float)
    e1, e2 = _E(40), _E(25)
    u = rng.integers(0, 40, 90)
    v = rng.integers(0, 25, 90)
    rel = PairRel("r", "fact", e1, e2, u, v, np.ones(90))
    eng = []
    for _ in range(n):
        rel.draw_params(rng)
        eng.append(rel.eta[0, 0])
    fit = _stan_draws("rate", {"C": 90, "E": 40.0 * 25.0, "a": rel.a_eta, "b": rel.b_eta}, seed, n)
    out.append(dict(_compare("Poisson block rate eta", fit["eta"][0], eng), block="rate"))

    # ---- occupancy: OccRel.draw_params, presence probability of one block
    o1, o2 = _E(30), _E(12)
    # every anchor row has a pair, so all are active and the pairs at risk are 30 x 12
    pairs = sorted({(i, j) for i in range(30) for j in range(12) if rng.random() < 0.3} |
                   {(i, int(rng.integers(12))) for i in range(30)})
    uu = np.array([p[0] for p in pairs])
    vv = np.array([p[1] for p in pairs])
    occ = OccRel("o", o1, o2, uu, vv, np.ones(len(pairs)), mult_by="none", a=0.5, b=0.5)
    eng = []
    for _ in range(n):
        occ.draw_params(rng)
        eng.append(occ.p[0, 0])
    pa, pb = occ.beta_ab()
    T = 30 * 12
    fit = _stan_draws("occupancy", {"T": int(T), "S": len(pairs), "a": float(np.ravel(pa)[0]), "b": float(np.ravel(pb)[0])},
                      seed, n)
    out.append(dict(_compare("presence probability p", fit["p"][0], eng), block="occupancy"))

    # ---- levels: Marginal.fit of a categorical column (posterior Dirichlet draw)
    x = pd.Series(rng.choice(list("abcd"), 300, p=[0.5, 0.3, 0.15, 0.05]))
    cd = Codec("c", {"type": "categorical"}, x)
    xc = cd.encode(x)
    eng = np.stack([Marginal.fit(cd, xc, rng, dirichlet_a=2.0).p for _ in range(n)])
    V = len(cd.levels)
    counts = np.bincount(xc[xc >= 0], minlength=V)
    fit = _stan_draws("levels", {"V": V, "y": counts.tolist(), "alpha": (2.0 / V + 1e-9) * np.ones(V)}, seed, n)
    for j in range(V):
        out.append(dict(_compare(f"level {cd.levels[j]} probability", fit["theta"][j], eng[:, j]), block="levels"))

    # ---- nig: Entity.draw_attr_params, one segment
    xs = rng.normal(1.3, 0.7, 60)
    ent = Entity("e", 60, 1, xs[:, None], np.zeros((60, 0), np.int64), [])
    mu_d, s2_d = [], []
    for _ in range(n):
        ent.draw_attr_params(rng)
        mu_d.append(ent.mu[0, 0])
        s2_d.append(ent.s2[0, 0])
    fit = _stan_draws("nig", {"N": 60, "x": xs.tolist(), "k0": NIG["k0"], "a0": NIG["a0"], "b0": NIG["b0"]}, seed, n)
    out.append(dict(_compare("segment mean mu", fit["mu"][0], mu_d), block="nig"))
    out.append(dict(_compare("segment variance s2", fit["s2"][0], s2_d), block="nig"))

    # ---- copula: continuous columns (one value per bin), so the latent normals are
    # the normal scores; the engine's chain against Stan's NUTS on z ~ N(0, R)
    P, N = 3, 150
    Rt = np.array([[1, 0.6, -0.3], [0.6, 1, 0.2], [-0.3, 0.2, 1]])
    Z = rng.multivariate_normal(np.zeros(P), Rt, N)
    eps = 1e-7
    cop = GaussianCopula(P).fit(Z - eps, Z + eps, rng, eta=2.0, iters=2400, burn_in=400)
    D = np.stack(cop.draws)
    fit = _stan_draws("copula", {"N": N, "P": P, "z": Z.tolist(), "eta": 2.0}, seed, n)
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        # the chain's draws are autocorrelated: use its effective size for the error
        d = D[:, i, j]
        ess = _ess(d)
        r = _compare(f"correlation R[{i + 1},{j + 1}]", fit["R"][i, j], d)
        se = np.sqrt(fit["R"][i, j].var() / len(fit["R"][i, j]) + d.var() / ess)
        r["z_mean"] = round(float(abs(fit["R"][i, j].mean() - d.mean()) / se), 2)
        r["engine_ess"] = round(ess)
        r["ok"] = bool(r["z_mean"] < 4 and 0.8 < r["sd_ratio"] < 1.25)
        out.append(dict(r, block="copula"))
    return {"available": True, "blocks": out, "all_ok": all(b["ok"] for b in out),
            "seconds": round(time.time() - t0, 1)}


def _ess(x):
    """Effective sample size of one chain (initial positive sequence, Geyer 1992)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4 or x.var() == 0:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / (np.arange(n, 0, -1) * x.var())
    s, k = 0.0, 1
    while k + 1 < n:
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        s += pair
        k += 2
    return float(n / (1 + 2 * s))
