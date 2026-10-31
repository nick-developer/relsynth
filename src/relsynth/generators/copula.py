# SPDX-License-Identifier: Apache-2.0
"""The Bayesian Gaussian copula as a generator plug-in (relsynth's default).

Group-specific Bayesian marginals plus a Gaussian copula per group (entities,
one group per latent segment) or a pooled copula (facts, one group per segment
block).  This is the attribute model of relsynth 1.0, moved behind the
generator contract without changing a single operation: for the same seed it
draws the same random numbers in the same order and gives byte-identical
datasets (Prop. 702).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ..copula import GaussianCopula
from ..marginals import Marginal
from .contract import Capabilities, TableGenerator


class SegmentMarginals(TableGenerator):
    """Marginal posteriors per group, shrunk top-down (table -> coarse group ->
    group); shared by the copula and the independent-marginals generators."""

    def _fit_marginals(self, rng, bootstrap):
        """Marginal posteriors top-down: the table's marginal, then each coarse
        group's (a Dirichlet / DP prior centred on the table's), then each group's
        (centred on its coarse group's).  bootstrap=False gives posterior means."""
        d = self.data
        mcfg, ccfg = d.cfg["marginals"], d.cfg["copula"]
        groups, parent, n = self._groups, self.parent, d.n_rows
        kw = dict(dirichlet_a=mcfg["dirichlet_prior"], grid=ccfg["grid"], bootstrap=bootstrap,
                  tail_trim=mcfg["tail_trim"], structural=self._pi_s, max_obs=int(mcfg.get("max_obs", 20000)))
        self.global_m = [Marginal.fit(cd, x, rng, **kw) for cd, x in zip(d.codecs, d.X)]
        self.group_m, self.group_n = {}, {}
        self.coarse_m = {}
        min_rows = ccfg["min_rows"]
        if parent:
            cg = np.array([parent.get(g, -1) for g in groups]) if n else np.array([])
            for c in np.unique(cg):
                if c < 0:
                    continue
                idx = cg == c
                self.coarse_m[c] = [Marginal.fit(cd, x[idx], rng, prior=gm, min_rows=min_rows, **kw)
                                    for cd, x, gm in zip(d.codecs, d.X, self.global_m)]
        for g in np.unique(groups):
            idx = groups == g
            prior = self.coarse_m.get(parent.get(g, -1), self.global_m)
            self.group_n[g] = int(idx.sum())
            self.group_m[g] = [Marginal.fit(cd, x[idx], rng, prior=pm, min_rows=min_rows, **kw)
                               for cd, x, pm in zip(d.codecs, d.X, prior)]

    def _redraw_marginals(self, rng, mode):
        if self.data.cfg["marginals"]["bayesian_bootstrap"]:
            # mean mode runs the posterior draws with their means (and median
            # indicators), so a level or a null rate that is structurally absent
            # with posterior probability above 1/2 stays absent, as in the other blocks
            from ..posterior import param_rng
            self._fit_marginals(param_rng(rng, mode), True)
        else:
            self._fit_marginals(rng, False)

    def fitted_groups(self):
        return set(self.group_m)


class CopulaGenerator(SegmentMarginals):
    name = "copula"
    version = "1.0"
    description = ("Bayesian Gaussian copula: Dirichlet / Bayesian-bootstrap marginals per latent segment and a "
                   "rank-likelihood copula (LKJ prior) per segment or pooled; conditions on parent attributes.")
    capabilities = Capabilities(conditioning=("group", "parent"), dp="none", gpu="never", pushdown=True,
                                posterior=True)

    def fit(self, data, context, privacy_budget=None):
        self.data = data
        groups, parent, rng, per_group_copula = context.groups, context.parent, context.rng, context.per_group
        cfg, mcfg, ccfg = data.cfg, data.cfg["marginals"], data.cfg["copula"]
        self._pi_s = cfg.get("structure", {}).get("structural_prior", 0.0)
        n = data.n_rows
        groups = np.zeros(n, dtype=np.int64) if groups is None else np.asarray(groups)
        self._groups, self.parent = groups, parent or {}
        self._fit_marginals(rng, mcfg["bayesian_bootstrap"])
        min_rows = ccfg["min_rows"]
        # latent bounds of every row under its own group's marginals
        p = len(data.cols)
        lo = np.zeros((n, p))
        hi = np.zeros((n, p))
        for g, ms in self.group_m.items():
            idx = groups == g
            for j, (m, x) in enumerate(zip(ms, data.X)):
                lo[idx, j], hi[idx, j] = m.latent_bounds(x[idx])
        # columns whose non-modal values occur in fewer than k rows carry no estimable
        # dependence (and copying their co-occurrences would single out those rows):
        # they are kept out of the copula, i.e. sampled independently given the group
        k = mcfg.get("min_dependence_rows", 0)
        self.independent = []
        if k and n:
            for j in range(p):
                cnt = pd.Series(data.X[j]).value_counts(dropna=False).to_numpy()
                if 0 < n - cnt.max() < k:
                    self.independent.append(data.cols[j])
                    lo[:, j], hi[:, j] = -np.inf, np.inf
        self.copulas = {}
        kw = dict(eta=ccfg["lkj_eta"], iters=ccfg["iterations"], burn_in=ccfg["burn_in"], mode=ccfg["mode"],
                  max_iters=ccfg.get("max_iterations"), rhat_target=ccfg.get("rhat_target", 1.1),
                  min_dep_rows=k, mh_steps=int(ccfg.get("mh_steps", 15)),
                  mcse_z=float(ccfg.get("mcse_z", 0.01)))
        cap = ccfg["sample_rows"]
        # the pooled copula is the one fact tables use; its posterior sd is about
        # 1/sqrt(rows fitted), and a dataset drawn from it shows that spread in every
        # correlation, so it is fitted on as many rows as a budget of rows x p^2
        # latent updates per sweep allows (the sampler's cost), at least 2 sample_rows
        n_pool = max(2 * cap, int(ccfg.get("pooled_cells", 1e6) // max(p * p, 1)))
        pooled_idx = rng.permutation(n)[: n_pool]
        self.pooled = GaussianCopula(p).fit(lo[pooled_idx], hi[pooled_idx], rng, **kw)
        if per_group_copula:
            for g in self.group_m:
                idx = np.flatnonzero(groups == g)
                if p >= 2 and len(idx) >= max(min_rows, 3 * p):
                    idx = rng.permutation(idx)[:cap]
                    self.copulas[g] = GaussianCopula(p).fit(lo[idx], hi[idx], rng, **kw)
        return self

    def posterior_draw(self, rng, mode="draw"):
        """Marginals drawn again from their posteriors (posterior means in mean
        mode) and one correlation matrix per group.  The rank-likelihood copula
        posterior does not depend on the marginals (Hoff 2007), so its stored
        draws stay valid when the marginals are redrawn."""
        self._redraw_marginals(rng, mode)
        for cop in [self.pooled] + list(self.copulas.values()):
            cop.set_current(rng, mode)
        return self

    def uncertainty(self):
        out = {"pooled": self.pooled.info}
        for g, c in self.copulas.items():
            out[f"group_{g}"] = c.info
        return out

    def sample(self, n, conditioning, rng):
        d = self.data
        groups, given = np.asarray(conditioning.groups), conditioning.given
        M = len(groups)
        p = len(d.cols)
        out = np.zeros((M, p))
        ai = [d.cols.index(a) for a in d.aux] if (d.aux and given is not None) else []
        ri = [j for j in range(p) if j not in ai]
        if ai:
            from ..table_model import _latent_draw
            # one array per column: discrete codes stay integers next to continuous values
            G = [d.codecs[j].encode(given[d.cols[j]].reset_index(drop=True)) for j in ai]
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            cop = self.copulas.get(g, self.pooled)
            ms = self.group_m.get(g, self.global_m)
            if ai:
                # the latent value of a given column is drawn uniformly on its bin
                # (a truncated standard normal), not fixed at the bin's centre
                za = np.column_stack([_latent_draw(*ms[j].latent_bounds(G[q][idx]), rng)
                                      for q, j in enumerate(ai)])
                za = np.clip(np.nan_to_num(za, nan=0.0, posinf=6, neginf=-6), -6, 6)
                Z = np.zeros((len(idx), p))
                Z[:, ri] = cop.sample_conditional(za, ai, ri, rng)
            else:
                Z = cop.sample(len(idx), rng) if p else np.zeros((len(idx), 0))
            U = ndtr(Z)
            for j, m in enumerate(ms):
                if j in ai:
                    continue
                out[idx, j] = m.from_uniform(U[:, j])
        return d.decode(out, rng)
