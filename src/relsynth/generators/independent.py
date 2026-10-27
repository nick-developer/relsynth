# SPDX-License-Identifier: Apache-2.0
"""Independent marginals given the latent group: the simplest generator.

Each column is drawn from its own Bayesian marginal posterior of the row's
group (the same marginals as the copula's, shrunk table -> coarse group ->
group), independently of the other columns of the row.  Dependence between
columns survives only through the group, i.e. through the latent segments
that also carry the links between tables (a latent class model, Goodman 1974;
Dunson and Xing 2009 for its Bayesian form).  It is cheap and has no MCMC, so
it is the reference plug-in that proves the contract and a baseline for
model selection.
"""
from __future__ import annotations

import numpy as np

from .contract import Capabilities
from .copula import SegmentMarginals


class IndependentGenerator(SegmentMarginals):
    name = "independent"
    version = "1.0"
    description = ("Independent Bayesian marginals given the latent group (latent class model): no copula, "
                   "no MCMC; columns depend on each other only through the row's segment.")
    capabilities = Capabilities(conditioning=("group",), dp="none", gpu="never", pushdown=True, posterior=True)

    def fit(self, data, context, privacy_budget=None):
        self.data = data
        self._pi_s = data.cfg.get("structure", {}).get("structural_prior", 0.0)
        n = data.n_rows
        self._groups = np.zeros(n, dtype=np.int64) if context.groups is None else np.asarray(context.groups)
        self.parent = context.parent or {}
        self._fit_marginals(context.rng, data.cfg["marginals"]["bayesian_bootstrap"])
        return self

    def posterior_draw(self, rng, mode="draw"):
        self._redraw_marginals(rng, mode)
        return self

    def uncertainty(self):
        return {"groups": len(self.group_m), "rows_per_group_min": min(self.group_n.values(), default=0)}

    def sample(self, n, conditioning, rng):
        d = self.data
        groups = np.asarray(conditioning.groups)
        p = len(d.cols)
        out = np.zeros((len(groups), p))
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            # an unfitted group comes as its coarse group (a sparse fact block) or with one
            ms = self.group_m.get(g) or self.coarse_m.get(g) or self.coarse_m.get(self.parent.get(g, -1)) \
                or self.global_m
            U = rng.random((len(idx), p))
            for j, m in enumerate(ms):
                if d.cols[j] in d.aux:
                    continue
                out[idx, j] = m.from_uniform(U[:, j])
        return d.decode(out, rng)
