# SPDX-License-Identifier: Apache-2.0
"""A minimal third-party generator plug-in for relsynth (see README.md)."""
from __future__ import annotations

import numpy as np

from relsynth.generators import Capabilities, TableGenerator


class EmpiricalGenerator(TableGenerator):
    name = "empirical"
    version = "0.1"
    description = "Each column drawn from the values of the row's group, Bayesian-bootstrap weighted (example)."
    capabilities = Capabilities(conditioning=("group",), posterior=True)

    def fit(self, data, context, privacy_budget=None):
        self.data = data
        groups = np.zeros(data.n_rows, np.int64) if context.groups is None else np.asarray(context.groups)
        self.own = [j for j, c in enumerate(data.cols) if c not in data.aux]
        # encoded values of every own column, per group (group -1: rows the engine left out)
        self.values = {g: [data.X[j][groups == g] for j in self.own] for g in np.unique(groups)}
        self.everything = [data.X[j] for j in self.own]
        self.posterior_draw(context.rng)
        return self

    def posterior_draw(self, rng, mode="draw"):
        def w(n):
            return np.full(n, 1.0 / n) if mode == "mean" else rng.dirichlet(np.ones(n))
        self.weights = {g: [w(len(v)) if len(v) else None for v in vs] for g, vs in self.values.items()}
        self.weights_all = [w(len(v)) if len(v) else None for v in self.everything]
        return self

    def fitted_groups(self):
        return set(self.values)

    def sample(self, n, conditioning, rng):
        d = self.data
        groups = np.asarray(conditioning.groups)
        out = np.zeros((n, len(d.cols)))
        for g in np.unique(groups):
            rows = np.flatnonzero(groups == g)
            vs, ws = self.values.get(g), self.weights.get(g)
            if vs is None:                       # a group the fit never saw: the whole table
                vs, ws = self.everything, self.weights_all
            for q, j in enumerate(self.own):
                v, wq = (vs[q], ws[q]) if len(vs[q]) else (self.everything[q], self.weights_all[q])
                out[rows, j] = v[rng.choice(len(v), size=len(rows), p=wq)] if len(v) else np.nan
        return d.decode(out, rng)
