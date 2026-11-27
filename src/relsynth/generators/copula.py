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
                  tail_trim=mcfg["tail_trim"], structural=self._pi_s, max_obs=int(mcfg.get("max_obs", 20000)),
                  point_mass_knots=bool(mcfg.get("point_mass_knots", False)))
        self.global_m = [Marginal.fit(cd, x, rng, **kw) for cd, x in zip(d.codecs, d.X)]
        self.group_m, self.group_n = {}, {}
        self.coarse_m = {}
        min_rows = ccfg["min_rows"]
        # prior weight, in rows, of the table's (or coarse group's) marginal in each
        # segment's marginal (G3; Props. 1720, 1724-1726).  marginals.segment_prior_rows is
        # a number of rows, or auto: estimated per column and level from the segments
        # themselves (empirical Bayes, relsynth.shrinkage, Prop. 1725).  The privacy floor
        # marginals.segment_floor_rows raises the weight of a segment of n rows to at least
        # floor - n (Prop. 1726).  marginals.segment_prior_mode picks the columns and the
        # prior (shape: continuous columns, the parent's shape at the segment's median and
        # spread).  Off (0 and no floor), and in fact tables, every mode is the 2.1 fit.
        seg = self._segment_prior()
        self.shrinkage = {}
        mode = str(mcfg.get("segment_prior_mode", "table")) if seg is not None else "table"
        covered = (lambda cd: seg is not None and (mode == "table" or cd.kind != "discrete"))
        if parent:
            cg = np.array([parent.get(g, -1) for g in groups]) if n else np.array([])
            a_c = self._concentrations(cg, None, seg, mode, "coarse")
            for c in np.unique(cg):
                if c < 0:
                    continue
                idx = cg == c
                self.coarse_m[c] = [Marginal.fit(cd, x[idx], rng, prior=gm, min_rows=min_rows,
                                                 **self._seg_kw(kw, cd, a_c[j], x[idx], seg, mode, covered))
                                    for j, (cd, x, gm) in enumerate(zip(d.codecs, d.X, self.global_m))]
            ref = cg
        else:
            ref = None
        a_g = self._concentrations(groups, ref, seg, mode, "segment")
        for g in np.unique(groups):
            idx = groups == g
            prior = self.coarse_m.get(parent.get(g, -1), self.global_m)
            self.group_n[g] = int(idx.sum())
            self.group_m[g] = [Marginal.fit(cd, x[idx], rng, prior=pm, min_rows=min_rows,
                                            **self._seg_kw(kw, cd, a_g[j], x[idx], seg, mode, covered))
                               for j, (cd, x, pm) in enumerate(zip(d.codecs, d.X, prior))]

    def _segment_prior(self):
        """The segment prior of an entity table: None when off (prior weight 0
        and no privacy floor), else (rows, floor) with rows a number or "auto"
        (marginals.segment_prior_rows, marginals.segment_floor_rows).  Entity
        tables only, since the segments of a fact table are blocks of the
        relational model, not records (Prop. 1726 states the floor per entity)."""
        if getattr(self.data, "role", None) != "entity":
            return None
        mcfg = self.data.cfg["marginals"]
        rows = mcfg.get("segment_prior_rows", 0)
        rows = "auto" if rows == "auto" else float(rows or 0)
        floor = float(mcfg.get("segment_floor_rows", 0) or 0)
        dp = float(mcfg["dirichlet_prior"])
        # every group has a row, so a floor of at most dirichlet_prior + 1 rows never
        # raises a weight: with such a floor and rows <= dirichlet_prior the fit is 2.1's
        if rows != "auto" and rows <= dp and floor <= dp + 1:
            return None
        return rows, floor

    def _concentrations(self, groups, ref, seg, mode, level):
        """Prior weight per column at one level of the hierarchy (coarse groups
        around the table, or segments around their coarse group): the fixed
        number of rows, or the empirical-Bayes estimate (Prop. 1725), recorded
        in self.shrinkage for the model summary."""
        d = self.data
        p = len(d.codecs)
        if seg is None:
            return [0.0] * p
        rows, _ = seg
        if rows != "auto":
            return [float(rows)] * p
        from ..shrinkage import concentration
        g = np.asarray(groups)
        ok = g >= 0
        r = None if ref is None else np.asarray(ref)[ok]
        out, info = [], {}
        for j, (cd, x) in enumerate(zip(d.codecs, d.X)):
            if cd.kind == "discrete" and mode != "table":
                out.append(0.0)
                continue
            est = concentration(cd.kind, np.asarray(x)[ok], g[ok], r, levels=len(getattr(cd, "levels", ()) or ()),
                                shape=(mode == "shape" and cd.kind != "discrete"), a_max=float(max(int(ok.sum()), 1)))
            out.append(est.a)
            info[d.cols[j]] = est.to_dict()
        self.shrinkage[level] = info
        return out

    def _seg_kw(self, kw, cd, a_col, x, seg, mode, covered):
        """Marginal.fit options of one column of one group (codes x): the prior
        weight max(dirichlet_prior, a, floor - n) (Prop. 1726) on the columns the
        mode covers, the 2.1 options on the others.  n counts the observations
        Marginal.fit uses (non-null values, at most max_obs for a continuous
        column), so the floor holds on the mixture that is actually fitted.  A
        segment whose weight neither the fixed prior nor the floor raises keeps
        the 2.1 options, shape mode included (Prop. 1726 (b))."""
        if not covered(cd):
            return kw
        from ..shrinkage import floor_weight
        x = np.asarray(x)
        if cd.kind == "discrete":
            n_obs = int((x >= 0).sum())
        else:
            n_obs = min(int((~np.isnan(x)).sum()), int(kw.get("max_obs", 20000)))
        a = max(float(kw["dirichlet_a"]), floor_weight(n_obs, a_col, seg[1]))
        if a <= float(kw["dirichlet_a"]):
            return kw
        out = {**kw, "dirichlet_a": a}
        if mode != "table":
            out["shift_prior"] = mode == "shape"
        return out

    def summary(self) -> dict:
        out = super().summary()
        if getattr(self, "shrinkage", None):
            out["segment_shrinkage"] = self.shrinkage
        return out

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
