# SPDX-License-Identifier: Apache-2.0
"""Sequential CART synthesis (synthpop style) as a generator plug-in.

The table's own columns are synthesised one after another (the visit sequence
is the column order).  Column j is drawn from a classification or regression
tree fitted to predict it from the row's latent group, the coarser group the
engine backs off to, the parent attributes joined onto the row, and the
columns before it (Reiter 2005; Nowok, Raab and Dibben 2016, method "cart").
A synthetic row walks down the tree with its own (already synthetic) values
and takes the value of a donor drawn from the observed rows of its leaf
(minimum leaf size ``min_leaf``, synthpop's ``minbucket = 5``).

Posterior: donors are drawn with Bayesian-bootstrap weights (Rubin 1981),
Dirichlet(1, ..., 1) over the rows of each leaf, redrawn for every dataset,
so datasets are draws of the leaf distributions given the trees.  The tree
structures themselves are fitted once (synthpop's ``proper = TRUE`` refits
them on a bootstrap sample instead; we keep them fixed to keep the cost of a
dataset low).  This is an approximation of a posterior predictive draw and is
documented as such (Prop. 703).

Everything is plain numpy: split search on at most ``bins`` quantile
thresholds per numeric predictor and on categories ordered by their mean
response (Breiman et al. 1984, sec. 9.4; exact for a binary or numeric
response, a heuristic for a multi-class one).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from .contract import Capabilities, TableGenerator

DEFAULTS = {"min_leaf": 5, "max_depth": 12, "bins": 32, "max_rows": 50000, "min_gain": 1e-8,
            "smoothing": "none"}


def _scores(x):
    """Normal scores of the ranks (nulls lowest): the regression response."""
    v = np.asarray(x, float)
    nan = np.isnan(v)
    if nan.any():
        v = np.where(nan, v[~nan].min() - 1 if (~nan).any() else 0.0, v)
    r = np.argsort(np.argsort(v, kind="stable"), kind="stable") + 1.0
    return ndtri(r / (len(r) + 1))


class _Tree:
    """A binary tree over predictors that are numeric (threshold splits) or
    categorical (subset splits).  Nodes are stored in flat lists."""

    def __init__(self):
        self.feat, self.thr, self.left_set, self.left, self.right, self.big_left = [], [], [], [], [], []
        self.leaf_rows = {}

    def _new(self):
        for a in (self.feat, self.thr, self.left_set, self.left, self.right, self.big_left):
            a.append(None)
        return len(self.feat) - 1

    def fit(self, P, kinds, y, classify, opts):
        n = len(y)
        if classify:
            yc = np.where(y < 0, y.max(initial=0) + 1, y).astype(np.int64)
            C = int(yc.max(initial=0)) + 1
            target = yc
        else:
            target = _scores(y)
            C = 0
        stack = [(self._new(), np.arange(n), 0)]
        while stack:
            node, I, depth = stack.pop()
            best = None
            if depth < opts["max_depth"] and len(I) >= 2 * opts["min_leaf"]:
                best = self._best_split(P, kinds, target, I, C, opts)
            if best is None:
                self.leaf_rows[node] = I
                continue
            f, thr, lset, go_left = best
            L, R = I[go_left], I[~go_left]
            self.feat[node], self.thr[node], self.left_set[node] = f, thr, lset
            self.big_left[node] = len(L) >= len(R)
            ln, rn = self._new(), self._new()
            self.left[node], self.right[node] = ln, rn
            stack.append((rn, R, depth + 1))
            stack.append((ln, L, depth + 1))
        return self

    @staticmethod
    def _impurity_curve(stats, C):
        """Weighted impurity of every left/right split of an ordered list of
        cells.  stats: (cells, C) class counts, or (cells, 3) count/sum/sum2."""
        if C:
            cl = np.cumsum(stats, axis=0)[:-1]
            tot = stats.sum(axis=0)
            cr = tot - cl
            nl, nr = cl.sum(1), cr.sum(1)
            gl = nl - (cl ** 2).sum(1) / np.maximum(nl, 1)
            gr = nr - (cr ** 2).sum(1) / np.maximum(nr, 1)
            return gl + gr, nl, nr, tot.sum() - (tot ** 2).sum() / max(tot.sum(), 1)
        cl = np.cumsum(stats, axis=0)[:-1]
        tot = stats.sum(axis=0)
        cr = tot - cl
        nl, nr = cl[:, 0], cr[:, 0]
        sl = cl[:, 2] - cl[:, 1] ** 2 / np.maximum(nl, 1)
        sr = cr[:, 2] - cr[:, 1] ** 2 / np.maximum(nr, 1)
        return sl + sr, nl, nr, tot[2] - tot[1] ** 2 / max(tot[0], 1)

    def _best_split(self, P, kinds, t, I, C, opts):
        ml = opts["min_leaf"]
        ti = t[I]
        best, best_val = None, None
        root = None
        for f, kind in enumerate(kinds):
            x = P[I, f]
            if kind == "num":
                qs = np.unique(np.quantile(x, np.linspace(0, 1, opts["bins"] + 1)[1:-1]))
                if not len(qs):
                    continue
                cell = np.searchsorted(qs, x, side="left")       # x <= qs[k] <=> cell <= k
                ncell = len(qs) + 1
                order = np.arange(ncell)
            else:
                cats, cell = np.unique(x, return_inverse=True)
                ncell = len(cats)
                if ncell < 2:
                    continue
            if C:
                stats = np.bincount(cell * C + ti, minlength=ncell * C).reshape(ncell, C).astype(float)
            else:
                stats = np.column_stack([np.bincount(cell, minlength=ncell),
                                         np.bincount(cell, weights=ti, minlength=ncell),
                                         np.bincount(cell, weights=ti * ti, minlength=ncell)]).astype(float)
            if kind != "num":
                cnt = stats.sum(1) if C else stats[:, 0]
                if C:
                    maj = stats.sum(0).argmax()
                    key = stats[:, maj] / np.maximum(cnt, 1)
                else:
                    key = stats[:, 1] / np.maximum(cnt, 1)
                order = np.argsort(key, kind="stable")
                stats = stats[order]
            imp, nl, nr, root = self._impurity_curve(stats, C)
            ok = (nl >= ml) & (nr >= ml)
            if not ok.any():
                continue
            imp = np.where(ok, imp, np.inf)
            k = int(np.argmin(imp))
            if best_val is None or imp[k] < best_val:
                best_val = imp[k]
                if kind == "num":
                    best = (f, float(qs[k]), None, x <= qs[k])
                else:
                    lset = set(cats[order[: k + 1]].tolist())
                    best = (f, None, lset, np.isin(x, list(lset)))
        if best is None or root is None or root - best_val <= opts["min_gain"] * max(root, 1e-12):
            return None
        return best

    def route(self, P):
        """Leaf of every row of P (rows: synthetic predictors)."""
        node = np.zeros(len(P), dtype=np.int64)
        todo = np.ones(len(P), bool)
        while todo.any():
            for nd in np.unique(node[todo]):
                rows = np.flatnonzero(todo & (node == nd))
                if self.left[nd] is None:
                    todo[rows] = False
                    continue
                x = P[rows, self.feat[nd]]
                if self.thr[nd] is not None:
                    gl = x <= self.thr[nd]
                else:
                    seen = self._seen[nd]
                    gl = np.isin(x, list(self.left_set[nd]))
                    unseen = ~np.isin(x, seen)
                    gl[unseen] = self.big_left[nd]
                node[rows] = np.where(gl, self.left[nd], self.right[nd])
        return node


class CartGenerator(TableGenerator):
    name = "cart"
    version = "1.0"
    description = ("Sequential CART (synthpop style): a tree per column on the row's group, parent attributes "
                   "and earlier columns; leaf donors with Bayesian-bootstrap weights.")
    capabilities = Capabilities(conditioning=("group", "parent"), dp="none", gpu="never", pushdown=True,
                                posterior=True)

    def fit(self, data, context, privacy_budget=None):
        opts = {**DEFAULTS, **(context.options or {})}
        bad = set(opts) - set(DEFAULTS)
        if bad:
            from .contract import GeneratorError
            raise GeneratorError(f"{context.table}: unknown cart options {sorted(bad)} (allowed: {sorted(DEFAULTS)})")
        if opts["smoothing"] not in ("none", "density"):
            from .contract import GeneratorError
            raise GeneratorError(f"{context.table}: cart smoothing must be none or density")
        self.opts, self.data = opts, data
        rng = context.rng
        n = data.n_rows
        groups = np.zeros(n, dtype=np.int64) if context.groups is None else np.asarray(context.groups)
        self.parent = dict(context.parent or {})
        self.groups_fit = set(np.unique(groups).tolist())
        self._groups = groups
        coarse = self._coarse(groups)
        rows = rng.permutation(n)[: opts["max_rows"]] if n > opts["max_rows"] else np.arange(n)
        cols = data.cols
        self.own = [j for j, c in enumerate(cols) if c not in data.aux]
        self.auxj = [j for j, c in enumerate(cols) if c in data.aux]
        # predictor columns: group, coarse group, parent attributes, then own columns in visit order
        self.kinds = ["cat", "cat"] + [self._kind(j) for j in self.auxj] + [self._kind(j) for j in self.own]
        self.fill = {}
        base = [groups[rows].astype(float), coarse[rows]]
        base += [self._pred(j, data.X[j][rows], fit=True) for j in self.auxj]
        P = np.column_stack(base + [self._pred(j, data.X[j][rows], fit=True) for j in self.own]) \
            if (base or self.own) else np.zeros((len(rows), 0))
        self.trees, self.donors = [], []
        npre = 2 + len(self.auxj)
        for q, j in enumerate(self.own):
            y = data.X[j][rows]
            classify = data.codecs[j].kind == "discrete"
            tr = _Tree().fit(P[:, : npre + q], self.kinds[: npre + q], y, classify, opts)
            tr._seen = {nd: np.unique(P[tr_rows, tr.feat[nd]]) for nd, tr_rows in self._node_rows(tr, P).items()
                        if tr.left[nd] is not None and tr.thr[nd] is None}
            self.trees.append(tr)
            self.donors.append({leaf: y[I] for leaf, I in tr.leaf_rows.items()})
            tr.leaf_rows = {leaf: len(I) for leaf, I in tr.leaf_rows.items()}
        self.npre = npre
        self.posterior_draw(rng)
        return self

    def _node_rows(self, tr, P):
        """Rows of the fitting data at every node (for the categories seen there)."""
        out = {0: np.arange(len(P))}
        stack = [0]
        while stack:
            nd = stack.pop()
            if tr.left[nd] is None:
                continue
            I = out[nd]
            x = P[I, tr.feat[nd]]
            gl = x <= tr.thr[nd] if tr.thr[nd] is not None else np.isin(x, list(tr.left_set[nd]))
            out[tr.left[nd]], out[tr.right[nd]] = I[gl], I[~gl]
            stack += [tr.left[nd], tr.right[nd]]
        return out

    def _kind(self, j):
        cd = self.data.codecs[j]
        if cd.kind == "continuous" or cd.ctype in ("integer", "float"):
            return "num"
        return "cat"

    def _pred(self, j, x, fit=False):
        """Predictor values of column j (encoded): nulls of a numeric column sit
        below its smallest value (a level of their own), as in the copula."""
        v = np.asarray(x, float)
        if self.data.codecs[j].kind == "discrete":
            return v
        if fit:
            self.fill[j] = float(np.nanmin(v)) - 1.0 if (~np.isnan(v)).any() else 0.0
        return np.where(np.isnan(v), self.fill.get(j, 0.0), v)

    def posterior_draw(self, rng, mode="draw"):
        """Bayesian-bootstrap donor weights per leaf (equal weights in mean mode)."""
        self.weights = []
        for don in self.donors:
            self.weights.append({leaf: (np.full(len(v), 1.0 / max(len(v), 1)) if mode == "mean"
                                        else rng.dirichlet(np.ones(len(v))) if len(v) else np.zeros(0))
                                 for leaf, v in don.items()})
        return self

    def uncertainty(self):
        leaves = [len(d) for d in self.donors]
        return {"trees": len(self.trees), "leaves_total": int(sum(leaves)),
                "min_leaf_rows": int(min((min(t.leaf_rows.values()) for t in self.trees if t.leaf_rows), default=0))}

    def fitted_groups(self):
        return set(self.groups_fit)

    def _coarse(self, groups):
        """The back-off group of each row: its parent group, or the group itself
        when it has none (an entity, or a fact row already given its coarse group)."""
        lut = {g: (c if c >= 0 else g) for g, c in self.parent.items()}
        return np.array([lut.get(g, g) for g in np.asarray(groups).tolist()], dtype=float)

    def sample(self, n, conditioning, rng):
        d = self.data
        groups = np.asarray(conditioning.groups)
        M = len(groups)
        given = conditioning.given
        # a group the fit never saw (the engine passes a sparse block as its coarse
        # group) is unknown to the group splits and known to the coarse ones
        seen = np.isin(groups, list(self.groups_fit))
        base = [np.where(seen, groups, np.nan).astype(float), self._coarse(groups)]
        for j in self.auxj:
            if given is not None and d.cols[j] in given:
                base.append(self._pred(j, d.codecs[j].encode(given[d.cols[j]].reset_index(drop=True))))
            else:
                base.append(np.full(M, np.nan))
        P = np.zeros((M, self.npre + len(self.own)))
        if base:
            P[:, : self.npre] = np.column_stack(base)
        out = np.zeros((M, len(d.cols)))
        for q, j in enumerate(self.own):
            tr = self.trees[q]
            leaf = tr.route(P[:, : self.npre + q]) if M else np.zeros(0, np.int64)
            vals = np.zeros(M)
            for lf in np.unique(leaf):
                rows = np.flatnonzero(leaf == lf)
                don, w = self.donors[q][lf], self.weights[q][lf]
                if not len(don):                      # a table fitted on no rows: nulls
                    vals[rows] = np.nan if d.codecs[j].kind == "continuous" else -1
                    continue
                v = don[rng.choice(len(don), size=len(rows), p=w)].astype(float)
                if self.opts["smoothing"] == "density" and d.codecs[j].kind == "continuous":
                    v = self._smooth(v, don, rng)
                vals[rows] = v
            out[:, j] = vals
            P[:, self.npre + q] = self._pred(j, vals)
        return d.decode(out, rng)

    @staticmethod
    def _smooth(v, don, rng):
        """Gaussian kernel smoothing of drawn numbers (synthpop smoothing =
        "density"): bandwidth by Silverman's rule on the leaf's values."""
        ok = ~np.isnan(don)
        if ok.sum() < 2:
            return v
        x = don[ok]
        sd = x.std(ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        h = 0.9 * min(sd, iqr / 1.34 if iqr > 0 else sd) * len(x) ** -0.2
        m = ~np.isnan(v)
        v = v.copy()
        v[m] = v[m] + rng.normal(0.0, h, m.sum()) if h > 0 else v[m]
        return v
