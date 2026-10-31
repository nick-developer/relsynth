# SPDX-License-Identifier: Apache-2.0
"""Attribute model of one table: the engine's encoding of its modelled columns
(codecs, joined dependent categoricals, features for segment discovery) and the
generator plug-in that fits and samples them (relsynth.generators; the default
is the Bayesian Gaussian copula of generators/copula.py)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln, ndtr, ndtri

from .marginals import MODELLED, Codec


def modelled_columns(tspec, exclude=()):
    return [c for c, cs in tspec["columns"].items()
            if cs["type"] in MODELLED and c not in exclude]


class JointCodec:
    """Strongly dependent low-cardinality categoricals modelled as one
    categorical over their observed combinations (a Dirichlet-categorical joint,
    as in a Bayesian network clique), so that their joint law is kept exactly
    instead of through a latent order.  Combinations never observed have no
    level, so they are structural zeros of the joint."""

    def __init__(self, parts, xs):
        self.parts = tuple(parts)
        self.name = "|".join(c.name for c in self.parts)
        self.ctype, self.kind, self.is_int = "categorical", "discrete", False
        self.min = self.max = None
        combo = pd.Series(list(zip(*xs)))
        self.levels = list(combo.value_counts().index)

    def encode_parts(self, xs):
        idx = {v: i for i, v in enumerate(self.levels)}
        return np.array([idx.get(t, -1) for t in zip(*xs)], dtype=np.int64)

    def decode_parts(self, arr, rng=None):
        arr = np.asarray(arr)
        k = len(self.parts)
        lv = self.levels + [(-1,) * k]
        out = []
        for q, cd in enumerate(self.parts):
            codes = np.array([lv[i][q] if i >= 0 else -1 for i in arr], dtype=np.int64)
            out.append(cd.decode(codes, rng).to_numpy())
        return out


def _latent_draw(lo, hi, rng):
    """Z ~ N(0, 1) truncated to (lo, hi]: inverse cdf of a uniform on (Phi(lo), Phi(hi)]."""
    a, b = ndtr(lo), ndtr(hi)
    u = a + (b - a) * rng.random(len(a))
    return ndtri(np.clip(u, 1e-12, 1 - 1e-12))


def _dm_logmarginal(counts, alpha=1.0):
    """Dirichlet(alpha)-multinomial log marginal of a sequence with these level counts."""
    counts = np.asarray(counts, float)
    L = len(counts)
    return float(gammaln(L * alpha) - gammaln(counts.sum() + L * alpha)
                 + (gammaln(counts + alpha) - gammaln(alpha)).sum())


def fd_log_bf(a, b, eps=1e-3):
    """log Bayes factor for "a determines b" (functional dependency, up to rare
    exceptions of probability eps) against "b independent of a".  Codes are ints
    with -1 = null, a level of its own.  Under both models b's values follow a
    Dirichlet(1)-categorical; under the dependency only the first row of each
    level of a draws from it, and every later row repeats that value w.p. 1-eps.
    Each repeated agreement therefore earns about -log p(b) nats, so a near-unique
    a (no repeats) never qualifies, however sparse the table."""
    a = np.asarray(a)
    b = np.asarray(b)
    if not len(a):
        return 0.0
    _, bi = np.unique(b, return_inverse=True)
    L = int(bi.max()) + 1
    if L < 2:
        return 0.0
    m0 = _dm_logmarginal(np.bincount(bi, minlength=L))
    order = np.lexsort((np.arange(len(a)), a))
    a_s, b_s = a[order], bi[order]
    first = np.r_[True, a_s[1:] != a_s[:-1]]
    m1 = _dm_logmarginal(np.bincount(b_s[first], minlength=L))
    head = np.maximum.accumulate(np.where(first, np.arange(len(a_s)), 0))
    agree = b_s[~first] == b_s[head[~first]]
    m1 += agree.sum() * np.log1p(-eps) + (~agree).sum() * (np.log(eps) - np.log(L - 1))
    return m1 - m0


def _cramers_v(x, y):
    ct = pd.crosstab(x, y).to_numpy().astype(float)
    if min(ct.shape) < 2:
        return 0.0
    n = ct.sum()
    e = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
    chi2 = ((ct - e) ** 2 / np.maximum(e, 1e-12)).sum()
    return float(np.sqrt(chi2 / n / (min(ct.shape) - 1)))


class TableModel:
    def __init__(self, tname, tspec, df: pd.DataFrame, cfg, exclude=(), aux=None):
        """aux: optional list of (name, column spec, series) of parent attributes
        joined onto each row.  They enter the copula, and at synthesis the row's
        own columns are drawn conditionally on its synthetic parent's values."""
        self.tname = tname
        self.cfg = cfg
        self.cols = modelled_columns(tspec, exclude)
        mcfg = cfg["marginals"]
        kmin = cfg.get("privacy", {}).get("suppress_small_counts", 5)
        self.codecs = [Codec(c, tspec["columns"][c], df[c], max_levels=mcfg["max_levels"], tail_min=kmin,
                             discrete_int_max=int(mcfg.get("discrete_int_max", 30)))
                       for c in self.cols]
        self.X = [cd.encode(df[cd.name]) for cd in self.codecs]
        self.n_rows = len(df)               # also when no column is modelled (keys and identifiers only)
        self.fd_groups = []
        self.aux = []
        for name, cs, ser in (aux or []):
            # numbers stay continuous: the codec is built on the values joined onto the
            # rows, which can be few (a fact over 20 stores) although the parent column
            # has many; a discrete codec would encode every other synthetic parent value
            # as null and condition the row on the wrong end of the scale
            cd = Codec(name, cs, ser, max_levels=mcfg["max_levels"], discrete_int_max=0)
            self.codecs.append(cd)
            self.X.append(cd.encode(ser.reset_index(drop=True)))
            self.cols.append(name)
            self.aux.append(name)
        self.out_cols = [c for c in self.cols if c not in self.aux]
        self._join_functional_dependencies(cfg["marginals"].get("fd_bf", 100.0))
        self._join_dependent_categoricals(cfg["marginals"].get("joint_max_levels", 64),
                                          cfg["marginals"].get("joint_min_cramers_v", 0.3))
        self._order_nominal_levels()
        self.generator: Any = None   # set by the engine (relsynth.generators); copula if left unset
        self.compute: Any = None

    def _join_functional_dependencies(self, bf):
        """Categoricals linked by functional dependencies (city -> state -> country,
        product -> category) are joined into one categorical over their observed
        combinations.  A dependency a -> b is accepted on a decisive Bayes factor
        (fd_log_bf); a group is joined only if it has no more combinations than its
        finest member has levels (plus the exceptions), so the joint discloses
        nothing beyond that column and keeps the dependency exactly."""
        cand = [i for i, cd in enumerate(self.codecs)
                if cd.kind == "discrete" and cd.ctype in ("categorical", "boolean") and cd.name not in self.aux
                and "__OTHER__" not in cd.levels]
        if len(cand) < 2:
            return
        n = self.n_rows
        lbf = np.log(bf)
        parent = {i: i for i in cand}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        # the test is run on a fixed subsample of large tables (the evidence only grows with n)
        fd_rows = int(self.cfg["marginals"].get("fd_sample_rows", 50000))
        sub = np.random.default_rng(0).permutation(n)[:fd_rows] if n > fd_rows else slice(None)
        for i in cand:
            for j in cand:
                if i != j and find(i) != find(j) and fd_log_bf(self.X[i][sub], self.X[j][sub]) > lbf:
                    parent[find(j)] = find(i)
        comps = {}
        for i in cand:
            comps.setdefault(find(i), []).append(i)
        joins = []
        for members in comps.values():
            if len(members) < 2:
                continue
            combos = len(set(zip(*[self.X[i].tolist() for i in members])))
            finest = max(len(np.unique(self.X[i])) for i in members)
            if combos <= finest + int(np.ceil(1e-3 * n)):
                joins.append(sorted(members))
        if not joins:
            return
        self.fd_groups = [[self.codecs[i].name for i in m] for m in joins]
        drop = []
        for m in joins:
            jc = JointCodec([self.codecs[i] for i in m], [self.X[i] for i in m])
            i0 = m[0]
            self.X[i0] = jc.encode_parts([self.X[i] for i in m])
            self.codecs[i0], self.cols[i0] = jc, jc.name
            drop += m[1:]
        for j in sorted(drop, reverse=True):
            del self.codecs[j], self.X[j], self.cols[j]

    def _join_dependent_categoricals(self, max_levels, min_v):
        cand = [i for i, cd in enumerate(self.codecs)
                if cd.kind == "discrete" and cd.ctype in ("categorical", "boolean") and cd.name not in self.aux
                and not isinstance(cd, JointCodec) and len(cd.levels) + 1 <= max_levels]
        pairs = []
        for ii, i in enumerate(cand):
            for j in cand[ii + 1:]:
                if (len(self.codecs[i].levels) + 1) * (len(self.codecs[j].levels) + 1) > max_levels:
                    continue
                v = _cramers_v(self.X[i], self.X[j])
                if v >= min_v:
                    pairs.append((v, i, j))
        used, joins = set(), []
        for _v, i, j in sorted(pairs, reverse=True):
            if i in used or j in used:
                continue
            used |= {i, j}
            joins.append((i, j))
        for i, j in joins:
            jc = JointCodec([self.codecs[i], self.codecs[j]], [self.X[i], self.X[j]])
            self.codecs[i], self.X[i], self.cols[i] = jc, jc.encode_parts([self.X[i], self.X[j]]), jc.name
        for j in sorted([j for _, j in joins], reverse=True):
            del self.codecs[j], self.X[j], self.cols[j]

    # Gaussian copulas need an order on nominal levels. We order each nominal
    # column's levels by the mean first-principal-component score of the other
    # columns within that level, so that the latent order tracks the dominant
    # dependence direction (binary columns need no ordering).
    def _order_nominal_levels(self):
        if len(self.cols) < 2:
            return
        S = np.column_stack([self._quick_scores(i) for i in range(len(self.cols))])
        for i, cd in enumerate(self.codecs):
            if cd.ctype != "categorical" or len(cd.levels) <= 2:
                continue
            # order by the mean of the most strongly associated other column
            # (largest correlation ratio), so that dependence stays monotone
            x = self.X[i]
            V = len(cd.levels)
            # rows of each level, in row order, by one stable sort (the same values in
            # the same order as v[x == l], so the means are bit-identical)
            order = np.argsort(x, kind="stable")
            start = np.searchsorted(x[order], np.arange(V), side="left")
            nl = np.bincount(x[x >= 0], minlength=V)[:V]
            best, means = -1.0, None
            for j in range(S.shape[1]):
                if j == i:
                    continue
                v = S[:, j]
                vs = v[order]
                mj = np.array([vs[start[l]:start[l] + nl[l]].mean() if nl[l] else 0.0 for l in range(V)])
                tot = v.var() * len(v)
                eta = (nl * (mj - v.mean()) ** 2).sum() / tot if tot > 0 else 0.0
                if eta > best:
                    best, means = eta, mj
            if means is None:
                continue
            perm = np.argsort(means)
            inv = np.empty_like(perm)
            inv[perm] = np.arange(len(perm))
            cd.levels = [cd.levels[j] for j in perm]
            self.X[i] = np.where(x >= 0, inv[np.maximum(x, 0)], -1)

    def _quick_scores(self, i):
        x = self.X[i]
        if self.codecs[i].kind == "discrete":
            v = x.astype(float)
            v[x < 0] = -1
        else:
            v = np.where(np.isnan(x), np.nanmin(x) - 1 if np.isfinite(np.nanmin(x)) else 0, x) \
                if np.isnan(x).any() else x
        r = pd.Series(v).rank(method="average").to_numpy()
        return ndtri(r / (len(r) + 1))

    # fitted parameters of the copula generator that 1.0 code read from the table model
    # (the science checks, the exported model); read through to the generator
    _GENERATOR_ATTRS = frozenset({"copulas", "pooled", "group_m", "global_m", "coarse_m", "group_n", "independent"})

    def __getattr__(self, name):
        gen = self.__dict__.get("generator")
        if name in TableModel._GENERATOR_ATTRS and gen is not None and hasattr(gen, name):
            return getattr(gen, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    # ---- the attribute generator (relsynth.generators): the encoding above is the
    # engine's; fitting and sampling the attribute values is the generator's
    @property
    def fitted(self):
        return self.generator is not None and self.generator.fitted

    def fit(self, groups=None, parent=None, rng=None, per_group_copula=True):
        """groups: int label per row (None = one group). parent: dict group -> coarser
        group label, used as the back-off of small groups."""
        from .generators.contract import FitContext
        rng = rng or np.random.default_rng()
        if self.generator is None:
            from .generators.registry import default_registry
            self.generator = default_registry().create("copula")
        ctx = FitContext(groups=None if groups is None else np.asarray(groups), parent=parent or {},
                         per_group=per_group_copula, rng=rng, config=self.cfg,
                         options=self.generator.options, compute=self.compute, table=self.tname)
        self.generator.fit(self, ctx)
        self.generator.fitted = True
        return self

    def redraw(self, rng, mode="draw"):
        """Parameters of one synthetic dataset: one posterior draw of the
        generator's parameters (posterior means in mean mode) and the weights of
        pooled tail levels."""
        if not self.fitted:
            return self
        self.generator.posterior_draw(rng, mode)
        for cd in self.codecs:
            for part in getattr(cd, "parts", [cd]):
                if hasattr(part, "redraw_tail"):
                    part.redraw_tail(rng, mode)
        return self

    def copula_info(self):
        return self.generator.uncertainty() if self.fitted else {}

    def fitted_groups(self):
        return self.generator.fitted_groups()

    def sample(self, groups, rng, given: pd.DataFrame | None = None) -> pd.DataFrame:
        from .generators.contract import Conditioning
        groups = np.asarray(groups)
        if given is not None and "parent" not in self.generator.capabilities.conditioning:
            given = None
        df = self.generator.sample(len(groups), Conditioning(groups, given), rng)
        if list(df.columns) != self.out_cols or (self.out_cols and len(df) != len(groups)):
            from .generators.contract import GeneratorError
            raise GeneratorError(f"{self.tname}: generator {self.generator.name} returned columns "
                                 f"{list(df.columns)} x {len(df)} rows, expected {self.out_cols} x {len(groups)}")
        return df

    def decode(self, out, rng) -> pd.DataFrame:
        """Encoded values (rows x cols, aux columns ignored) -> the table's own
        columns, decoded by their codecs (a joint codec gives each part)."""
        cols = {}
        for j, cd in enumerate(self.codecs):
            if cd.name in self.aux:
                continue
            v = out[:, j]
            if cd.kind == "discrete":
                v = v.astype(np.int64)
            if isinstance(cd, JointCodec):
                for part, vals in zip(cd.parts, cd.decode_parts(v, rng)):
                    cols[part.name] = vals
            else:
                cols[cd.name] = cd.decode(v, rng).to_numpy()
        return pd.DataFrame({c: cols[c] for c in self.out_cols})

    def features(self):
        """Per-row features for segment discovery: numeric normal scores and
        categorical codes."""
        num, cat, ncat = [], [], []
        for cd, x in zip(self.codecs, self.X):
            if cd.kind == "continuous":
                v = x.copy()
                ok = ~np.isnan(v)
                r = pd.Series(v[ok]).rank(method="average").to_numpy()
                s = np.zeros(len(v))
                s[ok] = ndtri(r / (ok.sum() + 1))
                num.append(s)
                if (~ok).any():
                    cat.append((~ok).astype(np.int64))
                    ncat.append(2)
            else:
                V = len(cd.levels)
                if V > 30:
                    continue
                cat.append(np.where(x < 0, V, x))
                ncat.append(V + 1)
        Xn = np.column_stack(num) if num else np.zeros((self.n_rows, 0))
        Xc = np.column_stack(cat) if cat else np.zeros((Xn.shape[0], 0), dtype=np.int64)
        return Xn, Xc, ncat
