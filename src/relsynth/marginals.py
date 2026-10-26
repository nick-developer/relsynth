# SPDX-License-Identifier: Apache-2.0
"""Column codecs and Bayesian marginal distributions.

Every modelled column is mapped to either
  * a discrete column: integer codes 0..V-1 over a fixed level order, or
  * a continuous column: float values (dates become days, datetimes seconds).
Nulls are kept as an extra lowest "level" of the latent scale, so the null
rate and its dependence on other columns pass through the copula.

Marginal posteriors (layer 1 algorithm 1, stage 1 of algorithm 2):
  * discrete:   p ~ Dirichlet(counts + a * prior_p)   (exact level counts)
  * continuous: Bayesian bootstrap (Rubin 1981): w ~ Dirichlet(1,..,1) over
                observations, quantile knots = weighted quantiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

MODELLED = ("integer", "float", "categorical", "boolean", "date", "datetime")
DAY_NS = 86400 * 10**9
OTHER = "__OTHER__"          # model level pooling the categories beyond max_levels


class Codec:
    def __init__(self, name, cspec, series: pd.Series, max_levels=200, discrete_int_max=30, tail_min=5):
        self.name = name
        self.ctype = cspec["type"]
        self.min = cspec.get("min")
        self.max = cspec.get("max")
        s = series.dropna()
        if self.ctype in ("categorical", "boolean"):
            self.kind = "discrete"
            vc = s.value_counts()
            levels = list(vc.index)
            self.tail, self.tail_share = None, 0.0
            if len(levels) > max_levels:
                # levels beyond the cap share one model level; at decoding it is
                # replaced by a tail level seen in at least tail_min rows (never by
                # a placeholder, and never by a value seen in fewer rows), or, if
                # the tail has none, by a kept level
                tail = vc.iloc[max_levels - 1:]
                self.tail_share = float(tail.sum() / max(len(s), 1))
                tail = tail[tail >= tail_min]
                self.tail = (list(tail.index), tail.to_numpy(dtype=float)) if len(tail) else None
                levels = levels[: max_levels - 1] + [OTHER]
            if self.ctype == "boolean":
                levels = [False, True]
            self.levels = levels
        elif self.ctype in ("integer", "float") and s.nunique() <= discrete_int_max:
            # few distinct numbers (counts, price points, rates): a categorical on the
            # ordered values, so no value between them is ever generated
            self.kind = "discrete"
            self.levels = sorted(pd.to_numeric(s).unique().tolist())
        else:
            self.kind = "continuous"
            self.levels = None
        self.is_int = self.ctype == "integer"
        self.decimals = cspec.get("decimals")
        if self.decimals is None and self.ctype == "float" and len(s):
            v = pd.to_numeric(s, errors="coerce").dropna().to_numpy()[:2000]
            for d in range(7):
                if np.allclose(v, np.round(v, d), atol=1e-9, rtol=0):
                    self.decimals = d
                    break

    def redraw_tail(self, rng, mode="draw"):
        """Weights of the pooled tail levels for one dataset: a Bayesian-bootstrap
        draw over their counts, or the count shares in mean mode."""
        if self.kind == "discrete" and getattr(self, "tail", None) is not None:
            cnt = self.tail[1]
            self.run_tail_w = cnt / cnt.sum() if mode == "mean" else rng.dirichlet(cnt)

    # ---- value <-> model space
    def encode(self, series: pd.Series) -> np.ndarray:
        """discrete -> int codes (-1 = null); continuous -> float (nan = null)."""
        if self.kind == "discrete":
            idx = {v: i for i, v in enumerate(self.levels)}
            other = idx.get(OTHER)
            out = np.full(len(series), -1, dtype=np.int64)
            vals = series.to_numpy(dtype=object)
            for i, v in enumerate(vals):
                if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
                    continue
                if self.ctype in ("integer", "float"):
                    try:
                        v = type(self.levels[0])(float(v)) if self.levels else v
                    except (TypeError, ValueError):
                        continue
                j = idx.get(v, other)
                out[i] = -1 if j is None else j
            return out
        if self.ctype == "date":
            v = series.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / DAY_NS
            v[series.isna().to_numpy()] = np.nan
            return v
        if self.ctype == "datetime":
            v = series.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / 1e9
            v[series.isna().to_numpy()] = np.nan
            return v
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

    def decode(self, arr, rng=None) -> pd.Series:
        if self.kind == "discrete":
            arr = np.asarray(arr)
            lv = np.array(self.levels + [None], dtype=object)
            out = lv[np.where(arr < 0, len(self.levels), arr)]
            # the pooled tail level can sit anywhere: levels are reordered for the copula
            oi = self.levels.index(OTHER) if OTHER in self.levels else None
            if oi is not None:
                oth = np.flatnonzero(arr == oi)
                if len(oth):
                    rng = rng or np.random.default_rng(0)
                    if self.tail is not None:
                        vals, cnt = self.tail
                        # Bayesian bootstrap over the tail, drawn once per dataset when set
                        w = self.run_tail_w if getattr(self, "run_tail_w", None) is not None \
                            else rng.dirichlet(cnt)
                        out[oth] = np.array(vals, dtype=object)[rng.choice(len(vals), size=len(oth), p=w)]
                    else:
                        kept = np.array([j for j in range(len(self.levels)) if j != oi])
                        out[oth] = lv[kept[rng.integers(0, len(kept), size=len(oth))]] if len(kept) else None
            s = pd.Series(out, dtype=object)
            if self.ctype == "integer":
                s = pd.Series(pd.array([None if v is None else int(v) for v in out], dtype="Int64"))
            elif self.ctype == "float":
                s = pd.Series([np.nan if v is None else float(v) for v in out], dtype=float)
            return s
        arr = np.asarray(arr, dtype=float)
        if self.min is not None:
            arr = np.where(np.isnan(arr), arr, np.maximum(arr, self.min))
        if self.max is not None:
            arr = np.where(np.isnan(arr), arr, np.minimum(arr, self.max))
        if self.ctype == "date":
            ns = np.where(np.isnan(arr), np.nan, np.round(arr) * DAY_NS)
            return pd.to_datetime(pd.Series(ns), errors="coerce").dt.normalize()
        if self.ctype == "datetime":
            return pd.to_datetime(pd.Series(np.where(np.isnan(arr), np.nan, arr * 1e9)), errors="coerce")
        if self.decimals is not None and not self.is_int:
            arr = np.round(arr, self.decimals)
        if self.is_int:
            return pd.Series(pd.array([None if np.isnan(v) else int(round(v)) for v in arr], dtype="Int64"))
        return pd.Series(arr)


def _null_rate(S, T, al, be, pi_s, rng, bootstrap=True):
    """Null probability of a column in a group: S nulls in T rows, prior
    (pi_s/2) delta_0 + (pi_s/2) delta_1 + (1 - pi_s) Beta(al, be), drawn from its
    exact posterior (posterior mean when bootstrap is off)."""
    from scipy.special import betaln
    lw = [np.log(pi_s / 2) if (pi_s > 0 and S == 0) else -np.inf,
          np.log(pi_s / 2) if (pi_s > 0 and S == T) else -np.inf,
          np.log1p(-pi_s) + betaln(al + S, be + T - S) - betaln(al, be)]
    w = np.exp(np.array(lw) - max(lw))
    w /= w.sum()
    if not bootstrap:
        return float(w[1] + w[2] * (al + S) / (al + be + T))
    r = rng.random()
    if r < w[0]:
        return 0.0
    if r < w[0] + w[1]:
        return 1.0
    return float(rng.beta(al + S, be + T - S))


class Marginal:
    """Posterior draw of one column's marginal, on the codec's model space."""

    def __init__(self, codec: Codec):
        self.codec = codec
        self.kind = codec.kind

    @classmethod
    def fit(cls, codec, x, rng, prior: Marginal | None = None, dirichlet_a=2.0,
            grid=48, bootstrap=True, min_rows=0, tail_trim=0.0, max_obs=20000, structural=0.0):
        """structural: prior probability pi_s that the null rate is structurally 0 or
        1 (pi_s/2 each) and that an unobserved category level is structurally absent;
        the rest of the prior is the usual Beta / Dirichlet slab."""
        m = cls(codec)
        n = len(x)
        if m.kind == "discrete":
            V = len(codec.levels)
            counts = np.bincount(x[x >= 0], minlength=V).astype(float)
            nnull = float((x < 0).sum())
            p0 = prior.p if prior is not None else np.full(V, 1.0 / max(V, 1))
            q0 = prior.p_null if prior is not None else 0.5
            a = dirichlet_a
            alpha = counts + a * p0 + 1e-9
            if bootstrap and structural > 0 and counts.sum() > 0:
                from .relational import sparse_dirichlet_draw
                m.p = sparse_dirichlet_draw(counts, a * p0 + 1e-9, structural, rng)
            else:
                m.p = rng.dirichlet(alpha) if bootstrap else alpha / alpha.sum()
            al, be = a * q0 + 1e-3, a * (1 - q0) + 1e-3
            m.p_null = _null_rate(nnull, n, al, be, structural, rng, bootstrap)
            if nnull == 0 and (prior is None or prior.p_null < 1e-6):
                m.p_null = 0.0
            return m
        obs = x[~np.isnan(x)]
        nnull = n - len(obs)
        if len(obs) == 0 and prior is not None:
            m.knots_v, m.knots_c = prior.knots_v.copy(), prior.knots_c.copy()
            m.p_null = 1.0 if prior.p_null > 0 or n > 0 else 0.0
            return m
        if len(obs) == 0:
            obs = np.array([0.0])
        if len(obs) > max_obs:
            obs = rng.choice(obs, max_obs, replace=False)
        w = rng.dirichlet(np.ones(len(obs))) if bootstrap else np.full(len(obs), 1 / len(obs))
        if prior is not None:
            # Dirichlet-process posterior: F ~ DP(a + n, (a F0 + sum delta_x) / (a + n)).
            # A draw is V * (Bayesian bootstrap of the group) + (1 - V) * F0, V ~ Beta(n, a);
            # F0 (the parent marginal) enters through its quantile knots.
            V = rng.beta(len(obs), dirichlet_a) if bootstrap else len(obs) / (len(obs) + dirichlet_a)
            pv = np.interp(np.linspace(0, 1, 4 * grid + 1)[1:-1], prior.knots_c, prior.knots_v)
            obs = np.concatenate([obs, pv])
            w = np.concatenate([V * w, np.full(len(pv), (1 - V) / len(pv))])
        o = np.argsort(obs)
        obs, w = obs[o], w[o]
        cw = np.cumsum(w)
        cw = (cw - w / 2) / cw[-1]
        tau = np.linspace(tail_trim, 1 - tail_trim, grid + 1)
        # end knots at 1/(2 n_eff) from each end, so that prior pseudo-points with
        # negligible weight never become the support's endpoints
        eps = 0.5 * float((w ** 2).sum() / w.sum() ** 2)
        tq = np.clip(tau, eps, 1 - eps)
        v = np.interp(tq, cw, obs)
        if tail_trim > 0:   # extrapolate the trimmed tails linearly instead of exposing min/max
            v[0] -= (v[1] - v[0]) * 0.5
            v[-1] += (v[-1] - v[-2]) * 0.5
            tau = np.linspace(0, 1, grid + 1)
        m.knots_v, m.knots_c = v, tau
        m.p_null = 0.0 if nnull == 0 else _null_rate(nnull, n, 1.0, 1.0, structural, rng, bootstrap)
        return m

    # ---- latent bins for the copula (Feldman & Kowal anchors)
    def latent_bounds(self, x):
        """(lo, hi] on the N(0,1) scale for each observation."""
        if self.kind == "discrete":
            x = np.asarray(x)
            if x.dtype.kind == "f":            # codes that travelled through a float array
                x = np.where(np.isnan(x), -1, x).astype(np.int64)
            cum = np.concatenate([[0.0, self.p_null],
                                  self.p_null + (1 - self.p_null) * np.cumsum(self.p)])
            cum = np.clip(cum, 0, 1)
            cum[-1] = 1.0
            idx = np.where(x < 0, 0, x + 1)
            lo, hi = cum[idx], cum[idx + 1]
        else:
            uv = np.unique(self.knots_v)
            # cdf at each unique knot value = largest tau with that value
            c = np.array([self.knots_c[self.knots_v == u].max() for u in uv])
            cum = self.p_null + (1 - self.p_null) * c
            cum[-1] = 1.0
            obs = ~np.isnan(x)
            q = np.searchsorted(uv, np.where(obs, x, uv[0]), side="left")
            q = np.minimum(q, len(uv) - 1)
            # x in (uv[q-1], uv[q]]  ->  latent (cum[q-1], cum[q]]
            first_atom = cum[0] > self.p_null + 1e-9
            lo_q = np.where(q == 0, self.p_null, cum[np.maximum(q - 1, 0)])
            hi_q = cum[q]
            if not first_atom and len(uv) > 1:   # the minimum knot has no mass: use the first interval
                lo_q = np.where(q == 0, cum[0], lo_q)
                hi_q = np.where(q == 0, cum[1], hi_q)
            lo = np.where(obs, lo_q, 0.0)
            hi = np.where(obs, hi_q, self.p_null)
        lo = np.clip(lo, 0, 1)
        hi = np.clip(hi, 0, 1)
        same = hi <= lo
        hi = np.where(same, np.minimum(lo + 1e-9, 1.0), hi)
        return ndtri(lo), ndtri(hi)

    # ---- inverse transform
    def from_uniform(self, u):
        u = np.asarray(u, dtype=float)
        if self.kind == "discrete":
            cum = self.p_null + (1 - self.p_null) * np.cumsum(self.p)
            idx = np.searchsorted(cum, u, side="left")
            idx = np.minimum(idx, len(self.p) - 1)
            return np.where(u < self.p_null, -1, idx)
        isnull = u < self.p_null
        uu = np.clip((u - self.p_null) / max(1 - self.p_null, 1e-12), 0, 1)
        v = np.interp(uu, self.knots_c, self.knots_v)
        return np.where(isnull, np.nan, v)

    def sample(self, n, rng):
        return self.from_uniform(rng.random(n))
