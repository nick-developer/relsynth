# SPDX-License-Identifier: Apache-2.0
"""A differentially private seed database (stage 1 of DP mode, docs/dp.md).

DP mode synthesises in two stages.  This module is stage 1: it reads the real
tables once, releases a fixed set of noisy integer statistics, and decodes them
into a *seed database* with the spec's tables, keys and links.  Stage 2 is the
ordinary relsynth engine fitted to the seed database.  Stage 2 never sees real
data, so everything it does (posterior draws, segments, rules, keys) is
post-processing and keeps the guarantee of stage 1 (Prop. 905).

Stage 1, per table owned by the privacy unit, parents first (Props. 901-904):

1. **Contribution bounding.**  The unit's own table has one row per unit.  A
   child table keeps at most D rows per parent row (D from a noisy degree
   histogram at a public quantile, or set in the spec), so one unit owns at
   most c_t = prod of D along its path rows of each table (PrivLava, Cai et al.
   2023).
2. **Discovery.**  Numeric ranges from a noisy histogram on a fixed log grid;
   category levels by Gaussian partition selection (approximate zCDP).
   Columns declared public in the spec skip this.
3. **Marginals.**  Noisy histograms of every modelled column, concatenated.
4. **Correlations.**  The noisy sum of z z^T over rows, where z are the
   rows' normal scores under the noisy marginals, clipped to norm R, scaled
   and rounded to integers.  The row's parent's scores and its number of
   children in each child table are coordinates too, so the copula links
   tables (a child row is drawn given its parent's scores).

Tables that hold no data of the unit are copied as they are and must be
declared public (`public_tables`); views are left to stage 2.
"""
from __future__ import annotations

import logging
import math
import secrets
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..errors import PrivacyError
from .config import DPConfig
from .mechanisms import Accountant, Charge, gaussian, sigma_for, threshold_for

log = logging.getLogger("relsynth")

NUMERIC = {"integer", "float", "date", "datetime"}
CATEGORICAL = {"categorical", "boolean", "constant"}
PLACEHOLDER = {"pii", "text", "binary"}
SCALE = 1024                      # fixed-point scale of the correlation sums
ZMAX = 4.0


def _rng():
    return np.random.default_rng(secrets.randbits(128))


def _hash_key() -> str:
    """The run's secret key for truncation priorities (16 characters)."""
    return secrets.token_hex(8)


def _keys(k):
    return list(k) if isinstance(k, (list, tuple)) else [k]


# ------------------------------------------------------------------ the plan
@dataclass
class TablePlan:
    name: str
    parent_rel: dict | None = None             # relationship to the owned parent (None for the unit)
    children: list = field(default_factory=list)   # relationships of owned children
    public_fks: list = field(default_factory=list)  # single-column links to public tables
    numeric: list = field(default_factory=list)
    categorical: list = field(default_factory=list)
    placeholder: list = field(default_factory=list)
    sequence: list = field(default_factory=list)
    other_ids: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    computed: list = field(default_factory=list)


@dataclass
class Plan:
    unit: str
    order: list
    tables: dict
    public: list
    views: list
    notes: list = field(default_factory=list)


def make_plan(spec: dict, cfg: DPConfig) -> Plan:
    """Which tables hold data of the unit, how each reaches it, and how each
    column is modelled.  Refuses what DP mode cannot account for, before any
    data is read."""
    from ..ir import lift
    from ..ir.privacy import trace
    from ..prepare import computed_columns
    tr = trace(lift(spec), cfg.unit)
    views = [t for t, T in spec["tables"].items() if T.get("role") == "view"]
    owned = [n for n in tr.owned() if n not in views]
    problems = []
    for n in owned:
        if len(tr.entities[n].paths) > 1:
            problems.append(f"{n} reaches {cfg.unit} along {len(tr.entities[n].paths)} paths: a row could belong to "
                            "several people; DP mode needs one path (exclude the others in the IR's "
                            "privacy.exclude_paths)")
    others = [t for t in spec["tables"] if t not in owned and t not in views]
    if cfg.public_tables == "all_unowned":
        public = others
    else:
        public = [t for t in others if t in cfg.public_tables]
        missing = [t for t in others if t not in cfg.public_tables]
        if missing:
            problems.append(f"tables {missing} hold no data of {cfg.unit} and would be released as they are: list them "
                            "in synthesis.privacy.dp.public_tables (or set it to all_unowned) to confirm they are "
                            "public; DP protects only the unit's data")
    rels = spec.get("relationships") or []
    comp = computed_columns(spec)
    plans: dict = {}
    for n in owned:
        p = TablePlan(n)
        T = spec["tables"][n]
        if n != cfg.unit:
            path = tr.entities[n].paths[0]
            rname = path[0]
            p.parent_rel = next(r for r in rels if r["name"] == rname)
        fk_cols: set = set()
        for r in rels:
            if r["child"] != n:
                continue
            cols = _keys(r["child_key"])
            if f"{n}.{cols[0]}" in cfg.drop_columns:
                p.dropped += cols
                fk_cols |= set(cols)
                continue
            if r is p.parent_rel:
                fk_cols |= set(cols)
            elif r["kind"] == "self_reference" or r["parent"] == n:
                problems.append(f"{n}.{'+'.join(cols)} links rows of {n} to each other; DP mode cannot bound it: add "
                                f"{n}.{cols[0]} to synthesis.privacy.dp.drop_columns")
            elif r["parent"] in owned:
                problems.append(f"{r['name']} links two tables of {cfg.unit} off the ownership path; add {n}.{cols[0]} "
                                "to synthesis.privacy.dp.drop_columns")
            elif r["parent"] in public and len(cols) == 1:
                p.public_fks.append(r)
                fk_cols |= set(cols)
            else:
                p.dropped += cols
                fk_cols |= set(cols)
        pk = set(_keys(T.get("primary_key") or []))
        child_keys = set()
        for r in rels:
            if r["parent"] == n and r["child"] in owned:
                child_keys |= set(_keys(r["parent_key"]))
                if r["kind"] != "self_reference" and r["child"] != n:
                    p.children.append(r)
        for c, cs in T["columns"].items():
            t = cs["type"]
            if c in fk_cols or c in pk or c in child_keys:
                continue
            if c in comp.get(n, ()) or t == "lookup":
                p.computed.append(c)
            elif f"{n}.{c}" in cfg.drop_columns:
                p.dropped.append(c)
            elif t in NUMERIC:
                p.numeric.append(c)
            elif t in CATEGORICAL:
                p.categorical.append(c)
            elif t in PLACEHOLDER:
                p.placeholder.append(c)
            elif t == "sequence":
                p.sequence.append(c)
            elif t == "id":
                p.other_ids.append(c)
            else:
                p.dropped.append(c)
        plans[n] = p
    if problems:
        raise PrivacyError("DP mode cannot run this spec yet (nothing was read):\n  - " + "\n  - ".join(problems))
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        par = plans[n].parent_rel
        if par is not None:
            visit(par["parent"])
        seen.add(n)
        order.append(n)
    for n in owned:
        visit(n)
    return Plan(cfg.unit, order, plans, public, views)


# ------------------------------------------------------------------ marginals
class Marginal:
    """A noisy one-dimensional distribution over ordered cells, the first of which
    is null; maps values to normal scores and normal scores back to values."""

    def __init__(self, kind, cells, counts, *, edges=None, levels=None, integer=False, dtype=None):
        self.kind, self.edges, self.levels, self.integer, self.dtype = kind, edges, levels, integer, dtype
        c = np.clip(np.asarray(counts, dtype=float), 0, None)
        self.raw = np.asarray(counts, dtype=float)
        self.total = float(self.raw.sum())
        if c.sum() <= 0:
            c = np.ones_like(c)
            c[0] = 0 if len(c) > 1 else 1
        self.p = c / c.sum()
        self.cum = np.concatenate([[0.0], np.cumsum(self.p)])
        self.cells = cells

    def _cell(self, v):
        """Cell index and position within the cell (0-1) of each value."""
        n = len(v)
        idx = np.zeros(n, dtype=int)
        pos = np.full(n, 0.5)
        if self.kind == "cat":
            look = {lv: i + 1 for i, lv in enumerate(self.levels)}
            s = pd.Series(v, dtype=object)
            idx = s.map(lambda x: look.get(x, 0) if not _isnull(x) else 0).to_numpy(dtype=int)
            return idx, pos
        x = pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype=float)
        ok = ~np.isnan(x)
        e = self.edges
        xc = np.clip(x[ok], e[0], e[-1])
        b = np.clip(np.searchsorted(e, xc, side="right") - 1, 0, len(e) - 2)
        w = e[b + 1] - e[b]
        if self.integer:
            p_ = (np.floor(xc) - e[b] + 0.5) / np.maximum(w, 1)
        else:
            p_ = np.where(w > 0, (xc - e[b]) / np.where(w > 0, w, 1), 0.5)
        idx[ok] = b + 1
        pos[ok] = np.clip(p_, 0, 1)
        return idx, pos

    def score(self, v) -> np.ndarray:
        idx, pos = self._cell(v)
        u = self.cum[idx] + pos * self.p[idx]
        return np.clip(norm.ppf(np.clip(u, 1e-9, 1 - 1e-9)), -ZMAX, ZMAX)

    def draw(self, z, rng) -> np.ndarray:
        u = norm.cdf(z)
        idx = np.clip(np.searchsorted(self.cum, u, side="right") - 1, 0, len(self.p) - 1)
        pos = np.where(self.p[idx] > 0, (u - self.cum[idx]) / np.where(self.p[idx] > 0, self.p[idx], 1), rng.random(len(u)))
        pos = np.clip(pos, 0, 1)
        return self._values(idx, pos)

    def draw_cells(self, idx, rng) -> np.ndarray:
        """Values in the given cells, at uniform positions within each cell (mst/aim)."""
        idx = np.clip(np.asarray(idx, dtype=int), 0, len(self.p) - 1)
        return self._values(idx, rng.random(len(idx)))

    def _values(self, idx, pos) -> np.ndarray:
        if self.kind == "cat":
            lv = np.array([None, *self.levels], dtype=object)
            return lv[idx]
        out = np.full(len(idx), np.nan)
        ok = idx > 0
        b = idx[ok] - 1
        lo, hi = self.edges[b], self.edges[b + 1]
        if self.integer:
            out[ok] = np.minimum(lo + np.floor(pos[ok] * np.maximum(hi - lo, 1)), np.maximum(hi - 1, lo))
        else:
            out[ok] = lo + pos[ok] * (hi - lo)
        return out


def _isnull(x):
    try:
        return x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA or x is pd.NaT
    except TypeError:
        return False


def _numeric_values(s: pd.Series, kind: str) -> np.ndarray:
    if kind in ("date", "datetime"):
        d = pd.to_datetime(s, errors="coerce")
        out = np.full(len(d), np.nan)
        ok = d.notna().to_numpy()
        out[ok] = d[ok].astype("int64").to_numpy() / 1e9
        return out
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


GRID = np.concatenate([-np.logspace(15, -6, 85), [0.0], np.logspace(-6, 15, 85)])   # public, data-free


def _grid_counts(x: np.ndarray) -> np.ndarray:
    x = x[~np.isnan(x)]
    b = np.clip(np.searchsorted(GRID, np.clip(x, GRID[0], GRID[-1]), side="right") - 1, 0, len(GRID) - 2)
    return np.bincount(b, minlength=len(GRID) - 1)


def _range_from_grid(noisy: np.ndarray, tau: float):
    hit = np.nonzero(noisy > tau)[0]
    if len(hit) == 0:
        return None
    return float(GRID[hit[0]]), float(GRID[hit[-1] + 1])


def _edges(lo, hi, bins, integer):
    if hi <= lo:
        hi = lo + 1
    if integer:
        lo, hi = math.floor(lo), math.ceil(hi) + 1
        if hi - lo <= bins:
            return np.arange(lo, hi + 1, dtype=float)
        return np.unique(np.round(np.linspace(lo, hi, bins + 1)))
    return np.linspace(lo, hi, bins + 1)


def _degree_edges(dmax):
    small = list(range(0, min(dmax, 16) + 1))
    e = small[:]
    x = 16
    while x < dmax:
        x = min(2 * x, dmax)
        e.append(x)
    return np.array(e + [dmax + 1], dtype=float)


# ------------------------------------------------------------------ correlation
def _nearest_corr(C):
    C = (C + C.T) / 2
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-3, None)
    C = (V * w) @ V.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def _vech_counts(Z, R):
    """Sum over rows of the upper triangle (with the diagonal) of z z^T, each row
    clipped to norm R and rounded to multiples of 1/SCALE; and the integer L2 norm
    bound of one row's contribution."""
    q = Z.shape[1]
    iu = np.triu_indices(q)
    nrm = np.linalg.norm(Z, axis=1)
    f = np.where(nrm > R, R / np.maximum(nrm, 1e-12), 1.0)
    Zc = Z * f[:, None]
    P = np.rint(SCALE * Zc[:, iu[0]] * Zc[:, iu[1]]).astype(np.int64)
    d = len(iu[0])
    bound = SCALE * R * R + 0.5 * math.sqrt(d)            # Prop. 902
    return P.sum(axis=0), bound, iu


# ------------------------------------------------------------------ fitting
@dataclass
class TableModel:
    name: str
    coords: list                 # (kind, name) in order: attr, fk, deg, parent
    marg: dict                   # coord name -> Marginal
    corr: np.ndarray | None
    n_hat: int
    cap: int                     # rows per unit (c_t)
    degree_cap: dict = field(default_factory=dict)   # child -> D
    parent_coords: list = field(default_factory=list)
    forest: object = None        # mst/aim: the fitted two-way model (select.Forest); corr is then None


@dataclass
class DPModel:
    plan: Plan
    tables: dict
    public: dict
    acct: Accountant
    report: dict = field(default_factory=dict)


def fit(spec: dict, real: dict, cfg: DPConfig, acct: Accountant, plan: Plan) -> DPModel:
    """Release stage 1's statistics.  `acct` refuses anything beyond the plan."""
    hkey = _hash_key()
    owned = plan.order
    rho_t = cfg.rho / len(owned)
    delta_levels = cfg.delta * cfg.levels_delta_share
    n_disc = sum(1 for t in owned for c in plan.tables[t].categorical if f"{t}.{c}" not in cfg.public) or 1
    rows: dict = {plan.unit: real[plan.unit].reset_index(drop=True)}
    caps = {plan.unit: 1}
    models: dict = {}
    scores_of: dict = {}          # table -> DataFrame of attribute normal scores, aligned with rows[t]
    tree = cfg.measure != "copula"
    cells_of: dict = {}           # mst/aim: table -> {coordinate: cell of each row of rows[t]}
    report: dict = {"tables": {}}
    for t in owned:
        P = plan.tables[t]
        df = rows[t]
        c_t = caps[t]
        T = spec["tables"][t]
        sh = dict(cfg.split)
        if not P.children:
            sh.pop("degrees")
        need_disc = [c for c in P.numeric if not _public_range(cfg, t, c)] + \
            [c for c in P.categorical if f"{t}.{c}" not in cfg.public]
        if not need_disc:
            sh.pop("discovery")
        tot = sum(sh.values())
        budget = {k: rho_t * v / tot for k, v in sh.items()}
        tr: dict = {"rows_per_unit_cap": c_t, "rho": rho_t}
        # ---------------- discovery
        ranges, levels = {}, {}
        var_of: dict = {}
        disc_num = [c for c in P.numeric if not _public_range(cfg, t, c)]
        disc_cat = [c for c in P.categorical if f"{t}.{c}" not in cfg.public]
        if need_disc:
            share = budget["discovery"] / (len(disc_num) + len(disc_cat))
            if disc_num:
                vec = np.concatenate([_grid_counts(_numeric_values(df[c], T["columns"][c]["type"])) for c in disc_num])
                r = share * len(disc_num)
                noisy = gaussian(vec, c_t * math.sqrt(len(disc_num)), r, acct, "discovery", f"{t}: numeric ranges")
                sig = sigma_for(c_t * math.sqrt(len(disc_num)), r)
                m = len(GRID) - 1
                for i, c in enumerate(disc_num):
                    ranges[c] = _range_from_grid(noisy[i * m:(i + 1) * m], 3 * sig) or (0.0, 1.0)
            for c in disc_cat:
                vc = df[c].dropna().astype(str).value_counts()
                noisy = gaussian(vc.to_numpy(), c_t, share, acct, "discovery", f"{t}.{c}: level selection")
                sig = sigma_for(c_t, share)
                d_c = delta_levels / n_disc
                tau = threshold_for(c_t, sig, d_c, c_t)
                acct.charge(Charge("discovery", f"{t}.{c}: level selection (delta)", 0.0, d_c))
                keep = noisy > tau
                levels[c] = (list(vc.index[keep]), noisy[keep])
                var_of[c] = sig ** 2
        # ---------------- marginals
        marg: dict = {}
        coords: list = []
        hist_parts: list = []
        makers: list = []
        for c in P.numeric:
            kind = T["columns"][c]["type"]
            lo, hi = _public_range(cfg, t, c) or ranges[c]
            integer = kind == "integer"
            if kind in ("date", "datetime"):
                lo, hi = _date_bound(lo), _date_bound(hi)
            e = _edges(lo, hi, cfg.bins, integer)
            x = _numeric_values(df[c], kind)
            ok = ~np.isnan(x)
            b = np.clip(np.searchsorted(e, np.clip(x[ok], e[0], e[-1]), side="right") - 1, 0, len(e) - 2)
            cnt = np.concatenate([[int((~ok).sum())], np.bincount(b, minlength=len(e) - 1)])
            hist_parts.append(cnt)
            makers.append((c, lambda k, e=e, integer=integer, kind=kind: Marginal("num", None, k, edges=e,
                                                                                  integer=integer, dtype=kind)))
        for c in P.categorical:
            if f"{t}.{c}" in cfg.public:
                lv = [str(x) for x in cfg.public[f"{t}.{c}"]["levels"]]
                s = df[c].astype(object).where(df[c].notna(), None).map(lambda x: None if x is None else str(x))
                look = {v: i + 1 for i, v in enumerate(lv)}
                cnt = np.bincount(s.map(lambda x: look.get(x, 0)).to_numpy(dtype=int), minlength=len(lv) + 1)
                hist_parts.append(cnt)
                makers.append((c, lambda k, lv=lv: Marginal("cat", None, k, levels=lv)))
            else:
                lv, noisy = levels[c]
                nnull = int(df[c].isna().sum())
                marg[c] = ("selected", lv, noisy, nnull)
        for r in P.public_fks:
            ck, pk = _keys(r["child_key"])[0], _keys(r["parent_key"])[0]
            lv = list(pd.Series(real[r["parent"]][pk]).dropna().astype(str).unique())
            look = {v: i + 1 for i, v in enumerate(lv)}
            s = df[ck].map(lambda x: 0 if _isnull(x) else look.get(str(_canon(x)), 0)).to_numpy(dtype=int)
            hist_parts.append(np.bincount(s, minlength=len(lv) + 1))
            makers.append((ck, lambda k, lv=lv: Marginal("cat", None, k, levels=lv)))
        pc_tree: list = []
        if tree and P.parent_rel is not None:
            # the parent row's cells, counted over this table's rows (nodes of the two-way model)
            par = P.parent_rel["parent"]
            pc_tree = [c for k, c in models[par].coords if k == "attr"][:cfg.max_parent_attrs]
            pidx = _parent_index(df, rows[par], _keys(P.parent_rel["child_key"]), _keys(P.parent_rel["parent_key"]))
            for c in pc_tree:
                kk = len(models[par].marg[c].p)
                hist_parts.append(np.bincount(cells_of[par][c][pidx], minlength=kk))
                makers.append((f"^{c}", lambda k: Marginal("cat", None, k, levels=[str(i) for i in range(len(k) - 1)])))
        null_counts = {c: v for c, v in marg.items()}
        marg = {}
        # the row count, released with the marginals (one more cell; a unit changes it by c_t)
        hist_parts.append(np.array([len(df)]))
        makers.append(("#rows", lambda k: Marginal("num", None, np.concatenate([[0.0], k]), edges=np.array([0.0, 1.0]))))
        if hist_parts:
            q = len(hist_parts)
            vec = np.concatenate(hist_parts)
            noisy = gaussian(vec, c_t * math.sqrt(q), budget["marginals"], acct, "fit", f"{t}: marginals")
            sig_m = sigma_for(c_t * math.sqrt(q), budget["marginals"])
            at = 0
            totals = []
            n_rows = None
            for (c, mk), part in zip(makers, hist_parts):
                k = noisy[at:at + len(part)]
                at += len(part)
                if c == "#rows":
                    n_rows = float(k[0])
                    continue
                var_of[c] = sig_m ** 2
                if c.startswith("^"):
                    var_of[c + "#y"] = k
                    continue
                marg[c] = mk(k)
                totals.append(marg[c].total)
        else:
            totals, n_rows = [], None
        for c, (_, lv, noisy_lv, _nnull) in null_counts.items():
            # levels and their counts come from the selection; the null cell is their complement
            n_guess = n_rows if n_rows is not None else float(np.sum(np.clip(noisy_lv, 0, None)))
            null_c = max(n_guess - float(np.sum(np.clip(noisy_lv, 0, None))), 0.0)
            marg[c] = Marginal("cat", None, np.concatenate([[null_c], noisy_lv]), levels=lv)
            totals.append(marg[c].total)
        attr_names = P.numeric + P.categorical + [_keys(r["child_key"])[0] for r in P.public_fks]
        coords = [("attr", c) for c in attr_names]
        # ---------------- degrees of the children (this table is their parent)
        degree_cap = {}
        if P.children:
            share = budget["degrees"] / len(P.children)
            for r in P.children:
                ch = r["child"]
                ck, pk = _keys(r["child_key"]), _keys(r["parent_key"])
                deg = _degrees(df, real[ch], pk, ck)
                dmax = cfg.max_degree_range
                e = _degree_edges(dmax)
                b = np.clip(np.searchsorted(e, np.minimum(deg, dmax), side="right") - 1, 0, len(e) - 2)
                cnt = np.bincount(b, minlength=len(e) - 1)
                noisy = gaussian(cnt, c_t, share, acct, "fit", f"{ch}: rows per {t} row")
                var_of[f"#{ch}"] = sigma_for(c_t, share) ** 2
                want = cfg.max_degree.get(ch, "auto")
                if want == "auto":
                    # post-processing: cells that do not stand above the noise are dropped when choosing
                    # the cap (a Bonferroni bound: all-noise cells pass with probability about 1 % in all)
                    sig = sigma_for(c_t, share)
                    z = float(norm.isf(0.01 / len(noisy)))
                    pr = np.where(noisy > z * sig, noisy, 0).astype(float)
                    if pr.sum() <= 0:
                        # the noise hides the whole histogram: a small cap keeps the later noise small
                        D = int(cfg.degree_fallback)
                        tr.setdefault("degree_fallback", []).append(ch)
                    else:
                        cum = np.cumsum(pr) / pr.sum()
                        j = int(np.searchsorted(cum, cfg.degree_quantile))
                        D = int(max(1, e[min(j + 1, len(e) - 1)] - 1))
                else:
                    D = int(want)
                degree_cap[ch] = D
                e2 = _degree_edges(D)
                # post-processing: fold the released cells above D into D's cell
                k2 = np.zeros(len(e2) - 1)
                for i in range(len(e) - 1):
                    v = min(int(e[i]), D)
                    jj = int(np.clip(np.searchsorted(e2, v, side="right") - 1, 0, len(e2) - 2))
                    k2[jj] += noisy[i]
                marg[f"#{ch}"] = Marginal("num", None, np.concatenate([[0.0], k2]), edges=e2, integer=True,
                                          dtype="integer")
                coords.append(("deg", f"#{ch}"))
                # bound the child: keep at most D rows per parent row, chosen at random
                kept = _truncate(real[ch], df, pk, ck, D, hkey)
                rows[ch] = kept
                caps[ch] = c_t * D
                df[f"#{ch}"] = np.minimum(deg, D)
                tr.setdefault("max_degree", {})[ch] = D
        if tree:
            # ---------------- two-way marginals (mst / aim; module select)
            from .select import Node, grouping, select_and_measure
            nodes: dict = {}
            cells: dict = {}
            groups: dict = {}

            def coarse_node(name, m_, y, var, fine, evidence=False, deg=False):
                g = grouping(m_.p, m_.kind != "cat", cfg.pair_bins)
                K = int(g.max()) + 1
                yc = np.bincount(g, weights=np.asarray(y, dtype=float), minlength=K)
                zero = None
                if deg:
                    zero = np.zeros(K, dtype=bool)
                    zero[0] = True
                nodes[name] = Node(name, K, evidence=evidence, y=yc, var=var * max(np.bincount(g).max(), 1),
                                   zero=zero)
                cells[name] = g[fine]
                groups[name] = g
            for kind, c in coords:
                m_ = marg[c]
                fine = m_._cell(df[c].to_numpy() if kind == "deg" else _for_score(df[c], m_))[0]
                coarse_node(c, m_, m_.raw, float(var_of.get(c, 1.0)), fine, deg=kind == "deg")
                cells_of.setdefault(t, {})[c] = fine
            for c in pc_tree:
                coarse_node(f"^{c}", models[par].marg[c], var_of[f"^{c}#y"], float(var_of[f"^{c}"]),
                            cells_of[par][c][pidx], evidence=True)
            n_hat_f = float(max(n_rows or 0.0, 1.0))      # the noisy row count (never len(df))
            forest = select_and_measure(cfg.measure, t, nodes, cells, n_hat_f, c_t, budget["correlations"], acct,
                                        gaussian, max_cells=cfg.max_pair_cells, select_share=cfg.select_share,
                                        rounds=cfg.aim_rounds, report=tr)
            forest.groups = groups
            n_hat = int(round(n_rows)) if n_rows is not None else 0
            models[t] = TableModel(t, coords, marg, None, max(n_hat, 0), c_t, degree_cap, pc_tree, forest)
            tr["coordinates"] = len(nodes)
            report["tables"][t] = tr
            continue
        # ---------------- correlations
        parent_coords = []
        Zcols = {}
        for kind, c in coords:
            vals = df[c] if kind != "deg" else df[c]
            Zcols[c] = marg[c].score(vals.to_numpy() if kind == "deg" else _for_score(vals, marg[c]))
        if P.parent_rel is not None:
            par = P.parent_rel["parent"]
            pc = [c for k, c in models[par].coords if k == "attr"][:cfg.max_parent_attrs]
            if pc:
                ps = scores_of[par]
                ck, pk = _keys(P.parent_rel["child_key"]), _keys(P.parent_rel["parent_key"])
                idx = _parent_index(df, rows[par], ck, pk)
                for c in pc:
                    Zcols[f"^{c}"] = ps[c].to_numpy()[idx]
                parent_coords = pc
        names = [c for _, c in coords] + [f"^{c}" for c in parent_coords]
        corr = None
        if len(names) >= 2 and len(df):
            Z = np.column_stack([Zcols[c] for c in names])
            R = cfg.clip * math.sqrt(len(names))
            S, bound, iu = _vech_counts(Z, R)
            noisy = gaussian(S, c_t * bound, budget["correlations"], acct, "fit", f"{t}: correlations")
            M = np.zeros((len(names), len(names)))
            M[iu] = noisy / SCALE
            M = M + M.T - np.diag(np.diag(M))
            dg = np.diag(M).copy()
            good = dg > 0
            C = np.eye(len(names))
            sub = np.ix_(good, good)
            C[sub] = M[sub] / np.sqrt(np.outer(dg[good], dg[good]))
            corr = _nearest_corr(np.clip(C, -1, 1))
        elif len(names) >= 2:
            corr = np.eye(len(names))
        scores_of[t] = pd.DataFrame({c: Zcols[c] for _, c in coords if c in Zcols})
        n_hat = int(round(n_rows)) if n_rows is not None else 0
        models[t] = TableModel(t, coords, marg, corr, max(n_hat, 0), c_t, degree_cap, parent_coords)
        tr["coordinates"] = len(names)
        report["tables"][t] = tr
    pub = {t: real[t] for t in plan.public if t in real}
    return DPModel(plan, models, pub, acct, report)


def _canon(x):
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def _for_score(vals, m: Marginal):
    if m.kind == "cat":
        return vals.map(lambda x: None if _isnull(x) else str(_canon(x))).to_numpy(dtype=object)
    if m.dtype in ("date", "datetime"):
        return _numeric_values(vals, m.dtype)
    return vals.to_numpy()


def _date_bound(x):
    return float(np.clip(x, -6e10, 2.5e11))     # years ~ 0..9900 in seconds


def _public_range(cfg, t, c):
    v = cfg.public.get(f"{t}.{c}")
    if v and "min" in v and "max" in v:
        lo, hi = v["min"], v["max"]
        if isinstance(lo, str) or hasattr(lo, "year"):
            lo, hi = pd.Timestamp(lo).value / 1e9, pd.Timestamp(hi).value / 1e9
        return float(lo), float(hi)
    return None


def _key_list(df, cols) -> list:
    """Each row's key as a tuple of canonical values (hashable, for any number of columns)."""
    return list(zip(*[df[c].astype(object).map(_canon) for c in cols]))


def _degrees(parent_df, child_df, pk, ck):
    from collections import Counter
    cnt = Counter(_key_list(child_df, ck))
    return np.array([cnt.get(k, 0) for k in _key_list(parent_df, pk)], dtype=int)


def _truncate(child_df, parent_df, pk, ck, D, key):
    """Child rows whose parent is among the kept parent rows, at most D per parent:
    the D with the smallest keyed hash of their values.  The hash key is secret
    and drawn per run, so the choice is uniform among distinct rows, and a row's
    priority does not depend on any other row: removing one unit leaves every
    other unit's kept rows unchanged (a stable transformation, Prop. 901)."""
    pkeys = set(_key_list(parent_df, pk))
    ck_list = _key_list(child_df, ck)
    mask = np.array([k in pkeys for k in ck_list], dtype=bool)
    sub = child_df[mask].reset_index(drop=True)
    g = pd.Series([k for k, m in zip(ck_list, mask) if m], dtype=object)
    r = pd.Series(pd.util.hash_pandas_object(sub.astype(str), index=False, hash_key=key).to_numpy(), dtype="uint64")
    rank = r.groupby(pd.factorize(g)[0]).rank(method="first") if len(sub) else r
    return sub[(rank <= D).to_numpy()].reset_index(drop=True)


def _parent_index(child_df, parent_df, ck, pk):
    pos: dict = {}
    for i, k in enumerate(_key_list(parent_df, pk)):
        pos.setdefault(k, i)
    return np.array([pos[k] for k in _key_list(child_df, ck)], dtype=int)


# ------------------------------------------------------------------ decoding
def decode(spec: dict, model: DPModel) -> dict:
    """The seed database: post-processing of the released statistics only (and of
    the public tables)."""
    rng = _rng()
    plan = model.plan
    out: dict = {}
    zs: dict = {}
    cells_out: dict = {}
    keys: dict = {}
    for t in plan.order:
        P, M = plan.tables[t], model.tables[t]
        names = [c for _, c in M.coords] + [f"^{c}" for c in M.parent_coords]
        if P.parent_rel is None:
            n = max(M.n_hat, 1)
            known = None
            parent_idx = None
        else:
            par = P.parent_rel["parent"]
            deg = out[par][f"#{t}"].to_numpy(dtype=int)
            parent_idx = np.repeat(np.arange(len(deg)), deg)
            n = len(parent_idx)
            known = np.column_stack([zs[par][c][parent_idx] for c in M.parent_coords]) \
                if M.parent_coords and M.forest is None else None
        q = len(names)
        k = len(M.parent_coords)
        u = q - k
        if M.forest is not None:
            from .select import sample
            from .select import refine
            G = M.forest.groups
            ev = {f"^{c}": G[f"^{c}"][cells_out[P.parent_rel["parent"]][c][parent_idx]] for c in M.parent_coords} \
                if parent_idx is not None else {}
            cl = sample(M.forest, n, ev, rng)
            cells_out[t] = {c: refine(cl[c], G[c], M.marg[c].p, rng) for _, c in M.coords}
            Z = None
        elif M.corr is None or q < 2:
            Z = rng.standard_normal((n, u))
        elif known is None or k == 0:
            C = M.corr[:u, :u]
            Z = rng.multivariate_normal(np.zeros(u), C, size=n, method="eigh") if n else np.zeros((0, u))
        else:
            C = M.corr
            Cuu, Cuk, Ckk = C[:u, :u], C[:u, u:], C[u:, u:]
            A = Cuk @ np.linalg.inv(Ckk)
            S = _nearest_corr_cov(Cuu - A @ Cuk.T)
            Z = known @ A.T + (rng.multivariate_normal(np.zeros(u), S, size=n, method="eigh") if n else 0)
        cols = {}
        zt = {}
        for j, (_kind, c) in enumerate(M.coords):
            if Z is None:
                cols[c] = M.marg[c].draw_cells(cells_out[t][c], rng)
                continue
            zt[c] = Z[:, j]
            cols[c] = M.marg[c].draw(Z[:, j], rng)
        zs[t] = zt
        df = pd.DataFrame(index=range(n))
        T = spec["tables"][t]
        pk = _keys(T.get("primary_key") or [])
        child_pk = set()
        for r in P.children:
            child_pk |= set(_keys(r["parent_key"]))
        for i, c in enumerate(pk + [x for x in sorted(child_pk) if x not in pk]):
            df[c] = np.arange(1, n + 1) if i == 0 else 1
        if P.parent_rel is not None:
            par = P.parent_rel["parent"]
            for ck, pkc in zip(_keys(P.parent_rel["child_key"]), _keys(P.parent_rel["parent_key"])):
                df[ck] = out[par][pkc].to_numpy()[parent_idx]
        for c, v in cols.items():
            if c.startswith("#"):
                df[c] = np.nan_to_num(v, nan=0.0).astype(int) if len(v) else v
                continue
            kind = T["columns"][c]["type"]
            if kind in ("date", "datetime"):
                df[c] = pd.to_datetime(pd.Series(v), unit="s", errors="coerce")
                if kind == "date":
                    df[c] = df[c].dt.normalize()
            else:
                df[c] = v
        for c in P.placeholder:
            cs = T["columns"][c]
            gen = (cs.get("pii") or {}).get("generator")
            if cs["type"] == "binary":
                df[c] = None
            elif gen == "email":
                df[c] = [f"user{i}@example.org" for i in range(n)]
            else:
                df[c] = [f"{c} {i}" for i in range(n)]
        for c in P.other_ids:
            df[c] = [f"{c[:2]}{i}" for i in range(n)]
        for c in P.sequence:
            start = int(T["columns"][c].get("start", 1))
            if parent_idx is not None:
                df[c] = pd.Series(parent_idx).groupby(parent_idx).cumcount().to_numpy() + start
            else:
                df[c] = np.arange(n) + start
        for c in P.dropped:
            df[c] = None
        out[t] = df
        keys[t] = pk
    res = {t: df[[c for c in df.columns if not str(c).startswith("#")]] for t, df in out.items()}
    for t, df in model.public.items():
        res[t] = df
    _enforce_checks(spec, model.plan, res)
    for t in model.plan.views:
        # a view is rebuilt from its base table by stage 2; it reads no rows of its own
        res[t] = pd.DataFrame({c: pd.Series(dtype=object) for c in spec["tables"][t]["columns"]})
    return res


def _enforce_checks(spec: dict, plan: Plan, res: dict) -> None:
    """Drop seed rows that break a check rule of the spec, then the rows whose owner
    was dropped (post-processing).  The seed is decoded from low-dimensional
    statistics, so cross-table rules (an order's store in the customer's country)
    often fail in it; stage 2 would learn the violations and spend its sampling
    on rejecting them."""
    from ..expr import check_mask
    from ..prepare import make_resolver
    for t in plan.order:
        P = plan.tables[t]
        if P.parent_rel is not None:
            r = P.parent_rel
            ck, pk = _keys(r["child_key"]), _keys(r["parent_key"])
            keep = set(_key_list(res[r["parent"]], pk))
            df = res[t]
            # a boolean array, not a list: an empty list would select no columns
            res[t] = df[np.array([k in keep for k in _key_list(df, ck)], dtype=bool)].reset_index(drop=True)
        for rule in spec.get("rules") or []:
            if rule.get("table") != t or rule.get("type") != "check" or not len(res[t]):
                continue
            df = res[t]
            try:
                ok = check_mask(df, rule["expr"], make_resolver(spec, t, df, res))
            except (KeyError, ValueError) as e:
                log.debug("DP seed: rule %r not checked: %s", rule["expr"], e)
                continue
            if (~ok).any():
                log.info("DP seed: %s: %d of %d rows break %r and are dropped", t, int((~ok).sum()), len(df),
                         rule["expr"])
                res[t] = df[ok].reset_index(drop=True)


def _nearest_corr_cov(S):
    S = (S + S.T) / 2
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-6, None)
    return (V * w) @ V.T
