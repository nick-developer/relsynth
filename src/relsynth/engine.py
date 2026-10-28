# SPDX-License-Identifier: Apache-2.0
"""Fit a whole-database model from a spec + tables and synthesise a new
database with the same structure (layer 1 algorithm 5, top-down synthesis).

v0.3 adds, on top of the v0.2 model:
  * strata: latent segments nested in the rows of small copied parents
  * structural zeros/ones in every rate, presence and categorical parameter
  * occupancy (grain) models for facts with unique or near-unique pairs
  * virtual dimensions: a categorical column in a fact's unique key
  * nested facts: a fact referencing another fact through a composite key
  * uniqueness of foreign-key tuples and key rules on entities (exact
    conditioning by Metropolis-Hastings on the constrained product measure)
  * lookup columns, cross-table aggregate rules, a safe rule language
  * DP count model for under-dispersed single-dimension facts
  * marginal back-off to the more informative dimension for small fact blocks
"""
from __future__ import annotations

import logging
import time
import traceback

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln, ndtri

from .counts import choose_count_model, nb_size_pooled
from .data import coerce
from .expr import check_mask, derive_values, names_in, split_assignment
from .pii import PiiGenerator
from .prepare import (_index, _keystr, _tuple_frame, compute_aggregate, lookup_column, lookup_deps, make_resolver,
                      prepare_source)
from .relational import (Entity, OccRel, PairRel, RelationalModel, SingleRel, TreeRel,
                         dispersion_alpha, gumbel_argmax, multiplicity_marginal, poisson_dup_test,
                         sparse_dirichlet_draw, tree_depth)
from .spec import SpecError, resolve_path, simple_fks, table_edges, topo_order
from .table_model import TableModel
from .text import TextModel
from .transform import Plan, offset_keys

_NULL_KEY = "\x00null"                    # group key of null values where nulls form a group

BUCKET_FREQ = {"day": "D", "week": "W", "month": "MS", "quarter": "QS", "year": "YS"}
VDIM = "__vdim__"
# hidden per-row draws (stated distributions, soft-rule choices, group-effect
# priorities) are kept in these columns while a table is built, so that rules are
# checked on the very draws that are kept and recomputing a frame reuses them;
# HIDDEN_SET marks the rows whose draws are made (False/missing: draw afresh)
HIDDEN_SET = "__hidden_set__"
HIDDEN_PREFIXES = ("__dist_", "__soft_z_", "__soft_d_", "__ge_r_")


def _fresh_rows(df, hc):
    """Rows whose hidden draw in column hc is still to be made: all rows when hc is
    absent; otherwise the rows whose HIDDEN_SET mask is False or missing (none
    when the frame has no mask: its draws are all made)."""
    if hc not in df.columns:
        return np.ones(len(df), bool)
    if HIDDEN_SET not in df.columns:
        return np.zeros(len(df), bool)
    return ~df[HIDDEN_SET].fillna(False).to_numpy(dtype=bool)


def _natural_sort(keys):
    """Key labels in their natural order: labels made of a common text prefix and a
    number ("O9" < "O10") by the number, anything else as np.sort does."""
    keys = np.asarray(keys)
    if keys.dtype == object and len(keys) and all(isinstance(k, str) for k in keys):
        parts = pd.Series(keys).str.extract(r"^(.*?)(\d+)$")
        if parts[1].notna().all():
            order = np.lexsort((keys, parts[1].astype(np.int64).to_numpy(), parts[0].to_numpy(dtype=object)))
            return keys[order]
    return np.sort(keys)


def _remap(m, col):
    """Values of col through the mapping m (unmapped values and nulls, pd.NA
    included, kept as they are)."""
    return pd.array([x if pd.isna(x) else m.get(x, x) for x in col.tolist()], dtype=object)


def _hidden_cols(df):
    return [c for c in df.columns if c == HIDDEN_SET or c.startswith(HIDDEN_PREFIXES)]


def _mark_fresh(df, idx):
    """Rows idx are redrawn: their hidden draws are made again on the next recompute."""
    if HIDDEN_SET in df.columns and len(idx):
        hs = df[HIDDEN_SET].fillna(False).to_numpy(dtype=bool, copy=True)
        hs[idx] = False
        df[HIDDEN_SET] = hs


def _scores(x):
    """Rank-based normal scores of a numeric vector (nan -> 0)."""
    ok = ~np.isnan(x)
    out = np.zeros(len(x))
    if ok.sum():
        r = pd.Series(x[ok]).rank(method="average").to_numpy()
        out[ok] = ndtri(r / (ok.sum() + 1))
    return out


def _is_null(v):
    return v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and v != v)


def _has_null(k):
    """A key with a null part never collides (SQL unique semantics); NaN and pd.NA
    are nulls as None is."""
    return any(_is_null(v) for v in k) if isinstance(k, tuple) else _is_null(k)


def _bias_corrected_mi(x, y):
    """Mutual information (nats) between two label vectors minus its
    Miller-Madow small-sample bias (R-1)(C-1)/(2n)."""
    n = len(x)
    if n == 0:
        return 0.0
    ct = pd.crosstab(pd.Series(x), pd.Series(y)).to_numpy().astype(float)
    p = ct / n
    pr, pc = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (pr @ pc)[nz])).sum()
    return float(mi - (ct.shape[0] - 1) * (ct.shape[1] - 1) / (2 * n))


def _surrogate_label(t, c):
    """Readable stem for generated labels: the column name, or the table name for
    generic columns (Artist.Name -> "Artist", Customer.Company -> "Company")."""
    import re as _re
    base = _re.sub(r"^(dim|fact|fct|tbl|ref|lkp)_", "", t, flags=_re.I)
    return base if _re.fullmatch(r"(?i)(name|title|label|description|desc|text)", c) else c


def _dp_floor(dp, mandatory):
    """DP count state whose base measure keeps mandatory participation: with every
    parent observed with a child (Prop. on mandatory participation) the base is
    1 + Poisson(lambda - 1), so new atoms are never 0 (one Contoso order had no lines)."""
    if dp is None or not mandatory:
        return dp
    return {**dp, "min": 1}


def _cover_segments(seg_of, zc, need, phi, rng):
    """Moves rows between parent segments until segment t holds at least need[t]
    rows (seg_of = -1: a null link, never moved).  Each deficit is filled from the
    segments with a surplus, a row moving from s to t with probability
    proportional to phi[z, t] / phi[z, s] (the rows most likely under t), so the
    number of moved rows is the total deficit and nothing changes when every
    segment is covered."""
    Kp = len(need)
    cnt = np.bincount(seg_of[seg_of >= 0], minlength=Kp)
    for t in np.flatnonzero(cnt < need):
        while cnt[t] < need[t]:
            src = np.flatnonzero((seg_of >= 0) & (seg_of != t) & (cnt[np.maximum(seg_of, 0)] > need[np.maximum(seg_of, 0)]))
            if not len(src):
                return seg_of
            k = int(min(need[t] - cnt[t], len(src)))
            s_ = seg_of[src]
            lw = np.log(np.maximum(phi[zc[src], t], 1e-300)) - np.log(np.maximum(phi[zc[src], s_], 1e-300))
            pick = src[np.argsort(-(lw + rng.gumbel(size=len(src))), kind="stable")]
            moved = 0
            for i in pick:              # a source segment never drops below its own need
                if moved == k:
                    break
                s0 = seg_of[i]
                if cnt[s0] > need[s0]:
                    seg_of[i] = t
                    cnt[s0] -= 1
                    cnt[t] += 1
                    moved += 1
            if not moved:
                return seg_of
    return seg_of


def _pooled_beta(Z, N, cmax=1e4):
    """Common Beta(c m, c (1 - m)) prior of K exchangeable shares with Z_k events
    in N_k trials: m = (sum Z + 1/2) / (sum N + 1) (Jeffreys on the pooled share)
    and c = m (1 - m) / tau^2 - 1, tau^2 the between-share variance beyond the
    binomial noise (Kleinman 1973), in [1, cmax]; tau^2 <= 0 gives cmax."""
    Z, N = np.asarray(Z, float), np.asarray(N, float)
    m = (Z.sum() + 0.5) / (N.sum() + 1.0)
    ok = N > 0
    if ok.sum() >= 2:
        r = Z[ok] / N[ok]
        w = N[ok] / N[ok].sum()
        tau2 = float((w * (r - m) ** 2).sum() - (w * m * (1 - m) / N[ok]).sum())
    else:
        tau2 = 0.0
    c = cmax if tau2 <= 0 else float(np.clip(m * (1 - m) / tau2 - 1, 1.0, cmax))
    return c * m, c * (1 - m)


def _mandatory(counts, z, K, min_expected=3.0):
    """Every parent row has at least one child (every album has a track) although
    the fitted Poisson rates would leave about sum_i exp(-mean of i's segment)
    parents empty: when that expectation is >= min_expected (observing no empty
    parent then has probability <= e^-3, about 5%), participation is mandatory."""
    counts = np.asarray(counts, float)
    if not len(counts) or counts.min() < 1:
        return False
    mean = np.bincount(z, weights=counts, minlength=K) / np.maximum(np.bincount(z, minlength=K), 1)
    return bool(np.exp(-mean[z]).sum() >= min_expected)


def _pg_logbf(y, codes):
    """log Bayes factor of a Poisson-gamma count model with one rate per level of
    `codes` against one pooled rate; Gamma(1/2, (1/2)/mean) prior on each rate."""
    y = np.asarray(y, float)
    m = max(y.mean(), 1e-9)
    a, b = 0.5, 0.5 / m

    def lm(yy):
        return a * np.log(b) - gammaln(a) + gammaln(a + yy.sum()) - (a + yy.sum()) * np.log(b + len(yy))
    return float(sum(lm(y[codes == k]) for k in np.unique(codes)) - lm(y))


def _cumulative_parts(df, rule, resolve=None):
    from .profiling import canon
    by = rule["by"] if isinstance(rule["by"], list) else [rule["by"]]
    step = pd.to_numeric(pd.Series(derive_values(df, f"_s_ = {rule['step']}", resolve)[1]),
                         errors="coerce").fillna(0.0).to_numpy(dtype=float)
    key = pd.MultiIndex.from_frame(df[by].astype(object).where(df[by].notna(), None).astype(str)).codes \
        if len(by) > 1 else [pd.factorize(df[by[0]].astype(str))[0]]
    key = np.ravel_multi_index(key, [max(int(k.max()) + 1, 1) for k in key]) if len(key) > 1 else np.asarray(key[0])
    o = pd.to_numeric(pd.Series(canon(df[rule["order"]])), errors="coerce").fillna(np.inf).to_numpy(dtype=float)
    ordr = np.lexsort((o, key))
    k = key[ordr]
    first = np.r_[True, k[1:] != k[:-1]]
    return step, ordr, k, first


def cumulative_openings(df, rule, resolve=None):
    """Opening balance of every group: y on its first row minus that row's step."""
    if not len(df):
        return np.array([])
    step, ordr, k, first = _cumulative_parts(df, rule, resolve)
    y = pd.to_numeric(df[rule["column"]], errors="coerce").to_numpy(dtype=float)
    return (y[ordr] - step[ordr])[first]


def cumulative_values(df, rule, resolve=None):
    """y_t = y_{t-1} + step_t within each group (ordered by rule['order']): a running
    balance.  The first row starts from the rule's opening balance when it has one
    (fitted when every group of the source opens at the same value, e.g. 0), else
    from y as generated on that row."""
    if rule.get("opening") is not None:
        step, ordr, k, first = _cumulative_parts(df, rule, resolve)
        c = np.where(first, float(rule["opening"]) + step[ordr], step[ordr])
        run = pd.Series(c).groupby(k).cumsum().to_numpy()
        res = np.empty(len(df))
        res[ordr] = run
        return res
    from .profiling import canon
    by = rule["by"] if isinstance(rule["by"], list) else [rule["by"]]
    step = pd.to_numeric(pd.Series(derive_values(df, f"_s_ = {rule['step']}", resolve)[1]),
                         errors="coerce").fillna(0.0).to_numpy(dtype=float)
    key = pd.MultiIndex.from_frame(df[by].astype(object).where(df[by].notna(), None).astype(str)).codes \
        if len(by) > 1 else [pd.factorize(df[by[0]].astype(str))[0]]
    key = np.ravel_multi_index(key, [max(int(k.max()) + 1, 1) for k in key]) if len(key) > 1 else np.asarray(key[0])
    o = pd.to_numeric(pd.Series(canon(df[rule["order"]])), errors="coerce").fillna(np.inf).to_numpy(dtype=float)
    ordr = np.lexsort((o, key))
    y = pd.to_numeric(df[rule["column"]], errors="coerce").to_numpy(dtype=float)
    k = key[ordr]
    first = np.r_[True, k[1:] != k[:-1]]
    c = np.where(first, np.nan_to_num(y[ordr]), step[ordr])
    run = pd.Series(c).groupby(k).cumsum().to_numpy()
    res = np.empty(len(df))
    res[ordr] = run
    return res


def _subtracted_column(step, df):
    """The column a step takes away (b in 'a - b', a in '0 - a'), when it is a plain column."""
    import re as _re
    m = _re.search(r"-\s*([A-Za-z_]\w*)\s*$", str(step))
    return m.group(1) if m and m.group(1) in df.columns else None


def cumulative_apply(df, rule, resolve=None):
    """Running balance of a cumulative rule written into df.  With a floor (a stock
    that never goes below 0 in the source), a row whose step would take the balance
    under the floor has its outflow (the subtracted column) reduced to what is
    available: withdrawals are capped by the balance, as the source's process does.
    Without a floor the balance is the plain running sum."""
    if not len(df):
        return df
    fl = rule.get("floor")
    out_c = _subtracted_column(rule["step"], df) if fl is not None else None
    if out_c is None:
        df[rule["column"]] = cumulative_values(df, rule, resolve)
        return df
    step, ordr, k, first = _cumulative_parts(df, rule, resolve)
    y0 = pd.to_numeric(df[rule["column"]], errors="coerce").to_numpy(dtype=float)
    outv = pd.to_numeric(df[out_c], errors="coerce").to_numpy(dtype=float).copy()
    st, fs = step[ordr], first
    opening = rule.get("opening")
    bal = np.empty(len(ordr))
    cut = np.zeros(len(ordr))
    b = 0.0
    for i in range(len(ordr)):
        if fs[i]:
            b = float(opening) if opening is not None else float(np.nan_to_num(y0[ordr[i]]) - st[i])
        nb = b + st[i]
        if nb < fl:
            cut[i] = min(fl - nb, np.nan_to_num(outv[ordr[i]]))
            nb += cut[i]
        bal[i] = nb
        b = nb
    res = np.empty(len(df))
    res[ordr] = bal
    outv[ordr] = outv[ordr] - cut
    df[rule["column"]] = res
    if cut.any():
        try:
            df[out_c] = pd.Series(outv, index=df.index).astype(df[out_c].dtype)
        except (TypeError, ValueError):
            df[out_c] = outv
    return df


def _lookup_order(lk):
    """Lookups ordered so that one whose path starts at another lookup column
    (a computed key) comes after it."""
    out, left = {}, dict(lk)
    while left:
        ready = [c for c, p in left.items() if not (lookup_deps(p) & set(left))]
        if not ready:
            ready = list(left)
        for c in ready:
            out[c] = left.pop(c)
    return out


_log = logging.getLogger("relsynth")


def say(msg):
    """Progress of long fits and samples, one INFO record per stage on the
    'relsynth' logger (the CLI shows them on stderr; RELSYNTH_QUIET=1 or
    --log-level warning hides them)."""
    _log.info(msg)


class Synthesizer:
    def __init__(self, spec):
        from .spec import validate
        self.user_spec = spec
        self.plan = Plan(spec)
        self.spec = self.plan.spec
        errs = validate(self.spec, keys=False)
        if errs:
            raise SpecError("internal model (order headers / offset keys) is invalid:\n  - " + "\n  - ".join(errs))
        self.cfg = self.spec["synthesis"]
        self.rng = np.random.default_rng(self.cfg["seed"])          # fitting stream
        self._post_fns = []          # parameter blocks drawn again for every dataset
        self._runs = 0
        self.log = {"warnings": [], "errors": [], "timings": {}, "source": {}, "decisions": {}}

    # ================================================================== fit
    def fit(self, tables: dict[str, pd.DataFrame], stats=None):
        """Fit the model.  `stats` (pushdown) maps fact tables given as uniform
        samples to their exact statistics (relsynth.pushdown.stats.TableStats):
        the relational model reads those facts only through the statistics."""
        t0 = time.time()
        spec, cfg, rng = self.spec, self.cfg, self.rng
        T = spec["tables"]
        self.fstats = {t: s for t, s in (stats or {}).items() if t in T and T[t]["role"] == "fact"}
        n_src = sum(self.fstats[t].n_rows if t in self.fstats else len(d) for t, d in tables.items())
        say(f"fit: preparing {len(tables)} tables ({n_src:,} rows"
            + (f"; {len(self.fstats)} fact table(s) from statistics and samples)" if self.fstats else ")"))
        tables = self.plan.to_internal({t: df.copy() for t, df in tables.items()},
                                       headers=self._pushed_headers(tables))
        for f, hl in self.plan.log.items():
            h = self.plan.headers[f]
            viol = "an unknown number of" if hl["header_violations"] is None else hl["header_violations"]
            self.log["decisions"].setdefault(f, {})["header"] = (
                f"{hl['header_rows']} {h['key']} values generated as header rows ({h['table']}); "
                f"{viol} cells differ from their header in the source")
        for k, v in self.plan.notes.items():
            self.log["decisions"].setdefault(k.split(".")[0], {})[k.split(".", 1)[1]] = v
        tables, prep = prepare_source(spec, tables, partial=set(self.fstats))
        self.log["source_computed"] = prep
        self._tables_for_fit = tables
        self.src_rows = {t: (self.fstats[t].n_rows if t in self.fstats else len(df)) for t, df in tables.items()}
        if self.fstats:
            self.log["pushdown"] = {t: {"rows": st.n_rows, "sample_rows": len(tables[t])} for t, st in self.fstats.items()}
        rels = spec["relationships"]
        self.pi_s = cfg["structure"]["structural_prior"]
        self.entities = [t for t in T if T[t]["role"] == "entity"]
        self.facts = [t for t in T if T[t]["role"] == "fact"]
        self.fk = {t: simple_fks(spec, t) for t in T}
        self.nest = {t: next((r for r in rels if r["child"] == t and r.get("composite")), None) for t in T}
        self.lookups = {t: _lookup_order({c: cs["lookup"] for c, cs in T[t]["columns"].items()
                                          if cs["type"] == "lookup"}) for t in T}
        self.derived_keys = {t: [r for r in rels if r["child"] == t and r.get("kind") == "derived"] for t in T}
        self.derived = {t: [split_assignment(r["expr"])[0] for r in spec["rules"]
                            if r["table"] == t and r["type"] == "derive"] for t in T}
        self.aggregated = {t: [r["column"] for r in spec["rules"] if r["table"] == t and r["type"] == "aggregate"]
                           for t in T}
        self.order = topo_order(self.entities, [(r["child"], r["parent"]) for r in rels
                                                if r["kind"] == "many_to_one" and r["child"] in self.entities
                                                and not r.get("composite")])
        self.all_order = topo_order(list(T), table_edges(spec))
        self.fact_order = [t for t in self.all_order if t in self.facts]
        self.copied = {e for e in self.entities if T[e].get("mode") == "copy"}
        self.grain = self._grain_layout(tables)
        # ---- keys
        self.key_index = {}
        for e in self.entities:
            pk = T[e]["primary_key"]
            keys = tables[e][pk]
            dup = int(keys.duplicated().sum())
            if dup:
                self.log["warnings"].append(f"{e}: {dup} duplicate primary keys (later rows ignored as parents)")
            self.key_index[e] = pd.Index(keys.drop_duplicates())
            self.log["source"].setdefault(e, {})["rows"] = len(keys)
        # ---- fact layout (anchor, partner, virtual dimension, nesting)
        self.fact_info, self.fact_key_index, self.fact_key_cols = {}, {}, {}
        for f in self.fact_order:
            self.fact_info[f] = self._fact_layout(f, tables)
        for c in self.facts:
            nr = self.nest[c]
            if nr is not None and nr["parent"] not in self.fact_key_index:
                p = nr["parent"]
                keys = _tuple_frame(self._path_values(p, tables[p], nr["parent_key"], tables))
                ix = _index(keys)
                if ix.has_duplicates:
                    self.log["warnings"].append(f"{p}: key {nr['parent_key']} is not unique in the source")
                self.fact_key_index[p] = ix
                self.fact_key_cols[p] = nr["parent_key"]
        self.attr_strata = self._attribute_strata(tables)
        self.copied_per_row = self._copied_layout(tables)
        # stored precision of aggregate columns (a sum of prices is stored to the cent)
        self.agg_decimals = {}
        for r in spec["rules"]:
            if r["type"] == "aggregate" and r["column"] in tables[r["table"]]:
                v = pd.to_numeric(tables[r["table"]][r["column"]], errors="coerce").dropna().to_numpy()[:5000]
                if len(v):
                    self.agg_decimals[(r["table"], r["column"])] = next(
                        (d for d in range(7) if np.allclose(v, np.round(v, d), atol=1e-9, rtol=0)), None)
        self.derive_decimals = {}
        for r in spec["rules"]:
            if r["type"] == "derive":
                lhs = split_assignment(r["expr"])[0]
                if lhs in tables[r["table"]] and T[r["table"]]["columns"].get(lhs, {}).get("type") == "float":
                    v = pd.to_numeric(tables[r["table"]][lhs], errors="coerce").dropna().to_numpy()[:20000]
                    if len(v):
                        self.derive_decimals[(r["table"], lhs)] = next(
                            (d for d in range(7) if np.allclose(v, np.round(v, d), atol=1e-9, rtol=0)), None)
        # opening balance of running balances: when every group of the source opens
        # at one value (stock starts at 0), the synthetic groups open there too
        for r in spec["rules"]:
            if r["type"] == "cumulative" and r.get("opening") is None and r["column"] in tables[r["table"]]:
                try:
                    op = cumulative_openings(self._opening_rows(r, tables), r)
                except (SpecError, KeyError, ValueError, TypeError):
                    continue
                op = op[np.isfinite(op)]
                if len(op) >= 2:
                    vals, cnt = np.unique(np.round(op, 9), return_counts=True)
                    if cnt.max() >= 0.999 * len(op):
                        r["opening"] = float(vals[cnt.argmax()])
                # a balance that is never negative in the source (a stock) keeps a floor of 0
                ny, ymin = self._balance_range(r, tables)
                if r.get("floor") is None and ny >= 30 and ymin is not None and ymin >= 0 and \
                        _subtracted_column(r["step"], tables[r["table"]]) is not None:
                    r["floor"] = 0.0
        # ---- attribute models (unfitted; give features for segmentation)
        say("fit: encoding columns")
        self.tm = {}
        for t in T:
            excl = self._excluded(t)
            aux = self._aux_columns(t, tables) if T[t]["role"] == "fact" else None
            self.tm[t] = TableModel(t, T[t], tables[t], cfg, exclude=excl, aux=aux)
        # ---- latent entity objects (strata = rows of small copied parents)
        E = {}
        sc = cfg["segments"]
        self.strata_info = {}
        for e in self.entities:
            n = len(tables[e])
            cap = sc["tables"].get(e, sc["default_cap"])
            Xn, Xc, ncat = self.tm[e].features()
            if e in self.copied and e in self.copied_per_row:
                # small reference table copied as-is: every row is its own fixed
                # segment, so links to it are modelled per row (exact structure)
                E[e] = Entity(e, n, n, Xn, Xc, ncat)
                E[e].fixed = True
                E[e].z = np.arange(n)
                E[e].copied = True
                continue
            if e in self.copied:
                # larger copied table (calendar, product list, geography): its rows keep
                # their identity but are clustered into latent segments like any entity,
                # and each row's share of its segment's links is its own observed degree
                # (degree-corrected block model, Karrer & Newman 2011).  One segment per
                # row would make every block array dense in the table's size.
                cap = max(1, min(cap, n // max(sc["min_rows_per_segment"], 1)))
                E[e] = Entity(e, n, cap, Xn, Xc, ncat)
                E[e].copied = True
                continue
            st = self._strata(e, tables)
            if st is not None:
                codes, labels, by = st[:3]
                ns = np.bincount(codes, minlength=len(labels))
                caps = np.maximum(1, np.minimum(cap, ns // max(sc["min_rows_per_segment"], 1)))
                E[e] = Entity(e, n, None, Xn, Xc, ncat, strata=codes, caps=caps)
                self.strata_info[e] = {"by": by, "labels": labels, "caps": caps.tolist()}
                if len(st) > 3:
                    self.strata_info[e]["attr"] = st[3]
            else:
                cap = max(1, min(cap, n // max(sc["min_rows_per_segment"], 1)))
                E[e] = Entity(e, n, cap, Xn, Xc, ncat)
        self._copy_rows = {e: tables[e].copy() for e in self.copied}
        self.E = E
        # virtual dimensions are fixed entities whose rows are the levels
        for f, info in self.fact_info.items():
            if info.get("vdim"):
                lv = info["vdim_levels"]
                ve = Entity(f"{VDIM}{f}", len(lv), len(lv), np.zeros((len(lv), 0)),
                            np.zeros((len(lv), 0), np.int64), [])
                ve.fixed = True
                ve.z = np.arange(len(lv))
                E[ve.name] = ve
        R = []
        self.edge_info = {}
        for r in rels:
            c, p = r["child"], r["parent"]
            if r.get("composite"):
                continue
            if c in self.entities and r["kind"] == "many_to_one":
                pidx = self._lookup(p, tables[c][r["child_key"]])
                ok = pidx >= 0
                nulls = int(tables[c][r["child_key"]].isna().sum())
                orphans = int((~ok).sum()) - nulls
                if orphans:
                    self.log["warnings"].append(f"{r['name']}: {orphans} orphan foreign keys treated as null")
                rel = PairRel(r["name"], "edge", E[c], E[p], np.flatnonzero(ok), pidx[ok], np.ones(ok.sum()))
                rel.null_mask = ~ok
                rel.child_only = c in self.grain and r["child_key"] in self.grain[c]["cols"]
                R.append(rel)
                self.edge_info[r["name"]] = {"rel": rel, "spec": r, "orphans": orphans, "nulls": nulls}
            elif r["kind"] == "self_reference":
                pidx = self._lookup(p, tables[c][r["child_key"]])
                rel = TreeRel(r["name"], E[c], pidx.copy())
                R.append(rel)
                self.edge_info[r["name"]] = {"rel": rel, "spec": r}
        for e, g in self.grain.items():
            ia = self._lookup(g["anchor"]["parent"], tables[e][g["anchor"]["child_key"]])
            ib = self._lookup(g["partner"]["parent"], tables[e][g["partner"]["child_key"]])
            ok = (ia >= 0) & (ib >= 0)
            pairs = pd.Series(1, index=pd.MultiIndex.from_arrays([ia[ok], ib[ok]])).groupby(level=[0, 1]).size()
            rel = OccRel(f"{e}.grain", E[g["anchor"]["parent"]], E[g["partner"]["parent"]],
                         pairs.index.get_level_values(0).to_numpy(), pairs.index.get_level_values(1).to_numpy(),
                         pairs.to_numpy(), table=e, mult_by="none")
            g["rel"] = rel
            R.append(rel)
            self.log["decisions"][e] = {"model": "grain (occupancy of its unique key)",
                                        "unique_key": g["cols"], "anchor": g["anchor"]["parent"],
                                        "partner": g["partner"]["parent"]}
        for f in self.fact_order:
            R += self._fact_relations(f, tables[f])
        t1 = time.time()
        say(f"fit: latent segments ({len(E)} entities, {len(R)} relations, "
            f"{cfg['gibbs']['iterations'] + cfg['gibbs']['final_iterations']} Gibbs sweeps)")
        rm = RelationalModel(E, R, cfg, rng)
        self._tie_headers(rm, tables)
        self.rm = rm.fit()
        self.log["timings"]["gibbs_s"] = round(time.time() - t1, 2)
        # ---- per-segment attribute models
        t2 = time.time()
        for e in self.entities:
            if e not in self.copied:
                say(f"fit: attributes of {e} ({len(tables[e]):,} rows, {len(self.tm[e].cols)} modelled columns)")
                self.tm[e].fit(groups=E[e].z, rng=rng, per_group_copula=True)
        self._post_edges()
        self.fact_entity = {}
        for f in self.fact_order:
            say(f"fit: fact {f} ({len(tables[f]):,} rows)")
            self._post_fact(f, tables[f])
        say("fit: text and identifiers")
        self._fit_text_pii(tables)
        self._fit_constraints(tables)
        self._fit_inclusion(tables)
        self._fit_soft(tables)
        self._fit_group_effects(tables)
        self._fit_affinity(tables)
        self._fit_participation(tables)
        self._fit_scd(tables)
        self.dist_src = {(t, c): tables[t][c].dropna().to_numpy(dtype=object)
                         for t, T_ in self.spec["tables"].items() for c, cs in T_["columns"].items()
                         if (cs.get("distribution") or {}).get("kind") == "empirical" and t in tables
                         and c in tables[t].columns}
        self.log["timings"]["attributes_s"] = round(time.time() - t2, 2)
        self.log["timings"]["fit_total_s"] = round(time.time() - t0, 2)
        say(f"fit: done in {time.time() - t0:.0f} s")
        return self

    def _opening_rows(self, r, tables):
        """The rows a running balance's openings are read from: every row, or for a
        pushed-down fact the first row of each group (in the rule's order) computed
        in the database."""
        t = r["table"]
        if t not in self.fstats:
            return tables[t]
        by = r["by"] if isinstance(r["by"], list) else [r["by"]]
        U = self.user_spec["tables"][t]["columns"]
        used = {r["column"], r["order"], *by} | {n.split(".")[0] for n in names_in(r["step"])}
        cols = [c for c in tables[t].columns if c in used and c in U and U[c]["type"] != "lookup"]
        H, _ = self.fstats[t].first_rows(by, cols, order=[r["order"]], differences=False)
        return coerce(H, {"columns": {c: U[c] for c in H.columns}}, table=t)

    def _balance_range(self, r, tables):
        """(non-null rows, smallest value) of a running balance column over every row."""
        t, c = r["table"], r["column"]
        if t in self.fstats:
            st = self.fstats[t]
            lo, _ = st.minmax(c)
            return st.notnull(c), (float(lo) if lo is not None else None)
        yv = pd.to_numeric(tables[t][c], errors="coerce").dropna()
        return len(yv), (float(yv.min()) if len(yv) else None)

    def _pushed_headers(self, tables):
        """Header rows of pushed-down facts with an order header, from all rows
        (the sample would miss whole orders): {fact: (header frame, violations)}."""
        out = {}
        U = self.user_spec["tables"]
        for f, h in self.plan.headers.items():
            if f not in self.fstats:
                continue
            st = self.fstats[f]
            have = set(tables[f].columns)
            cols = [c for c in h["columns"] if c in have and U[f]["columns"][c]["type"] != "lookup"]
            H, diff = st.first_rows(h["key"], cols)
            tsp = {"columns": {c: U[f]["columns"][c] for c in [h["key"]] + cols}}
            out[f] = (coerce(H, tsp, table=f), diff)
        return out

    def _tie_headers(self, rm, tables):
        """Order headers split off a fact (transform.Plan) take the segment of the
        parent the fact was anchored on (its customer): the header carries no
        information of its own about who buys what, and with ~2 lines per order the
        sampler cannot learn the customer -> order -> product path back on its own
        (the header segments collapse and product choice stops depending on the
        customer).  Tying restores the dependence the unsplit fact had."""
        if self.cfg["segments"].get("tie_headers", True) is False:
            return
        for _f, h in self.plan.headers.items():
            H, p = h["table"], h["scale_from"]
            if p is None or H not in rm.E or p not in rm.E:
                continue
            c, par = rm.E[H], rm.E[p]
            if par.fixed or getattr(par, "copied", False) or par.strata is not None or par.tree is not None \
                    or c.strata is not None or H in self.grain:
                continue
            r = next((x for x in self.fk[H] if x["parent"] == p), None)
            info = self.edge_info.get(r["name"]) if r else None
            if info is None or info["orphans"] or info["nulls"]:
                continue
            idx = self._lookup(p, tables[H][r["child_key"]])
            rm.tie(c, par, idx, info["rel"])
            info["tied"] = True
            self.log["decisions"].setdefault(H, {})["segments"] = f"tied to {p} (each order takes its {p} row's segment)"

    def _bb(self, key, n, rng):
        """Bayesian-bootstrap weights over n observed counts, fixed for the dataset."""
        rc = getattr(self, "run_cache", None)
        return rc.bb(key, n, rng) if rc is not None else rng.dirichlet(np.ones(n))

    def _register(self, fn, now=True):
        """A parameter block drawn at fit (now) and again for every dataset."""
        if now:
            fn(self.rng)
        self._post_fns.append(fn)

    def redraw(self, mode="draw", rng=None):
        """One posterior draw (posterior means in mean mode) of every parameter
        given the fitted partitions: segment weights, link and block rates,
        occupancy and activity probabilities, marginals, copula matrices, time
        profiles, extra-dimension laws, refined segment weights and rake targets.
        A dataset sampled after it is a draw from the posterior predictive
        distribution (Rubin 1993); the per-call draws of the samplers (count-law
        and tail weights) are fixed for the run as well (see posterior.py)."""
        from .posterior import param_rng
        base = self.rng if rng is None else rng
        prng = param_rng(base, mode)
        keep, keep_rm = self.rng, self.rm.rng
        self.rng = self.rm.rng = prng
        try:
            self.rm.draw_params(pi_s=self.pi_s)
            self._post_edges()
            for f in self.fact_order:
                rel = self.fact_info[f].get("rel")
                if rel is not None and rel not in self.rm.R:
                    rel.struct_kappa = self.cfg["structure"].get("structural_pooling", 3.0)
                    rel.draw_params(prng, self.pi_s)
                if rel is not None and self.fact_info[f].get("occ_min_one"):
                    self._truncate_occ(rel)
            for fn in self._post_fns:
                fn(prng)
            for _t, tm in self.tm.items():
                if hasattr(tm, "redraw") and hasattr(tm, "_groups"):
                    tm.redraw(base, mode)
            for _t, rk in getattr(self, "rake", {}).items():
                cnt = rk["counts"]
                w = (cnt + 0.5) / (cnt + 0.5).sum() if mode == "mean" else base.dirichlet(cnt.to_numpy() + 0.5)
                rk["dist"] = pd.Series(w, index=cnt.index)
        finally:
            self.rng, self.rm.rng = keep, keep_rm
        return self

    # ------------------------------------------------------------------ layout
    def _excluded(self, t):
        T = self.spec["tables"][t]
        ex = {T.get("primary_key")} | {r["child_key"] for r in self.fk[t]} | set(self.derived[t]) \
            | set(self.aggregated[t]) | set(self.lookups[t])
        nr = self.nest.get(t)
        if nr:
            ex |= {c for c in nr["child_key"] if "." not in c}
        ex |= {c for c, cs in T["columns"].items()
               if cs["type"] in ("id", "text", "pii", "constant", "lookup", "binary", "sequence")}
        if t in getattr(self, "attr_strata", {}):
            ex.add(self.attr_strata[t])
        if T["role"] == "fact":
            if T["fact"].get("time_column"):
                ex.add(T["fact"]["time_column"])
            info = getattr(self, "fact_info", {}).get(t, {})
            if info.get("vdim"):
                ex.add(info["vdim"])
        return ex

    def _grain_layout(self, tables):
        """Entities whose declared unique key is a pair of foreign keys (bridge,
        snapshot or report tables: one row per entity x period, customer x
        account, ...).  Their rows are a subset of the product of the two parents,
        so they are generated by an occupancy model of that product (each pair
        present at most once), not by choosing parents independently per row."""
        out = {}
        if self.cfg["uniqueness"].get("grain", "auto") in ("none", False, None):
            return out
        T = self.spec["tables"]
        for e in self.entities:
            if e in self.copied:
                continue
            fks = {r["child_key"]: r for r in self.fk[e]}
            for u in T[e].get("unique", []):
                if len(u) != 2 or not all(c in fks for c in u):
                    continue
                rs = [fks[c] for c in u]
                if any(T[r["parent"]]["role"] != "entity" or r["parent"] == e for r in rs):
                    continue
                if any(tables[e][r["child_key"]].isna().any() for r in rs):
                    continue                     # optional links: keep the per-row model
                # anchor: the parent that is synthesised (activity is modelled on it)
                rs.sort(key=lambda r: (r["parent"] in self.copied, u.index(r["child_key"])))
                out[e] = {"cols": list(u), "anchor": rs[0], "partner": rs[1]}
                break
        return out

    def _copied_layout(self, tables):
        """Copied tables whose rows are kept as one segment each (exact row-level
        structure: which partner rows each code meets).  The block arrays of the
        model are dense in the number of segments: a fact with anchor, partner and
        extra dimension costs K_a K_p (1 + sum K_x) cells, a link from an entity
        with n rows costs n K_parent per Gibbs sweep.  Starting from one segment
        per copied row, the copied table whose clustering saves most is clustered
        (rows keep their identity and their own share of links, see _weights)
        until the total is within synthesis.segments.copied_budget_cells."""
        sc = self.cfg["segments"]
        budget = float(sc.get("copied_budget_cells", 2e7))
        minr = max(sc["min_rows_per_segment"], 1)
        n = {e: len(tables[e]) for e in self.entities}
        cap = {e: max(1, min(sc["tables"].get(e, sc["default_cap"]), n[e] // minr)) for e in self.entities}
        per_row = {e for e in self.copied if not isinstance(sc.get("copied_rows"), dict)
                   or sc["copied_rows"].get(e, True)}
        forced = {e for e, v in (sc.get("copied_rows") or {}).items() if v is True} if isinstance(
            sc.get("copied_rows"), dict) else set()

        def k(e, pr):
            if e.startswith(VDIM):
                return len(self.fact_info[e[len(VDIM):]].get("vdim_levels", [1]))
            if e in self.fact_info:          # a parent fact (nested): about one segment per block
                fi = self.fact_info[e]
                ds = [r["parent"] for r in fi["fks"][:2] if not r.get("virtual")]
                return int(np.prod([k(d, pr) for d in ds])) if ds else 1
            return n[e] if e in pr else cap[e]

        def cost(pr):
            tot = 0.0
            for _f, fi in self.fact_info.items():
                dims = ([fi["anchor"]] if fi["nested"] else []) + \
                       [r["parent"] if not r.get("virtual") else r["parent"] for r in fi["fks"]]
                ks = [k(d, pr) for d in dims]
                if len(ks) >= 2:
                    tot += ks[0] * ks[1] * (1 + sum(ks[2:]))
                elif ks:
                    tot += ks[0]
            for r in self.spec["relationships"]:
                if r["kind"] == "many_to_one" and not r.get("composite") and r["child"] in n and r["parent"] in n:
                    tot += n[r["child"]] * k(r["parent"], pr)
            return tot
        # statistical side: one segment per copied row gives every row its own rate
        # against every segment of the other side (a saturated block model).  With a
        # few links per row those rates are not estimable, and each extra segment of
        # the other side then pays an Occam penalty over all copied rows (a calendar
        # of 730 days makes splitting products cost about 730 rates each), so the
        # other side collapses.  Rows are kept as segments only when every relation
        # carries at least min_rows_per_segment links per row; otherwise the table
        # is clustered and each row keeps its own share of links (degree correction).
        for e in sorted(per_row - forced):
            for r in self.spec["relationships"]:
                if r["parent"] != e or r.get("kind") != "many_to_one" or r.get("composite") \
                        or r["child"] not in tables or r["child"] == e:
                    continue
                links = self._notnull(r["child"], r["child_key"], tables)
                if links < minr * max(n[e], 1):
                    per_row.discard(e)
                    self.log["decisions"].setdefault(e, {})["copied rows"] = (
                        f"clustered into latent segments: {r['name']} has {links / max(n[e], 1):.1f} links per row "
                        f"(< {minr}), too few to give each row its own rates; each row keeps its own share of links")
                    break
        cur = cost(per_row)
        start = cur
        while cur > budget and per_row - forced:
            gains = {e: cur - cost(per_row - {e}) for e in per_row - forced}
            e = max(gains, key=gains.get)
            if gains[e] <= 0:
                break
            per_row.discard(e)
            cur -= gains[e]
            self.log["decisions"].setdefault(e, {})["copied rows"] = (
                f"clustered into latent segments (model size {start:.3g} cells with one segment per copied row, "
                f"budget {budget:.3g}); each row keeps its own share of links")
        return per_row

    def _attribute_strata(self, tables):
        """Entities too small for two latent segments (fewer than 2 x
        min_rows_per_segment rows) cannot learn which of their rows receive
        children (e.g. only sales agents have customers).  For such an entity, an
        own categorical column whose levels explain the children per row is used
        as observed strata: segments are its levels and the column is set from the
        segment.  The column is chosen by the Bayes factor of a Poisson-gamma count
        model per level against one pooled model, minus log(#candidates) for the
        choice, and must be decisive (> log 100)."""
        sc = self.cfg["segments"]
        mode = sc.get("attribute_strata", "auto")
        T = self.spec["tables"]
        out = {}
        if mode in ("none", False, None):
            return out
        minr = max(sc["min_rows_per_segment"], 1)
        for e in self.entities:
            df = tables[e]
            n = len(df)
            if e in self.copied or e in self.grain or n < 4:
                continue
            want = mode.get(e) if isinstance(mode, dict) else None
            if isinstance(mode, dict) and not want:
                continue
            fkc = {r["child_key"] for r in self.fk[e]}
            cand = [c for c, cs in T[e]["columns"].items()
                    if cs["type"] in ("categorical", "boolean") and c not in fkc and c != T[e].get("primary_key")
                    and 2 <= df[c].nunique(dropna=False) <= min(sc.get("stratify_max_levels", 20), n - 1)]
            if n // minr >= 2 and not want:
                # an entity large enough for latent segments is stratified only by a
                # column with a structural zero for its children (only sales staff have
                # quotas): segments mixing eligible and excluded rows would move the
                # child count between datasets; the strata keep them apart (a
                # cannot-link constrained mixture, Shental et al. 2003)
                cand = [c for c in cand if self._inclusion_level(e, c, tables)]
            if want:
                cand = [c for c in cand if c == want]
            Y = self._child_counts(e, tables)
            if not cand or not Y:
                continue
            best, bc = -np.inf, None
            for c in cand:
                codes = pd.factorize(df[c].astype(object), use_na_sentinel=False)[0]
                lbf = sum(max(0.0, _pg_logbf(y, codes)) for y in Y) - np.log(len(cand) * len(Y))
                if lbf > best:
                    best, bc = lbf, c
            if bc is not None and (want or best > np.log(100)):
                out[e] = bc
                self.log["decisions"].setdefault(e, {})["attribute_strata"] = {"column": bc, "log_bf": round(float(best), 2)}
        return out

    def _inclusion_level(self, e, c, tables, minexp=None):
        """A level of column c of entity e that a child relation never references
        although about D n_l / n references would be expected (at least
        structure.inclusion_min_expected), as in _fit_inclusion."""
        if minexp is None:
            minexp = float(self.cfg["structure"].get("inclusion_min_expected", 10.0))
        T = self.spec["tables"]
        df = tables[e]
        lev = _keystr(df[c]).fillna("\x00null").to_numpy(dtype=object)
        keys = _keystr(df[T[e]["primary_key"]])
        n = len(df)
        for r in self.spec["relationships"]:
            if r["parent"] != e or r["kind"] != "many_to_one" or r.get("composite") or r["child"] == e:
                continue
            ref = set(_keystr(self._ref_values(r["child"], r["child_key"], tables)).dropna())
            isref = keys.isin(ref).to_numpy()
            D = int(isref.sum())
            if D == 0 or D == n:
                continue
            for l_ in pd.unique(lev):
                at = lev == l_
                if at.sum() < n and not isref[at].any() and D * at.sum() / n >= minexp:
                    return True
        return False

    def _child_counts(self, e, tables):
        """Children per row of entity e, one array per relation that points to it."""
        T = self.spec["tables"]
        df = tables[e]
        rowpos = self.key_index[e].get_indexer(df[T[e]["primary_key"]])
        Y = []
        for r in self.spec["relationships"]:
            if r["parent"] != e or r["kind"] != "many_to_one" or r.get("composite"):
                continue
            y = self._key_counts(r["child"], r["child_key"], e, tables)
            Y.append(np.where(rowpos >= 0, y[np.maximum(rowpos, 0)], 0.0))
        return Y

    def _partner_evidence(self, e, tables, codes):
        """log Bayes factor that the levels `codes` of e's rows explain WHICH
        partners they meet in pair facts (tracks of a genre sit in that genre's
        playlists): a symmetric Dirichlet-multinomial over partner rows per level
        against one pooled Dirichlet-multinomial."""
        T = self.spec["tables"]
        rowpos = self.key_index[e].get_indexer(tables[e][T[e]["primary_key"]])
        lev = np.full(len(self.key_index[e]), -1)
        lev[rowpos[rowpos >= 0]] = codes[rowpos >= 0]
        tot = 0.0
        for f in self.facts:
            fks = self.fk[f]
            mine = [r for r in fks if r["parent"] == e]
            if not mine or len(fks) < 2:
                continue
            df = tables[f]
            pos = self.key_index[e].get_indexer(df[mine[0]["child_key"]]) if f not in self.fstats else None
            for r in fks:
                if r is mine[0] or r["parent"] == e:
                    continue
                if f in self.fstats:
                    # rows per (key into e, partner key): exact statistics
                    ca, cb = mine[0]["child_key"], r["child_key"]
                    gc = self.fstats[f].group_counts([ca, cb])
                    pos = self._stat_index(f, ca, e, gc[ca])
                    q = pd.factorize(_keystr(self._canon_vals(f, cb, gc[cb])), use_na_sentinel=True)[0]
                    wt = gc["n"].to_numpy(dtype=float)
                else:
                    q = pd.factorize(_keystr(df[r["child_key"]]), use_na_sentinel=True)[0]
                    wt = None
                ok = (pos >= 0) & (q >= 0)
                if not ok.any():
                    continue
                L, P = lev[pos[ok]], q[ok]
                W = None if wt is None else wt[ok]
                nP = int(P.max()) + 1
                alpha = 1.0 / nP

                def dm(cnt):
                    return float(gammaln(nP * alpha) - gammaln(cnt.sum() + nP * alpha)
                                 + (gammaln(cnt + alpha) - gammaln(alpha)).sum())
                # a relation the levels do not explain contributes nothing (its rates
                # may be equal across strata), not negative evidence
                tot += max(0.0, sum(dm(np.bincount(P[L == k], weights=None if W is None else W[L == k],
                                                   minlength=nP).astype(float)) for k in np.unique(L))
                           - dm(np.bincount(P, weights=W, minlength=nP).astype(float)))
        return tot

    def _strata(self, e, tables):
        """Observed strata of entity e: the rows of its small copied parents, or
        the levels of an own column for tiny entities (_attribute_strata)."""
        sc = self.cfg["segments"]
        if e in self.attr_strata:
            col = self.attr_strata[e]
            codes, labels = pd.factorize(tables[e][col].astype(object), use_na_sentinel=False)
            return codes.astype(np.int64), [(l,) for l in labels], [], col
        mode = sc.get("stratify", "auto")
        if mode in ("none", False, None):
            return None
        rels = [r for r in self.fk[e] if r["parent"] in self.copied]
        # implied dimensions (codes of the entity's own columns) are dropped before real
        # parents when the strata must shrink: their anchor keeps them modelled attributes,
        # while a real parent's link has no other carrier (AdventureWorks customers kept
        # education x occupation and lost their geography); alone (product subcategories)
        # they stratify like any small copied parent
        implied = {r["child_key"] for r in rels if self.spec["tables"][r["parent"]].get("virtual_dimension_of")}
        if isinstance(mode, dict):
            want = mode.get(e)
            if not want:
                return None
            rels = [r for r in rels if r["child_key"] in want]
        else:
            # small code lists always; larger ones when their levels explain how many
            # children the entity's rows have (decisive Bayes factor, as for
            # attribute strata), e.g. tracks of a genre appear in that genre's playlists
            Y = None
            keep = []
            for r in rels:
                if len(tables[r["parent"]]) <= sc.get("stratify_max_levels", 20):
                    keep.append(r)
                    continue
                if Y is None:
                    Y = self._child_counts(e, tables)
                codes = pd.factorize(_keystr(tables[e][r["child_key"]]), use_na_sentinel=False)[0]
                lbf = sum(max(0.0, _pg_logbf(y, codes)) for y in Y) + self._partner_evidence(e, tables, codes) \
                    - np.log(max(len(Y) + len(self.facts), 1))
                if lbf > np.log(100):
                    keep.append(r)
                    self.log["decisions"].setdefault(e, {})[f"stratify {r['child_key']}"] = {"log_bf": round(float(lbf), 1)}
            rels = keep
        if not rels:
            return None
        n = len(tables[e])
        minr = max(sc["min_rows_per_segment"], 1)
        # drop stratifiers (largest first) until every stratum can hold a segment and
        # a Gibbs sweep stays within budget: each row is scored against every segment,
        # so a sweep costs rows x segments (555 geography strata of 105,000 customers
        # would make 1,757 segments and 1.8e8 cells per sweep)
        cap = sc["tables"].get(e, sc["default_cap"])
        max_cells = float(sc.get("stratify_max_cells", 5e7))
        rels = sorted(rels, key=lambda r: (r["child_key"] in implied, len(tables[r["parent"]])))
        while rels:
            key = _tuple_frame([_keystr(tables[e][r["child_key"]]) for r in rels])
            codes, labels = pd.factorize(pd.Series(key), use_na_sentinel=False)
            K = int(np.maximum(1, np.minimum(cap, np.bincount(codes) // minr)).sum())
            if len(labels) <= max(1, n // minr) and n * K <= max_cells:
                return codes.astype(np.int64), list(labels), [r["child_key"] for r in rels]
            if n * K > max_cells:
                self.log["decisions"].setdefault(e, {})[f"stratify {rels[-1]['child_key']}"] = (
                    f"not stratified: {len(labels)} strata would need {K} segments, {n * K:.2g} cells per sweep "
                    f"(budget segments.stratify_max_cells = {max_cells:.2g})")
            rels = rels[:-1]
        return None

    def _fact_layout(self, f, tables):
        T = self.spec["tables"][f]
        df = tables[f]
        fk = self.fk[f]
        nest = self.nest[f]
        info = {"nested": nest is not None, "nest": nest, "vdim": None}
        if nest is not None:
            parent = nest["parent"]
            others = list(fk)
            partner = T["fact"].get("partner")
            if partner:
                others.sort(key=lambda r: r["parent"] != partner)
            info.update({"anchor": parent, "fks": others})
            return info
        anchor = T["fact"].get("anchor") or fk[0]["parent"]
        ak, pk_ = T["fact"].get("anchor_key"), T["fact"].get("partner_key")
        # a role named by its key column (anchor: ShipDateKey) wins over the table name
        fk_sorted = sorted(fk, key=lambda r: (r["parent"] != anchor, ak is not None and r["child_key"] != ak,
                                              fk.index(r)))
        if T["fact"].get("partner"):
            fk_sorted = [fk_sorted[0]] + sorted(fk_sorted[1:], key=lambda r: (
                r["parent"] != T["fact"]["partner"], pk_ is not None and r["child_key"] != pk_))
        info.update({"anchor": fk_sorted[0]["parent"], "fks": fk_sorted})
        # virtual dimension: a categorical column that completes the anchor key to a unique key
        if len(fk_sorted) == 1 or not T["fact"].get("partner"):
            akey = fk_sorted[0]["child_key"]
            fkcols = {r["child_key"] for r in fk_sorted}
            for u in T.get("unique", []):
                rest = [c for c in u if c != akey]
                if akey in u and len(rest) == 1 and rest[0] not in fkcols and \
                        T["columns"][rest[0]]["type"] in ("categorical", "integer", "boolean"):
                    lv = pd.Series(df[rest[0]]).dropna().unique().tolist() if f not in self.fstats else \
                        sorted(self._canon_vals(f, rest[0], self.fstats[f].distinct(rest[0])).dropna().unique().tolist(),
                               key=lambda x: (str(type(x)), x))
                    if 1 < len(lv) <= 1000:
                        info["vdim"] = rest[0]
                        info["vdim_levels"] = lv
                        info["fks"] = fk_sorted[:1] + [{"child_key": rest[0], "parent": f"{VDIM}{f}",
                                                         "virtual": True, "name": f"{f}.{rest[0]}"}] + fk_sorted[1:]
                        break
        return info

    def _aux_columns(self, f, tables):
        """Continuous attributes of the fact's parents (at most 2 per parent),
        joined onto every fact row: fact.condition_on_parents (default true)."""
        T = self.spec["tables"]
        opt = T[f]["fact"].get("condition_on_parents", True)
        if opt is False:
            return []
        out = []
        self.aux_map = getattr(self, "aux_map", {})
        self.aux_map[f] = []
        for r in self.fk[f]:
            p = r["parent"]
            pk = T[p]["primary_key"]
            excl = self._excluded(p)
            cands = [c for c, cs in T[p]["columns"].items()
                     if cs["type"] in ("float", "integer", "date", "datetime") and c not in excl]
            if isinstance(opt, list):
                cands = [c for c in cands if f"{p}.{c}" in opt]
            else:
                cands = [c for c in cands if tables[p][c].nunique() > 30][:2]
            par = tables[p].drop_duplicates(pk).set_index(pk)
            for c in cands:
                name = f"__aux__{r['child_key']}.{c}"
                ser = par[c].reindex(tables[f][r["child_key"]].to_numpy())
                out.append((name, T[p]["columns"][c], ser))
                self.aux_map[f].append((name, r["child_key"], p, c))
        return out

    def _lookup(self, parent, values):
        idx = self.key_index[parent].get_indexer(pd.Series(values).to_numpy())
        return np.asarray(idx, dtype=np.int64)

    # ------------------------------------------------------------------ pushdown
    # Facts given as (uniform sample, exact statistics): every quantity of the
    # relational model is read from the statistics, the row-level models from the
    # sample (relsynth.pushdown; docs/theory.md, Propositions P1-P4).
    def _canon_vals(self, t, col, values):
        """Values of t.col returned by a statistic, in the representation of the
        loaded frames (coerced to the column's declared type)."""
        user = self.user_spec["tables"].get(t, {}).get("columns", {})
        cs = user.get(col) if user.get(col, {}).get("type") not in (None, "lookup") else None
        cs = cs or self.spec["tables"][t]["columns"].get(col, {"type": "id"})
        df = coerce(pd.DataFrame({col: pd.Series(values, dtype=object).reset_index(drop=True)}), {"columns": {col: cs}})
        return df[col]

    def _stat_index(self, f, col, parent, values):
        """Row index in `parent` of each value of f.col returned by a statistic (-1:
        null or not a parent row)."""
        v = self._canon_vals(f, col, values)
        out = np.full(len(v), -1, dtype=np.int64)
        ok = v.notna().to_numpy()
        if ok.any():
            out[ok] = self.key_index[parent].get_indexer(v[ok].to_numpy())
        return out

    def _notnull(self, t, col, tables):
        """Rows of t with col set (every row, also for a pushed-down fact)."""
        if t in getattr(self, "fstats", {}):
            return int(self.fstats[t].notnull(col))
        return int(tables[t][col].notna().sum())

    def _ref_values(self, t, col, tables):
        """The values of t.col over every row (distinct values for a pushed-down fact)."""
        if t in getattr(self, "fstats", {}):
            return self._canon_vals(t, col, self.fstats[t].distinct(col))
        return tables[t][col]

    def _key_counts(self, t, col, parent, tables):
        """Rows of t per row of parent's key index, by t.col."""
        n = len(self.key_index[parent])
        if t in getattr(self, "fstats", {}):
            gc = self.fstats[t].counts(col)
            pos = self._stat_index(t, col, parent, gc[col])
            ok = pos >= 0
            return np.bincount(pos[ok], weights=gc["n"].to_numpy(dtype=float)[ok], minlength=n)
        pos = self.key_index[parent].get_indexer(tables[t][col].dropna())
        return np.bincount(pos[pos >= 0], minlength=n).astype(float)

    def _require(self, f, r):
        """Row filter of a statistic: f's key r points at a row of the internal
        parent (stored parents by a semi-join, minus the unknown members the model
        drops; implied dimensions by their key list)."""
        from .pushdown.stats import Require
        p = r["parent"]
        vals = self.key_index[p].to_numpy() if p in self.key_index else None
        user = self.user_spec["tables"]
        if p in user and user[p].get("primary_key") == self.spec["tables"][p].get("primary_key"):
            pk = self.spec["tables"][p]["primary_key"]
            unk = getattr(self.plan, "unknown_rows", {}).get(p)
            excl = [] if unk is None or pk not in unk else unk[pk].dropna().tolist()
            return Require(values=vals, table=p, column=pk, exclude=excl)
        return Require(values=vals)

    def _fk_stat_index(self, f, r, vals):
        """_dim_index of key values returned by a statistic of fact f."""
        if r.get("virtual"):
            lv = {v: i for i, v in enumerate(self.fact_info[f]["vdim_levels"])}
            return np.array([lv.get(v, -1) for v in self._canon_vals(f, r["child_key"], vals).tolist()],
                            dtype=np.int64)
        return self._stat_index(f, r["child_key"], r["parent"], vals)

    def _measures(self, f):
        """The (up to three) continuous columns whose normal scores segment discovery uses."""
        tm = self.tm[f]
        return [cd.name for cd in tm.codecs if cd.kind == "continuous" and cd.name not in tm.aux][:3]

    def _nested_groups(self, f):
        """GROUP BY statistics of a pushed-down nested fact over its composite key and
        dimension keys: (groups, parent-fact row of each group, dimension row index
        of each group per key in fact_info['fks'])."""
        info = self.fact_info[f]
        if "_ngroups" in info:
            return info["_ngroups"]
        nest, fks = info["nest"], info["fks"]
        tables = self._tables_for_fit
        plain = [c for c in nest["child_key"] if "." not in c]
        cols = list(dict.fromkeys(plain + [r["child_key"] for r in fks]))
        gc = self.fstats[f].group_counts(cols, self._measures(f))
        gk = pd.DataFrame({c: self._canon_vals(f, c, gc[c]) for c in cols})
        keys = _tuple_frame(self._path_values(f, gk, nest["child_key"], tables))
        pidx = np.asarray(self.fact_key_index[nest["parent"]].get_indexer(keys), dtype=np.int64)
        dims = [self._fk_stat_index(f, r, gc[r["child_key"]]) for r in fks]
        info["_ngroups"] = (gc, pidx, dims)
        return info["_ngroups"]

    def _nested_relations_pushed(self, f, df):
        """_fact_relations of a pushed-down nested fact: the projection onto the grand
        anchor, from the composite-key statistics."""
        info = self.fact_info[f]
        tables = self._tables_for_fit
        fks = info["fks"]
        gc, pidx, dims = self._nested_groups(f)
        up = self._grand_anchor_rows(info["anchor"], tables)
        g = np.where(pidx >= 0, up[np.maximum(pidx, 0)], -1)
        ga = self._grand_anchor(f)
        info["grand_anchor"] = ga
        idx = [g] + dims
        e_list = [self.E[ga]] + [self._dim_entity(f, r) for r in fks]
        sidx = [self._grand_anchor_rows(f, tables)] + [self._dim_index(f, r, df) for r in fks]
        info["ok"] = np.all(np.stack(sidx[:2]) >= 0, axis=0)
        ok = np.all(np.stack(idx[:2]) >= 0, axis=0)
        n = gc["n"].to_numpy(dtype=np.int64)
        dropped = int(n[~ok].sum())
        if dropped:
            self.log["warnings"].append(f"{f}: {dropped} rows with null/orphan dimension keys left out of the model")
        info.update({"dropped": dropped, "pushed": True})
        meas = self._measures(f)
        e1 = e_list[0]
        u, w = idx[0][ok], n[ok]
        Sc = np.column_stack([gc[f"s{j}"].to_numpy(dtype=float)[ok] for j in range(len(meas))]) if meas \
            else np.zeros((len(u), 0))
        Qc = np.column_stack([gc[f"q{j}"].to_numpy(dtype=float)[ok] for j in range(len(meas))]) if meas \
            else np.zeros((len(u), 0))
        if len(e_list) == 1:
            N = np.bincount(u, weights=w, minlength=e1.n).round().astype(np.int64)
            S = np.stack([np.bincount(u, weights=Sc[:, j], minlength=e1.n) for j in range(Sc.shape[1])], 1) \
                if Sc.shape[1] else np.zeros((e1.n, 0))
            Q2 = np.stack([np.bincount(u, weights=Qc[:, j], minlength=e1.n) for j in range(Qc.shape[1])], 1) \
                if Qc.shape[1] else np.zeros((e1.n, 0))
            rel = SingleRel(f, e1, N, S, Q2, table=f)
            info["rel"] = rel
            info["model"] = "single"
            return [rel]
        e2 = e_list[1]
        v = idx[1][ok]
        uk, inv = np.unique(u * e2.n + v, return_inverse=True)
        cnt = np.bincount(inv, weights=w).round().astype(np.int64)
        S = np.stack([np.bincount(inv, weights=Sc[:, j]) for j in range(Sc.shape[1])], 1) if Sc.shape[1] else None
        Q2 = np.stack([np.bincount(inv, weights=Qc[:, j]) for j in range(Qc.shape[1])], 1) if Qc.shape[1] else None
        model, mult = self._pair_model(f, info, uk // e2.n, uk % e2.n, cnt, e1.n, e2.n)
        if model == "occupancy":
            bp = self.cfg["structure"]["beta_prior"]
            rel = OccRel(f, e1, e2, uk // e2.n, uk % e2.n, cnt, S, Q2, table=f,
                         mult_by="block" if mult == "auto" else mult, a=bp, b=bp)
        else:
            rel = PairRel(f, "fact", e1, e2, uk // e2.n, uk % e2.n, cnt, S, Q2, table=f)
        info.update({"rel": rel, "model": model, "mult_setting": mult})
        rels = [rel]
        info["extra"] = []
        for j in range(2, len(e_list)):
            ex = e_list[j]
            r = fks[j - 1]
            x = idx[j][ok]
            m = x >= 0
            uk2, inv2 = np.unique(u[m] * ex.n + x[m], return_inverse=True)
            cnt2 = np.bincount(inv2, weights=w[m]).round().astype(np.int64)
            rx = PairRel(f"{f}.{r['child_key']}", "extra", e1, ex, uk2 // ex.n, uk2 % ex.n, cnt2, table=f)
            rx.row_idx = sidx[j][info["ok"]]
            rx.fan_exact = np.bincount(x[m], weights=w[m], minlength=ex.n)
            info["extra"].append(rx)
            rels.append(rx)
        return rels

    def _fact_relations_pushed(self, f, df):
        """_fact_relations of a pushed-down fact: the relations from GROUP BY
        statistics over every row; `ok` and the row indexes of the sample for the
        row-level models."""
        info = self.fact_info[f]
        if info["nested"]:
            return self._nested_relations_pushed(f, df)
        st = self.fstats[f]
        fks = info["fks"]
        e_list = [self._dim_entity(f, r) for r in fks]
        sidx = [self._dim_index(f, r, df) for r in fks]
        info["ok"] = np.all(np.stack(sidx[:2]) >= 0, axis=0)
        meas = self._measures(f)

        def index(r, vals):
            return self._fk_stat_index(f, r, vals)
        keys = [r["child_key"] for r in fks[:2]]
        gc = st.group_counts(keys, meas)
        idx = [index(r, gc[r["child_key"]]) for r in fks[:2]]
        ok = np.all(np.stack(idx) >= 0, axis=0)
        n = gc["n"].to_numpy(dtype=np.int64)
        dropped = int(n[~ok].sum())
        if dropped:
            self.log["warnings"].append(f"{f}: {dropped} rows with null/orphan dimension keys left out of the model")
        info.update({"dropped": dropped, "pushed": True})
        e1 = e_list[0]
        u, w = idx[0][ok], n[ok]
        Sc = np.column_stack([gc[f"s{j}"].to_numpy(dtype=float)[ok] for j in range(len(meas))]) if meas \
            else np.zeros((len(u), 0))
        Qc = np.column_stack([gc[f"q{j}"].to_numpy(dtype=float)[ok] for j in range(len(meas))]) if meas \
            else np.zeros((len(u), 0))
        if len(e_list) == 1:
            N = np.bincount(u, weights=w, minlength=e1.n).round().astype(np.int64)
            S = np.stack([np.bincount(u, weights=Sc[:, j], minlength=e1.n) for j in range(Sc.shape[1])], 1) \
                if Sc.shape[1] else np.zeros((e1.n, 0))
            Q2 = np.stack([np.bincount(u, weights=Qc[:, j], minlength=e1.n) for j in range(Qc.shape[1])], 1) \
                if Qc.shape[1] else np.zeros((e1.n, 0))
            rel = SingleRel(f, e1, N, S, Q2, table=f)
            info["rel"] = rel
            info["model"] = "single"
            return [rel]
        e2 = e_list[1]
        v = idx[1][ok]
        key = u * e2.n + v
        uk, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv, weights=w).round().astype(np.int64)
        S = np.stack([np.bincount(inv, weights=Sc[:, j]) for j in range(Sc.shape[1])], 1) if Sc.shape[1] else None
        Q2 = np.stack([np.bincount(inv, weights=Qc[:, j]) for j in range(Qc.shape[1])], 1) if Qc.shape[1] else None
        model, mult = self._pair_model(f, info, uk // e2.n, uk % e2.n, cnt, e1.n, e2.n)
        if model == "occupancy":
            bp = self.cfg["structure"]["beta_prior"]
            rel = OccRel(f, e1, e2, uk // e2.n, uk % e2.n, cnt, S, Q2, table=f,
                         mult_by="block" if mult == "auto" else mult, a=bp, b=bp)
        else:
            rel = PairRel(f, "fact", e1, e2, uk // e2.n, uk % e2.n, cnt, S, Q2, table=f)
        info.update({"rel": rel, "model": model, "mult_setting": mult})
        rels = [rel]
        info["extra"] = []
        # extra dimensions: rows with a valid anchor and partner (as in memory), the
        # partner checked in the store by a semi-join on its parent
        need = {} if fks[1].get("virtual") else {fks[1]["child_key"]: self._require(f, fks[1])}
        for j in range(2, len(e_list)):
            ex = e_list[j]
            r = fks[j]
            gx = st.group_counts([fks[0]["child_key"], r["child_key"]], require=need)
            a = index(fks[0], gx[fks[0]["child_key"]])
            x = index(r, gx[r["child_key"]])
            wx = gx["n"].to_numpy(dtype=float)
            okx = (a >= 0) & (x >= 0)
            kk = a[okx] * ex.n + x[okx]
            uk2, inv2 = np.unique(kk, return_inverse=True)
            cnt2 = np.bincount(inv2, weights=wx[okx]).round().astype(np.int64)
            rx = PairRel(f"{f}.{r['child_key']}", "extra", e1, ex, uk2 // ex.n, uk2 % ex.n, cnt2, table=f)
            rx.row_idx = sidx[j][info["ok"]]
            rx.fan_exact = np.bincount(x[okx], weights=wx[okx], minlength=ex.n)
            info["extra"].append(rx)
            rels.append(rx)
        return rels

    def _anchor_index(self, f, df, tables):
        """Row index of every fact row's anchor (entity row, or parent-fact row when nested)."""
        info = self.fact_info[f]
        if not info["nested"]:
            return self._lookup(info["anchor"], df[info["fks"][0]["child_key"]])
        nest = info["nest"]
        keys = _tuple_frame(self._path_values(f, df, nest["child_key"], tables))
        return np.asarray(self.fact_key_index[nest["parent"]].get_indexer(keys), dtype=np.int64)

    def _nest_groups(self, f, anchor_df, partner_df):
        """Eligibility codes of a nested fact's generation relation.  A component of
        the composite key that is a column of the partner ('CellKey.TemplateCode')
        must equal the parent row's key column: pairs with different values are
        structural zeros.  Returns (codes of anchor rows, codes of partner rows), or
        None when no component is a partner column."""
        info = self.fact_info[f]
        nest = info["nest"]
        if not info["fks"]:
            return None
        pfk = info["fks"][0]["child_key"]
        pairs = [(ck.split(".", 1)[1], pk) for ck, pk in zip(nest["child_key"], nest["parent_key"])
                 if ck.count(".") == 1 and ck.split(".", 1)[0] == pfk]
        if not pairs or any(c not in partner_df.columns or pk not in anchor_df.columns for c, pk in pairs):
            return None
        a = _tuple_frame([_keystr(anchor_df[pk]) for c, pk in pairs])
        b = _tuple_frame([_keystr(partner_df[c]) for c, pk in pairs])
        codes, _ = pd.factorize(pd.concat([pd.Series(a), pd.Series(b)], ignore_index=True), use_na_sentinel=False)
        return codes[:len(a)].astype(np.int64), codes[len(a):].astype(np.int64)

    def _path_values(self, t, df, cols, tables):
        """Columns or dotted lookup paths of table t, evaluated on df (source tables)."""
        out = []
        for c in cols:
            if "." not in c:
                out.append(_keystr(df[c]))
                continue
            hops, (pt, pc) = resolve_path(self.spec, t, c)
            cur = df[hops[0]["child_key"]]
            for i, h in enumerate(hops):
                par = tables[h["parent"]]
                pk = self.spec["tables"][h["parent"]]["primary_key"]
                ix = pd.Index(_keystr(par[pk])).get_indexer(_keystr(cur))
                nxt = hops[i + 1]["child_key"] if i + 1 < len(hops) else pc
                vals = par[nxt].to_numpy(dtype=object)
                cur = pd.Series([vals[j] if j >= 0 else None for j in ix])
            out.append(_keystr(cur))
        return out

    def _fact_relations(self, f, df):
        """Relations of fact f that take part in segment discovery (Gibbs)."""
        if f in self.fstats:
            return self._fact_relations_pushed(f, df)
        info = self.fact_info[f]
        tables = self._tables_for_fit
        if info["nested"]:
            # segment discovery uses the projection onto the grand anchor entity;
            # generation later uses the nested relation (see _post_fact)
            g = self._grand_anchor_rows(f, tables)
            ga = self._grand_anchor(f)
            idx = [g] + [self._dim_index(f, r, df) for r in info["fks"]]
            e_list = [self.E[ga]] + [self._dim_entity(f, r) for r in info["fks"]]
            info["grand_anchor"] = ga
        else:
            idx = [self._dim_index(f, r, df) for r in info["fks"]]
            e_list = [self._dim_entity(f, r) for r in info["fks"]]
        # the anchor and the partner define a row; further dimension keys may be
        # null (a null is a level of their distribution, see _extra_dist)
        ok = np.all(np.stack(idx[:2]) >= 0, axis=0)
        dropped = int((~ok).sum())
        if dropped:
            self.log["warnings"].append(f"{f}: {dropped} rows with null/orphan dimension keys left out of the model")
        tm = self.tm[f]
        meas = [i for i, cd in enumerate(tm.codecs) if cd.kind == "continuous" and cd.name not in tm.aux][:3]
        scores = np.column_stack([_scores(tm.X[i]) for i in meas]) if meas else np.zeros((len(df), 0))
        scores = scores[ok]
        info.update({"ok": ok, "dropped": dropped})
        e1 = e_list[0]
        u = idx[0][ok]
        if len(e_list) == 1:
            N = np.bincount(u, minlength=e1.n)
            S = np.stack([np.bincount(u, weights=scores[:, j], minlength=e1.n) for j in range(scores.shape[1])], 1) \
                if scores.shape[1] else np.zeros((e1.n, 0))
            Q2 = np.stack([np.bincount(u, weights=scores[:, j] ** 2, minlength=e1.n) for j in range(scores.shape[1])], 1) \
                if scores.shape[1] else np.zeros((e1.n, 0))
            rel = SingleRel(f, e1, N, S, Q2, table=f)
            info["rel"] = rel
            info["model"] = "single"
            return [rel]
        e2 = e_list[1]
        v = idx[1][ok]
        key = u * e2.n + v
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        S = np.stack([np.bincount(inv, weights=scores[:, j]) for j in range(scores.shape[1])], 1) \
            if scores.shape[1] else None
        Q2 = np.stack([np.bincount(inv, weights=scores[:, j] ** 2) for j in range(scores.shape[1])], 1) \
            if scores.shape[1] else None
        model, mult = self._pair_model(f, info, uk // e2.n, uk % e2.n, cnt, e1.n, e2.n)
        if model == "occupancy":
            bp = self.cfg["structure"]["beta_prior"]
            rel = OccRel(f, e1, e2, uk // e2.n, uk % e2.n, cnt, S, Q2, table=f,
                         mult_by="block" if mult == "auto" else mult, a=bp, b=bp)
        else:
            rel = PairRel(f, "fact", e1, e2, uk // e2.n, uk % e2.n, cnt, S, Q2, table=f)
        info.update({"rel": rel, "model": model, "mult_setting": mult})
        rels = [rel]
        info["extra"] = []
        for j in range(2, len(e_list)):
            ex = e_list[j]
            r = (info["fks"])[j - 1] if info["nested"] else info["fks"][j]
            x = idx[j][ok]
            kk = u[x >= 0] * ex.n + x[x >= 0]
            uk2, cnt2 = np.unique(kk, return_counts=True)
            rx = PairRel(f"{f}.{r['child_key']}", "extra", e1, ex, uk2 // ex.n, uk2 % ex.n, cnt2, table=f)
            rx.row_idx = x
            info["extra"].append(rx)
            rels.append(rx)
        return rels

    def _dim_entity(self, f, r):
        return self.E[r["parent"]]

    def _dim_index(self, f, r, df):
        if r.get("virtual"):
            lv = self.fact_info[f]["vdim_levels"]
            m = {v: i for i, v in enumerate(lv)}
            return np.array([m.get(v, -1) for v in df[r["child_key"]].to_numpy(dtype=object)], dtype=np.int64)
        return self._lookup(r["parent"], df[r["child_key"]])

    def _grand_anchor(self, f):
        info = self.fact_info[f]
        return self._grand_anchor(info["anchor"]) if info["nested"] else info["anchor"]

    def _grand_anchor_rows(self, f, tables):
        """Row index (in the grand-anchor entity) of every row of fact f."""
        info = self.fact_info[f]
        df = tables[f]
        if not info["nested"]:
            return self._lookup(info["anchor"], df[info["fks"][0]["child_key"]])
        pidx = self._anchor_index(f, df, tables)
        up = self._grand_anchor_rows(info["anchor"], tables)
        return np.where(pidx >= 0, up[np.maximum(pidx, 0)], -1)

    def _pair_model(self, f, info, u, v, c, n1, n2):
        """Poisson block model or occupancy (grain) model for a pair fact."""
        T = self.spec["tables"][f]
        setting = T["fact"].get("pairs", self.cfg["facts"]["pairs"])
        mult = T["fact"].get("multiplicity", self.cfg["facts"]["multiplicity"])
        keycols = set()
        if info["nested"]:
            keycols |= set(info["nest"]["child_key"])
            keycols |= {r["child_key"] for r in info["fks"][:1]}
        else:
            keycols |= {r["child_key"] for r in info["fks"][:2]}
        declared = any(set(uq) <= keycols for uq in T.get("unique", []))
        obs, exp_, z = poisson_dup_test(u, v, c, n1, n2, self.rng)
        dec = {"repeated_pairs_observed": obs, "repeated_pairs_expected_poisson": round(exp_, 1),
               "z": round(z, 2), "grain_declared": declared}
        if setting == "poisson":
            model = "poisson"
        elif setting == "occupancy" or declared or info.get("vdim") or info["nested"]:
            model = "occupancy"
        else:
            model = "occupancy" if (z < -4 and exp_ >= 10) else "poisson"
        if declared or (c.max(initial=1) <= 1 and mult == "auto" and model == "occupancy"):
            mult = "none"
        dec["model"], dec["multiplicity"] = model, mult
        self.log["decisions"][f] = dec
        return model, mult

    # ------------------------------------------------------------------ post-fit
    def _frailty_on(self, mode, D):
        thr = self.cfg["facts"]["frailty_threshold"]
        return mode is True or (mode == "auto" and np.nanmax(D) > thr)

    def _post_edges(self):
        rng = self.rng
        for name, info in self.edge_info.items():
            rel = info["rel"]
            if rel.kind == "tree":
                e = rel.e1
                info["psi"] = np.exp(rel.psi)
                info["lam"] = rel.lam
                info["pi_root"] = rel.pi_root
                info["roots"] = int((rel.par < 0).sum())
                info["max_depth"] = int(rel.depth.max()) if len(rel.depth) else 0
                info["lam_d"], info["psi_d"] = self._tree_levels(rel, info["psi"], rel.lam, info["max_depth"])
                alpha, D = dispersion_alpha(rel.nchild, e.z, e.K, rel.lam, self.cfg["facts"]["frailty_threshold"])
                info["frailty"] = alpha if self._frailty_on(self.cfg["trees"]["frailty"], D) else np.full(e.K, np.inf)
                info["dispersion"] = D
                continue
            c, p = rel.e1, rel.e2
            phi = rel.eta * p.counts()[None, :]
            tot = phi.sum(1, keepdims=True)
            info["phi"] = np.where(tot > 0, phi / np.where(tot > 0, tot, 1), 1.0 / phi.shape[1])
            nm = rel.null_mask
            nk = c.counts()
            nullk = np.bincount(c.z[nm], minlength=c.K)
            info["p_null"] = rng.beta(nullk + 0.5 * (nm.mean() if nm.any() else 0) + 1e-3,
                                      nk - nullk + 1) if nm.any() else np.zeros(c.K)
            fan = np.bincount(rel.v, minlength=p.n).astype(float)
            alpha, D = dispersion_alpha(fan, p.z, p.K, None, self.cfg["facts"]["frailty_threshold"])
            info["parent_frailty"] = alpha if self._frailty_on(self.cfg["facts"]["frailty"], D) else np.full(p.K, np.inf)
            info["dispersion"] = D
            info["fixed_parent"] = bool(p.fixed or getattr(p, "copied", False))
            info["obs"] = fan
            info["count_dp"] = _dp_floor(self._edge_count_model(name, fan, p, info["spec"]["child"]),
                                         not info["fixed_parent"] and _mandatory(fan, p.z, p.K))
            info["min_one"] = not info["count_dp"] and not info["fixed_parent"] and _mandatory(fan, p.z, p.K)
            if info["min_one"]:
                self.log["decisions"].setdefault(info["spec"]["child"], {})[name + " participation"] = \
                    "mandatory: every parent row has at least one child"
            if info["count_dp"]:
                info["parent_frailty"] = np.full(p.K, np.inf)
                self.log["decisions"].setdefault(info["spec"]["child"], {})[name] = \
                    "children per parent: DP counts given the total"

    def _post_fact(self, f, df):
        info, rng, cfg = self.fact_info[f], self.rng, self.cfg
        T = self.spec["tables"][f]
        mode = cfg["facts"]["frailty"]
        thr = cfg["facts"]["frailty_threshold"]
        tm = self.tm[f]
        tables = self._tables_for_fit
        bp = cfg["structure"]["beta_prior"]
        if info["nested"]:
            # generation relation: parent-fact rows (fixed segments) x partner
            pe = self.fact_entity[info["anchor"]]
            aidx = self._anchor_index(f, df, tables)
            sdims = [self._dim_index(f, r, df) for r in info["fks"]]
            info["ok"] = (aidx >= 0) & ((sdims[0] >= 0) if sdims else True)
            pushed = f in self.fstats
            if pushed:
                # pairs over every row from the composite-key statistics; the rows
                # of the sample feed the row-level models
                gc, gidx, dims = self._nested_groups(f)
                gw = gc["n"].to_numpy(dtype=float)
            else:
                gidx, dims, gw = aidx, sdims, np.ones(len(aidx))
            okg = (gidx >= 0) & ((dims[0] >= 0) if dims else True)
            u, w = gidx[okg], gw[okg]
            if dims:
                e2 = self._dim_entity(f, info["fks"][0])
                v = dims[0][okg]
                key = u * e2.n + v
                uk, inv = np.unique(key, return_inverse=True)
                cnt = np.bincount(inv, weights=w).round().astype(np.int64)
                elig = self._nest_groups(f, tables[info["anchor"]], tables[info["fks"][0]["parent"]])
                pe = self._refine_nested_parent(info["anchor"], f, pe, e2, uk // e2.n, uk % e2.n, cnt, bp, elig)
                rel = OccRel(f, pe, e2, uk // e2.n, uk % e2.n, cnt, table=f, mult_by="block", a=bp, b=bp)
                if elig is not None:
                    rel.set_eligibility(*elig)
                    info["eligibility"] = True
                rel.prior_mean = getattr(pe, "prior_mean", None)
                info["rel"] = rel
                info["model"] = "occupancy"
                info["mult_setting"] = info.get("mult_setting", T["fact"].get("multiplicity", cfg["facts"]["multiplicity"]))
                if cnt.max(initial=1) <= 1 and info["mult_setting"] == "auto":
                    info["mult_setting"] = "none"
                info["extra"] = []
                for j, r in enumerate(info["fks"][1:], start=1):
                    ex = self._dim_entity(f, r)
                    x = dims[j][okg]
                    m = x >= 0
                    uk2, inv2 = np.unique(u[m] * ex.n + x[m], return_inverse=True)
                    cnt2 = np.bincount(inv2, weights=w[m]).round().astype(np.int64)
                    rx = PairRel(f"{f}.{r['child_key']}", "extra", pe, ex, uk2 // ex.n, uk2 % ex.n, cnt2, table=f)
                    rx.row_idx = sdims[j][info["ok"]]
                    if pushed:
                        rx.fan_exact = np.bincount(x[m], weights=w[m], minlength=ex.n)
                    info["extra"].append(rx)
            else:
                N = np.bincount(u, weights=w, minlength=pe.n).round().astype(np.int64)
                rel = SingleRel(f, pe, N, np.zeros((pe.n, 0)), np.zeros((pe.n, 0)), table=f)
                rel.draw_params(rng)
                info["rel"] = rel
                info["model"] = "single"
        ok = info["ok"]
        rel = info["rel"]
        e1 = rel.e1
        a_idx = self._anchor_index(f, df, tables)
        z1 = e1.z[a_idx[ok]]
        parent = None
        if rel.kind == "single":
            alpha, D = dispersion_alpha(rel.N, e1.z, e1.K, rel.lam, thr)
            info["frailty1"] = alpha if self._frailty_on(mode, D) else np.full(e1.K, np.inf)
            info["dispersion1"] = D
            info["obs1"] = rel.N
            cp1 = e1.fixed or getattr(e1, "copied", False)
            info["count_dp"] = None if cp1 else _dp_floor(self._count_model(f, rel.N, e1), _mandatory(rel.N, e1.z, e1.K))
            info["min_one"] = not info["count_dp"] and not cp1 and _mandatory(rel.N, e1.z, e1.K)
            groups = np.full(len(df), -1)
            groups[ok] = z1
            blk_rows = z1
        else:
            e2 = rel.e2
            d2 = self._dim_index(f, info["fks"][0 if info["nested"] else 1], df)
            z2 = e2.z[d2[ok]]
            tot1 = np.bincount(rel.u, weights=rel.c, minlength=e1.n)
            tot2 = np.bincount(rel.v, weights=rel.c, minlength=e2.n)
            info["obs1"], info["obs2"] = tot1, tot2
            a1, D1 = dispersion_alpha(tot1, e1.z, e1.K, None, thr)
            a2, D2 = dispersion_alpha(tot2, e2.z, e2.K, None, thr)
            info["dispersion1"], info["dispersion2"] = D1, D2
            if rel.kind == "occ":
                self._finalise_occ(f, rel, info)
                info["frailty1"] = np.full(e1.K, np.inf)
                info["frailty2"] = np.full(e2.K, np.inf)
                # mandatory participation: every source anchor row has rows (an order
                # header always has lines), while the occupancy model would leave
                # about sum_i P(no pair | z_i) of them empty
                if len(tot1) and (tot1 > 0).all():
                    lq = np.log1p(-np.clip(rel.p, 0, 1 - 1e-12)) @ e2.counts().astype(float)
                    exp_empty = float(np.exp(lq)[e1.z].sum())
                    info["occ_min_one"] = exp_empty >= 3
                    if info["occ_min_one"]:
                        cs = self._truncate_occ(rel)
                        self.log["decisions"].setdefault(f, {})["participation"] = (
                            f"mandatory: every {e1.name} row has rows (the model alone would leave "
                            f"{exp_empty:.0f} empty); pairs are drawn conditioned on at least one, with "
                            f"rates rescaled for the truncation (factor {cs.min():.3f}-{cs.max():.3f})")
            else:
                info["frailty1"] = a1 if self._frailty_on(mode, D1) else np.full(e1.K, np.inf)
                info["frailty2"] = a2 if self._frailty_on(mode, D2) else np.full(e2.K, np.inf)
                cp1 = e1.fixed or getattr(e1, "copied", False)
                info["count_dp"] = None if cp1 else _dp_floor(self._count_model(f, tot1.astype(np.int64), e1),
                                                              _mandatory(tot1, e1.z, e1.K))
                info["min_one"] = not info["count_dp"] and not cp1 and _mandatory(tot1, e1.z, e1.K)
                if info["count_dp"] or info["min_one"]:
                    info["frailty1"] = np.full(e1.K, np.inf)
            self._fit_link_degree(f, info, rel)
            groups = np.full(len(df), -1)
            blk = z1 * e2.K + z2
            blk_rows = blk
            # blocks with few rows share the marginals of the more informative dimension
            fb = self._fallback_dim(f, z1, z2, ok)
            info["fallback"] = fb
            bc = np.bincount(blk, minlength=e1.K * e2.K)
            small = bc[blk] < cfg["facts"]["min_group_rows"]
            OFF1, OFF2 = 10**6, 2 * 10**6
            coarse = (OFF1 + z1) if fb == "anchor" else (OFF2 + z2)
            groups[ok] = np.where(small, coarse, blk)
            parent = {g: (g // e2.K + OFF1 if fb == "anchor" else g % e2.K + OFF2) if g < OFF1 else -1
                      for g in np.unique(groups[ok])}
            info["coarse_of_block"] = (OFF1, OFF2, e2.K)
            # extra dimensions: segment distribution per block, backed off to the
            # more informative block dimension, with structural zeros
            info["extra_p"] = []
            for rx in info["extra"]:
                ex = rx.e2
                xi = rx.row_idx
                s = np.where(xi >= 0, ex.z[np.maximum(xi, 0)], ex.K)       # null key -> level K
                d = self._extra_dist(z1, z2, s, e1.K, e2.K, ex)
                self._register(lambda rng, d=d: d.update(d["redraw"](rng)), now=False)
                fan = rx.fan_exact if getattr(rx, "fan_exact", None) is not None else \
                    np.bincount(xi[xi >= 0], minlength=ex.n).astype(float)
                ax, Dx = dispersion_alpha(fan, ex.z, ex.K, None, thr)
                d["frailty"] = ax if self._frailty_on(mode, Dx) else np.full(ex.K, np.inf)
                d["obs"] = fan
                info["extra_p"].append(d)
        if parent is None:
            parent = {}
        sub = groups.copy()
        sub[~ok] = -1
        tm.fit(groups=sub, parent=parent, rng=rng, per_group_copula=False)
        info["fitted_groups"] = set(tm.group_m)
        # this fact is a parent of a nested fact: its rows become a fixed entity
        if f in self.fact_key_index:
            self._make_fact_entity(f, df, blk_rows, ok)
        # ---- temporal model
        tc = T["fact"].get("time_column")
        if tc:
            b = T["fact"].get("time_bucket", "month")
            t = pd.to_datetime(df[tc], errors="coerce")
            per = t.dt.to_period({"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}[b])
            lo, hi = per.min(), per.max()
            buckets = pd.period_range(lo, hi, freq=lo.freq)
            # bucket number = ordinal difference (ordinals count periods of this freq)
            bidx = np.where(per.notna().to_numpy(), per.array.asi8 - lo.ordinal, -1).astype(np.int64)
            Bn = len(buckets)
            K = e1.K
            cnt = np.zeros((K, Bn))
            okt = ok & (bidx >= 0)
            np.add.at(cnt, (e1.z[a_idx[okt]], bidx[okt]), 1)
            glob = cnt.sum(0) / max(cnt.sum(), 1)
            nnull = int(t.isna().sum())
            info["time"] = {"col": tc, "buckets": buckets, "is_dt": T["columns"][tc]["type"] == "datetime",
                            "tmin": t.min(), "tmax": t.max()}

            n_time = len(t)

            def draw_time(rng, ti=info["time"], cnt=cnt, glob=glob, nnull=nnull, n=n_time):
                # theta_k ~ Dirichlet(n_k + 2 * global profile); null rate ~ Beta(1 + nulls, 1 + rest)
                ti["P"] = np.stack([rng.dirichlet(c + glob * 2 + 1e-6) for c in cnt])
                ti["null_rate"] = float(rng.beta(1 + nnull, 1 + n - nnull)) if nnull else 0.0
            self._register(draw_time)

    @staticmethod
    def _truncate_occ(rel):
        """Zero-truncated rates for mandatory participation.  Drawing the pairs of
        an anchor row conditioned on at least one raises its mean from mu to
        mu / (1 - P0), so rates fitted without the truncation over-generate.  Per
        anchor segment k the free rates p[k, l] (0 < p < 1) are scaled by c_k so
        that the truncated mean equals the observed pairs per anchor row:
        sum_rows E[N | N >= 1] = sum_rows n_i, with E[N | N >= 1] = E[N] / P(N >= 1)
        exact for the Poisson-binomial count of a row (method of moments; the
        maximum likelihood estimate in the Poisson limit).  The truncated mean
        increases in c, so the root is found by bisection.  With eligibility
        groups each (segment, group) cell uses its own partner counts."""
        e1, e2 = rel.e1, rel.e2
        n_obs = np.bincount(rel.u, minlength=e1.n).astype(float)
        g1 = getattr(rel, "g1", None)
        if g1 is not None and len(g1) == e1.n:
            G = rel.G
            m2g = np.bincount(e2.z * G + rel.g2, minlength=e2.K * G).reshape(e2.K, G).astype(float)
        else:
            G, g1 = 1, np.zeros(e1.n, np.int64)
            m2g = e2.counts().astype(float).reshape(e2.K, 1)
        cs = np.ones(e1.K)
        for k in range(e1.K):
            rows = e1.z == k
            if not rows.any():
                continue
            w = np.bincount(g1[rows], minlength=G).astype(float)
            target = n_obs[rows].sum()
            p = np.clip(rel.p[k].astype(float), 0, 1)
            free = (p > 0) & (p < 1)
            if not free.any():
                continue

            def tmean(c):
                q = np.where(free, np.minimum(p * c, 1 - 1e-12), p)
                mu = q @ m2g
                lp0 = np.log1p(-np.minimum(q, 1 - 1e-12)) @ m2g
                return float((w * mu / np.maximum(-np.expm1(lp0), 1e-300)).sum())

            lo, hi = 1e-6, (1 - 1e-9) / p[free].max()
            if tmean(hi) <= target:
                c = hi
            elif tmean(lo) >= target:
                c = lo
            else:
                for _ in range(80):
                    mid = np.sqrt(lo * hi)
                    lo, hi = (mid, hi) if tmean(mid) < target else (lo, mid)
                c = np.sqrt(lo * hi)
            rel.p[k] = np.where(free, np.minimum(p * c, 1 - 1e-9), rel.p[k])
            cs[k] = c
        return cs

    def _finalise_occ(self, f, rel, info):
        """Choose the multiplicity grouping by collapsed marginal likelihood and
        draw the generation parameters under the structural priors."""
        e1, e2 = rel.e1, rel.e2
        setting = info.get("mult_setting", "auto")
        E, T_, A, n1, m2, X = rel.stats(e1.z, e2.z, e1.K, e2.K)
        if setting == "none" or X.sum() == 0:
            choice = "none"
        elif setting in ("partner", "anchor", "block"):
            choice = setting
        else:
            sc = {by: multiplicity_marginal(E, X, by, rel.am, rel.bm, self.pi_s) for by in ("partner", "anchor", "block")}
            choice = max(sc, key=sc.get)
            info["multiplicity_scores"] = {k: round(v, 1) for k, v in sc.items()}
        rel.mult_by = choice
        rel.draw_params(self.rng, self.pi_s)
        info["multiplicity"] = choice
        self.log["decisions"].setdefault(f, {})["multiplicity_chosen"] = choice

    def _fallback_dim(self, f, z1, z2, ok):
        """Dimension whose segments explain more of the fact's attributes
        (bias-corrected mutual information, summed over modelled columns)."""
        setting = self.spec["tables"][f]["fact"].get("group_fallback", self.cfg["facts"]["group_fallback"])
        if setting in ("anchor", "partner"):
            return setting
        tm = self.tm[f]
        s1 = s2 = 0.0
        for cd, x in zip(tm.codecs, tm.X):
            if cd.name in tm.aux:
                continue
            xv = x[ok]
            if cd.kind == "continuous":
                r = pd.Series(xv).rank(pct=True).to_numpy()
                lab = np.where(np.isnan(xv), -1, np.minimum((np.nan_to_num(r) * 10).astype(int), 9))
            else:
                lab = xv
            s1 += _bias_corrected_mi(lab, z1)
            s2 += _bias_corrected_mi(lab, z2)
        return "partner" if s2 > s1 else "anchor"

    def _extra_dist(self, z1, z2, s, K1, K2, ex):
        """P(extra-dimension segment | block): sparse Dirichlet with a back-off
        prior towards P(segment | the more informative block dimension).  Level
        ex.K stands for a null key (only present when the source has nulls)."""
        pi_s, rng = self.pi_s, self.rng
        Kx = ex.K + 1 if (s == ex.K).any() else ex.K
        glob = np.bincount(s, minlength=Kx).astype(float)
        glob = glob / max(glob.sum(), 1)
        mi1, mi2 = _bias_corrected_mi(s, z1), _bias_corrected_mi(s, z2)
        by2 = mi2 > mi1
        coarse_lab, Kc = (z2, K2) if by2 else (z1, K1)
        Cc = np.zeros((Kc, Kx))
        np.add.at(Cc, (coarse_lab, s), 1)
        cnt = np.zeros((K1 * K2, Kx))
        np.add.at(cnt, (z1 * K2 + z2, s), 1)

        def draw(rng):
            Pc = np.stack([sparse_dirichlet_draw(c, glob + 1e-9, pi_s, rng) if c.sum() else glob for c in Cc])
            P = np.zeros_like(cnt)
            for b_ in range(K1 * K2):
                base = Pc[b_ % K2] if by2 else Pc[b_ // K2]
                P[b_] = sparse_dirichlet_draw(cnt[b_], base + 1e-9, pi_s, rng) if cnt[b_].sum() else base
            return {"P": P}
        return {**draw(rng), "backoff": "partner" if by2 else "anchor", "null": Kx > ex.K, "redraw": draw}

    def _count_model(self, f, counts, e, setting=None, key="count_model"):
        """Rows per anchor row: DP posterior of each segment's count distribution
        when it beats Poisson and NB by a decisive Bayes factor (counts.py)."""
        if setting is None:
            setting = self.spec["tables"][f].get("fact", {}).get("count_model", self.cfg["facts"]["count_model"])
        groups = [np.asarray(counts)[e.z == k] for k in range(e.K)]
        dp, det = choose_count_model(groups, setting, a=1.0, bf=self.cfg["facts"].get("count_model_bf", 100.0))
        if det:
            self.log["decisions"].setdefault(f, {})[key + "_evidence"] = det
        if dp:
            self.log["decisions"].setdefault(f, {})[key] = "dp"
        return dp

    def _fit_link_degree(self, f, info, rel):
        """Degree law of each key of a pair (link) fact, e.g. bonds per atom as the
        second atom or actors per movie.  The block model fixes how many rows join
        anchor segment k to partner segment l; within a segment it spreads them
        uniformly (occupancy) or by gamma frailties (Poisson pairs), so a key whose
        rows have near-constant degrees (every atom in about one bond) or heavy-tailed
        ones (actors per movie) is misfitted.  Each side's degrees, grouped by its
        segments, are scored as in counts.py:
          * the DP count law when it beats Poisson and NB by the count-model Bayes
            factor (at sampling time: DP posterior predictive given the total);
          * otherwise NB when it beats Poisson and the side has no frailty of its own
            (occupancy facts, or frailty off): one size r pooled over the segments,
            and i.i.d. NB(r) degrees given their total are Dirichlet-multinomial(r);
          * otherwise the segment law (Poisson degrees are what the block model gives).
        The rows are then matched to the block's links uniformly (_link_degrees).
        Sides that already have a degree law (DP counts per anchor), copied or fixed
        rows (their own observed shares), virtual dimensions and eligibility groups
        keep the segment law."""
        T = self.spec["tables"][f]
        setting = T["fact"].get("link_degree", self.cfg["facts"].get("link_degree", "segment"))
        info["link_degree"] = {}
        if setting == "segment" or rel.kind not in ("fact", "occ") or info.get("eligibility") or info.get("vdim"):
            return
        e1, e2 = rel.e1, rel.e2
        occ = rel.kind == "occ"
        sides = []
        if not (info.get("count_dp") or info.get("min_one")):
            obs1 = np.bincount(rel.u, minlength=e1.n) if occ else info["obs1"]
            sides.append(("anchor", e1, obs1, info["fks"][0]["child_key"], info.get("frailty1")))
        r2 = info["fks"][0 if info["nested"] else 1]
        obs2 = np.bincount(rel.v, minlength=e2.n) if occ else info["obs2"]
        sides.append(("partner", e2, obs2, r2["child_key"], info.get("frailty2")))
        dec = self.log["decisions"].setdefault(f, {})
        for side, e, obs, key, frail in sides:
            if getattr(e, "fixed", False) or getattr(e, "copied", False) or not len(obs):
                continue
            obs = np.asarray(obs).round().astype(np.int64)
            groups = [obs[e.z == k] for k in range(e.K)]
            dp, det = choose_count_model(groups, "dp" if setting == "dp" else "auto", a=1.0,
                                         bf=self.cfg["facts"].get("count_model_bf", 100.0))
            if det:
                dec[f"{key} degree_model_evidence"] = det
            if dp:
                info["link_degree"][side] = {"law": "dp", **dp}
                dec[f"{key} degree_model"] = "dp"
                continue
            no_frailty = occ or frail is None or not np.isfinite(frail).any()
            r = nb_size_pooled(groups) if det.get("parametric") == "nb" and no_frailty else np.inf
            if np.isfinite(r):
                info["link_degree"][side] = {"law": "nb", "r": r,
                                             "min": [bool(len(t) and t.min() >= 1) for t in groups]}
                dec[f"{key} degree_model"] = f"nb (size {r:.3g})"
        info["no_loops"] = e1 is e2 and not bool(np.any(rel.u == rel.v))

    def _link_degrees(self, f, info, rel, z1, z2, u, v, blk, elig1=None, elig2=None, unique=False, skip=None):
        """Rows of a pair fact with the DP degree law on its keys (_fit_link_degree).
        The block of every row is kept, so the block totals N_kl and with them the
        scale rule and the attribute groups are those of the block model.  Per side
        and segment k, the R_k rows whose block involves k get degrees d ~ F_k^R_k:
        i.i.d. from the DP posterior predictive F_k of the segment's source degrees,
        conditioned on summing to R_k (pairwise Gibbs, _conditional_counts), over
        the segment's eligible rows.  The R_k stubs (row i repeated d_i times) are
        dealt to the R_k rows in a uniformly random order, which splits each row's
        degree over the blocks by the multivariate hypergeometric law and, with the
        other side dealt independently, matches the stubs of a block uniformly (the
        block configuration model).  Repeated pairs (occupancy facts, whose pairs are
        unique) and self-links (a link table to one parent whose source has none) are
        then removed by swapping partners within the block, which keeps both degree
        sequences and the blocks (_repair_pairs).  skip = (anchor segments, partner
        segments) whose rows keep their draw (segments touching complete blocks)."""
        ld = info.get("link_degree")
        if not ld or not len(u):
            return u, v, blk
        K2 = rel.e2.K
        rng = self.rng
        sides = (("anchor", z1, u, blk // K2, elig1), ("partner", z2, v, blk % K2, elig2))
        out = {}
        for si, ((side, z, idx, seg, mask), (_, other_z, _, other_seg, _)) in enumerate(zip(sides, sides[::-1])):
            dp = ld.get(side)
            out[side] = idx
            if dp is None or (side == "anchor" and mask is not None):
                continue
            new = idx.copy()
            order = np.argsort(seg, kind="stable")
            ks, starts = np.unique(seg[order], return_index=True)
            members = z if mask is None else np.where(mask, z, -1)
            for k, E in zip(ks.tolist(), np.split(order, starts[1:])):
                if skip is not None and k in skip[si]:
                    continue
                M = np.flatnonzero(members == k)
                if not len(M):
                    continue
                R = len(E)
                if dp["law"] == "nb":
                    # i.i.d. NB(r) given the total: Dirichlet-multinomial(R; r, ..., r);
                    # shifted by one when every source row has a link
                    mn = 1 if k < len(dp["min"]) and dp["min"][k] and R >= len(M) else 0
                    w = rng.gamma(dp["r"], size=len(M))
                    d = mn + rng.multinomial(R - mn * len(M), w / w.sum())
                else:
                    obs = dp["counts"][k] if k < len(dp["counts"]) else np.zeros(0, np.int64)
                    mn = 1 if len(obs) and obs.min() >= 1 and R >= len(M) else 0
                    d = self._conditional_counts(R, len(M), obs, dp["a"], mn=mn, key=("link_degree", f, side, int(k)))
                if unique:
                    d = self._cap_degrees(d, len(np.flatnonzero(np.isin(other_z, np.unique(other_seg[E])))))
                new[E] = np.repeat(M, d)[rng.permutation(R)]
            out[side] = new
        u, v = out["anchor"], out["partner"]
        if unique or info.get("no_loops"):
            pz = z2 if elig2 is None else np.where(elig2, z2, -1)
            u, v, blk, moved, dropped = self._repair_pairs(u, v, blk, len(z2), K2, unique, info.get("no_loops", False),
                                                           partners=lambda b: np.flatnonzero(pz == b % K2))
            if moved or dropped:
                self.sample_log.setdefault("link_degree_repair", {})[f] = {"moved": moved, "dropped": dropped}
        return u, v, blk

    def _cap_degrees(self, d, cap):
        """Degrees of unique pairs cannot exceed the partner rows a row can reach:
        the excess over cap is moved to the rows below it, one link at a time with
        probability proportional to their degree plus one (the total is kept)."""
        d = d.copy()
        if cap <= 0 or d.max(initial=0) <= cap or d.sum() > cap * len(d):
            return d
        rng = self.rng
        excess = int(np.maximum(d - cap, 0).sum())
        d = np.minimum(d, cap)
        while excess:
            room = cap - d
            w = (d + 1.0) * (room > 0)
            add = np.minimum(rng.multinomial(excess, w / w.sum()), room)
            d += add
            excess -= int(add.sum())
        return d

    def _repair_pairs(self, u, v, blk, n2, K2, unique, no_loops, tries=200, partners=None):
        """Remove repeated pairs (unique) and self-links (no_loops) by degree-
        preserving swaps.  A bad row i = (a, b) and a good row j = (a', b') become
        (a, b') and (a', b); with j in the same partner segment the partners are
        exchanged, with j in the same anchor segment the anchors are, so each row
        keeps a block that matches its two segments and every degree and block
        total stays as drawn (rows of i's own block qualify both ways).  A swap is
        taken only when both new pairs are new and not self-links, so each accepted
        swap removes one bad row and adds none.  A row that finds no swap in `tries`
        attempts takes a uniformly chosen row of its block's partner segment
        (partners(b); default: the partners in the block) that it does not link to
        yet, which keeps the row and its block but moves one partner link; rows with
        no free partner at all are dropped.  Returns the rows and the numbers of
        moved and dropped rows."""
        rng = self.rng
        u, v, blk = u.copy(), v.copy(), blk.copy()
        key = u.astype(np.int64) * n2 + v
        first = ~pd.Series(key).duplicated().to_numpy()
        bad = np.zeros(len(u), bool)
        if unique:
            bad |= ~first
        if no_loops:
            bad |= u == v
        moved = dropped = 0
        if not bad.any():
            return u, v, blk, moved, dropped

        def pools(lab):
            o = np.argsort(lab, kind="stable")
            ks, st, ct = np.unique(lab[o], return_index=True, return_counts=True)
            return o, st, ct, {k: i for i, k in enumerate(ks.tolist())}
        by_partner, by_anchor = pools(blk % K2), pools(blk // K2)
        present = set(key[first].tolist()) if unique else set()
        ok_row = ~bad
        stuck = []
        for i in rng.permutation(np.flatnonzero(bad)).tolist():
            cands = []
            for o, st, ct, pos in (by_partner, by_anchor):
                p = pos[int(blk[i] % K2 if o is by_partner[0] else blk[i] // K2)]
                cands.append(o[st[p] + rng.integers(ct[p], size=tries // 2)])
            swap_u = np.repeat([False, True], [len(cands[0]), len(cands[1])])
            mix = rng.permutation(len(swap_u))
            done = False
            for j, su in zip(np.concatenate(cands)[mix].tolist(), swap_u[mix].tolist()):
                if not ok_row[j] or j == i:
                    continue
                ui, vi, uj, vj = int(u[i]), int(v[i]), int(u[j]), int(v[j])
                a, b = ui * n2 + vj, uj * n2 + vi
                if no_loops and (ui == vj or uj == vi):
                    continue
                if unique and (a == b or a in present or b in present):
                    continue
                if unique:
                    present.discard(uj * n2 + vj)
                    if first[i]:                       # a self-link that owned its pair
                        present.discard(ui * n2 + vi)
                    present.add(a)
                    present.add(b)
                if su:
                    u[i], u[j] = uj, ui
                else:
                    v[i], v[j] = vj, vi
                ok_row[i] = True
                done = True
                break
            if not done:
                stuck.append(i)
        if stuck:
            members = {}
            drop = np.zeros(len(u), bool)
            for i in stuck:
                bb = int(blk[i])
                if bb not in members:
                    if partners is not None:
                        members[bb] = partners(bb)
                    else:
                        o, st, ct, pos = by_partner
                        members[bb] = np.unique(v[blk == bb])
                ui = int(u[i])
                cand = members[bb]
                free = next((int(x) for x in cand[rng.permutation(len(cand))].tolist()
                             if not (no_loops and x == ui) and not (unique and ui * n2 + int(x) in present)), None)
                if free is None:
                    drop[i] = True
                    dropped += 1
                    continue
                if unique and first[i]:
                    present.discard(ui * n2 + int(v[i]))
                v[i] = free
                present.add(ui * n2 + free)
                moved += 1
            if drop.any():
                u, v, blk = u[~drop], v[~drop], blk[~drop]
        return u, v, blk, moved, dropped

    def _refine_nested_parent(self, p, f, pe, e2, u, v, cnt, bp, elig=None):
        """Latent segments for the rows of a parent fact, nested in its blocks and
        learned from the child fact's presence pattern (e.g. complete and partial
        filings of the same template).  Second stage of the fit: the parent
        partition is held fixed and the refinement uses the same Gibbs sampler,
        collapsed merge and structural priors, with the blocks as strata."""
        sc = self.cfg["segments"]
        cap = sc["tables"].get(p, sc["default_cap"])
        # each parent row carries one Bernoulli observation per partner row, far more
        # information than an attribute row: segments may be small, and the collapsed
        # merge (exact Bayes factors) decides how many survive
        ns = np.bincount(pe.z, minlength=pe.K)
        caps = np.maximum(1, np.minimum(cap, ns // 2))
        if caps.max() <= 1 or not len(u):
            pe.refined = None
            return pe
        ent = Entity(p, pe.n, None, np.zeros((pe.n, 0)), np.zeros((pe.n, 0), np.int64), [],
                     strata=pe.z, caps=caps)
        grp0 = np.asarray(pe.seg_group)
        rel = OccRel(f, ent, e2, u, v, cnt, table=f, mult_by="block", a=bp, b=bp)
        # hierarchical prior: sub-segment rates are centred on their block's rate
        has = np.bincount(u, minlength=pe.n) > 0
        Eb = np.zeros((pe.K, e2.K))
        np.add.at(Eb, (pe.z[u], e2.z[v]), 1.0)
        if elig is not None:
            rel.set_eligibility(*elig)
            N1g, N2g = rel.elig_counts(pe.z, e2.z, pe.K, e2.K, act=has)
            Tb = N1g @ N2g.T
        else:
            Tb = np.outer(np.bincount(pe.z[has], minlength=pe.K), e2.counts())
        rel.prior_mean = np.clip(np.where(Tb > 0, Eb / np.maximum(Tb, 1), 0.0), 0, 1)
        n_before = len(e2.rels)
        sub_cfg = {**self.cfg, "gibbs": {**self.cfg["gibbs"], "merge": False}}
        RelationalModel({p: ent, e2.name: e2}, [rel], sub_cfg, self.rng).fit()
        del e2.rels[n_before:]
        nmerge = self._merge_within_strata(ent, rel)
        ent.fixed = True
        ent.seg_group = grp0[ent.seg_stratum]
        ent.prior_mean = rel.prior_mean
        # within-block proportions of the refined segments (Dirichlet posterior draw)
        cnts = np.bincount(ent.z, minlength=ent.K).astype(float)
        ent.refined = {"block_of_seg": ent.seg_stratum.copy()}
        self._register(lambda rng, ref=ent.refined, c=cnts: ref.__setitem__("w", rng.dirichlet(c + 0.5)))
        self.log["decisions"].setdefault(f, {})["parent_segments"] = {
            "blocks": int(pe.K), "segments": int(ent.K), "merges": nmerge}
        self.fact_entity[p] = ent
        return ent

    @staticmethod
    def _merge_within_strata(ent, rel):
        """Greedy merge of segments inside each stratum on the collapsed marginal of
        the occupancy relation.  With the block grouping of multiplicities the
        marginal is a sum of per-segment terms, so the gain of merging a and b is
        s(a+b) - s(a) - s(b) and strata can be treated one at a time."""
        a_, b_ = rel.a, rel.b
        E, T, A, n1, m2, X = rel.stats(ent.z, rel.e2.z, ent.K, rel.e2.K)
        am, bm = rel.am, rel.bm
        PA, PB = rel.beta_ab()
        PA, PB = np.broadcast_to(PA, E.shape), np.broadcast_to(PB, E.shape)
        seg_ab = {k: (PA[k], PB[k]) for k in range(ent.K)}   # same prior for segments of a stratum

        def seg_score(E_, A_, n_, X_, ab):
            pa, pb = ab
            if np.ndim(A_):                              # per-group active rows (eligibility)
                T_ = A_ @ m2
                A_ = A_.sum()
            else:
                T_ = A_ * m2
            s_ = gammaln(max(n_, 1)) + betaln(a_ + A_, b_ + n_ - A_) - betaln(a_, b_)
            ok = T_ > 0
            s_ += (betaln(pa[ok] + E_[ok], pb[ok] + T_[ok] - E_[ok]) - betaln(pa[ok], pb[ok])).sum()
            g = E_ > 0
            s_ += (am * np.log(bm) - gammaln(am) + gammaln(am + X_[g]) - (am + X_[g]) * np.log(bm + E_[g])).sum()
            return s_
        if rel.g1 is not None:
            # eligible pairs at risk of a segment: its active rows per group x partners per group
            N1g, N2g = rel.elig_counts(ent.z, rel.e2.z, ent.K, rel.e2.K)
            m2 = N2g.T                                   # G x K2; A_ below is a G-vector
            A = N1g
        segs = {k: [E[k].copy(), A[k], n1[k], X[k].copy()] for k in range(ent.K)}
        score = {k: seg_score(*segs[k], seg_ab[k]) for k in segs}
        label = np.arange(ent.K)
        merges = 0
        for st in np.unique(ent.seg_stratum):
            live = [k for k in range(ent.K) if ent.seg_stratum[k] == st]
            while len(live) > 1:
                best, arg = 0.0, None
                for i, a in enumerate(live):
                    for b in live[i + 1:]:
                        m = [segs[a][0] + segs[b][0], segs[a][1] + segs[b][1], segs[a][2] + segs[b][2],
                             segs[a][3] + segs[b][3]]
                        d = seg_score(*m, seg_ab[a]) - score[a] - score[b]
                        if d > best:
                            best, arg = d, (a, b, m)
                if arg is None:
                    break
                a, b, m = arg
                segs[a], score[a] = m, seg_score(*m, seg_ab[a])
                label[label == b] = a
                live.remove(b)
                merges += 1
        keep = np.unique(label)
        remap = -np.ones(ent.K, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        ent.z = remap[label[ent.z]]
        ent.seg_stratum = ent.seg_stratum[keep]
        ent.K = len(keep)
        return merges

    def _make_fact_entity(self, f, df, blk_rows, ok):
        """Rows of fact f become a fixed entity (segments = their blocks) so that a
        nested fact can use them as its anchor."""
        lab = np.full(len(df), -1)
        lab[ok] = blk_rows
        u, z = np.unique(np.where(lab >= 0, lab, lab.max(initial=0) + 1), return_inverse=True)
        n = len(df)
        pe = Entity(f, n, len(u), np.zeros((n, 0)), np.zeros((n, 0), np.int64), [])
        pe.fixed = True
        pe.z = z.astype(np.int64)
        pe.K = len(u)
        pe.seg_stratum = np.zeros(pe.K, dtype=np.int64)
        # group of a block = the stratum group of its anchor segment (unmatched rows: own group)
        rel = self.fact_info[f].get("rel")
        if rel is not None and hasattr(rel, "e1"):
            K2 = rel.e2.K
            ag = np.asarray(rel.e1.seg_group)
            pe.seg_group = np.array([int(ag[b // K2]) if b // K2 < len(ag) and b >= 0 else -1 for b in u])
        self.fact_entity[f] = pe
        self.fact_block_map = getattr(self, "fact_block_map", {})
        self.fact_block_map[f] = {int(b): i for i, b in enumerate(u)}

    # ------------------------------------------------------------------ constraints
    def _fit_constraints(self, tables):
        """Unique keys and key rules that sampling must satisfy."""
        spec = self.spec
        T = spec["tables"]
        self.cons = {}
        self.rake = {}
        self.late = {t: set(self.aggregated[t]) for t in T}
        for _ in range(len(spec["rules"]) + 1):
            for r in spec["rules"]:
                if r["type"] == "derive":
                    lhs, rhs = split_assignment(r["expr"])
                    if names_in(rhs) & self.late[r["table"]]:
                        self.late[r["table"]].add(lhs)
        for t in T:
            fkcols = {r["child_key"] for r in self.fk[t]}
            # keys over links, and keys over links and modelled attributes (an account
            # number unique per customer): the attributes are drawn before the links,
            # so moving the links conditions on both (keys of attributes only: _finish)
            attrs = set(self.tm[t].out_cols) if t in getattr(self, "tm", {}) else set()
            ukeys = [u for u in T[t].get("unique", []) if set(u) & fkcols and set(u) <= fkcols | attrs]
            # links fixed by stratification are not moved by the link sampler: a rule that
            # reads only those links and the row's attributes is a rule on the attributes
            fixed = set(getattr(self, "strata_info", {}).get(t, {}).get("by", [])) if T[t]["role"] == "entity" else set()
            krules, arules = [], []
            for r in spec["rules"]:
                if r["table"] != t or r["type"] != "check":
                    continue
                used = names_in(r["expr"])
                if r.get("anchor_link") and T[t]["role"] == "entity":
                    continue                 # satisfied by construction (_choose_anchored), reported below
                links = {u.split(".")[0] for u in used if "." in u} | {u for u in used if u in fkcols}
                (krules if links and not links <= fixed else arules).append(r)
            # rules that read an aggregate (directly or through a derived column) can only be
            # evaluated once the child tables exist; they are applied/reported after the aggregates
            late = self.late[t]
            krules = [r for r in krules if not (names_in(r["expr"]) & late)]
            arules = [r for r in arules if not (names_in(r["expr"]) & late)]
            self.cons[t] = {"unique": ukeys, "key_rules": krules, "attr_rules": arules,
                            "anchor_rules": [r for r in spec["rules"] if r["table"] == t and r.get("anchor_link")
                                             and T[t]["role"] == "entity"]}
            if krules and T[t]["role"] == "entity" and t not in self.copied:
                self._fit_rake(t, krules, tables)
            for u in T[t].get("unique", []):
                if all(c in tables[t].columns for c in u):
                    d = int(tables[t].duplicated(u).sum())
                    if d:
                        self.log["warnings"].append(f"{t}: the source has {d} duplicate rows on unique key {u}")

    def _fit_rake(self, t, krules, tables, max_levels=50, max_cells=2000):
        """Source distribution of the cells used to calibrate key rules: the joint
        of the categorical columns (at most 50 levels each) that the rules read
        through the links being moved."""
        if not hasattr(self, "rake"):
            self.rake = {}
        df = tables[t]
        res = make_resolver(self.spec, t, df, tables)
        paths, parts = [], []
        for r in krules:
            for p in sorted(names_in(r["expr"])):
                if "." not in p or p in paths:
                    continue
                try:
                    v = pd.Series(np.asarray(res(p), dtype=object))
                except (KeyError, SpecError):
                    continue
                from .profiling import is_dateish
                if v.nunique() > max_levels or v.nunique() < 2 or is_dateish(v):
                    continue
                paths.append(p)
                parts.append(_keystr(v))
        if not paths:
            return
        cells = pd.Series(_tuple_frame(parts))
        counts = cells.value_counts().astype(float)
        if len(counts) > max_cells:
            return
        # the target cell shares: Dirichlet(counts + 1/2) posterior over the observed
        # cells, drawn for each dataset (its mean at fit)
        self.rake[t] = {"paths": paths, "counts": counts, "dist": (counts + 0.5) / (counts + 0.5).sum()}

    def _fit_text_pii(self, tables):
        self.text, self.pii = {}, {}
        tcfg = self.cfg["text"]
        for t, ts in self.spec["tables"].items():
            for c, cs in ts["columns"].items():
                if cs["type"] == "text":
                    o = cs.get("text") or {}
                    cond = o.get("condition_on")
                    classes = tables[t][cond].astype(str).tolist() if cond else self._row_segments(t, tables[t])
                    self.text[(t, c)] = TextModel(order=o.get("order", tcfg["order"]),
                                                  min_token_rows=o.get("min_token_rows", tcfg["min_token_rows"]),
                                                  max_copy_ngram=o.get("max_copy_ngram", tcfg["max_copy_ngram"]),
                                                  beta=tcfg["backoff_strength"], max_retries=tcfg["max_retries"],
                                                  whole=bool(o.get("whole_values", False))
                                                  ).fit(tables[t][c].tolist(), classes)
                    self.text[(t, c)].label = _surrogate_label(t, c)
                    if self.text[(t, c)].surrogate:
                        self.log["decisions"].setdefault(t, {})[f"{c} text"] = (
                            f"surrogate labels: only {self.text[(t, c)].word_coverage:.0%} of rows keep a word "
                            "seen in at least min_token_rows rows")
                elif cs["type"] == "pii":
                    self.pii[(t, c)] = PiiGenerator(cs["pii"], tables[t][c].tolist(), label=_surrogate_label(t, c))
                elif cs["type"] == "constant":
                    v = tables[t][c].dropna()
                    self.pii[(t, c)] = ("const", v.iloc[0] if len(v) else None)
                elif cs["type"] == "id" and c != ts.get("primary_key") and c in tables[t].columns:
                    # natural keys and codes ('0101001', 'BK-M68B-38'): new values of the
                    # source's shapes, unique; integers without leading zeros stay 1..n
                    v = tables[t][c].dropna().astype(str)
                    if len(v) and (pd.to_numeric(v, errors="coerce").isna().any() or v.str.match(r"^[+-]?0\d").any()):
                        self.pii[(t, c)] = PiiGenerator({"generator": "pattern"}, v.tolist(), label=_surrogate_label(t, c))

    def _row_segments(self, t, df):
        if t in self.E:
            return [str(x) for x in self.E[t].z]
        info = self.fact_info.get(t)
        if info:
            idx = self._anchor_index(t, df, self._tables_for_fit)
            z = self.fact_entity[info["anchor"]].z if info["nested"] else self.E[info["anchor"]].z
            return [str(z[i]) if i >= 0 else "na" for i in idx]
        return ["0"] * len(df)

    # ================================================================== sample
    def sample(self, scale=None, index=None):
        """One synthetic dataset: every parameter drawn once from its posterior
        (or set to its posterior mean) from the dataset's parameter stream, then
        the rows from its data stream.  index numbers the datasets (0, 1, ...);
        without it each call gives the next dataset."""
        from .posterior import run_streams, RunCache
        t0 = time.time()
        cfg = self.cfg
        pc = cfg.get("posterior", {})
        mode = pc.get("mode", "draw")
        index = self._runs if index is None else index
        self._runs = index + 1
        prng, drng = run_streams(cfg["seed"], pc.get("sample_seed"), index)
        self.redraw(mode, prng)
        self.rng = rng = drng
        self.run_cache = RunCache(mode)
        self._mode = mode
        self.posterior_log = {"mode": mode, "dataset": index}
        T = self.spec["tables"]
        sc = cfg["scale"] if scale is None else scale
        sc = self.plan.scale(sc)
        out, zs, keys = {}, {}, {}
        out.update(self.plan.ref_frames)        # reference tables read by keyed lookups
        weights_cache = {}
        self._wcache, self._zs = weights_cache, zs        # for redrawing anchored links in _finish
        self.sample_log = {"rule_violations": {}, "text_rejections": {}, "rows_removed_by_rules": {},
                           "uniqueness": {}, "warnings": []}       # warnings of this dataset only
        self._edge_masks = {}
        say("sample: entities")
        for e in self.order:
            ent = self.E[e]
            tree = next((i for i in self.edge_info.values() if i["rel"].kind == "tree" and i["rel"].e1 is ent), None)
            s = sc["tables"].get(e, sc["default"])
            if e in self.copied:
                df = self._copy_rows[e].reset_index(drop=True)
                keys[e] = df[T[e]["primary_key"]].to_numpy()
                zs[e] = ent.z.copy()
                out[e] = df
                continue
            if tree is not None:
                z, par = self._grow_tree(tree, s)
                # posterior predictive depth profiles for the evaluation, on their own
                # stream: the dataset's rows do not depend on whether it is computed
                ppc, data_rng = [], self.rng
                base = cfg["seed"] if pc.get("sample_seed") is None else pc["sample_seed"]
                self.rng = np.random.default_rng(np.random.SeedSequence([int(base), 7919, int(index), 104729]))
                try:
                    for _ in range(40):
                        _, p2 = self._grow_tree(tree, s)
                        ppc.append(np.bincount(tree_depth(p2.copy())).tolist())
                finally:
                    self.rng = data_rng
                self.sample_log.setdefault("tree_ppc", {})[tree["spec"]["name"]] = ppc
            elif e in self.grain:
                z, gk = self._sample_grain(e, zs, keys, out)
                par = None
            elif e in self.attr_strata and ent.n // max(self.cfg["segments"]["min_rows_per_segment"], 1) < 2:
                # a tiny entity: its segments' sizes as observed (scaled)
                z = np.repeat(np.arange(ent.K), np.round(s * np.bincount(ent.z, minlength=ent.K)).astype(np.int64))
                z = rng.permutation(z)
                par = None
            else:
                n = int(round(s * ent.n))
                z = allocate(ent.pi / ent.pi.sum(), n, rng)
                par = None
            df = self.tm[e].sample(z, rng)
            pk = T[e]["primary_key"]
            keys[e] = self._make_keys(e, len(z))
            ko = T[e].get("key_order")
            if ko is not None and ko in df.columns and len(z):
                # keys numbered in the order of a column (order numbers follow the order
                # dates): the k-th smallest value gets the k-th key; ties in row order
                from .expr import _days, to_array
                v = to_array(df[ko])
                v = _days(v) if v.dtype == object else v
                rank = np.argsort(np.argsort(np.where(np.isnan(v), np.inf, v), kind="stable"), kind="stable")
                keys[e] = keys[e][rank]
            df.insert(0, pk, keys[e])
            if par is not None:
                tr = tree["spec"]
                df[tr["child_key"]] = pd.array([keys[e][q] if q >= 0 else None for q in par], dtype=object)
            fixed_fk = self._stratum_keys(e, z, out)
            if e in self.attr_strata:
                df[self.attr_strata[e]] = fixed_fk.pop(self.attr_strata[e])
            if e in self.grain:
                fixed_fk.update(gk)
            for r in self.fk[e]:
                if r["child_key"] in fixed_fk:
                    df[r["child_key"]] = fixed_fk[r["child_key"]]
                    if r.get("anchor") and r["anchor"] in df.columns:
                        # an anchored link fixed by stratification: the segments nest in
                        # the dimension rows, so the anchor is the stratum's own value
                        df[r["anchor"]] = self._anchor_from_key(r, df, out)
                    continue
                info = self.edge_info[r["name"]]
                if r.get("anchor") and r["anchor"] in df.columns:
                    df[r["child_key"]] = self._choose_anchored(info, z, zs[r["parent"]], keys[r["parent"]],
                                                               weights_cache, r, df[r["anchor"]], out[r["parent"]])
                    continue
                mask = self._inclusion_mask(e, r, len(zs[r["parent"]]), out)
                self._edge_masks[r["name"]] = mask
                df[r["child_key"]] = self._choose_parents(info, z, zs[r["parent"]], keys[r["parent"]],
                                                          weights_cache, r["name"], mask=mask)
            zs[e] = z
            self._ixcache = {}          # parents' key indexes, fixed while e's links are moved
            try:
                out[e] = self._constrain_entity(e, df, z, zs, keys, out, weights_cache)
                out[e] = self._apply_affinity(e, out[e], out)
                out[e] = self._apply_participation(e, out[e], out, zs)
            finally:
                self._ixcache = None
            # lookups and derived columns now, so that the rules of later entities can
            # read them through their links (a customer's country from its geography)
            out[e] = self._computed(e, out[e], out, strict=False)
        # finish parents before children: lookups and rules read finished parents, and
        # facts are drawn against finished entities (their rules see every column)
        for t in self.all_order:
            if t in self.E:
                out[t] = self._finish(t, out[t], zs.get(t), out)
                if t in getattr(self, "scd", {}):
                    # columns computed from the version dates (a status, a current flag)
                    out[t] = self._computed(t, self._apply_scd(t, out[t], zs.get(t)), out)
                self._renumber_by_order(t, out, keys)
        fact_rows = {}
        for f in self.fact_order:
            say(f"sample: fact {f}")
            out[f], fact_rows[f] = self._sample_fact(f, zs, keys, sc, out, fact_rows)
            out[f] = self._apply_affinity(f, out[f], out)
            out[f] = self._apply_bridge(f, self._finish(f, out[f], None, out))
            say(f"sample: fact {f}: {len(out[f]):,} rows")
        for t in out:
            if t in self.spec["tables"] and _hidden_cols(out[t]):
                out[t] = out[t].drop(columns=_hidden_cols(out[t]))
        say("sample: aggregates")
        self._aggregates(out)
        self.sample_log["seconds"] = round(time.time() - t0, 2)
        say(f"sample: done in {time.time() - t0:.0f} s")
        self.internal_out = out
        return self.plan.to_external(out, self.rng, getattr(self, "_mode", "draw"))

    def _stratum_keys(self, e, z, out):
        """Strata are rows of small copied parents and segments are nested in them,
        so a row's stratum columns are determined by its segment (exact by
        construction, no leakage through the smoothed link probabilities)."""
        info = self.strata_info.get(e)
        if not info or not len(z):
            return {}
        ent = self.E[e]
        st = ent.seg_stratum[z]
        res = {}
        if info.get("attr"):
            res[info["attr"]] = np.array([l[0] for l in info["labels"]], dtype=object)[st]
        for pos, col in enumerate(info["by"]):
            r = next(r for r in self.fk[e] if r["child_key"] == col)
            pkeys = out[r["parent"]][self.spec["tables"][r["parent"]]["primary_key"]]
            back = dict(zip(_keystr(pkeys), pkeys.to_numpy(dtype=object)))
            lab = [(l if len(info["by"]) > 1 else (l,))[pos] for l in info["labels"]]
            vals = np.array([back.get(v) for v in lab], dtype=object)
            res[col] = vals[st]
        return res

    def _sample_grain(self, e, zs, keys, out):
        """Rows of a grain entity: occupied pairs of its two key parents from the
        occupancy model, then each row's segment from its conditional given the
        parents' segments, p(z | k, l) ∝ pi_z phi_A[z, k] phi_B[z, l] (Bayes' rule
        under the fitted link model), restricted to segments of the row's stratum."""
        g = self.grain[e]
        rel, ent, rng = g["rel"], self.E[e], self.rng
        ra, rb = g["anchor"], g["partner"]
        za, zb = zs[ra["parent"]], zs[rb["parent"]]
        u, v, _ = self._sample_occ(rel, za, zb)
        logw = np.log(np.maximum(ent.pi, 1e-300))[None, :] \
            + np.log(np.maximum(self.edge_info[ra["name"]]["phi"][:, za[u]].T, 1e-300)) \
            + np.log(np.maximum(self.edge_info[rb["name"]]["phi"][:, zb[v]].T, 1e-300))
        info = self.strata_info.get(e)
        if info:
            row_key = {ra["child_key"]: _keystr(keys[ra["parent"]][u]).to_numpy(dtype=object),
                       rb["child_key"]: _keystr(keys[rb["parent"]][v]).to_numpy(dtype=object)}
            by = info["by"]
            labs = [l if len(by) > 1 else (l,) for l in info["labels"]]
            ok = np.ones((len(u), ent.K), bool)
            for pos, col in enumerate(by):
                if col not in row_key:
                    continue
                seg_val = np.array([labs[s_][pos] for s_ in ent.seg_stratum], dtype=object)
                ok &= row_key[col][:, None] == seg_val[None, :]
            logw = np.where(ok, logw, -np.inf)
            dead = ~ok.any(1)
            if dead.any():          # pair of parents never seen in any stratum: no segment can host it
                keep = ~dead
                u, v, logw = u[keep], v[keep], logw[keep]
                self.sample_log["warnings"].append(f"{e}: {int(dead.sum())} sampled key pairs fall outside every "
                                            "observed stratum and were dropped")
        z = np.argmax(logw + rng.gumbel(size=logw.shape), axis=1) if len(u) else np.zeros(0, np.int64)
        gk = {ra["child_key"]: keys[ra["parent"]][u].astype(object),
              rb["child_key"]: keys[rb["parent"]][v].astype(object)}
        return z, gk

    def _renumber_by_order(self, t, out, keys):
        """Keys in the order of a column (Prop. keyorder) once the rows are final:
        attribute redraws in _finish can move the column after the keys were
        numbered.  The new numbering is a permutation of the same labels, applied to
        the table, its keys array and every link already pointing at it."""
        T = self.spec["tables"][t]
        ko = T.get("key_order")
        df = out[t]
        if ko is None or ko not in df.columns or not len(df) or t in self.copied:
            return
        from .expr import _days, to_array
        v = to_array(df[ko])
        v = _days(v) if v.dtype == object else v
        pk = T["primary_key"]
        cur = keys[t]
        srt = _natural_sort(cur)
        rank = np.argsort(np.argsort(np.where(np.isnan(v), np.inf, v), kind="stable"), kind="stable")
        new = srt[rank]
        if np.array_equal(new, cur):
            return
        m = dict(zip(cur.tolist(), new.tolist()))
        keys[t] = new
        df[pk] = new
        from .spec import path_fks
        for c in out:
            for r in path_fks(self.spec, c):
                if r["parent"] == t and r["child_key"] in out[c].columns:
                    col = out[c][r["child_key"]]
                    out[c][r["child_key"]] = _remap(m, col)
        for i in (self.edge_info or {}).values():
            if i["rel"].kind == "tree" and i["rel"].e1 is self.E.get(t):
                ck = i["spec"]["child_key"]
                if ck in df.columns:
                    df[ck] = _remap(m, df[ck])
        self.sample_log.setdefault("key_order_renumbered", {})[t] = int((new != cur).sum())

    def _make_keys(self, t, n):
        src = self.key_index.get(t)
        if src is not None and len(src) and isinstance(src[0], str):
            s0 = str(src[0])
            pre = s0.rstrip("0123456789")
            width = len(s0) - len(pre)
            return np.array([f"{pre}{i:0{max(width, 1)}d}" for i in range(1, n + 1)], dtype=object)
        return np.arange(1, n + 1)

    def _weights(self, alpha_k, z, cache, key, ent=None, obs=None):
        """Degree correction for the rows of a segment, rescaled to sum to the
        segment size: it spreads links unevenly across members without changing the
        segment's expected total (keeps the scale rule exact).  New rows get Gamma
        frailties.  Rows of a copied table are the source rows themselves, so their
        weights are a posterior draw of their own share of the segment's links,
        Dirichlet(observed links + 1/m_k) over the m_k rows of segment k (the degree
        parameters of a degree-corrected block model)."""
        if key is not None and key in cache:
            return cache[key]
        if ent is not None and getattr(ent, "copied", False) and obs is not None and len(obs) == len(z):
            obs = np.asarray(obs, float)
            K = len(alpha_k)
            m = np.bincount(z, minlength=K).astype(float)
            # prior concentration a_k of the row rates in segment k (Gamma(a_k) rates,
            # i.e. Dirichlet(a_k) shares), by the method of moments on the observed
            # counts: var = mean + mean^2 / a.  A posterior draw Gamma(obs + a_k)
            # averaged over data from the model is the prior again, so synthetic
            # counts per row keep the source's dispersion: rows that are alike
            # (orders per day) get near-equal shares instead of their noisy counts,
            # rows that differ (stock only on weekly dates) keep their own.
            mu = np.bincount(z, weights=obs, minlength=K) / np.maximum(m, 1)
            ex2 = np.bincount(z, weights=obs * obs, minlength=K) / np.maximum(m, 1)
            var = (ex2 - mu ** 2) * m / np.maximum(m - 1, 1)
            a = np.where((var > mu * (1 + 1e-9)) & (mu > 0), mu ** 2 / np.maximum(var - mu, 1e-12), 1e6)
            a = np.maximum(a, 1.0 / np.maximum(m, 1.0))
            # posterior mean mode: the Gamma's mean obs + a_k (no draw from the data stream)
            w = obs + a[z] if getattr(self, "_mode", "draw") == "mean" else self.rng.gamma(obs + a[z])
            tot = np.bincount(z, weights=w, minlength=len(alpha_k))
            w = w * (m[z] / np.maximum(tot[z], 1e-300))
            if key is not None:
                cache[key] = w
            return w
        a = alpha_k[z]
        w = np.where(np.isfinite(a), self.rng.gamma(np.where(np.isfinite(a), a, 1.0), 1 / np.where(np.isfinite(a), a, 1.0)), 1.0)
        tot = np.bincount(z, weights=w, minlength=len(alpha_k))
        cnt = np.bincount(z, minlength=len(alpha_k))
        w = w * (cnt[z] / np.maximum(tot[z], 1e-12))
        if key is not None:
            cache[key] = w
        return w

    def _edge_count_model(self, name, fan, p, child):
        """Children per parent row (e.g. every customer has 7 invoices): DP counts
        when they beat Poisson and NB by a decisive Bayes factor; the children of
        each parent segment are then spread by the DP posterior predictive of that
        segment's counts, conditioned on the segment's total."""
        if p.fixed or getattr(p, "copied", False) or not len(fan):
            return None
        return self._count_model(child, fan.astype(np.int64), p, setting=self.cfg["facts"]["count_model"],
                                 key=f"{name} count_model")

    def _conditional_counts(self, R, M, obs, a, sweeps=30, mn=0, key=None):
        """M counts, i.i.d. from the DP posterior predictive F (Bayesian-bootstrap
        weights on the observed counts plus mass a on Poisson(mean)), conditioned on
        summing to R.  Pairwise Gibbs: a pair's sum is kept and its split is redrawn
        from f(x) f(s - x), which leaves the conditional law invariant; the chain
        starts from the balanced allocation."""
        rng = self.rng
        if M == 0:
            return np.zeros(0, np.int64)
        n = len(obs)
        lam = obs.mean() if n else R / M
        cmax = int(max(R, obs.max(initial=0))) + 1
        x = np.arange(cmax + 1)
        # base measure mn + Poisson(lam - mn) (mn = 1 under mandatory participation)
        lb = max(lam - mn, 1e-12)
        xm = np.maximum(x - mn, 0)
        f = np.where(x >= mn, (a / (n + a)) * np.exp(xm * np.log(lb) - lb - gammaln(xm + 1)), 0.0)
        if n:
            w = self._bb(key if key is not None else ("counts", id(obs)), n, rng)
            np.add.at(f, np.minimum(obs, cmax), (n / (n + a)) * w)
        lf = np.log(np.maximum(f, 1e-300))
        N = np.full(M, R // M, dtype=np.int64)
        N[rng.permutation(M)[: R - N.sum()]] += 1
        if M < 2:
            return N
        for _ in range(sweeps):
            perm = rng.permutation(M)
            i, j = perm[0:M - 1:2], perm[1:M:2]
            ssum = N[i] + N[j]
            xs = np.arange(ssum.max() + 1)
            L = lf[np.minimum(xs[None, :], cmax)] + lf[np.minimum(np.maximum(ssum[:, None] - xs[None, :], 0), cmax)]
            L = np.where(xs[None, :] <= ssum[:, None], L, -np.inf)
            pick = gumbel_argmax(L, rng)
            N[i], N[j] = pick, ssum - pick
        return N

    @staticmethod
    def _member_law(info, w):
        """Within-segment law of a link's parent: by the degree weights w, or uniform
        when the link has a count model (exchangeable counts per member)."""
        return np.ones(len(w)) if info.get("count_dp") else w

    def _assign(self, info, t, rr, mm, wl, pkeys, res):
        """Rows rr linked to members mm of parent segment t: counts from the link's
        count model given their total (the source's dispersion), else i.i.d. by wl."""
        rng = self.rng
        dp = info.get("count_dp")
        if dp and len(rr) > 1:
            obs = dp["counts"][t] if t < len(dp["counts"]) else np.zeros(0, np.int64)
            cnt = self._conditional_counts(len(rr), len(mm), obs, dp["a"], mn=dp.get("min", 0),
                                           key=("dp", id(dp), int(t)))
            res[rng.permutation(rr)] = pkeys[np.repeat(mm, cnt)]
        else:
            ww = wl[mm] / wl[mm].sum()
            res[rr] = pkeys[mm[rng.choice(len(mm), size=len(rr), p=ww)]]

    def _cell_sampler(self, e, df, z, zs, keys, out, cache, movable, paths):
        """Exact draws of an entity's links under rules, calibrated (Proposition
        rake).  Without unique keys the rows are independent given their segments,
        so the target factorises over rows:
            pi_i(L) ∝ Q_i(L) 1[valid] w(c(L)) prod_j u_j(t_j(L)),
        with c the row's cell (categorical columns the rules read through the
        links) and t_j the segment of link j's parent.  A cell is drawn with
        probability ∝ w(c) prod_j sum_t phi_j[z_i, t] u_j(t) W_j(t, c_j) (W: the
        share of segment t's parents with the cell's value), then for each link a
        parent segment ∝ phi_j[z_i, t] u_j(t) W_j(t, c_j) and a parent among those
        with the value (by the link's count model or degree weights).  The row is
        kept if every rule holds, otherwise redrawn: the kept rows have law exactly
        pi_i.  Cell coordinates must be columns of a direct parent of a movable link
        or values the row already has (a link fixed by stratification, a column of
        the row); returns None otherwise (the Metropolis-Hastings kernel is used).
        draw(rows, cells, w, u) -> (links, cell index, parent segments)."""
        fkmap = {r["child_key"]: r for r in movable}
        by_link, fixed = {}, []
        res0 = self._resolver(e, df, out)
        for i, p in enumerate(paths):
            parts = p.split(".")
            if len(parts) == 2 and parts[0] in fkmap:
                by_link.setdefault(parts[0], []).append((i, parts[1]))
                continue
            if parts[0] in fkmap:
                return None                      # a path through a movable link, two hops away
            try:
                v = res0(p) if "." in p else df[p]
            except (KeyError, SpecError):
                return None
            fixed.append((i, np.asarray(_keystr(pd.Series(np.asarray(v, dtype=object))), dtype=object)))
        sep = "\x1f"

        def J(parts):                            # one string per coordinate tuple (None = null)
            return sep.join("\x00" if x is None else str(x) for x in parts)
        links = []
        for ck, r in fkmap.items():
            info = self.edge_info[r["name"]]
            phi = info["phi"]
            zp = zs[r["parent"]]
            Kp = phi.shape[1]
            w = self._member_law(info, self._weights(info["parent_frailty"], zp, cache, ("edge", r["name"]),
                                                     ent=info["rel"].e2, obs=info.get("obs")))
            if not (w > 0).any():
                w = np.ones(len(zp))
            m = np.bincount(zp, weights=w, minlength=Kp)
            m = np.where(m > 0, m, 1.0)
            par = out[r["parent"]]
            cols = [c for _, c in by_link.get(ck, [])]
            if any(c not in par.columns for c in cols):
                return None
            if cols:
                pv = pd.Series(J(x) for x in zip(*[_keystr(par[c]) for c in cols])).to_numpy(dtype=object)
            else:
                pv = np.full(len(zp), "", dtype=object)          # no coordinate: one level
            levels = pd.unique(pv)
            lix = {v: k for k, v in enumerate(levels)}
            code = np.array([lix[v] for v in pv], dtype=np.int64)
            Wv = np.zeros((Kp, len(levels)))
            np.add.at(Wv, (zp, code), w)
            Wv /= m[:, None]
            lix[J([None] * len(cols)) if cols else "\x00null"] = len(levels)       # a null link
            links.append({"r": r, "info": info, "zp": zp, "w": w, "Kp": Kp, "keys": keys[r["parent"]],
                          "coords": [i for i, _ in by_link.get(ck, [])], "lix": lix, "Wv": Wv,
                          "members": {(t, v): np.flatnonzero((zp == t) & (code == v))
                                      for t, v in zip(*np.nonzero(Wv > 0))}})
        rng = self.rng

        def coord(L, c):
            if not L["coords"]:
                return 0
            tup = c if isinstance(c, tuple) else (c,)
            return L["lix"].get(J(tup[i] for i in L["coords"]), -1)

        for L in links:
            # the link's overall mixture over parent segments: the fallback for rows whose
            # own segment gives every valid state zero weight
            phi = L["info"]["phi"]
            L["phi_mix"] = (np.bincount(z, minlength=phi.shape[0]).astype(float) @ phi) / max(len(z), 1)

        def draw(rows, cells, wcell, useg, mix=False):
            zc = np.zeros(len(rows), int) if mix else z[rows]
            Kc = 1 if mix else (int(zc.max()) + 1 if len(zc) else 0)
            R = np.tile(np.array([wcell.get(c, 1.0) for c in cells], float), (Kc, 1))
            for L in links:
                pn = L["info"]["p_null"][:Kc] if not mix else np.array([float(L["info"]["p_null"][z].mean())])
                u = useg.get(L["r"]["name"], np.ones(L["Kp"]))
                ph = L["phi_mix"][None, :] if mix else L["info"]["phi"][:Kc]
                M = (1 - pn)[:, None] * (ph @ (u[:, None] * L["Wv"]))
                M = np.concatenate([M, pn[:, None]], axis=1)
                cc = np.array([coord(L, c) for c in cells])
                R *= np.where(cc[None, :] >= 0, M[:, np.maximum(cc, 0)], 0.0)
            pick = np.full(len(rows), -1)
            if fixed:
                # coordinates the row already has: only the cells that agree with them
                fv = [J(x) for x in zip(*[f[rows] for _, f in fixed])]
                cfv = np.array([J((c if isinstance(c, tuple) else (c,))[i] for i, _ in fixed) for c in cells],
                               dtype=object)
                groups = pd.Series(np.arange(len(rows))).groupby([zc, fv]).indices.items()
            else:
                groups = ((k, np.flatnonzero(zc == k)) for k in np.unique(zc))
            for key, ii in groups:
                k = key[0] if isinstance(key, tuple) else key
                pk = R[k] * (cfv == key[1]) if fixed else R[k]
                if pk.sum() > 0:
                    pick[ii] = rng.choice(len(cells), size=len(ii), p=pk / pk.sum())
            prop, segs = {}, {}
            for L in links:
                r, info = L["r"], L["info"]
                u = useg.get(r["name"], np.ones(L["Kp"]))
                res = np.full(len(rows), None, dtype=object)
                if len(cells) == 1 or not L["coords"]:
                    cv = np.where(pick >= 0, coord(L, cells[0]) if len(cells) == 1 or not L["coords"] else -1, -1)
                    if not L["coords"]:
                        # one level, but a null link when the link's model draws one
                        pnr = info["p_null"][z[rows]]
                        cv = np.where(rng.random(len(rows)) < pnr, L["Wv"].shape[1], 0)
                        cv[pick < 0] = -1
                else:
                    cv = np.full(len(rows), -1)
                    for ci in np.unique(pick[pick >= 0]):
                        cv[pick == ci] = coord(L, cells[ci])
                tt = np.full(len(rows), -1)
                for v in np.unique(cv[cv >= 0]):
                    if v == L["Wv"].shape[1]:
                        continue                                      # a null link
                    for k in np.unique(zc[cv == v]):
                        rr = np.flatnonzero((cv == v) & (zc == k))
                        pt = (L["phi_mix"] if mix else info["phi"][k]) * u * L["Wv"][:, v]
                        if pt.sum() > 0:
                            tt[rr] = rng.choice(L["Kp"], size=len(rr), p=pt / pt.sum())
                ok_t = np.flatnonzero(tt >= 0)
                for (t, v), rr in pd.Series(np.arange(len(ok_t))).groupby([tt[ok_t], cv[ok_t]]).indices.items():
                    self._assign(info, t, ok_t[rr], L["members"][(t, v)], L["w"], L["keys"], res)
                prop[r["child_key"]] = res
                segs[r["name"]] = tt
            return prop, pick, segs
        draw.links = links
        return draw

    def _anchor_from_key(self, r, df, out):
        """The anchor column read from the dimension row each link points to."""
        par = out[r["parent"]]
        pk = self.spec["tables"][r["parent"]]["primary_key"]
        pos = _keystr(par[pk])
        mp = dict(zip(pos, par[r["anchor"]].to_numpy(dtype=object)))
        cur = df[r["anchor"]].to_numpy(dtype=object)
        new = np.array([mp.get(k, c) for k, c in zip(_keystr(df[r["child_key"]]), cur)], dtype=object)
        return new

    def _choose_anchored(self, info, zc, zp, pkeys, cache, r, avals, parent):
        """Link of an anchored dimension (the geography code of a customer whose
        country is modelled), drawn exactly from the link model restricted to the
        dimension rows with the row's anchor value:
            P(p | row) ∝ phi[z, t(p)] * w(p) / m_t(p) * 1[parent anchor = row anchor],
        sampled as a segment t ∝ phi[z, t] * W_t(a) (the valid share of segment t)
        and then a valid member of t ∝ w.  When the row's segment gives no valid
        row any weight, the segment is drawn from the link's overall mixture (the
        source law of the code within the anchor value); the count is logged."""
        rng = self.rng
        phi = info["phi"]
        Kp = phi.shape[1]
        w = self._member_law(info, self._weights(info["parent_frailty"], zp, cache, ("edge", r["name"]),
                                                 ent=info["rel"].e2, obs=info.get("obs")))
        if not (w > 0).any():
            w = np.ones(len(zp))
        m = np.bincount(zp, weights=w, minlength=Kp)
        m = np.where(m > 0, m, 1.0)
        # a null anchor is a value of its own: dimension rows without a state (cities of
        # countries that have none) are drawn for rows whose modelled state is null
        pa = _keystr(parent[r["anchor"]]).fillna(_NULL_KEY)
        ra = _keystr(pd.Series(np.asarray(avals, dtype=object))).fillna(_NULL_KEY)
        res = np.full(len(zc), None, dtype=object)
        isnull = rng.random(len(zc)) < info["p_null"][zc]
        mix = np.bincount(zc, minlength=phi.shape[0]).astype(float) @ phi
        fallback = unmatched = 0
        pa_idx = pd.Series(np.arange(len(pa))).groupby(pa).indices
        for a, rows in pd.Series(np.arange(len(zc))).groupby(ra).indices.items():
            rows = rows[~isnull[rows]]
            valid = pa_idx.get(a)
            if valid is None or not len(rows):
                unmatched += 0 if valid is not None else len(rows)
                continue
            wl = w if w[valid].sum() > 0 else np.ones(len(zp))
            wv = np.bincount(zp[valid], weights=wl[valid], minlength=Kp) / m
            members = [valid[zp[valid] == t] for t in range(Kp)]
            tt = np.full(len(rows), -1)
            for k in np.unique(zc[rows]):
                ii = np.flatnonzero(zc[rows] == k)
                pk = phi[k] * wv
                if pk.sum() <= 0:
                    pk = mix * wv
                    fallback += len(ii)
                if pk.sum() <= 0:
                    pk = (wv > 0).astype(float)
                tt[ii] = rng.choice(Kp, size=len(ii), p=pk / pk.sum())
            for t in np.unique(tt):
                self._assign(info, t, rows[tt == t], members[t], wl, pkeys, res)
        self.sample_log.setdefault("anchored_links", {})[r["name"]] = {"fallback_rows": int(fallback),
                                                                       "rows_without_valid_parent": int(unmatched)}
        return res

    def _choose_parents(self, info, zc, zp, pkeys, cache, name, mask=None):
        rng = self.rng
        phi = info["phi"]
        Kp = phi.shape[1]
        w = self._weights(info["parent_frailty"], zp, cache, ("edge", name), ent=info["rel"].e2, obs=info.get("obs"))
        if mask is None:
            mask = getattr(self, "_edge_masks", {}).get(name)
            mask = mask if mask is not None and len(mask) == len(zp) else None
        if mask is not None and not mask.all():
            w = self._renormalise(w * mask, zp, Kp)
            cache[("edge", name)] = w
        members = [np.flatnonzero(zp == t) for t in range(Kp)]
        if mask is not None and not mask.all():
            members = [m[w[m] > 0] for m in members]
        avail = np.array([len(m) > 0 for m in members])
        res = np.empty(len(zc), dtype=object)
        if not avail.any():
            # no parent row to point at (a parent scaled to no rows): the links are null
            if len(zc):
                self.sample_log["warnings"].append(f"{name}: the parent has no rows; {len(zc)} links left null")
            return res
        isnull = rng.random(len(zc)) < info["p_null"][zc]
        dp = info.get("count_dp")
        # mandatory participation: every parent row needs mn children, so parent
        # segment t needs need_t = mn * M_t rows
        mn = int(dp.get("min", 0)) if dp else (1 if info.get("min_one") else 0)
        need = mn * np.array([len(m) for m in members])
        seg_p = {}
        for k in np.unique(zc):
            pk = phi[k] * avail
            if pk.sum() == 0:
                pk = avail.astype(float)
            seg_p[k] = pk / pk.sum()

        def draw_segments():
            so = np.full(len(zc), -1)
            for k in seg_p:
                rows = np.flatnonzero((zc == k) & ~isnull)
                so[rows] = rng.choice(Kp, size=len(rows), p=seg_p[k])
            return so
        seg_of = draw_segments()
        if mn and need.sum() <= int((~isnull).sum()):
            # the rows' segments conditioned on covering every parent: rejection
            # (exact: a draw is kept only when N_t >= need_t for all t), then, if the
            # constraint is still unmet, the fewest moves that meet it
            for _ in range(self.cfg["rules"].get("max_rejection_rounds", 25)):
                if (np.bincount(seg_of[seg_of >= 0], minlength=Kp) >= need).all():
                    break
                seg_of = draw_segments()
            seg_of = _cover_segments(seg_of, zc, need, phi, rng)
        for t in np.unique(seg_of[seg_of >= 0]):
            rr = np.flatnonzero(seg_of == t)
            m = members[t]
            if dp and len(rr) == 1:
                # one row: the counts are exchangeable, so its parent is uniform over the members
                res[rr] = pkeys[m[rng.integers(len(m))]]
            elif dp:
                obs = dp["counts"][t] if t < len(dp["counts"]) else np.zeros(0, np.int64)
                cnt = self._conditional_counts(len(rr), len(m), obs, dp["a"], mn=dp.get("min", 0),
                                               key=("dp", id(dp), int(t)))
                res[rng.permutation(rr)] = pkeys[np.repeat(m, cnt)]
            elif info.get("min_one"):
                # one child for every parent first (as many as the rows allow), the
                # rest by the usual weights: the shifted count model N = 1 + M
                rr = rng.permutation(rr)
                k1 = min(len(rr), len(m))
                first = rng.permutation(m)[:k1]
                res[rr[:k1]] = pkeys[first]
                if len(rr) > k1:
                    ww = w[m] / w[m].sum()
                    res[rr[k1:]] = pkeys[m[rng.choice(len(m), size=len(rr) - k1, p=ww)]]
            else:
                ww = w[m] / w[m].sum()
                res[rr] = pkeys[m[rng.choice(len(m), size=len(rr), p=ww)]]
        res[isnull] = None
        return res

    # ---- uniqueness and key rules on entities: exact conditioning by MH
    def _constrain_entity(self, e, df, z, zs, keys, out, cache):
        """Condition the foreign keys of entity e on its unique keys and key rules.
        Given segments, the rows' FK draws are independent (product measure Q);
        the target is Q restricted to valid assignments.  An independence proposal
        from each row's own factor Q_i, accepted iff the new state is valid, is a
        Metropolis-Hastings kernel for that target (the acceptance ratio is 1 on
        valid states).  The repair phase reaches a valid state; each later sweep
        leaves the conditional law invariant.  Proposals move one link at a time
        and then all links together (a cycle of MH kernels with the same target).

        Calibration (raking): conditioning Q on rules over two links (the store's
        country is the customer's) shifts the links' margins when Q does not
        already respect the rule.  The target is tilted to Q * w(c) restricted to
        valid states, where c is the row's cell of the categorical columns the
        rules read through the links (customer country x store country), and w is
        updated by iterative proportional fitting, w <- w * P_source(c) / P_syn(c),
        so the synthetic cells match the source's: the I-projection of the
        restricted law onto that margin (Csiszar 1975).  Moves are accepted with
        probability min(1, w(new) / w(old)) on valid states."""
        cons = self.cons.get(e, {})
        ukeys, krules = cons.get("unique", []), cons.get("key_rules", [])
        orels = self._offset_rels(e)
        if (not ukeys and not krules and not orels) or not len(df):
            return df
        t0 = time.time()
        strat = set(self.strata_info.get(e, {}).get("by", []))
        need = {c for u in ukeys for c in u} | {r["offset_from"] for r in orels}
        for r in krules:
            for p in names_in(r["expr"]):
                if "." in p:
                    need.add(p.split(".")[0])
                elif p in {x["child_key"] for x in self.fk[e]}:
                    need.add(p)
        strat |= set(self.grain.get(e, {}).get("cols", []))       # fixed by construction
        strat |= {r["child_key"] for r in self.fk[e] if r.get("anchor")}   # drawn valid (_choose_anchored)
        movable = [r for r in self.fk[e] if r["child_key"] in need and r["child_key"] not in strat]
        if not movable:
            return df
        rng = self.rng
        sweeps = self.cfg["uniqueness"]["mh_sweeps"]
        for r in movable:
            df[r["child_key"]] = df[r["child_key"]].astype(object)
        blocks = [[r] for r in movable] + ([movable] if len(movable) > 1 else [])
        rake = self.rake.get(e) if krules else None
        rake_paths = rake["paths"] if rake else None

        def propose(rows, block):
            return {r["child_key"]: self._choose_parents(self.edge_info[r["name"]], z[rows], zs[r["parent"]],
                                                         keys[r["parent"]], cache, r["name"]) for r in block}

        def rule_ok(sub):
            res = self._offset_valid(e, sub, out) if orels else np.ones(len(sub), bool)
            if krules:
                sub = self._computed(e, sub.copy(), out, strict=False)
            for r in krules:
                res &= check_mask(sub, r["expr"], self._resolver(e, sub, out))
            return res, sub

        def cells_of(sub):
            res_ = self._resolver(e, sub, out)
            parts = [_keystr(pd.Series(np.asarray(res_(p) if "." in p else sub[p], dtype=object)))
                     for p in rake["paths"]]
            return _tuple_frame(parts)

        # the unique-key columns as object arrays (row access without pandas
        # indexing), kept equal to df: try_move writes accepted moves to both
        ucols = sorted({c for u in ukeys for c in u})
        karr = {c: df[c].to_numpy(dtype=object, copy=True) for c in ucols}

        def key_of(i, u):
            return tuple(karr[c][i] for c in u) if len(u) > 1 else karr[u[0]][i]

        valid, sub0 = rule_ok(df)
        cell = cells_of(sub0) if rake else None
        w = {}
        used = [dict() for _ in ukeys]
        # mandatory participation (every parent row has mn children, drawn so by
        # _choose_parents) is one more constraint of the target: a move is valid
        # only when it leaves its old parent with mn children or more
        mand = {}
        for r in movable:
            dp = self.edge_info[r["name"]].get("count_dp")
            mn = int(dp.get("min", 0)) if dp else (1 if self.edge_info[r["name"]].get("min_one") else 0)
            if mn:
                vals = df[r["child_key"]].to_numpy(dtype=object)
                deg = {}
                for v in vals:
                    if not _is_null(v):
                        deg[v] = deg.get(v, 0) + 1
                mand[r["child_key"]] = (mn, deg, vals.copy())
        for i in range(len(df)):
            if not valid[i]:
                continue
            ks = [key_of(i, u) for u in ukeys]
            if any(not _has_null(k) and k in used[ui] for ui, k in enumerate(ks)):
                valid[i] = False
                continue
            for ui, k in enumerate(ks):
                if not _has_null(k):
                    used[ui][k] = i

        def try_move(rows, prop, tilt):
            sub = df.iloc[rows].copy()
            for c, v in prop.items():
                sub[c] = v
            okr, subc = rule_ok(sub)
            cnew = cells_of(subc) if rake else None
            if tilt and w:
                ratio = np.array([w.get(a, 1.0) / w.get(b, 1.0) for a, b in zip(cnew, cell[rows])])
                okr = okr & (rng.random(len(rows)) < ratio)
            if not ukeys and not mand:
                # no unique keys or counts to keep track of: accept all valid moves at once
                acc = np.flatnonzero(okr)
                if len(acc):
                    ri = np.asarray(rows)[acc]
                    for c in prop:
                        col = df[c].to_numpy(dtype=object).copy()
                        col[ri] = sub[c].to_numpy(dtype=object)[acc]
                        df[c] = col
                    valid[ri] = True
                    if rake:
                        cell[ri] = cnew[acc]
                return
            sv = {c: sub[c].to_numpy(dtype=object) for c in set(ucols) | set(prop)}
            acc = []
            for j, i in enumerate(rows):
                if not okr[j]:
                    continue
                newk = [tuple(sv[c][j] for c in u) if len(u) > 1 else sv[u[0]][j] for u in ukeys]
                if any(not _has_null(nk) and used[ui].get(nk, i) != i for ui, nk in enumerate(newk)):
                    continue
                if any(c in mand and not _is_null(mand[c][2][i]) and mand[c][2][i] != sv[c][j]
                       and mand[c][1][mand[c][2][i]] <= mand[c][0] for c in prop):
                    continue
                for c in prop:
                    if c in mand and mand[c][2][i] != sv[c][j]:
                        mn_, deg_, cur_ = mand[c]
                        if not _is_null(cur_[i]):
                            deg_[cur_[i]] -= 1
                        if not _is_null(sv[c][j]):
                            deg_[sv[c][j]] = deg_.get(sv[c][j], 0) + 1
                        cur_[i] = sv[c][j]
                for ui, u in enumerate(ukeys):
                    oldk = key_of(i, u)
                    if used[ui].get(oldk) == i:
                        del used[ui][oldk]
                    if not _has_null(newk[ui]):
                        used[ui][newk[ui]] = i
                for c in ucols:
                    karr[c][i] = sv[c][j]
                acc.append(j)
                valid[i] = True
                if rake:
                    cell[i] = cnew[j]
            if acc:
                acc = np.asarray(acc)
                ri = np.asarray(rows)[acc]
                for c in prop:
                    col = df[c].to_numpy(dtype=object, copy=True)
                    col[ri] = sv[c][acc]
                    try:
                        df[c] = pd.Series(col, index=df.index).astype(df[c].dtype)
                    except (TypeError, ValueError):
                        df[c] = col

        exact = self._cell_sampler(e, df, z, zs, keys, out, cache, movable, rake["paths"] if rake else []) \
            if krules and not ukeys else None
        if exact is not None:
            # exact draws (rows are independent without unique keys), calibrated by
            # iterative proportional fitting of the cell weights w to the source's
            # cells.  The parent-segment weights u_j stay 1: calibrating them to the
            # link model's own segment margins as well (a second set of margins)
            # oscillated on Contoso, where the rules make those margins unattainable
            # together with the cells; their distance is logged instead
            src = rake["dist"] if rake else pd.Series([1.0], index=["*"])
            cells = list(src.index)
            wc, useg = {}, {}
            cell = np.array([None] * len(df), dtype=object)
            segarr = {L["r"]["name"]: np.full(len(df), -1) for L in exact.links}
            tgt = {}
            for L in exact.links:
                pn = L["info"]["p_null"][z]
                tg = ((1 - pn)[:, None] * L["info"]["phi"][z]).sum(axis=0)
                tgt[L["r"]["name"]] = tg / max(tg.sum(), 1e-300)

            mixed = [0]

            def draw_all():
                pending = np.arange(len(df))
                mixed[0] = 0
                for it in range(200):
                    if not len(pending):
                        break
                    if it == 100:
                        mixed[0] += len(pending)
                    prop, pick, segs = exact(pending, cells, wc, useg, mix=it >= 100)
                    sub = df.iloc[pending].copy()
                    for c, v in prop.items():
                        sub[c] = v
                    okr, _ = rule_ok(sub)
                    okr &= pick >= 0
                    acc = np.flatnonzero(okr)
                    ri = pending[acc]
                    for c in prop:
                        col = df[c].to_numpy(dtype=object).copy()
                        col[ri] = sub[c].to_numpy(dtype=object)[acc]
                        df[c] = col
                    for nm, tt in segs.items():
                        segarr[nm][ri] = tt[acc]
                    valid[ri] = True
                    cell[ri] = np.array([cells[k] for k in pick[acc]] + [None], dtype=object)[:-1]
                    pending = pending[~okr]
                return len(pending)

            def seg_tvd():
                out_ = {}
                for nm, tg in tgt.items():
                    sa = segarr[nm][valid & (segarr[nm] >= 0)]
                    sh = np.bincount(sa, minlength=len(tg)) / max(len(sa), 1)
                    out_[nm] = (sh, round(float(0.5 * np.abs(sh - tg).sum()), 4))
                return out_
            tries_left = draw_all()
            seg0 = {nm: v[1] for nm, v in seg_tvd().items()}
            for _ in range(int(self.cfg["uniqueness"].get("rake_rounds", 3)) + 2):
                ok = valid & np.array([c is not None for c in cell])
                syn = pd.Series(cell[ok]).value_counts(normalize=True) if ok.any() else pd.Series(dtype=float)
                eps = 0.5 / max(ok.sum(), 1)
                for c_ in cells:
                    wc[c_] = float(np.clip(wc.get(c_, 1.0) * (src[c_] + eps) / (syn.get(c_, 0.0) + eps), 1e-4, 1e4))
                tries_left = draw_all()
            sweeps = 0
            rake = None                       # calibrated already; the MH sweeps are not needed
            syn = pd.Series(cell[valid & np.array([c is not None for c in cell])]).value_counts(normalize=True)
            ix = syn.index.union(src.index)
            exact_log = {"method": "exact draws per row", "paths": rake_paths or [], "cells": len(src),
                         "rows_redrawn_by_mh": int(tries_left),
                         "rows_from_the_links_overall_mixture": int(mixed[0]),
                         "tvd_after": round(float(0.5 * (syn.reindex(ix, fill_value=0)
                                                          - src.reindex(ix, fill_value=0)).abs().sum()), 4),
                         "parent_segments_tvd_before": seg0,
                         "parent_segments_tvd_after": {nm: v[1] for nm, v in seg_tvd().items()}}
        else:
            exact_log = None

        rounds = 0
        while (~valid).any() and rounds < 200:
            rows = rng.permutation(np.flatnonzero(~valid))
            try_move(rows, propose(rows, blocks[rounds % len(blocks)]), False)
            rounds += 1

        def sweep(n):
            for _ in range(n):
                for b in blocks:
                    rows = rng.permutation(np.flatnonzero(valid))
                    if len(rows):
                        try_move(rows, propose(rows, b), True)
        sweep(sweeps)
        rlog = None
        if rake:
            src = rake["dist"]
            for _it in range(int(self.cfg["uniqueness"].get("rake_rounds", 3))):
                ok = valid
                syn = pd.Series(cell[ok]).value_counts(normalize=True) if ok.any() else pd.Series(dtype=float)
                eps = 0.5 / max(ok.sum(), 1)
                for c_ in set(src.index) | set(syn.index):
                    ratio = (src.get(c_, 0.0) + eps) / (syn.get(c_, 0.0) + eps)
                    w[c_] = float(np.clip(w.get(c_, 1.0) * ratio, 1e-3, 1e3))
                sweep(max(1, sweeps // 2))
            syn = pd.Series(cell[valid]).value_counts(normalize=True)
            ix = syn.index.union(src.index)
            rlog = {"paths": rake["paths"], "cells": len(src),
                    "tvd_after": round(float(0.5 * (syn.reindex(ix, fill_value=0)
                                                    - src.reindex(ix, fill_value=0)).abs().sum()), 4)}
        left = int((~valid).sum())
        self.sample_log["uniqueness"][e] = {"repair_rounds": rounds, "unresolved_rows": left, "sweeps": sweeps,
                                            "seconds": round(time.time() - t0, 1)}
        if exact_log:
            rlog = exact_log
        if rlog:
            self.sample_log["uniqueness"][e]["calibration"] = rlog
        if left:
            self.sample_log["warnings"].append(f"{e}: {left} rows could not satisfy its unique keys / key rules "
                                               "(not enough parent combinations at this scale)")
        return df

    def _resolver(self, t, df, out):
        """Values of 'fk.col' paths for the rows of df (table t) from the generated tables."""
        return make_resolver(self.spec, t, df, out, getattr(self, "_ixcache", None))

    def _grow_tree(self, info, s):
        """A forest with the source's level structure at scale s: round(s * n_kd)
        nodes of segment k at depth d (the sufficient statistic of a multi-type
        Galton-Watson process typed by segment and level), each node of depth d
        attached to a parent of depth d-1.  Given these counts the branching
        process allocates children to parents exactly as a multinomial with
        weights w_i lam_{k,d-1} psi_{k,d-1}(k') (independent Poisson counts
        conditioned on their totals; w_i the node's frailty).  The size, the depth
        profile and the composition then match the source, which an unconditioned
        branching process does not do: one draw near the root adds or removes a
        whole branch (a department under one manager), and a segment mixing
        managers and staff grows chains far deeper than the source."""
        rng, rel = self.rng, info["rel"]
        ent = rel.e1
        K, D = ent.K, info["max_depth"]
        lam_d, psi_d, alpha = info["lam_d"], info["psi_d"], info["frailty"]
        maxd = self.cfg["trees"]["max_depth"]
        cnt = np.zeros((D + 1, K))
        np.add.at(cnt, (rel.depth, ent.z), 1.0)
        target = np.round(s * cnt).astype(np.int64)
        if target[0].sum() == 0:
            target[0, int(np.argmax(cnt[0]))] = 1
        z = np.repeat(np.arange(K), target[0])
        par = np.full(len(z), -1, dtype=np.int64)
        prev = np.arange(len(z))
        for d in range(1, D + 1):
            if not len(prev) or not target[d].sum() or (maxd != "auto" and d > int(maxd)):
                break                   # a user's max_depth truncates the hierarchy
            zp = z[prev]
            w = np.ones(len(prev))
            fin = np.isfinite(alpha[zp])
            if fin.any():
                w[fin] = rng.gamma(alpha[zp[fin]], 1 / alpha[zp[fin]])
            base = w * lam_d[zp, d - 1]
            new_z, new_p = [], []
            for k2 in np.flatnonzero(target[d]):
                pr = base * psi_d[zp, d - 1, k2]
                if not pr.sum() > 0:
                    pr = base if base.sum() > 0 else np.ones(len(prev))
                new_p.append(prev[rng.choice(len(prev), size=target[d, k2], p=pr / pr.sum())])
                new_z.append(np.full(target[d, k2], k2))
            nz, npar = np.concatenate(new_z), np.concatenate(new_p)
            start = len(z)
            z, par = np.concatenate([z, nz]), np.concatenate([par, npar])
            prev = np.arange(start, len(z))
        return z.astype(np.int64), par

    def _tree_levels(self, rel, psi, lam, D, c=1.0):
        """Branching by (segment, depth): a multi-type Galton-Watson process whose
        type is the node's segment and its level.  Hierarchies are level-structured
        (an organisation chart: executives, managers, staff), and a segment that
        mixes levels would otherwise give leaves children and grow chains far deeper
        than the source.  The rate of segment k at depth d has the prior
        Gamma(c lam_k, c) centred on the segment's rate (c pseudo-nodes) and the
        children of the source nodes at that level as data; child segments likewise
        Dirichlet(c psi_k + counts).  No node is generated below the deepest source
        level (outside the source's support).  The forest is generated conditioned
        on the level x segment counts (_grow_tree); these rates are the weights of
        the parent of each node."""
        rng, e = self.rng, rel.e1
        K, dep, z = e.K, rel.depth, e.z
        nd = np.zeros((K, D + 1))
        Nd = np.zeros((K, D + 1))
        np.add.at(nd, (z, dep), 1.0)
        np.add.at(Nd, (z, dep), rel.nchild)
        nonroot = rel.par >= 0
        pa = rel.par[nonroot]
        B = np.zeros((K, D + 1, K))
        np.add.at(B, (z[pa], dep[pa], z[nonroot]), 1.0)
        lam_d = rng.gamma(c * lam[:, None] + Nd, 1.0 / (c + nd))
        base = psi / psi.sum(1, keepdims=True)
        psi_d = np.stack([[rng.dirichlet(c * base[k] + B[k, d] + 1e-9) for d in range(D + 1)] for k in range(K)])
        return lam_d, psi_d / psi_d.sum(2, keepdims=True)

    def _grow_tree_once(self, info, s):
        rng = self.rng
        maxd = self.cfg["trees"]["max_depth"]
        if maxd == "auto":
            # the deepest source level: a deeper node is outside the source's support,
            # and with the size conditioned (_grow_tree) a truncated branch makes
            # the tree wider, not smaller.  A segment that mixes managers and staff
            # would otherwise grow chains of 'managers' far below the source.
            maxd = int(info["max_depth"])
        psi, lam, alpha = info["psi"], info["lam"], info["frailty"]
        lam_d, psi_d = info.get("lam_d"), info.get("psi_d")
        K = len(lam)
        nroot = max(1, int(round(s * info["roots"])))
        z = list(rng.choice(K, size=nroot, p=info["pi_root"]))
        par = [-1] * nroot
        frontier = list(range(nroot))
        depth = 0
        while frontier and depth < maxd:
            nxt = []
            for u in frontier:
                k = z[u]
                w = rng.gamma(alpha[k], 1 / alpha[k]) if np.isfinite(alpha[k]) else 1.0
                if lam_d is not None:
                    if depth >= lam_d.shape[1] - 1:
                        continue                  # deepest source level: no children
                    lk, pk = lam_d[k, depth], psi_d[k, depth]
                else:
                    lk, pk = lam[k], psi[k]
                nc = rng.poisson(lk * w)
                if nc:
                    cz = rng.choice(K, size=nc, p=pk / pk.sum())
                    for c in cz:
                        z.append(int(c))
                        par.append(u)
                        nxt.append(len(z) - 1)
            frontier = nxt
            depth += 1
        return np.array(z, dtype=np.int64), np.array(par)

    # ---- facts
    def _anchor_state(self, f, zs, keys, fact_rows):
        """Segments and output key columns of the anchor rows of fact f."""
        info = self.fact_info[f]
        if not info["nested"]:
            a = info["anchor"]
            return zs[a], {info["fks"][0]["child_key"]: keys[a]}
        st = fact_rows[info["anchor"]]
        nest = info["nest"]
        cols = {ck: st["keycols"][pk] for ck, pk in zip(nest["child_key"], nest["parent_key"]) if "." not in ck}
        return st["z"], cols

    def _dim_state(self, f, r, zs, keys):
        if r.get("virtual"):
            lv = self.fact_info[f]["vdim_levels"]
            return np.arange(len(lv)), np.array(lv, dtype=object)
        return zs[r["parent"]], keys[r["parent"]]

    def _sample_fact(self, f, zs, keys, sc, ents, fact_rows):
        info = self.fact_info[f]
        rel = info["rel"]
        z1, acols = self._anchor_state(f, zs, keys, fact_rows)
        w1 = self._weights(info["frailty1"], z1, {}, None, ent=rel.e1, obs=info.get("obs1"))
        # conditional inclusion on the anchor (quotas only for sales staff): rows at an
        # excluded level take no part; the others take the segment's zero mass away
        elig = None
        if not info["nested"]:
            m1 = self._inclusion_mask(f, info["fks"][0], len(z1), ents)
            if m1.min() < 1:
                elig = m1 > 0
                w1 = self._renormalise(w1 * m1, z1, rel.e1.K)
        st = {"z1": z1, "acols": acols, "w1": w1, "extras": [], "elig": elig}
        dimcols = info["fks"] if info["nested"] else info["fks"][1:]
        if rel.kind == "single":
            if info.get("count_dp"):
                if elig is None:
                    n_rows = self._dp_counts(info["count_dp"], z1, rel.lam)
                else:
                    law, p0 = self._dp_law(info["count_dp"], rel.lam, rel.e1.K)
                    n_rows = self._anchor_thin(f, law(z1), z1, elig, rel.e1.K, lambda rows: law(z1[rows]), p0)
            else:
                n_rows = 1 + self.rng.poisson(np.maximum(rel.lam[z1] - 1, 0) * w1) if info.get("min_one") else \
                    self.rng.poisson(rel.lam[z1] * w1)
            u = np.repeat(np.arange(len(z1)), n_rows)
            v, blk = None, z1[u]
        else:
            r2 = dimcols[0]
            z2, k2 = self._dim_state(f, r2, zs, keys)
            w2 = self._weights(info["frailty2"], z2, {}, None, ent=rel.e2, obs=info.get("obs2"))
            m2 = self._inclusion_mask(f, r2, len(z2), ents)
            w2 = self._renormalise(w2 * m2, z2, rel.e2.K)
            st.update({"z2": z2, "k2": k2, "r2": r2, "w2": w2})
            if rel.kind == "occ":
                g = None
                if info.get("eligibility"):
                    pdf = ents[r2["parent"]]
                    ppk = self.spec["tables"][r2["parent"]]["primary_key"]
                    pos = pd.Index(_keystr(pdf[ppk]).to_numpy(dtype=object)).get_indexer(
                        _keystr(pd.Series(np.asarray(k2, dtype=object))).to_numpy(dtype=object))
                    g = self._nest_groups(f, pd.DataFrame(fact_rows[info["anchor"]]["keycols"]),
                                          pdf.iloc[np.maximum(pos, 0)].reset_index(drop=True))
                el2 = (m2 > 0) if m2.min() < 1 else None
                u, v, blk = self._sample_occ(
                    rel, z1, z2, g, min_one=info.get("occ_min_one", False), elig=elig, f=f, elig2=el2,
                    rewire=lambda u_, v_, b_, sk: self._link_degrees(f, info, rel, z1, z2, u_, v_, b_, elig, el2,
                                                                     unique=True, skip=sk))
            else:
                u, v, blk = self._sample_poisson_pairs(rel, z1, z2, info, w1, w2, elig=elig, f=f)
                u, v, blk = self._link_degrees(f, info, rel, z1, z2, u, v, blk, elig,
                                               (m2 > 0) if m2.min() < 1 else None)
            for xj, (r, xp) in enumerate(zip(dimcols[1:], info.get("extra_p", []))):
                zx, kx = self._dim_state(f, r, zs, keys)
                Kx = xp["P"].shape[1]
                wx = self._weights(xp["frailty"], zx, {}, None, ent=info["extra"][xj].e2, obs=xp.get("obs"))
                wx = wx * self._inclusion_mask(f, r, len(zx), ents)
                memx = [np.flatnonzero(zx == s_) for s_ in range(Kx)]
                availx = np.array([len(m) > 0 and wx[m].sum() > 0 for m in memx], dtype=float)
                if xp.get("null"):
                    availx[-1] = 1.0                                     # the null level
                st["extras"].append((r, xp, kx, wx, memx, availx))
        df = self._fact_rows(f, st, u, v, blk, ents)
        df = self._refill(f, st, df, ents)
        if info["nested"]:
            df = self._nested_consistency(f, df, ents, fact_rows)
        df = self._fact_unique(f, df, ents)
        self._given = getattr(self, "_given", {})
        self._given[f] = self._given_frame(f, df, ents)
        return df, self._fact_entity_state(f, df, ents)

    def _anchor_thin(self, f, n, z, elig, K, law, p0):
        """Rows per anchor under a conditional inclusion rule on the anchor.  In the
        source the excluded rows of segment k (share e_k) all have no rows, so the
        eligible rows follow the segment's count law with that zero mass removed:
        q(0) = (P0_k - e_k) / (1 - e_k), q(c) = P(c) / (1 - e_k) for c > 0.  Rejection:
        an eligible row drawn with 0 rows keeps it with probability r_k = 1 - e_k/P0_k,
        else is redrawn from the law.  Segment totals are kept in expectation; when
        e_k > P0_k (more excluded rows than zeros) r_k = 0 and the shortfall is logged."""
        n = np.asarray(n).copy()
        n[~elig] = 0
        m = np.bincount(z, minlength=K).astype(float)
        E = np.bincount(z, weights=elig.astype(float), minlength=K)
        e = 1 - E / np.maximum(m, 1)
        p0 = np.asarray(p0, float)
        r = np.clip(1 - e / np.maximum(p0, 1e-300), 0, 1)
        rows = np.flatnonzero(elig & (n == 0))
        rounds = 0
        while len(rows) and rounds < 200:
            rej = rows[self.rng.random(len(rows)) >= r[z[rows]]]
            if not len(rej):
                break
            n[rej] = law(rej)
            rows = rej[n[rej] == 0]
            rounds += 1
        self.sample_log.setdefault("anchor_inclusion", {})[f] = {
            "excluded_rows": int((~elig).sum()), "rounds": rounds,
            "segments_short": int(((e > p0 + 1e-12) & (m > 0)).sum())}
        return n

    def _dp_law(self, dp, lam, K):
        """The DP count law of _dp_counts with its Bayesian-bootstrap weights drawn
        once, as (draw(z), P0): rows are i.i.d. given the weights, so a redraw follows
        the same law as the first draw and P0 is its exact probability of no rows."""
        rng = self.rng
        W = {}
        p0 = np.ones(K)
        for k in range(K):
            obs = dp["counts"][k] if k < len(dp["counts"]) else np.zeros(0, np.int64)
            n, a = len(obs), dp["a"]
            w = self._bb(("dp", id(dp), int(k)), n, rng) if n else None
            W[k] = (obs, w, n / (n + a) if n else 0.0)
            mn = dp.get("min", 0)
            base = (0.0 if mn else float(np.exp(-lam[k]))) if k < len(lam) else float(not mn)
            p0[k] = W[k][2] * (float(w[obs == 0].sum()) if n else 0.0) + (1 - W[k][2]) * base

        def draw(z):
            out = np.zeros(len(z), dtype=np.int64)
            for k in np.unique(z):
                idx = np.flatnonzero(z == k)
                obs, w, s = W[k]
                from_obs = rng.random(len(idx)) < s
                if w is not None:
                    out[idx[from_obs]] = obs[rng.choice(len(obs), size=from_obs.sum(), p=w)]
                mn = dp.get("min", 0)
                out[idx[~from_obs]] = mn + rng.poisson(max(lam[k] - mn, 0.0), size=(~from_obs).sum())
            return out
        return draw, p0

    @staticmethod
    def _renormalise(w, z, K):
        """Rescale row weights to sum to the segment size again (after masking)."""
        tot = np.bincount(z, weights=w, minlength=K)
        cnt = np.bincount(z, minlength=K)
        return np.where(tot[z] > 0, w * cnt[z] / np.maximum(tot[z], 1e-300), 0.0)

    def _given_frame(self, f, df, ents):
        """Parent attributes joined onto the rows (fact.condition_on_parents)."""
        amap = getattr(self, "aux_map", {}).get(f)
        if not amap or not len(df):
            return None
        given = pd.DataFrame({name: ents[p].set_index(self.spec["tables"][p]["primary_key"])[c]
                              .reindex(df[fk].to_numpy()).to_numpy() for name, fk, p, c in amap if fk in df})
        return given if given.shape[1] == len(amap) else None

    def _fact_rows(self, f, st, u, v, blk, ents):
        """Rows of fact f for anchor rows u (and partner rows v, blocks blk): extra
        dimension keys given the block, attributes given the group, the time column
        and the sequence numbers.  Rows are independent given (u, v, blk)."""
        rng = self.rng
        info = self.fact_info[f]
        z1 = st["z1"]
        cols = {}
        if v is None:
            groups = blk
        else:
            cols[st["r2"]["child_key"]] = np.asarray(st["k2"], dtype=object)[v] if len(v) else np.array([], dtype=object)
            fitted = info["fitted_groups"]
            OFF1, OFF2, K2 = info["coarse_of_block"]
            coarse = (OFF1 + blk // K2) if info.get("fallback", "anchor") == "anchor" else (OFF2 + blk % K2)
            groups = np.where(np.isin(blk, list(fitted)), blk, coarse)
            for r, xp, kx, wx, memx, availx in st["extras"]:
                Kx = len(memx)
                res = np.empty(len(u), dtype=object)
                for b in np.unique(blk):
                    rows = np.flatnonzero(blk == b)
                    p = xp["P"][b] * availx
                    p = p / p.sum() if p.sum() > 0 else availx / availx.sum()
                    seg = rng.choice(Kx, size=len(rows), p=p)
                    for s_ in np.unique(seg):
                        rr = rows[seg == s_]
                        m = memx[s_]
                        if len(m) == 0 or wx[m].sum() <= 0:              # null key
                            res[rr] = None
                            continue
                        res[rr] = np.asarray(kx, dtype=object)[m[rng.choice(len(m), size=len(rr), p=wx[m] / wx[m].sum())]]
                cols[r["child_key"]] = res
        for c, vals in st["acols"].items():
            cols[c] = np.asarray(vals, dtype=object)[u] if len(u) else np.array([], dtype=object)
        given = self._given_frame(f, pd.DataFrame(cols), ents) if len(groups) else None
        df = self.tm[f].sample(groups, rng, given) if len(groups) else \
            pd.DataFrame(columns=self.tm[f].out_cols)
        for c, vals in cols.items():
            df[c] = vals
        if "time" in info:
            ti = info["time"]
            P = ti["P"]
            zrow = z1[u]
            b = _choice_rows(rng, P, zrow)
            starts = ti["buckets"].start_time.to_numpy()
            ends = ti["buckets"].end_time.to_numpy()
            span = (ends[b] - starts[b]).astype("timedelta64[s]").astype(np.int64)
            off = (rng.random(len(b)) * span).astype(np.int64)
            ts = starts[b] + off.astype("timedelta64[s]")
            ts = pd.Series(ts)
            if not ti["is_dt"]:
                ts = ts.dt.normalize()
            ts = ts.clip(ti["tmin"], ti["tmax"])
            ts[rng.random(len(ts)) < ti["null_rate"]] = pd.NaT
            df[ti["col"]] = ts.to_numpy()
        df["__group__"] = groups
        df["__seg__"] = z1[u] if len(u) else np.zeros(0, np.int64)
        df["__anchor_row__"] = u
        df["__blk__"] = blk
        self._sequences(f, df)
        return df

    def _sequences(self, t, df):
        """Sequence columns (line numbers): start..start+n-1 within their key (start 1
        unless declared), in row order."""
        for c, cs in self.spec["tables"][t]["columns"].items():
            if cs["type"] == "sequence" and cs.get("within") in df.columns:
                key = _keystr(df[cs["within"]]).fillna("\x00null").to_numpy(dtype=object)
                df[c] = pd.Series(key).groupby(key, sort=False).cumcount().to_numpy() + int(cs.get("start", 1))
        return df

    def _partner_given_anchor(self, rel, st, u):
        """Partner row of each row of anchor rows u under the Poisson block model:
        segment l with probability proportional to eta*_kl W2_l, then a row of l by
        its weight.  This is the law of one row given its anchor row (the Poisson
        process split by anchor), so redrawing a rejected row from it is exact."""
        rng = self.rng
        e2 = rel.e2
        z1, z2, w2 = st["z1"], st["z2"], st["w2"]
        m_real = e2.counts()
        m_syn = np.bincount(z2, minlength=e2.K).astype(float)
        mem2 = [np.flatnonzero(z2 == l) for l in range(e2.K)]
        W2 = np.array([w2[m].sum() for m in mem2])
        ok2 = (W2 > 0) & (m_real > 0)
        rate = rel.eta * np.where(ok2, m_real / np.maximum(m_syn, 1), 0.0)[None, :] * W2[None, :]
        v = np.zeros(len(u), dtype=np.int64)
        blk = np.zeros(len(u), dtype=np.int64)
        for k in np.unique(z1[u]):
            rows = np.flatnonzero(z1[u] == k)
            pk = rate[k]
            if pk.sum() <= 0:
                pk = ok2.astype(float)
            ls = rng.choice(e2.K, size=len(rows), p=pk / pk.sum())
            for l in np.unique(ls):
                rr = rows[ls == l]
                pv = w2[mem2[l]] / W2[l]
                v[rr] = mem2[l][rng.choice(len(pv), size=len(rr), p=pv)]
                blk[rr] = k * e2.K + l
        return v, blk

    def _offset_rels(self, t):
        return [r for r in self.spec["relationships"] if r["child"] == t and r.get("offset_column")]

    def _offset_valid(self, t, df, ents):
        """Rows whose offset keys land on a row of the parent (ship date = order date +
        gap must be a day of the calendar).  A draw that falls outside the parent is
        invalid like a broken rule, and is redrawn rather than left empty."""
        ok = np.ones(len(df), bool)
        for r in self._offset_rels(t):
            if r["offset_from"] not in df.columns or r["offset_column"] not in df.columns:
                continue
            par = ents[r["parent"]]
            pk = self.spec["tables"][r["parent"]]["primary_key"]
            k = offset_keys(df[r["offset_from"]], df[r["offset_column"]], par[pk], par[r["order_by"]], r["unit"])
            given = df[r["offset_from"]].notna().to_numpy() & \
                pd.to_numeric(df[r["offset_column"]], errors="coerce").notna().to_numpy()
            ok &= ~(given & pd.isna(pd.Series(k, dtype=object)).to_numpy())
        return ok

    def _row_validity(self, f, df, ents):
        """Rows that satisfy the fact's check rules and unique keys (the first row
        of a key is kept, later duplicates are invalid), evaluated with keys,
        lookups and derived columns computed.  None when nothing is declared."""
        T = self.spec["tables"][f]
        cons = self.cons.get(f, {})
        rules = cons.get("attr_rules", []) + cons.get("key_rules", [])
        keys = [(u, None) for u in T.get("unique", [])] + [(p["columns"], p["where"]) for p in T.get("partial_unique", [])]
        if (not rules and not keys and not self._offset_rels(f)) or not len(df):
            return None
        w = self._computed(f, df.copy(), ents)
        # the hidden draws the rules are checked on are the ones the rows keep
        for c in _hidden_cols(w):
            df[c] = w[c].to_numpy()
        res = self._resolver(f, w, ents)
        ok = self._offset_valid(f, df, ents)
        for r in rules:
            ok &= check_mask(w, r["expr"], res)
        for cols, where in keys:
            if not all(c in w.columns for c in cols):
                continue
            m = ok.copy() if where is None else ok & check_mask(w, where, res) & ~check_mask(w, f"not ({where})", res)
            sub = w.loc[m, cols].astype(object)
            sub = sub.where(sub.notna(), None)
            dup = sub.duplicated(keep="first").to_numpy()
            ok[np.flatnonzero(m)[dup]] = False
        return ok

    def _refill(self, f, st, df, ents):
        """Exact conditioning on the fact's rules without losing rows.  Given its
        anchor row, a row is drawn independently from Q(row | anchor) (Poisson
        block model: partner, extras, attributes; occupancy: extras and attributes
        given the present pair; single: attributes).  The target is Q restricted to
        valid rows with the count of rows per anchor kept, i.e. each row from
        Q(row | anchor, valid): rejection sampling per row.  Rows still invalid
        after synthesis.rules.max_rejection_rounds are reported (and removed later
        for key rules)."""
        ok = self._row_validity(f, df, ents)
        if ok is None or ok.all():
            return df
        rel = self.fact_info[f]["rel"]
        rounds, first_bad = 0, int((~ok).sum())
        maxr = self.cfg["rules"]["max_rejection_rounds"]
        u_all = df["__anchor_row__"].to_numpy()
        while not ok.all() and rounds < maxr:
            idx = np.flatnonzero(~ok)
            u = u_all[idx]
            if rel.kind == "single":
                v, blk = None, st["z1"][u]
            elif rel.kind == "occ":
                v = self._index_of(df, st, idx)
                blk = df["__blk__"].to_numpy()[idx]
            else:
                v, blk = self._partner_given_anchor(rel, st, u)
            new = self._fact_rows(f, st, u, v, blk, ents)
            new.index = df.index[idx]
            _mark_fresh(df, idx)
            for c in new.columns:
                if c not in df.columns:
                    df[c] = None
                dt = df[c].dtype
                col = df[c].to_numpy(dtype=object, copy=True)
                col[idx] = new[c].to_numpy(dtype=object)
                try:
                    df[c] = pd.Series(col, index=df.index).astype(dt)
                except (TypeError, ValueError):
                    df[c] = col
            ok = self._row_validity(f, df, ents)
            rounds += 1
        self._sequences(f, df)
        for c in ("__group__", "__seg__", "__anchor_row__", "__blk__"):
            df[c] = pd.to_numeric(df[c]).astype(np.int64)
        self.sample_log.setdefault("redrawn_rows", {})[f] = {"invalid_first_draw": first_bad, "rounds": rounds,
                                                            "still_invalid": int((~ok).sum())}
        return df

    def _index_of(self, df, st, idx):
        """Partner row index of rows idx (occupancy facts keep their pair)."""
        k2 = pd.Index(_keystr(pd.Series(np.asarray(st["k2"], dtype=object))).to_numpy(dtype=object))
        return np.asarray(k2.get_indexer(_keystr(df[st["r2"]["child_key"]].iloc[idx]).to_numpy(dtype=object)), np.int64)

    def _fact_unique(self, f, df, ents):
        """Declared unique keys of a fact (full or filtered with 'where').  For a key
        made of the generating pair, the occupancy model draws M_ij = 1 + Poisson
        independently per present pair, so keeping one row of a pair is exactly
        conditioning on M_ij = 1 for the pairs the key covers.  For any other key the
        duplicates are dropped as a repair and counted."""
        T = self.spec["tables"][f]
        keys = [(u, None) for u in T.get("unique", [])] + [(p["columns"], p["where"]) for p in T.get("partial_unique", [])]
        if not keys or not len(df):
            return df
        drop = np.zeros(len(df), bool)
        for cols, where in keys:
            if not all(c in df.columns for c in cols):
                continue
            m = np.ones(len(df), bool) if where is None else \
                (check_mask(df, where, self._resolver(f, df, ents)) &
                 ~check_mask(df, f"not ({where})", self._resolver(f, df, ents)))
            sub = df.loc[m & ~drop, cols].astype(object).where(df.loc[m & ~drop, cols].notna(), None)
            dup = sub.duplicated(keep="first")
            drop[np.flatnonzero(m & ~drop)[dup.to_numpy()]] = True
        if drop.any():
            self.sample_log["rows_removed_by_rules"][f"{f} (unique keys)"] = int(drop.sum())
            df = df[~drop].reset_index(drop=True)
        return df

    def _nested_consistency(self, f, df, ents, fact_rows):
        """Composite-key components that are lookups through another dimension
        (e.g. the cell's template) must agree with the parent row.  Pairs are
        independent given the parameters, so dropping the disagreeing ones is
        exact conditioning on the key constraint."""
        nest = self.fact_info[f]["nest"]
        st = fact_rows[nest["parent"]]
        keep = np.ones(len(df), bool)
        u = df["__anchor_row__"].to_numpy()
        for ck, pk in zip(nest["child_key"], nest["parent_key"]):
            if "." not in ck:
                continue
            vals = _keystr(pd.Series(self._resolver(f, df, ents)(ck).to_numpy(dtype=object))).to_numpy(dtype=object)
            want = _keystr(pd.Series(np.asarray(st["keycols"][pk], dtype=object)[u])).to_numpy(dtype=object)
            keep &= vals == want
        self.sample_log["rows_removed_by_rules"][f"{f}: composite key consistency"] = int((~keep).sum())
        return df[keep].reset_index(drop=True)

    def _fact_entity_state(self, f, df, ents):
        """If f is the parent of a nested fact, the segment and key columns of its rows."""
        if f not in self.fact_entity:
            return None
        bm = self.fact_block_map[f]
        z = np.array([bm.get(int(b), 0) for b in df["__blk__"].to_numpy()], dtype=np.int64)
        ref = getattr(self.fact_entity[f], "refined", None)
        if ref is not None and len(z):
            # refined segment of each new parent row, drawn within its block
            bos, w = ref["block_of_seg"], ref["w"]
            zr = np.zeros(len(z), dtype=np.int64)
            for b in np.unique(z):
                segs = np.flatnonzero(bos == b)
                rows = np.flatnonzero(z == b)
                if not len(segs):
                    continue
                pw = w[segs] / w[segs].sum()
                zr[rows] = segs[self.rng.choice(len(segs), size=len(rows), p=pw)]
            z = zr
        res = self._resolver(f, df, ents)
        keycols = {c: (res(c).to_numpy(dtype=object) if "." in c else df[c].to_numpy(dtype=object))
                   for c in self.fact_key_cols[f]}
        return {"z": z, "keycols": keycols}

    def _sample_poisson_pairs(self, rel, z1, z2, info, w1, w2, elig=None, f=None):
        rng = self.rng
        e1, e2 = rel.e1, rel.e2
        m_real = e2.counts()
        m_syn = np.bincount(z2, minlength=e2.K).astype(float)
        mem1 = [np.flatnonzero(z1 == k) for k in range(e1.K)]
        mem2 = [np.flatnonzero(z2 == l) for l in range(e2.K)]
        us, vs, bl = [], [], []
        if info.get("count_dp") or info.get("min_one"):
            # rows per anchor from the DP posterior (or 1 + Poisson when every anchor
            # has rows), then each row's partner segment
            # with probability proportional to its Poisson rate eta*_kl W2_l (given
            # the total, independent Poisson counts split multinomially)
            W2 = np.array([w2[m].sum() for m in mem2])
            ok2 = (W2 > 0) & (m_real > 0)
            rate = rel.eta * np.where(ok2, m_real / np.maximum(m_syn, 1), 0.0)[None, :] * W2[None, :]
            lam1 = (rel.eta * m_real[None, :]).sum(1)
            if elig is not None and info.get("count_dp"):
                law, p0 = self._dp_law(info["count_dp"], lam1, e1.K)
                n_u = self._anchor_thin(f, law(z1), z1, elig, e1.K, lambda rows: law(z1[rows]), p0)
            else:
                n_u = self._dp_counts(info["count_dp"], z1, lam1) if info.get("count_dp") else \
                    1 + rng.poisson(np.maximum(lam1[z1] - 1, 0))
            u = np.repeat(np.arange(len(z1)), n_u)
            v = np.zeros(len(u), dtype=np.int64)
            blk = np.zeros(len(u), dtype=np.int64)
            for k in np.unique(z1[u]):
                rows = np.flatnonzero(z1[u] == k)
                pk = rate[k]
                if pk.sum() <= 0:
                    pk = ok2.astype(float)
                ls = rng.choice(e2.K, size=len(rows), p=pk / pk.sum())
                for l in np.unique(ls):
                    rr = rows[ls == l]
                    pv = w2[mem2[l]] / W2[l]
                    v[rr] = mem2[l][rng.choice(len(pv), size=len(rr), p=pv)]
                    blk[rr] = k * e2.K + l
            perm = rng.permutation(len(u))
            return u[perm], v[perm], blk[perm]
        for k in range(e1.K):
            if not len(mem1[k]):
                continue
            W1 = w1[mem1[k]].sum()
            for l in range(e2.K):
                if not len(mem2[l]) or m_real[l] == 0:
                    continue
                eta_star = rel.eta[k, l] * m_real[l] / m_syn[l]      # scale rule, anchor = e1
                W2 = w2[mem2[l]].sum()
                N = rng.poisson(eta_star * W1 * W2)
                if not N:
                    continue
                pu = w1[mem1[k]] / W1
                pv = w2[mem2[l]] / w2[mem2[l]].sum()
                us.append(mem1[k][rng.choice(len(pu), size=N, p=pu)])
                vs.append(mem2[l][rng.choice(len(pv), size=N, p=pv)])
                bl.append(np.full(N, k * e2.K + l))
        u = np.concatenate(us) if us else np.zeros(0, dtype=np.int64)
        v = np.concatenate(vs) if vs else np.zeros(0, dtype=np.int64)
        blk = np.concatenate(bl) if bl else np.zeros(0, dtype=np.int64)
        perm = rng.permutation(len(u))
        return u[perm], v[perm], blk[perm]

    def _sample_occ(self, rel, z1, z2, groups=None, min_one=False, elig=None, f=None, elig2=None, rewire=None):
        """Draw the occupancy model: active anchors, present pairs, multiplicities.
        Structural blocks (p in {0, 1}) are kept as they are; other blocks follow
        the scale rule p* = min(1, p m_l / m*_l), so rows per anchor are preserved
        when the partner is resized.  With eligibility groups (g1, g2) only pairs
        of the same group are drawn (structural zeros, see OccRel).  With min_one
        (mandatory participation) the pairs of an active anchor are drawn
        conditioned on at least one being present: anchors left empty are redrawn
        (rejection); those still empty after 20 rounds are drawn by the exact
        sequential method (first_present).  Both are exact for the zero-truncated law.
        With a conditional inclusion rule on the partner (elig2) the excluded partner
        rows of block l had no pairs in the source, so the E_l eligible rows take the
        block's pairs: p* = min(1, p m_l / E*_l) on eligible rows only (Prop. anchorincl,
        partner side; the scale rule with E*_l in place of m*_l)."""
        rng = self.rng
        e1, e2 = rel.e1, rel.e2
        m_real = e2.counts()
        mem1 = [np.flatnonzero(z1 == k) for k in range(e1.K)]
        mem2 = [np.flatnonzero(z2 == l) for l in range(e2.K)]
        if elig2 is not None:
            mem2 = [J[elig2[J]] for J in mem2]
            self.sample_log.setdefault("partner_inclusion", {})[f] = {"excluded_rows": int((~elig2).sum())}
        m_syn = np.array([len(J) for J in mem2], dtype=float)
        ra = getattr(rel, "row_active", None)
        # rows of a copied anchor take part exactly when they did in the source
        if ra is not None and len(ra) == len(z1):
            active = ra.copy()
        elif elig is not None:
            # a conditional inclusion rule on the anchor: excluded rows are inactive and
            # the eligible rows of segment k active with pi_k m_k / E_k (the zero-mass
            # removal of _anchor_thin for a Bernoulli law), capped at 1
            m = np.bincount(z1, minlength=e1.K).astype(float)
            E = np.bincount(z1, weights=elig.astype(float), minlength=e1.K)
            pe = np.minimum(1.0, rel.pi * m / np.maximum(E, 1e-300))
            active = elig & (rng.random(len(z1)) < pe[z1])
            self.sample_log.setdefault("anchor_inclusion", {})[f] = {
                "excluded_rows": int((~elig).sum()),
                "segments_short": int(((rel.pi * m > E + 1e-9) & (m > 0)).sum())}
        else:
            active = rng.random(len(z1)) < rel.pi[z1]
        us, vs, bl = [], [], []

        def split(idx, g):
            if groups is None:
                return {None: idx}
            gi = groups[g][idx]
            o = np.argsort(gi, kind="stable")
            val, start = np.unique(gi[o], return_index=True)
            return dict(zip(val.tolist(), np.split(idx[o], start[1:])))
        parts2 = [split(J, 1) for J in mem2]

        def draw(sel):
            for k in range(e1.K):
                A = mem1[k][sel[mem1[k]]]
                if not len(A):
                    continue
                parts1 = split(A, 0)
                for l in range(e2.K):
                    p = rel.p[k, l]
                    if not len(mem2[l]) or p <= 0 or m_real[l] == 0:
                        continue
                    if p < 1:
                        p = min(1.0, p * m_real[l] / m_syn[l])
                    for g, Ag in parts1.items():
                        J = parts2[l].get(g)
                        if J is None or not len(J):
                            continue
                        if p >= 1:
                            ii, jj = np.repeat(Ag, len(J)), np.tile(J, len(Ag))
                        else:
                            ii, jj = _bernoulli_grid(Ag, J, p, rng)
                        if not len(ii):
                            continue
                        us.append(ii)
                        vs.append(jj)
                        bl.append(np.full(len(ii), k * e2.K + l))
        def first_present(rows):
            """Exact draw of an anchor's pairs given at least one, for anchors that
            rejection left empty (rates near zero).  Blocks are taken in a fixed
            order: the first non-empty block l* has probability
            (1 - P0_l*) prod_{l < l*} P0_l / (1 - P0); its count is binomial
            truncated at zero, later blocks are unconditioned binomials, and the
            partners are chosen uniformly within each block."""
            from scipy.stats import binom
            g1 = groups[0] if groups is not None else None
            for i in rows:
                k = z1[i]
                blocks = []
                for l in range(e2.K):
                    p = rel.p[k, l]
                    if not len(mem2[l]) or p <= 0 or m_real[l] == 0:
                        continue
                    if p < 1:
                        p = min(1.0, p * m_real[l] / m_syn[l])
                    J = parts2[l].get(g1[i] if g1 is not None else None)
                    if J is None or not len(J):
                        continue
                    blocks.append((l, J, p))
                if not blocks:
                    continue
                lp0 = np.array([len(J) * np.log1p(-min(p, 1 - 1e-15)) if p < 1 else -np.inf for _, J, p in blocks])
                w = -np.expm1(lp0) * np.exp(np.concatenate([[0.0], np.cumsum(lp0)[:-1]]))
                first = rng.choice(len(blocks), p=w / w.sum())
                for b, (l, J, p) in enumerate(blocks):
                    if b < first:
                        continue
                    m = len(J)
                    if b == first and p < 1:
                        pm = binom.pmf(np.arange(1, m + 1), m, p)
                        x = 1 + rng.choice(m, p=pm / pm.sum())
                    else:
                        x = m if p >= 1 else rng.binomial(m, p)
                    if x:
                        jj = rng.choice(J, size=x, replace=False)
                        us.append(np.full(x, i))
                        vs.append(jj)
                        bl.append(np.full(x, k * e2.K + l))

        draw(active)
        if min_one:
            for r in range(21):
                has = np.zeros(len(z1), bool)
                for ii in us:
                    has[ii] = True
                need = active & ~has
                if not need.any():
                    break
                if r < 20:
                    draw(need)
                else:
                    first_present(np.flatnonzero(need))
        u = np.concatenate(us) if us else np.zeros(0, dtype=np.int64)
        v = np.concatenate(vs) if vs else np.zeros(0, dtype=np.int64)
        blk = np.concatenate(bl) if bl else np.zeros(0, dtype=np.int64)
        if rewire is not None and len(u):
            # pairs of complete blocks (p = 1, e.g. every product on every date) are
            # structural: they and the segments they touch keep the occupancy draw
            free = rel.p.reshape(-1)[blk] < 1
            if free.all():
                u, v, blk = rewire(u, v, blk, None)
            elif free.any():
                skip = (set((blk[~free] // e2.K).tolist()), set((blk[~free] % e2.K).tolist()))
                uf, vf, bf = rewire(u[free], v[free], blk[free], skip)
                u, v, blk = np.concatenate([u[~free], uf]), np.concatenate([v[~free], vf]), \
                    np.concatenate([blk[~free], bf])
        mb = rel.mult_by
        if mb != "none" and len(u):
            if mb == "partner":
                mu = rel.mu[blk % e2.K]
            elif mb == "anchor":
                mu = rel.mu[blk // e2.K]
            else:
                mu = rel.mu.reshape(-1)[blk]
            rep = 1 + rng.poisson(mu)
            u, v, blk = np.repeat(u, rep), np.repeat(v, rep), np.repeat(blk, rep)
        perm = rng.permutation(len(u))
        return u[perm], v[perm], blk[perm]

    def _dp_counts(self, dp, z, lam):
        """Rows per anchor from the DP posterior predictive of each segment's count
        distribution: with probability n/(n+a) a Bayesian-bootstrap draw from the
        observed counts, otherwise a draw from the Poisson base measure."""
        rng = self.rng
        out = np.zeros(len(z), dtype=np.int64)
        for k in np.unique(z):
            idx = np.flatnonzero(z == k)
            obs = dp["counts"][k] if k < len(dp["counts"]) else np.zeros(0, np.int64)
            n, a = len(obs), dp["a"]
            from_obs = rng.random(len(idx)) < n / (n + a) if n else np.zeros(len(idx), bool)
            if n:
                w = self._bb(("dp", id(dp), int(k)), n, rng)
                out[idx[from_obs]] = obs[rng.choice(n, size=from_obs.sum(), p=w)]
            mn = dp.get("min", 0)
            out[idx[~from_obs]] = mn + rng.poisson(max(lam[k] - mn, 0.0), size=(~from_obs).sum())
        return out

    # ---- finishing: lookups, rules, text, pii
    def _finish(self, t, df, z, out):
        rng = self.rng
        T = self.spec["tables"][t]
        n = len(df)
        groups = df.pop("__group__").to_numpy() if "__group__" in df else (z if z is not None else np.zeros(n, int))
        seg = df.pop("__seg__").to_numpy() if "__seg__" in df else (z if z is not None else np.zeros(n, int))
        for c in ("__anchor_row__", "__blk__"):
            if c in df:
                df.pop(c)
        if t in self.copied:
            self.sample_log["rule_violations"][t] = {}
            return df[[c for c in T["columns"]]]
        cons = self.cons.get(t, {})
        tm = self.tm[t]
        df = self._computed(t, df, out)
        viol = {}
        arules = cons.get("attr_rules", [])
        # unique keys of an entity made only of modelled attribute columns (a product
        # code with its version date): later duplicates are redrawn from their own
        # conditional like rule violations (keys over links are handled in
        # _constrain_entity, keys of facts in _refill)
        akeys = [u for u in T.get("unique", []) if T["role"] != "fact" and u and all(c in tm.out_cols for c in u)]
        # keys over links and attributes (kept by the link moves): attribute redraws keep them too
        akeys += [u for u in cons.get("unique", []) if T["role"] != "fact" and not set(u) <= {r["child_key"] for r in self.fk[t]}]
        # an entity's rows whose attributes are redrawn must keep the link rules that read
        # those attributes (the customer active on the order date): rejection on all of them
        xrules = [r_ for r_ in cons.get("key_rules", []) if T["role"] != "fact"
                  and names_in(r_["expr"]) & set(tm.out_cols)] if arules or akeys else []

        def attr_bad(frame):
            ok = np.ones(len(frame), bool)
            res_ = self._resolver(t, frame, out)
            for r_ in arules + xrules:
                ok &= check_mask(frame, r_["expr"], res_)
            for u in akeys:
                sub = frame[u].astype(object)
                sub = sub.where(sub.notna(), None)
                ok &= ~(sub.duplicated(keep="first").to_numpy() & sub.notna().all(axis=1).to_numpy())
            return ~ok
        if arules or akeys:
            # rejection on the attributes jointly for all attribute rules (each redraw
            # is from the row's own conditional, so the accepted law is Q(x | rules))
            bad = attr_bad(df)
            rounds = 0
            redrawn = np.zeros(len(df), bool)
            while bad.any() and rounds < self.cfg["rules"]["max_rejection_rounds"] and tm.out_cols:
                idx = np.flatnonzero(bad)
                redrawn[idx] = True
                gv = getattr(self, "_given", {}).get(t)
                new = tm.sample(groups[idx], rng, gv.iloc[idx] if gv is not None else None)
                for c in tm.out_cols:
                    col = df[c].astype(object).to_numpy(copy=True)
                    col[idx] = new[c].to_numpy(dtype=object)
                    try:
                        df[c] = pd.Series(col, index=df.index).astype(df[c].dtype)
                    except (TypeError, ValueError):
                        df[c] = col
                _mark_fresh(df, idx)
                df = self._computed(t, df, out)
                bad = attr_bad(df)
                rounds += 1
            anch = [r for r in self.fk[t] if r.get("anchor") and r["anchor"] in df.columns]
            if redrawn.any() and anch and z is not None:
                # rows whose attributes were redrawn: their anchored links (the area of
                # the new country) are drawn again from the new anchor values
                idx = np.flatnonzero(redrawn)
                strat_by = set(self.strata_info.get(t, {}).get("by", []))
                for r in anch:
                    p_ = r["parent"]
                    if r["child_key"] in strat_by:
                        df[r["anchor"]] = self._anchor_from_key(r, df, out)
                        continue
                    col = df[r["child_key"]].astype(object).to_numpy(copy=True)
                    col[idx] = self._choose_anchored(self.edge_info[r["name"]], np.asarray(z)[idx], self._zs[p_],
                                                     out[p_][self.spec["tables"][p_]["primary_key"]].to_numpy(),
                                                     self._wcache, r, df[r["anchor"]].to_numpy(dtype=object)[idx],
                                                     out[p_])
                    df[r["child_key"]] = col
                df = self._computed(t, df, out)
            res_ = self._resolver(t, df, out)
            for r in arules:
                viol[r["expr"]] = int((~check_mask(df, r["expr"], res_)).sum())
        if T["role"] == "fact" and cons.get("key_rules") and t not in self.fact_entity:
            # rows still invalid after the per-row redraws of _refill: removed and reported
            keep = np.ones(len(df), bool)
            for r in cons["key_rules"]:
                keep &= check_mask(df, r["expr"], self._resolver(t, df, out))
            if (~keep).any():
                self.sample_log["rows_removed_by_rules"][t] = int((~keep).sum())
                df = df[keep].reset_index(drop=True)
                seg = seg[keep]
                n = len(df)
                self._sequences(t, df)
        for r in cons.get("key_rules", []) + cons.get("anchor_rules", []):
            viol[r["expr"]] = int((~check_mask(df, r["expr"], self._resolver(t, df, out))).sum())
        self.sample_log["rule_violations"][t] = viol
        for c, cs in T["columns"].items():
            if c in df.columns:
                continue
            if cs["type"] == "id" and (t, c) in self.pii:
                df[c] = _unique_codes(self.pii[(t, c)], n, rng)
            elif cs["type"] == "id":
                df[c] = np.arange(1, n + 1)
            elif cs["type"] == "text":
                tmod = self.text[(t, c)]
                o = cs.get("text") or {}
                classes = df[o["condition_on"]].astype(str).tolist() if o.get("condition_on") else [str(s) for s in seg]
                df[c] = tmod.generate(classes, rng)
                self.sample_log["text_rejections"][f"{t}.{c}"] = tmod.last_rejected
            elif cs["type"] == "pii":
                df[c] = self.pii[(t, c)].generate(n, rng)
            elif cs["type"] == "constant":
                df[c] = self.pii[(t, c)][1]
            elif cs["type"] == "binary":
                df[c] = None          # images and files are not synthesised
            else:
                df[c] = None          # filled by an aggregate rule
        # hidden draws stay with the frame until sample() ends (an SCD recompute reuses them)
        return df[[c for c in T["columns"]] + _hidden_cols(df)]

    def _fit_soft(self, tables):
        """Soft rules y = f(x) that hold on most rows: a Bayesian mixture

            z ~ Bernoulli(lambda),  lambda ~ Beta(1 + holds, 1 + violations),
            y = f(x) if z = 1;  otherwise f(x) + d with d from a Dirichlet(counts + 1/2)
            posterior over the observed offsets (violation: offset), or the modelled
            value (violation: model).

        lambda and the offset weights are drawn once per dataset."""
        self.soft = {}
        for _i, r in enumerate(self.spec["rules"]):
            if r["type"] != "soft" or r["table"] not in tables:
                continue
            t = r["table"]
            df = tables[t]
            lhs, _ = split_assignment(r["expr"])
            ctype = self.spec["tables"][t]["columns"].get(lhs, {}).get("type")
            try:
                _, fx = derive_values(df, r["expr"], self._resolver(t, df, tables), ctype)
            except (KeyError, SpecError):
                continue
            y = pd.to_numeric(df[lhs], errors="coerce").to_numpy(dtype=float) if ctype in ("integer", "float") else None
            if y is not None:
                fx = pd.to_numeric(pd.Series(np.asarray(fx, dtype=object)), errors="coerce").to_numpy(dtype=float)
                ok = ~np.isnan(y) & ~np.isnan(fx)
                hold = ok & (np.abs(y - fx) <= (r.get("tolerance") or 1e-9))
                off = (y - fx)[ok & ~hold]
            else:
                ok = df[lhs].notna().to_numpy()
                hold = ok & (df[lhs].astype(str).to_numpy() == pd.Series(fx).astype(str).to_numpy())
                off = np.zeros(0)
            h, v = int(hold.sum()), int((ok & ~hold).sum())
            vals, cnt = np.unique(off, return_counts=True) if len(off) else (np.zeros(0), np.zeros(0))
            viol = r.get("violation", "offset")
            if viol == "offset" and (y is None or len(vals) == 0 or len(vals) > 50):
                viol = "model"          # free text or diffuse offsets: the modelled value
            st = {"lhs": lhs, "ctype": ctype, "holds": h, "violations": v, "violation": viol,
                  "offsets": vals, "counts": cnt.astype(float), "rule": r}
            if h + v == 0:          # column not in the source: the declared share, fixed
                st["holds"], st["violations"] = float(r["holds"]) * 1e6, (1 - float(r["holds"])) * 1e6

            def draw(rng, st=st):
                st["lam"] = float(rng.beta(1 + st["holds"], 1 + st["violations"]))
                st["w"] = rng.dirichlet(st["counts"] + 0.5) if len(st["counts"]) else None
            self._register(draw)
            self.soft.setdefault(t, []).append(st)
            self.log["decisions"].setdefault(t, {})[f"soft rule {r['expr']}"] = (
                f"holds on {h} of {h + v} source rows; violations: {viol}"
                + (f" ({len(vals)} offsets)" if viol == "offset" else ""))

    def _apply_soft(self, t, df, out):
        """Soft rules of t on a finished frame.  Each row's indicator z (and offset)
        is drawn once and kept in a hidden column, so recomputing after attribute
        redraws re-applies the same choice."""
        rng = self.rng
        for st in getattr(self, "soft", {}).get(t, []):
            lhs = st["lhs"]
            try:
                _, fx = derive_values(df, st["rule"]["expr"], self._resolver(t, df, out), st["ctype"])
            except KeyError:
                continue
            zc, dc = f"__soft_z_{lhs}", f"__soft_d_{lhs}"
            rows = _fresh_rows(df, zc)
            if rows.any():
                n = int(rows.sum())
                zv = df[zc].to_numpy(dtype=bool, copy=True) if zc in df.columns else np.zeros(len(df), bool)
                zv[rows] = rng.random(n) < st["lam"]
                df[zc] = zv
                if st["violation"] == "offset" and st["w"] is not None:
                    dv = df[dc].to_numpy(copy=True) if dc in df.columns else \
                        np.zeros(len(df), dtype=np.asarray(st["offsets"]).dtype)
                    dv[rows] = st["offsets"][rng.choice(len(st["offsets"]), size=n, p=st["w"])]
                    df[dc] = dv
            z = df[zc].to_numpy(dtype=bool)
            fx = np.asarray(fx, dtype=object)
            new = df[lhs].to_numpy(dtype=object, copy=True)
            if st["ctype"] in ("integer", "float"):
                fxn = pd.to_numeric(pd.Series(fx), errors="coerce").to_numpy(dtype=float)
                new[z] = fxn[z]
                if dc in df.columns:
                    nz = ~z & ~np.isnan(fxn)
                    new[nz] = fxn[nz] + df[dc].to_numpy(dtype=float)[nz]
                vals = pd.to_numeric(pd.Series(new), errors="coerce")
                df[lhs] = vals.round().astype("Int64") if st["ctype"] == "integer" else vals
            else:
                new[z] = fx[z]
                df[lhs] = new
        return df

    def _apply_derive(self, t, df, rule, out):
        lhs, _ = split_assignment(rule["expr"])
        ctype = self.spec["tables"][t]["columns"].get(lhs, {}).get("type")
        lhs, vals = derive_values(df, rule["expr"], self._resolver(t, df, out), ctype)
        d = getattr(self, "derive_decimals", {}).get((t, lhs))
        if d is not None and ctype == "float":
            # stored to the source's precision (money to 4 decimals): the value the
            # database would hold, within the rule's tolerance of the exact product
            vals = np.round(pd.to_numeric(pd.Series(np.asarray(vals, dtype=object)), errors="coerce")
                            .to_numpy(dtype=float), d)
        df[lhs] = vals
        return df

    def _computed(self, t, df, out, strict=True):
        """Keys derived from other keys (offsets), lookups and derive rules, in that
        order: each reads only columns computed before it.  strict=False (an entity
        right after its links are drawn) skips what reads columns generated later."""
        for r in self.derived_keys.get(t, []):
            if r.get("offset_column") and r["offset_from"] in df.columns and r["offset_column"] in df.columns:
                par = out[r["parent"]]
                pk = self.spec["tables"][r["parent"]]["primary_key"]
                df[r["child_key"]] = offset_keys(df[r["offset_from"]], df[r["offset_column"]], par[pk],
                                                 par[r["order_by"]], r["unit"])
        res = self._resolver(t, df, out)
        for c in self.lookups[t]:
            try:
                df[c] = lookup_column(self.spec, t, c, df, out, res)
            except KeyError:
                if strict:
                    raise
        if strict:
            df = self._apply_distributions(t, df)
            df, _ = self._apply_group_effects(t, df, out)
        for r in self.spec["rules"]:
            if r["table"] == t and r["type"] == "derive" and split_assignment(r["expr"])[0] not in self.late[t]:
                try:
                    df = self._apply_derive(t, df, r, out)
                except KeyError:
                    if strict:
                        raise
        for r in self.spec["rules"]:
            if r["table"] == t and r["type"] == "cumulative" and len(df):
                df = cumulative_apply(df, r, self._resolver(t, df, out))
        if strict:
            df = self._apply_soft(t, df, out)
            df[HIDDEN_SET] = True               # every row's stated draws are now made
        return df

    def _fit_group_effects(self, tables):
        """Share of source groups (of 2 or more rows) on which each declared group
        effect holds, logged as a decision (g3-type measure, Parciak et al. 2024)."""
        for t, T in self.spec["tables"].items():
            for g in T.get("group_effects") or []:
                df = tables.get(t)
                if df is None or g["column"] not in df.columns:
                    continue
                try:
                    res = self._resolver(t, df, tables)
                    keys = [res(b) if "." in b else df[b] for b in g["by"]]
                except KeyError:
                    continue
                kf = pd.DataFrame({i: _keystr(pd.Series(np.asarray(k, dtype=object))) for i, k in enumerate(keys)})
                kf["v"] = _keystr(df[g["column"]]).to_numpy()
                gr = kf.groupby([c for c in kf.columns if c != "v"], dropna=True)["v"]
                size, nun = gr.size(), gr.nunique(dropna=False)
                multi = size >= 2
                share = float((nun[multi] == 1).mean()) if multi.any() else float("nan")
                self.log["decisions"].setdefault(t, {})[f"group effect {g['column']} by {', '.join(g['by'])}"] = (
                    f"one value per group on {share:.1%} of {int(multi.sum())} source groups with 2+ rows; "
                    "one value drawn per group")
                if share < 0.95:
                    self.log["warnings"].append(f"{t}: group effect on {g['column']} holds on only {share:.1%} of source groups; "
                         "the synthetic data will make it hold on all groups")

    def _fit_affinity(self, tables):
        """Link affinity (`link_affinity`): two links of a row point more often than
        their link models imply at parents that share a value (the customer's and the
        store's geography area).  Model: a mixture of the link model Q and Q
        restricted to the matching parents (a diagonal-inflated, quasi-independence
        table; Goodman 1968),
            P(b | row) = (1 - rho) Q(b | row) + rho Q(b | row, value(b) = value(a)),
        calibrated so that the share of matching rows equals s, with
        s ~ Beta(1 + h, 1 + n - h) (h of the n source rows match), drawn per dataset."""
        self.affinity = {}
        for t, T in self.spec["tables"].items():
            for la in T.get("link_affinity") or []:
                a, b = la["match"]
                st = {"a": a, "b": b, "fk": b.split(".")[0], "col": b.split(".", 1)[1]}
                if la.get("share") is not None:
                    st["h"], st["n"] = float(la["share"]) * 1e6, 1e6
                else:
                    df = tables.get(t)
                    try:
                        res = self._resolver(t, df, tables)
                        av, bv = _keystr(res(a) if "." in a else df[a]), _keystr(res(b))
                    except (KeyError, SpecError):
                        continue
                    st["h"], st["n"] = float(((av == bv) & av.notna()).sum()), float(len(df))

                def draw(rng, st=st):
                    st["s"] = float(rng.beta(1 + st["h"], 1 + st["n"] - st["h"]))
                self._register(draw)
                self.affinity.setdefault(t, []).append(st)
                self.log["decisions"].setdefault(t, {})[f"link affinity {a} = {b}"] = (
                    f"matching on {st['h'] / max(st['n'], 1):.2%} of source rows; {st['fk']} moved to a matching "
                    "parent on the extra share")

    def _apply_affinity(self, t, df, out):
        """Moves link fk of the rows needed to reach the drawn matching share s: rows
        not matching, in random order, each to a parent with the row's value of path
        a, chosen with probability proportional to the parent's degree in this table
        plus 1/2 (the link's popularity) and kept only when every key rule of the row
        still holds."""
        for st in getattr(self, "affinity", {}).get(t, []):
            r = next((x for x in self.fk[t] if x["child_key"] == st["fk"]), None)
            if r is None or not len(df) or any(st["fk"] in u for u in self.cons.get(t, {}).get("unique", [])):
                continue
            P = out[r["parent"]]
            pk = self.spec["tables"][r["parent"]]["primary_key"]
            if st["col"] not in P.columns:
                continue
            res = self._resolver(t, df, out)
            try:
                av = _keystr(pd.Series(np.asarray(res(st["a"]) if "." in st["a"] else df[st["a"]], dtype=object)))
                bv = _keystr(pd.Series(np.asarray(res(st["b"]), dtype=object)))
            except KeyError:
                continue
            av, bv = av.to_numpy(dtype=object), bv.to_numpy(dtype=object)
            same = np.array([x is not None and x == y for x, y in zip(av, bv)])
            need = int(round(st["s"] * len(df))) - int(same.sum())
            if need <= 0:
                continue
            pkeys = P[pk].to_numpy(dtype=object)
            deg = pd.Series(_keystr(df[st["fk"]]).to_numpy(dtype=object)).value_counts()
            pw = pd.Series(_keystr(P[pk]).to_numpy(dtype=object)).map(deg).fillna(0).to_numpy(dtype=float) + 0.5
            byval = pd.Series(np.arange(len(P))).groupby(_keystr(P[st["col"]]).to_numpy(dtype=object)).indices
            cand = np.flatnonzero(~same & np.array([x is not None and x in byval for x in av]))
            cand = cand[self.rng.permutation(len(cand))]
            # every row rule is re-checked: a moved link can also break a rule that the
            # sampler treats as an attribute rule (a link fixed by stratification)
            krules = self.cons.get(t, {}).get("key_rules", []) + self.cons.get(t, {}).get("attr_rules", [])
            moved = 0
            col = df[st["fk"]].to_numpy(dtype=object).copy()
            start = 0
            while moved < need and start < len(cand):
                # consecutive, non-overlapping windows: each candidate row is tried once
                rows = cand[start:start + 2 * (need - moved)]
                start += len(rows)
                newk = np.empty(len(rows), dtype=object)
                for v, ii in pd.Series(np.arange(len(rows))).groupby(av[rows]).indices.items():
                    m = byval[v]
                    newk[ii] = pkeys[m[self.rng.choice(len(m), size=len(ii), p=pw[m] / pw[m].sum())]]
                sub = df.iloc[rows].copy()
                sub[st["fk"]] = newk
                ok = np.ones(len(rows), bool)
                if krules:
                    sub = self._computed(t, sub, out, strict=False)
                    rs = self._resolver(t, sub, out)
                    for kr in krules:
                        ok &= check_mask(sub, kr["expr"], rs)
                acc = np.flatnonzero(ok)[:need - moved]
                col[rows[acc]] = newk[acc]
                moved += len(acc)
            df[st["fk"]] = col
            self.sample_log.setdefault("affinity_moves", {})[f"{t}.{st['fk']}"] = moved
        return df

    def _fit_participation(self, tables):
        """Participation of parents in an entity's link (customers with an order): a
        hurdle (Mullahy 1986).  The share of parent rows of segment t with no child,
        s0_t ~ Beta(c m + Z_t, c (1 - m) + N_t - Z_t) (Z_t of the N_t source parents
        have none), is drawn per dataset.  The segments share the prior Beta(c m,
        c (1 - m)) (exchangeable segments, BDA3 ch. 5): m = (sum Z + 1/2) / (sum N + 1)
        and c by the Beta-binomial moment estimator (as eb_beta_rows), so that a
        small segment of a link where no source parent is empty is not given a
        share 1 / (N_t + 2) of empty parents (a flat Beta(1, 1) per segment did:
        about 3% of AdventureWorks customers without a sale).  The link model (segment rates times gamma frailties)
        implies its own zero share (1 + lam/alpha)^-alpha, which the moment estimate of
        alpha can miss; when the drawn links leave more parents empty than the drawn
        share, children are moved to empty parents (_apply_participation)."""
        self.particip = {}
        cfg = self.cfg["structure"].get("participation", "auto")
        if cfg in (False, "none"):
            return
        for e in self.entities:
            if e in self.copied or e not in tables:
                continue
            for r in self.fk[e]:
                p = r["parent"]
                if p == e or p not in self.E or r.get("virtual") or r["child_key"] not in tables[e].columns:
                    continue
                if self.spec["tables"][p].get("virtual_dimension_of"):
                    continue        # a code implied inside the table: drawn with its anchor, columns read from it
                if r["child_key"] in getattr(self, "strata_info", {}).get(e, {}).get("by", []):
                    continue        # a link fixed by stratification is not moved
                idx = self._lookup(p, tables[e][r["child_key"]])
                deg = np.bincount(idx[idx >= 0], minlength=self.E[p].n)
                z = np.asarray(self.E[p].z)
                K = int(z.max()) + 1 if len(z) else 0
                N = np.bincount(z, minlength=K).astype(float)
                Z = np.bincount(z, weights=(deg == 0).astype(float), minlength=K)
                pa, pb = _pooled_beta(Z, N)
                st = {"Z": Z, "N": N, "a": pa, "b": pb, "s0": (Z + pa) / (N + pa + pb)}

                def draw(rng, st=st):
                    st["s0"] = rng.beta(st["a"] + st["Z"], st["b"] + st["N"] - st["Z"])
                self._register(draw)
                self.particip[(e, r["name"])] = st

    def _apply_participation(self, e, df, out, zs):
        """Children moved from parents with 2+ children, of the same segment first
        (chosen with probability proportional to degree - 1), to empty eligible parents of segments with more
        empty parents than round(s0_t N'_t), each move kept only when the row's key
        rules still hold."""
        for r in self.fk[e]:
            st = getattr(self, "particip", {}).get((e, r["name"]))
            p = r["parent"]
            if st is None or not len(df) or p not in out or p not in zs or \
                    any(r["child_key"] in u for u in self.cons.get(e, {}).get("unique", [])):
                continue
            pk = self.spec["tables"][p]["primary_key"]
            zp = np.asarray(zs[p])
            P = out[p]
            if len(zp) != len(P):
                continue
            pkeys = _keystr(P[pk]).to_numpy(dtype=object)
            pos = pd.Index(pkeys).get_indexer(_keystr(df[r["child_key"]]).to_numpy(dtype=object))
            deg = np.bincount(pos[pos >= 0], minlength=len(P))
            mask = self._edge_masks.get(r["name"]) if hasattr(self, "_edge_masks") else None
            elig = np.ones(len(P), bool) if mask is None or len(mask) != len(P) else np.asarray(mask) > 0
            K = len(st["s0"])
            krules = self.cons.get(e, {}).get("key_rules", []) + self.cons.get(e, {}).get("attr_rules", [])
            col = df[r["child_key"]].to_numpy(dtype=object).copy()
            rawkeys = P[pk].to_numpy(dtype=object)
            moved = 0
            # mandatory participation (_choose_parents covers every parent row): no
            # parent is to be empty, so moves only fill empty parents, never make one
            dp = self.edge_info.get(r["name"], {}).get("count_dp")
            mand = bool(dp.get("min", 0)) if dp else bool(self.edge_info.get(r["name"], {}).get("min_one"))
            for k in range(min(K, int(zp.max()) + 1 if len(zp) else 0)):
                seg = np.flatnonzero(zp == k)
                target = 0 if mand else int(round(float(st["s0"][k]) * len(seg)))
                empty = seg[(deg[seg] == 0) & elig[seg]]
                excess = int((deg[seg] == 0).sum()) - target
                if excess < 0:
                    # too few empty parents: the children of parents with one child move
                    # to parents of the segment that have children (probability by degree)
                    ones = seg[deg[seg] == 1]
                    ones = ones[self.rng.permutation(len(ones))][:-excess]
                    rows = np.flatnonzero(np.isin(pos, ones))
                    dest = seg[(deg[seg] >= 1) & ~np.isin(seg, ones)]
                    if not len(rows) or not len(dest):
                        continue
                    newp = dest[self.rng.choice(len(dest), size=len(rows), p=deg[dest] / deg[dest].sum())]
                    sub = df.iloc[rows].copy()
                    sub[r["child_key"]] = rawkeys[newp]
                    ok = np.ones(len(rows), bool)
                    if krules:
                        sub = self._computed(e, sub, out, strict=False)
                        rs = self._resolver(e, sub, out)
                        for kr in krules:
                            ok &= check_mask(sub, kr["expr"], rs)
                    for j in np.flatnonzero(ok):
                        deg[pos[rows[j]]] -= 1
                        deg[newp[j]] += 1
                        pos[rows[j]] = newp[j]
                        col[rows[j]] = rawkeys[newp[j]]
                    moved += int(ok.sum())
                    continue
                if excess == 0 or not len(empty):
                    continue
                empty = empty[self.rng.permutation(len(empty))][:excess]
                for rnd in range(4):
                    if not len(empty):
                        break
                    # donors from the same segment first: a child's other columns were
                    # drawn given its parent's segment (what a customer of that segment
                    # buys); other segments only for what the segment cannot supply
                    pp = np.maximum(pos, 0)
                    donors = np.flatnonzero((pos >= 0) & (deg[pp] >= 2) & (zp[pp] == k))
                    if rnd >= 2 or not len(donors):
                        donors = np.flatnonzero((pos >= 0) & (deg[pp] >= 2))
                    if not len(donors):
                        break
                    w = deg[pos[donors]] - 1.0
                    take = donors[self.rng.choice(len(donors), size=min(len(empty), len(donors)), replace=False,
                                                  p=w / w.sum())]
                    sub = df.iloc[take].copy()
                    sub[r["child_key"]] = rawkeys[empty[:len(take)]]
                    ok = np.ones(len(take), bool)
                    if krules:
                        sub = self._computed(e, sub, out, strict=False)
                        rs = self._resolver(e, sub, out)
                        for kr in krules:
                            ok &= check_mask(sub, kr["expr"], rs)
                    # one child per empty parent; a donor may have dropped to 1 meanwhile
                    acc = []
                    for j in np.flatnonzero(ok):
                        if deg[pos[take[j]]] >= 2:
                            acc.append(j)
                            deg[pos[take[j]]] -= 1
                            deg[empty[j]] += 1
                            pos[take[j]] = empty[j]
                            col[take[j]] = rawkeys[empty[j]]
                    moved += len(acc)
                    filled = empty[np.asarray(acc, dtype=int)] if len(acc) else empty[:0]
                    empty = np.setdiff1d(empty, filled)          # those not reached stay in the queue
                    empty = empty[self.rng.permutation(len(empty))]
            df[r["child_key"]] = col
            if moved:
                self.sample_log.setdefault("participation_moves", {})[f"{e}.{r['child_key']}"] = moved
        return df

    def _fit_scd(self, tables):
        """Type 2 slowly changing dimensions (`scd`): rows are versions of a natural
        key, valid from valid_from to valid_to, the latest one current.  Fitted:
          * versions per natural key: Dirichlet(counts + 1/2) over 1..M;
          * the share of keys whose latest version is current (valid_to null):
            Beta(1 + current, 1 + closed);
          * the gap valid_to(v) - valid_from(v + 1) in days: Dirichlet(counts + 1/2)
            over the observed gaps (0 or -1 day in a clean history);
          * stable columns: equal across the versions of a key on at least 99 % of
            keys with 2+ versions (repeated from the key's first version);
          * the current flag: a column with one value on current rows and another
            on the rest.
        All weights are drawn once per dataset."""
        self.scd = {}
        T = self.spec["tables"]
        for t, Tt in T.items():
            sc = Tt.get("scd")
            if not sc or t not in tables or t in self.copied:
                continue
            df = tables[t]
            nk, vf, vt = sc["natural_key"], sc["valid_from"], sc.get("valid_to")
            if nk not in df.columns or vf not in df.columns:
                continue
            key = _keystr(df[nk])
            start = pd.to_datetime(df[vf], errors="coerce", format="mixed")
            end = pd.to_datetime(df[vt], errors="coerce", format="mixed") if vt in df.columns else None
            ok = key.notna().to_numpy()
            sizes = key[ok].value_counts()
            M = int(min(sizes.max(), 50))
            vcount = np.bincount(np.minimum(sizes.to_numpy(), M), minlength=M + 1)[1:].astype(float)
            fr = pd.DataFrame({"k": key, "s": start, "e": end if end is not None else pd.NaT}).loc[ok]
            fr = fr.sort_values(["k", "s"])
            last = ~fr.duplicated("k", keep="last")
            cur_h = float(fr["e"][last].isna().sum()) if end is not None else float(last.sum())
            cur_n = float(last.sum())
            nxt = fr.groupby("k")["s"].shift(-1)
            gap = ((fr["e"] - nxt).dt.days).dropna()
            gv, gc = (np.unique(gap.to_numpy(), return_counts=True) if len(gap) else (np.zeros(0), np.zeros(0)))
            if len(gv) > 50:
                keep = np.argsort(gc)[-50:]
                gv, gc = gv[keep], gc[keep]
            # lengths of closed latest versions (days), for a latest version drawn closed
            dur = (fr["e"] - fr["s"])[last].dt.days.dropna().to_numpy(dtype=float)
            dur = np.quantile(dur, np.linspace(0, 1, 51)) if len(dur) > 51 else dur
            multi = sizes[sizes >= 2].index
            skip = {Tt.get("primary_key"), nk, vf, vt} | {r["child_key"] for r in self.fk[t]} | set(self.lookups[t]) \
                | set(self.derived[t])
            cur = sc.get("current")
            stable = sc.get("stable")
            if stable in (None, "auto"):
                stable = []
                inm = key.isin(set(multi)).to_numpy()
                for c, cs in Tt["columns"].items():
                    if c in skip or c not in df.columns or cs["type"] in ("id", "text", "binary") or not inm.any():
                        continue
                    nun = pd.Series(_keystr(df[c]).fillna("\x00").to_numpy()[inm]).groupby(key.to_numpy()[inm]).nunique()
                    if float((nun == 1).mean()) >= 0.99:
                        stable.append(c)
            if cur is None and end is not None:
                isnull = end.isna().to_numpy()
                for c, cs in Tt["columns"].items():
                    if c in skip or c not in df.columns or cs["type"] in ("id", "text", "binary"):
                        continue
                    v = _keystr(df[c])
                    a, b = v[isnull].fillna("\x00null"), v[~isnull].fillna("\x00null")
                    if len(a) and len(b) and a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] != b.iloc[0]:
                        raw = df[c]
                        cur = {"column": c, "value": raw[isnull].iloc[0],
                               "other": None if b.iloc[0] == "\x00null" else raw[~isnull].iloc[0]}
                        break
            st = {"nk": nk, "vf": vf, "vt": vt if vt in df.columns else None, "vcount": vcount,
                  "cur_h": cur_h, "cur_n": cur_n, "gaps": gv, "gcount": gc.astype(float), "stable": stable,
                  "current": cur, "dur": dur}

            def draw(rng, st=st):
                st["vw"] = rng.dirichlet(st["vcount"] + 0.5)
                st["pcur"] = float(rng.beta(1 + st["cur_h"], 1 + st["cur_n"] - st["cur_h"]))
                st["gw"] = rng.dirichlet(st["gcount"] + 0.5) if len(st["gcount"]) else None
            self._register(draw)
            self.scd[t] = st
            self.log["decisions"].setdefault(t, {})["scd type 2"] = (
                f"{int(sizes.size)} natural keys, {len(df)} versions (up to {M} per key); stable columns: "
                f"{', '.join(stable) or 'none'}; current flag: {cur['column'] if cur else 'none'}")

    def _apply_scd(self, t, df, seg):
        """Groups the synthetic rows into natural keys: group sizes from the drawn
        version law; rows of one segment grouped together (similar versions).  A
        row whose modelled valid_to is null (an open version, drawn by the row model
        with its segment, so the facts that point at it keep their link) becomes the
        latest version of its key: open rows are spread one per key where possible,
        and within a key the drawn start dates are given in order to the closed rows
        first and the open row last.  The key's code and its stable columns come
        from its first row; valid_to of each earlier version = next valid_from + a
        drawn gap.  Without a valid_to column the latest version is current with
        the drawn share."""
        st = getattr(self, "scd", {}).get(t)
        n = len(df)
        if st is None or not n:
            return df
        rng = self.rng
        M = len(st["vw"])
        segv = np.asarray(seg) if seg is not None and len(seg) == n else np.zeros(n, int)
        nk, vf, vt = st["nk"], st["vf"], st["vt"]
        # keys never span two strata (a column that stratifies the segments, such as a
        # sales-person flag, stays constant per key); inside a stratum, rows of one
        # segment are grouped together (similar versions)
        sc_ = getattr(self, "attr_strata", {}).get(t)
        blk = (pd.factorize(_keystr(df[sc_]).fillna("\x00"))[0] if sc_ in df.columns else np.zeros(n, int))
        perm = np.lexsort((rng.random(n), segv, blk))
        _, seg_n = np.unique(blk[perm], return_counts=True)
        parts = []
        for m in seg_n:
            draws = rng.choice(M, size=int(m), p=st["vw"]) + 1      # at most m keys are needed
            cum = np.cumsum(draws)
            G = int(np.searchsorted(cum, m)) + 1
            sz = draws[:G].copy()
            sz[-1] -= int(cum[G - 1]) - int(m)
            parts.append(sz[sz > 0])
        sizes = np.concatenate(parts) if parts else np.zeros(0, int)
        gid = np.empty(n, dtype=np.int64)
        gid[perm] = np.repeat(np.arange(len(sizes)), sizes)
        mo = pd.isna(df[vt]).to_numpy() if vt is not None else np.zeros(n, bool)
        if mo.any():
            # spread open rows one per key: an extra open row swaps places with a
            # closed row of a key that has none (same segment first)
            nopen = np.bincount(gid, weights=mo.astype(float), minlength=len(sizes)).astype(int)
            seen = np.zeros(len(sizes), bool)
            extra = []
            for i in rng.permutation(np.flatnonzero(mo)):
                if seen[gid[i]]:
                    extra.append(i)
                seen[gid[i]] = True
            free = np.flatnonzero(~mo & (nopen[gid] == 0))
            free = free[rng.permutation(len(free))]
            by_seg = pd.Series(free).groupby(segv[free]).apply(list).to_dict() if len(free) else {}
            pool = pd.Series(free).groupby(blk[free]).apply(list).to_dict() if len(free) else {}
            used = set()
            for i in extra:
                cand = by_seg.get(segv[i], [])
                j = None
                while cand:
                    c = cand.pop()
                    if c not in used and nopen[gid[c]] == 0:
                        j = c
                        break
                while j is None and pool.get(blk[i]):
                    c = pool[blk[i]].pop()
                    if c not in used and nopen[gid[c]] == 0:
                        j = c
                if j is None:
                    continue        # swaps stay inside a stratum
                used.add(j)
                gi, gj = gid[i], gid[j]
                gid[i], gid[j] = gj, gi
                nopen[gi] -= 1
                nopen[gj] += 1
        first = np.full(len(sizes), -1, dtype=np.int64)
        o1 = np.lexsort((mo, gid))
        first[gid[o1][::-1]] = o1[::-1]
        df[nk] = df[nk].to_numpy(dtype=object)[first[gid]]
        for c in st["stable"] + [h for h in _hidden_cols(df) if h.startswith("__dist_") and h[7:] in st["stable"]]:
            if c in df.columns:
                vals = df[c].to_numpy(dtype=object)[first[gid]]
                try:
                    df[c] = pd.Series(vals, index=df.index).astype(df[c].dtype)
                except (TypeError, ValueError):
                    df[c] = vals
        as_str = df[vf].dtype == object
        s_ = pd.to_datetime(df[vf], errors="coerce", format="mixed")
        s64 = s_.to_numpy(dtype="datetime64[ns]")
        # rows: closed versions by start, the open one last; start dates: sorted within the key
        order = np.lexsort((s64.astype(np.int64), mo, gid))
        sv = s64[np.lexsort((s64.astype(np.int64), gid))]
        g_o = gid[order]
        one = np.timedelta64(1, "D")
        for i in range(1, n):           # strictly increasing within a key
            if g_o[i] == g_o[i - 1] and not np.isnat(sv[i]) and not np.isnat(sv[i - 1]) and sv[i] <= sv[i - 1]:
                sv[i] = sv[i - 1] + one
        newstart = np.empty(n, dtype="datetime64[ns]")
        newstart[order] = sv
        is_last = np.r_[g_o[1:] != g_o[:-1], True]
        nxt = np.r_[sv[1:], np.datetime64("NaT")]
        cur_rows = np.zeros(n, bool)
        if vt is not None:
            e_ = pd.to_datetime(df[vt], errors="coerce", format="mixed").to_numpy(dtype="datetime64[ns]")[order]
            gap = np.zeros(n) if st["gw"] is None else st["gaps"][rng.choice(len(st["gaps"]), size=n, p=st["gw"])]
            if len(st["dur"]):
                dd = st["dur"][rng.integers(0, len(st["dur"]), size=n)]
                e_ = np.where(np.isnat(e_), sv + (dd * 86400e9).astype("timedelta64[ns]"), e_)
            ev = np.where(is_last, e_, nxt + (gap * 86400e9).astype("timedelta64[ns]"))
            # the open version is the row the model drew open (one per key after the spread)
            current = is_last & (mo[order] if mo.any() else (rng.random(n) < st["pcur"]))
            ev = np.where(current, np.datetime64("NaT"), ev)
            newend = np.empty(n, dtype="datetime64[ns]")
            newend[order] = ev
            cur_rows[order] = current
            fmt = (lambda a: pd.Series(pd.DatetimeIndex(a).strftime("%Y-%m-%d"), dtype=object).where(~np.isnat(a), None)
                   .to_numpy()) if as_str else (lambda a: a)
            df[vt] = fmt(newend)
        else:
            cur_rows[order] = is_last
        df[vf] = (pd.Series(pd.DatetimeIndex(newstart).strftime("%Y-%m-%d"), dtype=object).where(~np.isnat(newstart), None)
                  .to_numpy() if as_str else newstart)
        cur = st["current"]
        if cur and cur["column"] in df.columns:
            df[cur["column"]] = np.where(cur_rows, cur["value"], cur["other"])
        # a version date stored as an offset from another column (start = hire date +
        # gap): the hidden offset is set so that recomputing gives the new date
        from .expr import _days
        for o in self.plan.col_offsets:
            if o["table"] == t and o["column"] in (vf, vt) and o["base"] in df.columns and o["unit"] == "day":
                off = _days(df[o["column"]].to_numpy(dtype=object)) - _days(df[o["base"]].to_numpy(dtype=object))
                df[o["offset"]] = pd.array(np.where(np.isnan(off), np.nan, np.round(off)), dtype="Float64") \
                    .astype("Int64") if o["type"] == "integer" else off
        return df

    def _apply_distributions(self, t, df):
        """Columns whose law the user states (`distribution` on a column) are drawn
        from it, independently of the other columns, once per row (kept in a hidden
        column so that redraws of other columns do not change them):
          fixed {value}; categorical {probabilities: {level: p}}; uniform {min, max};
          normal {mean, sd, min?, max?}; lognormal {mean, sd} (of the log);
          poisson {mean}; date_range {start, end}; sequence {start, step};
          empirical (the source values, resampled with Bayesian-bootstrap weights:
          the column keeps its marginal but loses its dependence on the others).
        A null share `nulls` (0 by default) applies to every kind."""
        T = self.spec["tables"][t]
        for c, cs in T["columns"].items():
            d = cs.get("distribution")
            if not d or not len(df) or c not in df.columns and cs["type"] in ("lookup",):
                continue
            hc = f"__dist_{c}"
            rows = _fresh_rows(df, hc)
            if not rows.any():
                df[c] = df[hc].to_numpy(dtype=object) if df[hc].dtype == object else df[hc].to_numpy()
                continue
            n = int(rows.sum())
            rng = self.rng
            k = d["kind"]
            if k == "fixed":
                v = np.full(n, d["value"], dtype=object)
            elif k == "categorical":
                lv = list(d["probabilities"])
                p = np.array([float(d["probabilities"][x]) for x in lv])
                v = np.array(lv, dtype=object)[rng.choice(len(lv), size=n, p=p / p.sum())]
            elif k == "uniform":
                v = rng.uniform(float(d["min"]), float(d["max"]), n)
            elif k == "normal":
                v = rng.normal(float(d["mean"]), float(d["sd"]), n)
            elif k == "lognormal":
                v = np.exp(rng.normal(float(d["mean"]), float(d["sd"]), n))
            elif k == "poisson":
                v = rng.poisson(float(d["mean"]), n).astype(float)
            elif k == "date_range":
                a, b = pd.Timestamp(d["start"]), pd.Timestamp(d["end"])
                v = (a + pd.to_timedelta(rng.integers(0, (b - a).days + 1, n), "D")).strftime("%Y-%m-%d").to_numpy(
                    dtype=object)
            elif k == "sequence":
                v = float(d.get("start", 1)) + float(d.get("step", 1)) * np.flatnonzero(rows)
            elif k == "empirical":
                src = self.dist_src.get((t, c), np.zeros(0, dtype=object))
                v = src[rng.choice(len(src), size=n, p=self._bb(("dist", t, c), len(src), rng))] if len(src) \
                    else np.full(n, None, dtype=object)
            else:
                continue
            if k in ("uniform", "normal", "lognormal", "poisson", "sequence"):
                v = np.clip(v, float(d.get("min", -np.inf)), float(d.get("max", np.inf)))
                if cs["type"] == "integer":
                    v = np.round(v)
                elif d.get("decimals") is not None:
                    v = np.round(v, int(d["decimals"]))
            if d.get("nulls"):
                v = np.asarray(v, dtype=object)
                v[rng.random(n) < float(d["nulls"])] = None
            if rows.all():
                df[hc] = v
            else:
                # only rows drawn afresh (new or redrawn rows) get new values; a drawn
                # null is a value too and is kept
                col = df[hc].to_numpy(dtype=object, copy=True)
                col[rows] = np.asarray(v, dtype=object)
                df[hc] = col if np.asarray(v).dtype == object else pd.to_numeric(pd.Series(col), errors="coerce").to_numpy()
            df[c] = df[hc].to_numpy(dtype=object) if df[hc].dtype == object else df[hc].to_numpy()
        return df

    def _apply_bridge(self, t, df):
        """Bridge weights (`bridge: {group, weight}`): the weights of a group (the
        accounts of one customer, the diagnoses of one stay) sum to `total` (1 by
        default), as allocation factors must.  The modelled weights are kept up to
        a common factor: w_i <- total * w_i / sum of the group's w (a Dirichlet
        draw when the weights are Gamma draws), rounded to the source precision
        with the rounding remainder put on the group's largest weight."""
        br = self.spec["tables"][t].get("bridge")
        if not br or not len(df) or br.get("weight") not in df.columns or br.get("group") not in df.columns:
            return df
        total = float(br.get("total", 1.0))
        w = pd.to_numeric(df[br["weight"]], errors="coerce").to_numpy(dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        g = pd.factorize(_keystr(df[br["group"]]), use_na_sentinel=False)[0]
        tot = np.bincount(g, weights=w)
        cnt = np.bincount(g)
        w = np.where(tot[g] > 0, w / np.maximum(tot[g], 1e-300), 1.0 / cnt[g]) * total
        d = br.get("decimals")
        if d is not None:
            w = np.round(w, int(d))
            rem = total - np.bincount(g, weights=w)
            big = pd.Series(w).groupby(g).idxmax().to_numpy()
            w[big] = np.round(w[big] + rem, int(d))
        df[br["weight"]] = w
        return df

    def _apply_group_effects(self, t, df, out):
        """Group effects (`group_effects`): a value shared by all rows of a group
        (one price factor per category and day).  Model: u_g drawn once per group
        and given to its rows, a shared random effect (Hu, Reiter & Wang 2018).
        u_g is drawn from the row model's predictive law at a representative row
        of the group: the row with the smallest of a uniform priority drawn once per
        row and kept in a hidden column, so recomputing after redraws keeps the same
        representative.  Rows with a null group key keep their own value."""
        ges = self.spec["tables"][t].get("group_effects") or []
        if not ges or not len(df):
            return df, False
        res = self._resolver(t, df, out)
        changed = False
        for g in ges:
            c = g["column"]
            if c not in df.columns:
                continue
            try:
                keys = [res(b) if "." in b else df[b] for b in g["by"]]
            except KeyError:
                continue
            pc = f"__ge_r_{c}"
            if pc not in df.columns:
                df[pc] = self.rng.random(len(df))
            else:
                rows = _fresh_rows(df, pc)
                if rows.any():
                    pv = df[pc].to_numpy(dtype=float, copy=True)
                    pv[rows] = self.rng.random(int(rows.sum()))
                    df[pc] = pv
            kf = pd.DataFrame({i: _keystr(pd.Series(np.asarray(k, dtype=object))) for i, k in enumerate(keys)})
            code = kf.groupby(list(kf.columns), dropna=True, sort=False).ngroup().to_numpy()
            idx = np.flatnonzero(code >= 0)
            if not len(idx):
                continue
            rep = pd.Series(df[pc].to_numpy()[idx], index=idx).groupby(code[idx]).idxmin()
            vals = df[c].to_numpy(dtype=object)
            new = vals.copy()
            new[idx] = vals[rep.loc[code[idx]].to_numpy()]
            if not changed:
                same = pd.Series(new).astype(str).to_numpy() == pd.Series(vals).astype(str).to_numpy()
                changed = not bool(same.all())
            try:
                df[c] = pd.Series(new, index=df.index).astype(df[c].dtype)
            except (TypeError, ValueError):
                df[c] = new
        return df, changed

    def _fit_inclusion(self, tables):
        """Conditional inclusion dependencies (Bravo, Fan & Ma 2007): a link that
        never points at parent rows with a given level of a parent column (a null
        cost, a component that is never sold) although, if the referenced rows were
        a random subset, about D n_l / n of them would have that level (D distinct
        referenced rows, n_l of the n parent rows at the level).  With at least 10
        expected and none observed (probability about e^-10 under random
        selection) the level is excluded when the link chooses parent rows."""
        T = self.spec["tables"]
        self.cind = {}
        minexp = float(self.cfg["structure"].get("inclusion_min_expected", 10.0))
        if self.cfg["structure"].get("conditional_inclusion", "auto") in (False, "none"):
            return
        for t in T:
            fks = self.fk[t]
            for r in fks:
                p = r["parent"]
                if p == t or T[p]["role"] != "entity" or r["child_key"] not in tables[t]:
                    continue
                pk = T[p]["primary_key"]
                par = tables[p].drop_duplicates(pk)
                n = len(par)
                ref = set(_keystr(self._ref_values(t, r["child_key"], tables)).dropna())
                isref = _keystr(par[pk]).isin(ref).to_numpy()
                D = int(isref.sum())
                if n < 20 or D == 0 or D == n:
                    continue
                excl = []
                # strata columns are generated (exactly per stratum), so they can carry a rule
                skip = self._excluded(p) - set(self.lookups[p]) - set(self.derived[p]) \
                    - {getattr(self, "attr_strata", {}).get(p)}
                for c, cs in T[p]["columns"].items():
                    if c in skip or c not in par or cs["type"] in ("text", "pii", "id", "binary", "constant"):
                        continue
                    col = par[c]
                    if cs["type"] in ("categorical", "boolean") and col.nunique() <= 50:
                        lev = _keystr(col).fillna("\x00null").to_numpy(dtype=object)
                    else:
                        lev = np.where(col.isna().to_numpy(), "\x00null", "value").astype(object)
                    for l_ in pd.unique(lev):
                        at = lev == l_
                        nl = int(at.sum())
                        if nl == n or isref[at].any():
                            continue
                        if D * nl / n >= minexp:
                            excl.append((c, None if l_ == "\x00null" else l_, l_ == "value"))
                if excl:
                    self.cind[(t, r["child_key"])] = excl
                    self.log["decisions"].setdefault(t, {})[f"{r['name']} inclusion"] = (
                        "never references parent rows with " + "; ".join(
                            f"{c} {'null' if v is None else ('set' if isval else '= ' + str(v))}" for c, v, isval in excl))

    def _inclusion_mask(self, t, r, n, out):
        """1 for parent rows the link can point at, 0 for excluded levels."""
        rules = getattr(self, "cind", {}).get((t, r.get("child_key")))
        mask = np.ones(n)
        if not rules or r.get("virtual") or r["parent"] not in out:
            return mask
        par = out[r["parent"]]
        if len(par) != n:
            return mask
        for c, v, isval in rules:
            if c not in par:
                continue
            col = par[c]
            if v is None:
                mask[col.isna().to_numpy()] = 0
            elif isval:
                mask[col.notna().to_numpy()] = 0
            else:
                mask[(_keystr(col) == v).to_numpy()] = 0
        if mask.sum() == 0:
            return np.ones(n)
        return mask

    def _aggregates(self, out):
        """Cross-table aggregate rules: a parent column computed from its children
        (count, sum, ...), bottom-up; then lookups and derive rules of every
        synthesised table are re-applied in dependency order, so that columns
        defined from an aggregate (directly or through a lookup) stay exact."""
        rules = [r for r in self.spec["rules"] if r["type"] == "aggregate"]
        if not rules:
            return
        T = self.spec["tables"]
        for t in reversed(self.all_order):
            mine = [r for r in rules if r["table"] == t]
            for r in mine:
                val = compute_aggregate(self.spec, r, out)
                d = self.agg_decimals.get((t, r["column"]))
                if d is not None and T[t]["columns"][r["column"]]["type"] == "float":
                    val = np.round(pd.to_numeric(pd.Series(np.asarray(val)), errors="coerce").to_numpy(dtype=float), d)
                out[t][r["column"]] = val
            if mine and t not in self.copied:
                # columns derived from these aggregates (gross = 2 * total) are needed
                # now by the aggregates of the tables above (spend = sum of gross);
                # the top-down pass below recomputes them with the final lookups
                for r in [r for r in self.spec["rules"] if r["table"] == t and r["type"] == "derive"]:
                    out[t] = self._apply_derive(t, out[t], r, out)
        for t in self.all_order:
            if t in self.copied:
                continue
            df = out[t]
            for c in self.lookups[t]:
                # plain paths and keyed lookups (a dict) alike, as in _computed
                df[c] = lookup_column(self.spec, t, c, df, out, self._resolver(t, df, out))
            for r in [r for r in self.spec["rules"] if r["table"] == t and r["type"] == "derive"]:
                df = self._apply_derive(t, df, r, out)
            for r in [r for r in self.spec["rules"] if r["table"] == t and r["type"] == "cumulative"]:
                if len(df):
                    df = cumulative_apply(df, r, self._resolver(t, df, out))
            viol = self.sample_log["rule_violations"].setdefault(t, {})
            for r in [r for r in self.spec["rules"] if r["table"] == t and r["type"] == "check"]:
                viol[r["expr"]] = int((~check_mask(df, r["expr"], self._resolver(t, df, out))).sum())
            out[t] = df[[c for c in T[t]["columns"]]]

    # ================================================================== report
    def model_summary(self):
        h = self.rm.history
        out = {"segments": {"caps": h["caps"], "occupied_before_merge": h["occupied_before_merge"],
                            "final": h["segments_final"], "merges": {k: len(v) for k, v in h.get("merges", {}).items()},
                            "strata": {e: {"by": v["by"] or [v.get("attr")], "count": len(v["labels"])}
                                       for e, v in self.strata_info.items()}},
               "gibbs_loglik_trace": h["loglik_trace"],
               "copied_tables": sorted(getattr(self, "copied", [])),
               "copulas": {t: self.tm[t].copula_info() for t in self.tm
                           if self.tm[t].cols and t not in getattr(self, "copied", set())},
               "dependency_groups": {t: self.tm[t].fd_groups for t in self.tm
                                     if self.tm[t].fd_groups and t not in getattr(self, "copied", set())},
               "independent_columns": {t: list(getattr(self.tm[t], "independent", []))
                                       for t in self.tm if getattr(self.tm[t], "independent", None)},
               "relations": {}, "warnings": self.log["warnings"], "timings": self.log["timings"],
               "decisions": self.log["decisions"]}
        for name, info in self.edge_info.items():
            rel = info["rel"]
            d = {"kind": rel.kind, "dispersion_index": [float(x) for x in info.get("dispersion", [])]}
            if rel.kind == "tree":
                M = info["psi"] * info["lam"][:, None]
                d["mean_matrix_spectral_radius"] = float(np.max(np.abs(np.linalg.eigvals(M))))
                d["frailty_on"] = bool(np.isfinite(info["frailty"]).any())
            else:
                d["frailty_on"] = bool(np.isfinite(info["parent_frailty"]).any())
                d["orphans"], d["nulls"] = info["orphans"], info["nulls"]
                d["parent_copied"] = bool(info.get("fixed_parent"))
                d["count_model"] = "dp" if info.get("count_dp") else ("1+poisson" if info.get("min_one") else "poisson")
            out["relations"][name] = d
        for f, info in self.fact_info.items():
            rel = info["rel"]
            d = {"kind": rel.kind, "model": info.get("model"), "dropped_rows": info["dropped"],
                 "anchor": info["anchor"], "nested": info["nested"],
                 "virtual_dimension": info.get("vdim"),
                 "multiplicity": info.get("multiplicity"),
                 "group_fallback": info.get("fallback"),
                 "dispersion_index_anchor": [float(x) for x in info.get("dispersion1", [])],
                 "dispersion_index_partner": [float(x) for x in info.get("dispersion2", [])],
                 "frailty_on": bool(np.isfinite(info["frailty1"]).any()),
                 "count_model": "dp" if info.get("count_dp") else ("1+poisson" if info.get("min_one") else "poisson"),
                 "time_buckets": len(info["time"]["buckets"]) if "time" in info else 0}
            if rel.kind == "occ":
                d["structural_blocks"] = {"p=0": int((rel.p == 0).sum()), "p=1": int((rel.p == 1).sum()),
                                          "other": int(((rel.p > 0) & (rel.p < 1)).sum())}
                d["inactive_anchor_segments"] = int((rel.pi == 0).sum())
            out["relations"][f] = d
        return out


def _choice_rows(rng, P, k):
    """np.array([rng.choice(P.shape[1], p=P[k_i] / P[k_i].sum()) for k_i in k]) without the loop:
    Generator.choice(n, p=p) draws one uniform u and returns searchsorted(cdf, u,
    'right') with cdf = p.cumsum() / p.cumsum()[-1], and random(len(k)) gives the
    same uniforms as len(k) single calls, so the result and the random stream are
    identical."""
    k = np.asarray(k, dtype=np.int64)
    if not len(k):
        return np.zeros(0, dtype=np.int64)
    u = rng.random(len(k))
    out = np.empty(len(k), dtype=np.int64)
    for j in np.unique(k):
        cdf = (P[j] / P[j].sum()).cumsum()
        cdf /= cdf[-1]
        rows = np.flatnonzero(k == j)
        out[rows] = cdf.searchsorted(u[rows], side="right")
    return out


def _unique_codes(gen, n, rng, rounds=50):
    """n distinct values from a format-preserving generator: duplicates (and nulls,
    a key has none) are redrawn; when the formats cannot hold n distinct values a
    running number is appended to the rest."""
    gen.null_rate = 0.0
    out = pd.Series(gen.generate(n, rng), dtype=object)
    for _ in range(rounds):
        dup = out.duplicated(keep="first").to_numpy()
        if not dup.any():
            return out.to_numpy(dtype=object)
        out[dup] = gen.generate(int(dup.sum()), rng)
    dup = out.duplicated(keep="first").to_numpy()
    out[dup] = [f"{v}-{i}" for i, v in enumerate(out[dup])]
    return out.to_numpy(dtype=object)


def _bernoulli_grid(A, J, p, rng):
    """All (i, j) in A x J with independent Bernoulli(p) inclusion; geometric
    skipping for large grids (cost proportional to the number of successes)."""
    N = len(A) * len(J)
    if N == 0 or p <= 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    if N <= 2_000_000:
        hit = np.flatnonzero(rng.random(N) < p)
    else:
        exp_n = int(N * p + 10 * np.sqrt(N * p) + 10)
        pos = np.cumsum(rng.geometric(p, size=exp_n)) - 1
        while pos[-1] < N:
            pos = np.concatenate([pos, pos[-1] + np.cumsum(rng.geometric(p, size=exp_n))])
        hit = pos[pos < N]
    return A[hit // len(J)], J[hit % len(J)]


def allocate(p, n, rng):
    """Segment labels with counts n*p rounded by largest remainder (less noise
    than multinomial draws for small tables), in random order."""
    raw = p * n
    cnt = np.floor(raw).astype(int)
    rem = n - cnt.sum()
    if rem > 0:
        cnt[np.argsort(-(raw - cnt) + 1e-9 * rng.random(len(p)))[:rem]] += 1
    return rng.permutation(np.repeat(np.arange(len(p)), cnt))


def _eval_check(df, expr):
    """Backward-compatible helper: True where a single-table rule holds."""
    try:
        return check_mask(df, expr)
    except Exception:
        traceback.print_exc()
        return np.ones(len(df), dtype=bool)
