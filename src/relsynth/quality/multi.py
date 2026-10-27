# SPDX-License-Identifier: Apache-2.0
"""Multi-table fidelity metrics of the 2.0 report.

* Fan-out laws: per relationship, the law of the number of child rows per
  parent row (W1 relative to the real mean, share of parents without
  children, log ratio of the dispersion index), and the same for two-hop
  chains (grandchildren per grandparent).
* Join-preserving association: for every column pair across a foreign-key
  path of 1 or 2 hops, the column-pair trend score of SDMetrics
  (CorrelationSimilarity for two numeric columns, ContingencySimilarity
  otherwise), as used by ClavaDDPM and SyntheRela ("k-hop trends").
* Aggregation detection (C2ST-Agg, Hudovernik et al. TMLR 2026): each parent
  table gains, per child relationship, the child count, the mean of every
  numeric child column and the number of distinct values of every categorical
  child column; a classifier separates real from synthetic parent rows by
  stratified k-fold cross-validation on balanced samples, and its accuracy is
  reported (0.5 = indistinguishable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr, wasserstein_distance

from ..evaluate import _logreg, _num, keystr
from .encode import attribute_columns, codes, feature_matrix, is_categorical


def relations(spec):
    """Simple many-to-one relationships between two different tables:
    [(name, parent, pk, child, fk)]."""
    T = spec["tables"]
    out = []
    for r in spec["relationships"]:
        if r.get("composite") or r["kind"] == "self_reference" or r["parent"] == r["child"]:
            continue
        pk = r.get("parent_key") or T[r["parent"]].get("primary_key")
        if not pk or isinstance(r["child_key"], list) or isinstance(pk, list):
            continue
        out.append((r["name"], r["parent"], pk, r["child"], r["child_key"]))
    return out


def counts_per_parent(parent, pk, child, fk):
    """Number of child rows per parent row, in parent row order."""
    vc = keystr(child[fk]).dropna().value_counts()
    return keystr(parent[pk]).map(vc).fillna(0).to_numpy(dtype=float)


# ---------------------------------------------------------------- fan-out laws
def fanout_law(cr, cs):
    """Distances between two samples of per-parent counts."""
    if not len(cr) or not len(cs):
        return {}
    mr = float(cr.mean())
    disp = (lambda c: float(c.var() / c.mean()) if c.mean() > 0 else 1.0)
    return {"w1_rel_mean": float(wasserstein_distance(cr, cs) / mr) if mr > 0 else float(cs.mean() > 0),
            "zero_share_diff": float(abs((cr == 0).mean() - (cs == 0).mean())),
            "dispersion_log_ratio": float(abs(np.log(max(disp(cs), 1e-9) / max(disp(cr), 1e-9)))),
            "mean_real": mr, "mean_other": float(cs.mean()), "parents": [int(len(cr)), int(len(cs))]}


def two_hop_chains(spec):
    """(grandparent, gpk, parent, pk_parent, fk_parent_to_gp, child, fk_child_to_parent)."""
    rel = relations(spec)
    out = []
    for n1, gp, gpk, p, fk1 in rel:
        for n2, p2, ppk, c, fk2 in rel:
            if p2 == p and c != gp:
                out.append((f"{n1} | {n2}", gp, gpk, p, ppk, fk1, c, fk2))
    return out


def grandchild_counts(tables, gp, gpk, p, ppk, fk1, c, fk2):
    """Rows of c per row of gp through p."""
    cnt_p = pd.Series(counts_per_parent(tables[p], ppk, tables[c], fk2), index=keystr(tables[p][fk1]).to_numpy())
    per_gp = cnt_p.groupby(level=0).sum()
    return keystr(tables[gp][gpk]).map(per_gp).fillna(0).to_numpy(dtype=float)


# ---------------------------------------------------------------- k-hop trends
def _pair_trend(xa, xb, ya, yb, cat_x, cat_y, bins=10):
    """SDMetrics column-pair trend score in [0, 1] (1 = equal): 1 - |r_a - r_b| / 2
    with Pearson r for two numeric columns, else 1 - TVD of the joint table of the
    discretised columns (numeric columns in 10 equal-width bins on the real range)."""
    if not cat_x and not cat_y:
        fa, fb = _num(xa), _num(xb)
        ga, gb = _num(ya), _num(yb)
        oka, okb = ~(np.isnan(fa) | np.isnan(ga)), ~(np.isnan(fb) | np.isnan(gb))
        if oka.sum() < 3 or okb.sum() < 3 or np.ptp(fa[oka]) == 0 or np.ptp(ga[oka]) == 0:
            return None
        ra = pearsonr(fa[oka], ga[oka]).statistic
        rb = pearsonr(fb[okb], gb[okb]).statistic if np.ptp(fb[okb]) > 0 and np.ptp(gb[okb]) > 0 else 0.0
        return float(1 - abs(ra - rb) / 2)

    def disc(a, b, cat):
        if cat:
            ca, cb, _ = codes(a, b, True, max_levels=10 ** 6)
            return ca, cb
        fa, fb = _num(a), _num(b)
        ok = fa[~np.isnan(fa)]
        if not len(ok):
            return np.zeros(len(fa), int), np.zeros(len(fb), int)
        edges = np.histogram_bin_edges(ok, bins=bins)[1:-1]
        ea, eb = np.searchsorted(edges, fa, side="right"), np.searchsorted(edges, fb, side="right")
        ea[np.isnan(fa)] = bins
        eb[np.isnan(fb)] = bins
        return ea, eb
    xa_, xb_ = disc(xa, xb, cat_x)
    ya_, yb_ = disc(ya, yb, cat_y)
    ka = xa_.astype(np.int64) * (10 ** 6 + 2) + ya_
    kb = xb_.astype(np.int64) * (10 ** 6 + 2) + yb_
    _, inv = np.unique(np.r_[ka, kb], return_inverse=True)
    m = inv.max() + 1
    p = np.bincount(inv[:len(ka)], minlength=m) / max(len(ka), 1)
    q = np.bincount(inv[len(ka):], minlength=m) / max(len(kb), 1)
    return float(1 - 0.5 * np.abs(p - q).sum())


def _join(tables, child, fk, parent, pk, pcols, prefix):
    pdf = tables[parent][[pk] + [c for c in pcols if c != pk]].copy()
    pdf.columns = [prefix + c for c in pdf.columns]
    left = tables[child].assign(_k=keystr(tables[child][fk]).fillna("__null__").to_numpy())
    right = pdf.assign(_k=keystr(pdf[prefix + pk]).to_numpy())
    return left.merge(right, on="_k", how="left").drop(columns="_k")


def hop_views(spec, tables, max_hops=2):
    """Denormalised views along foreign-key paths: {(path name, hops):
    (view, bottom table, bottom columns, top table, top columns)} with top-table
    columns prefixed '^.'."""
    out = {}
    for name, p, pk, c, fk in relations(spec):
        pc = attribute_columns(spec, p)
        cc = attribute_columns(spec, c)
        if pc and cc:
            v = _join(tables, c, fk, p, pk, pc, "^.")
            out[(name, 1)] = (v, c, cc, p, pc)
    if max_hops >= 2:
        for name, gp, gpk, p, ppk, fk1, c, fk2 in two_hop_chains(spec):
            gc = attribute_columns(spec, gp)
            cc = attribute_columns(spec, c)
            if not gc or not cc:
                continue
            mid = tables[p][[ppk, fk1]].copy()
            mid.columns = ["_mid_pk", "_mid_fk"]
            v = tables[c].assign(_k=keystr(tables[c][fk2]).fillna("__null__").to_numpy()).merge(
                mid.assign(_k=keystr(mid["_mid_pk"]).to_numpy()), on="_k", how="left").drop(columns="_k")
            g = tables[gp][[gpk] + gc].copy()
            g.columns = ["^." + x for x in g.columns]
            v = v.assign(_k=keystr(v["_mid_fk"]).fillna("__null__").to_numpy()).merge(
                g.assign(_k=keystr(g["^." + gpk]).to_numpy()), on="_k", how="left").drop(columns="_k")
            out[(name, 2)] = (v, c, cc, gp, gc)
    return out


def hop_trends(spec, real, other, max_hops=2, max_rows=20000, seed=0):
    """{path: {"hops", "mean", "min", "worst", "pairs": {pair: [score, |diff| ]}}}
    for each 1- and 2-hop path, comparing pairs (bottom column, top column)."""
    vr, vo = hop_views(spec, real, max_hops), hop_views(spec, other, max_hops)
    out = {}
    for key, (a, c, cc, p, pc) in vr.items():
        if key not in vo:
            continue
        b = vo[key][0]
        if len(a) > max_rows:
            a = a.sample(max_rows, random_state=seed)
        if len(b) > max_rows:
            b = b.sample(max_rows, random_state=seed)
        pairs = {}
        for x in cc:
            cx = is_categorical(spec, c, x, real[c][x])
            for y in pc:
                cy = is_categorical(spec, p, y, real[p][y])
                s = _pair_trend(a[x], b[x], a["^." + y], b["^." + y], cx, cy)
                if s is not None:
                    pairs[f"{c}.{x}~{p}.{y}"] = s
        if pairs:
            w = min(pairs, key=pairs.get)
            out[f"{key[0]}"] = {"hops": key[1], "mean": float(np.mean(list(pairs.values()))),
                                "min": pairs[w], "worst": w, "k_pairs": len(pairs),
                                "pairs": {k: round(v, 4) for k, v in pairs.items()}}
    return out


# ---------------------------------------------------------------- aggregation detection
def aggregated(spec, tables, parent, factors=None):
    """The parent table's attribute columns plus, per child relationship, the
    child count (divided by factors[relationship] when the tables were
    resized), the mean of each numeric child column and the number of
    distinct values of each categorical child column (SyntheRela's C2ST-Agg
    features).  Returns (frame, feature kinds {column: categorical?})."""
    pk = spec["tables"][parent]["primary_key"]
    base = tables[parent]
    cols = attribute_columns(spec, parent)
    df = base[cols].copy()
    kinds = {c: is_categorical(spec, parent, c, base[c]) for c in cols}
    key = keystr(base[pk]).to_numpy()
    for name, p, _pk, c, fk in relations(spec):
        if p != parent:
            continue
        ch = tables[c]
        k = keystr(ch[fk])
        df[f"{name}:count"] = pd.Series(key).map(k.value_counts()).fillna(0).to_numpy() / (factors or {}).get(name, 1.0)
        kinds[f"{name}:count"] = False
        for x in attribute_columns(spec, c):
            cat = is_categorical(spec, c, x, ch[x])
            g = pd.DataFrame({"k": k.to_numpy(), "v": ch[x].astype(str).where(ch[x].notna(), None) if cat else _num(ch[x])})
            agg = g.groupby("k")["v"].nunique() if cat else g.groupby("k")["v"].mean()
            col = f"{name}:{x}:{'nunique' if cat else 'mean'}"
            df[col] = pd.Series(key).map(agg).to_numpy(dtype=float)
            if cat:
                df[col] = df[col].fillna(0)
            kinds[col] = False
    return df, kinds


def _as_spec_cols(kinds):
    """A minimal table spec for feature_matrix on an aggregated frame."""
    return {"tables": {"_agg": {"columns": {c: {"type": "categorical" if v else "float"} for c, v in kinds.items()}}}}


def detection_accuracy(real_df, other_df, kinds, classifier="logistic", folds=5, max_rows=5000, seed=0):
    """Stratified k-fold accuracy of a classifier separating balanced samples of
    real and other rows (0.5 = indistinguishable).  classifier: 'logistic'
    (L2 logistic regression on rank-normal scores, their squares, null
    indicators and one-hot levels; no extra dependency) or 'xgboost' (needs the
    xgboost package; raw numbers and level codes, as SyntheRela)."""
    n = min(len(real_df), len(other_df), max_rows)
    if n < folds * 2 or not kinds:
        return float("nan")
    rng = np.random.default_rng(seed)
    r = real_df.iloc[rng.permutation(len(real_df))[:n]].reset_index(drop=True)
    s = other_df.iloc[rng.permutation(len(other_df))[:n]].reset_index(drop=True)
    cols = list(kinds)
    if classifier == "xgboost":
        X = _raw_design(r, s, kinds)
    else:
        sp = _as_spec_cols(kinds)
        Xr, Xs = feature_matrix(r, s, sp, "_agg", cols)
        X = np.column_stack([np.r_[Xr, Xs], np.ones(2 * n)])
    y = np.r_[np.ones(n), np.zeros(n)]
    fold = np.r_[rng.permutation(np.arange(n) % folds), rng.permutation(np.arange(n) % folds)]
    hits = 0
    for f in range(folds):
        tr, te = fold != f, fold == f
        if classifier == "xgboost":
            import xgboost as xgb
            # the defaults of xgboost.XGBClassifier (100 trees, depth 6, eta 0.3), without scikit-learn
            m = xgb.train({"objective": "binary:logistic", "seed": seed + f, "nthread": 1, "verbosity": 0},
                          xgb.DMatrix(X[tr], label=y[tr]), num_boost_round=100)
            pred = m.predict(xgb.DMatrix(X[te])) > 0.5
        else:
            sd = X[tr].std(0)
            sd[sd == 0] = 1
            w = _logreg(X[tr] / sd, y[tr])
            pred = expit((X[te] / sd) @ w) > 0.5
        hits += int((pred == (y[te] == 1)).sum())
    return float(hits / (2 * n))


def _raw_design(r, s, kinds):
    cols = []
    for c, cat in kinds.items():
        if cat:
            a, b, _ = codes(r[c], s[c], True, max_levels=10 ** 6)
            cols.append(np.r_[a, b].astype(float))
        else:
            cols.append(np.r_[_num(r[c]), _num(s[c])])
    return np.column_stack(cols)


def parents_with_children(spec):
    return sorted({p for _n, p, _pk, _c, _fk in relations(spec)})
