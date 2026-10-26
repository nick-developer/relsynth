# SPDX-License-Identifier: Apache-2.0
"""Dependency discovery beyond keys, links and lookups (used by introspect).

Each method adds one kind of structure to the draft; every one is the same test
as the rest of introspect (profiling.py): it must hold on at least MIN_SHARE of
the rows that can show it, and that agreement must be beyond chance (one-sided
binomial test at ALPHA against the agreement expected when the inputs are
unrelated, measured by permuting one side).

* reference tables: a table without a primary key or links, with a composite
  key (exchange rates by date and currency pair), copied for keyed lookups;
* keyed lookups: a column equal to a reference table's value at the row whose
  key columns equal columns or constants of this row (rate(OrderDate, 'USD',
  CurrencyCode));
* mapped lookups: a code determined by a parent's code through a small map
  (currency of the customer's country);
* views: a table whose rows are exactly another table's rows (same key) and
  whose columns are that table's columns or lookups (sales = order lines joined
  to orders);
* sequences that start at 0 or 1 within a parent (line numbers);
* column offsets: a date that follows another date of the row by a gap taking
  few values (delivery date = order date + 0..7 days);
* implied dimensions: a code column that determines a group of columns of its
  table (geography key -> state, country, continent);
* ratio columns: y = x * f with f one of a few factors (unit price = list price
  x a price factor; net price = unit price x (1 - discount)), and two ratio
  columns that share their factor (unit cost = cost x the same factor);
* rules: an order between two columns of a row (cost below price, start before
  end), a date of the row inside a parent's date window (store open on the
  order date), and equalities between columns of two parents or of the
  row, with one exception member (the store's country is the customer's,
  except the online store; delivery on the order day, except online);
* key order: surrogate keys numbered in the order of a date (order numbers).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .data import canonical_keys
from .expr import names_in
from .profiling import MIN_ROWS, MIN_SHARE, canon, decimals, is_dateish, significant

CATLIKE = ("categorical", "integer", "boolean", "id")


def _lit(v):
    """A constant as an expression literal."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and float(v).is_integer()):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return repr(str(v))                     # a Python string literal (expressions are parsed by ast)


def _subsample(df, n, seed=0):
    """At most n rows of df: a seeded uniform sample in the original row order (the
    first n rows would be a biased slice, ordered by key or date)."""
    if len(df) > n:
        df = df.sample(n, random_state=seed).sort_index()
    return df.reset_index(drop=True)


def _add_months(xs: pd.Series, k) -> pd.Series:
    """xs + DateOffset(months=k) row by row, vectorised: the same calendar rule
    (keep the day, clipped to the last day of the target month; the time of day
    kept)."""
    k = np.asarray(k, dtype=np.int64)
    m0 = xs.dt.year.to_numpy().astype(np.int64) * 12 + xs.dt.month.to_numpy() - 1 + k
    y, m = m0 // 12, m0 % 12 + 1
    first = pd.to_datetime(pd.DataFrame({"year": y, "month": m, "day": 1}))
    last = (first + pd.offsets.MonthEnd(0)).dt.day.to_numpy()
    d = np.minimum(xs.dt.day.to_numpy(), last)
    out = first + pd.to_timedelta(d - 1, unit="D") + (xs - xs.dt.normalize()).to_numpy()
    return out.astype(xs.dtype)


def _perm_chance(ok_fn, a, b, rng, reps=3):
    """Share of rows satisfying ok_fn(a, b) when b is permuted (chance level)."""
    vals = []
    for _ in range(reps):
        vals.append(float(np.mean(ok_fn(a, b[rng.permutation(len(b))]))))
    return float(np.mean(vals))


def _eq_chance(a, b):
    """Expected share of equal pairs when b is permuted: sum over values of
    p_a(v) p_b(v) (the mean of the permutation statistic, exactly)."""
    pa = pd.Series(a, dtype=object).value_counts(normalize=True)
    pb = pd.Series(b, dtype=object).value_counts(normalize=True)
    return float((pa * pb.reindex(pa.index).fillna(0)).sum())


def _const_groups(k, y):
    """Share of groups of 2+ rows (codes k >= 0) in which y takes one value, and
    the rows those groups hold.  O(n log n)."""
    m = k >= 0
    k, y = k[m], y[m]
    if not len(k):
        return 0.0, 0
    o = np.lexsort((y, k))
    ks, ys = k[o], y[o]
    start = np.r_[True, ks[1:] != ks[:-1]]
    gid = np.cumsum(start) - 1
    size = np.bincount(gid)
    chg = np.zeros(len(size), bool)
    np.logical_or.at(chg, gid[1:][~start[1:]], ys[1:][~start[1:]] != ys[:-1][~start[1:]])
    multi = size >= 2
    if not multi.any():
        return 0.0, 0
    return float((~chg[multi]).mean()), int(size[multi].sum())


def fd_rows(x, y, rng):
    """Functional dependency x -> y measured on rows: (share of rows whose y is the
    most common y of their x, the same share with y permuted (chance), rows).
    Row-based, so a code with few large groups (8 countries) is tested on its
    thousands of rows, and singleton groups add nothing (they agree under
    permutation as well)."""
    kx = pd.factorize(pd.Series(x, dtype=object), use_na_sentinel=False)[0].astype(np.int64)
    ky = pd.factorize(pd.Series(y, dtype=object), use_na_sentinel=False)[0].astype(np.int64)
    n = len(kx)
    if n == 0:
        return 0.0, 1.0, 0
    ny = int(ky.max()) + 1

    def agree(a, b):
        # cells (x, y) sorted by x: the largest cell of each x, summed
        u, c = np.unique(a * ny + b, return_counts=True)
        start = np.r_[0, np.flatnonzero(np.diff(u // ny)) + 1]
        return float(np.maximum.reduceat(c, start).sum()) / n
    share = agree(kx, ky)
    chance = float(np.mean([agree(kx, ky[rng.permutation(n)]) for _ in range(3)]))
    return share, chance, n


def fd_accept(share, chance, n):
    """Accept a dependency measured on rows: at least MIN_SHARE, the binomial test
    of Proposition (signif), and the bounded-differences bound for a random
    permutation (one transposition moves the statistic by at most 2/n, so
    P(s - p0 >= t) <= exp(-n t^2 / 2), McDiarmid 1989) at the same alpha."""
    from .profiling import ALPHA
    return (n >= MIN_ROWS and share >= MIN_SHARE and significant(share, chance, n)
            and -n * max(share - chance, 0.0) ** 2 / 2 <= np.log(ALPHA))


class Discovery:
    # ------------------------------------------------------------ helpers
    def _hop(self, t, r, df=None):
        """(parent rows, position of each row's parent) through single link r."""
        df = self.df[t] if df is None else df
        p, pk = r["parent"], self.pks.get(r["parent"])
        if pk is None or p not in self.df:
            return None, None
        pr = self.parent_rows(p, pk, df[r["child_key"]].dropna().unique()).drop_duplicates(pk).reset_index(drop=True)
        ix = pd.Index(canonical_keys(pr[pk]).astype(str))
        pos = ix.get_indexer(canonical_keys(df[r["child_key"]]).astype(str))
        return pr, pos

    @staticmethod
    def _through(pr, pos, c):
        return np.where(pos >= 0, pr[c].to_numpy(dtype=object)[np.maximum(pos, 0)], None)

    def _ctype(self, t, c):
        return self.cols.get(t, {}).get(c, {}).get("type")

    def _modelled(self, t):
        return t in self.names and t not in self.copy and self.role.get(t) != "view"

    def _full(self, t):
        """All rows of a (reference) table."""
        if not hasattr(self, "_full_cache"):
            self._full_cache = {}
        if t not in self._full_cache:
            self._full_cache[t] = self.df[t] if self.whole[t] else self.src.sample(t, self.counts[t])[0]
        return self._full_cache[t]

    # ------------------------------------------------------- reference tables
    def composite_key(self, t, max_cols=3):
        """Smallest combination (2-3 columns, no measures) unique and non-null on the
        sample where independent columns would repeat at least 10 times, and unique
        and non-null on the whole table (read in one query)."""
        df = self.df[t]
        n = len(df)
        if n < 100:
            return None
        cand = []
        for c in df.columns:
            s = df[c]
            if s.isna().any():
                continue
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().all() and not np.allclose(num, np.round(num)):
                continue                     # measures are not key parts
            if 1 < s.nunique() < n:
                cand.append(c)
        cand = cand[:10]
        for k in range(2, max_cols + 1):
            for combo in itertools.combinations(cand, k):
                sub = df[list(combo)]
                if sub.duplicated().any():
                    continue
                space = float(np.prod([sub[c].nunique() for c in combo]))
                if n * (n - 1) / (2 * space) >= 10:
                    # the sample only screens: unique and non-null on every row
                    if not self.whole[t]:
                        full = self.src.columns(t, list(combo))
                        if full.isna().any().any() or full.duplicated().any():
                            continue
                    return list(combo)
        return None

    # ------------------------------------------------------- keyed lookups
    def keyed_lookups(self):
        refs = [R for R in self.names if R in self.copy and self.pks.get(R) is None and self.unique.get(R)]
        if not refs:
            return
        for t in self.names:
            if not self._modelled(t):
                continue
            df = _subsample(self.df[t], 5000)
            keys = self.key_cols(t)
            targets = [c for c, cs in self.cols[t].items() if c not in keys and cs["type"] in ("float", "integer")]
            if not targets or len(df) < MIN_ROWS:
                continue
            for R in refs:
                rdf = self._full(R)
                key = self.unique[R][0]
                rdate = {k: is_dateish(rdf[k]) for k in key}
                rcan = {k: canon(rdf[k], as_date=rdate[k]) for k in key}
                srcs = {}
                for k in key:
                    dom = set(v for v in rcan[k] if v is not None)
                    opts = []
                    for c in df.columns:
                        if self._ctype(t, c) in ("float", "binary", "text", "pii"):
                            continue
                        if is_dateish(df[c]) != rdate[k]:
                            continue
                        v = canon(df[c], as_date=rdate[k])
                        nn = [x for x in v if x is not None]
                        if nn and np.mean([x in dom for x in nn]) >= 0.99:
                            opts.append(("col", c, v))
                    if len(dom) <= 20:
                        for x in sorted(dom, key=str):
                            raw = rdf[k].to_numpy(dtype=object)[list(rcan[k]).index(x)]
                            opts.append(("const", raw, np.array([x] * len(df), dtype=object)))
                    srcs[k] = opts[:12]
                if not all(srcs.values()):
                    continue
                ridx = pd.Index(list(zip(*[rcan[k] for k in key])), tupleize_cols=False)
                keep = ~ridx.duplicated()
                ridx = ridx[keep]
                vcols = [c for c in rdf.columns if c not in key]
                best = {}
                for combo in itertools.islice(itertools.product(*[srcs[k] for k in key]), 600):
                    lk = pd.Index(list(zip(*[o[2] for o in combo])), tupleize_cols=False)
                    pos = ridx.get_indexer(lk)
                    found = pos >= 0
                    if found.mean() < 0.99:
                        continue
                    for vc in vcols:
                        rv = pd.to_numeric(rdf[vc], errors="coerce").to_numpy(dtype=float)[keep]
                        val = np.where(found, rv[np.maximum(pos, 0)], np.nan)
                        for c in targets:
                            y = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                            m = found & np.isfinite(y)
                            if m.sum() < MIN_ROWS:
                                continue
                            eqf = lambda a, b: np.isclose(a, b, rtol=1e-6, atol=1e-9)
                            share = float(np.mean(eqf(val[m], y[m])))
                            if share < MIN_SHARE:
                                continue
                            chance = _perm_chance(eqf, val[m], y[m], self.rng)
                            if not significant(share, chance, int(m.sum())):
                                continue
                            nconst = sum(o[0] == "const" for o in combo)
                            rank = (-share, nconst)
                            if c not in best or rank < best[c][0]:
                                best[c] = (rank, R, vc, combo, share, chance)
                for c, (_, R_, vc, combo, share, chance) in best.items():
                    match = {k: (o[1] if o[0] == "col" else _lit(o[1])) for k, o in zip(key, combo)}
                    self.cols[t][c] = {"type": "lookup", "lookup": {"table": R_, "match": match, "value": vc}}
                    self.review[:] = [x for x in self.review if not x.startswith(f"{t}.{c}: ")]
                    self.ask(f"{t}.{c}: keyed lookup {R_}.{vc} at " + ", ".join(f"{k} = {v}" for k, v in match.items())
                             + f" (equal on {share:.1%} of {len(df):,} sampled rows; {chance:.1%} by chance)")

    # ------------------------------------------------------- mapped lookups
    def mapped_lookups(self):
        for t in self.names:
            if not self._modelled(t):
                continue
            df = self.df[t]
            keys = self.key_cols(t)
            targets = [c for c, cs in self.cols[t].items() if c not in keys and cs["type"] == "categorical"
                       and 1 < df[c].nunique() <= 200]
            if not targets or len(df) < 100:
                continue
            hops = [(r, *self._hop(t, r)) for r in self.simple(t)]
            for c in targets:
                y = canon(df[c])
                best = None
                for r, pr, pos in hops:
                    if pr is None or (pos >= 0).mean() < 0.95:
                        continue
                    ok = pos >= 0
                    for pc in pr.columns:
                        if pc == self.pks.get(r["parent"]) or self._ctype(r["parent"], pc) not in CATLIKE:
                            continue
                        x = canon(self._through(pr, pos, pc))
                        L = len(set(x[ok]))
                        if L < 2 or L > 200 or L > ok.sum() / 20:
                            continue
                        agree, chance, nm = fd_rows(x[ok], y[ok], self.rng)
                        if not fd_accept(agree, chance, nm):
                            continue
                        mp = pd.DataFrame({"x": x[ok], "y": y[ok]}).dropna().groupby("x")["y"].agg(
                            lambda s: s.value_counts().index[0])
                        if all(str(k) == str(v) for k, v in mp.items()):
                            continue               # identity: a plain lookup
                        rank = (L, r["parent"] in self.copy, pc)
                        if best is None or rank < best[0]:
                            best = (rank, f"{r['child_key']}.{pc}", mp, agree, chance, nm, pr, pos, pc)
                if best is None:
                    continue
                _, path, mp, agree, chance, nm, pr, pos, pc = best
                raw = pd.DataFrame({"x": self._through(pr, pos, pc), "y": df[c].to_numpy(dtype=object)}).dropna()
                raw["k"] = canonical_keys(raw["x"]).astype(str)
                m = raw.groupby("k")["y"].agg(lambda s: s.value_counts().index[0])
                self.cols[t][c] = {"type": "lookup", "lookup": path,
                                   "map": {k: (v.item() if hasattr(v, "item") else v) for k, v in m.items()}}
                self.review[:] = [x for x in self.review if not x.startswith(f"{t}.{c}: ")]
                self.ask(f"{t}.{c}: mapped lookup of {path} ({len(m)} values; the map holds on {agree:.1%} of "
                         f"{nm:,} sampled rows, {chance:.1%} by chance)")

    # ------------------------------------------------------- views
    def views(self):
        self.view = {}
        order = sorted([t for t in self.names if t not in self.copy], key=lambda t: -len(self.cols[t]))
        for V in order:
            if V in self.view.values():
                continue
            ukeys = ([self.pk_cols[V]] if self.pk_cols[V] else []) + [u for u in self.unique[V] if u != self.pk_cols[V]]
            done = False
            for U in ukeys:
                for B in self.names:
                    if done or B == V or B in self.view or B in self.copy or self.counts[B] != self.counts[V]:
                        continue
                    bkeys = ([self.pk_cols[B]] if self.pk_cols[B] else []) + self.unique[B]
                    # key columns are paired by name; one column left on each side is
                    # paired with the other (a line number called RowNumber in the base)
                    UB = None
                    for kb_ in bkeys:
                        if not kb_ or len(kb_) != len(U):
                            continue
                        lv, lb = [c for c in U if c not in kb_], [c for c in kb_ if c not in U]
                        if len(lv) <= 1:
                            UB = [c if c in kb_ else lb[0] for c in U]
                            break
                    if UB is None or not all(c in self.df[B].columns for c in UB):
                        continue
                    dv = self.df[V]
                    k0 = next((c for c in U if c in UB), None)
                    if k0 is None:
                        continue
                    ks = self.rng.choice(dv[k0].dropna().unique(), size=min(2000, dv[k0].nunique()), replace=False)
                    sub = (dv[dv[k0].isin(ks)] if self.whole[V] else self.src.rows_in(V, k0, list(ks))).reset_index(drop=True)
                    bro = self.df[B][self.df[B][k0].isin(ks)] if self.whole[B] else self.src.rows_in(B, k0, list(ks))
                    kv = pd.Index(list(zip(*[canon(sub[c]) for c in U])), tupleize_cols=False)
                    kb = pd.Index(list(zip(*[canon(bro[c]) for c in UB])), tupleize_cols=False)
                    if kb.has_duplicates or len(sub) < MIN_ROWS:
                        continue
                    pos = kb.get_indexer(kv)
                    if (pos >= 0).mean() < MIN_SHARE or len(bro) != len(sub):
                        continue
                    okc, why = True, []
                    renamed = {v_: b_ for v_, b_ in zip(U, UB) if v_ != b_}
                    taken = set(UB)

                    def same(c, bc):
                        a = canon(sub[c])
                        b = canon(bro[bc].to_numpy(dtype=object)[np.maximum(pos, 0)])
                        return np.mean(a == b) >= MIN_SHARE
                    for c, cs in self.cols[V].items():
                        if cs["type"] == "lookup" or c in U:
                            continue
                        if c in bro.columns and same(c, c):
                            taken.add(c)
                            continue
                        # the same values under another name (OrderDate stored as DT)
                        alt = [bc for bc in bro.columns if bc not in taken and bc not in self.cols[V] and same(c, bc)]
                        if len(alt) == 1:
                            renamed[c] = alt[0]
                            taken.add(alt[0])
                            continue
                        okc = False
                        why.append(c)
                        break
                    if not okc:
                        continue
                    for c, bc in renamed.items():
                        self.cols[V][c] = {**self.cols[V][c], "base_column": bc}
                    self.view[V] = B
                    self.role[V] = "view"
                    done = True
                    self.unique[V] = []
                    self.rels = [r for r in self.rels if not (r["child"] == V and isinstance(r["child_key"], list))]
                    self.ask(f"{V}: view of {B} (the same {len(sub):,} sampled keys {U}; every column is a column of "
                             f"{B} or a lookup" + (", renamed: " + ", ".join(f"{c} = {B}.{bc}" for c, bc in
                                                                           renamed.items()) if renamed else "")
                             + f"); rebuilt from the synthetic {B}, so it equals the join exactly")

    # ------------------------------------------------------- line keys
    def line_keys(self):
        """Facts read from files carry no declared keys.  A pair (link k, integer
        column c) is a line key when c is distinct within k on complete groups of
        k (at least 99.9% of them, 30 or more with two or more rows) and numbers
        the rows from 0 or 1 without gaps on 95% of them: the pattern of order
        lines, which chance does not produce on that many groups.  The pair is
        added as a unique key; sequences() then decides the line numbering."""
        for t in self.names:
            if not self._modelled(t) or self.role[t] != "fact" or self.pks.get(t) is not None:
                continue
            if any(len(u) == 2 for u in self.unique[t]):
                continue
            fks = [r["child_key"] for r in self.simple(t)]
            df = self.df[t]
            cands = []
            for c in df.columns:
                if c in fks or self._ctype(t, c) not in ("id", "integer"):
                    continue
                v = pd.to_numeric(df[c], errors="coerce")
                if v.isna().any() or not np.allclose(v, np.round(v)) or v.min() not in (0, 1) or v.max() > 10000:
                    continue
                cands.append(c)
            found = None
            for k in fks:
                if found or not cands or df[k].nunique() < MIN_ROWS:
                    continue
                ks = self.rng.choice(df[k].dropna().unique(), size=min(self.n_groups, df[k].nunique()), replace=False)
                grp = df[df[k].isin(ks)] if self.whole[t] else self.src.rows_in(t, k, list(ks))
                kk = canonical_keys(grp[k]).astype(str)
                for c in cands:
                    v = pd.to_numeric(grp[c], errors="coerce")
                    g = pd.DataFrame({"k": kk, "v": v}).groupby("k")["v"]
                    mn, mx, n, nu = g.min(), g.max(), g.size(), g.nunique()
                    multi = n >= 2
                    if multi.sum() < MIN_ROWS:
                        continue
                    distinct = float(np.mean(nu[multi] == n[multi]))
                    nogap = float(np.mean((mx - mn + 1 == n)[multi]))
                    if distinct >= MIN_SHARE and nogap >= 0.95 and mn.mode().iloc[0] in (0, 1):
                        found = [k, c]
                        self.unique[t].insert(0, found)
                        self.ask(f"{t}: unique key {found} inferred ({c} numbers the rows of each {k} without "
                                 f"repeats in {distinct:.1%} of {int(multi.sum())} complete groups)")
                        break

    # ------------------------------------------------------- sequences
    def sequences(self):
        for t in self.names:
            if not self._modelled(t) or self.role[t] != "fact" or t in self.header:
                continue
            fks = {r["child_key"] for r in self.simple(t)}
            for u in self.unique[t]:
                if len(u) != 2:
                    continue
                k = next((c for c in u if c in fks), None)
                c = next((x for x in u if x != k), None)
                if k is None or c is None or c in fks or self._ctype(t, c) not in ("id", "integer", "categorical"):
                    continue
                df = self.df[t]
                ks = self.rng.choice(df[k].dropna().unique(), size=min(self.n_groups, df[k].nunique()), replace=False)
                grp = df[df[k].isin(ks)] if self.whole[t] else self.src.rows_in(t, k, list(ks))
                v = pd.to_numeric(grp[c], errors="coerce")
                if v.isna().any():
                    continue
                g = pd.DataFrame({"k": canonical_keys(grp[k]).astype(str), "v": v}).groupby("k")["v"]
                mn, mx, n, nu = g.min(), g.max(), g.size(), g.nunique()
                if (n >= 2).sum() < MIN_ROWS:
                    continue
                s0 = mn.mode().iloc[0]
                start = float(np.mean(mn == s0))
                distinct = float(np.mean(nu == n))
                nogap = float(np.mean(mx - mn + 1 == n))
                if s0 in (0, 1) and start >= MIN_SHARE and distinct >= MIN_SHARE and nogap >= 0.95:
                    self.cols[t][c] = {"type": "sequence", "within": k, "start": int(s0)}
                    self.review[:] = [x for x in self.review if not x.startswith(f"{t}.{c}: ")]
                    self.ask(f"{t}.{c}: line number {int(s0)}..n-{1 - int(s0)} within {k} (starts at {int(s0)} in "
                             f"{start:.1%} of {len(n)} complete groups; {1 - nogap:.1%} of groups skip numbers, which "
                             "the synthetic data does not)")

    # ------------------------------------------------------- column offsets
    def col_offsets(self):
        self.col_offset = {}
        for t in self.names:
            if not self._modelled(t):
                continue
            df = self.df[t]
            dts = [c for c, cs in self.cols[t].items() if cs["type"] in ("date", "datetime")]
            if len(dts) < 2 or len(df) < MIN_ROWS:
                continue
            d = {c: pd.to_numeric(pd.Series(canon(df[c], as_date=True)), errors="coerce").to_numpy(dtype=float)
                 for c in dts}
            cand = {}
            for a, b in itertools.permutations(dts, 2):
                m = np.isfinite(d[a]) & np.isfinite(d[b])
                if m.sum() < MIN_ROWS:
                    continue
                # b = a + gap is null wherever a is: b must (almost) never be set without a
                if np.mean(np.isfinite(d[a])[np.isfinite(d[b])]) < MIN_SHARE:
                    continue
                gap = d[b][m] - d[a][m]
                if np.mean(gap >= 0) < MIN_SHARE:
                    continue
                u = np.unique(gap)
                if len(u) > 60:
                    continue
                us = set(np.round(u, 6))
                inset = lambda x, y: np.isin(np.round(y - x, 6), list(us))
                chance = _perm_chance(inset, d[a][m], d[b][m], self.rng)
                if not significant(1.0, chance, int(m.sum())):
                    continue
                if b not in cand or len(u) < cand[b][1]:
                    cand[b] = (a, len(u), gap)
            for b in sorted(cand, key=lambda x: cand[x][1]):
                a, nu, gap = cand[b]
                if a in self.col_offset.get(t, {}) or b in {x for x, _ in self.col_offset.get(t, {}).values()}:
                    continue
                self.col_offset.setdefault(t, {})[b] = (a, nu)
                self.cols[t][b]["offset_from"] = a
                u, k = np.unique(gap, return_counts=True)
                top = ", ".join(f"{x:g} ({n / len(gap):.0%})" for x, n in sorted(zip(u, k), key=lambda z: -z[1])[:3])
                self.ask(f"{t}.{b}: offset from {a} (never earlier; {nu} distinct gaps, most common {top}); "
                         f"generated as {a} plus a modelled gap")

    # ------------------------------------------------------- soft rules (ages)
    def soft_rules(self, min_share=0.8):
        """An integer column that counts whole years (or months) from a date of the
        row to one reference date: an age or a tenure measured on the day the data
        were prepared.  Row i allows the reference dates in
        [x_i + y_i years, x_i + (y_i + 1) years); the date covered by most rows is
        found by a sweep over these intervals.  Kept when that share is at least
        min_share and beyond the share obtained with y permuted.  Holding on at
        least MIN_SHARE it is a derive rule; otherwise a soft rule that holds on
        the measured share (a Bayesian mixture of the identity and the observed
        offsets in the engine)."""
        from .expr import _months_between, _years_between
        taken = {(r["table"], r["expr"].split("=")[0].strip()) for r in self.rules
                 if r.get("type") in ("derive", "soft")}
        for t in self.names:
            if not self._modelled(t):
                continue
            df = self.df[t]
            keys = self.key_cols(t)
            dts = [c for c, cs in self.cols[t].items() if cs["type"] in ("date", "datetime")]
            ints = [c for c, cs in self.cols[t].items() if cs["type"] == "integer" and c not in keys
                    and (t, c) not in taken and not cs.get("offset_from") and not cs.get("ratio_of")]
            if not dts or not ints or len(df) < MIN_ROWS:
                continue
            for y in ints:
                yv = pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=float)
                ok_y = np.isfinite(yv)
                if ok_y.sum() < MIN_ROWS or np.nanmin(yv) < 0 or len(np.unique(yv[ok_y])) < 5:
                    continue
                best = None
                for x in dts:
                    # canon gives days since 1970-01-01
                    xd = pd.to_datetime(pd.Series(canon(df[x], as_date=True), dtype=float).floordiv(1),
                                        unit="D", errors="coerce")
                    m = ok_y & xd.notna().to_numpy()
                    if m.sum() < MIN_ROWS:
                        continue
                    for unit, fn, top in (("years", _years_between, 150), ("months", _months_between, 1800)):
                        if np.nanmax(yv[m]) > top:
                            continue
                        xs, ys = xd[m].reset_index(drop=True), yv[m].astype(int)
                        # screen: x + y units must pile up around one date
                        step = 365.2425 if unit == "years" else 30.436875
                        tday = (xs - pd.Timestamp("1970-01-01")).dt.days.to_numpy(dtype=float) + ys * step
                        if np.mean(np.abs(tday - np.median(tday)) <= step + 31) < min_share:
                            continue
                        mk = 12 if unit == "years" else 1
                        lo = _add_months(xs, ys * mk)
                        hi = _add_months(xs, (ys + 1) * mk)
                        ev = pd.concat([pd.Series(1, index=pd.DatetimeIndex(lo)),
                                        pd.Series(-1, index=pd.DatetimeIndex(hi))]).groupby(level=0).sum()
                        cov = ev.sort_index().cumsum()
                        d = cov.idxmax()
                        ref = np.full(len(xs), (d - pd.Timestamp("1970-01-01")).days, dtype=float)
                        f = fn((xs - pd.Timestamp("1970-01-01")).dt.days.to_numpy(dtype=float), ref)
                        share = float(np.mean(f == ys))
                        chance = float(np.mean(f == self.rng.permutation(ys)))
                        if share >= min_share and significant(share, chance, len(ys)) and \
                                (best is None or share > best[0]):
                            best = (share, x, unit, d, chance)
                if best is None:
                    continue
                share, x, unit, d, chance = best
                fn_name = "years_between" if unit == "years" else "months_between"
                rule = {"table": t, "type": "derive", "expr": f"{y} = {fn_name}({x}, '{d:%Y-%m-%d}')"}
                if share < MIN_SHARE:
                    rule["holds"] = round(share, 4)
                    rule["violation"] = "offset"
                self.rules.append(rule)
                taken.add((t, y))
                self.ask(f"{t}.{y}: whole {unit} from {x} to {d:%Y-%m-%d} on {share:.1%} of rows "
                         f"(by chance {chance:.1%}); " + ("derive rule" if share >= MIN_SHARE else
                                                          "soft rule: holds on that share, otherwise "
                                                          "the observed offsets"))

    # ------------------------------------------------------- unknown members
    UNKNOWN_WORDS = frozenset({"unknown", "n/a", "na", "not applicable", "none", "unk", "undefined", "unassigned",
                               "-", "not available", "missing", "?", "no value"})

    def unknown_members(self):
        """Rows of a dimension that stand for 'no member' (key -1 or 0, labels
        'Unknown' / 'N/A') and that other tables reference: a warehouse practice
        (Kimball) that keeps facts' foreign keys non-null.  Such rows are kept out
        of the model, copied, and referenced with the source's share."""
        self.unknown = {}
        for t in self.names:
            pk = self.pks.get(t)
            if not self._modelled(t) or self.role.get(t) != "entity" or not pk or len(self.df[t]) < 10:
                continue
            df = self.df[t]
            kv = canonical_keys(df[pk])
            kn = pd.to_numeric(kv, errors="coerce")
            txt = [c for c, cs in self.cols[t].items() if cs["type"] in ("categorical", "text") and c != pk]
            cand = set()
            if kn.notna().all() and (kn > 0).mean() >= 0.9:
                cand |= set(kv[kn <= 0].astype(str))
            if txt:
                low = df[txt].astype(str).apply(lambda col: col.str.strip().str.lower())
                unk = low.isin(self.UNKNOWN_WORDS) | df[txt].isna()
                frac = unk.mean(axis=1)
                hit = (frac >= 0.5) & low.isin(self.UNKNOWN_WORDS).any(axis=1)
                if hit.sum() <= 3:
                    cand |= set(kv[hit].astype(str))
            if not cand:
                continue
            refs = 0
            for r in self.rels:
                if r["parent"] == t and isinstance(r["child_key"], str) and r["child"] in self.df:
                    refs += int(canonical_keys(self.df[r["child"]][r["child_key"]]).astype(str).isin(cand).sum())
            if refs == 0:
                continue
            raw = [df[pk][kv.astype(str) == c].iloc[0] for c in sorted(cand)]
            raw = [int(x) if isinstance(x, (np.integer, int)) or (isinstance(x, float) and float(x).is_integer())
                   else x for x in raw]
            self.unknown[t] = raw
            self.ask(f"{t}: unknown member{'s' if len(raw) > 1 else ''} {raw} (referenced by {refs:,} rows): "
                     "copied as-is and referenced with the source's share; the model learns the real members only")

    # ------------------------------------------------------- bridge weights
    def bridges(self):
        """A bridge (Kimball: multivalued dimension, e.g. accounts per customer with
        allocation factors): a number column that sums to the same total (1 or 100)
        over the rows of each value of a foreign key, on at least MIN_SHARE of the
        groups with 2+ rows."""
        self.bridge = {}
        for t in self.names:
            if not self._modelled(t) or len(self.df[t]) < 50:
                continue
            df = self.df[t]
            fks = [r["child_key"] for r in self.simple(t)]
            nums = [c for c, cs in self.cols[t].items() if cs["type"] == "float" and c not in self.key_cols(t)]
            found = None
            for g in fks:
                key = canonical_keys(df[g]).astype(str)
                multi = key.map(key.value_counts()) >= 2
                if key[multi].nunique() < 10:
                    continue
                for c in nums:
                    v = pd.to_numeric(df[c], errors="coerce")
                    tot = v[multi].groupby(key[multi]).sum()
                    for total in (1.0, 100.0):
                        share = float(np.isclose(tot, total, rtol=0, atol=10.0 ** -(decimals(df[c]) or 4) * 2).mean())
                        if share >= MIN_SHARE:
                            found = {"group": g, "weight": c, **({"total": total} if total != 1.0 else {})}
                            d = decimals(df[c])
                            if d is not None:
                                found["decimals"] = int(d)
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                self.bridge[t] = found
                self.ask(f"{t}: bridge: {found['weight']} sums to {found.get('total', 1.0):g} over the rows of each "
                         f"{found['group']}; synthetic weights are normalised per group")

    # ------------------------------------------------------- slowly changing dimensions
    def scd(self, min_multi=5):
        """Type 2 slowly changing dimensions: an entity whose rows are versions of a
        natural key (a code repeated on a few rows), each version valid from a date,
        with distinct start dates within a key, and at most one open version (a null
        end date) per key, the latest one.  Kept when this holds on at least
        MIN_SHARE of the keys and at least min_multi keys have 2+ versions."""
        self.scd_info = {}
        start_w = ("start", "from", "valid", "effective", "begin")
        end_w = ("end", "to", "expir", "until", "thru")
        for t in self.names:
            if not self._modelled(t) or self.role.get(t) != "entity" or not self.pks.get(t):
                continue
            df = self.df[t]
            n = len(df)
            if n < 20:
                continue
            keys = self.key_cols(t)
            dates = [c for c, cs in self.cols[t].items() if cs["type"] in ("date", "datetime")]
            if not dates:
                continue
            best = None
            for nk, cs in self.cols[t].items():
                if nk in keys or cs["type"] not in ("categorical", "id", "pii", "integer"):
                    continue
                kv = canonical_keys(df[nk])
                if kv.isna().mean() > 0.01:
                    continue
                sizes = kv.value_counts()
                if not (0.3 * n <= len(sizes) < n) or sizes.max() > 50 or (sizes >= 2).sum() < min_multi:
                    continue
                k = kv.astype(str).to_numpy()
                for vf in dates:
                    sv = pd.to_datetime(df[vf], errors="coerce", format="mixed")
                    if sv.isna().mean() > 0.01:
                        continue
                    fr = pd.DataFrame({"k": k, "s": sv})
                    dup = fr.duplicated(["k", "s"]).sum()
                    if dup > (1 - MIN_SHARE) * n:
                        continue
                    # an end date with open versions is required: repeated codes with a
                    # start date alone (addresses, birth dates) are not versions
                    for vt in [d for d in dates if d != vf]:
                        if vt is not None:
                            ev = pd.to_datetime(df[vt], errors="coerce", format="mixed")
                            fr["open"] = ev.isna().to_numpy()
                            if not fr["open"].any():
                                continue
                            o = fr.sort_values(["k", "s"])
                            last = ~o.duplicated("k", keep="last")
                            # open versions: at most one per key, and it is the latest
                            per = o.groupby("k")["open"].sum()
                            ok_keys = float((per <= 1).mean())
                            open_last = float((o["open"] & last).sum() / max(o["open"].sum(), 1))
                            if ok_keys < MIN_SHARE or open_last < MIN_SHARE:
                                continue
                        sc = (sum(w in vf.lower() for w in start_w) + (sum(w in vt.lower() for w in end_w) if vt else 0),
                              vt is not None, int((sizes >= 2).sum()))
                        if best is None or sc > best[0]:
                            best = (sc, nk, vf, vt, sizes)
            if best is None:
                continue
            _, nk, vf, vt, sizes = best
            info = {"type": 2, "natural_key": nk, "valid_from": vf}
            if vt:
                info["valid_to"] = vt
            self.scd_info[t] = info
            if self.cols[t][nk]["type"] != "id":
                self.cols[t][nk] = {"type": "id"}
            self.ask(f"{t}: slowly changing dimension (type 2): {len(sizes)} values of {nk} with up to "
                     f"{int(sizes.max())} versions ({int((sizes >= 2).sum())} with 2+), valid from {vf}"
                     + (f" to {vt} (open versions are the latest)" if vt else "")
                     + "; versions are synthesised per natural key")

    # ------------------------------------------------------- implied dimensions
    def dimensions(self):
        self.dims = {}
        for t in self.names:
            if not self._modelled(t) or self.role[t] != "entity" or t in self.calendar:
                continue
            df = self.df[t]
            n = len(df)
            if n < 100:
                continue
            keys = self.key_cols(t)
            # a column given by a derive rule (reorder point = 0.75 x safety stock) keeps its formula
            derived = {r["expr"].split("=")[0].strip() for r in getattr(self, "rules", [])
                       if r.get("table") == t and r.get("type") == "derive"}
            usable = [c for c, cs in self.cols[t].items() if c not in keys and c not in derived
                      and cs["type"] in ("categorical", "integer", "boolean") and not cs.get("offset_from")
                      and df[c].nunique() >= 2]
            levels = {c: df[c].nunique() for c in usable}
            codes = {c: canon(df[c]) for c in usable}
            det = {}
            for x in usable:
                if not (2 <= levels[x] <= n / 5) or self._ctype(t, x) == "boolean":
                    continue
                ys = []
                for y in usable:
                    if y == x or levels[y] > levels[x]:
                        continue
                    agree, chance, nm = fd_rows(codes[x], codes[y], self.rng)
                    if fd_accept(agree, chance, nm):
                        ys.append(y)
                det[x] = ys
            pool = set(usable)
            for x in sorted(det, key=lambda c: (-len(det[c]), -levels[c])):
                ys = [y for y in det[x] if y in pool]
                if x not in pool or not ys:
                    continue
                big = levels[x] > 200 or any(levels[y] > 200 for y in ys)
                ints = any(self._ctype(t, c) == "integer" for c in [x] + ys)
                if len(ys) < 2 and not big and not ints:
                    continue
                d = {"key": x, "columns": ys}
                # anchor: the finest determined label with at most 50 levels stays a
                # modelled attribute (other columns keep their dependence on it)
                lab = [y for y in ys if self._ctype(t, y) == "categorical" and 2 <= levels[y] <= 50]
                if not lab and not big:
                    # without a label to anchor on, a link would leave the code's dependence on
                    # the other columns to the segments alone (resellers' size and business
                    # type).  A code the marginal model keeps exactly (a category, or an integer
                    # with at most 30 values) anchors itself: it stays modelled and the columns
                    # it determines are read from the dimension by its value
                    if not (self._ctype(t, x) == "categorical" or levels[x] <= 30):
                        continue
                    lab = None
                    d["anchor"] = x
                if lab:
                    d["anchor"] = max(lab, key=lambda y: (levels[y], -ys.index(y)))
                self_anchored = d.get("anchor") == x
                self.dims.setdefault(t, []).append(d)
                pool -= set(ys) | {x}
                for c in [x] + ys:          # a dimension code is not a foreign key to look for
                    self.review[:] = [r for r in self.review if not r.startswith(f"ASK {t}.{c}: looks like a foreign key")]
                self.ask(f"{t}: {x} ({levels[x]} values) determines {', '.join(ys)} on at least {MIN_SHARE:.1%} of "
                         f"the sampled rows; generated through a copied dimension of their observed combinations"
                         + (f"; {x} stays modelled with the other columns and the others are read by its value"
                            if self_anchored else
                            f", {d['anchor']} modelled with the other columns and {x} drawn within it"
                            if d.get("anchor") else ""))
            self._link_dimensions(t, df, pool)

    def _link_dimensions(self, t, df, pool):
        """A link that determines another link of the same row (a track's album fixes
        its genre): the second is read from the observed pairs, keyed by the first,
        instead of being drawn independently given the row's segment.  Only links to
        reference (copied) tables are read this way, and only when the dependency is
        one-way (album -> genre, not a one-to-one code pair)."""
        n = len(df)
        links = [r for r in self.rels if r["child"] == t and not isinstance(r["child_key"], list)
                 and r["parent"] != t and r["child_key"] in df.columns]
        if len(links) < 2:
            return
        lv = {r["child_key"]: df[r["child_key"]].nunique() for r in links}
        codes = {r["child_key"]: canon(df[r["child_key"]]) for r in links}
        taken = {c for d in self.dims.get(t, []) for c in [d["key"]] + d["columns"]}
        for rx in sorted(links, key=lambda r: -lv[r["child_key"]]):
            x = rx["child_key"]
            if x in taken or not (2 <= lv[x] <= n / 5):
                continue
            ys = []
            for ry in links:
                y = ry["child_key"]
                if y == x or y in taken or ry["parent"] not in self.copy or not (2 <= lv[y] < lv[x]):
                    continue
                agree, chance, nm = fd_rows(codes[x], codes[y], self.rng)
                back = fd_rows(codes[y], codes[x], self.rng)
                if fd_accept(agree, chance, nm) and not fd_accept(*back):
                    ys.append(y)
            if not ys:
                continue
            self.dims.setdefault(t, []).append({"key": x, "columns": ys, "anchor": x})
            taken |= {x, *ys}
            pool -= {x, *ys}
            self.ask(f"{t}: link {x} ({lv[x]} values) determines {', '.join(ys)} on at least {MIN_SHARE:.1%} of the "
                     f"sampled rows; {x} is drawn as a link and {', '.join(ys)} read from its observed pairs")

    # ------------------------------------------------- labels nested in a code
    def nested_labels(self, max_levels=200, min_share=0.8):
        """A categorical column with more levels than the marginal model keeps (cities)
        whose values mostly sit inside one value of a coarser column of the row (the
        city's state) is generated given that column: the Dirichlet posterior
        predictive of the label within its group, backed off to the overall labels
        (text model on whole values; labels held by fewer than min_token_rows rows are
        never released).  Measured on the rows whose label occurs at least twice in
        the sample (a label seen once agrees with any group) and tested against the
        permutation chance like a functional dependency."""
        self.nested = {}
        for t in self.names:
            if not self._modelled(t):
                continue
            df = self.df[t]
            keys = self.key_cols(t)
            dimcols = {c for d in getattr(self, "dims", {}).get(t, []) for c in d["columns"] + [d["key"]]}
            derived = {r["expr"].split("=")[0].strip() for r in getattr(self, "rules", [])
                       if r.get("table") == t and r.get("type") == "derive"}
            for c, cs in self.cols[t].items():
                if cs["type"] != "categorical" or c in keys or c in dimcols or c in derived:
                    continue
                lc = df[c].nunique()
                if lc <= max_levels:
                    continue
                vc = df[c].value_counts()
                held = df[c].map(vc).fillna(0).to_numpy()
                rep = held >= 2
                # labels: most sampled rows hold a label another sampled row shares (a code
                # or alternate key that repeats only across versions of one row is an identifier)
                if rep.sum() < MIN_ROWS or np.mean(rep[df[c].notna().to_numpy()]) < 0.5:
                    continue
                best = None
                for k, ks in self.cols[t].items():
                    if k == c or ks["type"] not in ("categorical", "integer") or ks.get("offset_from"):
                        continue
                    lk = df[k].nunique()
                    if not (2 <= lk <= lc / 5) or df[k].isna().mean() > 0.5:
                        continue
                    share, chance, n = fd_rows(canon(df[c])[rep], canon(df[k])[rep], self.rng)
                    if share >= min_share and significant(share, chance, n) and \
                            (best is None or (lk, share) > (best[1], best[2])):
                        best = (k, lk, share, chance, n)
                if best is None:
                    continue
                k, lk, share, chance, n = best
                cs["type"] = "text"
                cs["text"] = {"condition_on": k, "whole_values": True}
                self.nested[(t, c)] = k
                self.ask(f"{t}.{c}: {lc:,} labels, {share:.1%} of {n:,} sampled rows with a repeated label sit in the "
                         f"label's most common {k} ({chance:.1%} by chance); generated within {k} from the labels "
                         f"held by enough rows (text.whole_values)")

    # ------------------------------------------------------- ratio columns
    def ratios(self):
        derived = {(r["table"], r["expr"].split("=")[0].strip()) for r in self.rules if r.get("type") == "derive"}
        reads = {}
        for r in self.rules:
            if r.get("type") == "derive":
                lhs, rhs = r["expr"].split("=", 1)
                reads.setdefault((r["table"], lhs.strip()), set()).update(names_in(rhs))

        def computed_from(t, x, y, seen=()):
            # x is computed (through derive rules of t) from y: y = x * f would be a cycle
            for n in reads.get((t, x), ()):
                if n == y or (n not in seen and computed_from(t, n, y, seen + (n,))):
                    return True
            return False
        for t in self.names:
            if not self._modelled(t):
                continue
            df = _subsample(self.df[t], 20000)
            keys = self.key_cols(t)
            ys = [c for c, cs in self.cols[t].items() if cs["type"] == "float" and c not in keys
                  and (t, c) not in derived]
            if not ys or len(df) < 100:
                continue
            xs = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c, cs in self.cols[t].items()
                  if c not in keys and cs["type"] in ("float", "integer")}
            for r in self.simple(t):
                pr, pos = self._hop(t, r, df)
                if pr is None:
                    continue
                for pc in pr.columns:
                    if pc != self.pks.get(r["parent"]) and self._ctype(r["parent"], pc) in ("float", "integer"):
                        xs[f"{r['child_key']}.{pc}"] = pd.to_numeric(pd.Series(self._through(pr, pos, pc)),
                                                                    errors="coerce").to_numpy(dtype=float)
            cands = []
            for y in ys:
                yv = xs.get(y, pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=float))
                dd = decimals(df[y])
                dd = 9 if dd is None else dd
                tol = 0.5 * 10.0 ** -dd + 1e-12
                for x, xv in xs.items():
                    if x == y or computed_from(t, x, y):
                        continue
                    m = np.isfinite(xv) & np.isfinite(yv) & (xv != 0)
                    if m.sum() < 100:
                        continue
                    r_ = yv[m] / xv[m]
                    for k in range(2, 7):
                        lv = np.unique(np.round(r_, k))
                        if len(lv) > 50:
                            break
                        rep_ = np.isclose(np.round(xv[m] * np.round(r_, k), dd), yv[m], atol=tol)
                        if rep_.mean() < MIN_SHARE:
                            continue
                        if len(lv) == 1:
                            break                      # one factor: a derive rule, not a ratio
                        fl = lv

                        def reproduced(a, b, fl=fl):
                            return np.isclose(np.round(a[:, None] * fl[None, :], dd), b[:, None], atol=tol).any(1)
                        sub = self.rng.permutation(int(m.sum()))[:3000]
                        chance = _perm_chance(reproduced, xv[m][sub], yv[m][sub], self.rng, reps=2)
                        if significant(float(rep_.mean()), chance, int(m.sum())):
                            cands.append((y, (len(lv), "." not in x, x, k, float(rep_.mean()), chance)))
                        break
            # one base per column, without cycles: bases that are not ratio columns
            # themselves (list prices, costs) first, then ratios of ratios
            targets = {y for y, _ in cands}
            found = {}

            def reaches(a, b, seen=()):
                # a is computed from b through the ratios chosen so far or derive rules
                nxt = set(reads.get((t, a), ())) | ({found[a][2]} if a in found else set())
                return any(n == b or (n not in seen and reaches(n, b, seen + (n,))) for n in nxt)
            for y, c in sorted(cands, key=lambda z: (z[1][2] in targets, z[1][:2], z[0])):
                if y in found or reaches(c[2], y) or c[2] == y:
                    continue
                found[y] = c
            own = {}
            for y in sorted(found, key=lambda c: list(self.cols[t]).index(c)):
                nl, _, x, k, share, chance = found[y]
                self.cols[t][y]["ratio_of"] = x
                yv = pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=float)
                fy = np.round(yv / xs[x], k)
                same = None
                for y0, (_x0, _k0, f0) in own.items():
                    dd = decimals(df[y]) or 0
                    m = np.isfinite(f0) & np.isfinite(xs[x])
                    if m.sum() >= 100 and np.mean(np.isclose(np.round(xs[x][m] * f0[m], dd), yv[m],
                                                             atol=0.5 * 10.0 ** -dd + 1e-12)) >= MIN_SHARE:
                        same = y0
                        break
                if same:
                    self.cols[t][y]["ratio_same_as"] = same
                    self.ask(f"{t}.{y}: ratio of {x} with the factor of {same} ({share:.1%} of rows reproduced)")
                else:
                    own[y] = (x, k, fy)
                    self.ratio_k = getattr(self, "ratio_k", {})
                    self.ratio_k[(t, y)] = k
                    self.ask(f"{t}.{y}: ratio of {x}: {y} = {x} x one of {nl} factors (to {k} decimals; reproduces "
                             f"{share:.1%} of rows to the stored precision, {chance:.1%} by chance); the factor is "
                             "modelled, the value derived")

    # ------------------------------------------------------- group effects
    def group_effects(self, max_cols=2, screen_rows=60000):
        """A column whose value is shared by all rows of a group defined by one or
        two other columns (own columns or parent columns one link away), as a daily
        price factor per product category: (group) -> value on at least MIN_SHARE
        of the rows in groups of 2+ rows, beyond chance (rows permuted), with
        groups of 2+ rows holding at least a fifth of the rows.  Minimal: no single
        grouping column and no single link (a parent attribute) determines it.
        The value is then drawn once per group (a shared random effect)."""
        self.group_fx = {}
        derived = {(r["table"], r["expr"].split("=")[0].strip()) for r in self.rules if r.get("type") == "derive"}
        for t in self.names:
            if not self._modelled(t) or self.role.get(t) != "fact" or len(self.df[t]) < 1000:
                continue
            df = self.df[t]
            if len(df) > screen_rows:
                df = df.iloc[np.sort(self.rng.choice(len(df), screen_rows, replace=False))]
            df = df.reset_index(drop=True)
            n = len(df)
            keys = self.key_cols(t)
            hc = set(self.header.get(t, {}).get("columns", []))
            # targets: categorical or integer attributes, and the factor of a ratio column
            tv = {}
            for c, cs in self.cols[t].items():
                if c in keys or c in hc or (t, c) in derived or cs.get("offset_from") or cs["type"] == "lookup":
                    continue
                if cs.get("ratio_of") and not cs.get("ratio_same_as"):
                    x = cs["ratio_of"]
                    if "." in x:
                        r = next((r for r in self.simple(t) if r["child_key"] == x.split(".")[0]), None)
                        pr, pos = self._hop(t, r, df) if r is not None else (None, None)
                        if pr is None:
                            continue
                        xv = pd.to_numeric(pd.Series(self._through(pr, pos, x.split(".", 1)[1])), errors="coerce")
                    else:
                        xv = pd.to_numeric(df[x], errors="coerce")
                    k = getattr(self, "ratio_k", {}).get((t, c), 4)
                    f = (pd.to_numeric(df[c], errors="coerce") / xv.replace(0, np.nan)).round(k)
                    tv[c] = f.to_numpy(dtype=object)
                elif cs["type"] in ("categorical", "integer", "boolean") and not cs.get("ratio_of"):
                    tv[c] = df[c].to_numpy(dtype=object)
            tv = {c: v for c, v in tv.items() if 2 <= pd.Series(v).nunique() <= n / 10}
            if not tv:
                continue
            # grouping columns: own dates and codes, and parent columns one link away
            gv = {}
            for c, cs in self.cols[t].items():
                if c not in keys and cs["type"] in ("date", "datetime", "categorical", "boolean"):
                    gv[c] = df[c].to_numpy(dtype=object)
            links = {}
            for r in self.simple(t):
                links[r["child_key"]] = df[r["child_key"]].to_numpy(dtype=object)
                pr, pos = self._hop(t, r, df)
                if pr is None:
                    continue
                pk = self.pks.get(r["parent"])
                for pc in pr.columns:
                    ty = self._ctype(r["parent"], pc)
                    if pc != pk and ty in ("date", "datetime", "categorical", "boolean", "integer", "id"):
                        v = self._through(pr, pos, pc)
                        if 2 <= pd.Series(v).nunique() <= n / 2:
                            gv[f"{r['child_key']}.{pc}"] = v
            codes = {g: pd.factorize(pd.Series(v, dtype=object), use_na_sentinel=True)[0] for g, v in gv.items()}

            def holds(cols, y):
                k = codes[cols[0]].astype(np.int64)
                ok = k >= 0
                for c2 in cols[1:]:
                    k2 = codes[c2]
                    ok &= k2 >= 0
                    k = k * (int(k2.max()) + 2) + k2
                k = np.where(ok, k, -1)
                size = pd.Series(k).map(pd.Series(k).value_counts()).to_numpy()
                m = ok & (size >= 2)
                if m.sum() < max(MIN_ROWS, n / 5):
                    return None
                return fd_rows(k[m], y[m], self.rng)
            sub = self.rng.permutation(n)[:min(n, 10000)]
            pending = {}
            for y, yv in tv.items():
                ycode = pd.factorize(pd.Series(yv, dtype=object), use_na_sentinel=False)[0]
                single = [lk for lk, kv in links.items() if fd_rows(kv, yv, self.rng)[0] >= MIN_SHARE]
                if single:
                    continue
                ok1 = [g for g in gv if _const_groups(codes[g][sub].astype(np.int64), ycode[sub])[0] >= 0.99
                       and (lambda r_: r_ is not None and fd_accept(*r_))(holds([g], yv))]
                if not ok1:
                    pending[y] = ycode[sub]
            if not pending:
                continue
            # pairs of grouping columns: the sort by the pair's key is shared by all targets
            # (a group is constant iff neighbours in key order agree), screened on a subsample
            best = {}
            names = list(gv)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    ci, cj = codes[names[i]][sub].astype(np.int64), codes[names[j]][sub].astype(np.int64)
                    kk = np.where((ci >= 0) & (cj >= 0), ci * (int(cj.max()) + 2) + cj, -1)
                    m = kk >= 0
                    o = np.flatnonzero(m)[np.argsort(kk[m], kind="stable")]
                    if len(o) < 2:
                        continue
                    ks = kk[o]
                    first = np.r_[True, ks[1:] != ks[:-1]]
                    gid = np.cumsum(first) - 1
                    size = np.bincount(gid)
                    multi = size >= 2
                    if size[multi].sum() < len(sub) / 5:
                        continue
                    inner = ~first[1:]
                    for y, ys_ in pending.items():
                        ys = ys_[o]
                        bad = np.bincount(gid[1:][inner & (ys[1:] != ys[:-1])], minlength=len(size)) > 0
                        if float((~bad[multi]).mean()) < 0.99:
                            continue
                        r_ = holds([names[i], names[j]], tv[y])
                        if r_ is None or not fd_accept(*r_):
                            continue
                        ng = len(set(zip(codes[names[i]], codes[names[j]])))
                        if y not in best or ng < best[y][0]:      # the coarsest grouping
                            best[y] = (ng, [names[i], names[j]], r_)
            for y in pending:
                if y not in best:
                    continue
                ng, by, (share, chance, nr) = best[y]
                self.group_fx.setdefault(t, []).append({"column": y, "by": by})
                what = f"the factor of {y}" if self.cols[t][y].get("ratio_of") else y
                self.ask(f"{t}.{y}: {what} is one value per ({', '.join(by)}) on {share:.2%} of rows in groups of 2+ "
                         f"(by chance {chance:.1%}); group effect: one value drawn per group")

    # ------------------------------------------------------- link affinity
    def link_affinities(self, max_rows=20000):
        """Two links of a row that point at parents sharing a value (customer and
        store in the same area) more often than chance, without a rule: the share of
        matching rows s against the share with one side permuted, kept when
        significant, at least twice chance and below MIN_SHARE (a rule otherwise).
        Columns compared: same name, or at least half of the values shared.  Per
        pair of links the finest such column (most values) is kept; coarser ones
        follow from it.  The link to the smaller parent is the one moved."""
        self.affinity = {}
        in_rule = [names_in(r["expr"]) for r in self.rules if r.get("type") == "check"]
        for t in self.names:
            if not self._modelled(t):
                continue
            df = self.df[t]
            if len(df) > max_rows:
                df = df.iloc[np.sort(self.rng.choice(len(df), max_rows, replace=False))]
            df = df.reset_index(drop=True)
            if len(df) < 200:
                continue
            links = [r for r in self.simple(t) if self._ctype(t, r["child_key"]) != "lookup"
                     and r["parent"] not in self.calendar]
            vals = {}
            for r in links:
                pr, pos = self._hop(t, r, df)
                if pr is None:
                    continue
                pk = self.pks.get(r["parent"])
                for c in pr.columns:
                    if c == pk or self._ctype(r["parent"], c) not in ("categorical", "integer", "id"):
                        continue
                    v = pd.Series(self._through(pr, pos, c), dtype=object)
                    if v.nunique() >= 3 and not is_dateish(v.dropna().astype(str).head(200)):
                        k = canonical_keys(pd.to_numeric(v, errors="coerce") if v.dropna().map(
                            lambda x: isinstance(x, (int, float, np.integer, np.floating))).all() else v)
                        vals[(r["child_key"], c)] = (r, k.astype(object).where(k.notna(), None).astype(str)
                                                     .where(k.notna(), None))
            for i, r1 in enumerate(links):
                for r2 in links[i + 1:]:
                    if r1["parent"] == r2["parent"]:
                        continue
                    best = None
                    for (k1, c1), (_, v1) in vals.items():
                        if k1 != r1["child_key"]:
                            continue
                        for (k2, c2), (_, v2) in vals.items():
                            if k2 != r2["child_key"]:
                                continue
                            p1, p2 = f"{k1}.{c1}", f"{k2}.{c2}"
                            if any({p1, p2} <= u for u in in_rule):
                                continue
                            s1, s2 = set(v1.dropna()), set(v2.dropna())
                            if c1 != c2 and len(s1 & s2) < 0.5 * min(len(s1), len(s2)):
                                continue
                            a, b = v1.to_numpy(dtype=object), v2.to_numpy(dtype=object)
                            m = pd.notna(a) & pd.notna(b)
                            if m.sum() < 200:
                                continue
                            share = float(np.mean(a[m] == b[m]))
                            chance = _eq_chance(a[m], b[m])
                            if share >= MIN_SHARE or share < 2 * chance or share - chance < 0.01 or \
                                    not significant(share, chance, int(m.sum())):
                                continue
                            nl = (c1 == c2, len(s1 | s2))       # same-named columns first, then the finest
                            if best is None or nl > best[0]:
                                best = (nl, p1, p2, share, chance)
                    if best is None:
                        continue
                    _, p1, p2, share, chance = best
                    # the link to the smaller parent is moved (stores for customers)
                    if self.counts.get(r1["parent"], 0) < self.counts.get(r2["parent"], 0):
                        p1, p2 = p2, p1
                    self.affinity.setdefault(t, []).append({"match": [p1, p2]})
                    self.ask(f"{t}: {p1} = {p2} on {share:.1%} of rows (by chance {chance:.1%}); link affinity: "
                             f"{p2.split('.')[0]} is moved to a matching parent on the extra share")

    # ------------------------------------------------------- cross-table rules
    def cross_rules(self):
        for t in self.names:
            if not self._modelled(t):
                continue
            df = _subsample(self.df[t], 20000)
            if len(df) < 100:
                continue
            keys = self.key_cols(t)
            hops = {r["child_key"]: (r, *self._hop(t, r, df)) for r in self.simple(t)
                    if self._ctype(t, r["child_key"]) != "lookup"}
            hops = {k: v for k, v in hops.items() if v[1] is not None and (v[2] >= 0).mean() >= 0.95}
            # (c) order between two columns of the row (cost below price, start before end)
            skip = {c for c, cs in self.cols[t].items() if cs.get("ratio_of") or cs.get("offset_from")} | \
                {r["expr"].split("=")[0].strip() for r in self.rules if r.get("table") == t and r.get("type") == "derive"}
            ords = {}
            for c, cs in self.cols[t].items():
                if c in keys or c in skip or cs["type"] not in ("integer", "float", "date", "datetime"):
                    continue
                isd = cs["type"] in ("date", "datetime")
                v = pd.to_numeric(pd.Series(canon(df[c], as_date=isd)), errors="coerce").to_numpy(dtype=float)
                if np.isfinite(v).sum() >= 100 and len(np.unique(v[np.isfinite(v)])) > 2:
                    ords[c] = (v, isd)
            names_o = list(ords)
            for i, a in enumerate(names_o):
                for b in names_o[i + 1:]:
                    (va, da), (vb, db) = ords[a], ords[b]
                    if da != db:
                        continue
                    m = np.isfinite(va) & np.isfinite(vb)
                    if m.sum() < 100:
                        continue
                    for lo_, hi_, x, y in ((a, b, va[m], vb[m]), (b, a, vb[m], va[m])):
                        le = lambda p_, q_: p_ <= q_
                        share = float(np.mean(le(x, y)))
                        if share < MIN_SHARE:
                            continue
                        chance = _perm_chance(le, x, y, self.rng)
                        if not significant(share, chance, int(m.sum())):
                            continue
                        op = "<" if float(np.mean(x < y)) >= share else "<="
                        expr = f"days({lo_}) {op} days({hi_})" if da else f"{lo_} {op} {hi_}"
                        self.rules.append({"table": t, "type": "check", "expr": expr})
                        self.ask(f"{t}: check {expr} (holds on {share:.1%} of {int(m.sum()):,} sampled rows, "
                                 f"{chance:.1%} by chance)")
                        break
            if not hops:
                continue
            offs = set(getattr(self, "col_offset", {}).get(t, {}))
            dates = [c for c, cs in self.cols[t].items() if cs["type"] in ("date", "datetime") and c not in offs]
            # (a) dates inside a parent's date window
            for d in dates:
                dv = pd.to_numeric(pd.Series(canon(df[d], as_date=True)), errors="coerce").to_numpy(dtype=float)
                for fk, (r, pr, pos) in hops.items():
                    for pc in pr.columns:
                        if self._ctype(r["parent"], pc) not in ("date", "datetime"):
                            continue
                        pv = pd.to_numeric(pd.Series(canon(self._through(pr, pos, pc), as_date=True)),
                                           errors="coerce").to_numpy(dtype=float)
                        m = np.isfinite(dv) & np.isfinite(pv)
                        if m.sum() < MIN_ROWS:
                            continue
                        for lo, (a, b) in (("lower", (pv[m], dv[m])), ("upper", (dv[m], pv[m]))):
                            le = lambda x, y: x <= y
                            share = float(np.mean(le(a, b)))
                            if share < MIN_SHARE:
                                continue
                            chance = _perm_chance(le, a, b, self.rng)
                            if not significant(share, chance, int(m.sum())):
                                continue
                            strict = float(np.mean(a < b)) >= share
                            op = "<" if strict else "<="
                            expr = (f"days({fk}.{pc}) {op} days({d})" if lo == "lower" else
                                    f"days({d}) {op} days({fk}.{pc})")
                            self.rules.append({"table": t, "type": "check", "expr": expr})
                            self.ask(f"{t}: check {expr} (holds on {share:.1%} of {int(m.sum()):,} sampled rows with "
                                     f"both dates, {chance:.1%} by chance)")
            # (b) equalities with one exception member
            cols = {}
            for c, cs in self.cols[t].items():
                if c in keys or cs["type"] in ("float", "text", "pii", "binary", "constant", "lookup") \
                        or (cs["type"] == "lookup" and not is_dateish(df[c])):
                    continue
                cols[c] = (None, canon(df[c], as_date=is_dateish(df[c])), cs["type"] in ("date", "datetime"))
            for fk, (r, pr, pos) in hops.items():
                for pc in pr.columns:
                    pt = self._ctype(r["parent"], pc)
                    if pc == self.pks.get(r["parent"]) or pt in ("float", "text", "pii", "binary", "constant", None):
                        continue
                    v = self._through(pr, pos, pc)
                    isd = pt in ("date", "datetime")
                    cols[f"{fk}.{pc}"] = (fk, canon(v, as_date=isd), isd)
            exc_fks = [fk for fk, (r, _, _) in hops.items() if r["parent"] in self.copy]
            done_pairs = set()
            names = list(cols)
            # integer codes over one dictionary of every column's values: equal values
            # get equal codes (Python equality, as for the object arrays), None gets -1
            if names:
                allc = pd.factorize(pd.Series(np.concatenate([cols[c][1] for c in names]), dtype=object),
                                    use_na_sentinel=True)[0]
                ncode = int(allc.max()) + 1
                code = {c: allc[i * len(df):(i + 1) * len(df)] for i, c in enumerate(names)}
            kv_of = {}
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    fa, va, da = cols[a]
                    fb, vb, db = cols[b]
                    if da != db or (fa is not None and fa == fb) or (fa is None and fb is None and not da):
                        continue
                    pair = (fa or a, fb or b)
                    if pair in done_pairs:
                        continue
                    ca, cb = code[a], code[b]
                    m = (ca >= 0) & (cb >= 0)
                    if m.sum() < 100:
                        continue
                    sa = np.bincount(ca[m], minlength=ncode) > 0
                    sb = np.bincount(cb[m], minlength=ncode) > 0
                    n_ab, n_a, n_b = int((sa & sb).sum()), int(sa.sum()), int(sb.sum())
                    if n_ab < 0.3 * min(n_a, n_b) or n_a + n_b - n_ab < 2:
                        continue
                    eq = ca == cb
                    share = float(eq[m].mean())
                    if share >= MIN_SHARE and fa is not None and fb is not None and fa != fb:
                        # the same value through two different links (the store's country is
                        # the customer's): a rule between the links, not a lookup
                        chance = _eq_chance(va[m], vb[m])
                        if fd_accept(share, chance, int(m.sum())):
                            ea, eb = (f"days({a})", f"days({b})") if da else (a, b)
                            self.rules.append({"table": t, "type": "check", "expr": f"{ea} == {eb}"})
                            done_pairs.add(pair)
                            self.ask(f"{t}: check {ea} == {eb} (equal on {share:.2%} of {int(m.sum()):,} sampled "
                                     f"rows, {chance:.1%} by chance)")
                        continue
                    if share >= MIN_SHARE or share < 0.2:
                        continue                         # exact equalities are lookups; rare ones are noise
                    ex = None
                    for fk in exc_fks:
                        if fk not in kv_of:
                            kv_of[fk] = canon(df[fk])
                        kv = kv_of[fk]
                        bad = m & ~eq
                        top = pd.Series(kv[bad]).value_counts()
                        if not len(top):
                            continue
                        val = top.index[0]
                        if top.iloc[0] < MIN_SHARE * bad.sum():
                            continue
                        rest = m & (kv != val)
                        s_rest = float(eq[rest].mean()) if rest.sum() else 0.0
                        if s_rest < MIN_SHARE:
                            continue
                        chance = _perm_chance(lambda x, y: x == y, va[rest], vb[rest], self.rng)
                        if not significant(s_rest, chance, int(rest.sum())):
                            continue
                        rawv = df[fk].to_numpy(dtype=object)[np.flatnonzero(kv == val)[0]]
                        conv = float(np.mean(~eq[m & (kv == val)])) if (m & (kv == val)).any() else 0.0
                        ex = (fk, rawv, s_rest, chance, int(rest.sum()), conv)
                        break
                    if ex is None:
                        continue
                    fk, rawv, s_rest, chance, nrest, conv = ex
                    ea, eb = (f"days({a})", f"days({b})") if da else (a, b)
                    rule = f"{ea} == {eb} or {fk} == {_lit(rawv)}"
                    self.rules.append({"table": t, "type": "check", "expr": rule})
                    done_pairs.add(pair)
                    msg = (f"{t}: check {rule} (equal on {s_rest:.1%} of {nrest:,} sampled rows outside "
                           f"{fk} = {rawv}, {chance:.1%} by chance)")
                    if conv >= MIN_SHARE:
                        conv_rule = f"{ea} != {eb} or {fk} != {_lit(rawv)}"
                        self.rules.append({"table": t, "type": "check", "expr": conv_rule})
                        msg += f"; and never equal for {fk} = {rawv}: check {conv_rule}"
                    self.ask(msg)

    # ------------------------------------------------------- key order
    def key_orders(self):
        self.key_order = {}
        for t in self.names:
            if not self._modelled(t) or self.role[t] != "entity" or self.pks.get(t) is None:
                continue
            df = self.df[t]
            k = pd.to_numeric(df[self.pks[t]], errors="coerce")
            if k.isna().any() or len(df) < 100:
                continue
            srt = df.assign(__k=k.to_numpy()).sort_values("__k")
            for c, cs in self.cols[t].items():
                if cs["type"] not in ("date", "datetime") or cs.get("offset_from"):
                    continue
                v = pd.to_numeric(pd.Series(canon(srt[c], as_date=True)), errors="coerce").to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if len(v) < 100:
                    continue
                share = float(np.mean(np.diff(v) >= 0))
                if share < MIN_SHARE:
                    continue
                chance = float(np.mean([np.mean(np.diff(self.rng.permutation(v)) >= 0) for _ in range(3)]))
                if significant(share, chance, len(v) - 1):
                    self.key_order[t] = c
                    self.ask(f"{t}: keys numbered in the order of {c} (non-decreasing on {share:.1%} of "
                             f"{len(v) - 1:,} consecutive sampled keys, {chance:.1%} by chance)")
                    break
