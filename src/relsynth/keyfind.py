# SPDX-License-Identifier: Apache-2.0
"""Data-driven discovery of primary keys and links (v0.8).

The model is derived from the data that is given; no schema template, naming
convention list or known database is assumed.  Column and table names are part
of the data and may be used as one piece of evidence, weighted by how rare each
word is in this database (so words every table carries, whatever they are, count
for nothing); `evidence="data"` switches names off entirely.

Primary keys (unique column combinations, UCCs).  Every column that is non-null
and unique over the whole table is a candidate.  Candidates are scored as in
holistic key detection (Jiang & Naumann 2020): short values, early position and
(with names) an identifier-like word; floats with decimals, date-times, long
texts and binary columns are never keys.  The best candidate is the key when it
clears the next one by the largest gap ("cliff") or is the only one.

Links (unary inclusion dependencies, IND).  For each column c of table t and each
single-column key K of another table, the share of c's distinct values found in
K is the inclusion sigma; candidates need sigma >= 0.95.  An inclusion is only
evidence when it is unlikely by chance:

* integers: if c's v distinct values were spread at random over [min c, max c],
  the chance that all of them fall in K is C(k-2, v-2) / C(w-2, v-2) <= d^(v-2),
  d = k / w the share of that range occupied by K (chance_log10).  A
  surrogate key 1..N fills its range (d = 1), so inclusion in it carries no
  evidence at all, and the column/parent pair must be decided by the other
  features;
* strings: the same hypergeometric chance over the code space of the child's
  value shapes, the w strings of the lengths the child uses over the character
  classes it uses (26 capitals, 26 small letters, 10 digits, each other
  character seen), k of them keys: C(k, v) / C(w, v).  Short codes (grades
  A-F, sizes S/M/L) are no evidence; long identifiers (SO43697) overwhelming;
* randomness (Zhang et al. 2010): the distinct values of a real foreign key are
  spread over the parent key like a sample, so their quantile positions in K are
  close to uniform; R = 1 - 2 W1(positions, U(0,1)) (1 = uniform, 0 = all at one
  end);
* coverage C = share of K referenced;
* name similarity N (optional): the parent's table or key words found in the
  column name, weighted by inverse frequency (Chaudhuri et al. 2003).  A name
  is evidence only together with values that look like a sample of the key:
  R >= R_crit(v) = min(0.5, 1 - 2/sqrt(v)) (below the 0.1% point of R for v
  random keys), or chance inclusion; a measure named after a table (OrderQty
  1..20 in Orders' keys 1..1000, R = 0.02) is not a link.

A self-reference (a column pointing at its own table's key) is a candidate when
every non-null pointer lands on a row of the table, the pointers form a forest
(no cycle) and the tops are few and explicit: between 1 and max(3, 5% of rows)
null pointers.  The pointers and keys are read for the whole table, also when
the table is sampled.  The forest is evidence only when the column's own values,
assigned to the rows at random, would rarely form one: a conditional
permutation test (Good 2005) with B = 199 permutations and a fixed seed,

    p = (1 + #{permutations that form a forest}) / (B + 1),

accepted when p <= 0.01, stopped as soon as the count rules that out.  A
code column 1..3 with one null in 1,000 rows forms a forest under about 60% of
its permutations (the three rows 1..3 only need no cycle among themselves) and
is rejected, while uniform pointers over all n keys (forest_chance, the
unconditional chance, kept as a bound) would put it at 0.005.  A forest that is
not significant (Chinook's 8 employees with one top: 65% of the permutations
form one too) is asked about when its trees hang from its explicit tops, that
is no row points at itself: a code column forms a forest almost only through
self-pointers, its few target rows holding their own value.

Each column links to at most one parent (the best score); a near tie between
parents with no name evidence is reported, not guessed.  Every decision is
written to the review list with its evidence.
"""
from __future__ import annotations

import math
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .data import canonical_keys

MIN_SIGMA = 0.95
MAX_KEY_LEN = 40
CHANCE_LOG10 = -6.0          # inclusion counts as evidence when its chance is below 1e-6
NAME_MATCH = 0.8


# ---------------------------------------------------------------- names as data
def tokens(name: str) -> list[str]:
    """Words of an identifier: camelCase, snake_case, digits split off; plural s
    dropped so customer and customers are one word."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    out = []
    for w in re.split(r"[^A-Za-z0-9]+|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])", s):
        w = w.lower()
        if not w:
            continue
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


class Names:
    """Inverse-frequency weights of words over every table and column name of the
    database being read: a word in many names (whatever it is) says little."""

    def __init__(self, tables: dict[str, list[str]]):
        docs = [set(tokens(t)) for t in tables] + [set(tokens(c)) for cols in tables.values() for c in cols]
        n = max(len(docs), 1)
        df = {}
        for d in docs:
            for w in d:
                df[w] = df.get(w, 0) + 1
        self.w = {w: math.log((n + 1) / (k + 0.5)) for w, k in df.items()}

    def weight(self, w):
        return self.w.get(w, 1.0)

    def contained(self, small: list[str], big: list[str]) -> float:
        """Weighted share of the words of `small` found (or nearly found) in `big`."""
        if not small or not big:
            return 0.0
        num = den = 0.0
        for a in small:
            wa = self.weight(a)
            best = max((1.0 if a == b else SequenceMatcher(None, a, b).ratio()) for b in big)
            num += wa * (best if best >= 0.8 else 0.0)
            den += wa
        return num / den if den else 0.0

    def link(self, col, parent, pkey) -> float:
        """Evidence that column `col` names table `parent` (key `pkey`)."""
        tc = tokens(col)
        # a column whose words are exactly the parent's key words (CustomerKey ->
        # Customer.CustomerKey), or that contains the parent table's distinctive words
        a = self.contained(tokens(pkey), tc) if set(tokens(pkey)) - {w for w in tokens(pkey) if self.weight(w) < 0.5} else 0.0
        b = self.contained(tokens(parent), tc)
        return max(a, b)


# ---------------------------------------------------------------- value features
def _keyvals(s: pd.Series):
    k = canonical_keys(pd.Series(s)).dropna()
    return k


def is_keylike(s: pd.Series, binary=False) -> bool:
    """Could hold key values: integers or short strings (never measures with
    decimals, date-times, long texts or binary)."""
    if binary:
        return False
    v = s.dropna()
    if not len(v):
        return False
    if pd.api.types.is_bool_dtype(v):
        return False
    if pd.api.types.is_datetime64_any_dtype(v):
        return False
    if pd.api.types.is_numeric_dtype(v):
        return bool(np.allclose(v.astype(float), np.round(v.astype(float))))
    sv = v.astype(str)
    if sv.str.len().max() > MAX_KEY_LEN:
        return False
    # dates stored as text are not keys
    if sv.head(200).str.match(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?").mean() > 0.9:
        return False
    num = pd.to_numeric(sv, errors="coerce")
    if num.notna().all() and not np.allclose(num, np.round(num)):
        return False
    return True


def _as_int(vals: np.ndarray):
    try:
        a = np.asarray(vals, dtype=np.int64)
        return a
    except (TypeError, ValueError, OverflowError):
        return None


def chance_log10(child: np.ndarray, parent_sorted: np.ndarray) -> float:
    """log10 of the chance that v distinct integers spread at random over the range
    [min, max] of the child values all fall in the parent key set.  Given the two
    extremes (which are keys), the other v - 2 distinct values are a uniform draw
    without replacement from the w - 2 interior integers, k - 2 of them keys:

        P = C(k - 2, v - 2) / C(w - 2, v - 2)  <=  d^(v - 2),  d = k / w

    (hypergeometric, exact).  Strings: string_chance_log10."""
    ci, pi = _as_int(child), _as_int(parent_sorted)
    v = len(child)
    if ci is None or pi is None:
        return string_chance_log10(child, parent_sorted)
    lo, hi = int(ci.min()), int(ci.max())
    w = hi - lo + 1
    k = int(np.searchsorted(pi, hi, side="right") - np.searchsorted(pi, lo, side="left"))
    if v <= 2 or k >= w:
        return 0.0
    if k < v:
        return -np.inf
    lnc = lambda a, b: gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)
    return float((lnc(k - 2, v - 2) - lnc(w - 2, v - 2)) / math.log(10))


def _ln_choose(a, b: int) -> float:
    """ln C(a, b) for 0 <= b <= a, a possibly far beyond 2^53 (a float): the sum of
    ln(a - i) over i < b, minus ln b!, without the cancellation of gammaln(a + 1) -
    gammaln(a - b + 1) at large a."""
    i = np.arange(b, dtype=float)
    return float(np.sum(math.log(a) + np.log1p(-i / a)) - gammaln(b + 1))


def string_chance_log10(child, parent_sorted) -> float:
    """log10 of the chance that v distinct random codes of the child's shapes all
    fall in the parent key set.  The code space holds the w strings of the lengths
    the child's values have, over the character classes they use: 26 capitals, 26
    small letters and 10 digits (ASCII) when any is used, plus each other character
    seen.  The codes are a uniform draw without replacement from it, k of whose
    strings are keys:

        P = C(k, v) / C(w, v)  <=  (k / w)^v

    (hypergeometric, exact).  Three grades in A-F: 20 / 2600, 10^-2.1, no
    evidence; 30 order numbers SO43697.. among 30,000: 10^-192."""
    child = [str(x) for x in child]
    v = len(child)
    if v <= 2:
        return 0.0
    chars = set("".join(child))
    up = any("A" <= ch <= "Z" for ch in chars)
    lo = any("a" <= ch <= "z" for ch in chars)
    dg = any("0" <= ch <= "9" for ch in chars)
    other = {ch for ch in chars if not ("A" <= ch <= "Z" or "a" <= ch <= "z" or "0" <= ch <= "9")}
    A = 26 * up + 26 * lo + 10 * dg + len(other)
    lens = {len(x) for x in child}
    w = sum(float(A) ** L for L in lens)
    cls = ("A-Z" if up else "") + ("a-z" if lo else "") + ("0-9" if dg else "") + \
        "".join(re.escape(ch) for ch in sorted(other))
    K = pd.Series(np.asarray(parent_sorted, dtype=object)).astype(str)
    K = K[K.str.len().isin(lens)]
    k = int(K.str.fullmatch(f"[{cls}]+").sum()) if len(K) else 0
    if k >= w:
        return 0.0
    if k < v:
        return -np.inf
    return (_ln_choose(k, v) - _ln_choose(w, v)) / math.log(10)


def randomness(child: np.ndarray, parent_sorted: np.ndarray) -> float:
    """R = 1 - 2 W1(quantile positions of the child values in the parent key,
    Uniform(0,1)); 1 when they spread like a sample, 0 when they sit at one end
    (Zhang et al. 2010, one dimension, exact)."""
    n = len(parent_sorted)
    if n == 0 or len(child) == 0:
        return 0.0
    pos = np.searchsorted(parent_sorted, np.sort(child))
    u = (pos + 0.5) / n
    m = len(u)
    # W1 between the empirical law of u and U(0,1), exactly: integral of |F_m(x) - x|
    grid = np.concatenate(([0.0], np.sort(u), [1.0]))
    w1 = 0.0
    for i in range(m + 1):
        a, b, f = grid[i], grid[i + 1], i / m
        # integral over [a, b] of |f - x|
        if f <= a:
            w1 += (b * b - a * a) / 2 - f * (b - a)
        elif f >= b:
            w1 += f * (b - a) - (b * b - a * a) / 2
        else:
            w1 += ((f - a) ** 2 + (b - f) ** 2) / 2
    return float(max(0.0, 1 - 2 * w1))


def r_crit(v) -> float:
    """Randomness a real foreign key with v distinct values reaches: R >= min(0.5,
    1 - 2 / sqrt(v)).  sqrt(v) W1 of v random keys has its 0.999 quantile below 1
    (simulated for v = 5..1000), so a real key falls short with probability below
    0.1%; for v <= 4 the bound is not positive and asks nothing."""
    return min(0.5, 1 - 2 / math.sqrt(v)) if v > 0 else 1.0


FOREST_ALPHA = 0.01
FOREST_PERMS = 199           # (1 + 0) / (199 + 1) = 0.005: p <= FOREST_ALPHA is attainable


def _forest_rows(pos, idx, steps) -> bool:
    """Do the pointers row -> pos (row positions, -1 for a null or outside pointer)
    form a forest?  Pointer doubling: roots (null, outside, self) are made fixed
    points and 2^steps >= n jumps take every row that reaches a root onto it;
    a row on or above a cycle never does.  Same answer as is_forest."""
    root = (pos < 0) | (pos == idx)
    if not root.any():
        return False
    nxt = np.where(root, idx, pos)
    for _ in range(steps):
        nxt = nxt[nxt]
    return bool(root[nxt].all())


def forest_perm_test(pointer, key, B=FOREST_PERMS, alpha=FOREST_ALPHA, seed=0):
    """Conditional permutation test of 'the pointers form a forest' (Good 2005):
    the column's own values (nulls included) are assigned to the rows at random B
    times with a fixed seed, and

        p = (1 + #{permutations forming a forest}) / (B + 1)

    is a valid p-value under the null that the values are exchangeable over the
    rows.  The draws stop as soon as 1 + count exceeds alpha (B + 1), when
    p > alpha is certain.  Returns (p, count, permutations drawn); after an early
    stop p is a lower bound."""
    at = {k: i for i, k in enumerate(key)}
    pos = np.array([-1 if p is None else at.get(p, -1) for p in pointer], dtype=np.int64)
    n = len(pos)
    idx = np.arange(n)
    steps = max(1, int(math.ceil(math.log2(max(n, 2)))))
    rng = np.random.default_rng(seed)
    hits = drawn = 0
    while drawn < B:
        drawn += 1
        hits += _forest_rows(pos[rng.permutation(n)], idx, steps)
        if 1 + hits > alpha * (B + 1):
            break
    return (1 + hits) / (B + 1), hits, drawn


def forest_chance(n, r):
    """Probability that n - r pointers drawn uniformly from n keys, plus r null
    pointers, form a forest (self-pointers counting as tops).  With s self-pointers
    the other n - r - s pointers form a forest rooted at the r + s tops, of which
    there are (r + s) n^(n - r - s - 1) (Cayley); summing over s and dividing by
    n^(n - r) gives (1/n) [r (1 + 1/n)^m + (m/n) (1 + 1/n)^(m-1)] with m = n - r,
    which is (r + 1) (1 + 1/n)^(m-1) / n <= e (r + 1) / n.  The unconditional
    chance: it ignores the column's own values (a code 1..3 is far more likely to
    form a forest than uniform pointers), so the decision uses forest_perm_test."""
    m = n - r
    if n <= 0 or m <= 0:
        return 1.0
    return min(1.0, (r + 1) * (1 + 1 / n) ** (m - 1) / n)


def forest_depth(pointer, key) -> int:
    """Longest chain of the pointers key -> pointer when they form a forest (0 if
    they have a cycle or no root).  A real hierarchy has chains of length >= 2."""
    if not is_forest(pointer, key):
        return 0
    nxt = {c: p for c, p in zip(key, pointer)}
    depth = {}

    def d(x):
        path = []
        while x in nxt and nxt[x] is not None and nxt[x] != x and x not in depth:
            path.append(x)
            x = nxt[x]
        base = depth.get(x, 0)
        for z in reversed(path):
            base += 1
            depth[z] = base
        return depth.get(path[0], 0) if path else base

    return max((d(x) for x in nxt), default=0)


def is_forest(pointer, key) -> bool:
    """Do the pointers key -> pointer (one per row of a table) form a forest: no
    cycle, at least one root (null, self-pointing or pointing outside)?"""
    nxt = {}
    for c, p in zip(key, pointer):
        nxt[c] = p
    roots = sum(1 for c, p in nxt.items() if p is None or p == c or p not in nxt)
    if roots == 0:
        return False
    state = {}
    for start in nxt:
        path = []
        x = start
        while x in nxt and state.get(x) is None:
            state[x] = 1
            path.append(x)
            y = nxt[x]
            if y is None or y == x:
                break
            if state.get(y) == 1:
                return False                # back to a point on this path: cycle
            x = y
        for z in path:
            state[z] = 2
    return True


# ---------------------------------------------------------------- the search
class KeyFinder:
    def __init__(self, P, evidence="names+data"):
        self.P = P
        self.use_names = evidence != "data"
        self.names = Names({t: list(P.df[t].columns) for t in P.names})
        self.notes = []
        self._ks = {}

    def keyset(self, t, c):
        if (t, c) not in self._ks:
            s = self.P.df[t][c] if self.P.whole[t] else self.P.src.column(t, c)
            k = _keyvals(s)
            arr = np.array(pd.unique(k.astype(object)), dtype=object)
            ints = _as_int(arr)
            self._ks[(t, c)] = np.sort(ints) if ints is not None else np.sort(arr.astype(str))
        return self._ks[(t, c)]

    # ------------------------------------------------------------ keys
    def primary_key(self, t):
        """(column, evidence) or (None, reason)."""
        P = self.P
        df = P.df[t]
        binary = set(P.src.meta[t]["binary"])
        cands = []
        cols = list(df.columns)
        for i, c in enumerate(cols):
            s = df[c]
            if c in binary or s.isna().any() or not s.is_unique or not is_keylike(s):
                continue
            if not P.whole[t] and not P.src.column(t, c).is_unique:
                continue
            L = s.astype(str).str.len().max()
            score_len = 1.0 / max(1.0, L - 8)
            score_pos = 1.0 / (i + 1)
            feats = [score_len, score_pos]
            if self.use_names:
                own = self.names.contained(tokens(t), tokens(c)) if tokens(t) else 0.0
                feats.append(max(own, 1.0 if c.lower() in ("id", "key") else 0.0))
            cands.append((float(np.mean(feats)), c))
        if not cands:
            return None, "no column is unique and non-null"
        cands.sort(reverse=True)
        if len(cands) == 1:
            return cands[0][1], f"only unique non-null key-like column (score {cands[0][0]:.2f})"
        gap = cands[0][0] - cands[1][0]
        if gap < 0.05:
            return cands[0][1], (f"best of {len(cands)} unique columns but close to {cands[1][1]} "
                                 f"({cands[0][0]:.2f} vs {cands[1][0]:.2f}); check")
        return cands[0][1], f"best of {len(cands)} unique columns (score {cands[0][0]:.2f}, next {cands[1][0]:.2f})"

    # ------------------------------------------------------------ links
    def candidates(self, t, c, parents):
        """Scored parents for column c of t: list of dicts."""
        P = self.P
        s = P.df[t][c]
        if not is_keylike(s, c in P.src.meta[t]["binary"]):
            return []
        # screen on the sample's distinct values, confirm on the whole column
        sv = _keyvals(s)
        sv = np.array(pd.unique(sv.astype(object)), dtype=object)
        si = _as_int(sv)
        sv = np.sort(si) if si is not None else np.sort(sv.astype(str))
        if len(sv) < 1:
            return []
        screened = []
        for pt, pk in parents:
            if pt == t and c == pk:
                continue
            K = self.keyset(pt, pk)
            if len(K) and (sv.dtype.kind in "iu") == (K.dtype.kind in "iu") and np.isin(sv, K).mean() >= 0.9:
                screened.append((pt, pk))
        if not screened:
            return []
        vals = self.keyset(t, c)
        out = []
        for pt, pk in screened:
            K = self.keyset(pt, pk)
            if len(K) == 0 or (vals.dtype.kind in "iu") != (K.dtype.kind in "iu"):
                continue
            hit = np.isin(vals, K)
            sigma = float(hit.mean())
            if sigma < MIN_SIGMA:
                continue
            inc = vals[hit]
            cl = chance_log10(inc, K)
            R = randomness(inc, K)
            C = len(inc) / len(K)
            N = self.names.link(c, pt, pk) if self.use_names else 0.0
            x = {"parent": pt, "key": pk, "sigma": sigma, "chance": cl, "R": R, "C": C, "N": N,
                 "R_crit": r_crit(len(inc)), "forest": None, "p_forest": None, "v": len(vals)}
            if pt == t:
                # pointers and keys of every row, also when the table is sampled
                rows = P.df[t] if P.whole[t] else P.src.columns(t, [pk, c])
                kk = canonical_keys(rows[pk]).astype(object).where(rows[pk].notna(), None).tolist()
                cc = canonical_keys(rows[c]).astype(object).where(rows[c].notna(), None).tolist()
                # a hierarchy: every pointer lands on a row of the table, the pointers
                # form a forest, and its tops are few and explicit (null pointers):
                # at most max(3, 5% of rows) roots, at least 2 rows below them
                roots = sum(1 for a in cc if a is None)
                below = sum(1 for a, b in zip(cc, kk) if a is not None and a != b)
                forest = (sigma == 1.0 and 1 <= roots <= max(3, 0.05 * len(cc)) and below >= 2
                          and forest_depth(cc, kk) >= 1)
                if not forest:
                    continue
                x["forest"] = True
                x["p_forest"], x["perm_hits"], x["perms"] = forest_perm_test(cc, kk)
                # trees that hang from the explicit tops only (no row points at itself)
                x["tops_explicit"] = not any(a is not None and a == b for a, b in zip(cc, kk, strict=True))
            out.append(x)
        return out

    @staticmethod
    def evidence(x, use_names):
        """Is this candidate supported by more than chance?  Returns the reason or None."""
        if x["forest"] and x["p_forest"] <= FOREST_ALPHA:
            return (f"pointers form a forest; {x['perm_hits']} of {x['perms']} random permutations of the column "
                    f"do (p = {x['p_forest']:.3f})")
        if use_names and x["N"] >= NAME_MATCH:
            # a name is evidence together with values that could be the key's
            if x["forest"]:
                if x["tops_explicit"]:
                    return "name names the parent, and the pointers form a forest under explicit tops"
            elif x["R"] >= x["R_crit"]:
                return f"name names the parent, and the values spread over its key (R >= {max(x['R_crit'], 0.0):.2f})"
        if x["v"] < 2:
            return None
        if x["chance"] <= CHANCE_LOG10:
            return f"inclusion unlikely by chance (1e{max(x['chance'], -99):.0f})"
        return None

    @staticmethod
    def score(x, use_names):
        return (2.0 * x["N"] if use_names else 0.0) + x["R"] + x["C"] + (0.5 if x["chance"] <= CHANCE_LOG10 else 0.0)

    def key_words(self, parents):
        """Words that at least half of this database's primary-key names share (for
        example 'id' or 'key'): learnt from the keys found, not from a list."""
        if getattr(self, "_kw", None) is None:
            names = [set(tokens(pk)) for _, pk in parents if isinstance(pk, str)]
            cnt = {}
            for ws in names:
                for w in ws:
                    cnt[w] = cnt.get(w, 0) + 1
            self._kw = {w for w, k in cnt.items() if len(names) >= 3 and k >= len(names) / 2}
        return self._kw

    def best_parent(self, t, c, parents):
        """(chosen candidate with 'why', ambiguous alternatives) or (None, [])."""
        cands, weak = [], []
        named_like_key = self.use_names and bool(set(tokens(c)) & self.key_words(parents))
        for x in self.candidates(t, c, parents):
            why = self.evidence(x, self.use_names)
            x["score"] = self.score(x, self.use_names)
            if why:
                x["why"] = why
                cands.append(x)
            elif x["forest"] and x["tops_explicit"]:
                x["why"] = (f"the pointers form a forest under explicit tops, but so do {x['perm_hits']} of "
                            f"{x['perms']} random permutations of the column (p {'=' if x['perms'] == FOREST_PERMS else '>='} "
                            f"{x['p_forest']:.2g})")
                weak.append(x)
            elif named_like_key and x["sigma"] == 1.0:
                x["why"] = "named like this database's keys, but its values alone do not establish the parent"
                weak.append(x)
            elif x["v"] >= 5 and x["R"] >= x["R_crit"] and x["C"] >= 0.5 and (not self.use_names or x["N"] >= 0.5):
                x["why"] = "inclusion alone is no evidence (the parent key fills its range)"
                weak.append(x)
        if not cands:
            # values that fit a dense key range: a question, never a guessed link
            return None, sorted(weak, key=lambda x: -x["score"])
        cands.sort(key=lambda x: -x["score"])
        top = cands[0]
        close = [x for x in cands[1:] if top["score"] - x["score"] < 0.05]
        if close and top["N"] < NAME_MATCH:
            return None, [top] + close
        return top, []


def describe(x):
    return (f"{x['sigma']:.0%} of values in the parent, coverage {x['C']:.0%}, randomness {x['R']:.2f}"
            + (f", name {x['N']:.2f}" if x["N"] else "") + f"; {x['why']}")


def acyclic(rels):
    """Drop the weakest link of each cycle between different tables (a schema's
    links between tables form no cycle; self-references are kept)."""
    rels = list(rels)
    while True:
        g = {}
        for r in rels:
            if r["child"] != r["parent"] and isinstance(r["child_key"], str):
                g.setdefault(r["child"], []).append(r)
        cyc = _find_cycle(g)
        if not cyc:
            return rels
        weakest = min((r for r in cyc if r.get("_score") is not None), key=lambda r: r["_score"], default=None)
        if weakest is None:
            return rels                      # declared links only: kept as declared
        rels.remove(weakest)


def _find_cycle(g):
    state, stack = {}, []

    def dfs(u):
        state[u] = 1
        for r in g.get(u, []):
            v = r["parent"]
            stack.append(r)
            if state.get(v) == 1:
                i = next(i for i, e in enumerate(stack) if e["child"] == v)
                return stack[i:]
            if state.get(v) is None:
                c = dfs(v)
                if c:
                    return c
            stack.pop()
        state[u] = 2
        return None

    for u in list(g):
        if state.get(u) is None:
            c = dfs(u)
            if c:
                return c
    return None
