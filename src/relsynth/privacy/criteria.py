"""Attacks for the three criteria of EDPB Guidelines 02/2026 (draft for public
consultation, adopted 7 July 2026, sec. 3.4): no record isolation, no linkage,
no inference.  They follow the attacker model of Anonymeter (Giomi, Boenisch,
Wehmeyer and Tasnádi, PoPETs 2023): the attacker holds the synthetic data and
some auxiliary knowledge, makes guesses about the real people, and the risk is
the success rate on the training people above the success rate on control
people that the generator never saw:

    risk = (rate_train - rate_control) / (1 - rate_control)

The control set removes what the synthetic data teaches about the population
(utility) and keeps what it reveals about the people in the training data
(Prop. 601).  A random-guess baseline checks that the attack is informative.

The attacks run on one row per person (relsynth.privacy.unit.profiles), so a
person's rows in every table are attacked together.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .encode import Encoder, column_kinds, kneighbors
from .stats import Risk, risk

OPS = {"==": operator.eq, "!=": operator.ne, "<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}


# ------------------------------------------------------------- record isolation
@dataclass(frozen=True)
class Predicate:
    """A conjunction of (column, op, value) terms; value None means 'is null'."""
    terms: tuple

    def __str__(self):
        return " & ".join(f"{c} is null" if v is None and o == "==" else f"{c} is not null" if v is None
                          else f"{c} {o} {v!r}" for c, o, v in self.terms)


class _Table:
    """Column arrays of a frame for fast predicate counting."""

    def __init__(self, df: pd.DataFrame, kinds):
        self.n = len(df)
        self.col = {}
        for c in df.columns:
            if kinds.get(c) == "num":
                self.col[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            else:
                self.col[c] = df[c].astype(object).where(df[c].notna(), None).to_numpy()

    def mask(self, p: Predicate):
        m = np.ones(self.n, bool)
        for c, o, v in p.terms:
            x = self.col[c]
            if v is None:
                isnull = np.isnan(x) if x.dtype.kind == "f" else np.array([e is None for e in x])
                m &= isnull if o == "==" else ~isnull
            else:
                with np.errstate(invalid="ignore"):
                    r = OPS[o](x, v)
                m &= np.asarray(r, bool)
            if not m.any():
                break
        return m

    def count(self, p: Predicate) -> int:
        return int(self.mask(p).sum())


def _univariate_queries(syn: _Table, kinds, rng):
    qs = []
    for c, x in syn.col.items():
        isnull = np.isnan(x) if x.dtype.kind == "f" else np.array([e is None for e in x])
        if isnull.sum() == 1:
            qs.append(Predicate(((c, "==", None),)))
        v = x[~isnull]
        if kinds[c] == "num" and len(v):
            qs += [Predicate(((c, "<=", float(v.min())),)), Predicate(((c, ">=", float(v.max())),))]
        vals, cnt = np.unique(v.astype(str) if kinds[c] == "cat" else v, return_counts=True)
        for val in vals[cnt == 1]:
            if kinds[c] == "cat":
                # map back to the original object (a 1 must stay an int when the column is mixed)
                orig = next(e for e in v if str(e) == val)
                qs.append(Predicate(((c, "==", orig),)))
            else:
                qs.append(Predicate(((c, "==", float(val)),)))
    order = rng.permutation(len(qs))
    return [qs[i] for i in order]


def _multivariate_queries(syn: _Table, kinds, n_cols, n_queries, rng, max_attempts, batch=1000):
    cols = list(syn.col)
    med = {c: np.nanmedian(syn.col[c]) for c in cols if kinds[c] == "num" and np.isfinite(syn.col[c]).any()}
    out, seen, tries = [], set(), 0
    while len(out) < n_queries and tries < max_attempts:
        for r in rng.integers(0, syn.n, batch):
            tries += 1
            sel = sorted(rng.choice(cols, size=min(n_cols, len(cols)), replace=False).tolist())
            terms = []
            for c in sel:
                v = syn.col[c][r]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    terms.append((c, "==", None))
                elif kinds[c] == "num":
                    terms.append((c, ">=" if v > med.get(c, v) else "<=", float(v)))
                else:
                    terms.append((c, "==", v))
            p = Predicate(tuple(terms))
            if p not in seen and syn.count(p) == 1:
                seen.add(p)
                out.append(p)
                if len(out) >= n_queries:
                    break
    return out


def _random_queries(syn: _Table, kinds, n_cols, n, rng):
    cols = list(syn.col)
    uniq = {c: pd.unique(syn.col[c]) for c in cols}
    qs = []
    for _ in range(n):
        sel = sorted(rng.choice(cols, size=min(n_cols, len(cols)), replace=False).tolist())
        terms = []
        for c in sel:
            v = uniq[c][rng.integers(len(uniq[c]))]
            ops = ["==", "!="] if kinds[c] == "cat" else list(OPS)
            o = ops[rng.integers(len(ops))]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                terms.append((c, "==" if o == "==" else "!=", None))
            else:
                terms.append((c, o, float(v) if kinds[c] == "num" else v))
        qs.append(Predicate(tuple(terms)))
    return qs


def _so_curve(n, w):
    """Expected share of queries of weight w that single out in n rows,
    integrated over w in [0, w_eff] (Cohen and Nissim 2020; Anonymeter)."""
    return ((1.0) - (n * w + 1) * (1 - w) ** n) / (n + 1)


@dataclass
class IsolationResult:
    risk: Risk
    mode: str
    n_queries: int
    queries: list = field(default_factory=list)
    isolated_ids: list = field(default_factory=list)

    def to_dict(self):
        return {"criterion": "record_isolation", "attack": f"singling_out_{self.mode}", **self.risk.to_dict(),
                "n_queries": self.n_queries, "example_queries": [str(q) for q in self.queries[:5]]}


def record_isolation(ori: pd.DataFrame, syn: pd.DataFrame, control: pd.DataFrame | None = None, *,
                     mode: str = "multivariate", n_attacks: int = 500, n_cols: int = 3, seed: int = 0,
                     max_attempts: int = 200_000, size_correction: str = "subsample",
                     level: float = 0.95) -> IsolationResult:
    """Singling-out attack (Giomi et al. sec. 3.1).  Queries that isolate one
    synthetic row are guessed to isolate one real person; a guess succeeds when
    the predicate matches exactly one training person.

    size_correction: how the control rate is made comparable when the control set
    is smaller than the training set.  'subsample' (default) evaluates the
    queries on a uniform subset of the training people of the control's size,
    which makes the two rates exactly equal in expectation for a generator that
    ignores individuals (Prop. 601); it costs power (a person outside the subset
    cannot be caught), so assess() splits training and control people in equal
    numbers and no subsampling happens.  'anonymeter' rescales the control count
    by Anonymeter's fitted singling-out curve; it is kept to reproduce
    Anonymeter's numbers, but it is not exact under the null: with 2.5 times
    more training than control people and a generator that never saw them, it
    reported risks of 0.10-0.14 with intervals excluding 0 (test_privacy.py)."""
    rng = np.random.default_rng(seed)
    cols = [c for c in syn.columns if c in ori.columns]
    ori, syn = ori[cols].drop_duplicates(), syn[cols].drop_duplicates()
    control = control[cols].drop_duplicates() if control is not None else None
    kinds = column_kinds([ori, syn] + ([control] if control is not None else []), cols)
    S = _Table(syn, kinds)
    if mode == "univariate":
        cand = _univariate_queries(S, kinds, rng)
        qs, seen = [], set()
        for p in cand:
            if p not in seen and S.count(p) == 1:
                seen.add(p)
                qs.append(p)
            if len(qs) >= n_attacks:
                break
        k = 1
    else:
        qs = _multivariate_queries(S, kinds, n_cols, n_attacks, rng, max_attempts)
        k = n_cols
    if not qs:
        r = risk(n_attacks, 0, level=level)
        return IsolationResult(r, mode, 0)
    n_eval = n_attacks                      # rates are over the attacks attempted, as in Anonymeter
    target = ori
    if control is not None and size_correction == "subsample" and len(control) < len(ori):
        target = ori.sample(len(control), random_state=int(rng.integers(2**31)))
    O = _Table(target, kinds)
    hits = [p for p in qs if O.count(p) == 1]
    base = _random_queries(S, kinds, k, n_attacks, rng)
    n_base = sum(O.count(p) == 1 for p in base)
    n_ctrl = None
    if control is not None:
        C = _Table(control, kinds)
        n_ctrl = float(sum(C.count(p) == 1 for p in qs))
        if size_correction == "anonymeter" and len(control) != len(ori):
            n_ctrl *= _anonymeter_correction(control, kinds, qs, len(ori), rng)
    r = risk(n_eval, len(hits), n_eval if n_ctrl is not None else None, n_ctrl, n_eval, n_base, level)
    ids = [int(np.flatnonzero(O.mask(p))[0]) for p in hits[:50]]
    ids = [target.index[i] for i in ids]
    return IsolationResult(r, mode, len(qs), hits, ids)


def _anonymeter_correction(control, kinds, qs, n_ori, rng):
    from scipy.optimize import curve_fit
    sizes, succ = [], []
    lo = min(1000, len(control))
    for m in np.linspace(lo, len(control), 10).astype(int):
        for _ in range(5):
            sub = _Table(control.sample(m, random_state=int(rng.integers(2**31))), kinds)
            succ.append(sum(sub.count(p) == 1 for p in qs))
            sizes.append(m)
    sizes, succ = np.array(sizes, float), np.array(succ, float)
    f = lambda x, w, a: a * _so_curve(x, w)  # noqa: E731
    w0 = 1 / sizes.max()
    try:
        (w, a), _ = curve_fit(f, sizes, succ, p0=(w0, 1 / max(_so_curve(sizes.max(), w0), 1e-12)),
                              bounds=(0, (1, np.inf)))
        return float(f(n_ori, w, a) / max(f(len(control), w, a), 1e-12))
    except RuntimeError:
        return 1.0


# ------------------------------------------------------------------- linkage
@dataclass
class LinkageResult:
    risk: Risk
    aux_cols: tuple
    n_neighbors: int
    linked_ids: list = field(default_factory=list)

    def to_dict(self):
        return {"criterion": "linkage", "attack": "nearest_neighbour_linkage", **self.risk.to_dict(),
                "aux_cols": [list(a) for a in self.aux_cols], "n_neighbors": self.n_neighbors}


def linkage(ori, syn, control=None, *, aux_cols, n_attacks: int = 500, n_neighbors: int = 1, seed: int = 0,
            level: float = 0.95) -> LinkageResult:
    """Linkability attack (Giomi et al. sec. 3.2).  The attacker holds two
    datasets about the same people with disjoint columns A and B (for example
    demographics and behaviour).  For each person it looks up the n_neighbors
    nearest synthetic rows using A, and using B; it claims a link when the two
    neighbour sets intersect."""
    rng = np.random.default_rng(seed)
    A, B = list(aux_cols[0]), list(aux_cols[1])
    frames = [ori, syn] + ([control] if control is not None else [])
    encA, encB = Encoder(frames, A), Encoder(frames, B)

    def attack(df):
        n = min(n_attacks, len(df))
        t = df.sample(n, random_state=int(rng.integers(2**31)))
        ia, _ = kneighbors(encA, t[A], syn[A], n_neighbors, rng=rng)
        ib, _ = kneighbors(encB, t[B], syn[B], n_neighbors, rng=rng)
        ok = np.array([len(set(a) & set(b)) > 0 for a, b in zip(ia, ib)])
        return n, int(ok.sum()), t.index[ok].tolist()

    n_o, k_o, ids = attack(ori)
    n_c, k_c = (attack(control)[:2] if control is not None else (None, None))
    # random guessing: two random neighbour sets of size k out of len(syn) intersect
    ra = np.array([rng.choice(len(syn), n_neighbors, replace=False) for _ in range(n_o)])
    rb = np.array([rng.choice(len(syn), n_neighbors, replace=False) for _ in range(n_o)])
    k_b = int(sum(len(set(a) & set(b)) > 0 for a, b in zip(ra, rb)))
    return LinkageResult(risk(n_o, k_o, n_c, k_c, n_o, k_b, level), (tuple(A), tuple(B)), n_neighbors, ids)


# ------------------------------------------------------------------ inference
@dataclass
class InferenceResult:
    risk: Risk
    attribute: str
    aux_cols: list
    regression: bool
    correct_ids: list = field(default_factory=list)

    def to_dict(self):
        return {"criterion": "inference", "attack": "nearest_neighbour_inference", "attribute": self.attribute,
                **self.risk.to_dict(), "aux_cols": self.aux_cols, "regression": self.regression}


def guesses_correct(guess, truth, regression: bool, tolerance: float = 0.05):
    g, s = np.asarray(guess), np.asarray(truth)
    both_null = pd.isna(g) & pd.isna(s)
    if regression:
        g, s = g.astype(float), s.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = np.abs(g - s) <= tolerance * np.maximum(np.abs(g), 1e-12)
    else:
        ok = g.astype(object) == s.astype(object)
    return np.asarray(ok | both_null, bool)


def inference(ori, syn, control=None, *, attribute: str, aux_cols=None, n_attacks: int = 500,
              regression: bool | None = None, seed: int = 0, level: float = 0.95) -> InferenceResult:
    """Attribute inference attack (Giomi et al. sec. 3.3): the attacker knows the
    aux columns of a person, finds the nearest synthetic row and guesses its
    attribute.  The baseline guesses the attribute of a random synthetic row."""
    rng = np.random.default_rng(seed)
    aux = [c for c in (aux_cols or [c for c in ori.columns if c != attribute]) if c != attribute]
    if regression is None:
        regression = ori[attribute].dtype.kind in "fiu" and ori[attribute].nunique() > 20
    frames = [ori, syn] + ([control] if control is not None else [])
    enc = Encoder(frames, aux)

    def attack(df):
        n = min(n_attacks, len(df))
        t = df.sample(n, random_state=int(rng.integers(2**31)))
        i, _ = kneighbors(enc, t[aux], syn[aux], 1, rng=rng)
        ok = guesses_correct(syn[attribute].to_numpy()[i[:, 0]], t[attribute].to_numpy(), regression)
        return n, int(ok.sum()), t.index[ok].tolist()

    n_o, k_o, ids = attack(ori)
    n_c, k_c = (attack(control)[:2] if control is not None else (None, None))
    t = ori.sample(min(n_attacks, len(ori), len(syn)), random_state=int(rng.integers(2**31)))
    g = syn[attribute].sample(len(t), random_state=int(rng.integers(2**31))).to_numpy()
    k_b = int(guesses_correct(g, t[attribute].to_numpy(), regression).sum())
    return InferenceResult(risk(n_o, k_o, n_c, k_c, len(t), k_b, level), attribute, aux, bool(regression), ids)
