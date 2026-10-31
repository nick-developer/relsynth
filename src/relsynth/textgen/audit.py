# SPDX-License-Identifier: Apache-2.0
"""Leakage audit of a text generator (docs/text.md, Props. 1500, 1504, 1505).

* **Canaries and exposure** (Carlini et al., "The Secret Sharer", USENIX
  Security 2019).  Fresh people are added to the training data, each holding
  `repeats` texts with a canary "<prefix> s_1 ... s_k" whose secret is drawn
  uniformly from a public candidate set R (k words from a public list).  After
  training, the model's log-likelihood ranks every candidate; the exposure of a
  canary is log2 |R| - log2 rank (ties broken at random).  Under (eps, delta)-DP
  for the canary's person, P(exposure >= t) <= e^eps 2^-t + delta (Prop. 1500),
  and E[exposure] <= (eps + 1) / ln 2 + delta log2 |R| (Prop. 1504), whatever
  the model and the rest of the data.  `exposure_bound` evaluates it for a
  zCDP rho, minimising over the conversion's delta.
* **Exact and near duplicates**: every synthetic text against every training
  text, by MinHash over word 3-gram shingles with LSH banding (Broder 1997)
  and an exact Jaccard check of every candidate pair (Prop. 1505).
* **Extraction attack** (Carlini et al., "Extracting Training Data from Large
  Language Models", USENIX Security 2021): many samples are drawn, ranked by a
  membership score (the model's likelihood against a zlib-entropy reference,
  or how often a sample repeats), and the top ones are matched against the
  training texts held by fewer than `k` people.
"""
from __future__ import annotations

import hashlib
import math
import zlib
from collections import Counter, defaultdict

import numpy as np

CANARY_PREFIX = "my reference words are"


# ------------------------------------------------------------------ canaries
def candidate_words(texts, n: int = 24, skip: int = 20) -> list[str]:
    """`n` lower-case words that are common in the corpus (ranks skip..skip+n of
    the word frequency), so every model can produce them: a canary built from
    them tests memorisation of the *combination*, not of a new word."""
    from ..text import WORD, tokenize
    c: Counter = Counter()
    for t in texts:
        if t is None or (isinstance(t, float) and math.isnan(t)):
            continue
        c.update({w.lower() for w in tokenize(str(t)) if WORD.search(w) and w.isalpha() and len(w) > 2})
    ws = [w for w, _ in c.most_common(skip + n)][skip: skip + n]
    if len(ws) < n:
        ws = [w for w, _ in c.most_common(n)]
    return sorted(ws)


def make_canaries(words, k: int, n: int, seed: int) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    return [tuple(int(x) for x in rng.integers(len(words), size=k)) for _ in range(n)]


def canary_text(words, secret) -> str:
    return CANARY_PREFIX + " " + " ".join(words[j] for j in secret)


def insert_canaries(texts, owners, classes, words, secrets_, repeats: int, seed: int, owner_prefix="canary"):
    """The training rows with one fresh person per canary holding `repeats`
    copies of it (with a class drawn from the existing ones)."""
    rng = np.random.default_rng(seed)
    T, O, C = list(texts), list(owners), list(classes) if classes is not None else [None] * len(texts)
    levels = sorted({str(c) for c in C})
    cls = []
    for i, s in enumerate(secrets_):
        c = levels[int(rng.integers(len(levels)))] if levels else None
        cls.append(c)
        for _ in range(repeats):
            T.append(canary_text(words, s))
            O.append(f"{owner_prefix}-{i}")
            C.append(c)
    return T, O, C, cls


def exposures(logprob, words, k: int, secrets_, classes, seed: int = 0) -> np.ndarray:
    """Exposure of each canary: `logprob(tokens, class)` scores all |words|^k
    candidates; rank 1 is the most likely."""
    from ..text import tokenize
    rng = np.random.default_rng(seed)
    prefix = tokenize(CANARY_PREFIX)
    grid = np.array(np.meshgrid(*[np.arange(len(words))] * k, indexing="ij")).reshape(k, -1).T
    R = len(grid)
    out = []
    cache: dict = {}
    for s, c in zip(secrets_, classes):
        key = str(c)
        if key not in cache:
            cache[key] = np.array([logprob(prefix + [words[j] for j in g], c) for g in grid])
        lp = cache[key]
        idx = int(np.ravel_multi_index(tuple(s), [len(words)] * k))
        mine = lp[idx]
        higher = int(np.sum(lp > mine))
        ties = int(np.sum(lp == mine)) - 1
        rank = 1 + higher + int(rng.integers(ties + 1))
        out.append(math.log2(R) - math.log2(rank))
    return np.array(out)


def exposure_bound(t: float, rho: float | None = None, delta_extra: float = 0.0, epsilon: float | None = None,
                   delta: float | None = None) -> float:
    """Upper bound on P(exposure >= t) for one person's canary (Prop. 1500): with
    (epsilon, delta) given, e^eps 2^-t + delta; with a zCDP rho, the minimum of
    that over the (eps(d), d) pairs rho implies, plus `delta_extra` (the
    partition-selection delta)."""
    if epsilon is not None:
        return min(1.0, math.exp(epsilon) * 2.0 ** (-t) + (delta or 0.0) + delta_extra)
    from ..governance.accounting import cks_delta, epsilon_of
    best = 1.0
    for d in np.logspace(-12, -0.5, 60):
        e = 0.0 if cks_delta(rho, 0.0) <= d else epsilon_of(rho, float(d))
        best = min(best, math.exp(e) * 2.0 ** (-t) + float(d))
    return min(1.0, best + delta_extra)


def expected_exposure_bound(epsilon: float, delta: float, R: int) -> float:
    """Prop. 1504: E[exposure] <= (eps + 1) / ln 2 + delta log2 |R|."""
    return (epsilon + 1) / math.log(2) + delta * math.log2(R)


# ------------------------------------------------------------------ near duplicates
def _shingles(text, n=3):
    from ..text import WORD
    import re
    w = [x.lower() for x in re.findall(r"\w+", str(text or "")) if WORD.search(x)]
    if len(w) < n:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _minhash(sh: set, perms: int, seed: int = 1505) -> np.ndarray:
    if not sh:
        return np.full(perms, np.iinfo(np.uint64).max, dtype=np.uint64)
    h = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") for s in sh],
                 dtype=np.uint64)
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2 ** 61 - 1, perms, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2 ** 61 - 1, perms, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return (h[None, :] * a[:, None] + b[:, None]).min(axis=1)


def near_duplicates(train, synthetic, threshold: float = 0.7, bands: int = 32, rows: int = 4) -> dict:
    """For each synthetic text, the best Jaccard similarity of its word 3-gram
    shingles to any training text among the LSH candidates (exactly computed),
    and the counts of exact copies and of near duplicates (>= threshold).  A
    pair of similarity s becomes a candidate with probability
    1 - (1 - s^rows)^bands (0.9998 at s = 0.7 with the defaults)."""
    perms = bands * rows
    tsh = [_shingles(t) for t in train]
    index: dict = defaultdict(list)
    for i, sh in enumerate(tsh):
        if not sh:
            continue
        m = _minhash(sh, perms)
        for b in range(bands):
            index[(b, m[b * rows:(b + 1) * rows].tobytes())].append(i)
    train_exact = Counter(" ".join(str(t).split()).lower() for t in train if t is not None)
    best, exact, near = [], 0, 0
    for s in synthetic:
        if s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        if train_exact.get(" ".join(str(s).split()).lower()):
            exact += 1
        sh = _shingles(s)
        if not sh:
            best.append(0.0)
            continue
        m = _minhash(sh, perms)
        cand = set()
        for b in range(bands):
            cand.update(index.get((b, m[b * rows:(b + 1) * rows].tobytes()), ()))
        j = max((len(sh & tsh[i]) / len(sh | tsh[i]) for i in cand), default=0.0)
        best.append(j)
        near += j >= threshold
    best_a = np.array(best) if best else np.zeros(0)
    n = len(best_a)
    return {"n": n, "exact_copies": exact, "exact_rate": exact / max(n, 1), "near_duplicates": int(near),
            "near_rate": near / max(n, 1), "threshold": threshold,
            "jaccard_p50": float(np.median(best_a)) if n else 0.0,
            "jaccard_p99": float(np.quantile(best_a, 0.99)) if n else 0.0}


# ------------------------------------------------------------------ extraction
def zlib_entropy(text: str) -> float:
    return float(len(zlib.compress(str(text).encode("utf-8"))))


def extraction_attack(samples, train, owners=None, k: int = 3, top: int = 100, logprob=None, near: float = 0.8) -> dict:
    """Rank `samples` by a membership score and count how many of the `top`
    ones reproduce (exactly, or with Jaccard >= `near`) a training text held by
    fewer than `k` people.  Score: log-likelihood / zlib entropy when `logprob`
    is given (Carlini et al. 2021, sec. 5.2), else the number of times a sample
    repeats."""
    samples = [s for s in samples if s is not None and str(s).strip()]
    rep = Counter(samples)
    uniq = list(rep)
    if logprob is not None:
        from ..text import tokenize
        score = [logprob(tokenize(s)) / max(zlib_entropy(s), 1.0) for s in uniq]
    else:
        score = [rep[s] for s in uniq]
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")[:top]
    owners = list(owners) if owners is not None else list(range(len(train)))
    held: dict = defaultdict(set)
    for t, o in zip(train, owners):
        if t is not None:
            held[" ".join(str(t).split()).lower()].add(o)
    rare = [t for t, o in held.items() if len(o) < k]
    rare_sh = [_shingles(t) for t in rare]
    rare_set = set(rare)
    hits = []
    for j in order:
        s = uniq[j]
        key = " ".join(s.split()).lower()
        if key in rare_set:
            hits.append(s)
            continue
        sh = _shingles(s)
        if sh and any(r and len(sh & r) / len(sh | r) >= near for r in rare_sh):
            hits.append(s)
    return {"samples": len(samples), "distinct": len(uniq), "top": len(order), "rare_training_texts": len(rare),
            "extracted": len(hits), "precision": len(hits) / max(len(order), 1), "examples": hits[:5]}
