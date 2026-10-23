"""Conditional text synthesis with a hierarchical Dirichlet back-off n-gram model.

p_c(w | h_n) = (n_c(h_n, w) + beta * p_c(w | h_{n-1})) / (n_c(h_n) + beta)
p_c(w | -)   = (n_c(w)      + beta * p_G(w))         / (n_c       + beta)
p_G(w)       = (n_G(w) + 1) / (n_G + V)

which is the posterior predictive of a Dirichlet prior on each next-token
distribution centred on the shorter context (MacKay & Peto 1995).  c is the
value of a conditioning column (or the row's latent segment).

Privacy guards, enforced by construction:
  * vocabulary k-anonymity: a token is in the vocabulary only if it occurs in
    at least `min_token_rows` distinct source rows; other tokens never appear;
  * no verbatim copying of rare content: an output is rejected if it contains
    a run of max_copy_ngram tokens that occurs in fewer than `min_token_rows`
    source rows, or equals a source text that occurs in fewer than
    `min_token_rows` rows (common phrases shared by many rows are allowed);
  * emails / long digit runs are replaced by typed placeholders before training.
This is a memorisation guard, not differential privacy (see docs).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

TOK = re.compile(r"<[A-Z]+>|\w+|[^\w\s]")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
DIGITS = re.compile(r"\d{5,}")
BOS, EOS = "<s>", "</s>"
CDF_CACHE_FLOATS = 30_000_000      # about 240 MB of cached next-token laws per text model


WORD = re.compile(r"[^\W_]")


def scrub(t: str) -> str:
    return DIGITS.sub("<NUM>", EMAIL.sub("<EMAIL>", t))


def tokenize(t: str):
    return TOK.findall(scrub(t))


def detok(tokens):
    s = " ".join(tokens)
    s = re.sub(r" ([.,!?;:%)])", r"\1", s)
    s = re.sub(r"([($]) ", r"\1", s)
    s = re.sub(r" ' ", "'", s)
    return s


class TextModel:
    def __init__(self, order=3, min_token_rows=3, max_copy_ngram=8, beta=2.0, max_retries=30, whole=False):
        self.N, self.k, self.L, self.beta, self.R = order, min_token_rows, max_copy_ngram, beta, max_retries
        # whole=True: each value is one token (a city within its state), so the model is
        # the Dirichlet posterior predictive of the value given the class, backed off to
        # the overall distribution; values held by fewer than k rows are never released
        self.whole = whole
        if whole:
            self.N = 1

    def fit(self, texts, classes=None):
        texts = [("" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t)) for t in texts]
        classes = [None] * len(texts) if classes is None else list(classes)
        toks = [([t] if t else []) for t in texts] if self.whole else [tokenize(t) for t in texts]
        rows_with = Counter()
        for ts in toks:
            rows_with.update(list(dict.fromkeys(ts)))      # first-seen order: the vocabulary (and the
            # draws) must not depend on the hash seed of the process
        self.vocab = [w for w, c in rows_with.items() if c >= self.k] + [EOS]
        self.widx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        self.null_rate = float(np.mean([len(t) == 0 for t in texts])) if texts else 0.0
        self.counts = defaultdict(lambda: defaultdict(Counter))   # class -> context -> Counter
        G = np.ones(V)
        ngram_rows = Counter()
        text_freq = Counter(" ".join(ts) for ts in toks)
        self.rare_texts = {t for t, c in text_freq.items() if c < self.k}
        for ts, c in zip(toks, classes):
            if not ts:
                continue
            ngram_rows.update({tuple(ts[i:i + self.L]) for i in range(len(ts) - self.L + 1)})
            seq = [BOS] * self.N + [w for w in ts if w in self.widx] + [EOS]
            for i in range(self.N, len(seq)):
                w = self.widx[seq[i]]
                G[w] += 1
                for n in range(self.N):
                    h = tuple(seq[i - n:i]) if n else ()
                    self.counts[c][h][w] += 1
        # n-grams of length L that occur in fewer than k source rows are identifying
        self.rare_ngrams = {g for g, c in ngram_rows.items() if c < self.k}
        # share of non-empty rows that keep at least one word once rare tokens are
        # removed; below one half the words are mostly unique to their row (names,
        # titles), the n-gram model can only recombine punctuation and filler words,
        # and surrogate labels ("Artist 00042") are generated instead
        keep = [any(w in self.widx and (self.whole or WORD.search(w)) for w in ts) for ts in toks if ts]
        self.word_coverage = float(np.mean(keep)) if keep else 0.0
        self.surrogate = self.word_coverage < 0.5
        self.pG = G / G.sum()
        self.classes = set(self.counts)
        self._cdf_cache, self._whole_cache = {}, {}
        return self

    def dist(self, c, hist):
        p = self.pG
        tab = self.counts.get(c) if c in self.classes else None
        if tab is None:
            return p
        for n in range(self.N):
            h = tuple(hist[len(hist) - n:]) if n else ()
            cnt = tab.get(h)
            if not cnt:
                break
            v = np.zeros(len(self.vocab))
            idx = np.fromiter(cnt.keys(), dtype=np.int64)
            v[idx] = np.fromiter(cnt.values(), dtype=float)
            p = (v + self.beta * p) / (v.sum() + self.beta)
        return p

    def _cdf(self, c, hist):
        """Normalised cumulative next-token law for (class, last N-1 tokens),
        cached: the law depends on nothing else.  Drawing u ~ U(0,1) and taking
        searchsorted(cdf, u, 'right') is exactly what Generator.choice(V, p=p) does,
        so the draws (and the random stream) are identical to the uncached code."""
        key = (c, tuple(hist[len(hist) - self.N + 1:]) if self.N > 1 else ())
        cache = self.__dict__.setdefault("_cdf_cache", {})
        cdf = cache.get(key)
        if cdf is None:
            cdf = self.dist(c, hist).cumsum()
            cdf /= cdf[-1]
            if len(cache) * len(cdf) > CDF_CACHE_FLOATS:
                cache.clear()
            cache[key] = cdf
        return cdf

    def _one(self, c, rng, max_len=120):
        hist = [BOS] * self.N
        out = []
        for _ in range(max_len):
            w = self.vocab[int(self._cdf(c, hist).searchsorted(rng.random(), side="right"))]
            if w == EOS:
                break
            out.append(w)
            hist.append(w)
        return out

    def violates(self, toks):
        if " ".join(toks) in self.rare_texts:
            return True
        for i in range(len(toks) - self.L + 1):
            if tuple(toks[i:i + self.L]) in self.rare_ngrams:
                return True
        return False

    def generate(self, classes, rng):
        out, rejected = [], 0
        label = getattr(self, "label", "Item")
        for c in classes:
            if rng.random() < self.null_rate:
                out.append(None)
                continue
            if getattr(self, "surrogate", False):
                out.append(f"{label} {rng.integers(10**5):05d}")
                continue
            if self.whole:
                cache = self.__dict__.setdefault("_whole_cache", {})
                cdf = cache.get(c)
                if cdf is None:
                    p = self.dist(c, [BOS]).copy()
                    p[self.widx[EOS]] = 0.0
                    cdf = (p / p.sum()).cumsum()
                    cdf /= cdf[-1]
                    cache[c] = cdf
                out.append(self.vocab[int(cdf.searchsorted(rng.random(), side="right"))])
                continue
            for _ in range(self.R):
                t = _trim(self._one(c, rng))
                if t and not self.violates(t):
                    break
                rejected += 1
            else:
                t = []
            out.append(fill_placeholders(detok(t), rng) if t else f"{label} {rng.integers(10**5):05d}")
        self.last_rejected = rejected
        return out


def _trim(toks):
    """Drop punctuation that opens or closes a generated text and reject texts
    without a word (", &" is not a label)."""
    while toks and not WORD.search(toks[0]) and toks[0] not in ("(", '"', "'", "<NUM>", "<EMAIL>"):
        toks = toks[1:]
    while toks and not WORD.search(toks[-1]) and toks[-1] not in (")", '"', "'", ".", "!", "?", "<NUM>", "<EMAIL>"):
        toks = toks[:-1]
    if toks.count("(") != toks.count(")") or toks.count("[") != toks.count("]"):
        return []
    return toks if any(WORD.search(w) for w in toks) else []


def fill_placeholders(s, rng):
    s = s.replace("<EMAIL>", f"user{rng.integers(10**5)}@example.com")
    while "<NUM>" in s:
        s = s.replace("<NUM>", str(rng.integers(10**5, 10**8)), 1)
    return s
