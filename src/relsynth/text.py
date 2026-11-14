# SPDX-License-Identifier: Apache-2.0
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
CHUNK_TOKENS = 4_000_000           # tokens per span of the packed fit
RARE_PART = 8_000_000              # runs per hash partition of the no-copy guard
FOLD_ROWS = 4_000_000              # span table entries folded into the running context table


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
        self._tabs = None
        # tokens are interned to integer ids in first-seen order and every pass below runs
        # over spans of about CHUNK_TOKENS tokens, so long free text costs 4 bytes a token
        # and a bounded working set, instead of a string per token, a tuple per n-gram and
        # a Counter per context (rel-trial's descriptions needed more than 12 GiB that way)
        tid: dict = {}
        rows = []
        for t in texts:
            ts = ([t] if t else []) if self.whole else tokenize(t)
            rows.append(np.fromiter((tid.setdefault(w, len(tid)) for w in ts), dtype=np.int32, count=len(ts)))
        tokens = list(tid)
        del tid
        nt = len(tokens)
        lens = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
        spans = _spans(lens, CHUNK_TOKENS)

        def span(lo, hi):
            n = lens[lo:hi]
            ids = np.concatenate(rows[lo:hi]).astype(np.int64) if n.sum() else np.zeros(0, np.int64)
            return ids, n, np.cumsum(n) - n

        # rows holding each token: the vocabulary keeps tokens of at least k rows, in
        # first-seen order (it, and the draws, must not depend on the hash seed)
        rows_with = np.zeros(nt, np.int64)
        for lo, hi in spans:
            ids, n, _ = span(lo, hi)
            if len(ids):
                u = np.unique(np.repeat(np.arange(len(n), dtype=np.int64), n) * nt + ids)
                rows_with += np.bincount(u % nt, minlength=nt)
        self.vocab = [w for w, c in zip(tokens, rows_with) if c >= self.k] + [EOS]
        self.widx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        self.null_rate = float(np.mean([len(t) == 0 for t in texts])) if texts else 0.0
        tmap = np.fromiter((self.widx.get(w, -1) for w in tokens), dtype=np.int64, count=nt)
        good = np.fromiter(((self.widx.get(w, -1) >= 0) and (self.whole or bool(WORD.search(w))) for w in tokens),
                           dtype=bool, count=nt)
        # word coverage (share of non-empty rows that keep a word of the vocabulary) and the
        # whole texts held by fewer than k rows; outputs are made of vocabulary tokens, so
        # only texts made entirely of them can ever be matched
        kept, text_freq = [], Counter()
        for lo, hi in spans:
            ids, n, st = span(lo, hi)
            ne = n > 0
            if ne.any():
                kept.append(np.logical_or.reduceat(good[ids], st[ne]))
            inv = np.zeros(len(ids) + 1, np.int64)
            np.cumsum(tmap[ids] < 0, out=inv[1:])
            for j in np.flatnonzero(inv[st + n] == inv[st]):
                text_freq[" ".join(tokens[x] for x in rows[lo + j])] += 1
        keep = np.concatenate(kept) if kept else np.zeros(0, bool)
        self.word_coverage = float(np.mean(keep)) if len(keep) else 0.0
        self.surrogate = self.word_coverage < 0.5
        self.rare_texts = {t for t, c in text_freq.items() if c < self.k}
        del text_freq, kept, keep
        self._rare_ngram_index(rows, lens, spans, span, tmap)
        # class -> context -> next-token counts, packed: key = class, then each context
        # token, in base V + 1 (the start padding is V unless "<s>" is itself a token)
        cls: dict = {}
        cid_row = np.fromiter((cls.setdefault(c, len(cls)) if n else -1 for c, n in zip(classes, lens.tolist())),
                              dtype=np.int64, count=len(lens))
        self.classes = set(cls)
        B, bos, N = V + 1, self.widx.get(BOS, V), self.N
        if B ** max(N - 1, 0) * max(len(cls), 1) >= 2 ** 62:
            del rows
            return self._fit_dicts(texts, classes)
        self._cls, self._B, self._bos = cls, B, bos
        G = np.ones(V)
        parts: list = [[] for _ in range(N)]
        for lo, hi in spans:
            ids, n, st = span(lo, hi)
            ne = n > 0
            if not ne.any():
                continue
            A = tmap[ids]
            q = A[A >= 0]
            m = np.add.reduceat((A >= 0).astype(np.int64), st[ne])
            del A, ids
            seg = m + N + 1                                # padding + vocabulary tokens + end
            S = np.full(int(seg.sum()), bos, np.int64)
            sstart = np.cumsum(seg) - seg
            P = np.arange(int((m + 1).sum())) - np.repeat(np.cumsum(m + 1) - (m + 1), m + 1) \
                + np.repeat(sstart + N, m + 1)
            qpos = np.arange(len(q)) - np.repeat(np.cumsum(m) - m, m)
            S[np.repeat(sstart, m) + N + qpos] = q
            S[sstart + seg - 1] = self.widx[EOS]
            del q, qpos
            w = S[P]
            G += np.bincount(w, minlength=V)[:V]
            tc = np.repeat(cid_row[lo:hi][ne], m + 1)
            for k in range(N):
                key = tc.copy()
                for j in range(k):
                    key *= B
                    key += S[P - k + j]
                parts[k].append(_runs(key, w.astype(np.int32), np.ones(len(w), np.int32)))
                # fold the span tables into one as soon as they outgrow it: the running table
                # plus what waits stays near twice the final size, not the sum of all spans
                if len(parts[k]) > 1 and sum(len(x[0]) for x in parts[k][1:]) >= max(len(parts[k][0][0]), FOLD_ROWS):
                    parts[k] = [_runs(*(np.concatenate([x[i] for x in parts[k]]) for i in range(3)))]
            del S, P, w, tc
        self._tabs = []
        for k in range(N):
            ks = np.concatenate([x[0] for x in parts[k]]) if parts[k] else np.zeros(0, np.int64)
            ws = np.concatenate([x[1] for x in parts[k]]) if parts[k] else np.zeros(0, np.int32)
            cs = np.concatenate([x[2] for x in parts[k]]) if parts[k] else np.zeros(0, np.int32)
            parts[k] = None
            ck, cw, cnt = _runs(ks, ws, cs)
            del ks, ws, cs
            newc = np.ones(len(ck), bool)
            newc[1:] = ck[1:] != ck[:-1]
            off = np.append(np.flatnonzero(newc), len(ck))
            self._tabs.append((ck[newc], off, cw, cnt))       # counts stay int32 (exact; cast in dist)
        self.counts = None
        self.pG = G / G.sum()
        self._cdf_cache, self._whole_cache = {}, {}
        return self

    def _rare_ngram_index(self, rows, lens, spans, span, tmap):
        """Runs of L vocabulary tokens that occur in fewer than k source rows, kept as
        sorted pairs of independent 64-bit hashes (a false match needs a 128-bit
        collision).  Runs with a token outside the vocabulary are left out: outputs
        are made of vocabulary tokens only, so they can never match one.  The runs are
        counted in hash partitions of at most about 8 million, hashed span by span."""
        L = self.L
        self.rare_ngrams = None
        empty = np.zeros(0, np.uint64)
        self._rare = (empty, empty)
        total = int(np.maximum(lens - L + 1, 0).sum())
        if not total:
            return
        nparts = max(1, -(-total // RARE_PART))
        rare1, rare2 = [], []
        for p in range(nparts):
            got = []
            for lo, hi in spans:
                ids, n, st = span(lo, hi)
                ok = n >= L
                if not ok.any():
                    continue
                A = tmap[ids]
                inv = np.zeros(len(A) + 1, np.int64)
                np.cumsum(A < 0, out=inv[1:])
                nw = n[ok] - L + 1
                ws = np.repeat(st[ok], nw) + (np.arange(int(nw.sum())) - np.repeat(np.cumsum(nw) - nw, nw))
                row = np.repeat(np.flatnonzero(ok) + lo, nw)
                clean = inv[ws + L] == inv[ws]
                ws, row = ws[clean], row[clean]
                if not len(ws):
                    continue
                h1, h2 = _window_hashes(A, ws, L)
                if nparts > 1:
                    sel = (h1 % np.uint64(nparts)) == p
                    h1, h2, row = h1[sel], h2[sel], row[sel]
                got.append((h1, h2, row))
            if not got:
                continue
            a = np.concatenate([g[0] for g in got])
            b = np.concatenate([g[1] for g in got])
            r = np.concatenate([g[2] for g in got])
            del got
            o = np.lexsort((r, b, a))
            a, b, r = a[o], b[o], r[o]
            del o
            new = np.ones(len(a), bool)
            new[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1]) | (r[1:] != r[:-1])
            a, b = a[new], b[new]                     # one entry per (n-gram, row)
            grp = np.ones(len(a), bool)
            grp[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
            at = np.flatnonzero(grp)
            nrows = np.diff(np.append(at, len(a)))
            rare1.append(a[at][nrows < self.k])
            rare2.append(b[at][nrows < self.k])
        if rare1:
            r1, r2 = np.concatenate(rare1), np.concatenate(rare2)
            o = np.lexsort((r2, r1))
            self._rare = (r1[o], r2[o])

    def _fit_dicts(self, texts, classes):
        """The reference implementation with nested Counters (used when the packed
        context keys of `fit` would not fit in 64 bits)."""
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
        if getattr(self, "_tabs", None) is not None:
            return self._dist_packed(c, hist)
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

    def _dist_packed(self, c, hist):
        p = self.pG
        if c not in self.classes:
            return p
        base = self._cls[c]
        for n in range(self.N):
            key = base
            for tok in (hist[len(hist) - n:] if n else ()):
                j = self._bos if tok == BOS else self.widx.get(tok)
                if j is None:                  # a token the model never saw as context
                    return p
                key = key * self._B + j
            keys, off, ws, cnt = self._tabs[n]
            i = int(np.searchsorted(keys, key))
            if i == len(keys) or keys[i] != key:
                break
            v = np.zeros(len(self.vocab))
            v[ws[off[i]:off[i + 1]]] = cnt[off[i]:off[i + 1]]
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
        if self.rare_ngrams is None:
            return self._rare_window(toks)
        for i in range(len(toks) - self.L + 1):
            if tuple(toks[i:i + self.L]) in self.rare_ngrams:
                return True
        return False

    def _rare_window(self, toks):
        r1, r2 = self._rare
        if len(toks) < self.L or not len(r1):
            return False
        ids = np.fromiter((self.widx.get(w, -1) for w in toks), dtype=np.int64, count=len(toks))
        if (ids < 0).any():                    # not an output of the model: check runs of known tokens
            return any(self._rare_window(list(g)) for g in _known_runs(toks, ids))
        h1, h2 = _window_hashes(ids, np.arange(len(toks) - self.L + 1), self.L)
        lo, hi = np.searchsorted(r1, h1, "left"), np.searchsorted(r1, h1, "right")
        return any(bool((r2[a:b] == y).any()) for a, b, y in zip(lo, hi, h2) if b > a)

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


def _spans(lens, size):
    """Consecutive row ranges [lo, hi) of about `size` tokens each (one row may exceed it)."""
    out, lo, acc = [], 0, 0
    for i, n in enumerate(lens.tolist()):
        acc += n
        if acc >= size:
            out.append((lo, i + 1))
            lo, acc = i + 1, 0
    if lo < len(lens):
        out.append((lo, len(lens)))
    return out


def _runs(key, w, cnt):
    """(key, w) pairs sorted, with the counts of equal pairs summed."""
    if not len(key):
        return key, w, cnt
    o = np.lexsort((w, key))
    ks, ws, cs = key[o], w[o], cnt[o]
    new = np.ones(len(ks), bool)
    new[1:] = (ks[1:] != ks[:-1]) | (ws[1:] != ws[:-1])
    at = np.flatnonzero(new)
    return ks[at], ws[at], np.add.reduceat(cs, at)


_M = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F))
_SEED = (np.uint64(0x243F6A8885A308D3), np.uint64(0x13198A2E03707344))


def _mix(h):
    h = h ^ (h >> np.uint64(30))
    h = h * np.uint64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> np.uint64(27))
    h = h * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def _window_hashes(ids, st, L):
    """Two independent 64-bit hashes of the runs ids[s:s + L] for each s in st."""
    out = []
    with np.errstate(over="ignore"):
        for m, seed in zip(_M, _SEED):
            h = np.full(len(st), seed, np.uint64)
            for j in range(L):
                h = _mix((h ^ (ids[st + j].astype(np.uint64) + np.uint64(1))) * m)
            out.append(h)
    return out[0], out[1]


def _known_runs(toks, ids):
    run = []
    for w, j in zip(toks, ids):
        if j < 0:
            if run:
                yield run
            run = []
        else:
            run.append(w)
    if run:
        yield run


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
