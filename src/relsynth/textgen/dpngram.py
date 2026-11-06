# SPDX-License-Identifier: Apache-2.0
"""A differentially private, row-conditioned n-gram text model (Props. 1501, 1502).

It needs no neural network, no GPU and no model weights, so it runs in an
air-gapped installation and is the fallback of every other text method.

The release follows DP n-gram extraction (Kim, Gopi, Kulkarni and Yekhanin,
NeurIPS 2021) with user-level contribution bounding (Wilson et al., PoPETs
2020), level by level:

* each person keeps at most `max_texts` texts (chosen at random) of at most
  `max_tokens` tokens, after PII redaction;
* level n (n = 0 .. order-1) counts, for every key (class, context of n
  tokens, next token), the number of *people* whose texts contain it; each
  person adds 1 to at most `max_keys` keys (a random subset), so one person
  changes the count vector by at most sqrt(max_keys) in L2 and any single key
  by at most 1;
* the counts get discrete Gaussian noise (OpenDP, as in DP mode) and a key is
  released when its noisy count exceeds the partition-selection threshold of
  Prop. 903; level n only counts keys whose shorter context and whose prefix
  were released at level n-1, a public restriction;
* the share of empty texts is released with its own small budget.

The released noisy counts take the place of the counts in the hierarchical
Dirichlet back-off model of `relsynth.text` (Prop. 12), so generation is
post-processing.  The guarantee is user-level zCDP plus the partition
selection delta, for add/remove of one person with all their texts; the
charges go to the same `Accountant` as DP mode.
"""
from __future__ import annotations

import math
import secrets
from collections import Counter, defaultdict

import numpy as np

from ..text import BOS, EOS, WORD, TextModel, tokenize


def _rng() -> np.random.Generator:
    return np.random.default_rng(secrets.randbits(128))


class DPNgramModel(TextModel):
    """`TextModel` whose counts are released with user-level DP."""

    def __init__(self, order: int = 2, max_texts: int = 4, max_tokens: int = 40, max_keys: int = 64,
                 beta: float = 2.0, max_retries: int = 30, null_share: float = 0.05, level_weights=None):
        super().__init__(order=order, min_token_rows=1, max_copy_ngram=10 ** 6, beta=beta, max_retries=max_retries)
        self.max_texts, self.max_tokens, self.max_keys = int(max_texts), int(max_tokens), int(max_keys)
        self.null_share = float(null_share)
        self.level_weights = list(level_weights) if level_weights else [1.0] * order
        if len(self.level_weights) != order:
            raise ValueError("level_weights needs one weight per n-gram level")
        self.release: dict = {}

    # ------------------------------------------------------------ bounding
    def _bound(self, texts, owners, classes, rng):
        by: dict = defaultdict(list)
        for i, o in enumerate(owners):
            by[o].append(i)
        people = []
        for idx in by.values():
            if len(idx) > self.max_texts:
                idx = list(rng.choice(idx, self.max_texts, replace=False))
            rows = []
            for i in idx:
                t = texts[i]
                empty = t is None or (isinstance(t, float) and math.isnan(t)) or str(t) == ""
                rows.append((None if empty else tokenize(str(t))[: self.max_tokens], str(classes[i])))
            people.append(rows)
        return people

    def _keys_of(self, rows, n, vocab, rng):
        keys = set()
        for toks, c in rows:
            if toks is None:
                continue
            ts = toks if vocab is None else [w for w in toks if w in vocab]
            seq = [BOS] * self.N + ts + [EOS]
            for i in range(self.N, len(seq)):
                keys.add((c, tuple(seq[i - n:i]) if n else (), seq[i]))
        keys = list(keys)
        if len(keys) > self.max_keys:
            keys = [keys[j] for j in rng.choice(len(keys), self.max_keys, replace=False)]
        return keys

    # ------------------------------------------------------------ fit
    def fit_dp(self, texts, owners, classes, acct, rho: float, delta: float, stage: str = "fit", label: str = "text"):
        """Release the model from (text, owner, class) rows with `rho` zCDP and
        `delta`, charged to `acct` (an `relsynth.dp.mechanisms.Accountant`)."""
        from ..dp.mechanisms import Charge, gaussian, sigma_for, threshold_for
        rng = _rng()
        texts = list(texts)
        owners = list(owners)
        classes = ["None"] * len(texts) if classes is None else [str(c) for c in classes]
        people = self._bound(texts, owners, classes, rng)
        rho_null = rho * self.null_share
        rho_lv = [(rho - rho_null) * w / sum(self.level_weights) for w in self.level_weights]
        d_lv = delta / self.N
        # share of empty texts: a person adds at most max_texts rows to (empty, non-empty)
        nulls = np.array([sum(t is None for t, _ in p) for p in people], dtype=np.int64)
        tot = np.array([len(p) for p in people], dtype=np.int64)
        noisy = gaussian([int(nulls.sum()), int((tot - nulls).sum())], self.max_texts, rho_null, acct, stage,
                         f"{label}: empty share")
        self.null_rate = float(np.clip(noisy[0], 0, None) / max(np.clip(noisy, 0, None).sum(), 1))
        released: list[dict] = []
        vocab = None
        sens = math.sqrt(self.max_keys)
        for n in range(self.N):
            cnt: Counter = Counter()
            for rows in people:
                for k in self._keys_of(rows, n, vocab, rng):
                    if n == 0 or (self._allowed(k, released[n - 1])):
                        cnt[k] += 1
            keys = list(cnt)
            sig = sigma_for(sens, rho_lv[n])
            tau = threshold_for(1, sig, d_lv, self.max_keys)
            noisy = gaussian([cnt[k] for k in keys], sens, rho_lv[n], acct, stage, f"{label}: {n + 1}-grams")
            acct.charge(Charge(stage, f"{label}: {n + 1}-gram selection (delta)", 0.0, d_lv))
            rel = {k: float(v) for k, v in zip(keys, noisy) if v > tau}
            released.append(rel)
            if n == 0:
                vocab = {w for (_, _, w) in rel if w != EOS}
            self.release[f"level{n + 1}"] = {"candidates": len(keys), "released": len(rel), "sigma": sig,
                                             "threshold": tau, "rho": rho_lv[n], "delta": d_lv}
        self._build(released, vocab or set())
        self.release.update({"people": len(people), "rows_kept": int(tot.sum()), "max_texts": self.max_texts,
                             "max_tokens": self.max_tokens, "max_keys": self.max_keys, "rho": rho, "delta": delta,
                             "vocabulary": len(self.vocab) - 1})
        return self

    @staticmethod
    def _allowed(key, prev: dict) -> bool:
        c, h, w = key
        # the shorter context (h[1:], w) and the context itself (h[:-1] -> h[-1]) were
        # released; the start padding is public
        return (c, h[1:], w) in prev and (h[-1] == BOS or (c, h[:-1], h[-1]) in prev)

    def _build(self, released, vocab):
        order = sorted(vocab, key=lambda w: (-sum(v for (c, h, ww), v in released[0].items() if ww == w), w))
        self.vocab = order + [EOS]
        self.widx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        self.counts = defaultdict(lambda: defaultdict(Counter))
        G = np.ones(V)
        for lvl in released:
            for (c, h, w), v in lvl.items():
                if w in self.widx:
                    self.counts[c][h][self.widx[w]] = v
                    if not h:
                        G[self.widx[w]] += v
        self.counts = {c: dict(t) for c, t in self.counts.items()}
        self.pG = G / G.sum()
        self.classes = set(self.counts)
        self.rare_texts, self.rare_ngrams = set(), set()     # DP replaces the memorisation guards
        self.word_coverage = 1.0 if any(WORD.search(w) for w in self.vocab) else 0.0
        self.surrogate = self.word_coverage == 0.0
        self._cdf_cache, self._whole_cache = {}, {}

    # ------------------------------------------------------------ scoring (audits)
    def logprob(self, tokens, c) -> float:
        """log P(tokens, EOS | class) under the model (natural log).  A token
        outside the vocabulary, which the model never emits, scores a fixed
        log(1e-12) and stays in the history (the next token backs off past it),
        so candidates that share such a token are still ranked by the rest."""
        hist = [BOS] * self.N
        lp = 0.0
        for w in list(tokens) + [EOS]:
            j = self.widx.get(w)
            if j is None:
                lp += math.log(1e-12)
                hist.append(w)
                continue
            p = self.dist(c, hist)
            lp += math.log(max(p[j] / p.sum(), 1e-300))
            hist.append(w)
        return lp


def ngram_logprob(model: TextModel, tokens, c) -> float:
    """log P(tokens, EOS | class) of any `TextModel` (the non-private one too)."""
    return DPNgramModel.logprob(model, tokens, c)  # type: ignore[arg-type]  # uses TextModel fields only
