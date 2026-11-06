# SPDX-License-Identifier: Apache-2.0
"""Text embeddings that need no model and learn nothing from the data.

Word unigrams, word bigrams and character 3-grams are hashed into 2^18
buckets with a sign hash (Weinberger et al. ICML 2009), sublinear counts,
L2-normalised, then multiplied by a fixed sparse random projection (Achlioptas
2003; Li, Hastie and Church KDD 2006) to `dim` dimensions.  Both hashes and the
projection are functions of a public constant only, so embedding a private text
reveals nothing about any other text: Private Evolution can use it for its
nearest-neighbour votes without spending budget on the embedding (Prop. 1503).
By Johnson-Lindenstrauss, inner products of the hashed vectors are preserved up
to a relative error of order sqrt(log n / dim).

A local sentence-transformers model can replace it (`SentenceEmbedder`).
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_W = re.compile(r"[^\W_]+|<[A-Z]+>", re.UNICODE)
BUCKETS = 1 << 18
SEED = 1503                         # public constant: the projection is the same in every installation


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8, key=b"relsynth-text").digest(), "little")


def features(text: str) -> dict[int, float]:
    t = str(text or "").lower()
    words = _W.findall(t)
    grams = words + [a + " " + b for a, b in zip(words, words[1:])]
    padded = f" {t} "
    grams += ["#" + padded[i:i + 3] for i in range(len(padded) - 2)]
    f: dict[int, float] = {}
    for g in grams:
        h = _h(g)
        b = h % BUCKETS
        f[b] = f.get(b, 0.0) + (1.0 if (h >> 63) & 1 else -1.0)
    return f


class HashEmbedder:
    """Data-independent embedding: hashed n-gram features x sparse random projection."""
    name = "hash-rp"

    def __init__(self, dim: int = 256, density: float = 1 / 32):
        self.dim = dim
        rng = np.random.default_rng(SEED)
        # Achlioptas / Li sparse projection, one row per bucket, generated lazily per
        # bucket from the bucket's own seed so memory stays O(touched buckets)
        self._density = density
        self._cache: dict[int, np.ndarray] = {}
        self._rng_seed = int(rng.integers(2 ** 62))

    def _row(self, b: int) -> np.ndarray:
        r = self._cache.get(b)
        if r is None:
            g = np.random.default_rng((self._rng_seed, b))
            u = g.random(self.dim)
            s = 1 / np.sqrt(self._density * self.dim)
            r = np.where(u < self._density / 2, -s, np.where(u < self._density, s, 0.0))
            if len(self._cache) < 2_000_000:
                self._cache[b] = r
        return r

    def __call__(self, texts) -> np.ndarray:
        X = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            f = features(t)
            if not f:
                continue
            v = np.array(list(f.values()))
            v = np.sign(v) * np.log1p(np.abs(v))
            v /= np.linalg.norm(v) or 1.0
            for b, w in zip(f.keys(), v):
                X[i] += w * self._row(b)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(n > 0, n, 1.0)


class SentenceEmbedder:
    """A local sentence-transformers model (folder; never downloads)."""

    def __init__(self, model: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device=device, local_files_only=True)
        self.name = f"st:{model}"

    def __call__(self, texts) -> np.ndarray:
        return np.asarray(self.m.encode([str(t or "") for t in texts], normalize_embeddings=True))


def make_embedder(cfg=None):
    if cfg in (None, "hash", "hash-rp"):
        return HashEmbedder()
    if isinstance(cfg, dict) and cfg.get("name") in ("hash", "hash-rp"):
        return HashEmbedder(dim=int(cfg.get("dim", 256)))
    if isinstance(cfg, dict) and cfg.get("name") == "sentence-transformers":
        return SentenceEmbedder(cfg["model"], cfg.get("device"))
    from ..spec import SpecError
    raise SpecError(f"unknown text embedder {cfg!r}: hash or {{name: sentence-transformers, model: <folder>}}")
