# SPDX-License-Identifier: Apache-2.0
"""Counter-based random streams.

A stream is a Philox4x64 generator (Salmon, Moraes, Dror and Shaw, "Parallel
random numbers: as easy as 1, 2, 3", SC 2011) whose 128-bit key is the first
16 bytes of SHA-256 over the stream's labels, e.g. (seed, dataset, table,
partition).  Its draws are a function of those labels alone: not of the order
in which partitions run, the worker that runs them or how many workers there
are (Prop. 900).  Philox passes BigCrush with distinct keys giving
statistically independent streams; SHA-256 makes two label tuples share a key
with probability 2^-128 per pair.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def stream_key(*labels) -> int:
    """The 128-bit Philox key of a label tuple (ints and strings)."""
    text = json.dumps([x if isinstance(x, (int, str)) else str(x) for x in labels], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:16], "little")


def stream(*labels) -> np.random.Generator:
    """A generator whose draws depend only on the labels."""
    return np.random.Generator(np.random.Philox(key=stream_key(*labels)))


class KeyedRunCache:
    """posterior.RunCache whose draws come from streams keyed by the cached
    quantity's name, so a value is the same whichever partition (or worker)
    first asks for it.  Keys that name an object by its memory address (the
    count laws of facts, `("dp", id(law), segment)`) are named by their fact
    through `bind`; keys that cannot be named use the caller's stream."""

    def __init__(self, base, seed: int, index: int):
        self.mode = base.mode
        self.store = dict(base.store)
        self.seed, self.index = int(seed), int(index)
        self._names: dict[int, str] = {}

    def bind(self, syn) -> KeyedRunCache:
        """Names of the count laws of syn's facts in this process."""
        self._names = {id(info["count_dp"]): f for f, info in syn.fact_info.items() if info.get("count_dp")}
        return self

    def _label(self, key):
        if isinstance(key, tuple) and key and key[0] == "dp":
            name = self._names.get(key[1])
            return None if name is None else ("dp", name) + tuple(key[2:])
        if isinstance(key, tuple) and all(isinstance(x, (str, int, np.integer)) for x in key) and key[:1] != ("counts",):
            return tuple(int(x) if isinstance(x, np.integer) else x for x in key)
        return None

    def bb(self, key, n, rng):
        label = self._label(key)
        skey = ("keyed",) + label if label is not None else key
        w = self.store.get(skey)
        if w is None or len(w) != n:
            if self.mode == "mean":
                w = np.full(n, 1.0 / n)
            else:
                g = stream(self.seed, self.index, "bb", *label) if label is not None else rng
                w = g.dirichlet(np.ones(n))
            self.store[skey] = w
        return w
