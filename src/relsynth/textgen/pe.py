# SPDX-License-Identifier: Apache-2.0
"""Private Evolution for row-conditioned text (Prop. 1503).

Private Evolution (Lin et al. ICLR 2024; for text Xie et al., Aug-PE,
ICML 2024) never trains on the private texts.  A *public* generator (an API:
a local pretrained language model, or an n-gram model fitted to a public
corpus) proposes texts; the private texts only vote, through a DP histogram,
for the proposals they are closest to; the winners are varied by the API and
the loop repeats.

Per iteration, per class (the row's conditioning values):

1. every private text (each person keeps at most `max_texts`) finds its
   nearest proposal of its own class in a data-independent embedding
   (`relsynth.textgen.embed`);
2. each person's votes are scaled to L2 norm `SCALE` and floored, so a person
   moves the integer histogram over all proposals by at most SCALE in L2;
3. the histogram gets discrete Gaussian noise (OpenDP) and the proposals are
   resampled in proportion to max(noisy - H, 0), H = `threshold` sigma;
4. the API varies the resampled proposals into the next generation.

T iterations cost rho = T / (2 sigma'^2) with sigma' = sigma / SCALE, plus the
release of the class levels (partition selection); the guarantee is
user-level.  The output is post-processing of the noisy histograms and the
public API.
"""
from __future__ import annotations

import math
import secrets

import numpy as np

from .embed import make_embedder

SCALE = 64


class NgramApi:
    """Public API from an n-gram model fitted to a *public* corpus."""

    def __init__(self, texts, classes=None, order: int = 3, seed: int = 0):
        from ..text import TextModel
        self.m = TextModel(order=order, min_token_rows=1, max_copy_ngram=10 ** 6).fit(list(texts), classes)
        self.rng = np.random.default_rng(seed)

    def random(self, cls: str, n: int) -> list[str]:
        out = self.m.generate([cls] * n, self.rng)
        return [t or "" for t in out]

    def variation(self, texts, cls: str) -> list[str]:
        from ..text import BOS, EOS, _trim, detok, tokenize
        out = []
        for t in texts:
            toks = tokenize(t)
            cut = int(self.rng.integers(0, max(len(toks), 1)))
            hist, new = [BOS] * self.m.N + toks[:cut], list(toks[:cut])
            for _ in range(60):
                w = self.m.vocab[int(self.m._cdf(cls, hist).searchsorted(self.rng.random(), side="right"))]
                if w == EOS:
                    break
                new.append(w)
                hist.append(w)
            new = _trim(new)
            out.append(detok(new) if new else t)
        return out


class LMApi:
    """Public API from a local pretrained causal LM (no fine-tuning)."""

    def __init__(self, model: str, seed: int = 0, **kw):
        from .lm import LMTextModel
        self.lm = LMTextModel(model=model, seed=seed, **kw)
        self.lm._load()
        self.lm.null_rate = 0.0
        self.rng = np.random.default_rng(seed)
        self.key = "condition"

    def random(self, cls: str, n: int) -> list[str]:
        return [t or "" for t in self.lm.generate([{self.key: cls}] * n, self.rng)]

    def variation(self, texts, cls: str) -> list[str]:
        """Keep a random prefix of each text (at least one word) and let the model continue it."""
        torch = __import__("torch")
        from .lm import prompt_of
        out = []
        with torch.no_grad():
            for t in texts:
                w = t.split()
                pre = " ".join(w[: int(self.rng.integers(1, max(len(w), 2)))])
                p = self.lm.tok.encode(prompt_of({self.key: cls}) + pre)
                ids = torch.tensor([p], device=self.lm.dev)
                res = self.lm.net.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                           top_p=self.lm.top_p, max_new_tokens=self.lm.max_new_tokens,
                                           pad_token_id=self.lm.tok.pad_token_id, eos_token_id=self.lm.tok.eos_token_id)
                row = res[0, len(p):].tolist()
                if self.lm.tok.eos_token_id in row:
                    row = row[: row.index(self.lm.tok.eos_token_id)]
                out.append((pre + self.lm.tok.decode(row)).strip() or t)
        return out


class PETextModel:
    def __init__(self, api, iterations: int = 4, pool: int = 64, max_texts: int = 4, threshold: float = 2.0,
                 embed=None, class_share: float = 0.05):
        self.api, self.T, self.N, self.k = api, int(iterations), int(pool), int(max_texts)
        self.H, self.embed = float(threshold), make_embedder(embed)
        self.class_share = float(class_share)
        self.release: dict = {}
        self.weights: dict = {}
        self.pool: dict = {}

    def fit_dp(self, texts, owners, classes, acct, rho: float, delta: float, stage: str = "fit", label: str = "text"):
        from ..dp.mechanisms import Charge, gaussian, sigma_for, threshold_for
        rng = np.random.default_rng(secrets.randbits(128))
        classes = ["None"] * len(texts) if classes is None else [str(c) for c in classes]
        by: dict = {}
        n_rows = len(texts)
        for t, o, c in zip(texts, owners, classes):
            empty = t is None or (isinstance(t, float) and math.isnan(t)) or str(t) == ""
            by.setdefault(o, []).append((None if empty else str(t), c))
        people = []
        for rows in by.values():
            if len(rows) > self.k:
                rows = [rows[j] for j in rng.choice(len(rows), self.k, replace=False)]
            people.append(rows)
        # class levels and the empty share: one partition selection over (class) and (empty) keys
        rho_c = rho * self.class_share
        keys: dict = {}
        for rows in people:
            for key in {("class", c) for t, c in rows if t is not None} | ({("empty",)} if any(t is None for t, _ in rows)
                                                                          else set()):
                keys[key] = keys.get(key, 0) + 1
        kl = list(keys)
        sens = math.sqrt(self.k + 1)
        noisy = gaussian([keys[k] for k in kl], sens, rho_c, acct, stage, f"{label}: class levels")
        tau = threshold_for(1, sigma_for(sens, rho_c), delta, self.k + 1)
        acct.charge(Charge(stage, f"{label}: class selection (delta)", 0.0, delta))
        rel = {k: float(v) for k, v in zip(kl, noisy) if v > tau}
        levels = sorted(k[1] for k in rel if k[0] == "class")
        tot = sum(max(float(v), 0.0) for k, v in zip(kl, noisy) if k[0] == "class")
        self.null_rate = float(max(rel.get(("empty",), 0.0), 0.0) / max(tot + max(rel.get(("empty",), 0.0), 0.0), 1.0))
        self.levels = levels
        if not levels:
            self.pool, self.weights = {}, {}
            self.release = {"levels": 0, "note": "no class level passed the threshold"}
            return self
        rho_it = (rho - rho_c) / self.T
        sigma = sigma_for(SCALE, rho_it)
        pool = {c: self.api.random(c, self.N) for c in levels}
        priv = [[(t, c) for t, c in rows if t is not None and c in pool] for rows in people]
        flat = [(i, t, c) for i, rows in enumerate(priv) for t, c in rows]
        P = self.embed([t for _, t, _ in flat]) if flat else np.zeros((0, 1))
        hist_log = []
        for it in range(self.T):
            offs: dict = {}
            allp: list[str] = []
            for c in levels:
                offs[c] = len(allp)
                allp += pool[c]
            S = self.embed(allp)
            votes: dict = {}
            for (i, _t, c), x in zip(flat, P):
                lo = offs[c]
                j = lo + int(np.argmax(S[lo:lo + len(pool[c])] @ x))
                votes.setdefault(i, {}).setdefault(j, 0)
                votes[i][j] += 1
            h = np.zeros(len(allp), dtype=np.int64)
            for v in votes.values():
                norm = math.sqrt(sum(x * x for x in v.values()))
                for j, x in v.items():
                    h[j] += int(math.floor(SCALE * x / norm))
            noisy = gaussian(h, SCALE, rho_it, acct, stage, f"{label}: PE votes {it + 1}").astype(float)
            w_all = np.clip(noisy - self.H * sigma, 0, None)
            hist_log.append(float((w_all > 0).mean()))
            new = {}
            self.weights = {}
            for c in levels:
                lo = offs[c]
                w = w_all[lo:lo + len(pool[c])]
                p = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
                self.weights[c] = p
                if it == self.T - 1:
                    continue
                pick = [pool[c][j] for j in rng.choice(len(w), self.N, p=p)]
                new[c] = self.api.variation(pick, c)
            if it < self.T - 1:
                pool = new
        self.pool = pool
        self.release = {"method": "Private Evolution", "levels": len(levels), "iterations": self.T, "pool": self.N,
                        "max_texts": self.k, "sigma_votes": sigma / SCALE, "rho": rho, "delta": delta,
                        "voted_share": hist_log, "people": len(people), "rows": n_rows}
        return self

    def generate(self, classes, rng):
        out = []
        for c in classes:
            if rng.random() < self.null_rate or not self.pool:
                out.append(None)
                continue
            c = str(c)
            if c not in self.pool:
                c = self.levels[int(rng.integers(len(self.levels)))]
            out.append(self.pool[c][int(rng.choice(len(self.pool[c]), p=self.weights[c]))])
        return out
