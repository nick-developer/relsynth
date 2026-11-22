# SPDX-License-Identifier: Apache-2.0
"""Hierarchical context VAE: a row VAE (level 1) and a family prior over row latents (level 2).

Level 1 - row VAE.  Encoder: one token per column (numeric x*w + b, categorical embedding;
FT-Transformer / TabSyn tokenizer), a CLS token, a PreNorm transformer, then mean and log
variance of the row latent z (m numbers).  Variant ``mlp``: the same inputs flattened through
an MLP.  Decoder: MLP z -> numeric means (Gaussian with a learned scale per column) and one
softmax per categorical column; it is factorised over columns given z, so it cannot ignore z
(the posterior-collapse mechanism of autoregressive decoders, Li et al. 2019, is absent).
Free bits per latent dimension and a linear KL warm-up are applied anyway.

Level 2 - family prior p(z_i | z_<i, context, N, gap_i).  The context is the parent's columns
through their own tokenizer plus log N.  Input token i carries z_{i-1} (zeros for i = 1), the
gap to row i (sequences; 0 for unordered siblings) and the context.  Variants: ``none`` (no
history: rows independent given the context, the tvae-like baseline), ``gru``, and
``transformer`` (causal PreNorm self-attention with learned positions over a window).  The
output is a K-component diagonal Gaussian mixture (mixture density network) for z_i.

Training is two-stage, as TabSyn: stage 1 fits the row VAE on rows; stage 2 fits the prior on
the encoder's posterior means.  Unordered families are seen in a fresh random order each step
(order-agnostic autoregression) and sampled in random order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import Families


@dataclass
class Config:
    encoder: str = "transformer"       # transformer | mlp
    prior: str = "transformer"         # none | gru | transformer
    d_token: int = 32
    layers: int = 2
    heads: int = 4
    latent: int = 16
    d_prior: int = 96
    prior_layers: int = 2
    mixtures: int = 5
    window: int = 64
    beta: float = 0.5
    free_bits: float = 0.05
    vae_steps: int = 3000
    prior_steps: int = 3000
    batch_rows: int = 512
    batch_families: int = 64
    lr: float = 2e-3
    ordered: bool = False


class Tokenizer(nn.Module):
    def __init__(self, n_num: int, widths: list[int], d: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_num, d))
        self.emb = nn.ModuleList(nn.Embedding(k, d) for k in widths)

    def forward(self, X, C):                                  # -> (B, k, d)
        toks = []
        if X.shape[1]:
            toks.append(X.unsqueeze(-1) * self.w + self.b)
        for j, e in enumerate(self.emb):
            toks.append(e(C[:, j]).unsqueeze(1))
        return torch.cat(toks, 1)


class RowVAE(nn.Module):
    def __init__(self, n_num: int, widths: list[int], cfg: Config):
        super().__init__()
        self.cfg, self.n_num, self.widths = cfg, n_num, list(widths)
        d, m = cfg.d_token, cfg.latent
        if cfg.encoder == "transformer":
            self.tok = Tokenizer(n_num, widths, d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d, cfg.heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * m))
        else:
            inp = n_num + sum(widths)
            self.mlp = nn.Sequential(nn.Linear(inp, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2 * m))
        out = n_num + sum(widths)
        self.dec = nn.Sequential(nn.Linear(m, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out))
        self.log_sigma = nn.Parameter(torch.zeros(n_num))

    def encode(self, X, C):
        if self.cfg.encoder == "transformer":
            t = self.tok(X, C)
            t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
            h = self.head(self.body(t)[:, 0])
        else:
            oh = [F.one_hot(C[:, j], k).float() for j, k in enumerate(self.widths)]
            h = self.mlp(torch.cat([X] + oh, 1))
        mu, logvar = h.chunk(2, -1)
        return mu, logvar.clamp(-8, 8)

    def decode(self, z):
        o = self.dec(z)
        return o[:, : self.n_num], o[:, self.n_num:]

    def recon_loss(self, X, C, z):
        mean, logits = self.decode(z)
        ls = self.log_sigma
        nll = (0.5 * ((X - mean) / ls.exp()) ** 2 + ls + 0.5 * math.log(2 * math.pi)).sum(1) if self.n_num else 0.0
        o = 0
        for j, k in enumerate(self.widths):
            nll = nll + F.cross_entropy(logits[:, o:o + k], C[:, j], reduction="none")
            o += k
        return nll

    @torch.no_grad()
    def sample_rows(self, z, gen):
        mean, logits = self.decode(z)
        X = mean + self.log_sigma.exp() * torch.randn(mean.shape, generator=gen)
        cats, o = [], 0
        for k in self.widths:
            cats.append(torch.multinomial(F.softmax(logits[:, o:o + k], -1), 1, generator=gen)[:, 0])
            o += k
        C = torch.stack(cats, 1) if cats else torch.zeros(len(z), 0, dtype=torch.long)
        return X, C


class MDN(nn.Module):
    def __init__(self, d: int, m: int, k: int):
        super().__init__()
        self.m, self.k = m, k
        self.out = nn.Linear(d, k * (1 + 2 * m))

    def params(self, h):
        o = self.out(h)
        logit = o[..., : self.k]
        mu, ls = o[..., self.k:].reshape(*o.shape[:-1], self.k, 2 * self.m).chunk(2, -1)
        return logit, mu, ls.clamp(-6, 3)

    def nll(self, h, z):                                       # z: (..., m)
        logit, mu, ls = self.params(h)
        lp = -0.5 * (((z.unsqueeze(-2) - mu) / ls.exp()) ** 2 + 2 * ls + math.log(2 * math.pi)).sum(-1)
        return -(torch.logsumexp(F.log_softmax(logit, -1) + lp, -1))

    @torch.no_grad()
    def sample(self, h, gen):
        logit, mu, ls = self.params(h)
        j = torch.multinomial(F.softmax(logit, -1), 1, generator=gen)[:, 0]
        ar = torch.arange(len(j))
        return mu[ar, j] + ls[ar, j].exp() * torch.randn(mu[ar, j].shape, generator=gen)


class FamilyPrior(nn.Module):
    def __init__(self, ctx_num: int, ctx_widths: list[int], cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, m = cfg.d_prior, cfg.latent
        self.ctx_tok = Tokenizer(ctx_num, ctx_widths, d) if (ctx_num or ctx_widths) else None
        self.ctx_n = nn.Linear(1, d)
        self.inp = nn.Linear(m + 1, d)
        if cfg.prior == "transformer":
            self.pos = nn.Embedding(cfg.window, d)
            layer = nn.TransformerEncoderLayer(d, cfg.heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, cfg.prior_layers, enable_nested_tensor=False)
        elif cfg.prior == "gru":
            self.body = nn.GRU(d, d, num_layers=cfg.prior_layers, batch_first=True)
        else:
            self.body = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.mdn = MDN(d, m, cfg.mixtures)

    def context(self, cX, cC, n):
        c = self.ctx_n(torch.log(n.float()).unsqueeze(-1))
        if self.ctx_tok is not None:
            c = c + self.ctx_tok(cX, cC).mean(1)
        return c                                               # (B, d)

    def hidden(self, zprev, gap, ctx, mask=None, state=None, pos0=0):
        """zprev, gap: (B, L, m), (B, L); ctx (B, d) -> hidden (B, L, d)."""
        u = self.inp(torch.cat([zprev, gap.unsqueeze(-1)], -1)) + ctx.unsqueeze(1)
        if self.cfg.prior == "none":
            u = ctx.unsqueeze(1) + self.inp(torch.cat([torch.zeros_like(zprev), gap.unsqueeze(-1)], -1))
            return self.norm(self.body(u)), None
        if self.cfg.prior == "gru":
            h, state = self.body(u, state)
            return self.norm(h), state
        L = u.shape[1]
        u = u + self.pos(torch.arange(pos0, pos0 + L).clamp(max=self.cfg.window - 1))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        h = self.body(u, mask=causal, src_key_padding_mask=None if mask is None else ~mask)
        return self.norm(h), None


def _windows(fam: Families, idx, mu: np.ndarray, cfg: Config, rng: np.random.Generator):
    """A batch of padded windows (one per family in idx) of posterior means and gaps."""
    W = cfg.window
    B, m = len(idx), mu.shape[1]
    Z = np.zeros((B, W, m), np.float32)
    G = np.zeros((B, W), np.float32)
    M = np.zeros((B, W), bool)
    for b, f in enumerate(idx):
        r = np.arange(fam.off[f], fam.off[f + 1])
        if cfg.ordered:
            if len(r) > W:
                s = rng.integers(0, len(r) - W + 1)
                r = r[s:s + W]
        else:
            r = rng.permutation(r)[:W]
        Z[b, :len(r)] = mu[r]
        G[b, :len(r)] = fam.gap[r] if cfg.ordered else 0.0
        M[b, :len(r)] = True
    return Z, G, M


class ContextVAE:
    def __init__(self, n_num, widths, ctx_num, ctx_widths, cfg: Config, seed: int):
        torch.manual_seed(seed)
        self.cfg = cfg
        self.vae = RowVAE(n_num, widths, cfg)
        self.prior = FamilyPrior(ctx_num, ctx_widths, cfg)
        self.log = {}

    # ---------------------------------------------------------------- stage 1
    def fit_vae(self, fam: Families, rng):
        cfg, n = self.cfg, len(fam.X)
        X, C = torch.as_tensor(fam.X), torch.as_tensor(fam.C)
        opt = torch.optim.AdamW(self.vae.parameters(), lr=cfg.lr, weight_decay=1e-5)
        warm = max(1, cfg.vae_steps // 5)
        for step in range(cfg.vae_steps):
            b = torch.as_tensor(rng.integers(0, n, cfg.batch_rows))
            mu, lv = self.vae.encode(X[b], C[b])
            z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
            kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv)              # (B, m)
            kl_fb = torch.clamp(kl.mean(0), min=cfg.free_bits).sum()
            beta = cfg.beta * min(1.0, (step + 1) / warm)
            loss = self.vae.recon_loss(X[b], C[b], z).mean() + beta * kl_fb
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.vae.parameters(), 5.0)
            opt.step()
        with torch.no_grad():
            mu, lv = self.encode_all(fam)
            kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv)
            self.log["active_units"] = int((mu.var(0) > 0.01).sum())   # Burda et al. 2016 activity
            self.log["kl_per_dim"] = float(kl.mean())
        return self

    @torch.no_grad()
    def encode_all(self, fam: Families):
        out_mu, out_lv = [], []
        X, C = torch.as_tensor(fam.X), torch.as_tensor(fam.C)
        for s in range(0, len(X), 8192):
            mu, lv = self.vae.encode(X[s:s + 8192], C[s:s + 8192])
            out_mu.append(mu)
            out_lv.append(lv)
        return torch.cat(out_mu), torch.cat(out_lv)

    # ---------------------------------------------------------------- stage 2
    def _batch_nll(self, fam, idx, mu, rng):
        Z, G, M = _windows(fam, idx, mu, self.cfg, rng)
        Z, G, M = torch.as_tensor(Z), torch.as_tensor(G), torch.as_tensor(M)
        zprev = torch.cat([torch.zeros_like(Z[:, :1]), Z[:, :-1]], 1)
        n = torch.as_tensor(fam.sizes[idx])
        ctx = self.prior.context(torch.as_tensor(fam.ctx_X[idx]), torch.as_tensor(fam.ctx_C[idx]), n)
        h, _ = self.prior.hidden(zprev, G, ctx, mask=M)
        nll = self.prior.mdn.nll(h, Z)
        return (nll * M).sum() / M.sum()

    def fit_prior(self, fam: Families, rng):
        mu = self.encode_all(fam)[0].numpy()
        self.mu_train = mu
        opt = torch.optim.AdamW(self.prior.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        for _ in range(self.cfg.prior_steps):
            idx = rng.integers(0, len(fam), self.cfg.batch_families)
            loss = self._batch_nll(fam, idx, mu, rng)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.prior.parameters(), 5.0)
            opt.step()
        return self

    @torch.no_grad()
    def prior_nll(self, fam: Families, rng, families=512):
        """Mean per-row prior NLL of families (held out ones: generalisation; two orders: order sensitivity)."""
        mu = self.encode_all(fam)[0].numpy()
        idx = np.arange(min(len(fam), families))
        return float(self._batch_nll(fam, idx, mu, rng))

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, sizes: np.ndarray, ctx_X, ctx_C, gaps: list[np.ndarray] | None, seed: int, chunk: int = 512):
        """Rows for families of the given sizes and contexts (the engine's counts); returns X, C, off.
        Families are drawn in chunks of similar size, so a short family does not wait for the longest."""
        gen = torch.Generator().manual_seed(seed)
        order = np.argsort(sizes, kind="stable")
        Zs = [None] * len(sizes)
        for s0 in range(0, len(order), chunk):
            ix = order[s0:s0 + chunk]
            Z = self._sample_latents(sizes[ix], ctx_X[ix], ctx_C[ix],
                                     [gaps[i] for i in ix] if gaps is not None else None, gen)
            for b, i in enumerate(ix):
                Zs[i] = Z[b, :sizes[i]]
        X, C = self.vae.sample_rows(torch.cat(Zs, 0), gen)
        off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        return X.numpy(), C.numpy(), off

    def _sample_latents(self, sizes, ctx_X, ctx_C, gaps, gen):
        cfg, W, m = self.cfg, self.cfg.window, self.cfg.latent
        B, L = len(sizes), int(sizes.max())
        n = torch.as_tensor(sizes)
        ctx = self.prior.context(torch.as_tensor(ctx_X), torch.as_tensor(ctx_C), n)
        G = np.zeros((B, L), np.float32)
        if gaps is not None:
            for b, g in enumerate(gaps):
                G[b, :len(g)] = g
        G = torch.as_tensor(G)
        Z = torch.zeros(B, L, m)
        state = None
        for i in range(L):
            if cfg.prior == "transformer":
                s = max(0, i - W + 1)
                zprev = torch.cat([torch.zeros(B, 1, m), Z[:, :i]], 1)[:, s:i + 1]
                h, _ = self.prior.hidden(zprev, G[:, s:i + 1], ctx)
                h = h[:, -1]
            elif cfg.prior == "gru":
                zp = Z[:, i - 1:i] if i else torch.zeros(B, 1, m)
                h, state = self.prior.hidden(zp, G[:, i:i + 1], ctx, state=state)
                h = h[:, -1]
            else:
                h, _ = self.prior.hidden(torch.zeros(B, 1, m), G[:, i:i + 1], ctx)
                h = h[:, -1]
            Z[:, i] = self.prior.mdn.sample(h, gen)
        return Z
