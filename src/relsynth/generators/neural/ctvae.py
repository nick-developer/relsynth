# SPDX-License-Identifier: Apache-2.0
"""``ctvae``: the context transformer VAE, a family generator (contract 1.1).

The rows of a fact that share an anchor row (the lines of an invoice, the
transactions of an account) form a *family*.  Row-wise generators (copula, cart,
tvae, ddpm) draw a family's rows independently given their context, so the
dependence *between* siblings (shared genres in an invoice, autocorrelated
balances in an account) is lost.  ``ctvae`` models a family jointly:

* a **row VAE** maps each row to a latent ``z`` (column tokens and a PreNorm
  transformer over them, as in FT-Transformer, Gorishniy et al. 2021, and
  TabSyn's VAE, Zhang et al. 2024), trained with free bits (Kingma et al. 2016)
  and a KL warm-up;
* a **family prior** (causal transformer by default, GRU as an option) draws the
  family's latents one row after the other given the earlier rows, the row's own
  context (segment, coarse segment, parent attributes), the family size and, for
  ordered families, the time gap to the previous row; a mixture density network
  gives p(z_i | history), two-stage training as in TabSyn;
* a **residual categorical head**: a row's levels are drawn from the decoder's
  logits for its latent plus a correction from the family history (initialised at
  zero), p(c_i | history, z_i).  This is the variant that met the acceptance rule
  of the ablation (experiments/context_vae, design record docs/research/
  context-transformer-vae.md); a latent-only prior lost categorical sharing.

Unordered siblings are trained in random orders (the prior is then a model of the
exchangeable family, up to the order dependence the run reports); ordered families
follow the time column, which the engine draws before the attributes.

Columns are encoded and decoded through relsynth's Bayesian marginals exactly as
in tvae (base.NeuralGenerator): Props. 1800-1805 apply unchanged, and the family
factorisation adds Props. 1860-1863 (docs/proofs/sec_v74_ctvae.tex).

**Privacy.**  With ``dp``, both stages run DP-SGD on *families*: families are
Poisson-sampled, each family's gradient (over at most ``window`` of its rows) is
clipped as one unit and Gaussian noise is added, so the bound covers one anchor
row's whole history in this table (entity level, Prop. 1863).  One RDP accountant
composes the two stages.  As for tvae, the encoding marginals, kept levels and
segments are fitted without noise, so this is not an end-to-end guarantee.
"""
from __future__ import annotations

import math
import secrets

import numpy as np

from ..contract import Capabilities, GeneratorError, PrivacyCost
from . import accountant
from . import train as T
from .base import NeuralGenerator, _number, _pos_int
from .features import FeatureSpace, table_marginals

DEFAULTS = {"prior": "transformer", "categorical_head": "residual", "latent": 16, "d_column": 32, "layers": 2,
            "heads": 4, "d_prior": 96, "prior_layers": 2, "mixtures": 5, "window": 64, "beta": 0.5,
            "free_bits": 0.05, "vae_steps": 3000, "prior_steps": 3000, "batch_size": 512, "batch_families": 64,
            "lr": 2e-3, "weight_decay": 1e-5, "max_rows": 200000, "max_levels": 30, "cond_dim": 16,
            "sample_chunk": 512, "calibrate": True, "calib_families": 2000, "dp": None}


def _families(fam, order, n, tie=None):
    """Index arrays of the rows of every family, each sorted by its order key (NaN last, ties by
    ``tie``, else in row order); rows without a family (-1, or no family ids at all) are families of
    one row."""
    if fam is None:
        return [np.array([i]) for i in range(n)]
    fam = np.asarray(fam, np.int64)
    key = np.where(fam >= 0, fam, -1 - np.arange(n))          # singletons get distinct negative ids
    o = np.zeros(n) if order is None else np.where(np.isnan(order), np.inf, order)
    idx = np.lexsort((np.arange(n), np.arange(n) if tie is None else np.asarray(tie), o, key))
    ks = key[idx]
    cuts = np.flatnonzero(np.diff(ks)) + 1
    return np.split(idx, cuts)


def _gaps(order, rows):
    """log1p(days since the previous row of the family), 0 for the first row or an unknown time."""
    if order is None or len(rows) < 2:
        return np.zeros(len(rows), np.float32)
    t = np.asarray(order, float)[rows] / 86400.0
    d = np.diff(t, prepend=t[0])
    d[0] = 0.0
    return np.log1p(np.clip(np.nan_to_num(d, nan=0.0), 0, None)).astype(np.float32)


class CTVAEGenerator(NeuralGenerator):
    name = "ctvae"
    version = "1.0"
    contract = "1.1"
    description = ("Context transformer VAE: rows of a family (the facts sharing an anchor row, in time order "
                   "when the table has a time column) drawn jointly by a causal transformer prior over row "
                   "latents with a residual categorical head; marginal-decoded, optional entity-level DP-SGD. "
                   "Needs relsynth[neural].")
    capabilities = Capabilities(conditioning=("group", "parent", "family"), dp="none", gpu="optional",
                                pushdown=False, posterior=True)

    # ---------------------------------------------------------------- options
    def _options(self, context):
        raw = dict(context.options or {})
        t = context.table
        bad = set(raw) - set(DEFAULTS)
        if bad:
            raise GeneratorError(f"{t}: unknown ctvae options {sorted(bad)} (allowed: {sorted(DEFAULTS)})")
        o = {**DEFAULTS, **raw}
        plan_bs = getattr(context.compute, "batch_size", None) if context.compute is not None else None
        if "batch_size" not in raw and plan_bs:
            o["batch_size"] = int(plan_bs)
        if o["prior"] not in ("transformer", "gru", "none"):
            raise GeneratorError(f"{t}: ctvae option prior must be transformer, gru or none")
        if o["categorical_head"] not in ("residual", "decoder"):
            raise GeneratorError(f"{t}: ctvae option categorical_head must be residual or decoder")
        for k in ("latent", "d_column", "layers", "heads", "d_prior", "prior_layers", "mixtures", "window",
                  "vae_steps", "prior_steps", "batch_size", "batch_families", "max_rows", "max_levels", "cond_dim",
                  "sample_chunk", "calib_families"):
            if not _pos_int(o[k]):
                raise GeneratorError(f"{t}: ctvae option {k} must be a positive integer")
        if o["d_column"] % o["heads"] or o["d_prior"] % o["heads"]:
            raise GeneratorError(f"{t}: ctvae options d_column and d_prior must be multiples of heads")
        for k in ("beta", "free_bits", "lr", "weight_decay"):
            if not _number(o[k]) or o[k] < 0:
                raise GeneratorError(f"{t}: ctvae option {k} must be a number >= 0")
        if not isinstance(o["calibrate"], bool):
            raise GeneratorError(f"{t}: ctvae option calibrate must be true or false")
        o["dp"] = T.DPOptions.parse(o["dp"], t)
        if o["dp"] is not None and "calibrate" not in raw:
            # calibration reads the training families' sizes and contexts, which the DP-SGD accountant
            # does not cover (Prop. 1863); under dp it runs only when the spec asks for it
            o["calibrate"] = False
        return o

    # ---------------------------------------------------------------- fit
    def fit(self, data, context, privacy_budget=None):
        torch = T.torch_module()
        o = self._options(context)
        self.opts, self.data, self.table = o, data, context.table
        rng = context.rng
        n = data.n_rows
        groups = np.zeros(n, dtype=np.int64) if context.groups is None else np.asarray(context.groups, np.int64)
        fam_all = None if context.family is None else np.asarray(context.family, np.int64)
        order_all = None if context.order is None else np.asarray(context.order, float)
        self.ordered = order_all is not None
        use = np.flatnonzero(groups >= 0) if n else np.zeros(0, np.int64)
        fams = _families(None if fam_all is None else fam_all[use],
                         None if order_all is None else order_all[use], len(use))
        fams = [use[f] for f in fams]                             # back to table row indices
        if len(use) > o["max_rows"]:                              # whole families, never part of one
            keep, tot = [], 0
            for i in rng.permutation(len(fams)):
                if tot >= o["max_rows"]:
                    break
                keep.append(i)
                tot += len(fams[i])
            fams = [fams[i] for i in sorted(keep)]
        self.n_fit = int(sum(len(f) for f in fams))
        self.n_families = len(fams)
        self._groups = groups
        self.parent = {int(k): int(v) for k, v in (context.parent or {}).items()}
        self.enc_m = table_marginals(data, rng, bootstrap=False)
        self.dec_m = self.enc_m
        self.own = [j for j, c in enumerate(data.cols) if c not in data.aux]
        self.auxj = [j for j, c in enumerate(data.cols) if c in data.aux]
        rows = np.concatenate(fams) if fams else np.zeros(0, np.int64)
        # rare columns stay out of the network and are drawn from their marginals (as tvae; not under DP)
        k = int(data.cfg["marginals"].get("min_dependence_rows", 0) or 0)
        self.indep = []
        if k and len(rows) and o["dp"] is None:
            for j in self.own:
                x = np.asarray(data.X[j])[rows]
                x = np.where(np.isnan(x), -1, x) if x.dtype.kind == "f" else x
                cnt = np.unique(x, return_counts=True)[1]
                if 0 < len(rows) - cnt.max() < k:
                    self.indep.append(j)
        self.independent = [data.cols[j] for j in self.indep]
        self.space = FeatureSpace(data, [j for j in self.own if j not in self.indep], o["max_levels"], self.enc_m)
        self.cond_space = FeatureSpace(data, self.auxj, o["max_levels"], self.enc_m)
        g_fit = sorted(set(groups[rows].tolist()))
        self.group_ix = {g: i + 1 for i, g in enumerate(g_fit)}
        coarse = self._coarse(groups[rows]) if len(rows) else np.zeros(0, np.int64)
        self.coarse_ix = {c: i + 1 for i, c in enumerate(sorted(set(coarse.tolist())))}
        self.device = T.device_of(context.compute)
        T.dtype_of(context.compute)
        self.dtype_name = "float32"
        self.report = None
        self.cat_bias, self.cat_bias_target, self.num_maps = None, None, None
        if self.n_fit == 0 or self.space.dim == 0:
            self.weights = None
            self.fitted = True
            return self
        with T.seeded(rng, context.compute) as gen:
            Z, S = self.space.encode(data.X, rows, rng)
            A = self.cond_space.dense(data.X, rows, rng)
            g = np.array([self.group_ix[int(x)] for x in groups[rows]])
            c = np.array([self.coarse_ix[int(x)] for x in coarse])
            off = np.concatenate([[0], np.cumsum([len(f) for f in fams])]).astype(np.int64)
            gap = np.concatenate([_gaps(order_all, f) for f in fams]) if self.ordered \
                else np.zeros(len(rows), np.float32)
            dev = self.device
            tt = lambda a, **kw: torch.as_tensor(a, device=dev, **kw)       # noqa: E731
            self.tensors = {"z": tt(Z), "s": tt(S), "g": tt(g), "c": tt(c), "a": tt(A, dtype=torch.float32),
                            "gap": tt(gap)}
            self.off = off
            self.model = self._build().to(dev)
            if o["dp"] is None:
                self.report = self._fit_plain(gen)
            else:
                self.report = self._fit_dp(gen)
            self.weights = {kk: v.detach().cpu().numpy() for kk, v in self.model.state_dict().items()}
            self.model.eval()
            if o["calibrate"]:
                self._calibrate_families(gen)
            del self.tensors
        self.fitted = True
        return self

    # ---------------------------------------------------------------- network
    def _build(self):
        torch = T.torch_module()
        nn, F = torch.nn, torch.nn.functional
        sp, o = self.space, self.opts
        n_num, widths = sp.n_num, list(sp.cat_widths)
        L, d, dp_, H = o["latent"], o["d_column"], o["d_prior"], o["heads"]
        K, W, prior = o["mixtures"], o["window"], o["prior"]
        residual = o["categorical_head"] == "residual"
        cond = self._cond_module()
        offs = [int(x) for x in sp.cat_offsets]

        class Block(nn.Module):
            """PreNorm transformer block written with plain tensor operations (vmap-safe for DP-SGD)."""
            def __init__(self, dm):
                super().__init__()
                self.n1, self.n2 = nn.LayerNorm(dm), nn.LayerNorm(dm)
                self.qkv, self.proj = nn.Linear(dm, 3 * dm), nn.Linear(dm, dm)
                self.ff = nn.Sequential(nn.Linear(dm, 2 * dm), nn.GELU(), nn.Linear(2 * dm, dm))
                self.h = H

            def forward(self, x, causal=False):
                B, Lq, D = x.shape
                q, k, v = self.qkv(self.n1(x)).reshape(B, Lq, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
                att = q @ k.transpose(-1, -2) / math.sqrt(D // self.h)
                if causal:
                    att = att.masked_fill(torch.triu(torch.ones(Lq, Lq, dtype=torch.bool, device=x.device), 1),
                                          float("-inf"))
                y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, Lq, D)
                x = x + self.proj(y)
                return x + self.ff(self.n2(x))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # row VAE
                self.w = nn.Parameter(torch.randn(max(n_num, 1), d) * 0.1)
                self.b = nn.Parameter(torch.zeros(max(n_num, 1), d))
                self.emb = nn.ModuleList(nn.Embedding(k, d) for k in widths)
                self.cls = nn.Parameter(torch.zeros(1, 1, d))
                self.enc = nn.ModuleList(Block(d) for _ in range(o["layers"]))
                self.enc_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * L))
                self.dec = nn.Sequential(nn.Linear(L, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                         nn.Linear(128, n_num + sum(widths)))
                self.log_s = nn.Parameter(torch.zeros(max(n_num, 1)))
                # family prior
                self.cond = cond
                self.cond_in = nn.Linear(cond.out_dim, dp_)
                self.n_in = nn.Linear(1, dp_)
                self.inp = nn.Linear(L + 1, dp_)
                self.cat_in = nn.ModuleList(nn.Embedding(k + 1, dp_) for k in widths)   # index k: no row before
                if prior == "transformer":
                    self.pos = nn.Embedding(W, dp_)
                    self.body = nn.ModuleList(Block(dp_) for _ in range(o["prior_layers"]))
                elif prior == "gru":
                    self.body = nn.GRU(dp_, dp_, num_layers=o["prior_layers"], batch_first=True)
                else:
                    self.body = nn.Sequential(nn.Linear(dp_, dp_), nn.ReLU(), nn.Linear(dp_, dp_))
                self.norm = nn.LayerNorm(dp_)
                self.mdn = nn.Linear(dp_, K * (1 + 2 * L))
                # residual categorical head (zero at initialisation: starts as the decoder's own logits)
                self.z_in = nn.Linear(L, dp_)
                self.cat_cond = nn.ModuleList(nn.Embedding(k, dp_) for k in widths)
                self.cat_out = nn.ModuleList(nn.Linear(dp_, k) for k in widths)
                for lin in self.cat_out:
                    nn.init.zeros_(lin.weight)
                    nn.init.zeros_(lin.bias)

            def forward(self, fn, *args):
                """fn(self, *args): lets torch.func.functional_call run any loss of this module with
                swapped-in parameters (per-family gradients for DP-SGD)."""
                return fn(self, *args)

            # ---- row VAE
            def encode(self, z, s):
                toks = [z.unsqueeze(-1) * self.w[:n_num] + self.b[:n_num]] if n_num else []
                toks += [e(s[:, j]).unsqueeze(1) for j, e in enumerate(self.emb)]
                t = torch.cat([self.cls.expand(len(s), -1, -1)] + toks, 1)
                for blk in self.enc:
                    t = blk(t)
                mu, lv = self.enc_head(t[:, 0]).chunk(2, -1)
                return mu, lv.clamp(-8, 8)

            def decode(self, zl):
                out = self.dec(zl)
                return out[..., :n_num], out[..., n_num:]

            def recon(self, z, s, zl):
                mean, logits = self.decode(zl)
                nll = torch.zeros(len(zl), device=zl.device)
                if n_num:
                    ls = self.log_s[:n_num].clamp(-5, 2)
                    nll = nll + (0.5 * ((z - mean) / ls.exp()) ** 2 + ls).sum(1)
                for j, k in enumerate(widths):
                    nll = nll + F.cross_entropy(logits[:, offs[j]:offs[j] + k], s[:, j], reduction="none")
                return nll

            # ---- prior
            def hidden(self, zprev, gap, cprev, rowc, logn, pos0=0, state=None):
                """(B, Lw, L), (B, Lw), (B, Lw, n_cat), (B, Lw, cond), (B, 1) -> (B, Lw, d_prior)."""
                u = self.cond_in(rowc) + self.n_in(logn).unsqueeze(1)
                if prior != "none":
                    u = u + self.inp(torch.cat([zprev, gap.unsqueeze(-1)], -1))
                    for j, e in enumerate(self.cat_in):
                        u = u + e(cprev[..., j])
                else:
                    u = u + self.inp(torch.cat([torch.zeros_like(zprev), gap.unsqueeze(-1)], -1))
                if prior == "transformer":
                    Lw = u.shape[1]
                    u = u + self.pos((torch.arange(Lw, device=u.device) + pos0).clamp(max=W - 1))
                    for blk in self.body:
                        u = blk(u, causal=True)
                    return self.norm(u), None
                if prior == "gru":
                    h, state = self.body(u, state)
                    return self.norm(h), state
                return self.norm(self.body(u)), None

            def mdn_params(self, h):
                out = self.mdn(h)
                logit = out[..., :K]
                mu, ls = out[..., K:].reshape(*out.shape[:-1], K, 2 * L).chunk(2, -1)
                return logit, mu, ls.clamp(-6, 3)

            def mdn_nll(self, h, zt):
                logit, mu, ls = self.mdn_params(h)
                lp = -0.5 * (((zt.unsqueeze(-2) - mu) / ls.exp()) ** 2 + 2 * ls + math.log(2 * math.pi)).sum(-1)
                return -torch.logsumexp(F.log_softmax(logit, -1) + lp, -1)

            def cat_logits(self, h, zl, s_prev_cols, j):
                """Logits of categorical column j given the history h, the row latent and earlier columns."""
                # the decoder is fitted in stage 1 and fixed in stage 2: its logits enter as a constant
                lg = self.decode(zl)[1].detach()[..., offs[j]:offs[j] + widths[j]]
                if residual:
                    gq = h + self.z_in(zl)
                    for i in range(j):
                        gq = gq + self.cat_cond[i](s_prev_cols[..., i])
                    lg = lg + self.cat_out[j](gq)
                return lg

        net = Net()
        net.widths, net.n_num, net.L = widths, n_num, L
        return net

    # ---------------------------------------------------------------- batches of families
    def _windows(self, idx, gen, rand_rows):
        """Padded windows of the families idx: rows (B, W) as row positions (-1: padding), and the row
        just before each window (-1 at the family's start).  A family longer than the window gives a
        random contiguous window of its order (time order, or a random permutation of siblings), with
        the previous row as the first slot's history: the layout the sampler's sliding window sees."""
        W = self.opts["window"]
        out = np.full((len(idx), W), -1, np.int64)
        prev = np.full(len(idx), -1, np.int64)
        u = rand_rows
        for b, f in enumerate(idx):
            r = np.arange(self.off[f], self.off[f + 1])
            if not self.ordered:
                r = u.permutation(r)
            if len(r) > W:
                s0 = int(u.integers(0, len(r) - W + 1))
                if s0:
                    prev[b] = r[s0 - 1]
                r = r[s0:s0 + W]
            out[b, :len(r)] = r
        return out, prev

    def _window_tensors(self, win, mu=None):
        torch = T.torch_module()
        tz, dev = self.tensors, self.device
        pos, prev = win
        P = torch.as_tensor(pos, device=dev)
        M = P >= 0
        Pc = P.clamp(min=0)
        sizes = torch.as_tensor(np.diff(self.off), device=dev, dtype=torch.float32)
        fam_of = torch.as_tensor(np.searchsorted(self.off, np.maximum(pos[:, 0], 0), side="right") - 1, device=dev)
        b = {"m": M, "z": tz["z"][Pc], "s": tz["s"][Pc], "g": tz["g"][Pc], "c": tz["c"][Pc], "a": tz["a"][Pc],
             "gap": tz["gap"][Pc] * M, "logn": torch.log(sizes[fam_of]).unsqueeze(-1)}
        if mu is not None:
            b["mu"] = mu[Pc]
            pv = torch.as_tensor(prev, device=dev)
            has = (pv >= 0).unsqueeze(-1)
            pc = pv.clamp(min=0)
            b["zp0"] = mu[pc] * has
            pad = torch.as_tensor(self.space.cat_widths, device=dev, dtype=torch.long)
            b["cp0"] = torch.where(has, tz["s"][pc], pad.expand(len(pv), -1)) if self.space.cats \
                else tz["s"][pc]
        return b

    def _prior_loss(self, net, b):
        """Per-family mean prior NLL (MDN over latents + residual categorical head) of padded windows."""
        torch = T.torch_module()
        F = torch.nn.functional
        B, Wl = b["m"].shape
        zt = b["mu"]
        zprev = torch.cat([b["zp0"].unsqueeze(1), zt[:, :-1]], 1)
        s = b["s"]
        cprev = torch.cat([b["cp0"].unsqueeze(1), s[:, :-1]], 1) if net.widths else s
        rowc = net.cond(b["g"].reshape(-1), b["c"].reshape(-1), b["a"].reshape(B * Wl, -1)).reshape(B, Wl, -1)
        h, _ = net.hidden(zprev, b["gap"], cprev, rowc, b["logn"])
        nll = net.mdn_nll(h, zt)
        for j, k in enumerate(net.widths):
            lg = net.cat_logits(h, zt, s, j)
            nll = nll + F.cross_entropy(lg.reshape(-1, k), s[..., j].reshape(-1), reduction="none").reshape(B, Wl)
        m = b["m"].float()
        return (nll * m).sum(1) / m.sum(1).clamp(min=1)

    def _vae_loss(self, net, z, s, eps, beta, m=None):
        """Per-row VAE loss with free bits on the batch's mean KL per latent unit."""
        torch = T.torch_module()
        mu, lv = net.encode(z, s)
        zl = mu + (0.5 * lv).exp() * eps
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv)
        w = torch.ones(len(z), device=z.device) if m is None else m.float()
        kl_mean = (kl * w.unsqueeze(1)).sum(0) / w.sum().clamp(min=1)
        fb = torch.clamp(kl_mean, min=self.opts["free_bits"]).sum()
        return net.recon(z, s, zl), beta * fb

    # ---------------------------------------------------------------- plain training
    def _fit_plain(self, gen):
        torch = T.torch_module()
        o, net, tz = self.opts, self.model, self.tensors
        n = int(tz["z"].shape[0])
        rep = T.TrainReport()
        vae_p, prior_p = self._stage_params()
        opt = torch.optim.AdamW(vae_p.values(), lr=o["lr"], weight_decay=o["weight_decay"])
        warm = max(1, o["vae_steps"] // 5)
        net.train()
        for step in range(o["vae_steps"]):
            idx = torch.randint(0, n, (o["batch_size"],), generator=gen, device=gen.device)
            eps = torch.randn(len(idx), o["latent"], generator=gen, device=gen.device)
            rec, kl = self._vae_loss(net, tz["z"][idx], tz["s"][idx], eps, o["beta"] * min(1.0, (step + 1) / warm))
            loss = rec.mean() + kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_p.values(), 5.0)
            opt.step()
            rep.steps += 1
            if step % 100 == 0:
                rep.train_loss.append(float(loss.detach()))
        mu = self._posterior_means()
        opt = torch.optim.AdamW(prior_p.values(), lr=o["lr"], weight_decay=o["weight_decay"])
        rows = np.random.default_rng(int(torch.randint(0, 2 ** 62, (1,), generator=gen, device=gen.device)))
        nf = len(self.off) - 1
        for step in range(o["prior_steps"]):
            idx = rows.integers(0, nf, o["batch_families"])
            b = self._window_tensors(self._windows(idx, gen, rows), mu)
            loss = self._prior_loss(net, b).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior_p.values(), 5.0)
            opt.step()
            rep.steps += 1
            if step % 100 == 0:
                rep.valid_loss.append(float(loss.detach()))
        net.eval()
        self._active_units(mu)
        return rep

    VAE_PARAMS = ("w", "b", "cls", "log_s", "emb.", "enc.", "enc_head.", "dec.")

    def _stage_params(self):
        """The row VAE's parameters (stage 1) and the family prior's (stage 2): each stage updates its
        own, so the decoder fitted in stage 1 is not moved by the prior's loss."""
        vae, prior = {}, {}
        for k, v in self.model.named_parameters():
            (vae if k in self.VAE_PARAMS or k.startswith(self.VAE_PARAMS[4:]) else prior)[k] = v
        return vae, prior

    def _posterior_means(self):
        torch = T.torch_module()
        tz, net = self.tensors, self.model
        with torch.no_grad():
            return torch.cat([net.encode(tz["z"][i:i + 8192], tz["s"][i:i + 8192])[0]
                              for i in range(0, int(tz["z"].shape[0]), 8192)])

    def _active_units(self, mu):
        self.active_units = int((mu.var(0) > 0.01).sum()) if len(mu) > 1 else 0

    # ---------------------------------------------------------------- entity-level DP-SGD (Prop. 1863)
    def _fit_dp(self, gen):
        torch = T.torch_module()
        from torch.func import functional_call, grad, vmap
        o, net, dp = self.opts, self.model, self.opts["dp"]
        nf = len(self.off) - 1
        npub = dp.public_count(nf)          # q, delta, sigma and the divisor come from this public count
        q = min(1.0, o["batch_families"] / npub)
        steps1, steps2 = o["vae_steps"], o["prior_steps"]
        delta = dp.resolve_delta(npub)
        sigma = accountant.calibrate_sigma(dp.epsilon, delta, q, steps1 + steps2)
        eps, order = accountant.epsilon(q, sigma, steps1 + steps2, delta)
        secure = torch.Generator(device=gen.device)
        secure.manual_seed(secrets.randbits(63))            # never seeded from the run seed (train.py)
        rand_rows = np.random.default_rng(secrets.randbits(63))
        stages = self._stage_params()
        warm = max(1, steps1 // 5)
        rep = T.TrainReport()

        def vae_fam(mod, z, s, e, beta, m):
            # one family: its window's rows as a batch, padded rows weighted 0
            rec, kl = self._vae_loss(mod, z, s, e, beta, m)
            return (rec * m.float()).sum() / m.float().sum().clamp(min=1) + kl

        def prior_fam(mod, b):
            return self._prior_loss(mod, {k: v.unsqueeze(0) for k, v in b.items()}).sum()

        frozen: dict = {}

        def vae_one(p, b, e, beta):
            return functional_call(net, {**frozen, **p}, (vae_fam, b["z"], b["s"], e, beta, b["m"]))

        def prior_one(p, b):
            return functional_call(net, {**frozen, **p}, (prior_fam, b))

        net.train()
        for stage, steps in ((1, steps1), (2, steps2)):
            # noise and updates touch only this stage's parameters (the other stage's are constants)
            params = stages[stage - 1]
            frozen = {k: v.detach() for k, v in stages[2 - stage].items()}
            opt = torch.optim.AdamW(params.values(), lr=o["lr"], weight_decay=o["weight_decay"])
            mu = self._posterior_means() if stage == 2 else None
            per = vmap(grad(vae_one), in_dims=(None, 0, 0, None)) if stage == 1 else \
                vmap(grad(prior_one), in_dims=(None, 0))
            for step in range(steps):
                idx = np.flatnonzero(T.os_uniform(nf) < q)
                summed = {k: torch.zeros_like(v) for k, v in params.items()}
                det = {k: v.detach() for k, v in params.items()}
                for i in range(0, len(idx), 64):
                    sub = idx[i:i + 64]
                    b = self._window_tensors(self._windows(sub, gen, rand_rows), mu)
                    if stage == 1:
                        b = {k: b[k] for k in ("z", "s", "m")}
                        e = torch.randn(len(sub), o["window"], o["latent"], generator=secure, device=gen.device)
                        g = per(det, b, e, o["beta"] * min(1.0, (step + 1) / warm))
                    else:
                        g = per(det, b)
                    norms = torch.sqrt(sum((gi.reshape(len(sub), -1) ** 2).sum(1) for gi in g.values()))
                    factor = torch.clamp(dp.clip / (norms + 1e-12), max=1.0)
                    for k, gi in g.items():
                        summed[k] += torch.einsum("b,b...->...", factor, gi)
                opt.zero_grad(set_to_none=True)
                for k, p in params.items():
                    noise = torch.as_tensor(T.os_normal(p.numel()), dtype=p.dtype, device=gen.device) \
                        .reshape(p.shape) * (sigma * dp.clip)
                    p.grad = (summed[k] + noise) / (q * npub)
                opt.step()
                rep.steps += 1
        net.eval()
        self.active_units = None             # a statistic of the raw rows, outside the DP-SGD accounting
        rep.epochs_run = rep.steps * q
        rep.dp = {"mechanism": "DP-SGD on families (Poisson-subsampled Gaussian, RDP accountant, entity level)",
                  "epsilon": eps, "delta": delta, "noise_multiplier": sigma, "clip": dp.clip, "sampling_rate": q,
                  "steps": steps1 + steps2, "rdp_order": order, "epsilon_target": dp.epsilon, "unit": "family"}
        return rep

    # ---------------------------------------------------------------- sampling
    def _sample_families(self, fams, g, c, A, gaps, gen, fixed=None, Zfix=None, Sfix=None):
        """Latents and slots of the rows of families (lists of positions into g, c, A, gaps).  Rows with
        fixed[pos] are given (kept siblings of a redraw): their latents and slots are read, not drawn,
        and the other rows are drawn given them (teacher forcing)."""
        torch = T.torch_module()
        net, o, dev = self.model, self.opts, self.device
        L, W, prior = o["latent"], o["window"], o["prior"]
        nc = len(net.widths)
        Zout = np.zeros((len(g), L), np.float32)
        Sout = np.zeros((len(g), nc), np.int64)
        sizes = np.array([len(f) for f in fams])
        order = np.argsort(sizes, kind="stable")
        bias = self._bias_tensor(dev)
        pad = torch.as_tensor(net.widths, device=dev, dtype=torch.long)
        for s0 in range(0, len(order), o["sample_chunk"]):
            ch = [fams[i] for i in order[s0:s0 + o["sample_chunk"]]]
            B, Lmax = len(ch), max(len(f) for f in ch)
            P = np.full((B, Lmax), -1, np.int64)
            for b, f in enumerate(ch):
                P[b, :len(f)] = f
            Pc = np.maximum(P, 0)
            gt, ct = torch.as_tensor(g[Pc], device=dev), torch.as_tensor(c[Pc], device=dev)
            at = torch.as_tensor(A[Pc], device=dev, dtype=torch.float32)
            rowc = net.cond(gt.reshape(-1), ct.reshape(-1), at.reshape(B * Lmax, -1)).reshape(B, Lmax, -1)
            gp = torch.as_tensor(gaps[Pc] * (P >= 0), device=dev)
            logn = torch.log(torch.as_tensor([len(f) for f in ch], device=dev, dtype=torch.float32)).unsqueeze(-1)
            Z = torch.zeros(B, Lmax, L, device=dev)
            S = torch.zeros(B, Lmax, nc, dtype=torch.long, device=dev)
            if fixed is not None:
                F_ = torch.as_tensor(fixed[Pc] & (P >= 0), device=dev)
                Zf = torch.as_tensor(Zfix[Pc], device=dev, dtype=torch.float32)
                Sf = torch.as_tensor(Sfix[Pc], device=dev, dtype=torch.long)
            state = None
            for i in range(Lmax):
                if prior == "transformer":
                    a0 = max(0, i - W + 1)
                    # history slots a0..i hold the rows before them (slot 0: the start token); slice first
                    if a0 == 0:
                        zp = torch.cat([torch.zeros(B, 1, L, device=dev), Z[:, :i]], 1)
                        cp = torch.cat([pad.expand(B, 1, -1), S[:, :i]], 1)
                    else:
                        zp, cp = Z[:, a0 - 1:i], S[:, a0 - 1:i]
                    h, _ = net.hidden(zp, gp[:, a0:i + 1], cp, rowc[:, a0:i + 1], logn)
                elif prior == "gru":
                    zp = Z[:, i - 1:i] if i else torch.zeros(B, 1, L, device=dev)
                    cp = S[:, i - 1:i] if i else pad.expand(B, 1, -1)
                    h, state = net.hidden(zp, gp[:, i:i + 1], cp, rowc[:, i:i + 1], logn, state=state)
                else:
                    h, _ = net.hidden(torch.zeros(B, 1, L, device=dev), gp[:, i:i + 1],
                                      pad.expand(B, 1, -1), rowc[:, i:i + 1], logn)
                h = h[:, -1]
                logit, mu, ls = net.mdn_params(h)
                k = torch.multinomial(logit.softmax(-1), 1, generator=gen)[:, 0]
                ar = torch.arange(B, device=dev)
                Z[:, i] = mu[ar, k] + ls[ar, k].exp() * torch.randn(B, L, generator=gen, device=dev)
                if fixed is not None:
                    Z[:, i] = torch.where(F_[:, i:i + 1], Zf[:, i], Z[:, i])
                for j, w in enumerate(net.widths):
                    lg = net.cat_logits(h, Z[:, i], S[:, i], j)
                    if bias is not None:
                        lg = lg + bias[self.space.cat_offsets[j]:self.space.cat_offsets[j] + w]
                    S[:, i, j] = torch.multinomial(lg.softmax(-1), 1, generator=gen)[:, 0]
                    if fixed is not None:
                        S[:, i, j] = torch.where(F_[:, i], Sf[:, i, j], S[:, i, j])
            keep = P >= 0
            Zout[P[keep]] = Z.cpu().numpy()[keep]
            Sout[P[keep]] = S.cpu().numpy()[keep]
        return Zout, Sout

    def _numeric(self, Zl, gen):
        """Numeric scores of the rows from their latents (decoder mean plus its noise)."""
        torch = T.torch_module()
        net = self.model
        if not net.n_num:
            return np.zeros((len(Zl), 0))
        with torch.no_grad():
            zt = torch.as_tensor(Zl, device=self.device)
            mean, _ = net.decode(zt)
            s = net.log_s[:net.n_num].clamp(-5, 2).exp()
            return (mean + s * torch.randn(mean.shape, generator=gen, device=self.device)).cpu().numpy()

    def sample(self, n, conditioning, rng):
        d = self.data
        M = len(conditioning)
        out = np.zeros((M, len(d.cols)))
        if M == 0:
            return d.decode(out, rng)
        if self.weights is None:
            for j in self.own:
                out[:, j] = self.dec_m[j].sample(M, rng)
            return d.decode(out, rng)
        torch = T.torch_module()
        g, c, A = self._cond_inputs(M, conditioning, rng)
        fam = getattr(conditioning, "family", None)
        order = getattr(conditioning, "order", None) if self.ordered else None
        self._ensure_model()
        sib = getattr(conditioning, "siblings", None)
        fixed = Zfix = Sfix = tie = None
        if fam is not None and sib is not None and len(sib):
            fam, order, g, c, A, fixed, Zfix, Sfix = self._with_siblings(sib, fam, order, g, c, A, rng)
            pos = getattr(conditioning, "position", None)
            if pos is not None and "__pos__" in sib and len(pos) == M:
                # equal order keys keep the original table order, so a kept sibling that preceded a
                # redrawn row still conditions it
                tie = np.r_[np.asarray(pos, np.int64), np.asarray(sib["__pos__"], np.int64)]
        N = len(g)
        fams = _families(fam, order if self.ordered else (None if fixed is None else (~fixed).astype(float)), N,
                         tie if self.ordered else None)
        gaps = np.zeros(N, np.float32)
        if self.ordered and order is not None:
            for f in fams:
                gaps[f] = _gaps(order, f)
        with T.seeded(rng, device=self.device) as gen, torch.no_grad():
            Zl, S = self._sample_families(fams, g, c, A, gaps, gen, fixed, Zfix, Sfix)
            Zl, S = Zl[:M], S[:M]
            Z = self._numeric(Zl, gen)
        Z = self._apply_num_maps(Z) if self.space.n_num else Z
        self.space.decode_into(out, Z, S, self.dec_m, rng)
        for j in getattr(self, "indep", []):
            out[:, j] = self.dec_m[j].sample(M, rng)
        return d.decode(out, rng)

    def _with_siblings(self, sib, fam, order, g, c, A, rng):
        """Append the kept siblings of a redraw after the M requested rows: their family ids, order keys,
        contexts (their own group when given, else their family's requested row's; parent attributes
        when given, else neutral) and their encoded latents and slots."""
        torch = T.torch_module()
        d, M = self.data, len(g)
        fam = np.asarray(fam, np.int64)
        sf = np.asarray(sib["__family__"], np.int64)
        Ms = len(sib)
        X = {}
        for j in set(self.space.num) | {cc.col for cc in self.space.cats} | set(self.auxj):
            col, cd = d.cols[j], d.codecs[j]
            parts = getattr(cd, "parts", None)
            if parts is not None and all(pc.name in sib for pc in parts):
                # joint categorical (TableModel's JointCodec): encode each part, then the combination
                X[j] = cd.encode_parts([np.asarray(pc.encode(sib[pc.name].reset_index(drop=True)))
                                        for pc in parts])
            elif col in sib:
                X[j] = np.asarray(cd.encode(sib[col].reset_index(drop=True)))
            else:
                X[j] = np.full(Ms, -1 if d.codecs[j].kind == "discrete" else np.nan)
        Zs, Ss = self.space.encode(X, np.arange(Ms), rng)
        As = self.cond_space.dense(X, np.arange(Ms), rng) if self.auxj else np.zeros((Ms, 0), np.float32)
        first = {}
        for i, f in enumerate(fam.tolist()):
            first.setdefault(f, i)
        ref = np.array([first.get(int(f), 0) for f in sf], np.int64)
        if "__group__" in sib:
            gs_raw = np.asarray(sib["__group__"], np.int64)
            gs = np.array([self.group_ix.get(int(x), 0) for x in gs_raw], np.int64)
            cs = np.array([self.coarse_ix.get(int(x), 0) for x in self._coarse(gs_raw)], np.int64)
            gs = np.where(gs > 0, gs, g[ref])
            cs = np.where(cs > 0, cs, c[ref])
        else:
            gs, cs = g[ref], c[ref]
        with torch.no_grad():
            mu = self.model.encode(torch.as_tensor(Zs, device=self.device),
                                   torch.as_tensor(Ss, device=self.device))[0].cpu().numpy()
        L = self.opts["latent"]
        fixed = np.r_[np.zeros(M, bool), np.ones(Ms, bool)]
        Zfix = np.r_[np.zeros((M, L), np.float32), mu.astype(np.float32)]
        Sfix = np.r_[np.zeros((M, Ss.shape[1]), np.int64), Ss]
        so = np.asarray(sib["__order__"], float) if "__order__" in sib else np.full(Ms, np.nan)
        order = None if order is None else np.r_[np.asarray(order, float), so]
        A2 = np.concatenate([A, As.astype(A.dtype)], 0) if A.shape[1] else np.zeros((M + Ms, 0), np.float32)
        return np.r_[fam, sf], order, np.r_[g, gs], np.r_[c, cs], A2, fixed, Zfix, Sfix

    # ---------------------------------------------------------------- calibration (Prop. 1805 on families)
    def _calibrate_families(self, gen):
        """Monotone score maps and categorical logit offsets so that the aggregate law of families drawn
        for the training families' contexts matches the encoding marginals (as tvae, Prop. 1805)."""
        torch = T.torch_module()
        sp, o, tz = self.space, self.opts, self.tensors
        nf = len(self.off) - 1
        pick = torch.randperm(nf, generator=gen, device=gen.device)[:o["calib_families"]].cpu().numpy()
        pick.sort()
        pos = np.concatenate([np.arange(self.off[f], self.off[f + 1]) for f in pick])
        fams, start = [], 0
        for f in pick:
            k = int(self.off[f + 1] - self.off[f])
            fams.append(np.arange(start, start + k))
            start += k
        g, c = tz["g"].cpu().numpy()[pos], tz["c"].cpu().numpy()[pos]
        A, gaps = tz["a"].cpu().numpy()[pos], tz["gap"].cpu().numpy()[pos]
        target = self._slot_target(self.enc_m) if sp.cats else None
        with torch.no_grad():
            for _ in range(2 if sp.cats else 1):
                Zl, S = self._sample_families(fams, g, c, A, gaps, gen)
                m = len(S)
                if sp.cats:
                    q = np.zeros(sp.n_cat)
                    for i, K in enumerate(sp.cat_widths):
                        off = int(sp.cat_offsets[i])
                        q[off:off + K] = (np.bincount(S[:, i], minlength=K)[:K] + 0.5) / (m + 0.5 * K)
                    step = np.log(target) - np.log(q)
                    self.cat_bias = step if self.cat_bias is None else self.cat_bias + step
                    self.cat_bias_target = target
            if sp.n_num:
                from .base import score_map
                Z = self._numeric(Zl, gen)
                self.num_maps = [score_map(Z[:, i]) for i in range(sp.n_num)]

    # ---------------------------------------------------------------- reports
    def uncertainty(self):
        r = self.report
        out = {"rows_fitted": getattr(self, "n_fit", 0), "families_fitted": getattr(self, "n_families", 0),
               "ordered": bool(getattr(self, "ordered", False)), "groups": len(getattr(self, "group_ix", {})),
               "features": getattr(getattr(self, "space", None), "dim", 0),
               "active_units": getattr(self, "active_units", None),
               "posterior_frozen": bool(getattr(self, "posterior_frozen", False))}
        if r is not None:
            out.update({"steps": r.steps, "vae_loss_last": r.train_loss[-1] if r.train_loss else None,
                        "prior_loss_last": r.valid_loss[-1] if r.valid_loss else None})
        return out

    def privacy_cost(self) -> PrivacyCost:
        r = self.report
        if r is not None and r.dp:
            return PrivacyCost(dp=False, epsilon=r.dp["epsilon"], delta=r.dp["delta"], mechanism=r.dp["mechanism"],
                               note=("ctvae: DP-SGD bounds what the network weights reveal about any one family "
                                     "(all rows of one anchor row in this table). The encoding marginals, kept "
                                     "levels and latent groups are fitted without noise, so this is not an "
                                     "end-to-end guarantee; use relsynth run --dp for one (docs/neural.md)."))
        return PrivacyCost(dp=False, note="ctvae: no formal privacy guarantee")

    def state(self):
        st = super().state()
        st.pop("tensors", None)
        st.pop("off", None)
        return st


__all__ = ["CTVAEGenerator"]
