# SPDX-License-Identifier: Apache-2.0
"""Conditional tabular VAE (TVAE, Xu et al. 2019) as a generator plug-in.

Encoder q(z | x, c) and decoder p(x | z, c) are MLPs; c is the conditioning
vector of base.py (group, coarse group, parent attributes), concatenated to
both, which makes it a conditional VAE (Sohn, Lee and Yan 2015).  The decoder
gives, per continuous score, a mean and a learned per-column standard
deviation (Gaussian likelihood, as in TVAE), and per discrete column the
logits of a categorical likelihood.  The loss per row is the negative ELBO
with TVAE's reconstruction weight ``loss_factor``:

    loss = loss_factor * ( sum_num [(z - mu)^2 / (2 s^2) + log s] + sum_cat CE ) + KL(q || N(0, I)).

Sampling draws the latent from N(0, I), then each score from N(mu, s^2) and
each category from its softmax (TVAE takes the arg max; sampling keeps the
decoder's calibrated uncertainty, which matters for rare levels).
"""
from __future__ import annotations

from .base import NeuralGenerator, mlp
from .train import GeneratorError, torch_module


class TVAEGenerator(NeuralGenerator):
    name = "tvae"
    version = "1.0"
    description = ("Conditional tabular VAE (Xu et al. 2019) on relsynth's marginal scores; conditions on the "
                   "row's group and parent attributes; optional DP-SGD. Needs relsynth[neural].")
    DEFAULTS = {"epochs": 300, "batch_size": 500, "lr": 1e-3, "patience": 30, "hidden": [128, 128],
                "latent": 64, "loss_factor": 2.0}

    def _check_options(self, o, table):
        h = o["hidden"]
        if not (isinstance(h, (list, tuple)) and h and all(isinstance(x, int) and x > 0 for x in h)):
            raise GeneratorError(f"{table}: tvae option hidden must be a list of positive integers")
        if not isinstance(o["latent"], int) or o["latent"] < 1 or not o["loss_factor"] > 0:
            raise GeneratorError(f"{table}: tvae needs latent >= 1 and loss_factor > 0")

    def _build(self):
        torch = torch_module()
        nn = torch.nn
        sp, o = self.space, self.opts
        cond = self._cond_module()
        L, H = o["latent"], list(o["hidden"])
        widths = list(sp.cat_widths)
        n_num = sp.n_num

        class TVAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.cond = cond
                self.enc = mlp([sp.dim + cond.out_dim, *H, 2 * L], torch)
                self.dec = mlp([L + cond.out_dim, *H[::-1], sp.dim], torch)
                self.log_s = nn.Parameter(torch.full((max(n_num, 1),), -1.0))

            def decode(self, z, cv):
                out = self.dec(torch.cat([z, cv], 1))
                return out[:, :n_num], out[:, n_num:]

            def forward(self, g, c, a, x, eps):
                cv = self.cond(g, c, a)
                h = self.enc(torch.cat([x, cv], 1))
                mu, lv = h[:, :L], h[:, L:].clamp(-10, 10)
                mean, logits = self.decode(mu + torch.exp(0.5 * lv) * eps, cv)
                return mu, lv, mean, logits, self.log_s

        m = TVAE()
        m.widths, m.L, m.n_num = widths, L, n_num
        return m

    def _noise(self, n, gen):
        torch = torch_module()
        dev = gen.device
        return {"eps": torch.randn(n, self.opts["latent"], generator=gen, device=dev),
                "ug": torch.rand(n, generator=gen, device=dev), "uc": torch.rand(n, generator=gen, device=dev)}

    def _loss_rows(self, params, batch, noise):
        torch = torch_module()
        from torch.func import functional_call
        m = self.model
        g, c = self._drop_cond(batch, noise)

        x = torch.cat([batch["z"], batch["x1h"]], 1)
        mu, lv, mean, logits, log_s = functional_call(m, params, (g, c, batch["a"], x, noise["eps"]))
        rec = torch.zeros(x.shape[0], device=x.device)
        if m.n_num:
            ls = log_s[: m.n_num].clamp(-5, 2)
            rec = rec + (((batch["z"] - mean) ** 2) / (2 * torch.exp(2 * ls)) + ls).sum(1)
        off = 0
        for i, w in enumerate(m.widths):
            lp = torch.log_softmax(logits[:, off:off + w], dim=1)
            rec = rec - lp.gather(1, batch["s"][:, i:i + 1]).squeeze(1)
            off += w
        kl = 0.5 * (mu ** 2 + torch.exp(lv) - 1 - lv).sum(1)
        return self.opts["loss_factor"] * rec + kl

    def _generate(self, n, cond, gen):
        torch = torch_module()
        m = self.model
        cv = m.cond(cond["g"], cond["c"], cond["a"])
        z = torch.randn(n, m.L, generator=gen, device=cv.device)
        mean, logits = m.decode(z, cv)
        if m.n_num:
            s = torch.exp(m.log_s[: m.n_num].clamp(-5, 2))
            num = mean + s * torch.randn(mean.shape, generator=gen, device=cv.device)
        else:
            num = torch.zeros((n, 0), device=cv.device)
        cats, off = [], 0
        bias = self._bias_tensor(cv.device)
        if bias is not None:
            logits = logits + bias
        for w in m.widths:
            p = torch.softmax(logits[:, off:off + w], dim=1)
            cats.append(torch.multinomial(p, 1, generator=gen).squeeze(1))
            off += w
        S = torch.stack(cats, 1) if cats else torch.zeros((n, 0), dtype=torch.long, device=cv.device)
        return num, S
