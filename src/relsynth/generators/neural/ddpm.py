# SPDX-License-Identifier: Apache-2.0
"""Conditional Gaussian-multinomial diffusion (TabDDPM) as a generator plug-in.

Continuous scores follow Gaussian diffusion (Ho, Jain and Abbeel 2020) and
discrete columns multinomial diffusion (Hoogeboom et al. 2021), with one
cosine noise schedule (Nichol and Dhariwal 2021) and one MLP denoiser that
predicts the Gaussian noise and the logits of each clean category, the design
of TabDDPM (Kotelnikov et al. 2023).  The denoiser is conditioned on the
step (sinusoidal embedding) and on base.py's conditioning vector (the row's
group, coarse group and parent attributes), which is how ClavaDDPM (Pang et al.
2024) conditions a child table on its parent's cluster: relsynth's latent
segments play the part of ClavaDDPM's clusters.

Forward processes, for step t with abar_t = prod_{i<=t} (1 - beta_i):

    q(z_t | z_0) = N(sqrt(abar_t) z_0, (1 - abar_t) I)
    q(x_t | x_0) = Cat(abar_t x_0 + (1 - abar_t) / K)          (x one-hot over K levels)

Loss per row: the mean squared error of the predicted noise over the scores,
plus, per discrete column, KL(q(x_{t-1} | x_t, x_0) || q(x_{t-1} | x_t, p0))
for t >= 2 and -log q(x_0 | x_1, p0) for t = 1, where p0 is the predicted
distribution of the clean category (Hoogeboom et al. 2021, eq. 13), averaged
over discrete columns (TabDDPM), plus ``aux_weight`` times the cross-entropy of the
predicted clean category (the hybrid loss of Austin et al. 2021, sec. 3.4).  The
KL terms alone carry a signal of order beta_t per step, too weak with 1000
steps for the denoiser to learn how the categories depend on each other: on
UCI Adult every pair of categorical columns came out independent without the
cross-entropy (Cramér's V 0.01-0.03 against 0.40-0.49 in the data).

Sampling runs the reverse chain on a respaced sub-sequence of ``sample_steps``
steps (Nichol and Dhariwal 2021, sec. 4).  For steps s < t both posteriors
are exact for the coarser chain:

    q(z_s | z_t, z_0) = N( sqrt(abar_s) b / (1 - abar_t) z_0 + sqrt(a) (1 - abar_s) / (1 - abar_t) z_t,
                           b (1 - abar_s) / (1 - abar_t) ),  a = abar_t / abar_s, b = 1 - a
    q(x_s | x_t, x_0) proportional to (a x_t + b / K) * (abar_s x_0 + (1 - abar_s) / K).
"""
from __future__ import annotations

import math

import numpy as np

from .base import NeuralGenerator, _number, _pos_int, mlp
from .features import ZCLIP
from .train import GeneratorError, torch_module


def cosine_abar(T: int, s: float = 0.008) -> np.ndarray:
    """abar_0..abar_T of the cosine schedule (abar_0 = 1), betas clipped at 0.999."""
    f = np.cos(((np.arange(T + 1) / T) + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    beta = np.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
    return np.concatenate([[1.0], np.cumprod(1 - beta)])


class DDPMGenerator(NeuralGenerator):
    name = "ddpm"
    version = "1.0"
    description = ("Conditional Gaussian-multinomial diffusion (TabDDPM) on relsynth's marginal scores, "
                   "conditioned on the row's group like ClavaDDPM; optional DP-SGD. Needs relsynth[neural].")
    DEFAULTS = {"epochs": 2000, "batch_size": 1024, "lr": 1e-3, "patience": 200, "max_steps": 20000,
                "hidden": [256, 512, 512, 256], "emb_dim": 128, "timesteps": 1000, "sample_steps": 200,
                "aux_weight": 1.0}

    def _check_options(self, o, table):
        h = o["hidden"]
        if not (isinstance(h, (list, tuple)) and h and all(_pos_int(x) for x in h)):
            raise GeneratorError(f"{table}: ddpm option hidden must be a list of positive integers")
        for k in ("emb_dim", "timesteps", "sample_steps"):
            if not _pos_int(o[k]):
                raise GeneratorError(f"{table}: ddpm option {k} must be a positive integer")
        if not _number(o["aux_weight"]) or o["aux_weight"] < 0:
            raise GeneratorError(f"{table}: ddpm option aux_weight must be a number >= 0")
        if o["sample_steps"] > o["timesteps"]:
            raise GeneratorError(f"{table}: ddpm sample_steps may not exceed timesteps")

    def _build(self):
        torch = torch_module()
        nn = torch.nn
        sp, o = self.space, self.opts
        cond = self._cond_module()
        E, H = o["emb_dim"], list(o["hidden"])
        self._ab = cosine_abar(o["timesteps"])

        class Denoiser(nn.Module):
            def __init__(self):
                super().__init__()
                self.cond = cond
                self.inp = nn.Linear(sp.dim, E)
                self.temb = nn.Sequential(nn.Linear(E, E), nn.SiLU(), nn.Linear(E, E))
                self.cproj = nn.Linear(cond.out_dim, E)
                self.net = mlp([E, *H, sp.dim], torch)
                self.register_buffer("freqs", torch.exp(-math.log(10000.0) * torch.arange(E // 2) / max(E // 2, 1)))

            def forward(self, x, t, g, c, a):
                ang = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
                te = torch.cat([torch.cos(ang), torch.sin(ang)], 1)
                if te.shape[1] < E:
                    te = torch.cat([te, torch.zeros(te.shape[0], E - te.shape[1], device=te.device)], 1)
                h = self.inp(x) + self.temb(te) + self.cproj(self.cond(g, c, a))
                return self.net(torch.relu(h))

        return Denoiser()

    # ---- the processes
    def _abar(self, t):
        torch = torch_module()
        if not hasattr(self, "_ab"):
            self._ab = cosine_abar(self.opts["timesteps"])
        ab = torch.as_tensor(self._ab, dtype=torch.float32, device=t.device)
        return ab[t]

    def _noise(self, n, gen):
        torch = torch_module()
        dev = gen.device
        return {"t": torch.randint(1, self.opts["timesteps"] + 1, (n,), generator=gen, device=dev),
                "eps": torch.randn(n, max(self.space.n_num, 1), generator=gen, device=dev),
                "gum": torch.rand(n, max(self.space.n_cat, 1), generator=gen, device=dev),
                "ug": torch.rand(n, generator=gen, device=dev), "uc": torch.rand(n, generator=gen, device=dev)}

    @staticmethod
    def _sample_onehot(logp, u, torch):
        """One-hot draw from log probabilities by the Gumbel-max trick with uniforms u."""
        gmb = -torch.log(-torch.log(u.clamp(1e-12, 1 - 1e-7)))
        idx = (logp + gmb).argmax(1)
        # a comparison instead of one_hot, which vmap (per-row gradients of DP-SGD) cannot batch
        ks = torch.arange(logp.shape[1], device=logp.device)
        return (ks.unsqueeze(0) == idx.unsqueeze(1)).float(), idx

    def _cat_posterior(self, xt, p0, a_ts, ab_s, K, torch):
        """log q(x_s | x_t, p0), normalised: (a x_t + b/K) * (abar_s p0 + (1 - abar_s)/K)."""
        left = a_ts * xt + (1 - a_ts) / K
        right = ab_s * p0 + (1 - ab_s) / K
        lp = torch.log(left.clamp_min(1e-30)) + torch.log(right.clamp_min(1e-30))
        return lp - torch.logsumexp(lp, 1, keepdim=True)

    def _loss_rows(self, params, batch, noise):
        torch = torch_module()
        from torch.func import functional_call
        sp = self.space
        g, c = self._drop_cond(batch, noise)
        t = noise["t"]
        ab_t = self._abar(t).unsqueeze(1)
        ab_tm1 = self._abar(t - 1).unsqueeze(1)
        parts = []
        if sp.n_num:
            eps = noise["eps"][:, : sp.n_num]
            parts.append(torch.sqrt(ab_t) * batch["z"] + torch.sqrt(1 - ab_t) * eps)
        xts = []
        for i, K in enumerate(sp.cat_widths):
            o = int(sp.cat_offsets[i])
            x0 = batch["x1h"][:, o:o + K]
            q = ab_t * x0 + (1 - ab_t) / K
            xt, _ = self._sample_onehot(torch.log(q), noise["gum"][:, o:o + K], torch)
            xts.append(xt)
        parts += xts
        x_in = torch.cat(parts, 1)
        out = functional_call(self.model, params, (x_in, t, g, c, batch["a"]))
        loss = torch.zeros(x_in.shape[0], device=x_in.device)
        if sp.n_num:
            loss = loss + ((out[:, : sp.n_num] - noise["eps"][:, : sp.n_num]) ** 2).mean(1)
        if sp.cats:
            a_ts = ab_t / ab_tm1
            first = (t == 1).float()
            closs = torch.zeros_like(loss)
            for i, K in enumerate(sp.cat_widths):
                o = int(sp.cat_offsets[i])
                x0 = batch["x1h"][:, o:o + K]
                p0 = torch.softmax(out[:, sp.n_num + o: sp.n_num + o + K], 1)
                lq_model = self._cat_posterior(xts[i], p0, a_ts, ab_tm1, K, torch)
                lq_true = self._cat_posterior(xts[i], x0, a_ts, ab_tm1, K, torch)
                kl = (torch.exp(lq_true) * (lq_true - lq_model)).sum(1)
                nll = -(x0 * lq_model).sum(1)
                # hybrid loss (Austin et al. 2021, sec. 3.4): the cross-entropy of the predicted clean
                # category, a proper scoring rule whose minimiser is p(x_0 | x_t), the quantity p0 stands for
                ce = -(x0 * torch.log_softmax(out[:, sp.n_num + o: sp.n_num + o + K], 1)).sum(1)
                closs = closs + first * nll + (1 - first) * kl + self.opts["aux_weight"] * ce
            loss = loss + closs / len(sp.cats)
        return loss

    def _generate(self, n, cond, gen):
        torch = torch_module()
        sp, o = self.space, self.opts
        dev = cond["g"].device
        m = self.model
        T, S = o["timesteps"], o["sample_steps"]
        # respaced times from T down towards 1 (a one-step chain goes from T straight to 0)
        steps = np.unique(np.round(np.linspace(T, 1, S)).astype(int))[::-1]
        z = torch.randn(n, sp.n_num, generator=gen, device=dev)
        xs = []
        for K in sp.cat_widths:              # x_T: uniform over the levels
            xs.append(torch.nn.functional.one_hot(torch.randint(0, K, (n,), generator=gen, device=dev), K).float())
        bias = self._bias_tensor(dev)
        for k, t in enumerate(steps):
            s = int(steps[k + 1]) if k + 1 < len(steps) else 0
            ab_t, ab_s = float(self._ab[t]), float(self._ab[s])
            a, b = ab_t / ab_s, 1 - ab_t / ab_s
            tt = torch.full((n,), int(t), device=dev, dtype=torch.long)
            out = m(torch.cat([z, *xs], 1), tt, cond["g"], cond["c"], cond["a"])
            if sp.n_num:
                eps = out[:, : sp.n_num]
                z0 = ((z - math.sqrt(1 - ab_t) * eps) / math.sqrt(ab_t)).clamp(-ZCLIP, ZCLIP)
                mean = (math.sqrt(ab_s) * b / (1 - ab_t)) * z0 + (math.sqrt(a) * (1 - ab_s) / (1 - ab_t)) * z
                if s > 0:
                    var = b * (1 - ab_s) / (1 - ab_t)
                    z = mean + math.sqrt(var) * torch.randn(z.shape, generator=gen, device=dev)
                else:
                    z = mean
            for i, K in enumerate(sp.cat_widths):
                off = sp.n_num + int(sp.cat_offsets[i])
                lg = out[:, off:off + K]
                if bias is not None:
                    o0 = int(sp.cat_offsets[i])
                    lg = lg + bias[o0:o0 + K]
                p0 = torch.softmax(lg, 1)
                lp = self._cat_posterior(xs[i], p0, a, ab_s, K, torch)
                xs[i], _ = self._sample_onehot(lp, torch.rand(n, K, generator=gen, device=dev), torch)
        cats = [x.argmax(1) for x in xs]
        Sx = torch.stack(cats, 1) if cats else torch.zeros((n, 0), dtype=torch.long, device=dev)
        return z, Sx
