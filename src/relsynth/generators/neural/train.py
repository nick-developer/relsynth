# SPDX-License-Identifier: Apache-2.0
"""Training loop of the neural generators: ordinary minibatch training with
early stopping on held-out rows, or DP-SGD.

**Determinism (Prop. 701).**  Every random number of a non-private fit comes
from torch generators seeded from the engine's numpy generator, inside
``torch.random.fork_rng`` so that torch's global state is left as it was; on
the CPU a seed then gives the same model.  On a GPU results are reproducible
up to floating point unless the compute preset asks for deterministic kernels.

**DP-SGD (Prop. 1802).**  With ``dp`` options the network is trained by
DP-SGD (Abadi et al. 2016): Poisson sampling of rows at rate q = batch / n,
per-row gradients (torch.func.vmap of torch.func.grad), each clipped to L2
norm C, summed, plus N(0, sigma^2 C^2) noise, divided by the expected batch
size, then an Adam step (post-processing).  sigma is calibrated so that the
Renyi accountant (accountant.py) gives the requested epsilon at delta after
all steps.  As in relsynth's DP mode, the noise and the sampling are drawn
from the operating system's CSPRNG and are never seeded, so a DP fit is not
bit-reproducible.  No early stopping is done (choosing an epoch on held-out
real rows would spend budget that is not accounted).

The loss functions receive every random number they use as an input tensor
(per row), so the per-row gradients under vmap are those of the loss the
non-private path minimises.
"""
from __future__ import annotations

import contextlib
import math
import secrets
from dataclasses import dataclass, field

import numpy as np

from ..contract import GeneratorError
from . import accountant

INSTALL_HINT = "neural generators need PyTorch: pip install 'relsynth[neural]'"


def torch_module():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - exercised without torch
        raise GeneratorError(INSTALL_HINT) from e
    return torch


def torch_missing() -> str | None:
    import importlib.util
    return None if importlib.util.find_spec("torch") is not None else INSTALL_HINT


def device_of(plan) -> str:
    """The torch device of a compute plan (relsynth.compute.ComputePlan)."""
    torch = torch_module()
    if plan is None or not getattr(plan, "gpu", False):
        return "cpu"
    ids = list(getattr(plan, "device_ids", []) or [])
    if not torch.cuda.is_available():         # the compute resolver already applied the fallback rule
        raise GeneratorError("the compute plan asks for a GPU but torch sees none (pip install a CUDA or ROCm build)")
    return f"cuda:{ids[0]}" if ids else "cuda"


def dtype_of(plan):
    torch = torch_module()
    prec = getattr(plan, "precision", "fp32") if plan is not None else "fp32"
    if prec in ("fp32", "tf32"):
        return torch.float32
    raise GeneratorError(f"neural generators train in fp32; compute precision {prec} is not supported")


@contextlib.contextmanager
def seeded(rng: np.random.Generator, plan=None, device: str | None = None):
    """A torch generator seeded from the engine's generator, inside a fork of
    torch's global random state (layer initialisation draws from it).  The generator
    lives on ``device`` when given (the fitted model's device), else on the plan's."""
    torch = torch_module()
    seed = int(rng.integers(0, 2 ** 63 - 1))
    threads = getattr(plan, "threads", None) if plan is not None else None
    old_threads = torch.get_num_threads()
    dev = device or device_of(plan)
    devices = [] if dev == "cpu" else list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if threads:
            torch.set_num_threads(int(threads))
        try:
            gen = torch.Generator(device=dev)
            gen.manual_seed(seed)
            yield gen
        finally:
            torch.set_num_threads(old_threads)


@dataclass
class DPOptions:
    epsilon: float
    delta: float | None = None     # None: 1 / (n log n) is used... see resolve()
    clip: float = 1.0

    @classmethod
    def parse(cls, raw, table: str):
        if raw in (None, False):
            return None
        if not isinstance(raw, dict):
            raise GeneratorError(f"{table}: generator option dp must be a mapping {{epsilon, delta, clip}}")
        bad = set(raw) - {"epsilon", "delta", "clip"}
        if bad:
            raise GeneratorError(f"{table}: unknown dp options {sorted(bad)} (allowed: epsilon, delta, clip)")
        try:
            eps = float(raw["epsilon"])
        except (KeyError, TypeError, ValueError) as e:
            raise GeneratorError(f"{table}: dp.epsilon must be a positive number") from e
        delta = raw.get("delta")
        clip = float(raw.get("clip", 1.0))
        if not (eps > 0 and math.isfinite(eps)) or clip <= 0 or (delta is not None and not 0 < float(delta) < 1):
            raise GeneratorError(f"{table}: dp needs epsilon > 0, clip > 0 and 0 < delta < 1")
        return cls(eps, None if delta is None else float(delta), clip)

    def resolve_delta(self, n: int) -> float:
        # the usual choice delta << 1/n (Dwork and Roth 2014, sec. 2.3): 1 / (10 n), at most 1e-5
        return self.delta if self.delta is not None else min(1e-5, 1.0 / (10.0 * max(n, 1)))


@dataclass
class TrainReport:
    steps: int = 0
    epochs_run: float = 0.0
    best_epoch: int | None = None
    train_loss: list = field(default_factory=list)
    valid_loss: list = field(default_factory=list)
    dp: dict | None = None


def _batches(n, batch, gen, torch):
    perm = torch.randperm(n, generator=gen, device=gen.device)
    for i in range(0, n, batch):
        yield perm[i:i + batch]


def train(model, loss_rows, tensors: dict, noise_fn, *, epochs: int, batch_size: int, lr: float,
          weight_decay: float, gen, dp: DPOptions | None = None, valid_share: float = 0.1,
          patience: int = 20, max_steps: int | None = None) -> TrainReport:
    """Fit `model` by minimising the mean of loss_rows(model_params, batch_tensors, noise).

    loss_rows(params, batch, noise) -> per-row losses (a 1-D tensor), written
    with torch.func.functional_call so that it also runs under vmap.
    noise_fn(n, gen) -> dict of per-row random tensors for a batch of n rows.
    tensors: dict of row-aligned tensors (the training data).
    """
    torch = torch_module()
    from torch.func import functional_call, grad, vmap  # noqa: F401  (functional_call used by callers)

    n = int(next(iter(tensors.values())).shape[0])
    rep = TrainReport()
    params = {k: v for k, v in model.named_parameters()}
    opt = torch.optim.AdamW(params.values(), lr=lr, weight_decay=weight_decay)

    if dp is None:
        nval = int(round(valid_share * n)) if n >= 50 and valid_share > 0 else 0
        perm = torch.randperm(n, generator=gen, device=gen.device)
        vidx, tidx = perm[:nval], perm[nval:]
        train_t = {k: v[tidx] for k, v in tensors.items()}
        valid_t = {k: v[vidx] for k, v in tensors.items()}
        valid_noise = noise_fn(nval, gen) if nval else None
        ntr = n - nval
        bs = max(1, min(batch_size, ntr))
        best, best_state, bad = math.inf, None, 0
        for ep in range(epochs):
            model.train()
            tot, cnt = 0.0, 0
            for idx in _batches(ntr, bs, gen, torch):
                batch = {k: v[idx] for k, v in train_t.items()}
                loss = loss_rows(params, batch, noise_fn(len(idx), gen)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                rep.steps += 1
                tot += float(loss.detach()) * len(idx)
                cnt += len(idx)
                if max_steps and rep.steps >= max_steps:
                    break
            rep.train_loss.append(tot / max(cnt, 1))
            rep.epochs_run = ep + 1
            if nval:
                model.eval()
                with torch.no_grad():
                    v = float(loss_rows(params, valid_t, valid_noise).mean())
                rep.valid_loss.append(v)
                if v < best - 1e-6 * abs(best if math.isfinite(best) else 1.0):
                    best, bad, rep.best_epoch = v, 0, ep + 1
                    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
            if max_steps and rep.steps >= max_steps:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return rep

    # ---- DP-SGD
    delta = dp.resolve_delta(n)
    bs = max(1, min(batch_size, n))
    q = bs / n
    steps = max(1, int(math.ceil(epochs * n / bs)))
    if max_steps:
        steps = min(steps, int(max_steps))
    sigma = accountant.calibrate_sigma(dp.epsilon, delta, q, steps)
    eps, order = accountant.epsilon(q, sigma, steps, delta)
    secure = torch.Generator(device=gen.device)
    secure.manual_seed(secrets.randbits(63))           # never seeded from the run seed (see the module doc)

    def one(p, row, nz):
        b = {k: v.unsqueeze(0) for k, v in row.items()}
        z = {k: v.unsqueeze(0) for k, v in nz.items()}
        return loss_rows(p, b, z).sum()

    per_row = vmap(grad(one), in_dims=(None, 0, 0))
    detached = {k: v.detach() for k, v in params.items()}
    chunk = 512
    model.train()
    for _ in range(steps):
        mask = torch.rand(n, generator=secure, device=gen.device) < q
        idx = torch.nonzero(mask).flatten()
        summed = {k: torch.zeros_like(v) for k, v in params.items()}
        for i in range(0, len(idx), chunk):
            sub = idx[i:i + chunk]
            batch = {k: v[sub] for k, v in tensors.items()}
            noise = noise_fn(len(sub), secure)
            for k in detached:
                detached[k] = params[k].detach()
            g = per_row(detached, batch, noise)
            norms = torch.sqrt(sum((gi.reshape(len(sub), -1) ** 2).sum(1) for gi in g.values()))
            factor = torch.clamp(dp.clip / (norms + 1e-12), max=1.0)
            for k, gi in g.items():
                summed[k] += torch.einsum("b,b...->...", factor, gi)
        opt.zero_grad(set_to_none=True)
        for k, p in params.items():
            noise = torch.randn(p.shape, generator=secure, device=gen.device) * (sigma * dp.clip)
            p.grad = (summed[k] + noise) / (q * n)
        opt.step()
        rep.steps += 1
    rep.epochs_run = steps * q
    model.eval()
    rep.dp = {"mechanism": "DP-SGD (Poisson-subsampled Gaussian, RDP accountant)", "epsilon": eps, "delta": delta,
              "noise_multiplier": sigma, "clip": dp.clip, "sampling_rate": q, "steps": steps, "rdp_order": order,
              "epsilon_target": dp.epsilon}
    return rep
