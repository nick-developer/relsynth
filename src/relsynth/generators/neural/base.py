# SPDX-License-Identifier: Apache-2.0
"""What the neural generators share: options, the feature space, the
conditioning on the row's group, coarse group and parent attributes, the
Bayesian marginals used to decode, persistence and the privacy report.

**Conditioning.**  A row's context enters the network as a learned embedding
of its group, a learned embedding of its coarse group (the engine's back-off,
a fact block's dimension segment) and the encoded parent attributes.  Index 0
of each embedding is a *null* token: during training the group is replaced by
it for a share ``p_uncond`` of the rows (and the coarse group for half that
share), as in classifier-free guidance training (Ho and Salimans 2022), so the
same network is also a model of the rows given only their coarse group, and
given nothing.  A group the fit never saw (a sparse fact block passed with its
coarse label, Prop. 700) is sampled through the null token with its coarse
group, which is the neural form of the engine's back-off (Prop. 1801).

**Posterior (Prop. 1800).**  The network is fitted once (a point estimate);
the marginals it decodes through are relsynth's Bayesian table marginals,
redrawn by posterior_draw, so datasets are posterior predictive draws of the
marginals given a fixed dependence model, the same split as cart's
(Prop. 706).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
from scipy.special import ndtri

from ..contract import Capabilities, GeneratorError, PrivacyCost, TableGenerator
from . import train as T
from .features import FeatureSpace, table_marginals

COMMON = {"epochs": None, "batch_size": None, "lr": None, "weight_decay": 1e-5, "max_rows": 200000,
          "max_levels": 64, "valid_share": 0.1, "patience": None, "max_steps": None, "cond_dim": 16,
          "p_uncond": 0.1, "dp": None, "sample_batch": 65536, "calibrate": True, "calib_rows": 20000}


def _pos_int(v) -> bool:
    return bool(isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v >= 1)


def _number(v) -> bool:
    return bool(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(v))


def score_map(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Knots (x, y) of the strictly increasing map that sends the empirical law of the scores z to
    N(0, 1) (Prop. 1805): 1999 inner quantiles and the sample's ends.  The end knots' probabilities stay
    outside the inner grid, so the map is increasing for any sample size."""
    z = np.asarray(z, np.float64)
    m = len(z)
    grid = np.linspace(0.0, 1.0, 2001)[1:-1]
    e = min(0.5 / m, grid[0] / 2)
    x = np.concatenate([[z.min()], np.quantile(z, grid), [z.max()]])
    y = ndtri(np.concatenate([[e], grid, [1 - e]]))
    x = np.maximum.accumulate(x + np.arange(len(x)) * 1e-9)   # strictly increasing knots
    return x, y


class NeuralGenerator(TableGenerator):
    """Base class; subclasses set DEFAULTS and implement _build, _loss_rows,
    _noise and _generate."""
    DEFAULTS: dict = {}
    capabilities = Capabilities(conditioning=("group", "parent"), dp="none", gpu="optional", pushdown=True,
                                posterior=True)

    @classmethod
    def unavailable(cls) -> str | None:
        return T.torch_missing()

    # ---------------------------------------------------------------- options
    def _options(self, context):
        allowed = {**COMMON, **self.DEFAULTS}
        raw = dict(context.options or {})
        bad = set(raw) - set(allowed)
        if bad:
            raise GeneratorError(f"{context.table}: unknown {self.name} options {sorted(bad)} "
                                 f"(allowed: {sorted(allowed)})")
        o = {**allowed, **raw}
        plan_bs = getattr(context.compute, "batch_size", None) if context.compute is not None else None
        if raw.get("batch_size") is None and plan_bs:
            o["batch_size"] = int(plan_bs)
        for k in ("epochs", "batch_size", "lr", "patience"):     # null means "the generator's default"
            if o[k] is None:
                o[k] = self.DEFAULTS[k]
        if not isinstance(o["calibrate"], bool):
            raise GeneratorError(f"{context.table}: {self.name} option calibrate must be true or false")
        for k in ("epochs", "batch_size", "max_rows", "max_levels", "patience", "cond_dim", "sample_batch",
                  "calib_rows"):
            if not _pos_int(o[k]):
                raise GeneratorError(f"{context.table}: {self.name} option {k} must be a positive integer")
        if o["max_steps"] is not None and not _pos_int(o["max_steps"]):
            raise GeneratorError(f"{context.table}: {self.name} option max_steps must be a positive integer or null")
        for k in ("lr", "weight_decay", "valid_share", "p_uncond"):
            v = o[k]
            if not _number(v) or v < 0:
                raise GeneratorError(f"{context.table}: {self.name} option {k} must be a number >= 0")
        if not 0 <= o["valid_share"] < 0.5 or not 0 <= o["p_uncond"] < 1:
            raise GeneratorError(f"{context.table}: {self.name} needs 0 <= valid_share < 0.5 and 0 <= p_uncond < 1")
        o["dp"] = T.DPOptions.parse(o["dp"], context.table)
        self._check_options(o, context.table)
        return o

    def _check_options(self, o, table):
        """Subclass hook for its own options."""

    # ---------------------------------------------------------------- fit
    def fit(self, data, context, privacy_budget=None):
        torch = T.torch_module()
        o = self._options(context)
        self.opts, self.data, self.table = o, data, context.table
        rng = context.rng
        n = data.n_rows
        groups = np.zeros(n, dtype=np.int64) if context.groups is None else np.asarray(context.groups, np.int64)
        use = np.flatnonzero(groups >= 0) if n else np.zeros(0, np.int64)
        if len(use) > o["max_rows"]:
            use = np.sort(rng.permutation(use)[: o["max_rows"]])
        self.n_fit = int(len(use))
        self._groups = groups
        self.parent = {int(k): int(v) for k, v in (context.parent or {}).items()}
        # the encoding marginals: the table's posterior-mean marginals (decoding uses posterior draws)
        self.enc_m = table_marginals(data, rng, bootstrap=False)
        self.dec_m = self.enc_m
        self.own = [j for j, c in enumerate(data.cols) if c not in data.aux]
        self.auxj = [j for j, c in enumerate(data.cols) if c in data.aux]
        # columns whose non-modal values occur in fewer than marginals.min_dependence_rows rows carry
        # no estimable dependence, and learning their co-occurrences would let the network reproduce
        # the rows that hold them: as in the copula, they stay out of the network and are drawn
        # independently from their marginals.  Under DP-SGD the selection is skipped: it depends on the
        # rows and would change the network's input width (and so the published weight shapes) between
        # neighbouring tables, which the DP-SGD bound does not cover; every column is then learned
        k = int(data.cfg["marginals"].get("min_dependence_rows", 0) or 0)
        self.indep = []
        if k and len(use) and o["dp"] is None:
            for j in self.own:
                x = np.asarray(data.X[j])[use]
                x = np.where(np.isnan(x), -1, x) if x.dtype.kind == "f" else x
                cnt = np.unique(x, return_counts=True)[1]
                if 0 < len(use) - cnt.max() < k:
                    self.indep.append(j)
        self.independent = [data.cols[j] for j in self.indep]
        net = [j for j in self.own if j not in self.indep]
        self.space = FeatureSpace(data, net, o["max_levels"], self.enc_m)
        self.cond_space = FeatureSpace(data, self.auxj, o["max_levels"], self.enc_m)
        g_fit = sorted(set(groups[use].tolist()))
        self.group_ix = {g: i + 1 for i, g in enumerate(g_fit)}
        coarse = self._coarse(groups[use]) if len(use) else np.zeros(0, np.int64)
        c_fit = sorted(set(coarse.tolist()))
        self.coarse_ix = {c: i + 1 for i, c in enumerate(c_fit)}
        self.device = T.device_of(context.compute)
        T.dtype_of(context.compute)                # validation only: rejects bf16/fp16, training runs in fp32
        self.dtype_name = "float32"
        self.report = None
        if self.n_fit == 0 or self.space.dim == 0:
            self.weights = None
            self.fitted = True
            return self
        with T.seeded(rng, context.compute) as gen:
            Z, S = self.space.encode(data.X, use, rng)
            A = self.cond_space.dense(data.X, use, rng)
            dev = self.device
            tensors = {"z": torch.as_tensor(Z, device=dev), "s": torch.as_tensor(S, device=dev),
                       "x1h": torch.as_tensor(self.space.one_hot(S), device=dev),
                       "g": torch.as_tensor(np.array([self.group_ix[int(g)] for g in groups[use]]), device=dev),
                       "c": torch.as_tensor(np.array([self.coarse_ix[int(c)] for c in coarse]), device=dev),
                       "a": torch.as_tensor(A, device=dev)}
            self.model = self._build().to(dev)
            self.report = T.train(self.model, self._loss_rows, tensors, self._noise, epochs=o["epochs"],
                                  batch_size=o["batch_size"], lr=o["lr"], weight_decay=o["weight_decay"], gen=gen,
                                  dp=o["dp"], valid_share=o["valid_share"], patience=o["patience"],
                                  max_steps=o["max_steps"])
            self.weights = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
            self.model.eval()
            self.num_maps, self.cat_bias, self.cat_bias_target = None, None, None
            if o["calibrate"]:
                self._calibrate(tensors, gen)
        self.fitted = True
        return self

    # ---------------------------------------------------------------- marginal calibration (Prop. 1805)
    def _slot_target(self, marginals) -> np.ndarray:
        """Probability of every one-hot slot under the given marginals (null, kept levels, other)."""
        out = np.zeros(self.space.n_cat)
        for i, c in enumerate(self.space.cats):
            m = marginals[c.col]
            p = np.asarray(m.p, float)
            p = p / p.sum() if p.sum() > 0 else p
            o, off = int(self.space.cat_offsets[i]), 1 if c.has_null else 0
            pn = float(m.p_null) if c.has_null else 0.0
            if c.has_null:
                out[o] = pn
            out[o + off: o + off + len(c.keep)] = (1 - pn) * p[c.keep]
            if c.has_other:
                out[o + c.width - 1] = (1 - pn) * (1 - p[c.keep].sum())
            blk = out[o:o + c.width]
            out[o:o + c.width] = (blk + 1e-6) / (blk + 1e-6).sum()
        return out

    def _calibrate(self, tensors, gen):
        """Monotone quantile maps of the scores and logit offsets of the categories, so that the
        network's aggregate law over the training rows' contexts matches the encoding marginals."""
        torch = T.torch_module()
        sp = self.space
        n = int(tensors["g"].shape[0])
        m = min(n, self.opts["calib_rows"])
        idx = torch.randperm(n, generator=gen, device=gen.device)[:m]
        cond = {"g": tensors["g"][idx], "c": tensors["c"][idx], "a": tensors["a"][idx]}
        target = self._slot_target(self.enc_m) if sp.cats else None
        rounds = 2 if sp.cats else 1
        with torch.no_grad():
            for _ in range(rounds):
                Z, S = self._generate(m, cond, gen)
                if sp.cats:
                    S = S.cpu().numpy()
                    q = np.zeros(sp.n_cat)
                    for i, K in enumerate(sp.cat_widths):       # slot frequencies, add-1/2 smoothed
                        o = int(sp.cat_offsets[i])
                        q[o:o + K] = (np.bincount(S[:, i], minlength=K)[:K] + 0.5) / (m + 0.5 * K)
                    step = np.log(target) - np.log(q)
                    self.cat_bias = step if self.cat_bias is None else self.cat_bias + step
                    self.cat_bias_target = target
        if sp.n_num:
            Z = Z.cpu().numpy()
            self.num_maps = [score_map(Z[:, i]) for i in range(sp.n_num)]

    def _apply_num_maps(self, Z: np.ndarray) -> np.ndarray:
        if not getattr(self, "num_maps", None):
            return Z
        out = np.empty_like(Z, dtype=float)
        for i, (x, y) in enumerate(self.num_maps):
            z = np.asarray(Z[:, i], float)
            v = np.interp(z, x, y)
            # beyond the calibration sample: continue with the end slopes
            lo, hi = z < x[0], z > x[-1]
            v[lo] = y[0] + (z[lo] - x[0]) * (y[1] - y[0]) / max(x[1] - x[0], 1e-12)
            v[hi] = y[-1] + (z[hi] - x[-1]) * (y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-12)
            out[:, i] = v
        return out

    def _bias_tensor(self, device):
        """Logit offsets of the current posterior draw: the calibrated offsets plus the shift from the
        encoding marginals to the decoding marginals (a prior-shift correction)."""
        b = getattr(self, "cat_bias", None)
        if b is None:
            return None
        torch = T.torch_module()
        if self.dec_m is not self.enc_m and getattr(self, "cat_bias_target", None) is not None:
            b = b + np.log(self._slot_target(self.dec_m)) - np.log(self.cat_bias_target)
        return torch.as_tensor(b, dtype=torch.float32, device=device)

    def _coarse(self, groups) -> np.ndarray:
        lut = {g: (c if c >= 0 else g) for g, c in self.parent.items()}
        return np.array([lut.get(int(g), int(g)) for g in np.asarray(groups).tolist()], dtype=np.int64)

    # ---------------------------------------------------------------- conditioning
    def _cond_inputs(self, n_rows: int, conditioning, rng):
        groups = np.asarray(conditioning.groups, np.int64)
        coarse = self._coarse(groups)
        g = np.array([self.group_ix.get(int(x), 0) for x in groups], dtype=np.int64)
        c = np.array([self.coarse_ix.get(int(x), 0) for x in coarse], dtype=np.int64)
        d = self.data
        if self.auxj:
            given = conditioning.given
            X = {}
            for j in self.auxj:
                col = d.cols[j]
                if given is not None and col in given:
                    X[j] = np.asarray(d.codecs[j].encode(given[col].reset_index(drop=True)))
                else:
                    X[j] = np.full(n_rows, -1 if d.codecs[j].kind == "discrete" else np.nan)
            A = self.cond_space.dense(X, np.arange(n_rows), rng)
        else:
            A = np.zeros((n_rows, 0), np.float32)
        return g, c, A

    def _ensure_model(self):
        torch = T.torch_module()
        if getattr(self, "model", None) is None:
            # layer initialisation draws from torch's global generators: keep the caller's state
            devs = list(range(torch.cuda.device_count())) if self.device != "cpu" else []
            with torch.random.fork_rng(devices=devs):
                self.model = self._build()
            self.model.load_state_dict({k: torch.as_tensor(v) for k, v in self.weights.items()})
            dev = self.device if (self.device == "cpu" or torch.cuda.is_available()) else "cpu"
            self.model = self.model.to(dev)
            self.device = dev
            self.model.eval()
        return self.model

    # ---------------------------------------------------------------- sample
    def sample(self, n, conditioning, rng):
        d = self.data
        M = len(conditioning)
        out = np.zeros((M, len(d.cols)))
        if M == 0:
            return d.decode(out, rng)
        if self.weights is None:                   # fitted on no rows: draw the marginals independently
            for j in self.own:
                out[:, j] = self.dec_m[j].sample(M, rng)
            return d.decode(out, rng)
        torch = T.torch_module()
        g, c, A = self._cond_inputs(M, conditioning, rng)
        self._ensure_model()
        Zs, Ss = [], []
        with T.seeded(rng, device=self.device) as gen, torch.no_grad():
            bs = self.opts["sample_batch"]
            for i in range(0, M, bs):
                sl = slice(i, i + bs)
                cond = {"g": torch.as_tensor(g[sl], device=self.device), "c": torch.as_tensor(c[sl], device=self.device),
                        "a": torch.as_tensor(A[sl], device=self.device)}
                z, s = self._generate(len(g[sl]), cond, gen)
                Zs.append(z.cpu().numpy())
                Ss.append(s.cpu().numpy())
        Z = self._apply_num_maps(np.concatenate(Zs)) if self.space.n_num else np.zeros((M, 0))
        S = np.concatenate(Ss) if self.space.cats else np.zeros((M, 0), np.int64)
        self.space.decode_into(out, Z, S, self.dec_m, rng)
        for j in getattr(self, "indep", []):        # rare columns: independent marginal draws
            out[:, j] = self.dec_m[j].sample(M, rng)
        return d.decode(out, rng)

    # ---------------------------------------------------------------- posterior
    def posterior_draw(self, rng, mode="draw"):
        from ...posterior import param_rng
        if self.data.cfg["marginals"]["bayesian_bootstrap"]:
            if getattr(self.data, "X", None) is None:
                # a loaded model holds no source rows, so the marginals cannot be redrawn: every dataset
                # decodes through the marginals saved with it (posterior_frozen says so in the report)
                self.posterior_frozen = True
                return self
            self.dec_m = table_marginals(self.data, param_rng(rng, mode), bootstrap=True)
        else:
            self.dec_m = self.enc_m
        return self

    # ---------------------------------------------------------------- reports
    def fitted_groups(self):
        return set(getattr(self, "group_ix", {}))

    def uncertainty(self):
        r = self.report
        out = {"rows_fitted": getattr(self, "n_fit", 0), "groups": len(getattr(self, "group_ix", {})),
               "features": getattr(getattr(self, "space", None), "dim", 0),
               "posterior_frozen": bool(getattr(self, "posterior_frozen", False))}
        if r is not None:
            out.update({"steps": r.steps, "epochs": round(r.epochs_run, 2), "best_epoch": r.best_epoch,
                        "train_loss_last": r.train_loss[-1] if r.train_loss else None,
                        "valid_loss_best": min(r.valid_loss) if r.valid_loss else None})
        return out

    def privacy_cost(self) -> PrivacyCost:
        r = self.report
        if r is not None and r.dp:
            return PrivacyCost(dp=False, epsilon=r.dp["epsilon"], delta=r.dp["delta"], mechanism=r.dp["mechanism"],
                               note=(f"{self.name}: DP-SGD bounds what the network weights reveal about any one row "
                                     "of this table (row-level, this table only). The encoding marginals, category "
                                     "levels and latent groups are fitted without noise, so this is not an "
                                     "end-to-end guarantee; use relsynth run --dp for one (docs/neural.md)."))
        return PrivacyCost(dp=False, note=f"{self.name}: no formal privacy guarantee")

    def summary(self):
        out = {"generator": self.name, "version": self.version}
        r = self.report
        if r is not None and r.dp:
            out["dp_sgd"] = dict(r.dp)
        return out

    # ---------------------------------------------------------------- persistence
    def state(self):
        """The persisted state: no torch model (rebuilt from the numpy weights on load) and no source
        rows: the table data is a detached copy without its encoded rows (X) and without the back-
        reference to the generator, and the per-row segment labels are dropped.  A loaded model samples
        with the marginals it was saved with (posterior_draw needs the rows)."""
        st = dict(self.__dict__)
        st.pop("model", None)
        st.pop("_groups", None)
        d = st.get("data")
        if d is not None:
            d = copy.copy(d)
            d.X = None
            if hasattr(d, "generator"):
                d.generator = None
            st["data"] = d
        return st

    # ---------------------------------------------------------------- shared pieces for subclasses
    def _cond_module(self):
        """The conditioning encoder: group and coarse embeddings and the parent attributes."""
        torch = T.torch_module()
        nn = torch.nn
        cd, a_dim = self.opts["cond_dim"], self.cond_space.dim
        G, C = len(self.group_ix) + 1, len(self.coarse_ix) + 1

        class Cond(nn.Module):
            def __init__(self):
                super().__init__()
                self.g = nn.Embedding(G, cd)
                self.c = nn.Embedding(C, cd)
                self.out_dim = 2 * cd + a_dim

            def forward(self, g, c, a):
                return torch.cat([self.g(g), self.c(c), a], dim=1)

        return Cond()

    def _drop_cond(self, batch, noise):
        """Group indices with the null token for the rows drawn by p_uncond."""
        p = self.opts["p_uncond"]
        g = batch["g"] * (noise["ug"] >= p).long()
        c = batch["c"] * (noise["uc"] >= p / 2).long()
        return g, c


def mlp(sizes, torch, act="relu"):
    nn = torch.nn
    layers = []
    for a, b in zip(sizes[:-2], sizes[1:-1]):
        layers += [nn.Linear(a, b), nn.ReLU() if act == "relu" else nn.SiLU()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


__all__ = ["NeuralGenerator", "mlp", "pd"]
