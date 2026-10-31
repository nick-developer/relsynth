# SPDX-License-Identifier: Apache-2.0
"""Row-conditioned text from a local causal language model, optionally fine-tuned
with user-level differential privacy (Prop. 1502).

The model is read from a local folder (`local_files_only`; relsynth never
downloads or bundles weights: the user fetches them under the model's own
licence, see `relsynth text models`), or is the built-in `stub`: a two-layer
GPT-2-shaped network over bytes, randomly initialised, for tests and for
air-gapped installations without any weights.

Each row becomes the prompt "col1: v1 | col2: v2\\n" (the conditioning values);
the text follows and the loss is taken on the text only.  Fine-tuning updates
LoRA adapters (Hu et al. ICLR 2022, through `peft`) when `lora` is set, else
every weight.

**DP fine-tuning** (`dp`): DP-GD with user-level clipping (Abadi et al. CCS
2016; McMahan et al. ICLR 2018).  Each person keeps at most `max_texts` texts;
in every step each person's gradient (of their mean loss) is clipped to L2 norm
C, the clipped gradients are summed over *all* people (no sampling, so the
Gaussian mechanism's zCDP map is exact: rho = T / (2 sigma^2) for T steps,
Bun and Steinke 2016), Gaussian noise N(0, sigma^2 C^2) is added and the sum
is divided by a noisy count of people.  Full batches are the regime where
DP training works best anyway (De et al. 2022; Sander et al. ICML 2023).  The
noise is floating point, as in Opacus; the charges go to the DP-mode
`Accountant`.  Subsampled DP-SGD with an RDP accountant is an open item
(RFC-T14-01).
"""
from __future__ import annotations

import math
import secrets

import numpy as np

STUB = "stub"


def _torch():
    try:
        import torch
    except ImportError as e:
        from ..spec import SpecError
        raise SpecError("text method 'lm' needs PyTorch and transformers: pip install 'relsynth[text-lm]'") from e
    return torch


def prompt_of(values: dict) -> str:
    return " | ".join(f"{k}: {v}" for k, v in values.items()) + "\n"


class ByteTokenizer:
    """Bytes 0-255, 256 = end of text / padding.  Data-independent, so it
    releases nothing about the training texts."""
    eos_token_id = 256
    pad_token_id = 256
    vocab_size = 257

    def encode(self, s: str) -> list[int]:
        return list(s.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="ignore")


class _HFTok:
    def __init__(self, tok):
        self.t = tok
        self.eos_token_id = tok.eos_token_id
        self.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def encode(self, s):
        return self.t.encode(s, add_special_tokens=False)

    def decode(self, ids):
        return self.t.decode(ids, skip_special_tokens=True)


class LMTextModel:
    def __init__(self, model: str = STUB, max_tokens: int = 64, steps: int = 200, batch_size: int = 32, lr: float = 1e-3,
                 lora: dict | None = None, dp: dict | None = None, device: str | None = None, temperature: float = 1.0,
                 top_p: float = 0.95, max_new_tokens: int = 64, seed: int = 0, stub_size: dict | None = None):
        self.model_name, self.max_tokens, self.steps, self.bs, self.lr = model, int(max_tokens), int(steps), \
            int(batch_size), float(lr)
        self.lora, self.dp, self.device = lora, dp, device
        self.temperature, self.top_p, self.max_new_tokens, self.seed = float(temperature), float(top_p), \
            int(max_new_tokens), int(seed)
        self.stub_size = dict(stub_size or {"n_embd": 64, "n_layer": 2, "n_head": 2})
        self.release: dict = {}
        self.cond_cols: list = []

    # ------------------------------------------------------------ model
    def _load(self):
        torch = _torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
        torch.manual_seed(self.seed)
        if self.model_name == STUB:
            self.tok = ByteTokenizer()
            cfg = GPT2Config(vocab_size=self.tok.vocab_size, n_positions=512, bos_token_id=256, eos_token_id=256,
                             **self.stub_size)
            self.net = GPT2LMHeadModel(cfg)
        else:
            self.tok = _HFTok(AutoTokenizer.from_pretrained(self.model_name, local_files_only=True))
            self.net = AutoModelForCausalLM.from_pretrained(self.model_name, local_files_only=True)
        if self.lora:
            from peft import LoraConfig, get_peft_model
            o = dict(self.lora)
            self.net = get_peft_model(self.net, LoraConfig(r=int(o.get("r", 8)), lora_alpha=int(o.get("alpha", 16)),
                                                          lora_dropout=0.0, target_modules=o.get("targets"),
                                                          task_type="CAUSAL_LM"))
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dev = torch.device(dev)
        self.net.to(self.dev)
        self.params = [p for p in self.net.parameters() if p.requires_grad]

    def _example(self, prompt: str, text: str):
        p = self.tok.encode(prompt)
        t = self.tok.encode(text)[: self.max_tokens] + [self.tok.eos_token_id]
        return p + t, [-100] * len(p) + t

    def _batch(self, exs):
        torch = _torch()
        L = max(len(x) for x, _ in exs)
        ids = torch.full((len(exs), L), self.tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(exs), L), -100, dtype=torch.long)
        att = torch.zeros((len(exs), L), dtype=torch.long)
        for i, (x, y) in enumerate(exs):
            ids[i, : len(x)] = torch.tensor(x)
            lab[i, : len(y)] = torch.tensor(y)
            att[i, : len(x)] = 1
        return ids.to(self.dev), lab.to(self.dev), att.to(self.dev)

    def _loss(self, exs):
        ids, lab, att = self._batch(exs)
        return self.net(input_ids=ids, attention_mask=att, labels=lab).loss

    # ------------------------------------------------------------ fit
    def fit(self, texts, conds: list[dict], owners=None, acct=None, rho: float | None = None, delta: float = 0.0):
        """texts[i] with its conditioning values conds[i] ({column: value}).  With
        `dp`, owners[i] is the person of row i and (acct, rho) the budget."""
        torch = _torch()
        self._load()
        rows = [(t, c) for t, c in zip(texts, conds) if t is not None and not (isinstance(t, float) and math.isnan(t))
                and str(t) != ""]
        self.null_rate = 1 - len(rows) / max(len(texts), 1)
        self.cond_cols = list(conds[0]) if conds else []
        opt = torch.optim.Adam(self.params, lr=self.lr)
        self.net.train()
        if not self.dp:
            exs = [self._example(prompt_of(c), str(t)) for t, c in rows]
            g = np.random.default_rng(self.seed)
            for _ in range(self.steps):
                b = [exs[j] for j in g.choice(len(exs), min(self.bs, len(exs)), replace=False)]
                opt.zero_grad()
                self._loss(b).backward()
                opt.step()
            self.net.eval()
            return self
        return self._fit_dp(texts, conds, owners, acct, rho, opt)

    def _fit_dp(self, texts, conds, owners, acct, rho, opt):
        torch = _torch()
        from ..dp.mechanisms import Charge, gaussian
        if acct is None or rho is None or owners is None:
            raise ValueError("DP fine-tuning needs owners, an accountant and rho")
        o = dict(self.dp)
        C = float(o.get("clip", 1.0))
        k = int(o.get("max_texts", 4))
        T = self.steps
        count_share = float(o.get("count_share", 0.02))
        rng = np.random.default_rng(secrets.randbits(128))
        by: dict = {}
        for t, c, u in zip(texts, conds, owners):
            if t is None or (isinstance(t, float) and math.isnan(t)) or str(t) == "":
                continue
            by.setdefault(u, []).append(self._example(prompt_of(c), str(t)))
        people = []
        for exs in by.values():
            if len(exs) > k:
                exs = [exs[j] for j in rng.choice(len(exs), k, replace=False)]
            people.append(exs)
        n_noisy = int(gaussian([len(people)], 1, rho * count_share, acct, "fit", "text lm: number of people")[0])
        n_div = max(n_noisy, 1)
        rho_steps = rho * (1 - count_share)
        sigma = math.sqrt(T / (2 * rho_steps))          # noise multiplier: rho_steps = T / (2 sigma^2)
        acct.charge(Charge("fit", f"text lm: DP-GD ({T} steps, clip {C}, sigma {sigma:.4g})", rho_steps))
        gen = torch.Generator(device="cpu")
        gen.manual_seed(secrets.randbits(63))
        clipped_share = []
        for _ in range(T):
            acc = [torch.zeros_like(p) for p in self.params]
            n_clip = 0
            for exs in people:
                self.net.zero_grad(set_to_none=True)
                self._loss(exs).backward()
                gs = [p.grad.detach() if p.grad is not None else torch.zeros_like(p) for p in self.params]
                norm = math.sqrt(sum(float((g.float() ** 2).sum()) for g in gs))
                f = min(1.0, C / (norm + 1e-12))
                n_clip += f < 1.0
                for a, g in zip(acc, gs):
                    a.add_(g, alpha=f)
            for p, a in zip(self.params, acc):
                noise = torch.randn(a.shape, generator=gen).to(a.device) * (sigma * C)
                p.grad = (a + noise) / n_div
            opt.step()
            clipped_share.append(n_clip / max(len(people), 1))
        self.net.eval()
        self.release = {"method": "DP-GD (user-level, full batch)", "people": len(people), "max_texts": k, "clip": C,
                        "steps": T, "noise_multiplier": sigma, "rho": rho, "clipped_share_last": clipped_share[-1]}
        return self

    # ------------------------------------------------------------ sample and score
    def generate(self, conds: list[dict], rng: np.random.Generator) -> list:
        torch = _torch()
        torch.manual_seed(int(rng.integers(2 ** 62)))
        out: list = [None] * len(conds)
        groups: dict = {}
        for i, c in enumerate(conds):
            if rng.random() < self.null_rate:
                continue
            groups.setdefault(prompt_of(c), []).append(i)
        with torch.no_grad():
            for prm, idx in groups.items():
                p = self.tok.encode(prm)
                for s in range(0, len(idx), 64):
                    chunk = idx[s:s + 64]
                    ids = torch.tensor([p] * len(chunk), device=self.dev)
                    res = self.net.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                            temperature=self.temperature, top_p=self.top_p,
                                            max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                            eos_token_id=self.tok.eos_token_id)
                    for j, row in zip(chunk, res[:, len(p):].tolist()):
                        if self.tok.eos_token_id in row:
                            row = row[: row.index(self.tok.eos_token_id)]
                        out[j] = self.tok.decode(row).strip() or None
        return out

    def logprob(self, text: str, cond: dict) -> float:
        torch = _torch()
        x, y = self._example(prompt_of(cond), text)
        ids, lab, att = self._batch([(x, y)])
        with torch.no_grad():
            logits = self.net(input_ids=ids, attention_mask=att).logits[0, :-1]
            tgt = lab[0, 1:]
            lp = torch.log_softmax(logits.float(), -1)
            m = tgt != -100
            return float(lp[m].gather(1, tgt[m][:, None]).sum())
