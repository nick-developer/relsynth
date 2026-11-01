# SPDX-License-Identifier: Apache-2.0
"""Synthetic free text conditioned on the row (roadmap T14; guide docs/text.md,
Props. 1500-1507).

A text column picks a method in its spec (`columns.<c>.text`):

    body:
      type: text
      text:
        condition_on: [category, priority]     # one column or several
        method: ngram | lm | pe                # default ngram (relsynth 1.0)
        pii: {recognizers: [regex, {name: gliner, model: /models/gliner_multi_pii-v1}]}
        dp: {epsilon: 2.0, delta: 1.0e-6, unit_column: customer_id}
        lm: {model: /models/SmolLM2-135M, lora: {r: 8}, steps: 200}
        pe: {api: {kind: lm, model: /models/SmolLM2-135M}, iterations: 4, pool: 64}

* `ngram` without `dp` is the 1.0 model, byte for byte; with `dp` it is the
  DP n-gram release of `dpngram` (no weights, no GPU).
* `lm` fine-tunes a local causal language model (`lm`), with DP-GD under
  `dp`.
* `pe` is Private Evolution (`pe`), always DP; its API is public (a local
  pretrained model or an n-gram model of a public corpus).

`dp` here protects the people of `unit_column` in *this column*: the rest of
an ordinary run is not differentially private, and the run's statement says
so.  In DP mode (`relsynth run --dp`), text columns are still left empty;
see RFC-T14-01.
"""
from __future__ import annotations

from ..spec import SpecError

METHODS = ("ngram", "lm", "pe")
NEW_KEYS = {"method", "dp", "pii", "lm", "pe"}
DP_KEYS = {"epsilon", "rho", "delta", "unit_column", "max_texts", "max_tokens", "max_keys", "clip", "selection_delta_share"}


def is_extended(o: dict) -> bool:
    """True when a text column uses anything beyond the 1.0 options (the 1.0
    path, and its random stream, are kept otherwise)."""
    return bool(set(o or {}) & NEW_KEYS) or isinstance((o or {}).get("condition_on"), list)


def condition_cols(o: dict) -> list[str]:
    c = (o or {}).get("condition_on")
    return [] if not c else [c] if isinstance(c, str) else list(c)


def class_keys(df, cols, seg=None) -> list[str]:
    if not cols:
        return [str(s) for s in seg]
    if len(cols) == 1:
        return df[cols[0]].astype(str).tolist()
    return [" | ".join(f"{c}={v}" for c, v in zip(cols, vals)) for vals in zip(*[df[c].astype(str) for c in cols])]


def cond_dicts(df, cols, seg=None) -> list[dict]:
    if not cols:
        return [{"segment": str(s)} for s in seg]
    return [dict(zip(cols, vals)) for vals in zip(*[df[c].astype(str) for c in cols])]


def dp_budget(d: dict) -> tuple[float, float, float]:
    """(rho for the mechanisms, delta for partition selection, delta stated)."""
    from ..governance.accounting import rho_of
    if not d.get("unit_column"):
        raise SpecError("text.dp.unit_column: the column holding the person each text belongs to (customer_id)")
    delta = d.get("delta")
    if delta is None or not 0 < float(delta) <= 1e-5:
        raise SpecError("text.dp.delta: set delta in (0, 1e-5] (NIST SP 800-226)")
    delta = float(delta)
    share = float(d.get("selection_delta_share", 0.5))
    if d.get("rho") is not None and d.get("epsilon") is not None:
        raise SpecError("text.dp: give epsilon or rho, not both")
    if d.get("rho") is not None:
        rho = float(d["rho"])
    elif d.get("epsilon") is not None:
        rho = rho_of(float(d["epsilon"]), delta * (1 - share))
    else:
        raise SpecError("text.dp: set epsilon (with delta) or rho; relsynth has no default epsilon")
    if not rho > 0:
        raise SpecError("text.dp: the budget must be positive")
    return rho, delta * share, delta


class TextColumn:
    """One text column under an extended configuration: PII redaction, the
    chosen method, DP accounting and typed-placeholder filling."""

    def __init__(self, table: str, column: str, o: dict, settings: dict | None = None):
        self.table, self.column, self.o = table, column, dict(o or {})
        self.method = self.o.get("method", "ngram")
        if self.method not in METHODS:
            raise SpecError(f"{table}.{column}: text.method must be one of {', '.join(METHODS)}")
        if self.method == "pe" and not self.o.get("dp"):
            raise SpecError(f"{table}.{column}: text.method pe is differentially private by design: set text.dp")
        self.cols = condition_cols(self.o)
        self.settings = settings or {}
        self.info: dict = {"method": self.method}
        self.pii = None
        if self.o.get("pii"):
            from .pii import PiiPipeline
            p = self.o["pii"]
            rec = p.get("recognizers", ["regex"]) if isinstance(p, dict) else ["regex"]
            self.pii = PiiPipeline(rec)

    def fit(self, df, seg=None):
        texts = df[self.column].tolist()
        if self.pii is not None:
            texts, counts = self.pii.redact_all(texts)
            self.info["pii"] = {"recognizers": self.pii.names, "redacted": counts}
        d = self.o.get("dp")
        if d:
            from ..dp.mechanisms import Accountant
            rho, d_sel, delta = dp_budget(d)
            if d["unit_column"] not in df.columns:
                raise SpecError(f"{self.table}.{self.column}: text.dp.unit_column {d['unit_column']} is not a column")
            owners = df[d["unit_column"]].tolist()
            acct = Accountant(rho=rho, delta=d_sel)
        cls = class_keys(df, self.cols, seg)
        t = self.settings
        if self.method == "ngram" and not d:
            from ..text import TextModel
            self.model = TextModel(order=self.o.get("order", t.get("order", 3)),
                                   min_token_rows=self.o.get("min_token_rows", t.get("min_token_rows", 3)),
                                   max_copy_ngram=self.o.get("max_copy_ngram", t.get("max_copy_ngram", 8)),
                                   beta=t.get("backoff_strength", 2.0), max_retries=t.get("max_retries", 30)
                                   ).fit(texts, cls)
        elif self.method == "ngram":
            from .dpngram import DPNgramModel
            self.model = DPNgramModel(order=self.o.get("order", 2), max_texts=d.get("max_texts", 4),
                                      max_tokens=d.get("max_tokens", 40), max_keys=d.get("max_keys", 64),
                                      beta=t.get("backoff_strength", 2.0), max_retries=t.get("max_retries", 30)
                                      ).fit_dp(texts, owners, cls, acct, rho, d_sel, label=f"{self.table}.{self.column}")
        elif self.method == "lm":
            from .lm import LMTextModel
            lo = dict(self.o.get("lm") or {})
            if d:
                lo["dp"] = {"clip": d.get("clip", 1.0), "max_texts": d.get("max_texts", 4)}
            self.model = LMTextModel(**lo)
            conds = cond_dicts(df, self.cols, seg)
            if d:
                self.model.fit(texts, conds, owners=owners, acct=acct, rho=rho)
            else:
                self.model.fit(texts, conds)
        else:
            from .pe import LMApi, NgramApi, PETextModel
            po = dict(self.o.get("pe") or {})
            api = po.pop("api", None) or {}
            if api.get("kind") == "lm":
                a = LMApi(api["model"], **{k: v for k, v in api.items() if k not in ("kind", "model")})
            elif api.get("kind") == "ngram" and api.get("corpus"):
                import pandas as pd
                corpus = pd.read_csv(api["corpus"])
                col = api.get("column", "text")
                a = NgramApi(corpus[col].tolist(), corpus[api["class_column"]].astype(str).tolist()
                             if api.get("class_column") else None)
            else:
                raise SpecError(f"{self.table}.{self.column}: text.pe.api: {{kind: lm, model: <folder>}} or "
                                "{kind: ngram, corpus: <public CSV>, column: text, class_column: ...}")
            self.model = PETextModel(a, iterations=po.get("iterations", 4), pool=po.get("pool", 64),
                                     max_texts=d.get("max_texts", 4), threshold=po.get("threshold", 2.0),
                                     embed=po.get("embed")).fit_dp(texts, owners, cls, acct, rho, d_sel,
                                                                    label=f"{self.table}.{self.column}")
        if d:
            from ..governance.accounting import epsilon_of
            r_spent, d_spent = acct.spent
            self.info["dp"] = {"scope": f"column {self.table}.{self.column} only; the rest of the run is not DP",
                               "unit_column": d["unit_column"], "rho": r_spent, "selection_delta": d_spent,
                               "delta": delta, "epsilon": epsilon_of(r_spent, delta - d_spent) if r_spent > 0 else 0.0,
                               "charges": [c.__dict__ for c in acct.charges],
                               "release": getattr(self.model, "release", {})}
        return self

    def generate(self, df, seg, rng) -> list:
        if self.method == "lm":
            out = self.model.generate(cond_dicts(df, self.cols, seg), rng)
        else:
            out = self.model.generate(class_keys(df, self.cols, seg), rng)
        from .pii import fill
        return [fill(x, rng) if isinstance(x, str) else x for x in out]

    @property
    def last_rejected(self) -> int:
        return int(getattr(self.model, "last_rejected", 0))


def logprob_fn(model):
    """A log-likelihood function (tokens, class) -> float for the audits."""
    from .dpngram import ngram_logprob
    from ..text import TextModel
    if isinstance(model, TextModel):
        return lambda toks, c: ngram_logprob(model, toks, c)
    if hasattr(model, "logprob") and hasattr(model, "net"):
        from ..text import detok
        return lambda toks, c: model.logprob(detok(toks), c if isinstance(c, dict) else {"condition": c})
    return None


__all__ = ["TextColumn", "METHODS", "is_extended", "dp_budget", "logprob_fn", "class_keys", "cond_dicts"]
