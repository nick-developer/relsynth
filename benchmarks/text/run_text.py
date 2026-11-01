# SPDX-License-Identifier: Apache-2.0
"""Synthetic-text benchmark (roadmap T14; numbers in the T14 report).

    python benchmarks/text/run_text.py --out bench-out/text.json [--quick] [--lm /models/pythia-14m]

Corpora:
  S6, S7   the fake ticket scenarios (tests/fakedata/scenarios.py);
  banking77  PolyAI BANKING77 (CC BY 4.0; Casanueva et al. 2020): 13,083 customer
           service queries in 77 intents.  Fetched at run time from the
           authors' repository into the bench cache and checked against pinned
           SHA-256 digests; only aggregate numbers are written.  The queries
           are given to synthetic people (1-3 queries each), the intent and its
           topic (card, transfer, top up, ...) are the row values the text is
           conditioned on.  The official test split stands in for a public
           corpus for Private Evolution's API (same domain: optimistic).

Methods: the 1.0 n-gram model; the DP n-gram release at eps 1, 4, 8; Private
Evolution at eps 4 with an n-gram API on the public split; with --lm a local
model fine-tuned without and with DP.  Metrics: MAUVE (hashed embedding),
consistency with the intent and topic, TSTR, exact/near copies, and the
canary audit (64 canaries x 4 copies).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.datasets import CACHE, DOWNLOADS, download  # noqa: E402

B77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
DOWNLOADS.setdefault("banking77_train.csv", (B77 + "train.csv",
                                             "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"))
DOWNLOADS.setdefault("banking77_test.csv", (B77 + "test.csv",
                                            "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"))
TOPICS = ("card", "transfer", "top_up", "cash", "exchange", "pin", "refund", "payment", "balance", "account",
          "identity", "verify", "fee", "charge", "atm", "country", "disposable", "virtual", "apple", "beneficiary")


def topic(intent: str) -> str:
    for t in TOPICS:
        if t in intent:
            return t
    return "other"


def banking77(seed=77):
    rng = np.random.default_rng(seed)
    tr = pd.read_csv(download("banking77_train.csv"))
    te = pd.read_csv(download("banking77_test.csv"))
    out = []
    for df in (tr, te):
        df = df.rename(columns={"category": "intent"}).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        sizes = []
        while sum(sizes) < len(df):
            sizes.append(int(rng.integers(1, 4)))
        owner = np.repeat(np.arange(len(sizes)), sizes)[: len(df)]
        df["customer_id"] = owner + (0 if not out else 10 ** 6)
        df["topic"] = df.intent.map(topic)
        out.append(df)
    return out[0], out[1]


def scenario(name):
    sys.path.insert(0, str(ROOT / "tests"))
    from fakedata.scenarios import enterprise, text
    tk = (text(6) if name == "S6" else enterprise(7))[0]["ticket"]
    tk = tk.rename(columns={"body": "text", "category": "intent", "priority": "topic"})
    m = np.random.default_rng(1).random(len(tk)) < 0.7
    return tk[m].reset_index(drop=True), tk[~m].reset_index(drop=True)


def evaluate(train, test, opts, quick, audit=True):
    from relsynth.textgen import TextColumn
    from relsynth.textgen.audit import run_audit
    from relsynth.textgen.metrics import consistency, mauve, tstr
    from relsynth.textgen.audit import near_duplicates
    t0 = time.time()
    tc = TextColumn("bench", "text", opts).fit(train)
    fit_s = time.time() - t0
    rng = np.random.default_rng(5)
    syn = test[["intent", "topic", "customer_id"]].copy()
    syn["text"] = tc.generate(syn, None, rng)
    r = {"fit_seconds": round(fit_s, 1), "null_rate": float(syn.text.isna().mean())}
    n = min(len(test), 1500 if quick else 3000)
    r["mauve"] = mauve(test.text.sample(n, random_state=0).tolist(), syn.text.dropna().sample(
        min(n, syn.text.notna().sum()), random_state=0).tolist())
    r["consistency"] = {c: {k: v[k] for k in ("real", "synthetic", "ratio", "synthetic_ci", "chance")}
                        for c, v in consistency(pd.concat([train, test]), syn, "text", ["intent", "topic"]).items()}
    r["tstr"] = tstr(pd.concat([test]), syn.assign(text=syn.text), "text", "intent")
    tstr_real = tstr(pd.concat([train, test]), syn, "text", "intent")
    r["tstr_trainreal"] = tstr_real
    r["copies"] = near_duplicates(train.text.tolist(), syn.text.tolist())
    if audit:
        a = run_audit(train, "text", opts, canaries=32 if quick else 64, repeats=4, words=12, samples=500 if quick else 1500,
                      owner_column="customer_id")
        r["audit"] = {"exposure": {k: v for k, v in (a.get("exposure") or {}).items() if k != "values"},
                      "duplicates": a["duplicates"], "extraction": {k: v for k, v in a["extraction"].items()
                                                                    if k != "examples"},
                      "canaries_generated": a["canaries_generated"]}
    if tc.info.get("dp"):
        r["dp"] = {k: tc.info["dp"][k] for k in ("epsilon", "delta", "rho")}
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench-out/text.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--corpus", action="append", help="S6, S7, banking77 (default: all)")
    ap.add_argument("--lm", help="local model folder for method lm (fine-tuning without and with DP)")
    ap.add_argument("--lm-steps", type=int, default=150)
    a = ap.parse_args(argv)
    res = {}
    for name in a.corpus or ["S6", "S7", "banking77"]:
        train, test = banking77() if name == "banking77" else scenario(name)
        unit = {"unit_column": "customer_id", "delta": 1e-6}
        methods = {"ngram-1.0": {"condition_on": "intent"},
                   "ngram-1.0-two-cols": {"condition_on": ["intent", "topic"]}}
        for eps in (1.0, 4.0, 8.0):
            methods[f"dp-ngram-eps{eps:g}"] = {"condition_on": "intent", "method": "ngram", "dp": {**unit, "epsilon": eps}}
        pub = test if name == "banking77" else scenario("S7" if name == "S6" else "S6")[1]
        corpus = CACHE / "text" / f"public_{name}.csv"           # stays in the ignored cache
        corpus.parent.mkdir(parents=True, exist_ok=True)
        pub[["text", "intent"]].to_csv(corpus, index=False)
        methods["pe-eps4"] = {"condition_on": "intent", "method": "pe", "dp": {**unit, "epsilon": 4.0},
                              "pe": {"api": {"kind": "ngram", "corpus": str(corpus), "column": "text",
                                             "class_column": "intent"}, "iterations": 4, "pool": 48}}
        if a.lm:
            lm = {"model": a.lm, "steps": a.lm_steps, "batch_size": 16, "lr": 5e-4, "max_tokens": 48,
                  "max_new_tokens": 48}
            methods["lm"] = {"condition_on": "intent", "method": "lm", "lm": lm}
            methods["lm-dp-eps8"] = {"condition_on": "intent", "method": "lm", "lm": {**lm, "steps": 20, "lr": 2e-3},
                                     "dp": {**unit, "epsilon": 8.0, "max_texts": 2}}
        res[name] = {"rows_train": len(train), "rows_test": len(test), "people": int(train.customer_id.nunique()),
                     "methods": {}}
        for m, o in methods.items():
            print(f"{name} {m} ...", flush=True)
            r = evaluate(train, test, o, a.quick, audit=not m.startswith("lm"))
            res[name]["methods"][m] = r
            c = r["consistency"].get("intent", {})
            print(f"  mauve {r['mauve']:.3f}  consistency {c.get('ratio', float('nan')):.3f}  "
                  f"tstr {r['tstr'].get('ratio', float('nan')):.3f}  exact copies {r['copies']['exact_rate']:.3f}  "
                  f"exposure {((r.get('audit') or {}).get('exposure') or {}).get('mean', float('nan')):.2f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
