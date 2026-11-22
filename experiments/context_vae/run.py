# SPDX-License-Identifier: Apache-2.0
"""Run the context-VAE ablation on one dataset and seed; writes one JSON report.

    python -m experiments.context_vae.run --dataset chinook --seed 1 --out out/chinook-1.json

Variants are ``encoder:prior`` (encoder transformer|mlp, prior none|gru|transformer) plus
``shuffle`` (real rows permuted across families: exact marginals, no within-family dependence;
the reference value of every within-family metric under independence).  Every variant gets the
real training families' counts, contexts and gaps, as the engine would hand them over.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import metrics as M
from .data import DATASETS, Codec, families, load, split, subset

DEFAULT_VARIANTS = "transformer:none,transformer:gru,transformer:transformer,mlp:transformer,shuffle"


def _db(name: str, given: str | None) -> str:
    if given:
        return given
    import bench.datasets as bd
    loc = bd.DATASETS[DATASETS[name].builder].location()
    if not str(loc).startswith("sqlite:"):
        raise SystemExit(f"{name}: expected a SQLite bench source, got {loc}")
    return str(loc)[len("sqlite:"):]


def evaluate(real, syn_X, syn_C, syn_off, widths, ordered, ctx_C, hold, seed):
    r_off = real.off
    out = {"marginals": M.marginals(real.X, real.C, syn_X, syn_C, widths)}
    wr, ws = M.within(real.X, real.C, r_off, widths), M.within(syn_X, syn_C, syn_off, widths)
    out["within_real"], out["within_syn"] = wr, ws
    cat = [abs(ws[k] - wr[k]) for k in wr if k.endswith("_share")]
    num = [abs(ws[k] - wr[k]) for k in wr if k.endswith("_within")]
    out["err_cat_share"] = float(np.mean(cat)) if cat else None
    out["err_num_within"] = float(np.mean(num)) if num else None
    if ordered:
        sr, ss = M.sequence(real.X, real.C, r_off, widths), M.sequence(syn_X, syn_C, syn_off, widths)
        lag = [abs(ss[k] - sr[k]) for k in sr if k.endswith("_lag1") and sr[k] is not None and ss[k] is not None]
        tv = [M.trans_tvd(sr[k], ss[k]) for k in sr if k.endswith("_trans")]
        out["err_lag1"] = float(np.mean(lag)) if lag else None
        out["trans_tvd"] = float(np.mean(tv)) if tv else None
    Fr = M.family_features(real.X, real.C, r_off, widths, real.ctx_C)
    Fs = M.family_features(syn_X, syn_C, syn_off, widths, ctx_C)
    Fh = M.family_features(hold.X, hold.C, hold.off, widths, hold.ctx_C)
    out["c2st_agg"] = M.c2st(Fr, Fs, seed)
    # row-level C2ST separates row fidelity (decoder, marginals) from family structure
    rng = np.random.default_rng(seed)
    pr = rng.choice(len(real.X), min(len(real.X), 20000), replace=False)
    ps = rng.choice(len(syn_X), min(len(syn_X), 20000), replace=False)
    out["c2st_row"] = M.c2st(M.row_matrix(real.X[pr], real.C[pr], widths), M.row_matrix(syn_X[ps], syn_C[ps], widths), seed)
    # the family's size and context come from the engine (copied from a training family here), so
    # they are left out of the distance: it measures what the model generated
    nctx = real.ctx_C.shape[1]
    keep = slice(1, Fr.shape[1] - nctx)
    sb = lambda off: np.floor(np.log2(np.diff(off))).astype(int)
    out["dcr_family"] = M.dcr_share_strata(Fs[:, keep], sb(syn_off), Fr[:, keep], sb(r_off), Fh[:, keep], sb(hold.off), seed)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(syn_X), min(len(syn_X), 20000), replace=False)
    out["dcr_row"] = M.dcr_share(M.row_matrix(syn_X[pick], syn_C[pick], widths),
                                 M.row_matrix(real.X, real.C, widths), M.row_matrix(hold.X, hold.C, widths), seed)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--db")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--vae-steps", type=int, default=3000)
    ap.add_argument("--prior-steps", type=int, default=3000)
    ap.add_argument("--max-families", type=int, default=-1, help="subsample families (0 = all, -1 = dataset default)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    from .model import Config, ContextVAE

    d = DATASETS[a.dataset]
    df = load(d, _db(a.dataset, a.db))
    rng = np.random.default_rng(a.seed)
    keys = df["fam"].unique()
    if a.max_families < 0:
        a.max_families = d.max_families
    if a.max_families and len(keys) > a.max_families:
        keys = rng.choice(keys, a.max_families, replace=False)
        df = df[df["fam"].isin(keys)].reset_index(drop=True)
    codec = Codec(df, d.num, d.cat)
    ctx = Codec(df, d.ctx_num, d.ctx_cat)
    fam = families(df, d, codec, ctx)
    tr, ho = split(len(fam), rng)
    train, hold = subset(fam, tr), subset(fam, ho)
    widths = codec.widths
    report = {"dataset": a.dataset, "seed": a.seed, "families_train": len(train), "families_hold": len(hold),
              "rows_train": int(len(train.X)), "mean_size": float(train.sizes.mean()), "max_size": int(train.sizes.max()),
              "variants": {}}
    gaps = [train.gap[train.off[f]:train.off[f + 1]] for f in range(len(train))] if d.order else None
    for v in a.variants.split(","):
        t0 = time.time()
        rng = np.random.default_rng([a.seed, zlib.crc32(v.encode())])   # per variant: results do not depend on the list
        if v == "shuffle":
            p = rng.permutation(len(train.X))
            sX, sC, soff = train.X[p], train.C[p], train.off
            info = {}
        else:
            enc, pri = v.split(":")
            cfg = Config(encoder=enc, prior=pri, ordered=d.order is not None,
                         vae_steps=a.vae_steps, prior_steps=a.prior_steps)
            model = ContextVAE(train.X.shape[1], widths, train.ctx_X.shape[1], ctx.widths, cfg, a.seed)
            model.fit_vae(train, rng).fit_prior(train, rng)
            info = dict(model.log)
            info["prior_nll_train"] = model.prior_nll(train, np.random.default_rng(0))
            info["prior_nll_hold"] = model.prior_nll(hold, np.random.default_rng(0))
            if not d.order and pri != "none":           # order sensitivity: two random orders of held-out families
                info["order_gap"] = abs(model.prior_nll(hold, np.random.default_rng(1)) - info["prior_nll_hold"])
            info["config"] = asdict(cfg)
            sX, sC, soff = model.sample(train.sizes, train.ctx_X, train.ctx_C, gaps, a.seed + 1000)
            sX = codec.snap(sX)                         # the engine decodes through the marginals
        res = evaluate(train, sX, sC, soff, widths, d.order is not None, train.ctx_C, hold, a.seed)
        res.update(info)
        res["seconds"] = round(time.time() - t0, 1)
        report["variants"][v] = res
        print(f"[{a.dataset} s{a.seed}] {v}: cat-share err {res['err_cat_share']}, num-within err {res['err_num_within']}, "
              f"c2st {res['c2st_agg']:.3f} row {res['c2st_row']:.3f}, ks {res['marginals']['ks_mean']}, "
              f"tvd {res['marginals']['tvd_mean']}, lag1 {res.get('err_lag1')}, trans {res.get('trans_tvd')}, "
              f"dcr fam {res['dcr_family']:.3f} row {res['dcr_row']:.3f}, {res['seconds']} s",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
