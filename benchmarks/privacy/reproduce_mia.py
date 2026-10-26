"""Sanity checks of the membership attacks against published findings.

1. Raw data release (the generator returns its training rows): every target
   is detectable, AUC close to 1 (Stadler et al. 2022, 'raw' baseline).
2. A generator that ignores individuals (independent marginals of the
   population, not of the training data): AUC close to 0.5 for every target
   (no information about membership exists).
3. relsynth: outliers chosen by the Meeus et al. (2023) score are more
   exposed than random records (Stadler et al. sec. 6; Meeus et al. Table 2).

    python benchmarks/privacy/reproduce_mia.py ADULT.csv SPEC.yaml OUT.json [--n-jobs 4]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import relsynth as rs
from relsynth.privacy import mia
from relsynth.privacy.assess import relsynth_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--shadows", type=int, default=32)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--targets", type=int, default=8)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    key = "pid"
    df = df.set_index(key, drop=False)
    rng = np.random.default_rng(0)
    ids = rng.permutation(df.index.to_numpy())
    ref_ids, pool = ids[:5000], ids[5000:25000]
    prof = df.drop(columns=[key])
    ref = prof.loc[ref_ids]
    vul = mia.vulnerability(pd.concat([ref, prof.loc[pool[:5000]]]))
    cand = vul.loc[pool[:5000]].sort_values(ascending=False)
    top = list(cand.index[:a.targets])
    rnd = list(rng.choice(cand.index[a.targets:], a.targets, replace=False))
    kinds = pd.Series({**{i: "vulnerable" for i in top}, **{i: "random" for i in rnd}})
    targets = prof.loc[top + rnd]
    popl = np.setdiff1d(pool, top + rnd)
    spec = rs.Spec.load(a.spec).resolved
    rel = relsynth_generator(spec)
    pop_marg = prof.loc[ref_ids]

    gens = {
        "raw_release": lambda ids_, sd: prof.loc[ids_].reset_index(drop=True),
        "population_marginals": lambda ids_, sd: pd.DataFrame(
            {c: pop_marg[c].sample(len(ids_), replace=True, random_state=sd + i).to_numpy()
             for i, c in enumerate(pop_marg.columns)}),
        "relsynth": lambda ids_, sd: rel({"adult": df.loc[ids_].reset_index(drop=True)}, sd)["adult"].drop(
            columns=[key]),
    }
    out = {}
    for name, g in gens.items():
        t = time.time()
        r = mia.shadow_attack(popl, targets, kinds, g, shadow_size=a.size, n_train=a.shadows, n_test=a.shadows,
                              seed=1, ref_profiles=ref, vuln=vul, n_jobs=a.n_jobs if name == "relsynth" else 1)
        d = r.to_dict()
        out[name] = {"pooled": d["pooled"], "seconds": time.time() - t,
                     "targets": [{k: v for k, v in x.items() if k in ("kind", "vulnerability", "classifier_auc",
                                                                       "lira_auc", "classifier_auc_ci")}
                                 for x in d["targets"]]}
        p = d["pooled"]
        print(name, {ro: (round(v["auc"], 3), round(v.get("mean_auc_vulnerable", np.nan), 3),
                          round(v.get("mean_auc_random", np.nan), 3)) for ro, v in p.items()},
              f"{time.time() - t:.0f}s", flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
