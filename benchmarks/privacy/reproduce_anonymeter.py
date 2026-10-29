# SPDX-License-Identifier: Apache-2.0
"""Reproduce the published Anonymeter results on Adult (Giomi et al. 2023,
example notebook of github.com/statice/anonymeter) with relsynth's attacks.

Data: adults_train.csv, adults_control.csv and adults_syn_ctgan.csv from the
Anonymeter repository (notebooks/datasets; Adult is CC BY 4.0).  They are not
copied into relsynth: pass the folder that holds them.

    python benchmarks/privacy/reproduce_anonymeter.py DATA_DIR OUT.json [--seeds 5] [--oracle]

--oracle also runs the anonymeter package itself on the same data (optional
dependency) so both implementations are compared seed by seed.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from relsynth.privacy import criteria as K

# published values: anonymeter notebook outputs (one run each; they vary per run)
PUBLISHED = {
    "singling_out_univariate": {"value": 0.0137, "ci": [0.0, 0.0341]},
    "singling_out_multivariate": {"value": 0.0288, "ci": [0.0, 0.1380]},
    "linkability_k10": {"value": 0.0, "ci": [0.0, 0.0019], "attack_rate": 0.0035, "control_rate": 0.0054},
}
AUX = [["type_employer", "education", "hr_per_week", "capital_loss", "capital_gain"],
       ["race", "sex", "fnlwgt", "age", "country"]]


def ours(ori, syn, ctl, seed):
    out = {}
    t = time.time()
    r = K.record_isolation(ori, syn, ctl, mode="univariate", n_attacks=500, seed=seed, size_correction="anonymeter")
    out["singling_out_univariate"] = {"value": r.risk.value, "ci": list(r.risk.ci), "s": time.time() - t}
    t = time.time()
    r = K.record_isolation(ori, syn, ctl, mode="multivariate", n_attacks=100, n_cols=4, seed=seed,
                           size_correction="anonymeter")
    out["singling_out_multivariate"] = {"value": r.risk.value, "ci": list(r.risk.ci), "s": time.time() - t}
    t = time.time()
    r = K.linkage(ori, syn, ctl, aux_cols=AUX, n_attacks=2000, n_neighbors=10, seed=seed)
    out["linkability_k10"] = {"value": r.risk.value, "ci": list(r.risk.ci), "attack_rate": r.risk.attack.value,
                              "control_rate": r.risk.control.value, "s": time.time() - t}
    return out


def oracle(ori, syn, ctl, seed):
    from anonymeter.evaluators import LinkabilityEvaluator, SinglingOutEvaluator
    out = {}
    t = time.time()
    e = SinglingOutEvaluator(ori=ori, syn=syn, control=ctl, n_attacks=500, seed=seed).evaluate(mode="univariate")
    r = e.risk()
    out["singling_out_univariate"] = {"value": r.value, "ci": list(r.ci), "s": time.time() - t}
    t = time.time()
    e = SinglingOutEvaluator(ori=ori, syn=syn, control=ctl, n_attacks=100, n_cols=4, seed=seed).evaluate(
        mode="multivariate")
    r = e.risk()
    out["singling_out_multivariate"] = {"value": r.value, "ci": list(r.ci), "s": time.time() - t}
    t = time.time()
    np.random.seed(seed)
    e = LinkabilityEvaluator(ori=ori, syn=syn, control=ctl, n_attacks=2000, aux_cols=AUX, n_neighbors=10)
    e.evaluate(n_jobs=-1)
    res = e.results()
    r = res.risk()
    out["linkability_k10"] = {"value": r.value, "ci": list(r.ci), "attack_rate": res.attack_rate.value,
                              "control_rate": res.control_rate.value, "s": time.time() - t}
    return out


def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--oracle", action="store_true")
    a = ap.parse_args()
    ori = pd.read_csv(f"{a.data}/adults_train.csv")
    syn = pd.read_csv(f"{a.data}/adults_syn_ctgan.csv")
    ctl = pd.read_csv(f"{a.data}/adults_control.csv")
    runs = []
    for s in range(a.seeds):
        row = {"seed": s, "relsynth": ours(ori, syn, ctl, s)}
        if a.oracle:
            row["anonymeter"] = oracle(ori, syn, ctl, s)
        runs.append(row)
        print(json.dumps(row), flush=True)
    summary = {}
    for k, pub in PUBLISHED.items():
        ov = [overlap(r["relsynth"][k]["ci"], pub["ci"]) for r in runs]
        summary[k] = {"published": pub, "relsynth_mean": float(np.mean([r["relsynth"][k]["value"] for r in runs])),
                      "relsynth_ci_overlaps_published": f"{sum(ov)}/{len(ov)}"}
        if a.oracle:
            summary[k]["anonymeter_mean"] = float(np.mean([r["anonymeter"][k]["value"] for r in runs]))
            summary[k]["relsynth_ci_overlaps_anonymeter_same_seed"] = \
                f"{sum(overlap(r['relsynth'][k]['ci'], r['anonymeter'][k]['ci']) for r in runs)}/{len(runs)}"
    json.dump({"summary": summary, "runs": runs}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
