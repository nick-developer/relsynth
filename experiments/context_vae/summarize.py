# SPDX-License-Identifier: Apache-2.0
"""Summarise context-VAE reports (JSON files) as a Markdown table, mean over seeds.

    python -m experiments.context_vae.summarize out/*.json >> "$GITHUB_STEP_SUMMARY"
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

KEYS = [("err_cat_share", "cat share err"), ("err_num_within", "num within err"), ("err_lag1", "lag-1 err"),
        ("trans_tvd", "transition TVD"), ("c2st_agg", "C2ST-Agg acc"), ("c2st_row", "C2ST row acc"), ("dcr_family", "DCR family"),
        ("dcr_row", "DCR row"), ("active_units", "active units"), ("seconds", "seconds")]


def main(paths):
    acc = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    for p in paths:
        r = json.load(open(p))
        seeds[r["dataset"]].add(r["seed"])
        for v, res in r["variants"].items():
            for k, _ in KEYS:
                x = res.get(k)
                if isinstance(x, (int, float)):
                    acc[(r["dataset"], v)][k].append(float(x))
            m = res.get("marginals", {})
            for k in ("ks_mean", "tvd_mean"):
                if isinstance(m.get(k), (int, float)):
                    acc[(r["dataset"], v)][k].append(float(m[k]))
    cols = KEYS + [("ks_mean", "KS mean"), ("tvd_mean", "TVD mean")]
    print("| dataset | variant | " + " | ".join(n for _, n in cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for (ds, v), d in sorted(acc.items()):
        cells = []
        for k, _ in cols:
            x = d.get(k)
            cells.append("" if not x else (f"{np.mean(x):.3f}" + (f" ± {np.std(x):.3f}" if len(x) > 1 else "")))
        print(f"| {ds} | {v} | " + " | ".join(cells) + " |")
    print()
    print("Seeds: " + ", ".join(f"{ds} {sorted(s)}" for ds, s in sorted(seeds.items())) +
          ". Lower is better for errors and TVD; C2ST-Agg accuracy and DCR shares are best at 0.5.")


if __name__ == "__main__":
    main(sys.argv[1:])
