#!/usr/bin/env python3
"""Run the data model of spec.yaml exactly as `relsynth run` does (bit-identical output).

    python run_model.py [--out synthetic] [--datasets 1] [--scale 1.0]

Needs relsynth (the package this model was exported from) and access to the source
named in the specification (for the fit).  Every step is a public relsynth call:
load the source, fit the Bayesian model (latent segments by blocked Gibbs, then
the conjugate posteriors of every block), then for each dataset draw theta from
the posterior (parameter stream m) and the tables given theta (data stream m).
"""
import argparse
from pathlib import Path

from relsynth.cli import _dataset_target
from relsynth.data import coerce, load_tables, write_tables
from relsynth.engine import Synthesizer
from relsynth.spec import load_spec

SPEC = Path(__file__).with_name("spec.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output folder (default: the spec's target)")
    ap.add_argument("--datasets", type=int, default=None, help="number of synthetic datasets")
    ap.add_argument("--scale", type=float, default=None)
    a = ap.parse_args()
    spec = load_spec(str(SPEC))
    if a.scale is not None:
        spec["synthesis"]["scale"]["default"] = a.scale
    n = a.datasets or int(spec["synthesis"]["posterior"].get("datasets", 1))
    real = load_tables(spec)                          # the source (fit data)
    syn = Synthesizer(spec).fit(real)                 # posterior p(theta, z | data)
    for i in range(n):
        fake = syn.sample(index=i)                    # theta^(i) ~ posterior, D*^(i) ~ p(D* | theta^(i))
        fake = {t: coerce(df, spec["tables"][t]) for t, df in fake.items()}
        write_tables(spec, fake, _dataset_target(spec, a.out, i))
        print(f"dataset {i + 1}: " + ", ".join(f"{t} {len(df):,}" for t, df in fake.items()))


if __name__ == "__main__":
    main()
