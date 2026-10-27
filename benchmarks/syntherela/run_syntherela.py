# SPDX-License-Identifier: Apache-2.0
"""relsynth on the SyntheRela benchmark databases (Hudovernik et al., TMLR 2026).

    python benchmarks/syntherela/run_syntherela.py --data ~/syntherela-data --download
    python benchmarks/syntherela/run_syntherela.py --data ~/syntherela-data --seeds 1 2 3

Downloads the databases from the SDV public demo bucket (the copies SyntheRela
starts from) into --data, which must be outside the repository: the data and
the synthetic data derived from it are never committed (see the dataset
register; IMDb MovieLens is non-commercial).  For each database and seed it
fits relsynth with default settings, draws one synthetic database of the same
size and writes the SyntheRela-compatible summaries of the 2.0 quality report
(C2ST-Agg accuracy with XGBoost, cardinality similarity, k-hop trends) and the
quality vector to --out.  Only aggregate numbers leave the machine.

Database preparation follows SyntheRela's Table 5: the derived composite
identifiers (`a_id_b_id`) and SDV's `add_numerical` helper columns are dropped,
so the column counts match (Biodegradability 6, CORA 2, IMDB MovieLens 14).
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BUCKET = "https://sdv-datasets-public.s3.amazonaws.com/multi_table/{name}/{file}"
DATASETS = {"Biodegradability": "Biodegradability", "CORA": "CORA", "IMDB": "imdb_MovieLens"}
# categorical columns SDV stores as numbers but SyntheRela treats as categories
CATEGORICAL = {("content", "word_cited_id")}


def download(root: Path, name: str) -> Path:
    d = root / name
    if (d / "metadata.json").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    for f in ("metadata.json", "data.zip"):
        url = BUCKET.format(name=name, file=f)
        with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310  # nosec B310 - fixed https URL
            blob = r.read()
        if f.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for m in z.namelist():
                    if m.endswith(".csv"):
                        (d / Path(m).name).write_bytes(z.read(m))
        else:
            (d / f).write_bytes(blob)
    return d


def build(d: Path):
    """(spec dict, {table: DataFrame}) from an SDV metadata.json and its CSVs."""
    meta = json.loads((d / "metadata.json").read_text())
    rels = meta["relationships"]
    fks = {(r["child_table_name"], r["child_foreign_key"]) for r in rels}
    parents = {r["parent_table_name"] for r in rels}
    tables, spec_t = {}, {}
    for t, tm in meta["tables"].items():
        f = d / f"{t}.csv"
        if not f.exists():
            f = d / "data" / "data" / f"{t}.csv"
        df = pd.read_csv(f, low_memory=False)
        cols, keep = {}, []
        pk = tm.get("primary_key")
        n_fk = sum(1 for (c, _k) in fks if c == t)
        for c, cm in tm["columns"].items():
            st = cm["sdtype"]
            if c == "add_numerical":
                continue
            if st == "id":
                if c == pk and t not in parents and n_fk >= 2:
                    continue                    # derived composite id of a link table
                cols[c] = "id"
            elif st == "categorical" or (t, c) in CATEGORICAL:
                cols[c] = "categorical"
            elif st == "numerical":
                cols[c] = "integer" if cm.get("computer_representation", "").startswith("Int") else "float"
            elif st == "datetime":
                cols[c] = "datetime"
            elif st == "boolean":
                cols[c] = "boolean"
            else:
                cols[c] = "categorical"
            keep.append(c)
        tables[t] = df[keep]
        ts = {"columns": cols}
        if pk in cols:
            ts["primary_key"] = pk
        if t not in parents and n_fk >= 1 and pk not in cols:
            ts["role"] = "fact"
        spec_t[t] = ts
    spec = {"name": d.name, "source": {"kind": "csv", "path": str(d)}, "tables": spec_t,
            "relationships": [{"child": r["child_table_name"], "child_key": r["child_foreign_key"],
                               "parent": r["parent_table_name"]} for r in rels]}
    return spec, tables


def run_one(name, d, seed, classifier, splits):
    import relsynth as rs
    from relsynth.data import coerce
    from relsynth.engine import Synthesizer
    from relsynth.evaluate import evaluate
    from relsynth.quality import assess
    raw, frames = build(d)
    spec = rs.Spec(raw).resolved
    spec["synthesis"]["seed"] = int(seed)
    real = {t: coerce(df, spec["tables"][t]) for t, df in frames.items()}
    t0 = time.time()
    syn_ = Synthesizer(spec).fit(real)
    fit_s = time.time() - t0
    syn = syn_.sample(scale=spec["synthesis"]["scale"], index=0)
    syn = {t: coerce(df, spec["tables"][t]) for t, df in syn.items()}
    sample_s = time.time() - t0 - fit_s
    base = evaluate(spec, real, syn, privacy=False, sample_log=syn_.sample_log)
    rep = assess(spec, real, syn, base=base, config={"classifier": classifier, "null_splits": splits,
                                                     "multi_splits": splits, "seed": int(seed)})
    sr = rep["benchmark"]["syntherela"]
    hops = sr["k_hop_trend"]
    return {"database": name, "seed": seed, "classifier": classifier,
            "rows_real": int(sum(len(v) for v in real.values())), "rows_syn": int(sum(len(v) for v in syn.values())),
            "c2st_agg_max": sr["c2st_agg_accuracy_max"], "c2st_agg": sr["c2st_agg_accuracy"],
            "cardinality_mean": (sum(sr["cardinality_similarity"].values()) / len(sr["cardinality_similarity"])
                                 if sr["cardinality_similarity"] else None),
            "cardinality": sr["cardinality_similarity"], "hop1": hops.get("1-hop"), "hop2": hops.get("2-hop"),
            "vector": {k: {x: rep["vector"][k][x] for x in ("checks", "failed", "allowed_failures", "pass")}
                       for k in ("fidelity", "utility", "integrity")},
            "scorecard_1": rep["scorecard_1"], "fit_seconds": round(fit_s, 1), "sample_seconds": round(sample_s, 1),
            "failed_checks": [c["id"] for c in rep["checks"] if not c["pass"]]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, type=Path, help="folder for the downloaded databases (not in the repo)")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--classifier", default="xgboost", choices=["xgboost", "logistic"])
    ap.add_argument("--splits", type=int, default=10, help="real-vs-real splits for the null limits")
    ap.add_argument("--out", type=Path, default=Path("syntherela_results.json"))
    a = ap.parse_args(argv)
    results = json.loads(a.out.read_text()) if a.out.exists() else []
    done = {(r["database"], r["seed"], r["classifier"]) for r in results}
    for name in a.datasets:
        d = download(a.data, DATASETS[name]) if a.download else a.data / DATASETS[name]
        for seed in a.seeds:
            if (name, seed, a.classifier) in done:
                continue
            r = run_one(name, d, seed, a.classifier, a.splits)
            print(json.dumps({k: r[k] for k in ("database", "seed", "c2st_agg_max", "cardinality_mean", "hop1", "hop2")}),
                  flush=True)
            results.append(r)
            a.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
