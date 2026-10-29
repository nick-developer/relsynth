# relsynth on the SyntheRela benchmark

SyntheRela (Hudovernik, Jurkovič and Štrumbelj, *Benchmarking the Fidelity and Utility of Synthetic Relational
Data*, arXiv:2410.03411, TMLR) is the reference benchmark for multi-table synthetic data. This folder runs
relsynth 2.0 with **default settings and an unedited spec** (built from the SDV metadata) on the SyntheRela
databases that the project's dataset register allows, and compares the results with published numbers.

## Results

relsynth: mean ± sd over seeds 1–3, one synthetic database of the real size per seed, XGBoost detector.
Published: Table 1 of RelDiff (Hudovernik et al., arXiv:2506.00710), which reports SyntheRela metrics for
TabARGN, ClavaDDPM, RCTGAN, REaLTabFormer, SDV and RelDiff. All values ×100. C2ST-Agg is the accuracy of
the best discriminator over the parent tables (50 = indistinguishable, **lower is better**); the others are
similarities (**higher is better**). "–" means the method was not run on that database; "DNC" did not complete.
The rank counts the published methods with a number plus relsynth.

### Biodegradability (3 seeds, 21,895 rows)

| Metric | relsynth | TabARGN | ClavaDDPM | RCTGAN | REaLTabFormer | SDV | RelDiff | relsynth rank |
|---|---|---|---|---|---|---|---|---|
| C2ST-Agg accuracy (lower is better) | **87.83 ± 0.38** | 88.86 | – | 83.82 | – | 98.02 | 47.04 | 3 of 5 |
| Cardinality similarity | **88.72 ± 0.43** | 79.53 | – | 85.22 | – | 61.17 | 100.00 | 2 of 5 |
| 1-hop trends | **94.79 ± 0.81** | 61.36 | – | 75.80 | – | 49.09 | 89.37 | 1 of 5 |
| 2-hop trends | **87.98 ± 0.79** | 60.54 | – | 77.04 | – | 47.80 | 86.59 | 1 of 5 |

### CORA (3 seeds, 57,353 rows)

| Metric | relsynth | TabARGN | ClavaDDPM | RCTGAN | REaLTabFormer | SDV | RelDiff | relsynth rank |
|---|---|---|---|---|---|---|---|---|
| C2ST-Agg accuracy (lower is better) | **73.13 ± 0.46** | 68.80 | – | 73.74 | – | 99.59 | 69.30 | 3 of 5 |
| Cardinality similarity | **92.54 ± 0.94** | 96.27 | – | 90.48 | – | 68.82 | 100.00 | 3 of 5 |
| 1-hop trends | **61.50 ± 0.17** | 80.42 | – | 68.39 | – | 4.95 | 72.16 | 4 of 5 |

### IMDB (3 seeds, 1,249,411 rows)

| Metric | relsynth | TabARGN | ClavaDDPM | RCTGAN | REaLTabFormer | SDV | RelDiff | relsynth rank |
|---|---|---|---|---|---|---|---|---|
| C2ST-Agg accuracy (lower is better) | **90.41 ± 0.33** | 73.76 | 65.00 | 81.56 | – | – | 53.29 | 5 of 5 |
| Cardinality similarity | **84.51 ± 0.12** | 81.19 | 98.95 | 79.53 | – | DNC | 100.00 | 3 of 5 |
| 1-hop trends | **95.52 ± 0.09** | 88.64 | 91.57 | 81.76 | – | – | 94.84 | 1 of 5 |

### What the table says

- **Column-pair trends across joins: relsynth is first on Biodegradability (1-hop and 2-hop) and IMDB
  (1-hop), and fourth of five on CORA (61.5 against 68–80).** Its latent segments carry the dependence
  between a parent's attributes and its children's. CORA's only attribute pairs run through the citation
  self-links, whose two keys relsynth does not tie to each other's topic.
- **Cardinality: second on Biodegradability, third on CORA and IMDB.** RelDiff (100 by construction, it
  generates the graph structure first) and ClavaDDPM on IMDB are better. relsynth's weak relationships are
  the second foreign key of a link table to the same parent (`bond.atom_id2`, 0.68) and the many-to-many
  link tables of IMDB (`movies2actors`, `movies2directors`, 0.76–0.80): the second key of a link table is
  drawn from the segment law, not from a degree model of its own.
- **C2ST-Agg: third on Biodegradability and CORA, last on IMDB (90.4 against 53–82).** The discriminator
  separates IMDB movies by their aggregated children (counts of actors, directors and ratings per movie and
  the number of distinct values among them) — the same link-table degree weakness. On CORA relsynth is
  close to RCTGAN and TabARGN; on Biodegradability it is between RCTGAN and TabARGN.
- **relsynth is not the best generator on this benchmark overall.** RelDiff is better on C2ST-Agg and
  cardinality on all three databases, and ClavaDDPM on IMDB. relsynth's strength is cross-table correlation at small
  cost: fit plus sample takes 6–13 s on Biodegradability and CORA and 65–100 s on IMDB (1.25 M rows) on 4 CPU
  cores, with no GPU.

The 2.0 quality vector (computed at commit 17bb29d; later commits change only area rules and resized comparisons, not the four summary scores above) agrees: integrity passes on every run, and the fidelity area fails on all three
databases (Biodegradability 11–18 of 105 checks, CORA 6–7 of 39, IMDB 22–24 of 154). The failed checks are
mostly fan-out laws and aggregation detection on the relationships named above. A degree model for the
second key of link tables is the engine change these results point to; no roadmap thread owns it yet.

## Caveats

- **Re-implemented metrics.** The scores are computed by `relsynth.quality` (cardinality = 1 − KS of
  per-parent counts; k-hop trends = mean SDMetrics CorrelationSimilarity / ContingencySimilarity with
  10 equal-width bins; C2ST-Agg = SyntheRela's aggregation features, balanced sample, 5-fold, XGBoost
  defaults, maximum over parents). The SyntheRela package itself was not run: its pinned dependencies do not
  install on Python 3.13. Small differences in sampling and binning are possible.
- **Published numbers were read from the PDFs** of arXiv:2506.00710 (and checked against the DDA accuracies
  of arXiv:2505.15668: Biodegradability RCTGAN 0.88 and SDV 0.98; CORA RCTGAN 0.73; IMDB RCTGAN 0.95 and
  ClavaDDPM 0.83, which differ for IMDB). Verify them against the papers before citing. They are stored in
  [published.json](published.json).
- **Seeds.** Three seeds; the published numbers are means over the papers' own runs.
- **Databases left out.** Berka: the SDV copy has 1.08 M rows against 757,722 in SyntheRela's Table 5, so its
  numbers are not comparable. AirBnB, Rossmann and Walmart: Kaggle competition data, which the dataset
  register does not allow. Row counts of the three databases run here match Table 5 exactly
  (Biodegradability 21,895; CORA 57,353; IMDB MovieLens 1,249,411).

## Licences and data

The databases are downloaded from the SDV public demo bucket to a folder **outside the repository**. No real
or synthetic rows are committed. IMDB MovieLens is for non-commercial use, so it is benchmarked locally
only and only aggregate numbers are reported here, as the dataset register requires.

## Running it

```bash
pip install "relsynth[benchmark]"                      # xgboost
python benchmarks/syntherela/run_syntherela.py --data ~/syntherela-data --download --seeds 1 2 3 \
    --out syntherela_results.json
python benchmarks/syntherela/summarise.py syntherela_results.json
```

`--splits` sets the real-vs-real splits of the calibrated limits (default 10). `--classifier logistic` avoids the xgboost dependency but is not comparable with the
published numbers.
