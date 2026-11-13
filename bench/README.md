# bench

The benchmark harness: `python -m bench all` runs every suite and compares the
results with `baseline.json`. The full description, the regression rule and how
to change the baseline are in [docs/development/benchmarks.md](../docs/development/benchmarks.md).

| Path | Content |
|---|---|
| `__main__.py` | the command line |
| `datasets.py` | the golden databases: pinned downloads, builders, spec and checks of each |
| `runner.py` | runs relsynth per dataset and seed, writes a history record |
| `assess.py` | the 2.0 quality report and the privacy attacks of one run, as checks |
| `regress.py` | baseline and regression rule (R1–R3) |
| `specs/` | the frozen spec of each database |
| `builders/`, `checks/` | the build scripts and domain checks: the layer-2 ones kept as they were written, the G7 builders (`sdv_sqlite.py`, `stress_schema.py`, `tpc_parquet.py`, ...) and the schema-level checks X1–X6 (`checks/generic.py`) |
| `history/` | history records (aggregates only), committed with each baseline change |
| `baseline.json` | the current baseline |
| `.cache/` | downloads and built databases (not committed) |

## The benchmark matrix (G7)

The datasets cover as many categories, sizes and schema shapes as the
[dataset register](../docs/datasets.md) allows, in four tiers. A tier runs only
its own datasets (`python -m bench run --tier weekly`); each dataset runs on its
own seeds unless `--seeds` is given. The 2.0 golden databases keep their frozen
specs; every dataset added by G7 runs with the unedited `relsynth introspect`
draft of the code under test (made once per commit into the cache; its SHA-256
is in the history record), and with the schema-level checks X1–X6 of
`checks/generic.py` (primary keys unique, links resolve, no cycles in
self-references: identities; fan-out skew, row ratio, NULL shares: recorded).
Data that cannot be fetched by checksum is generated (TPC-H/DS by DuckDB's
dbgen/dsdgen, checked by row counts; Synthea by the pinned release jar; the
stress schema by `builders/stress_schema.py` from a seed).

### Tier `pr`: every pull request (ci.yml `bench`, `--jobs 4`), under 15 minutes in all

| Dataset | What | Seeds | Runner |
|---|---|---|---|
| `chinook` | Chinook 1.4.5 (11 tables); spec = examples/chinook_model/spec.yaml | 1, 2, 3, 4, 5 | GitHub-hosted |
| `adult` | UCI Adult (32,561 people, 15 columns; SDV copy, CC BY 4.0) | 1 | GitHub-hosted |
| `stress_s` | schema stress generator, small (27 tables, depth 4, 60-column table, 26k rows) | 1 | GitHub-hosted |
| `mongodb_samples` | MongoDB sample datasets (supplies, analytics, geospatial, training) through relsynth.documents | 1 | GitHub-hosted |

### Tier `weekly`: Saturdays (bench.yml), one job per dataset within a hosted runner's 6 hours

| Dataset | What | Seeds | Runner |
|---|---|---|---|
| `adventureworks` | AdventureWorks DW 2022 as SQLite (27 tables, 1.05M rows); checks G1-G14 | 1, 2, 3 | GitHub-hosted |
| `contoso_parquet` | Contoso V2 100k, raw Parquet files; checks T1-T17 | 1, 2, 3, 4, 5 | GitHub-hosted |
| `berka` | Berka / PKDD'99 Czech bank (SyntheRela 'financial', 8 tables, 1.08M rows); checks F1-F16 | 1, 2, 3, 4, 5 | GitHub-hosted |
| `acs` | ACS PUMS 2019, NIST CRC national excerpt (27,253 people, 24 columns); checks A1-A14 | 1, 2, 3, 4, 5 | GitHub-hosted |
| `covtype` | UCI CoverType (581,012 rows, 55 columns: 44 binary indicators; SDV copy, CC BY 4.0) | 1, 2, 3 | GitHub-hosted |
| `census_kdd` | UCI Census-Income KDD (299,285 people, 42 columns; SDV copy, CC BY 4.0) | 1, 2, 3 | GitHub-hosted |
| `synthea_1k` | Synthea 1K sample (18 tables, 1.84M rows, 1,163 patients; SDV copy, Apache-2.0) | 1 | GitHub-hosted |
| `mimic_demo` | MIMIC-IV Clinical Database Demo 2.2 (100 patients, 31 tables; ODbL 1.0) | 1, 2, 3 | GitHub-hosted |
| `online_retail_ii` | UCI Online Retail II (1.07M invoice lines as a 4-table star; CC BY 4.0) | 1, 2, 3 | GitHub-hosted |
| `stackexchange` | coffee.stackexchange.com data dump (7 tables; CC BY-SA 4.0) | 1, 2, 3 | GitHub-hosted |
| `rel_trial` | RelBench rel-trial (15 tables, about 5.5M rows; ClinicalTrials.gov via AACT) | 1 | GitHub-hosted |
| `electricity` | UCI ElectricityLoadDiagrams20112014, last year hourly (321 meters, 2.8M rows); T12 spec | 1 | GitHub-hosted |
| `nyc_tlc_1m` | NYC TLC yellow taxi trips 2024-01 (about 3M trips) and the zone lookup | 1 | GitHub-hosted |
| `tpch_sf1` | TPC-H data generator at SF1 (8 tables, 8.7M rows; not TPC-H results) | 1 | GitHub-hosted |
| `tpcds_sf1` | TPC-DS data generator at SF1 (24 tables; not TPC-DS results) | 1 | GitHub-hosted |
| `stress_m` | schema stress generator, medium (120 tables, depth 8, 500-column table, 340k rows) | 1 | GitHub-hosted |

### Tier `monthly`: the 2nd of each month (bench.yml); the sets marked for a bigger runner run with `python -m bench all --tier monthly` on the maintainer's machine

| Dataset | What | Seeds | Runner |
|---|---|---|---|
| `contoso_sqlite` | Contoso V2 100k as SQLite with the generator's keys; checks T1-T17 | 1 | GitHub-hosted |
| `synthea_100k` | Synthea 3.4.0, 100,000 generated patients (Massachusetts, seed 7) | 1 | bigger runner or maintainer machine |
| `synthea_1m` | Synthea 3.4.0, 1,000,000 generated patients (Massachusetts, seed 7) | 1 | bigger runner or maintainer machine |
| `nyc_tlc_12m` | NYC TLC yellow taxi trips 2024 (12 months, about 41M trips) | 1 | GitHub-hosted |
| `open_food_facts` | Open Food Facts, 200,000-product extract of the 8 GB export (ODbL) | 1 | GitHub-hosted |
| `tpch_sf10` | TPC-H data generator at SF10 (87M rows; not TPC-H results) | 1 | GitHub-hosted |
| `tpch_sf100` | TPC-H data generator at SF100 (870M rows; not TPC-H results) | 1 | bigger runner or maintainer machine |
| `tpcds_sf10` | TPC-DS data generator at SF10 (24 tables; not TPC-DS results) | 1 | GitHub-hosted |
| `stress_l` | schema stress generator, large (300 tables, depth 8, 500-column table, 5M fact rows) | 1 | GitHub-hosted |

### Tier `local`: only where the data is configured (private or research-only terms); aggregates only

| Dataset | What | Seeds | Runner |
|---|---|---|---|
| `bi_corep_lr` | private; RELSYNTH_BENCH_BI_DIR with data/, spec.yaml and checks.py (B1-B8) | 1, 2, 3, 4, 5 | local only |
| `imdb_movielens` | research only: IMDb x MovieLens (CTU copy via SDV; non-commercial), aggregates only | 1 | local only |

### Coverage list

Each feature of the roadmap's list is exercised by at least two datasets
(`python -m bench coverage --check`, also a unit test); features without the
two-set requirement are marked extra in `bench/datasets.py` (`REQUIRED_FEATURES`).

| Feature | Datasets (tier) | Sets |
|---|---|---|
| one table (microdata) (`single_table`) | acs (weekly), adult (pr), covtype (weekly), census_kdd (weekly) | 4 |
| 10 or more tables (`tables_10_plus`) | adventureworks (weekly), synthea_1k (weekly), synthea_100k (monthly), synthea_1m (monthly), mimic_demo (weekly), rel_trial (weekly), tpcds_sf1 (weekly), tpcds_sf10 (monthly), stress_s (pr) | 9 |
| 100 or more tables (`tables_100_plus`) | stress_m (weekly), stress_l (monthly) | 2 |
| KB to tens of MB (`size_mb`) | chinook (pr), adult (pr), stress_s (pr), mongodb_samples (pr) | 4 |
| about 100 MB to 1 GB (`size_100mb`) | contoso_parquet (weekly), berka (weekly), covtype (weekly), census_kdd (weekly), electricity (weekly), nyc_tlc_1m (weekly) | 6 |
| 1 GB to 10 GB (`size_gb`) | synthea_1k (weekly), rel_trial (weekly), nyc_tlc_12m (monthly), tpch_sf1 (weekly), tpcds_sf1 (weekly), stress_l (monthly) | 6 |
| 10 GB to 100 GB (`size_10gb`) | synthea_100k (monthly), tpch_sf10 (monthly), tpcds_sf10 (monthly) | 3 |
| 100 GB and more (`size_100gb`) | synthea_1m (monthly), tpch_sf100 (monthly) | 2 |
| snowflake chains of depth 3 (`snowflake_depth_3`) | adventureworks (weekly), tpch_sf1 (weekly), tpcds_sf1 (weekly) | 3 |
| snowflake chains of depth 4 or more (`snowflake_depth_4`) | stress_s (pr), stress_m (weekly), stress_l (monthly) | 3 |
| self-references (trees) (`self_reference`) | chinook (pr), adventureworks (weekly), stackexchange (weekly), stress_s (pr), stress_m (weekly) | 5 |
| a cycle of links between tables (`cyclic_keys`) | stress_s (pr), stress_m (weekly), stress_l (monthly) | 3 |
| many-to-many bridges (`many_to_many`) | chinook (pr), berka (weekly), rel_trial (weekly), stress_s (pr), stress_m (weekly), imdb_movielens (local) | 6 |
| composite primary and foreign keys (`composite_keys`) | mimic_demo (weekly), tpch_sf1 (weekly), tpch_sf10 (monthly), tpch_sf100 (monthly), tpcds_sf1 (weekly), tpcds_sf10 (monthly), stress_s (pr), stress_m (weekly) | 8 |
| slowly changing dimensions (type 2) (`scd`) | adventureworks (weekly), tpcds_sf1 (weekly), tpcds_sf10 (monthly) | 3 |
| dated events (facts with a time column) (`events`) | chinook (pr), adventureworks (weekly), contoso_sqlite (monthly), contoso_parquet (weekly), berka (weekly), synthea_1k (weekly), synthea_100k (monthly), synthea_1m (monthly), mimic_demo (weekly), online_retail_ii (weekly), rel_trial (weekly), nyc_tlc_1m (weekly), nyc_tlc_12m (monthly) | 13 |
| regular time series (one value per entity and step) (`regular_series`) | electricity (weekly), stress_s (pr), stress_m (weekly) | 3 |
| long free text (`long_text`) | stackexchange (weekly), rel_trial (weekly), open_food_facts (monthly) | 3 |
| nested documents and lists (`nested_documents`) | open_food_facts (monthly), mongodb_samples (pr) | 2 |
| wide tables (50 or more columns) (`wide_tables`) | covtype (weekly), open_food_facts (monthly), stress_m (weekly), stress_l (monthly) | 4 |
| binary columns (BLOB) (`binary_columns`) | stress_s (pr), stress_m (weekly), stress_l (monthly) | 3 |
| high-cardinality categoricals (`high_cardinality`) | contoso_sqlite (monthly), contoso_parquet (weekly), acs (weekly), bi_corep_lr (local), census_kdd (weekly), synthea_1k (weekly), online_retail_ii (weekly), open_food_facts (monthly), mongodb_samples (pr) | 9 |
| skew and imbalance (fan-out, classes) (`skew`) | berka (weekly), bi_corep_lr (local), adult (pr), covtype (weekly), census_kdd (weekly), online_retail_ii (weekly), stackexchange (weekly), nyc_tlc_1m (weekly), nyc_tlc_12m (monthly), stress_s (pr), stress_m (weekly), stress_l (monthly), imdb_movielens (local) | 13 |
| messy values (codes for missing, negatives, outliers, stray text) (`messy_values`) | acs (weekly), adult (pr), census_kdd (weekly), online_retail_ii (weekly), stackexchange (weekly), nyc_tlc_1m (weekly), nyc_tlc_12m (monthly), open_food_facts (monthly), stress_s (pr), stress_m (weekly) | 10 |
| small privacy units (under 200 people) (`small_privacy_units`) | chinook (pr), mimic_demo (weekly) | 2 |

## History record format (schema 1)

```json
{"schema": 1, "relsynth": "1.0.0", "git": {"sha": "...", "branch": "...", "dirty": false},
 "machine": {"node": "...", "platform": "...", "cpu": "...", "cpus": 4, "python": "3.13.0",
             "numpy": "...", "pandas": "...", "scipy": "...", "ci": false},
 "started": "...", "finished": "...", "label": "...", "seeds": [1, 2, 3, 4, 5],
 "suites": ["scorecard", "quality", "privacy", "repro"],
 "runs": [{"dataset": "chinook", "seed": 1, "seconds": 25.6, "rows": {...}, "scorecard": {"passed": 141, "total": 142},
           "cost": {"cpu_seconds": 40.2, "peak_mib": 339.4}, "quality": {"seconds": 55.8, ...},
           "checks": [{"id": "scorecard:<area>|<metric>", "pass": true, "value": 0.03, "limit": 0.1},
                      {"id": "domain:<check>", "pass": null, "value": 2.5, "real": 2.4},
                      {"id": "quality:<check id>", "pass": true, "value": 0.01, "limit": 0.05, "alpha": 0.01},
                      {"id": "area:fidelity", "pass": true, "value": 0.98, "limit": 0},
                      {"id": "goal:referential_integrity", "pass": true, "value": 1.0, "real": 1.0},
                      {"id": "privacy:<criterion>|<attack>", "pass": true, "value": 0.0, "limit": 0.1, "upper": 0.01,
                       "status": "pass"},
                      {"id": "metric:<name>", "pass": null, "value": 0.91, "direction": "higher"}]},
          {"dataset": "bi_corep_lr", "skipped": "RELSYNTH_BENCH_BI_DIR is not set ..."},
          {"dataset": "fake", "seed": null, "checks": [{"id": "science:C00", "pass": true}]}]}
```
