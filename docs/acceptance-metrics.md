# Roadmap acceptance metrics

Every acceptance criterion of the relsynth roadmap (`future_versions_roadmap.md`, its north-star metrics and the
thread criteria of the implementation plan) is defined here as a measurement: the metric and how relsynth computes
it, the datasets, the baseline it is compared with, the confidence level and the thread that owns it. A criterion
that is not on this page is not yet measurable and must be added before its release.

Conventions used throughout:

- **Golden databases**: bi/COREP, Chinook, AdventureWorks DW and Contoso V2 (100k), each with its unedited
  `introspect` draft spec from `layer2/realdata`, unless a row says otherwise.
- **Seeds**: 5 seeds (1–5) unless stated. A result "with CI" is the mean over seeds with a 95% Student-t interval.
- **Report fields** are those of the 2.0 quality report (`relsynth.assess_quality`, schema
  `quality-report-2.0`); metric ids are defined in [metrics.md](metrics.md).
- **"Passes calibrated limits"** means the area of the quality vector passes its rule (Prop. 111) at
  `area_confidence = 0.99`, with checks at `confidence = 0.99` and the default tolerances.
- **"Unchanged in outcome"** means the same pass/fail for every 1.0 scorecard item and every 2.0 check, for the
  same spec and seed, and identical synthetic tables where the row says so.

## North-star and release metrics from the roadmap

| Roadmap metric (target) | Definition and computation | Datasets | Baseline | Confidence | Owner |
|---|---|---|---|---|---|
| `referential_integrity` (100%) | Share of non-null foreign-key values in the synthetic database that match a parent key, over all relationships, simple and composite: `fk_integrity` items of the report (integrity area). Every relationship must be 1.0. | Golden databases, fake suite S1–S11, SyntheRela databases | 1.0: keys are generated from parent keys, never sampled | Identity: no sampling error, no tolerance | T05 (metric), each generator thread |
| `business_rule_preservation` (>99%) | For every declared hard rule (`check`, `derive`, `aggregate`, `cumulative`): 1 − violation rate on synthetic rows (`rule_violation_rate`). Reported per rule; the criterion is the minimum over rules ≥ 0.99, and hard rules that relsynth enforces exactly must be 1.0. Rules the real data itself breaks are compared with the real rate. | Golden databases, S7–S11 | Real data's own violation rate | Identity for enforced rules; for soft rules the calibrated `soft_rule_rate_diff` | T05, T03 |
| `generation_failures` (<0.1%) | Share of runs (fit + sample + write) that end with an engine error, over the 5 seeds × golden databases × supported engines in the T01 harness; plus the share of generated rows the sampler removed (`rows_removed_share`, per table ≤ 1%). | T01 harness matrix | 0 failures | Exact count; with 0 failures in N runs the 95% upper bound is 3/N, so the harness needs N ≥ 3000 runs to claim <0.1% | T01 |
| `privacy_report` (mandatory) and `attack_success_rate` (below baseline) | Defined by T06: advantage of each attack over its baseline with a 95% CI. This page only fixes the form: an attack passes when the upper CI bound of its advantage is ≤ the stated margin. | Golden databases, Adult, ACS PUMS | Attack baselines of T06 | 95% CI | T06 |
| `utility_gap` / `utility_loss` (<5%) | TSTR gap `tstr_gap` = TRTR − TSTR on held-out real rows (real data split 70/30 by cluster before fitting; the model is fitted on the 70%), for each table with a declared target and for the multi-table task (`mt_tstr_gap`). Scores: weighted one-vs-rest AUC (classification) or R² on the rank-normal target (regression). Criterion: gap < 0.05 absolute in score units, and the calibrated check passes. Relative loss (gap / (TRTR − naive)) is reported alongside. | Golden databases (targets listed in `benchmarks/targets.yaml` when T01 adds it; default heuristic otherwise), SyntheRela databases with published targets | TRTR on the same split | Calibrated check at 0.99; mean over 5 seeds with CI | T05 |
| `distribution_match` (95%+) | Share of passed checks in the fidelity area of the quality vector (`vector.fidelity.pass_rate`), together with the area verdict. Criterion: pass_rate ≥ 0.95 **and** the area passes its calibrated rule. | Golden databases | Real-vs-real: by Prop. 109 a real split fails about 1% of checks | 0.99 per check, 0.99 per area | T05 |
| `structure_preservation` (>99%) and `schema_errors` (<0.5%) | Documents: share of synthetic documents that validate against the inferred JSON schema, and share of documents with a schema error. | MongoDB samples, StackExchange | 100% of real documents validate | Exact counts, 95% Clopper–Pearson bound | T13 |
| `privacy_risk`, `generation_cost` (decreasing), `dataset_coverage` (increasing) | Tracked per release by the T01 benchmark harness: risk = T06's maximum attack advantage; cost = CPU-seconds and peak memory per million generated rows on Contoso; coverage = number of schema shapes with a passing golden test. A release passes when the value is no worse than the previous release beyond the 5-seed spread. | As listed | Previous release | 5-seed spread | T01, T06 |
| Benchmark regression (">2% degradation" flag) | For each SyntheRela database and golden database: C2ST-Agg accuracy, cardinality similarity, 1-hop trend and the area pass rates. Flag when a value is worse than the previous release by more than max(0.02, 2 × the 3-seed sd). | SyntheRela databases, golden databases | Previous release | 3 seeds | T05, T01 |

## Thread criteria that need a metric

| Criterion (thread) | Definition | Datasets | Baseline | Confidence |
|---|---|---|---|---|
| "reproduces 1.0 numbers within the 5-seed spread" (T01) | For every 1.0 scorecard item value: the new value lies within [min, max] over seeds 1–5 of release 1.0, widened by 10% of the range. | Golden databases | Release 1.0 runs | Range of 5 seeds |
| "identical synthetic output for the same seed" (T02, T03, T04, T18) | Every synthetic table equal cell by cell after canonical row order (Prop. 101). | Golden databases, fake suite | 1.0 output | Identity |
| "same scorecard" / "scorecard equal in outcome" (T04, T10, T18) | Same pass/fail for every 1.0 scorecard item and every 2.0 check (Prop. 112). Values may differ only where the data path differs (sampled pushdown). | As listed | Reference run (SQLite, in memory, or 1 GB sample) | Outcome identity |
| "within the proved pushdown tolerance" (T04, T18) | Relative error of each in-database statistic below the bound of Prop. 102 (Acklam) and Prop. 100; peak memory reported in MiB. | Contoso Parquet, warehouses | In-memory statistics | Bound, not a test |
| "passes calibrated limits" (T12, T13, T15) | The thread's new checks are added to the 2.0 report with a split-calibrated null (Prop. 109) and pass the area rule (Prop. 111). New checks register their metric in `quality/registry.py`. | As listed | Real-vs-real splits | 0.99 |
| "forecasting TSTR within a stated gap" (T12) | `tstr_gap` with a forecasting learner on a temporal hold-out (last 20% of time); the thread states the gap before running. | A public time series from the register | TRTR | Calibrated at 0.99 |
| "benchmark tables with CIs over 5 seeds" (T11) | The SyntheRela block and the quality vector per generator, mean and 95% Student-t interval over seeds 1–5. | SyntheRela databases, golden databases | relsynth copula generator | 95% |
| "utility-vs-ε curves" (T07) | `tstr_gap` and the fidelity pass rate at ε ∈ {0.5, 1, 2, 4, 8}, mean and CI over 5 seeds, against MST and AIM on the single tables. | Chinook, Contoso, AdventureWorks | Non-DP relsynth; MST/AIM | 95% |
| "consistency of generated text with row values above a stated rate" (T14) | Share of synthetic texts whose conditioning attributes are recovered by the text classifier of `text_metrics`, compared with the same rate on real texts. | S6/S7, a public corpus | Real rate | 95% Clopper–Pearson |
| "degree and clustering checks" (T15) | KS of degree distributions and W1 of local clustering coefficients, registered as metrics with split-calibrated nulls. | SNAP, OGB graphs | Real-vs-real splits | 0.99 |
| "agent answers match ground truth at a stated rate" (T16) | Share of `review:` questions answered correctly, with a 95% Clopper–Pearson interval; every wrong answer must be rejected at the approval step in the tests. | Golden databases | Stated rate | 95% |

## Decisions (2026-10-28)

- **Utility targets** of the golden databases are declared in
  [`benchmarks/targets.yaml`](../benchmarks/targets.yaml). Derived columns (tax and freight as a share of the
  amount, calendar attributes, alternate keys) are not targets. Tables not listed, and the private bi database,
  use the documented heuristic, and each check's scope names the target used.
- **Tolerances** (practical floors) are kept as stated in [metrics.md](metrics.md). An organisation can set its
  own per metric; the calibrated limit still applies when it is larger.
- **Quality gate.** In 2.x the default gate stays `--fail-under` on the 1.0 share of passed checks; the 2.0
  vector gate is opt-in with `relsynth evaluate --quality FILE --quality-gate` or `QualityReport.require()`.
  The default moves to the 2.0 vector in 3.0.
