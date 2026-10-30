# Benchmark harness

One command runs every suite of a clean checkout and says whether anything got worse:

```bash
pip install -e ".[dev,bench]" datasketches
python -m bench all
```

`all` runs ruff, the unit and end-to-end tests, the live-server tests when
`RELSYNTH_TEST_PG_URL`, `RELSYNTH_TEST_MSSQL_URL` or `RELSYNTH_TEST_ORACLE_URL`
is set ([testing](../testing.md)), the golden databases on seeds 1 to 5, and the
statistical check suite C00–C78. It then compares the results with
`bench/baseline.json` and exits with 1 on a regression. The first run downloads
and builds the public databases (about 150 MB, 10 minutes); a full run takes
about 4 hours on 4 cores, most of it AdventureWorks (about 30 minutes per seed alone, 55 with five seeds in parallel; timings are in `bench/baseline.json`).

## Commands

| Command | What it does |
|---|---|
| `python -m bench list` | the datasets and whether each is available here |
| `python -m bench fetch` | download (checked against pinned SHA-256 sums) and build the public databases |
| `python -m bench run [--datasets ...] [--seeds 1,2,3,4,5] [--suites ...] [--no-fake \| --fake-only]` | synthesise and score; writes one history record |
| `python -m bench compare [RECORD ...] [--full]` | a record (default: the newest in `bench/history/`), or the records of one run split over several jobs, against the baseline |
| `python -m bench baseline RECORD...` | rebuild `bench/baseline.json` from history records |
| `python -m bench all` | everything above in order, then `compare --full` |

## The golden databases

| Dataset | Source | Checks |
|---|---|---|
| `chinook` | Chinook 1.4.5, SQLite (MIT) | scorecard, quality, determinism |
| `chinook_x2` | Chinook drawn at `synthesis.scale` 2 | scorecard, quality (fan-out normalised by the scale) |
| `adventureworks` | AdventureWorks DW install script, built into SQLite (MIT) | scorecard, G1–G14, quality (5/3 null splits) |
| `contoso_sqlite` | Contoso V2 100k Parquet release plus the generator's DDL, built into SQLite (MIT) | scorecard, T1–T17, quality, privacy (customers) |
| `contoso_parquet` | the same release read as raw Parquet files | scorecard, T1–T17, quality |
| `berka` | Berka / PKDD'99 Czech bank (SyntheRela "financial": 8 tables, 1.08M rows), built into SQLite with SDV's keys | scorecard, F1–F16, quality, privacy (accounts) |
| `acs` | ACS PUMS 2019, NIST CRC national excerpt (27,253 people, 24 columns), built into SQLite | scorecard, A1–A14, quality, privacy with shadow models (people), determinism |
| `bi_corep_lr` | the maintainer's private EBA COREP extract | scorecard, B1–B8, quality |
| `fake` | the statistical check suite (pytest -m science) | C00–C78, C600–C606 and the other science checks |

Berka and ACS were added on 2026-10-30 as the most complex databases of the
[dataset register](../datasets.md) that the harness can fetch: Berka is the
richest allowed SyntheRela database (a bridge table between clients and
accounts, two parents of the district table, a 1M-row transaction fact with a
time column, an `order` table whose name is an SQL keyword, derived and
aggregate rules) and ACS the register's reference privacy dataset (structural
missing values, a geography code, survey weights, sensitive income and
disability items). TPC-DS (`dsdgen`, 24 tables) and MIMIC-IV demo (clinical,
ODbL) are the next candidates; they need the DuckDB extension repository and
physionet.org, which the environment that built this baseline could not reach.

The download URLs and checksums are in `bench/datasets.py`; licences and terms
are in the [dataset register](../datasets.md). Nothing downloaded is committed:
the cache is `bench/.cache` (or `RELSYNTH_BENCH_CACHE`). History records hold
only aggregates (scores, shares, counts), never rows.

bi/COREP is private. It runs only where `RELSYNTH_BENCH_BI_DIR` points to a
folder with `data/*.csv`, `spec.yaml` and `checks.py`; elsewhere it is reported
as skipped, which is not a regression. Its results, even the aggregates, never
enter the repository: a run writes them to `<RELSYNTH_BENCH_BI_DIR>/bench/history/`
under the same file name as the public record (which only notes that the
dataset ran), and `baseline` writes its baseline to
`<RELSYNTH_BENCH_BI_DIR>/bench/baseline.json`. `compare` merges both where they
are present.

Each database runs with a frozen spec in `bench/specs/`: Chinook with the
example spec, the others with the unedited 1.0 `introspect` draft. Freezing the
spec separates a change of the engine from a change of what `introspect`
proposes (a fresh Chinook draft adds a dependency AlbumId → MediaTypeId that
costs two checks); `introspect` itself is covered by the unit and end-to-end
tests.

## What a result is

Every real database is synthesised once per seed with `relsynth run` in a fresh
process and scored by relsynth's scorecard (one pass or fail per item) and by
its domain checks; then, each in a fresh process (`bench/assess.py`), the
suites that apply to the dataset (`--suites`, default all of them):

| Suite | What runs | Items in the record |
|---|---|---|
| `scorecard` | relsynth's 1.0 scorecard and the domain checks | `scorecard:<area>\|<metric>`, `domain:<check>` |
| `quality` | the calibrated 2.0 quality report (`relsynth.assess_quality`, targets from `benchmarks/targets.yaml`) | `quality:<check id>` (pass under the calibrated limit), `area:<fidelity\|utility\|integrity>` (the area rule of Prop. 111, exact), `goal:<metric>`, `metric:<value>` |
| `privacy` | `relsynth.privacy.assess` on a release fitted to 40% of the people, the split drawn with the run's seed (settings in `benchmarks/privacy.yaml`) | `privacy:<criterion>\|<attack>` (passes when the upper 95% bound is under the internal context's limit; per-person results are information only), `goal:privacy_internal`, `goal:privacy_<external context>_needs_dp`, `metric:privacy_<attack>` |
| `repro` | the same seed synthesised twice | `goal:deterministic` (identical files, exact) |

Every run also records its generation cost from the operating system
(`metric:cpu_seconds_per_million_rows`, `metric:peak_memory_mib`).

### Which goal each item covers

The goals are the roadmap's acceptance metrics ([acceptance metrics](../acceptance-metrics.md))
and the privacy criteria of the EDPB guidelines that T06 implements. Each is
covered by the cheapest item that decides it, on the datasets where it means
something:

| Goal | Item | Datasets |
|---|---|---|
| Referential integrity 100% | `goal:referential_integrity` (minimum `fk_integrity`), scorecard integrity items | all |
| Business rules preserved (>99%) | `goal:business_rule_preservation` (every hard rule within the real data's own violation rate), domain identities | all; Berka F1–F11 and ACS A1–A12 test rules the draft spec may miss |
| Distributions (95%+, calibrated) | `goal:distribution_match`, `area:fidelity`, `quality:` column, pair, 3-way, precision/recall checks | all |
| Cross-table structure | `quality:` fan-out, 2-hop fan-out, hop-trend and detection checks; `metric:cardinality_similarity_*`, `metric:k_hop_trend_*`, `metric:c2st_agg_accuracy_max` | multi-table databases |
| Utility gap (<5%) | `goal:utility_gap`, `area:utility`, `metric:utility_gap_max` | all with targets |
| Scale control | `chinook_x2`: integrity and rate-normalised fan-out at twice the size | Chinook |
| Generation failures (<0.1%) | R3 (a run that errors) | all |
| Reproducibility | `goal:deterministic` | Chinook, ACS |
| Singling out, linkage, inference (EDPB) | `privacy:record_isolation\|*`, `privacy:linkage\|*`, `privacy:inference\|*` | ACS, Berka, Contoso |
| Attribute leakage of sensitive items | `privacy:inference\|attribute_*` (members minus non-members) | ACS (income decile, disabilities), Berka (loan status, balance), Contoso (spend) |
| Membership inference | `privacy:membership\|domias_*` everywhere privacy runs; shadow-model LiRA and classifier where a fit is cheap | ACS (shadows), Berka, Contoso |
| External releases need DP | `goal:privacy_partner_needs_dp`, `goal:privacy_public_needs_dp` (blocked without DP) | ACS, Berka, Contoso |
| Detecting a leak at all | positive controls C600–C606 in the statistical suite | `fake` |
| Generation cost | `metric:cpu_seconds_per_million_rows`, `metric:peak_memory_mib` (notes only) | all |

AdventureWorks runs the quality report with 5/3 null splits (the default 20/10
take about an hour per seed) and no privacy suite (67 minutes per seed with
shadows in T06); Chinook has no privacy suite because its 59 customers are
below the small-population limit, where T06 showed that the attacks succeed
and only DP helps.

A domain check is judged only when it is an identity: the
real value is 0 or 1 (no duplicate, every row obeys the rule), and it passes
when the synthetic value is exactly the same. Other domain measures (a mean, a
KS distance) are recorded without a verdict. Each statistical check C00–C78 is
one pass or fail.

A history record (`bench/history/<time>-<commit>-<machine>.json`) stores the
suites that ran, and the
relsynth version, the commit and whether the tree was dirty, the machine (CPU,
Python, numpy, pandas, scipy), the seeds, and every check of every run with its
value.

## The regression rule

The baseline summarises the records of one commit over five seeds. For every
dataset it keeps the per-seed pass rate (passed / judged) and, for every check,
the seeds on which it failed. A new record regresses when

- **R1** a *calibrated* check fails: one that passed on every baseline seed (at
  least three; one for the deterministic statistical checks). An exact check
  (an identity, a scorecard item with limit 0, a statistical-suite check)
  regresses on any failing seed; a statistical scorecard item (a distance under
  a positive limit) only when it fails on more than half of the new seeds;
- **R2** the mean pass rate over the new seeds, on the checks the baseline
  has (a new kind of check counts once a baseline with it is built), falls below
  the baseline mean by
  more than the spread, max(2 × standard deviation of the baseline's per-seed
  rates, 1 / number of judged checks);
- **R3** a run errors, or (with `--full`) a dataset of the baseline did not run
  and was not skipped as unavailable;
- **R4** a tracked value (`metric:` items: SyntheRela summaries, area pass
  rates, the utility gap, attack values) moves the wrong way: its mean over the
  new seeds is worse than the baseline mean by more than
  max(0.02 × max(1, |baseline mean|), 2 × the baseline's standard deviation).
  This is the "benchmark regression flag" of the acceptance metrics; it catches
  a slow drift that no single pass or fail shows. Cost values depend on the
  machine and are only noted when they grow by more than 25%.

A check that failed on some baseline seed is unstable: a borderline statistical
item that flips with the seed. It counts towards R2 but cannot trigger R1. The
rule is a heuristic chosen for this harness, not a test with a stated level:
two standard deviations would be a 95% band if per-seed rates were normal, and
the floor of one check stops a baseline whose seeds all agree from flagging
every single flip.

Results depend slightly on the machine (BLAS and thread counts change the
order of floating-point sums), so compare records from the same kind of
machine; the CI baseline and a laptop baseline can differ by an unstable check
or two. Five seeds cannot show that a statistical item never fails: of the
roughly 1,300 AdventureWorks items, a few pass five times by chance and then
fail on one seed of a GitHub runner (seen on main after the harness merged:
two items on one or two seeds each). The majority rule keeps these flips out
of R1 and lists them as notes, while a broken table, which fails on most
seeds, is still reported. A pull-request run of one seed keeps the
single-failure rule.

## In CI

- `ci.yml`, job `bench`, on every pull request: Chinook on five seeds,
  Contoso (Parquet) and Berka on seed 1, compared with the baseline (about 40
  minutes with the quality and privacy suites).
- `bench.yml`, on every push to main, weekly and on demand: every public
  database on five seeds and the statistical suite, one job per database in
  parallel, then one `compare --full` over all the records. The records are
  uploaded as artefacts.

## Changing the baseline

A change that improves results, or that changes them on purpose, updates the
baseline in the same pull request:

```bash
python -m bench run --label baseline          # all golden databases, five seeds
python -m bench baseline bench/history/<the new record>.json
git add bench/baseline.json bench/history/<the new record>.json
```

The pull request says why the numbers moved. A baseline built on a different
machine from the previous one says so in its description.

## Dependencies

The harness needs `py7zr` (the `bench` extra) to unpack the Contoso release.
py7zr is LGPL-2.1-or-later; it is a development tool and never a dependency of
the relsynth package ([licence check](../../tools/check_licenses.py)).
