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
| `python -m bench run [--datasets ...] [--seeds 1,2,3,4,5] [--no-fake]` | synthesise and score; writes one history record |
| `python -m bench compare [RECORD] [--full]` | a record (default: the newest in `bench/history/`) against the baseline |
| `python -m bench baseline RECORD...` | rebuild `bench/baseline.json` from history records |
| `python -m bench all` | everything above in order, then `compare --full` |

## The golden databases

| Dataset | Source | Checks |
|---|---|---|
| `chinook` | Chinook 1.4.5, SQLite (MIT) | scorecard |
| `adventureworks` | AdventureWorks DW install script, built into SQLite (MIT) | scorecard, G1–G14 |
| `contoso_sqlite` | Contoso V2 100k Parquet release plus the generator's DDL, built into SQLite (MIT) | scorecard, T1–T17 |
| `contoso_parquet` | the same release read as raw Parquet files | scorecard, T1–T17 |
| `bi_corep_lr` | the maintainer's private EBA COREP extract | scorecard, B1–B8 |
| `fake` | the statistical check suite (pytest -m science) | C00–C78 |

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
its domain checks. A domain check is judged only when it is an identity: the
real value is 0 or 1 (no duplicate, every row obeys the rule), and it passes
when the synthetic value is exactly the same. Other domain measures (a mean, a
KS distance) are recorded without a verdict. Each statistical check C00–C78 is
one pass or fail.

A history record (`bench/history/<time>-<commit>-<machine>.json`) stores the
relsynth version, the commit and whether the tree was dirty, the machine (CPU,
Python, numpy, pandas, scipy), the seeds, and every check of every run with its
value.

## The regression rule

The baseline summarises the records of one commit over five seeds. For every
dataset it keeps the per-seed pass rate (passed / judged) and, for every check,
the seeds on which it failed. A new record regresses when

- **R1** a *calibrated* check fails: one that passed on every baseline seed (at
  least three; one for the deterministic statistical checks);
- **R2** the mean pass rate over the new seeds falls below the baseline mean by
  more than the spread, max(2 × standard deviation of the baseline's per-seed
  rates, 1 / number of judged checks);
- **R3** a run errors, or (with `--full`) a dataset of the baseline did not run
  and was not skipped as unavailable.

A check that failed on some baseline seed is unstable: a borderline statistical
item that flips with the seed. It counts towards R2 but cannot trigger R1. The
rule is a heuristic chosen for this harness, not a test with a stated level:
two standard deviations would be a 95% band if per-seed rates were normal, and
the floor of one check stops a baseline whose seeds all agree from flagging
every single flip.

Results depend slightly on the machine (BLAS and thread counts change the
order of floating-point sums), so compare records from the same kind of
machine; the CI baseline and a laptop baseline can differ by an unstable check
or two.

## In CI

- `ci.yml`, job `bench`, on every pull request: Chinook on five seeds and
  Contoso (Parquet) on seed 1, compared with the baseline (about 15 minutes).
- `bench.yml`, on every push to main, weekly and on demand: every public
  database on five seeds and the statistical suite, `compare --full`. The
  record is uploaded as an artefact.

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
