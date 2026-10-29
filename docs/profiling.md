# Where relsynth 1.0 spends time and memory

This profile decides which engine changes are worth making. It measures relsynth 1.0.0 (`main`) on two real databases from the dataset register, stage by stage: CPU, memory and I/O. The rule from the roadmap is to change only what the profile justifies.

## Method

- **Machine:** 4 vCPUs, Linux, Python 3.13, numpy with its default BLAS threads.
- **Runs:** one process per database, run with the unedited specs from the real-data runs and the default installation (no `relsynth[speed]`). The stage tables come from runs on an otherwise idle machine. A first pair of runs, made while other jobs shared the machine, was 30–40% slower in every stage; its results are kept with the scripts but not used here. Contoso V2 100k is read as Parquet (8 tables, 753k rows). AdventureWorks DW is read from SQLite (30 tables, 1.05M rows).
- **Stages:** load, fit, sample, evaluate, evaluate with privacy, write CSV and write Parquet. Each is timed separately.
- **Measurements:**
  - Wall and CPU time come from `time.perf_counter` and `time.process_time`.
  - Resident memory is sampled every 20 ms by a `psutil` thread; the table gives the start and peak per stage.
  - Read and written bytes come from `/proc/self/io`.
  - A second run under `cProfile` records the function profile. The sampler thread's own calls (`time.sleep`, `psutil`, reads of `/proc`) are removed from the shares below.
- **Scripts:** `profile_stages.py` in the project's T04 report folder.

## Stages

**Contoso V2 100k (Parquet)**: scorecard 253/253, overall peak 1,425 MiB.

| Stage | Wall s | CPU s | RSS start → peak MiB | Read MiB | Written MiB |
|---|---:|---:|---:|---:|---:|
| load | 15.4 | 12.7 | 156 → 696 | 16.4 | 0 |
| fit | 171.4 | 287.6 | 556 → 973 | 0.3 | 0 |
| sample | 75.7 | 76.1 | 690 → 1,129 | 0.1 | 0 |
| evaluate | 54.9 | 73.9 | 1,125 → 1,381 | 0.1 | 0 |
| evaluate + privacy | 54.2 | 73.7 | 1,326 → 1,425 | 0.1 | 0 |
| write CSV | 5.1 | 4.2 | 1,336 → 1,337 | 4.2 | 53.7 |
| write Parquet | 1.6 | 1.5 | 1,336 → 1,358 | 0 | 17.2 |

**AdventureWorks DW (SQLite)**: scorecard 1229/1286, the same as 1.0's recorded result. Overall peak 2,202 MiB.

| Stage | Wall s | CPU s | RSS start → peak MiB | Read MiB | Written MiB |
|---|---:|---:|---:|---:|---:|
| load | 13.4 | 13.4 | 160 → 736 | 102.9 | 0 |
| fit | 855.2 | 2,351.6 | 542 → 1,039 | 1.3 | 0 |
| sample | 55.7 | 56.4 | 896 → 1,373 | 0.1 | 0 |
| evaluate | 75.0 | 80.0 | 1,313 → 2,159 | 0.3 | 0 |
| evaluate + privacy | 74.2 | 80.2 | 1,391 → 2,202 | 0.1 | 0 |
| write CSV | 4.5 | 4.3 | 1,438 → 1,442 | 0 | 59.1 |
| write Parquet | 1.0 | 1.0 | 1,442 → 1,471 | 0.1 | 13.5 |

Fit CPU time is 1.7–2.7 times wall time because numpy's BLAS uses all four cores.

## Function profile

These come from the first pair of runs (under `cProfile`, which slows Python code but not compiled code). They are shares of the main thread's own time (cProfile `tottime`) after the sampler thread's calls are removed.

| | Contoso | AdventureWorks |
|---|---:|---:|
| relsynth Python code | 40% | 52% |
| numpy and scipy | 24% | 31% |
| pandas | 25% | 13% |
| pyarrow and file I/O | 1.5% | 0.8% |
| SQLAlchemy and SQLite | 0% | 0.1% |
| other | 9% | 3% |

The largest cumulative relsynth functions on AdventureWorks are:

| Function | Cumulative s |
|---|---:|
| `engine.fit` | 1,249 |
| `relational.fit` | 997 |
| `relational.run` (the Gibbs sweeps over segments) | 825 |
| `engine._refine_nested_parent` | 415 |
| `relational._merge_gain` | 263 |
| `copula.fit` | 156 |
| `relational.update_z` | 132 |
| `relational.gumbel_argmax` | 74 |
| `copula.rtruncnorm` | 71 |

On Contoso the order is the same, and `kernels.gauss_ll` (86 s) and `engine.sample` (120 s) follow.

Most pandas time is not in DataFrame operations. It comes from iterating element by element over Arrow-backed string and datetime columns: `ArrowExtensionArray.__iter__` takes 14 s on Contoso and 12 s on AdventureWorks, followed by `_from_sequence`, `isna` on strings and datetime iteration. No single relsynth function accounts for most of it.

## The optional numba kernels

`relsynth[speed]` already compiles one kernel, the Gaussian log-likelihood of the segment sweep (`kernels.gauss_ll`), which gives bit-identical results. The same runs with numba installed, on the idle machine:

| Database | Fit without numba | Fit with numba | Peak memory | Scorecard |
|---|---:|---:|---:|---|
| Contoso | 171 s | 108 s (−37%) | 1,425 → 1,588 MiB | 253/253 both |
| AdventureWorks | 855 s | 850 s (−1%) | 2,202 → 2,297 MiB | 1229/1286 both |

Contoso's fit is dominated by numeric columns in the segment sweep, which the kernel covers. AdventureWorks spends its fit in `_refine_nested_parent`, `_merge_gain` and the categorical parts of `relational.run`, which have no kernel yet. Numba adds 100–160 MiB (the LLVM compiler).

## What the profile justifies

| Candidate | Evidence | Decision |
|---|---|---|
| Replace pandas with Polars | pandas is 13–25% of main-thread time, and most of that is Python-level iteration over string columns, which Polars would not remove | **No.** The cost would be a rewrite of the data layer for at most that share |
| Arrow for file reads and writes | load reads 16–103 MiB in 13–15 s, mostly type normalisation; writes take 1–6 s; I/O is under 2% of the profile | **No** for 1.0's in-memory path. Arrow does pay where results cross a process boundary: see DuckDB below and warehouse bulk loads (T18) |
| Columnar fetch from DuckDB | grouped statistics of a 22.4M-row fact returned as Python row tuples used 6.3 GB | **Yes, done**: `DuckDBFiles.frame()` |
| DuckDB for large file sources | loading the 22.4M-row fact whole peaks at 13 GB | **Yes, done**: DuckDB pushdown |
| More compiled kernels for the relational Gibbs sweep | `relational.fit` takes 997 of `engine.fit`'s 1,249 cumulative seconds on AdventureWorks (80%), mostly in `relational.run`, `_refine_nested_parent` and `_merge_gain`, which loop over segments in Python | **Candidate.** The one existing kernel cuts Contoso's fit by 37% and does nothing for AdventureWorks. Kernels for `_refine_nested_parent`, `_merge_gain` and the categorical likelihood, with the same bit-identity checks (C69), are the change most likely to pay off. T04 does not change the engine; this goes to the engine thread |
| Element-wise iteration over Arrow strings | 10–20 s per run | **Candidate.** Convert once to NumPy object arrays where code iterates. A small change, left for the engine owner |

Fit memory is not the limit at this size: the peak comes from evaluate (AdventureWorks 2.2 GB), which holds the real and synthetic tables together. On a large fact, the limit is loading the fact itself, which pushdown removes.
