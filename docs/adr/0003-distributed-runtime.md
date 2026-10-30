# ADR 0003: Partitioned generation with counter-based streams; local, process and Spark executors

- Status: proposed (roadmap thread T10, release 3.5)
- Date: 2026-10-30

## Context

relsynth 1.0 samples a dataset in one process from one random stream. Lake-scale output (hundreds of millions of fact rows) needs the work spread over cores and machines, with three constraints the roadmap sets: the same seed must give identical output with 1, 8 or 64 workers; keys and hard rules must hold across partitions; and the synthetic data must keep the guarantees of a single-node run (the same law). The profile of 1.0 (docs/profiling.md) shows sampling time is in the attribute models and rule redraws of fact rows, which are per row, and in the relational step for entities, which is not.

## Decision

1. **Two phases.** The driver draws the parameters and every entity, then each fact's state (anchor and partner rows with their weights) once. Fact rows are generated in partitions of the anchor rows. Facts whose joins couple rows across the table (occupancy, degree laws) get their joins on the driver and their attributes in partitions; facts with constraints across anchors that cannot be split run as one partition. The rule that decides is `relsynth.runtime.partition_rule`, with a stated reason.
2. **Counter-based streams.** Each partition draws from Philox4x64 keyed by SHA-256 of (seed, dataset, fact, partition) (Salmon et al., SC 2011; numpy's `Philox`). Cached per-dataset draws (Bayesian-bootstrap weights) are keyed by their name. No state passes between partitions, so the result is a function of the labels only.
3. **Partitions fixed by the model.** Block boundaries follow from the anchor rows, the expected (or, for driver-drawn joins, the actual) rows per anchor and `partition_rows`, never from the number of workers.
4. **Executors behind one interface** (`map(fn, items, shared)`, results in item order): `local`, `process` (fork on Linux: the fitted model and entities are inherited, not copied), `spark` (one broadcast of the model, one task per partition). The default stays the 1.0 sampler (`executor: none`), byte-identical.
5. **Keys after the join.** Surrogate keys are numbered over the joined partitions on the driver (Spark's `zipWithIndex` pattern), generated codes are drawn over the join, so they are unique and dense.

## Alternatives considered

| Option | Why not (now) |
|---|---|
| One stream split by `SeedSequence.spawn` per worker | The draws would depend on how partitions are assigned to workers; spawning per partition label is what we do, through a hash so labels can be strings |
| Per-row counters (Philox counter = row number) | Exact per row, but every engine step would have to draw by row index; per-partition keys give the same worker independence with no change to the samplers |
| Ray (Moritz et al., OSDI 2018) | Good fit for Python tasks, but adds a cluster runtime most of our users do not run; Spark is already in Databricks, Fabric, EMR and Synapse, which are the warehouse order the user chose. Ray can be added as a fourth executor behind the same interface |
| Dask | Similar to Ray; the same reason |
| Generating partitions in SQL inside the warehouse | The engine's samplers (copula, rule redraws) are Python; pushing them into SQL is a rewrite, and the fit already runs in the database (pushdown) |
| Partitioning entities too | Entities carry the relational constraints (link moves, uniqueness across the table, participation); splitting them needs its own law-preserving scheme. Deferred: see the T10 report |

## Benchmark

The scale study in `roadmap/reports/T10.md` (script `roadmap/reports/T10/scripts/scale_study.py`) measures throughput per executor and worker count on one 4-core machine, with identical digests across executors as the check. On one machine, `process` is the fastest executor; Spark `local[n]` has a fixed cost per job of several seconds (JVM start, Python worker start, broadcast) and pays off only on a cluster. No cluster run has been made yet.

## Consequences

- A partitioned dataset is a different draw from the 1.0 sampler's with the same law; `partition_rows` is part of its identity.
- The driver still holds the entities and the joined facts; streaming partitions from workers straight into Iceberg or Delta files is the next step.
- `structure` facts keep their joins on the driver: three integer arrays per fact row (24 bytes per row).
