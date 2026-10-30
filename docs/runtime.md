# Partitioned generation: workers, processes and Spark

By default relsynth samples a synthetic dataset in one process, as release 1.0 did. `synthesis.runtime` makes it generate the fact tables in **partitions** that run in parallel: in this process, in a pool of worker processes, or as the tasks of a Spark job. The output depends on the seed, the dataset index and `partition_rows`, and never on the executor or the number of workers.

```yaml
synthesis:
  runtime:
    executor: process        # none (default) | local | process | spark
    workers: 8               # processes, or Spark tasks at once
    partition_rows: 1000000  # fact rows per partition, about
```

```python
import relsynth as rs
from relsynth.runtime import sample_partitioned

model = rs.fit("spec.yaml")
tables, run = sample_partitioned(model.synthesizer, executor={"executor": "spark"}, partition_rows=2_000_000)
print(run.to_dict())          # per fact: mode, partitions, rows, seconds
```

The run report is also in the dataset's sample log (`runtime`) and the diagnostics bundle.

## How a dataset is generated

1. **Driver.** The posterior draw of the parameters, then every entity (dimension) with its segments and rules, as in 1.0. Then, per fact, the state its rows are drawn from: the anchor rows (segments, keys, degree weights, eligibility) and the partner rows. These are drawn once for the dataset.
2. **Partitions.** The anchor rows of a fact are cut into consecutive blocks of about `partition_rows` fact rows. Partition *p* draws the rows of block *p* from its own random stream, keyed by (seed, dataset, fact, *p*): rows per anchor or pairs, attributes, rule redraws, unique keys, finishing.
3. **Driver.** The partitions are joined in partition order. Surrogate keys of the fact are numbered 1..*n* over the join (and generated codes drawn over it), as Spark's `zipWithIndex` does. Facts nested on a fact (order lines on orders) are drawn from the joined rows of their parent the same way. Then the aggregates and the external form of the tables, from the stream (seed, dataset, `tail`).

How each fact is split is reported as its **mode**:

| Mode | When | What runs where |
|---|---|---|
| `partition` | Rows per anchor row or Poisson pairs (most facts) | Everything per partition |
| `structure` | Occupancy facts (a pair at most once) and link tables with degree laws (`link_degree`) | The joins (which anchor and partner rows each row links) on the driver, for the whole table; attributes, rules and keys per partition |
| `whole` | Something couples rows across anchor rows after the joins: a unique key or a group effect that does not contain the anchor key, link affinity, a running balance over groups that cross anchors, a sequence numbered over the table | One partition |

The reason of every `structure` and `whole` fact is in the run report.

## Guarantees

* **Same output for any number of workers** (Prop. 900). Partitions are fixed before any task runs, and each partition's random numbers come from a Philox counter-based generator (Salmon et al., SC 2011) whose key is a hash of (seed, dataset, fact, partition). Run order, worker and executor do not enter. Tested with 1, 2, 8 and 64 processes and with Spark (`tests/unit/test_runtime.py`).
* **Same law as one process** (Props. 901, 902). In `partition` mode the rows of different anchor rows are independent given the driver's state (Poisson thinning and superposition), so drawing them per block gives the law of drawing them together. In `structure` mode the joins are drawn exactly as in one process; the rest is per row. The output is a different draw from the 1.0 sampler's (other streams), with the same distribution.
* **Keys and rules across partitions** (Prop. 903). Primary keys are unique and foreign keys resolve in the joined tables; unique keys and rules that contain the anchor key hold because an anchor's rows are never split; the others make the fact `whole`.
* **No change without `runtime`.** With `executor: none` (the default) the sampler and its output are those of 1.0, byte for byte.

## Executors

| Executor | Use | Notes |
|---|---|---|
| `local` | Testing, debugging | One partition after the other in this process |
| `process` | One machine with several cores | Linux forks the workers, which share the model and the entities without copying them; elsewhere they are spawned and the model is pickled once per worker (`cloudpickle` when installed) |
| `spark` | A cluster: Databricks, EMR, HDInsight, Synapse, Kubernetes | Uses the active `SparkSession`, or starts `local[workers]` (`master` to choose another). The model and entities go to the executors once, as a broadcast variable; each partition is a task. Needs `pyspark` and Java |

Ray is not included; the choice is recorded in [ADR 0003](adr/0003-distributed-runtime.md).

## Does the law change across partitions?

The engine fits one law per table. A lake table partitioned by time can hold a law that drifts from month to month, which one law averages out. `partition_drift` tells you before the fit:

```python
from relsynth.connectors import connect
from relsynth.runtime import partition_drift

lake = connect({"kind": "iceberg", "path": "./lake", "schema": "sales"})
report = partition_drift(lake, table="Sales")              # by the table's partition fields
report = partition_drift(lake, "month(OrderDate)", table="Sales")
print(report.to_markdown(), report.drifting)
```

It computes per-partition sufficient statistics in DuckDB in one pass per column (counts, means and variances of numbers and times; level counts of categories), tests homogeneity with Welch's ANOVA or the chi-square test, adjusts the p-values of all columns with Holm's procedure and flags a column when the adjusted p-value is below `alpha` (0.05) and the effect is not negligible (eta squared at least 0.01, Cramér's V at least 0.1; `min_effect` changes them). Under one law across partitions the chance of flagging any column is at most `alpha` (Prop. 904). A data frame or Arrow table works too: `partition_drift(df, "Region")`.

relsynth does not yet model a law that changes across partitions; split the table by period, or wait for the time-varying models of the time-series work.

## Limits of this release

* Entities (dimensions) are generated on the driver. In schemas where order headers or customers are entities with millions of rows, the driver's share of the time grows; facts are where partitions help (see the scale study in the T10 report).
* The driver holds the joined tables, so a dataset must fit in the driver's memory. Writing partitions straight from the workers to Iceberg or Delta data files, with the aggregates combined from per-partition partials, is the next step.
* `partition_rows` is part of the dataset's identity: change it and the draws change (the law does not).
