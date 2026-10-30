# SPDX-License-Identifier: Apache-2.0
"""Distributed runtime: synthetic datasets generated in partitions by an
executor (this process, a pool of processes, or Spark), with counter-based
random streams so that the output does not depend on the number of workers.

    from relsynth.runtime import sample_partitioned
    tables, run = sample_partitioned(model, executor={"executor": "process", "workers": 8},
                                     partition_rows=1_000_000)
"""
from .executors import EXECUTORS, Executor, LocalExecutor, ProcessExecutor, SparkExecutor, make_executor
from .partitioned import DEFAULT_PARTITION_ROWS, PartitionedRun, blocks_for, partition_rule, sample_partitioned
from .streams import KeyedRunCache, stream, stream_key

__all__ = ["EXECUTORS", "Executor", "LocalExecutor", "ProcessExecutor", "SparkExecutor", "make_executor",
           "DEFAULT_PARTITION_ROWS", "PartitionedRun", "blocks_for", "partition_rule", "sample_partitioned",
           "KeyedRunCache", "stream", "stream_key"]
