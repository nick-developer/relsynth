# SPDX-License-Identifier: Apache-2.0
"""Executors: where the partitions of a synthetic dataset are generated.

An executor maps a task function over a list of items and returns the results
in item order.  Every task gets the same read-only `shared` object (the fitted
model and the entities of the dataset), shipped once per map:

* `local`: in this process, one item after the other;
* `process`: a pool of worker processes on this machine (fork: the workers
  inherit `shared` without copying it, POSIX only; spawn elsewhere);
* `spark`: the tasks of one Spark job (Zaharia et al., NSDI 2012), `shared`
  sent once as a broadcast variable; for clusters (Databricks, EMR, HDInsight,
  Synapse, Kubernetes).

The result never depends on the executor or its number of workers: partitions
are fixed before the map and every task draws from its own counter-based
stream (relsynth.runtime.streams, Prop. 950).  docs/adr/0001-executors.md
records why these three and how they compare.
"""
from __future__ import annotations

import logging
import os
import pickle  # nosec B403 - only objects this process created are unpickled by its own workers
import sys
from typing import Any, Callable

log = logging.getLogger("relsynth.runtime")


def dumps(obj) -> bytes:
    """Pickle with cloudpickle when installed: a fitted model holds closures
    (its posterior draw functions), which the standard pickle refuses."""
    try:
        import cloudpickle
        return cloudpickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except ImportError:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

EXECUTORS = ("local", "process", "spark")


class Executor:
    """Maps fn(shared, item) over items; results in item order."""

    name = "base"
    workers = 1

    def map(self, fn: Callable[[Any, Any], Any], items: list, shared: Any = None) -> list:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"executor": self.name, "workers": self.workers}

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release workers."""


class LocalExecutor(Executor):
    name = "local"

    def map(self, fn, items, shared=None):
        return [fn(shared, x) for x in items]


# the shared object of the running map, inherited by forked workers
_SHARED: Any = None


def _call(arg):
    fn, item = arg
    return fn(_SHARED, item)


def _init_spawned(blob):
    global _SHARED
    _SHARED = pickle.loads(blob)  # noqa: S301  # nosec B301 - written by the parent process


class ProcessExecutor(Executor):
    """A pool of worker processes.  start: fork (default on Linux; workers inherit
    the shared object copy-on-write) or spawn (the object is pickled once per
    worker)."""

    name = "process"

    def __init__(self, workers: int | None = None, start: str | None = None):
        self.workers = int(workers or os.cpu_count() or 1)
        if start is None:
            start = "fork" if sys.platform.startswith("linux") else "spawn"
        if start not in ("fork", "spawn", "forkserver"):
            raise ValueError("process executor: start must be fork, spawn or forkserver")
        self.start = start

    def map(self, fn, items, shared=None):
        global _SHARED
        import multiprocessing as mp
        if not items:
            return []
        ctx = mp.get_context(self.start)
        n = min(self.workers, len(items))
        if self.start == "fork":
            _SHARED = shared
            try:
                pool = ctx.Pool(n)
            finally:
                _SHARED = None
        else:
            pool = ctx.Pool(n, initializer=_init_spawned, initargs=(dumps(shared),))
        try:
            return pool.map(_call, [(fn, x) for x in items], chunksize=1)
        finally:
            # close and join, not terminate: terminating workers that hold a large
            # inherited heap takes seconds, an orderly exit does not
            pool.close()
            pool.join()

    def describe(self):
        return {"executor": self.name, "workers": self.workers, "start": self.start}


_BROADCAST_CACHE: dict[int, Any] = {}


def _spark_task(fn, bc):
    def run(pair):
        i, item = pair
        key = bc.id if hasattr(bc, "id") else id(bc)
        shared = _BROADCAST_CACHE.get(key)
        if shared is None:
            shared = pickle.loads(bc.value)  # noqa: S301  # nosec B301 - broadcast by the driver of this job
            _BROADCAST_CACHE.clear()
            _BROADCAST_CACHE[key] = shared
        return i, dumps(fn(shared, item))
    return run


class SparkExecutor(Executor):
    """Tasks of a Spark job.  spark: a SparkSession (default: the active one, or a
    local[workers] session); workers: the number of tasks run at once (default:
    the context's default parallelism)."""

    name = "spark"

    def __init__(self, spark=None, workers: int | None = None, master: str | None = None):
        try:
            from pyspark.sql import SparkSession
        except ImportError as e:      # pragma: no cover - depends on the environment
            raise ImportError("executor spark needs pyspark (pip install 'relsynth[spark]') or a Spark runtime") from e
        if spark is None:
            spark = SparkSession.getActiveSession()
        if spark is None:
            m = master or f"local[{int(workers or os.cpu_count() or 1)}]"
            spark = (SparkSession.builder.master(m).appName("relsynth")
                     .config("spark.ui.enabled", "false").config("spark.driver.maxResultSize", "0").getOrCreate())
            self._own = True
        else:
            self._own = False
        self.spark = spark
        self.workers = int(workers or spark.sparkContext.defaultParallelism)

    def map(self, fn, items, shared=None):
        if not items:
            return []
        sc = self.spark.sparkContext
        bc = sc.broadcast(dumps(shared))
        try:
            rdd = sc.parallelize(list(enumerate(items)), numSlices=len(items))
            got = rdd.map(_spark_task(fn, bc)).collect()
        finally:
            bc.unpersist()
        got.sort(key=lambda p: p[0])
        return [pickle.loads(b) for _, b in got]  # noqa: S301  # nosec B301 - results of our own tasks

    def describe(self):
        return {"executor": self.name, "workers": self.workers, "master": self.spark.sparkContext.master}

    def close(self):
        if self._own:
            self.spark.stop()


def make_executor(spec: dict | str | Executor | None = None) -> Executor:
    """An executor from a runtime section ({executor, workers, start, master}),
    a name or an Executor (returned as is)."""
    if isinstance(spec, Executor):
        return spec
    if spec is None:
        return LocalExecutor()
    cfg = {"executor": spec} if isinstance(spec, str) else dict(spec)
    name = cfg.get("executor", "local")
    if name == "local":
        return LocalExecutor()
    workers = int(cfg["workers"]) if cfg.get("workers") is not None else None
    if name == "process":
        return ProcessExecutor(workers, cfg.get("start"))
    if name == "spark":
        return SparkExecutor(workers=workers, master=cfg.get("master"))
    raise ValueError(f"runtime.executor must be one of {', '.join(EXECUTORS)}, not {name!r}")
