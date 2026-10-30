# SPDX-License-Identifier: Apache-2.0
"""Partitioned generation: one synthetic dataset with its fact tables split
into partitions that executors generate in parallel.

Two phases (docs/runtime.md):

1. On the driver, as in `Synthesizer.sample`: the posterior draw of the
   parameters, then every entity (dimension) with its segments, finished.
   Then, per fact, the state its rows are drawn from (anchor segments, key
   columns and degree weights; partner rows and their weights), drawn once
   for the whole dataset from the fact's own stream.
2. The fact's anchor rows are cut into consecutive blocks of a fixed number of
   rows; block p becomes partition p.  A task draws the rows of one partition
   from the stream (seed, dataset, fact, p): rows per anchor or pairs,
   attributes, rule redraws, unique keys and finishing.  The partitions are
   joined in partition order, surrogate keys are numbered 1..n over the join
   (as Spark's zipWithIndex), and nested facts are drawn from the joined rows
   of their parent the same way.  The aggregates and the external form of the
   tables follow on the driver, from the stream (seed, dataset, "tail").

The blocks depend only on the model, the dataset and `partition_rows`, never on
the executor, so the output is identical for any number of workers (Prop. 900).
Given phase 1, the rows of different anchor rows are independent in the fact
models that qualify (Poisson rows per anchor, Poisson pairs; Prop. 901), so the
partitioned dataset has the same law as an unpartitioned one; facts whose
model couples rows across anchors (occupancy, degree laws over the whole
table, link affinity, keys or groups not containing the anchor key) are drawn
whole, as one partition, and listed with the reason in the run report.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .executors import Executor, make_executor
from .streams import KeyedRunCache, stream

log = logging.getLogger("relsynth.runtime")

DEFAULT_PARTITION_ROWS = 1_000_000


@dataclass
class FactPlan:
    fact: str
    mode: str            # partition | structure | whole
    reason: str
    anchor_rows: int
    blocks: list = field(default_factory=list)          # (partition, start, stop) over anchor rows
    rows: int = 0
    seconds: float = 0.0

    def to_dict(self):
        return {"fact": self.fact, "mode": self.mode, "reason": self.reason,
                "anchor_rows": self.anchor_rows, "partitions": len(self.blocks), "rows": self.rows,
                "seconds": round(self.seconds, 2)}


def _anchor_columns(syn, f) -> set:
    info = syn.fact_info[f]
    if info["nested"]:
        return {c for c in info["nest"]["child_key"] if "." not in c}
    return {info["fks"][0]["child_key"]}


PARTITION, STRUCTURE, WHOLE = "partition", "structure", "whole"


def partition_rule(syn, f) -> tuple[str, str]:
    """How fact f is split (Props. 901, 902):

    * partition: rows per anchor row, or Poisson pairs, are independent across
      anchor rows given the state, so each partition draws its own;
    * structure: the joins (occupancy pairs, degree laws over the whole table)
      are drawn once on the driver, then the attributes of each partition's
      rows in parallel;
    * whole: something couples rows across anchor rows after the joins (a key or
      group not containing the anchor key, link affinity): one partition.

    Returns (mode, reason)."""
    info = syn.fact_info[f]
    rel = info["rel"]
    T = syn.spec["tables"][f]
    if getattr(syn, "affinity", {}).get(f):
        return WHOLE, "link affinity: the matching share is a share of the whole table"
    if info["nested"] and info["nest"].get("parent") != info.get("anchor"):
        return WHOLE, "nested on a fact other than its anchor"
    anchor = _anchor_columns(syn, f)
    keys = [list(u) for u in T.get("unique", [])] + [list(p["columns"]) for p in T.get("partial_unique", [])]
    for k in keys:
        if not set(k) & anchor:
            return WHOLE, f"unique key {k} does not contain the anchor key {sorted(anchor)}"
    for g in T.get("group_effects") or []:
        if not set(g.get("by", [])) & anchor:
            return WHOLE, f"group effect on {g.get('column')} is shared by groups that cross anchor rows"
    for r in syn.spec.get("rules", []):
        if r["table"] == f and r["type"] == "cumulative":
            by = r["by"] if isinstance(r["by"], list) else [r["by"]]
            if not set(by) & anchor:
                return WHOLE, f"running balance of {r['column']} is kept over groups that cross anchor rows"
    for c, cs in T["columns"].items():
        if cs["type"] == "sequence" and cs.get("within") not in anchor:
            return WHOLE, f"sequence {c} is numbered within {cs.get('within')}, not within the anchor"
        if (cs.get("distribution") or {}).get("kind") == "sequence":
            return WHOLE, f"{c} follows a sequence over the whole table"
    if rel.kind == "occ":
        return STRUCTURE, "occupancy model: pairs drawn on the driver (a pair occurs at most once in the table)"
    if info.get("link_degree"):
        return STRUCTURE, "link degree law: joins drawn on the driver (degrees conditioned on table totals)"
    return PARTITION, ""


def _expected_rows(syn, f, st) -> float:
    """Expected number of rows of fact f given its state (sets the block size)."""
    info = syn.fact_info[f]
    rel = info["rel"]
    z1, w1 = np.asarray(st["z1"]), np.asarray(st["w1"], dtype=float)
    if not len(z1):
        return 0.0
    if rel.kind == "single":
        lam = np.asarray(rel.lam, dtype=float)
        return float((lam[z1] * (w1 if not info.get("count_dp") else 1.0)).sum())
    if rel.kind == "occ":
        return float(len(z1))
    z2, w2 = np.asarray(st["z2"]), np.asarray(st["w2"], dtype=float)
    K1, K2 = rel.e1.K, rel.e2.K
    W1 = np.bincount(z1, weights=w1, minlength=K1)
    W2 = np.bincount(z2, weights=w2, minlength=K2)
    m_real = np.asarray(rel.e2.counts(), dtype=float)
    m_syn = np.bincount(z2, minlength=K2).astype(float)
    eta = np.asarray(rel.eta, dtype=float) * np.where(m_syn > 0, m_real / np.maximum(m_syn, 1), 0.0)[None, :]
    return float((eta * W1[:, None] * W2[None, :]).sum())


def blocks_for(n_anchor: int, expected_rows: float, partition_rows: int) -> list[tuple[int, int, int]]:
    """Consecutive blocks of anchor rows holding about partition_rows fact rows
    each: (partition, start, stop)."""
    if n_anchor <= 0:
        return [(0, 0, 0)]
    per = expected_rows / n_anchor if expected_rows > 0 else 1.0
    size = max(1, int(math.ceil(partition_rows / max(per, 1e-12))))
    return [(p, a, min(a + size, n_anchor)) for p, a in enumerate(range(0, n_anchor, size))]


def blocks_by_rows(u, n_anchor: int, partition_rows: int) -> tuple[np.ndarray, list]:
    """Blocks of anchor rows for a structure drawn on the driver: consecutive
    anchor rows until a block holds partition_rows rows (an anchor's rows never
    split).  Returns (row order by anchor, [(partition, start, stop, r0, r1)])."""
    u = np.asarray(u, dtype=np.int64)
    order = np.argsort(u, kind="stable")
    per = np.bincount(u, minlength=n_anchor) if n_anchor else np.zeros(0, np.int64)
    cum = np.concatenate([[0], np.cumsum(per)])
    blocks, a, p = [], 0, 0
    while a < n_anchor or (n_anchor == 0 and not blocks):
        b = int(np.searchsorted(cum, cum[a] + partition_rows, side="left")) if n_anchor else 0
        b = min(max(b, a + 1), n_anchor) if n_anchor else 0
        blocks.append((p, a, b, int(cum[a]) if n_anchor else 0, int(cum[b]) if n_anchor else 0))
        a, p = b, p + 1
        if n_anchor == 0:
            break
    return order, blocks


def _slice_rows(state, rows):
    """A fact's nested-anchor state ({z, keycols}) restricted to rows."""
    return {"z": np.asarray(state["z"])[rows], "keycols": {c: np.asarray(v)[rows] for c, v in state["keycols"].items()}}


def _join_rows(states):
    states = [s for s in states if s is not None]
    if not states:
        return None
    return {"z": np.concatenate([np.asarray(s["z"]) for s in states]),
            "keycols": {c: np.concatenate([np.asarray(s["keycols"][c], dtype=object) for s in states])
                        for c in states[0]["keycols"]}}


def _fresh_log():
    return {"rule_violations": {}, "text_rejections": {}, "rows_removed_by_rules": {}, "uniqueness": {}, "warnings": []}


def _merge_log(into: dict, part: dict) -> None:
    """Counts add up over partitions; lists are joined; other values: the first."""
    for k, v in part.items():
        if k not in into:
            into[k] = v
        elif isinstance(v, dict) and isinstance(into[k], dict):
            _merge_log(into[k], v)
        elif isinstance(v, list) and isinstance(into[k], list):
            into[k].extend(x for x in v if x not in into[k])
        elif isinstance(v, (int, float, np.integer, np.floating)) and isinstance(into[k], (int, float, np.integer, np.floating)) \
                and not isinstance(v, bool):
            into[k] = into[k] + v


def _task(shared, block):
    """Partition p of fact f: its rows, the state of facts nested on them, the
    surrogate key columns it numbered and its sample log."""
    syn = shared["syn"]
    f = shared["fact"]
    p, a, b = block[:3]
    syn.rng = stream(shared["seed"], shared["index"], f, "partition", p)
    if isinstance(syn.run_cache, KeyedRunCache):
        syn.run_cache.bind(syn)
    keep_log = syn.sample_log
    syn.sample_log = _fresh_log()
    try:
        rows = np.arange(a, b)
        st = syn._restrict_state(shared["state"], rows)
        fact_rows = dict(shared["fact_rows"])
        info = syn.fact_info[f]
        if info["nested"]:
            fact_rows[info["anchor"]] = _slice_rows(fact_rows[info["anchor"]], rows)
        out = shared["out"]
        if shared.get("structure") is not None:
            # joins drawn on the driver: this partition's rows, anchors renumbered
            su, sv, sblk = shared["structure"]
            idx = shared["order"][block[3]:block[4]]
            u = su[idx] - a
            v = sv[idx] if sv is not None else None
            df, nested_state = syn._fact_attributes(f, st, u, v, sblk[idx], out, fact_rows)
        else:
            df, nested_state = syn._fact_block(f, st, out, fact_rows)
        before = set(df.columns)
        df = syn._apply_affinity(f, df, out)
        df = syn._apply_bridge(f, syn._finish(f, df, None, out))
        T = syn.spec["tables"][f]
        # keys generated by _finish (surrogate numbers, unique codes) are made again
        # over the joined partitions on the driver, so they are unique in the table
        numbered = [c for c, cs in T["columns"].items() if cs["type"] == "id" and c not in before]
        return df, nested_state, numbered, syn.sample_log
    finally:
        syn.sample_log = keep_log


@dataclass
class PartitionedRun:
    """What sample_partitioned did: the plan of every fact, the executor and the time."""
    facts: list
    executor: dict
    partition_rows: int
    seconds: float

    def to_dict(self):
        return {"executor": self.executor, "partition_rows": self.partition_rows, "seconds": round(self.seconds, 2),
                "facts": [p.to_dict() for p in self.facts]}


def sample_partitioned(syn, scale=None, index=None, executor: Executor | dict | str | None = None,
                       partition_rows: int = DEFAULT_PARTITION_ROWS):
    """One synthetic dataset of a fitted Synthesizer, its facts generated per
    partition by `executor` (local, process or spark; see executors.py).
    Returns (tables, PartitionedRun).  The same seed, dataset index and
    partition_rows give the same tables with every executor and number of
    workers."""
    from ..engine import say
    if int(partition_rows) < 1:
        raise ValueError("partition_rows must be at least 1")
    t0 = time.time()
    ex = make_executor(executor)
    ctx = syn._sample_entities(scale, index)
    index = int(ctx["index"])
    pc = syn.cfg.get("posterior", {})
    seed = int(syn.cfg["seed"] if pc.get("sample_seed") is None else pc["sample_seed"])
    syn.run_cache = KeyedRunCache(syn.run_cache, seed, index).bind(syn)
    out, zs, keys = ctx["out"], ctx["zs"], ctx["keys"]
    fact_rows: dict = {}
    plans = []
    for f in syn.fact_order:
        t1 = time.time()
        say(f"sample: fact {f} (partitioned)")
        info = syn.fact_info[f]
        syn.rng = stream(seed, index, f, "state")
        st = syn._fact_state(f, zs, keys, out, fact_rows, extras=True)
        n1 = len(st["z1"])
        mode, why = partition_rule(syn, f)
        anchor_rows = {info["anchor"]: fact_rows[info["anchor"]]} if info["nested"] else {}
        shared = {"syn": syn, "fact": f, "state": st, "out": out, "seed": seed, "index": index,
                  "fact_rows": anchor_rows}
        if mode == PARTITION:
            blocks = blocks_for(n1, _expected_rows(syn, f, st), int(partition_rows))
        elif mode == STRUCTURE:
            syn.rng = stream(seed, index, f, "structure")
            u, v, blk = syn._fact_structure(f, st, out, {**fact_rows})
            order, blocks = blocks_by_rows(u, n1, int(partition_rows))
            shared.update({"structure": (np.asarray(u, np.int64), None if v is None else np.asarray(v, np.int64),
                                         np.asarray(blk, np.int64)), "order": order})
        else:
            blocks = [(0, 0, n1)]
        if len(blocks) > 1:
            parts = ex.map(_task, blocks, shared)
        else:
            parts = [_task(shared, blocks[0])]
        frames = [r[0] for r in parts]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].reset_index(drop=True)
        for c in parts[0][2]:
            if (f, c) in syn.pii:
                from ..engine import _unique_codes
                df[c] = _unique_codes(syn.pii[(f, c)], len(df), stream(seed, index, f, "codes", c))
            else:
                df[c] = np.arange(1, len(df) + 1)
        for r in parts:
            _merge_log(syn.sample_log, r[3])
        out[f] = df
        fact_rows[f] = _join_rows([r[1] for r in parts])
        plans.append(FactPlan(f, mode, why, n1, [b[:3] for b in blocks], len(df), time.time() - t1))
        say(f"sample: fact {f}: {len(df):,} rows in {len(blocks)} partition(s)")
    syn.rng = stream(seed, index, "tail")
    tables = syn._sample_end(ctx)
    run = PartitionedRun(plans, ex.describe(), int(partition_rows), time.time() - t0)
    syn.sample_log["runtime"] = run.to_dict()
    return tables, run
