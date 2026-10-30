# SPDX-License-Identifier: Apache-2.0
"""Source data for a fit: every table loaded, except pushed-down facts, which come
as (uniform sample, exact statistics).

    data = load_for_fit(spec)          # FitData(tables, stats, plan)
    Synthesizer(spec).fit(data.tables, stats=data.stats)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from . import planner
from .stats import TableStats

log = logging.getLogger("relsynth.pushdown")


@dataclass
class FitData:
    """Prepared source tables (samples for pushed-down facts), their exact
    statistics, the plan, and the load notes."""
    tables: dict[str, pd.DataFrame]
    stats: dict[str, TableStats] = field(default_factory=dict)
    plan: planner.PushdownPlan = field(default_factory=planner.PushdownPlan)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def source_rows(self) -> dict[str, int]:
        """Rows of every source table (exact also for sampled facts)."""
        return {t: (self.stats[t].n_rows if t in self.stats else len(df)) for t, df in self.tables.items()}


def canonical_order(df: pd.DataFrame, pk=None, unique=()) -> pd.DataFrame:
    """The rows in an order that depends on their values only: by primary key, else
    by the first unique key that holds, else by a hash of the whole row.  Databases
    return rows in any order (and a parallel scan in a different one each time); the
    fit visits rows in order, so without this the same source could give different
    models on two servers."""
    if not len(df):
        return df
    for cand in [pk, *list(unique or [])]:
        keys = [cand] if isinstance(cand, str) else list(cand or [])
        if not keys or any(k not in df.columns for k in keys):
            continue
        try:
            out = df.sort_values(keys, kind="stable", na_position="last")
            if not out[keys].duplicated().any():
                return out.reset_index(drop=True)
        except TypeError:            # mixed types in a key column
            pass
    h = pd.util.hash_pandas_object(df.astype(str), index=False).to_numpy()
    return df.iloc[h.argsort(kind="stable")].reset_index(drop=True)


def load_for_fit(spec: dict, conn=None, mode: str | None = None) -> FitData:
    """Read the spec's source for fitting, pushing large facts down per
    synthesis.pushdown (mode overrides its mode)."""
    from ..data import load_tables, source_connector
    t0 = time.time()
    own = conn is None
    conn = conn or source_connector(spec)
    try:
        if mode is not None:
            spec = {**spec, "synthesis": {**spec["synthesis"], "pushdown": {**spec["synthesis"]["pushdown"], "mode": mode}}}
        names = list(spec["tables"])
        missing = [t for t in names if not conn.has_table(t)]
        if missing:
            where = conn.location()
            if conn.kind in ("csv", "parquet"):
                raise FileNotFoundError(f"{missing[0]}: no .csv, .csv.gz or .parquet file in {where}")
            raise ValueError(f"tables not in the source {where}: {', '.join(missing)}")
        stored = None
        if any(r.get("type") == "aggregate" for r in spec.get("rules", [])):
            stored = {t: set(conn.describe(t).column_names) for t in {r["table"] for r in spec["rules"]
                                                                       if r.get("type") == "aggregate"}}
        p = planner.plan(spec, conn, stored=stored)
        raw, stats = {}, {}
        seed = int(spec["synthesis"].get("seed", 7))
        for t in names:
            if t in p.pushed:
                df, whole = conn.sample(t, p.sample_rows, seed=seed)
                stats[t] = conn.stats(t)
                log.info("pushdown: %s: %s of %s rows sampled for its attribute models; links from statistics",
                         t, f"{len(df):,}", f"{p.pushed[t]:,}")
            elif spec["tables"][t].get("role") == "view" and p.mode != "off" and \
                    conn.count(t) >= int(spec["synthesis"]["pushdown"].get("min_rows", 1_000_000)):
                # a view is rebuilt from its base, never fitted: a sample serves the evaluation
                df, whole = conn.sample(t, p.sample_rows, seed=seed)
                stats[t] = conn.stats(t)
            else:
                df = conn.read_table(t)
            raw[t] = canonical_order(df, spec["tables"][t].get("primary_key"), spec["tables"][t].get("unique"))
        load_tables.notes = []
        tables = load_tables(spec, tables=raw, partial=set(stats))
        notes = list(load_tables.notes)
        if callable(getattr(conn, "versions", None)) and conn.versions():
            # lakehouse sources: the version of every table read (pin with source.snapshot)
            notes.append("versions read: " + ", ".join(f"{t}={v}" for t, v in sorted(conn.versions().items())))
        return FitData(tables=tables, stats=stats, plan=p, notes=notes,
                       seconds=round(time.time() - t0, 1))
    finally:
        if own:
            conn.close()
