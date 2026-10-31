# SPDX-License-Identifier: Apache-2.0
"""Partition-aware statistics: does a table's law change across its partitions?

A lake table is usually partitioned by time (`month(OrderDate)`) or by a
business key (`Region`).  The engine fits one law per table, so before a fit
it is worth knowing whether the columns' distributions differ between
partitions.  `partition_drift` computes, in one DuckDB pass per column, the
sufficient statistics per partition (count, mean and variance of numbers;
level counts of categories) and tests homogeneity:

* numbers: Welch's heteroscedastic one-way ANOVA (Welch, Biometrika 1951),
  valid when variances differ between partitions;
* categories: Pearson's chi-square test of homogeneity, with levels lumped
  into "other" until every expected count is at least 5 (Cochran 1954).

The p-values of all columns are adjusted with Holm's step-down procedure
(Holm, Scand. J. Stat. 1979), so the chance of flagging any column of a table
whose law does not change is at most `alpha` (Prop. 904).  With millions of
rows tiny differences are significant, so a column is flagged only when the
adjusted p-value is below `alpha` *and* the effect size (eta squared for
numbers, Cramér's V for categories) reaches `min_effect`.

The result says where a time-varying law should be modelled (a time-varying
segment model, roadmap thread T12) or the table split; relsynth 1.x does not
model it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as st

MIN_EFFECT = {"numeric": 0.01, "categorical": 0.1}      # eta^2 "small" (Cohen 1988); Cramér's V "small"
MAX_LEVELS = 50


@dataclass
class DriftReport:
    table: str
    by: list[str]
    partitions: int
    rows: int
    alpha: float
    columns: list[dict] = field(default_factory=list)

    @property
    def drifting(self) -> list[str]:
        return [c["column"] for c in self.columns if c["drift"]]

    def to_dict(self) -> dict:
        return {"table": self.table, "by": self.by, "partitions": self.partitions, "rows": self.rows,
                "alpha": self.alpha, "drifting": self.drifting, "columns": self.columns}

    def to_markdown(self) -> str:
        lines = [f"## {self.table}: law across {self.partitions} partitions of {', '.join(self.by)}", "",
                 "| Column | Test | Statistic | p (Holm) | Effect | Drift |", "|---|---|---|---|---|---|"]
        for c in self.columns:
            lines.append(f"| {c['column']} | {c['test']} | {c['statistic']:.3g} | {c['p_holm']:.3g} | "
                         f"{c['effect_name']} {c['effect']:.3f} | {'yes' if c['drift'] else 'no'} |")
        return "\n".join(lines) + "\n"


def holm(p: np.ndarray) -> np.ndarray:
    """Holm's step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def welch_anova(n, mean, var) -> tuple[float, float, float]:
    """(F, p, eta^2) of Welch's ANOVA from per-group counts, means and sample
    variances.  Groups with fewer than two rows or no variance are left out of
    the test (they carry no variance estimate); eta^2 uses every group."""
    n, mean, var = (np.asarray(x, dtype=float) for x in (n, mean, var))
    tot = n.sum()
    grand = (n * mean).sum() / tot
    ssb = (n * (mean - grand) ** 2).sum()
    ssw = (np.where(n > 1, n - 1, 0) * np.nan_to_num(var)).sum()
    eta = float(ssb / (ssb + ssw)) if ssb + ssw > 0 else 0.0
    ok = (n > 1) & (var > 0)
    n, mean, var = n[ok], mean[ok], var[ok]
    k = len(n)
    if k < 2:
        return 0.0, 1.0, eta
    w = n / var
    W = w.sum()
    mw = (w * mean).sum() / W
    a = (w * (mean - mw) ** 2).sum() / (k - 1)
    tmp = ((1 - w / W) ** 2 / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
    f = a / b
    df2 = (k ** 2 - 1) / (3 * tmp) if tmp > 0 else np.inf
    return float(f), float(st.f.sf(f, k - 1, df2)), eta


def chi2_homogeneity(table: np.ndarray) -> tuple[float, float, float]:
    """(chi^2, p, Cramér's V) of a partitions x levels count table, after
    lumping the rarest levels until every expected count is at least 5."""
    t = np.asarray(table, dtype=float)
    t = t[t.sum(axis=1) > 0]
    t = t[:, t.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0, 1.0, 0.0
    order = np.argsort(t.sum(axis=0))
    t = t[:, order]
    while t.shape[1] > 2:
        exp = np.outer(t.sum(axis=1), t.sum(axis=0)) / t.sum()
        if float(np.min(exp)) >= 5:
            break
        t = np.column_stack([t[:, 0] + t[:, 1], t[:, 2:]])         # merge the two rarest levels
    chi2, p, _, _ = st.chi2_contingency(t, correction=False)
    v = np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1)))
    return float(chi2), float(p), float(v)


def _expr(by: str) -> tuple[str, str]:
    """(SQL expression, its name) of a partition field."""
    from ..connectors.lakehouse import parse_partition
    fn, col, width = parse_partition(by)
    q = '"' + col.replace('"', '""') + '"'
    ts = f"TRY_CAST({q} AS TIMESTAMP)"
    sql = {"identity": q, "year": f"year({ts})", "month": f"strftime({ts}, '%Y-%m')",
           "day": f"CAST({ts} AS DATE)", "hour": f"date_trunc('hour', {ts})",
           "bucket": f"hash({q}) % {width}",
           "truncate": f"CASE WHEN typeof({q}) IN ('VARCHAR') THEN left(CAST({q} AS VARCHAR), {width}) "
                       f"ELSE CAST(floor(TRY_CAST({q} AS DOUBLE) / {width}) * {width} AS VARCHAR) END"}[fn]
    return sql, by


def partition_fields(conn, table: str) -> list[str]:
    """The partition fields of a lake table, in `partition_by` notation."""
    kind = getattr(conn, "kind", "")
    if kind == "iceberg":
        tbl = conn._table(table)
        sch = tbl.schema()
        out = []
        for f in tbl.spec().fields:
            col = sch.find_field(f.source_id).name
            tr = str(f.transform).lower()
            if tr == "identity":
                out.append(col)
            elif "[" in tr:                                   # bucket[16], truncate[4]
                fn, w = tr.rstrip("]").split("[")
                out.append(f"{fn}({w}, {col})")
            else:
                out.append(f"{tr}({col})")
        return out
    if kind == "delta":
        return list(conn._dt(table).metadata().partition_columns)
    return []


def partition_drift(data, by=None, *, table: str | None = None, columns: list[str] | None = None,
                    alpha: float = 0.05, min_effect: dict | None = None) -> DriftReport:
    """Test whether the columns' laws differ across partitions.

    data: a DataFrame, an Arrow table or dataset, or a lake connector (then
    `table` names the table and `by` defaults to its partition fields).
    by: partition fields ('Region', 'month(OrderDate)', ...), one or several.
    """
    import duckdb
    name = table or "table"
    rel = data
    if hasattr(data, "dataset") and table is not None:
        if by is None:
            by = partition_fields(data, table)
        rel = data.dataset(table)
    by = [by] if isinstance(by, str) else list(by or [])
    if not by:
        raise ValueError(f"{name}: no partition fields given and the table has none")
    eff = {**MIN_EFFECT, **(min_effect or {})}
    con = duckdb.connect()
    try:
        con.register("rs_t", rel)
        cols = [r[0] for r in con.execute("DESCRIBE rs_t").fetchall()]
        types = {r[0]: r[1] for r in con.execute("DESCRIBE rs_t").fetchall()}
        exprs = [_expr(b) for b in by]
        part = " || '|' || ".join(f"COALESCE(CAST({e} AS VARCHAR), '<null>')" for e, _ in exprs)
        con.execute(f"CREATE TEMP VIEW rs_p AS SELECT *, {part} AS rs_part FROM rs_t")   # noqa: S608 - quoted names
        k, rows = con.execute("SELECT count(DISTINCT rs_part), count(*) FROM rs_p").fetchone() or (0, 0)
        used = {parse_col(b) for b in by}
        todo = [c for c in (columns or cols) if c not in used and c in types]
        res = []
        for c in todo:
            q = '"' + c.replace('"', '""') + '"'
            t = types[c].upper()
            if t.startswith(("DATE", "TIMESTAMP")):
                q, t = f"epoch(CAST({q} AS TIMESTAMP))", "DOUBLE"           # times are compared as numbers
            if any(x in t for x in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL", "NUMERIC")) and "[]" not in t:
                g = con.execute(f"SELECT count({q}), avg({q}), var_samp({q}) FROM rs_p GROUP BY rs_part").fetchall()  # noqa: S608
                g = [x for x in g if x[0]]
                if len(g) < 2:
                    continue
                n, m, v = zip(*g)
                stat, p, e = welch_anova(n, m, [x if x is not None else 0.0 for x in v])
                res.append({"column": c, "test": "Welch ANOVA", "statistic": stat, "p": p,
                            "effect_name": "eta2", "effect": e, "kind": "numeric"})
            elif not any(x in t for x in ("[]", "STRUCT", "MAP", "BLOB")):
                top = [r[0] for r in con.execute(
                    f"SELECT CAST({q} AS VARCHAR) v FROM rs_p WHERE {q} IS NOT NULL GROUP BY v "  # noqa: S608
                    f"ORDER BY count(*) DESC, v LIMIT {MAX_LEVELS}").fetchall()]
                if len(top) < 2:
                    continue
                lv = con.execute(
                    f"SELECT rs_part, CASE WHEN CAST({q} AS VARCHAR) IN (SELECT unnest(?)) THEN CAST({q} AS VARCHAR) "  # noqa: S608
                    f"ELSE '<other>' END v, count(*) FROM rs_p WHERE {q} IS NOT NULL GROUP BY ALL", [top]).fetchdf()
                tab = lv.pivot_table(index=lv.columns[0], columns="v", values=lv.columns[2], aggfunc="sum",
                                    fill_value=0).to_numpy()
                stat, p, e = chi2_homogeneity(tab)
                res.append({"column": c, "test": "chi-square", "statistic": stat, "p": p,
                            "effect_name": "V", "effect": e, "kind": "categorical"})
    finally:
        con.close()
    adj = holm(np.array([r["p"] for r in res]))
    for r, a in zip(res, adj):
        r["p_holm"] = float(a)
        r["drift"] = bool(a < alpha and r["effect"] >= eff[r["kind"]])
    return DriftReport(name, by, int(k), int(rows), alpha, res)


def parse_col(by: str) -> str:
    from ..connectors.lakehouse import parse_partition
    return parse_partition(by)[1]


def drift_frame(report: DriftReport) -> pd.DataFrame:
    return pd.DataFrame(report.columns)
