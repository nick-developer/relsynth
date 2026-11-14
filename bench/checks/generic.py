# SPDX-License-Identifier: Apache-2.0
"""Schema-level domain checks for any database of the matrix (G7).

The checks need no knowledge of the domain: they read the links and keys from
the spec the run used (the frozen spec or the introspect draft) and measure the
same things on the real source and on the synthetic CSV files, in DuckDB, so
that a 60-million-row TPC-H fact is never loaded into pandas.

| Check | Value | Judged |
|---|---|---|
| `X1 pk_duplicates\\|<table>` | share of rows whose primary key repeats | identity (real 0) |
| `X2 fk_orphans\\|<child>.<key>` | share of non-NULL keys without a parent row | identity (real 0) |
| `X3 selfref_cycle\\|<table>.<key>` | share of rows of a self-reference that sit on a cycle | identity (real 0) |
| `X4 fanout_top1pct\\|<child>.<key>` | share of child rows on the top 1% of parents (skew) | recorded |
| `X5 rows_ratio` | synthetic rows / real rows over all tables | recorded |
| `X6 null_share_mad\\|<table>` | mean absolute difference of the columns' NULL shares | recorded |

A value of exactly 0 or 1 on the real data makes a check an identity
(bench/runner.py `domain_item`); X4 to X6 are recorded without a verdict
(real value None where it would otherwise look like an identity).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

LIMIT_SELFREF_ROWS = 2_000_000     # the cycle walk loads parent links into memory


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _lit(path) -> str:
    """A file path as an SQL string literal (a view cannot take a bound parameter)."""
    return "'" + str(path).replace("'", "''") + "'"


def _connect(location: str, syn_dir: Path, tables: list[str]):
    """One DuckDB connection with real.<t> and syn.<t> views of every table."""
    import duckdb
    import pandas as pd
    con = duckdb.connect()
    con.execute("CREATE SCHEMA real")
    con.execute("CREATE SCHEMA syn")
    kind, path = location.split(":", 1)
    src = Path(path)
    for t in tables:
        f = syn_dir / f"{t}.csv"
        if f.exists():
            # every column as text: keys compare as written, whatever type the sniffer would guess
            con.execute(f"CREATE VIEW syn.{_q(t)} AS SELECT * FROM read_csv({_lit(f)}, header = true, "
                        "all_varchar = true)")
    if kind == "sqlite":
        lite = sqlite3.connect(src)
        try:
            for t in tables:
                df = pd.read_sql_query(f"SELECT * FROM {_q(t)}", lite)  # noqa: S608 (name from the spec, quoted)
                for c in df.columns:            # binary values: only whether they are present matters here
                    if df[c].map(lambda v: isinstance(v, bytes | bytearray | memoryview)).any():
                        df[c] = df[c].map(lambda v: None if v is None else len(v))
                con.register(f"_real_{t}", df)
                con.execute(f"CREATE VIEW real.{_q(t)} AS SELECT * FROM {_q('_real_' + t)}")
        finally:
            lite.close()
    else:
        reader = {"parquet": "read_parquet", "csv": "read_csv"}[kind]
        for t in tables:
            hits = [p for ext in (kind, "pq") for p in (src / f"{t}.{ext}", src / t) if p.exists()]
            if not hits:
                continue
            p = hits[0]
            glob = str(p / f"*.{kind}") if p.is_dir() else str(p)
            con.execute(f"CREATE VIEW real.{_q(t)} AS SELECT * FROM {reader}({_lit(glob)})")
    return con


def _has(con, schema: str, t: str) -> bool:
    return bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
                            [schema, t]).fetchone()[0])


def _txt(col: str) -> str:
    """A key column as text, so 12, 12.0 and '12' compare equal."""
    c = _q(col)
    return f"regexp_replace(CAST({c} AS VARCHAR), '\\.0+$', '')"


def checks(spec_path: Path, location: str, syn_dir: Path) -> dict:
    """{check: (real, synthetic)} for the runner's domain items."""
    from relsynth.spec import load_spec
    spec = load_spec(str(spec_path))
    tables = [t for t, ts in spec["tables"].items() if ts.get("mode") != "skip"]
    con = _connect(location, Path(syn_dir), tables)
    out: dict = {}
    try:
        both = [t for t in tables if _has(con, "real", t) and _has(con, "syn", t)]
        n_real = n_syn = 0
        for t in both:
            nr = con.execute(f"SELECT count(*) FROM real.{_q(t)}").fetchone()[0]
            ns = con.execute(f"SELECT count(*) FROM syn.{_q(t)}").fetchone()[0]
            n_real, n_syn = n_real + nr, n_syn + ns
            pk = spec["tables"][t].get("primary_key")
            if isinstance(pk, str):
                vals = []
                for s, n in (("real", nr), ("syn", ns)):
                    d = con.execute(f"SELECT count(DISTINCT {_txt(pk)}) FROM {s}.{_q(t)}").fetchone()[0]
                    vals.append(round(1 - d / n, 6) if n else 0.0)
                out[f"X1 pk_duplicates|{t}"] = tuple(vals)
            cols = [c for c in con.execute(f"SELECT * FROM real.{_q(t)} LIMIT 0").fetchdf().columns]
            scols = set(con.execute(f"SELECT * FROM syn.{_q(t)} LIMIT 0").fetchdf().columns)
            cols = [c for c in cols if c in scols]
            if cols and nr and ns:
                def shares(s, n, cols=cols, t=t):
                    sel = ", ".join(f"count({_q(c)})" for c in cols)
                    row = con.execute(f"SELECT {sel} FROM {s}.{_q(t)}").fetchone()
                    return [1 - v / n for v in row]
                r, s_ = shares("real", nr), shares("syn", ns)
                out[f"X6 null_share_mad|{t}"] = (None, round(sum(abs(a - b) for a, b in zip(r, s_)) / len(cols), 6))
        if n_real:
            out["X5 rows_ratio"] = (None, round(n_syn / n_real, 6))
        for rel in spec["relationships"]:
            c, p, ck, pk = rel["child"], rel["parent"], rel["child_key"], rel.get("parent_key")
            if c not in both or p not in both or not isinstance(ck, str) or not isinstance(pk, str):
                continue
            name = f"{c}.{ck}"
            vals, top = [], []
            for s in ("real", "syn"):
                n, orphans = con.execute(
                    f"SELECT count(*), count(*) FILTER (WHERE p.k IS NULL) FROM "
                    f"(SELECT {_txt(ck)} AS k FROM {s}.{_q(c)} WHERE {_q(ck)} IS NOT NULL) ch "
                    f"LEFT JOIN (SELECT DISTINCT {_txt(pk)} AS k FROM {s}.{_q(p)}) p USING (k)").fetchone()
                vals.append(round(orphans / n, 6) if n else 0.0)
                share = con.execute(
                    f"WITH f AS (SELECT {_txt(ck)} AS k, count(*) AS n FROM {s}.{_q(c)} WHERE {_q(ck)} IS NOT NULL "
                    f"GROUP BY 1), r AS (SELECT n, row_number() OVER (ORDER BY n DESC) AS i, count(*) OVER () AS m "
                    f"FROM f) SELECT sum(n) FILTER (WHERE i <= greatest(1, ceil(m * 0.01))) / sum(n) FROM r"
                ).fetchone()[0]
                top.append(None if share is None else round(float(share), 6))
            out[f"X2 fk_orphans|{name}"] = tuple(vals)
            if c != p:
                real_top = top[0] if top[0] not in (0.0, 1.0) else None
                out[f"X4 fanout_top1pct|{name}"] = (real_top, top[1])
            else:
                out[f"X3 selfref_cycle|{name}"] = tuple(_cycle_share(con, s, c, pk, ck) for s in ("real", "syn"))
    finally:
        con.close()
    return out


def _cycle_share(con, schema: str, table: str, pk: str, ck: str) -> float | None:
    """Share of rows whose chain of parents comes back to a row it passed."""
    rows = con.execute(f"SELECT {_txt(pk)}, {_txt(ck)} FROM {schema}.{_q(table)} LIMIT {LIMIT_SELFREF_ROWS + 1}"
                       ).fetchall()
    if not rows or len(rows) > LIMIT_SELFREF_ROWS:
        return None
    parent = {k: v for k, v in rows if k is not None}
    state: dict = {}                       # 1 on the current path, 2 done (no cycle), 3 on a cycle
    for start in parent:
        path, u = [], start
        while u in parent and u not in state:
            state[u] = 1
            path.append(u)
            u = parent[u]
        cyc = u in state and state[u] == 1
        hit = path.index(u) if cyc else len(path)
        for i, v in enumerate(path):
            state[v] = 3 if cyc and i >= hit else 2
    return round(sum(1 for v in state.values() if v == 3) / len(parent), 6)
