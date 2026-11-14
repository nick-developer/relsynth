# SPDX-License-Identifier: Apache-2.0
"""Parameterised schema stress generator (G7): a SQLite database whose schema is
made to be hard for introspection and synthesis, built from a seed alone.

    python stress_schema.py OUT.db [--tables 120] [--depth 8] [--wide 500] [--rows 200000]
                                   [--branches 0] [--skew 1.3] [--seed 7]

What the schema holds (each part is named so that a failure points at it):

* snowflake branches `b<i>_l<k>`: dimension chains of depth `--depth`; level k
  references level k+1, the leaf level `b<i>_l1` is referenced by the facts.
  Row counts grow towards the leaf (3 rows at the top, x2 to x4 per level).
* facts `fact_<j>`: each references the leaf of three branches, a date and an
  amount that depends on the leaf's category; one link per fact has an extreme
  Zipf fan-out (`--skew`), so a few parents own most rows.
* a composite key: `ck_account(region_code, account_no)` and `ck_txn`, whose
  link to it is the composite pair.
* cyclic keys: `employee.manager_id -> employee` (a tree, the root has none) and
  the two-table cycle `department.head_id -> employee`,
  `employee.department_id -> department`.
* a many-to-many bridge `bridge_emp_project(employee_id, project_id)`.
* binary columns (`BLOB`): `document.payload` (bytes of a few lengths) and
  `document.thumbnail` (often NULL).
* a wide table `wide_profile` with `--wide` columns (numbers, categories,
  flags, nulls) linked to a leaf dimension.
* messy values: a free-text `note` with stray whitespace and mixed case,
  numbers with outliers, a category with a rare long tail.
* a regular series: `sensor` and `sensor_reading(sensor_id, hour, value)`,
  one reading per sensor and hour (a daily cycle plus AR(1) noise).
* filler tables `ref_<k>` (small code lists referenced by the wide table) until
  the schema has `--tables` tables.

Declared PRIMARY and FOREIGN KEYs, so introspect reads the schema the way it
reads a production catalog.  The same arguments always build the same file
contents (one numpy Generator from `--seed`).
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
from sqlite_ddl import create_table, insert_rows


def build(out: Path, tables: int, depth: int, wide: int, rows: int, branches: int, skew: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out.unlink(missing_ok=True)
    con = sqlite3.connect(out)
    counts: dict[str, int] = {}

    def add(name, cols, data, pk=(), fks=(), unique=()):
        create_table(con, name, cols, primary_key=list(pk), foreign_keys=list(fks), unique=list(unique))
        insert_rows(con, name, [c for c, _, _ in cols], data)
        counts[name] = len(data)

    # fixed parts: 4 facts, ck pair, employee/department/project/bridge, document, wide, sensor pair = 14 tables
    n_fixed = 14
    n_facts = 4
    if branches <= 0:   # as many branches as fit, leaving at least 3 code lists for the wide table
        branches = max(3, (tables - n_fixed - 3) // depth)
    n_refs = max(3, tables - n_fixed - branches * depth)
    cats = ["alpha", "beta", "gamma", "delta", "epsilon"]

    # ------------------------------------------------------------ snowflake branches
    leaf_rows, leaf_cat = {}, {}
    for b in range(branches):
        parent_n = None
        for k in range(depth, 0, -1):
            name = f"b{b}_l{k}"
            n = 3 if parent_n is None else int(min(parent_n * rng.integers(2, 5), 4000))
            ids = np.arange(1, n + 1)
            cat = rng.choice(cats, size=n, p=[0.4, 0.25, 0.2, 0.1, 0.05])
            weight = np.round(rng.gamma(2.0, 10.0, size=n), 2)
            cols = [("id", "INTEGER", True), ("name", "TEXT", True), ("category", "TEXT", False),
                    ("weight", "REAL", False)]
            data = [[int(i), f"{name}-{i}", str(c), float(w)] for i, c, w in zip(ids, cat, weight)]
            fks = []
            if parent_n is not None:
                # each parent gets at least one child, the rest at random: a hierarchy without gaps
                par = np.concatenate([np.arange(1, parent_n + 1), rng.integers(1, parent_n + 1, size=n - parent_n)])
                rng.shuffle(par)
                cols.append(("parent_id", "INTEGER", True))
                for row, p in zip(data, par):
                    row.append(int(p))
                fks = [(["parent_id"], f"b{b}_l{k + 1}", ["id"])]
            add(name, cols, data, pk=["id"], fks=fks)
            parent_n = n
        leaf_rows[b], leaf_cat[b] = parent_n, cat

    # ------------------------------------------------------------ code lists
    for r in range(n_refs):
        n = int(rng.integers(3, 30))
        add(f"ref_{r}", [("code", "TEXT", True), ("label", "TEXT", True)],
            [[f"R{r}C{i}", f"label {r}.{i}"] for i in range(n)], pk=["code"])

    # ------------------------------------------------------------ employees, departments, projects (cycles, m:n)
    n_emp, n_dept, n_proj = max(50, rows // 200), max(5, rows // 4000), max(10, rows // 1000)
    manager = np.zeros(n_emp, dtype=int)
    for i in range(1, n_emp):                       # a tree: each manager has a smaller id
        manager[i] = int(rng.integers(1, i + 1))
    dept_of = rng.integers(1, n_dept + 1, size=n_emp)
    con.execute("PRAGMA foreign_keys = OFF")        # the two-table cycle is loaded in one go
    create_table(con, "department", [("id", "INTEGER", True), ("name", "TEXT", True), ("head_id", "INTEGER", False),
                                     ("budget", "REAL", False)], primary_key=["id"],
                 foreign_keys=[(["head_id"], "employee", ["id"])])
    create_table(con, "employee", [("id", "INTEGER", True), ("name", "TEXT", True),
                                   ("manager_id", "INTEGER", False), ("department_id", "INTEGER", True),
                                   ("hired", "DATE", True), ("salary", "REAL", True), ("note", "TEXT", False)],
                 primary_key=["id"], foreign_keys=[(["manager_id"], "employee", ["id"]),
                                                   (["department_id"], "department", ["id"])])
    heads = rng.choice(np.arange(1, n_emp + 1), size=n_dept, replace=False)
    insert_rows(con, "department", ["id", "name", "head_id", "budget"],
                [[d + 1, f"dept {d + 1}", int(h) if rng.random() > 0.1 else None, float(np.round(rng.lognormal(12, 1), 2))]
                 for d, h in enumerate(heads)])
    hired = np.datetime64("2010-01-01") + rng.integers(0, 5000, size=n_emp).astype("timedelta64[D]")
    salary = np.round(rng.lognormal(10.8, 0.35, size=n_emp), 2)
    salary[rng.random(n_emp) < 0.01] *= 25          # outliers
    notes = np.array(["  Senior ", "junior", "JUNIOR", "contractor  ", "", None, "Remote", "remote "], dtype=object)
    insert_rows(con, "employee", ["id", "name", "manager_id", "department_id", "hired", "salary", "note"],
                [[i + 1, f"emp {i + 1}", int(manager[i]) or None, int(dept_of[i]), str(hired[i]), float(salary[i]),
                  notes[rng.integers(0, len(notes))]] for i in range(n_emp)])
    counts["department"], counts["employee"] = n_dept, n_emp
    con.execute("PRAGMA foreign_keys = ON")
    add("project", [("id", "INTEGER", True), ("title", "TEXT", True), ("started", "DATE", True)],
        [[p + 1, f"project {p + 1}", str(np.datetime64("2015-01-01") + np.timedelta64(int(rng.integers(0, 3000)), "D"))]
         for p in range(n_proj)], pk=["id"])
    pairs = set()
    for e in range(1, n_emp + 1):
        for p in rng.choice(np.arange(1, n_proj + 1), size=int(rng.integers(0, 4)), replace=False):
            pairs.add((e, int(p)))
    add("bridge_emp_project", [("employee_id", "INTEGER", True), ("project_id", "INTEGER", True),
                               ("role", "TEXT", True)],
        [[e, p, ["lead", "member", "member", "reviewer"][(e + p) % 4]] for e, p in sorted(pairs)],
        pk=["employee_id", "project_id"],
        fks=[(["employee_id"], "employee", ["id"]), (["project_id"], "project", ["id"])])

    # ------------------------------------------------------------ composite key
    regions = ["NORTH", "SOUTH", "EAST", "WEST"]
    acc = [(r, a) for r in regions for a in range(1, max(20, rows // 400) + 1)]
    add("ck_account", [("region_code", "TEXT", True), ("account_no", "INTEGER", True), ("opened", "DATE", True),
                       ("tier", "TEXT", True)],
        [[r, a, str(np.datetime64("2012-01-01") + np.timedelta64(int(rng.integers(0, 4000)), "D")), ["basic", "gold", "platinum"][a % 3]]
         for r, a in acc], pk=["region_code", "account_no"])
    n_txn = max(200, rows // 4)
    pick = rng.integers(0, len(acc), size=n_txn)
    add("ck_txn", [("id", "INTEGER", True), ("region_code", "TEXT", True), ("account_no", "INTEGER", True),
                   ("amount", "REAL", True)],
        [[i + 1, acc[j][0], acc[j][1], float(np.round(rng.normal(100, 40), 2))] for i, j in enumerate(pick)],
        pk=["id"], fks=[(["region_code", "account_no"], "ck_account", ["region_code", "account_no"])])

    # ------------------------------------------------------------ binary columns
    n_doc = max(50, rows // 100)
    lens = rng.choice([16, 64, 256], size=n_doc, p=[0.5, 0.3, 0.2])
    add("document", [("id", "INTEGER", True), ("employee_id", "INTEGER", True), ("kind", "TEXT", True),
                     ("payload", "BLOB", True), ("thumbnail", "BLOB", False)],
        [[i + 1, int(rng.integers(1, n_emp + 1)), ["pdf", "png", "txt"][int(lens[i]) % 3],
          rng.integers(0, 256, size=int(lens[i]), dtype=np.uint8).tobytes(),
          rng.integers(0, 256, size=8, dtype=np.uint8).tobytes() if rng.random() < 0.3 else None]
         for i in range(n_doc)], pk=["id"], fks=[(["employee_id"], "employee", ["id"])])

    # ------------------------------------------------------------ regular series
    n_sensor, hours = max(4, rows // 5000), 24 * 14
    add("sensor", [("id", "INTEGER", True), ("kind", "TEXT", True)],
        [[i + 1, ["temperature", "humidity", "pressure"][i % 3]] for i in range(n_sensor)], pk=["id"])
    t0 = np.datetime64("2024-03-01T00:00")
    readings = []
    for i in range(n_sensor):
        level, e = rng.normal(20, 5), 0.0
        for h in range(hours):
            e = 0.8 * e + rng.normal(0, 0.5)
            readings.append([i + 1, str(t0 + np.timedelta64(h, "h")).replace("T", " ") + ":00",
                             round(float(level + 3 * np.sin(2 * np.pi * h / 24) + e), 3)])
    add("sensor_reading", [("sensor_id", "INTEGER", True), ("hour", "DATETIME", True), ("value", "REAL", True)],
        readings, pk=["sensor_id", "hour"], fks=[(["sensor_id"], "sensor", ["id"])])

    # ------------------------------------------------------------ wide table
    n_wide = max(100, rows // 50)
    wcols = [("id", "INTEGER", True), ("leaf_id", "INTEGER", True)]
    wfks = [(["leaf_id"], "b0_l1", ["id"])]
    cols_data = [np.arange(1, n_wide + 1), rng.integers(1, leaf_rows[0] + 1, size=n_wide)]
    base = rng.normal(size=n_wide)                  # a shared factor: the wide columns are correlated
    for k in range(max(0, wide - 2)):
        kind = k % 5
        if kind in (0, 1):
            v = np.round(base * rng.uniform(0.2, 1.0) + rng.normal(size=n_wide), 3).astype(object)
            ty = "REAL"
        elif kind == 2:
            v = np.array([f"c{int(x)}" for x in np.clip(np.round(base + rng.normal(size=n_wide)) + 3, 0, 6)], dtype=object)
            ty = "TEXT"
        elif kind == 3:
            v = (base + rng.normal(size=n_wide) > 0).astype(int).astype(object)
            ty = "INTEGER"
        else:                                       # a code from a code list
            r = k % n_refs
            n_codes = counts[f"ref_{r}"]
            v = np.array([f"R{r}C{int(i)}" for i in np.minimum(rng.zipf(1.5, size=n_wide) - 1, n_codes - 1)], dtype=object)
            ty = "TEXT"
            if k < 5 * n_refs:                      # each code list referenced once
                wfks.append(([f"w{k:03d}"], f"ref_{r}", ["code"]))
        if kind != 4:
            v[rng.random(n_wide) < rng.uniform(0, 0.3)] = None
        wcols.append((f"w{k:03d}", ty, kind == 4))
        cols_data.append(v)
    add("wide_profile", wcols, [list(r) for r in zip(*[[x.item() if hasattr(x, "item") else x for x in c]
                                                       for c in cols_data])], pk=["id"], fks=wfks)

    # ------------------------------------------------------------ facts
    per_fact = max(100, rows // n_facts)
    for j in range(n_facts):
        bs = [(j + i) % branches for i in range(3)]
        fk_vals = []
        for i, b in enumerate(bs):
            n = leaf_rows[b]
            if i == 0:                              # the skewed link: Zipf ranks over a shuffled leaf
                order = rng.permutation(n) + 1
                ranks = np.minimum(rng.zipf(skew, size=per_fact), n) - 1
                fk_vals.append(order[ranks])
            else:
                fk_vals.append(rng.integers(1, n + 1, size=per_fact))
        lc = leaf_cat[bs[0]][fk_vals[0] - 1]
        effect = np.select([lc == "alpha", lc == "beta"], [3.0, 1.5], 0.5)
        amount = np.round(rng.lognormal(effect, 0.5), 2)
        amount[rng.random(per_fact) < 0.002] *= 1000   # outliers
        day = np.datetime64("2020-01-01") + rng.integers(0, 1461, size=per_fact).astype("timedelta64[D]")
        status = rng.choice(["ok"] * 50 + ["late"] * 8 + ["cancelled"] * 2 + [f"rare_{i}" for i in range(40)],
                            size=per_fact)
        cols = [("id", "INTEGER", True)] + [(f"b{b}_id", "INTEGER", True) for b in bs] + \
            [("day", "DATE", True), ("amount", "REAL", True), ("quantity", "INTEGER", True), ("status", "TEXT", True)]
        qty = rng.poisson(2, size=per_fact) + 1
        add(f"fact_{j}", cols,
            [[i + 1, *(int(f[i]) for f in fk_vals), str(day[i]), float(amount[i]), int(qty[i]), str(status[i])]
             for i in range(per_fact)],
            pk=["id"], fks=[([f"b{b}_id"], f"b{b}_l1", ["id"]) for b in bs])
    con.commit()
    con.close()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--tables", type=int, default=120)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--wide", type=int, default=500)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--branches", type=int, default=0)
    ap.add_argument("--skew", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    counts = build(a.out, a.tables, a.depth, a.wide, a.rows, a.branches, a.skew, a.seed)
    print(f"{len(counts)} tables, {sum(counts.values())} rows")


if __name__ == "__main__":
    main()
