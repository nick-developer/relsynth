# SPDX-License-Identifier: Apache-2.0
"""v0.9 checks of data-model identification (keyfind, introspect, discovery, profiling).

Each test returns [(what, value, target, ok), ...].  Run all of them with
    python3 -P checks/v09_ident.py            or one:  python3 -P checks/v09_ident.py t_pii_anchors
The failing cases are those of the v0.9 review: a code column taken for a hierarchy,
short string codes taken for links, a measure taken for a link on its name, a
composite key decided on a sample, self-references skipped in sampled tables, PII
words matched inside other words, quotes dropped from rule literals; and the
identity of the faster profiling kernels with the code they replace.
"""
from __future__ import annotations

import ast
import itertools
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fakedata"))
os.environ.setdefault("RELSYNTH_QUIET", "1")

CHINOOK = "/tmp/claude-0/-home-claude/bb3ce53c-120b-56d9-8782-51605abd2883/scratchpad/chinook/Chinook.db"


def _db(tables: dict, ddl: dict | None = None) -> str:
    db = Path(tempfile.mkdtemp()) / "t.db"
    con = sqlite3.connect(db)
    for t, df in tables.items():
        if ddl and t in ddl:
            con.execute(ddl[t])
            df.to_sql(t, con, index=False, if_exists="append")
        else:
            df.to_sql(t, con, index=False)
    con.commit()
    con.close()
    return str(db)


def _links(sp):
    return sorted((r["child"], r["child_key"], r["parent"]) for r in sp["relationships"])


class _P:
    """The few attributes KeyFinder reads, for whole tables in memory."""

    def __init__(self, tables):
        self.df = tables
        self.names = list(tables)
        self.whole = {t: True for t in tables}
        self.src = type("Src", (), {"meta": {t: {"binary": []} for t in tables}})()


# ------------------------------------------------------------------ S1 forest
def t_forest_code_column():
    """A code 1..3 with one null in 1,000 rows is never a self-reference, nor asked."""
    from relsynth.keyfind import KeyFinder, forest_chance
    rng = np.random.default_rng(0)
    acc = ask = 0
    R = 60
    for _ in range(R):
        df = pd.DataFrame({"id": np.arange(1, 1001), "ptype": rng.integers(1, 4, 1000).astype(float)})
        df.loc[rng.choice(1000, 1), "ptype"] = np.nan
        best, weak = KeyFinder(_P({"T": df}), "data").best_parent("T", "ptype", [("T", "id")])
        acc += best is not None
        ask += bool(weak)
    return [("code column 1..3 accepted as a self-reference", f"{acc}/{R}", "0", acc == 0),
            ("code column 1..3 asked about", f"{ask}/{R}", "0", ask == 0),
            ("unconditional forest_chance(1000, 1) (kept as a bound, no longer decides)",
             round(forest_chance(1000, 1), 4), "0.0054", abs(forest_chance(1000, 1) - 0.00542) < 1e-4)]


def t_forest_perm_test():
    """The permutation test: its forest check equals is_forest, its p-value is
    (1 + hits) / (B + 1) with early stopping, and real hierarchies are accepted or
    asked."""
    from relsynth.keyfind import (FOREST_PERMS, _forest_rows, forest_perm_test, is_forest, KeyFinder)
    rng = np.random.default_rng(1)
    same = True
    for _ in range(2000):
        n = int(rng.integers(1, 30))
        ptr = [None if rng.random() < 0.1 else int(x) for x in rng.integers(0, n + 2, n)]   # some outside
        key = list(range(n))
        at = {k: i for i, k in enumerate(key)}
        pos = np.array([-1 if p is None else at.get(p, -1) for p in ptr], dtype=np.int64)
        same &= _forest_rows(pos, np.arange(n), max(1, int(np.ceil(np.log2(max(n, 2)))))) == is_forest(ptr, key)
    out = [("pointer-doubling forest check == is_forest (2,000 random pointer sets)", bool(same), True, bool(same))]
    # a balanced hierarchy of 2,000 rows with one top: significant
    n = 2000
    ptr = [None] + [int(1 + (i - 1) // 4) for i in range(2, n + 1)]
    p, hits, drawn = forest_perm_test(ptr, list(range(1, n + 1)))
    out.append(("balanced tree of 2,000 rows: permutation p", p, "<= 0.01", p <= 0.01 and drawn == FOREST_PERMS))
    # Chinook's 8 employees (one top): not significant, asked
    e = pd.DataFrame({"EmployeeId": range(1, 9), "ReportsTo": [None, 1, 2, 2, 1, 1, 6, 6]})
    e["ReportsTo"] = e["ReportsTo"].astype(float)
    best, weak = KeyFinder(_P({"Employee": e}), "data").best_parent("Employee", "ReportsTo", [("Employee", "EmployeeId")])
    out.append(("Chinook Employee.ReportsTo (8 rows): asked, not accepted", (best is None, len(weak)), "(True, 1)",
                best is None and len(weak) == 1))
    # the tree scenario's manager_id: accepted or asked (permutation p about 0.02)
    import scenarios as S
    T = S.tree()[0]
    kf = KeyFinder(_P({"employee": T["employee"]}), "data")
    best, weak = kf.best_parent("employee", "manager_id", [("employee", "employee_id")])
    out.append(("tree scenario manager_id: found or asked", "found" if best else ("asked" if weak else "lost"),
                "found or asked", bool(best) or bool(weak)))
    return out


def t_selfref_sampled():
    """A self-reference in a sampled table is tested on the whole table."""
    import scenarios as S
    from relsynth.introspect import introspect
    T = S.tree()[0]
    db = _db(T)
    out = []
    for n in (20000, 2000):
        sp = introspect({"kind": "sqlite", "path": db}, sample_rows=n, declared=False)
        found = ("employee", "manager_id", "employee") in _links(sp)
        asked = any(x.startswith("ASK employee.manager_id") for x in sp["review"])
        out.append((f"sample_rows={n} (employee has {len(T['employee'])} rows): manager_id",
                    "found" if found else ("asked" if asked else "skipped"), "found or asked", found or asked))
    return out


# ------------------------------------------------------------------ S2 strings
def t_string_chance():
    from relsynth.keyfind import chance_log10, string_chance_log10
    rng = np.random.default_rng(2)
    cur = [a + b + c for a in "ABCDEFGHIJ" for b in "ABCDEFGHIJ" for c in "ABCDEFGH"][:105]
    out = [("Section {A,B,C} in grades A-F", round(chance_log10(np.array(list("ABC"), dtype=object),
                                                                 np.array(list("ABCDEF"), dtype=object)), 2),
            "> -6 (no evidence)", chance_log10(np.array(list("ABC"), dtype=object), np.array(list("ABCDEF"), dtype=object)) > -6),
           ("10 currency codes in 105 three-letter codes", round(string_chance_log10(cur[:10], cur), 1), "<= -6",
            string_chance_log10(cur[:10], cur) <= -6),
           ("30 order numbers SO43697.. among 30,000", round(string_chance_log10(
               [f"SO{43697 + i}" for i in range(0, 3000, 100)], [f"SO{43697 + i}" for i in range(30000)]), 1),
            "<= -100", string_chance_log10([f"SO{43697 + i}" for i in range(0, 3000, 100)],
                                           [f"SO{43697 + i}" for i in range(30000)]) <= -100),
           ("two distinct strings (Y/N) carry no evidence", string_chance_log10(["N", "Y"], ["N", "Y"]), 0.0,
            string_chance_log10(["N", "Y"], ["N", "Y"]) == 0.0)]
    # exact against the simulated null: 4 random distinct codes of 2 capitals among 676, 60 of them keys
    space = [a + b for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    keys = sorted(rng.choice(space, 300, replace=False))
    ks = set(keys)
    sims = np.array([all(x in ks for x in rng.choice(space, 4, replace=False)) for _ in range(40000)])
    exact = 10 ** string_chance_log10(keys[:4], keys)
    se = np.sqrt(exact * (1 - exact) / len(sims))
    out.append(("string chance exact vs simulated null (v=4, k=300, w=676)", round(exact, 4),
                f"{sims.mean():.4f} +- {4 * se:.4f}", abs(exact - sims.mean()) <= 4 * se))
    # end to end: Section and ShirtSize of students are no links from values alone
    grade = pd.DataFrame({"GradeCode": list("ABCDEF"), "Points": [4.0, 3.5, 3.0, 2.0, 1.0, 0.0]})
    size = pd.DataFrame({"SizeCode": ["S", "M", "L", "XL"], "Label": ["small", "medium", "large", "x-large"]})
    st = pd.DataFrame({"StudentId": np.arange(1, 501), "Section": rng.choice(list("ABC"), 500),
                       "ShirtSize": rng.choice(["S", "M", "L"], 500), "Age": rng.integers(18, 30, 500)})
    from relsynth.introspect import introspect
    db = _db({"Grade": grade, "Size": size, "Student": st})
    for ev in ("names+data", "data"):
        got = _links(introspect({"kind": "sqlite", "path": db}, declared=False, evidence=ev))
        out.append((f"{ev}: Student.Section -> Grade", ("Student", "Section", "Grade") in got, False,
                    ("Student", "Section", "Grade") not in got))
    return out


# ------------------------------------------------------------------ S3, S6 names need values
def t_names_need_values():
    from relsynth.introspect import introspect
    from relsynth.keyfind import r_crit, randomness
    rng = np.random.default_rng(3)
    N = 1000
    orders = pd.DataFrame({"OrderKey": np.arange(1, N + 1), "Amount": rng.gamma(2, 50, N).round(2)})
    cust = pd.DataFrame({"CustomerKey": np.arange(1, 301), "CustomerName": [f"c{i}" for i in range(300)]})
    lines = pd.DataFrame({"LineKey": np.arange(1, 5001), "OrderKey": rng.integers(1, N + 1, 5000),
                          "OrderQty": rng.integers(1, 21, 5000), "Price": rng.gamma(2, 10, 5000).round(2)})
    got = _links(introspect({"kind": "sqlite", "path": _db({"Orders": orders, "Customer": cust, "OrderLine": lines})},
                            declared=False))
    out = [("OrderLine.OrderQty -> Orders (a measure named after the table)", ("OrderLine", "OrderQty", "Orders") in got,
            False, ("OrderLine", "OrderQty", "Orders") not in got),
           ("OrderLine.OrderKey -> Orders kept", ("OrderLine", "OrderKey", "Orders") in got, True,
            ("OrderLine", "OrderKey", "Orders") in got),
           ("R_crit(v) = min(0.5, 1 - 2/sqrt(v)) at v = 4, 25, 100", [r_crit(4), r_crit(25), r_crit(100)],
            [0.0, 0.5, 0.5], [r_crit(4), r_crit(25), r_crit(100)] == [0.0, 0.5, 0.5])]
    # R_crit is below the 0.1% point of R for v random keys of 1,000
    K = np.arange(1, 1001)
    ok = []
    for v in (5, 10, 20, 50):
        q = np.quantile([randomness(np.sort(rng.choice(K, v, replace=False)), K) for _ in range(3000)], 0.001)
        ok.append((v, round(float(q), 3), round(r_crit(v), 3), bool(q >= r_crit(v))))
    out.append(("0.1% quantile of R for v random keys >= R_crit(v)", ok, "all True", all(x[3] for x in ok)))
    # every declared Chinook link passes the name guard
    if os.path.exists(CHINOOK):
        con = sqlite3.connect(CHINOOK)
        bad = []
        for (t,) in con.execute("select name from sqlite_master where type='table'"):
            for row in con.execute(f"pragma foreign_key_list('{t}')"):
                p, c, pc = row[2], row[3], row[4]
                a = pd.read_sql(f'select "{c}" from "{t}"', con)[c].dropna().astype(np.int64).unique()
                k = np.sort(pd.read_sql(f'select "{pc}" from "{p}"', con)[pc].astype(np.int64).unique())
                a = np.sort(a[np.isin(a, k)])
                if randomness(a, k) < r_crit(len(a)):
                    bad.append(f"{t}.{c}")
        out.append(("Chinook declared links below R_crit", bad, [], not bad))
    return out


# ------------------------------------------------------------------ S4 composite keys
def t_composite_key_whole_table():
    """A pair unique on the sample but repeated in the whole table is not a key."""
    from relsynth.discovery import Discovery
    from relsynth.source import Source
    rng = np.random.default_rng(4)
    pairs = pd.DataFrame(list(itertools.product(range(200), range(100))), columns=["a", "b"]).sample(19000, random_state=1)
    full = pd.concat([pairs, pairs.iloc[:5]], ignore_index=True)          # 5 repeated pairs
    full["v"] = rng.gamma(2, 10, len(full)).round(2)
    d = Path(tempfile.mkdtemp())
    full.to_csv(d / "R.csv", index=False)
    clean = full.iloc[:19000]
    clean.to_csv(d / "C.csv", index=False)
    src = Source({"kind": "csv", "path": str(d)})
    out = []
    for t, df, want in (("R", full, None), ("C", clean, ["a", "b"])):
        smp = df.sample(2000, random_state=3).reset_index(drop=True)
        assert not smp[["a", "b"]].duplicated().any()
        obj = type("Obj", (), {})()
        obj.df, obj.whole, obj.src = {t: smp}, {t: False}, src
        got = Discovery.composite_key(obj, t)
        out.append((f"{'with' if want is None else 'without'} repeats outside the sample: composite key", got, want,
                    got == want))
    # grains(): a fact whose (CustomerKey, ProductKey) repeats outside the sample
    from relsynth.introspect import introspect
    cust = pd.DataFrame({"CustomerKey": np.arange(200), "Name": [f"c{i}" for i in range(200)]})
    prod = pd.DataFrame({"ProductKey": np.arange(100), "Label": [f"p{i}" for i in range(100)]})
    sales = full[["a", "b", "v"]].rename(columns={"a": "CustomerKey", "b": "ProductKey", "v": "Amount"})
    sales.insert(0, "SalesKey", np.arange(len(sales)))
    ddl = {"Sales": "create table Sales (SalesKey integer primary key, CustomerKey integer references Customer(CustomerKey), "
                    "ProductKey integer references Product(ProductKey), Amount real)",
           "Customer": "create table Customer (CustomerKey integer primary key, Name text)",
           "Product": "create table Product (ProductKey integer primary key, Label text)"}
    sp = introspect({"kind": "sqlite", "path": _db({"Customer": cust, "Product": prod, "Sales": sales}, ddl)},
                    sample_rows=2000)
    uq = sp["tables"]["Sales"].get("unique", [])
    rej = any("repeats in the whole table" in x for x in sp["review"])
    out.append(("grains: [CustomerKey, ProductKey] with 5 repeated pairs among 19,005 rows",
                (uq, "rejected" if rej else "not proposed"), "not a unique key", ["CustomerKey", "ProductKey"] not in uq))
    return out


# ------------------------------------------------------------------ B1, B2, B3
def t_pii_anchors():
    from relsynth.introspect import PII_TOKENS, snake
    neg = ["Elevation", "IsPrivate", "ReservationCount", "Derivative", "Fairfax", "Motivation", "ZipperCount", "Intel",
           "Hotel"]
    pos = ["EmailAddress", "Email2", "Phone", "MobilePhone", "FirstName", "ContactName", "AddressLine1", "PostalCode",
           "ZipCode", "Zipcode", "VatNumber", "TaxId", "CardNumber", "EmergencyContactPhone", "Fax", "SSN", "IBAN",
           "Telephone", "AccountNumber", "PassportNo"]
    hit = lambda nm: [g for g, p in PII_TOKENS.items() if p.search(snake(nm))]
    fp = [n for n in neg if hit(n)]
    fn = [n for n in pos if not hit(n)]
    return [("PII words found inside other words", fp, [], not fp), ("PII names missed", fn, [], not fn)]


def t_lit_quotes():
    from relsynth.discovery import _lit
    vals = ["O'Brien", "''", 'say "hi"', "back\\slash", "Zürich", "plain"]
    back = [ast.literal_eval(_lit(v)) for v in vals]
    return [("rule literals parse back to the value", back == vals, True, back == vals),
            ("numbers and booleans unchanged", [_lit(3), _lit(2.0), _lit(0.25), _lit(True)], ["3", "2", "0.25", "True"],
             [_lit(3), _lit(2.0), _lit(0.25), _lit(True)] == ["3", "2", "0.25", "True"])]


def t_subsample():
    from relsynth.discovery import _subsample
    df = pd.DataFrame({"k": np.arange(50000)})
    a, b = _subsample(df, 5000), _subsample(df, 5000)
    small = _subsample(df.iloc[:100], 5000)
    return [("seeded: the same rows twice", a.equals(b), True, a.equals(b)),
            ("not the first rows (mean key of the subsample)", round(float(a.k.mean())), "about 25,000",
             abs(a.k.mean() - 25000) < 1500),
            ("rows kept in order", bool(a.k.is_monotonic_increasing), True, bool(a.k.is_monotonic_increasing)),
            ("a small table is kept whole", small.equals(df.iloc[:100].reset_index(drop=True)), True,
             small.equals(df.iloc[:100].reset_index(drop=True)))]


# ------------------------------------------------------------------ performance: identity
def _fd_rows_ref(x, y, rng):
    """fd_rows as in v0.8 (value_counts)."""
    kx = pd.factorize(pd.Series(x, dtype=object), use_na_sentinel=False)[0]
    ky = pd.factorize(pd.Series(y, dtype=object), use_na_sentinel=False)[0]
    n = len(kx)
    if n == 0:
        return 0.0, 1.0, 0

    def agree(a, b):
        ct = pd.DataFrame({"a": a, "b": b}).value_counts()
        return float(ct.groupby(level=0).max().sum()) / n
    return agree(kx, ky), float(np.mean([agree(kx, ky[rng.permutation(n)]) for _ in range(3)])), n


def t_identity_fd_rows():
    from relsynth.discovery import fd_rows
    rng = np.random.default_rng(5)
    same = True
    for Lx, Ly, n in ((20, 5, 3000), (500, 30, 3000), (3000, 3000, 3000), (3, 2, 50), (1, 1, 10)):
        x = rng.integers(0, Lx, n).astype(object)
        y = (np.asarray(x, dtype=int) % Ly).astype(object) if Ly < Lx else rng.integers(0, Ly, n).astype(object)
        x[:7] = None
        y[3:9] = "s"
        r1, r2 = np.random.default_rng(1), np.random.default_rng(1)
        same &= fd_rows(x, y, r1) == _fd_rows_ref(x, y, r2) and r1.random() == r2.random()
    same &= fd_rows(np.array([], dtype=object), np.array([], dtype=object), rng) == (0.0, 1.0, 0)
    return [("fd_rows (sorted cells) == v0.8 value_counts, results and random stream", bool(same), True, bool(same))]


def t_identity_add_months():
    from relsynth.discovery import _add_months
    rng = np.random.default_rng(6)
    xs = pd.Series(pd.Timestamp("1950-01-01") + pd.to_timedelta(rng.integers(0, 25000, 5000), unit="D"))
    xs = pd.concat([xs, pd.Series(pd.to_datetime(["2000-02-29", "2004-02-29", "2001-01-31", "1999-12-31",
                                                   "2003-03-31"]))], ignore_index=True)
    out = []
    for unit, mk, top in (("years", 12, 70), ("months", 1, 900)):
        k = rng.integers(0, top, len(xs))
        ref = pd.DatetimeIndex([a + pd.DateOffset(**{unit: int(j)}) for a, j in zip(xs, k, strict=True)])
        new = pd.DatetimeIndex(_add_months(xs, k * mk))
        out.append((f"vectorised + DateOffset({unit}=k) == per-row DateOffset", bool(ref.equals(new)), True,
                    bool(ref.equals(new))))
    return out


def _alg_pairs_ref(df, cols, dec, rng, screen=2000):
    """The two-column rules of algebraic_rules as in v0.8 (one holds() per target, pair, operation)."""
    from relsynth.profiling import MIN_ROWS, MIN_SHARE, _share, significant
    X = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cols}
    cols = [c for c in cols if np.isfinite(X[c]).sum() >= MIN_ROWS and len(np.unique(X[c][np.isfinite(X[c])])) >= 3]
    n = len(df)
    idx_s = np.arange(n) if n <= screen else rng.choice(n, screen, replace=False)
    perm = rng.permutation(n)
    out = []
    for c in cols:
        y = X[c]
        d = dec.get(c)
        tol = (10.0 ** -d if d else 0.0) + 1e-9 * np.abs(y) + 1e-9
        for a, b in itertools.combinations([a for a in cols if a != c], 2):
            xa, xb = X[a], X[b]
            for expr, f, fp in ((f"{a} + {b}", xa + xb, xa + xb[perm]), (f"{a} - {b}", xa - xb, xa - xb[perm]),
                                (f"{b} - {a}", xb - xa, xb - xa[perm]), (f"{a} * {b}", xa * xb, xa * xb[perm])):
                if _share(y[idx_s], f[idx_s], tol[idx_s])[0] < MIN_SHARE:
                    continue
                share, n_ = _share(y, f, tol)
                if share >= MIN_SHARE and significant(share, _share(y, fp, tol)[0], n_):
                    out.append((c, expr, [a, b], 2, float(10.0 ** -dec[c]) if dec.get(c) else 0.0))
    return out


def t_identity_algebraic_rules():
    from relsynth.profiling import algebraic_rules, decimals
    rng = np.random.default_rng(7)
    n = 6000
    df = pd.DataFrame({f"c{i}": rng.gamma(2, 10, n).round(2) for i in range(12)})
    df["c3"] = (df.c1 + df.c2).round(2)
    df["c5"] = (df.c7 - df.c9).round(2)
    df["c6"] = (df.c8 * df.c10).round(2)
    df.loc[rng.choice(n, 40), "c5"] = np.nan
    df.loc[rng.choice(n, 3), "c3"] += 1.0                  # 3 violations: still >= 99.9%
    dec = {c: decimals(df[c]) for c in df}
    r1, r2 = np.random.default_rng(0), np.random.default_rng(0)
    new = [x for x in algebraic_rules(df, list(df), dec, r1) if x[3] == 2]
    ref = _alg_pairs_ref(df, list(df), dec, r2)
    same = new == ref and r1.random() == r2.random()
    return [("algebraic_rules pair rules (one screen per pair) == v0.8 loop, and random stream", bool(same), True,
             bool(same)), ("pair rules found", [x[:2] for x in new], "c3, c5, c6", len(new) >= 3)]


def t_identity_cross_codes():
    """cross_rules (b): set sizes and equality from shared integer codes equal the
    object-array computation (Python equality: 1 == 1.0, None)."""
    from relsynth.profiling import canon
    rng = np.random.default_rng(8)
    n = 3000
    cols = []
    for L in (3, 10, 200, 3000):
        v = canon(pd.Series(rng.choice([f"v{i}" for i in range(L)], n).astype(object)))
        v[rng.random(n) < 0.05] = None
        cols.append(v)
    cols.append(canon(pd.Series(rng.integers(0, 5, n).astype(float))))
    w = np.asarray(rng.integers(0, 5, n), dtype=object)
    w[:10] = None
    cols.append(w)                                          # ints vs floats of the same values
    allc = pd.factorize(pd.Series(np.concatenate(cols), dtype=object), use_na_sentinel=True)[0]
    K = int(allc.max()) + 1
    C = [allc[i * n:(i + 1) * n] for i in range(len(cols))]
    same = True
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            va, vb = cols[i], cols[j]
            pa = np.array([x is not None for x in va])
            pb = np.array([x is not None for x in vb])
            m = pa & pb
            sa, sb = set(va[m]), set(vb[m])
            ref = (int(m.sum()), len(sa & sb), len(sa), len(sb), len(sa | sb), np.asarray(va == vb, dtype=bool).tolist())
            ca, cb = C[i], C[j]
            m2 = (ca >= 0) & (cb >= 0)
            xa = np.bincount(ca[m2], minlength=K) > 0
            xb = np.bincount(cb[m2], minlength=K) > 0
            new = (int(m2.sum()), int((xa & xb).sum()), int(xa.sum()), int(xb.sum()),
                   int(xa.sum() + xb.sum() - (xa & xb).sum()), (ca == cb).tolist())
            same &= ref == new
    return [("cross-rule pair statistics from integer codes == object arrays", bool(same), True, bool(same))]


def t_identity_aggregates():
    """aggregates() with the per-relation cache finds the planted aggregates (the
    cached value is a pure function of (agg, of, date))."""
    from relsynth.introspect import _Profiler
    rng = np.random.default_rng(0)
    nd, nf = 300, 6000
    dim = pd.DataFrame({"DateKey": np.arange(nd)} | {f"I{i}": rng.integers(0, 40, nd) for i in range(4)})
    fact = pd.DataFrame({"SalesKey": np.arange(nf), "DateKey": rng.integers(0, nd, nf)}
                        | {f"m{i}": rng.gamma(2, 10, nf).round(2) for i in range(5)})
    dim["I0"] = fact.groupby("DateKey").size().reindex(dim.DateKey).fillna(0).to_numpy().astype(int)
    dim["I1"] = fact.groupby("DateKey").m3.max().reindex(dim.DateKey).to_numpy()
    P = _Profiler.__new__(_Profiler)
    P.rng = np.random.default_rng(7)
    P.review, P.rules, P.copy = [], [], set()
    P.df, P.whole, P.pks = {"D": dim, "F": fact}, {"D": True, "F": True}, {"D": "DateKey", "F": "SalesKey"}
    P.role = {"D": "entity", "F": "fact"}
    P.rels = [{"child": "F", "child_key": "DateKey", "parent": "D"}]
    P.cols = {"D": {c: {"type": "id" if c == "DateKey" else "integer"} for c in dim},
              "F": {c: {"type": "id" if "Key" in c else "float"} for c in fact}}
    P.cols["D"]["I1"] = {"type": "float"}
    P.ask = lambda m: P.review.append(m)
    P.aggregates()
    got = sorted((r["column"], r["agg"], r.get("of")) for r in P.rules)
    want = [("I0", "count", None), ("I1", "max", "m3")]
    return [("aggregates found with the cache", got, want, got == want)]


TESTS = [t_forest_code_column, t_forest_perm_test, t_selfref_sampled, t_string_chance, t_names_need_values,
         t_composite_key_whole_table, t_pii_anchors, t_lit_quotes, t_subsample, t_identity_fd_rows,
         t_identity_add_months, t_identity_algebraic_rules, t_identity_cross_codes, t_identity_aggregates]


def main(argv):
    import time
    sel = [t for t in TESTS if not argv or t.__name__ in argv]
    bad = 0
    for t in sel:
        t0 = time.time()
        try:
            items = t()
        except Exception as e:                      # noqa: BLE001
            items = [(f"{type(e).__name__}: {e}", None, None, False)]
        ok = all(i[3] for i in items)
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {t.__name__} ({time.time() - t0:.1f}s)")
        for what, value, target, good in items:
            print(f"   {'ok ' if good else 'BAD'} {what}: {value} (target {target})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
