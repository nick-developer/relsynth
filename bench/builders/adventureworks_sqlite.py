# SPDX-License-Identifier: Apache-2.0
"""Build a SQLite copy of AdventureWorksDW from Microsoft's install script folder.

Parses the T-SQL DDL generically (CREATE TABLE, ALTER TABLE ... PRIMARY KEY / UNIQUE /
FOREIGN KEY, including composite keys) and loads the pipe-delimited data files.
varbinary columns are stored as BLOBs (hex decoded) so binary handling is exercised.
"""
import re
import sqlite3
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
ddl = (src / "instawdbdw.sql").read_text(encoding="utf-8-sig", errors="replace")

TYPE = [(r"^(tinyint|smallint|int|bigint|bit)$", "INTEGER"), (r"^(money|smallmoney|decimal|numeric)", "NUMERIC"),
        (r"^(float|real)", "FLOAT"), (r"^datetime", "DATETIME"), (r"^date$", "DATE"), (r"^time", "TIME"),
        (r"^(varbinary|binary|image)", "BLOB"), (r"^xml", "TEXT"), (r"", "TEXT")]


def sqltype(t):
    t = t.lower()
    for pat, s in TYPE:
        if re.search(pat, t):
            if s == "TEXT" and re.match(r"n?(var)?char\]?\((\d+)\)", t):
                return "VARCHAR(%s)" % re.search(r"\((\d+|max)\)", t).group(1).replace("max", "4000")
            return s
    return "TEXT"


tables = {}
for m in re.finditer(r"CREATE TABLE \[dbo\]\.\[(\w+)\]\s*\(((?:(?!CREATE TABLE).)*?)\n\s*\)", ddl, re.S):
    cols = []
    for line in m.group(2).splitlines():
        c = re.match(r"\s*\[(\w+)\]\s*\[(\w+)\]\s*(\([^)]*\))?(.*)", line)
        if c:
            cols.append((c.group(1), sqltype(c.group(2) + (c.group(3) or "").strip()), "NOT NULL" in c.group(4).upper(),
                         c.group(2).lower()))
    tables[m.group(1)] = {"cols": cols, "pk": None, "uq": [], "fk": []}

for m in re.finditer(r"ALTER TABLE \[dbo\]\.\[(\w+)\][^;]*?CONSTRAINT \[\w+\] (PRIMARY KEY|UNIQUE)\s*\w*\s*\((.*?)\)", ddl, re.S):
    cols = re.findall(r"\[(\w+)\]", m.group(3))
    if m.group(2) == "PRIMARY KEY":
        tables[m.group(1)]["pk"] = cols
    else:
        tables[m.group(1)]["uq"].append(cols)
for blk in re.split(r"(?=ALTER TABLE)|;|\nGO\b", ddl):
    a = re.search(r"ALTER TABLE \[dbo\]\.\[(\w+)\]", blk)
    if not a:
        continue
    for m in re.finditer(r"FOREIGN KEY\s*\((.*?)\)\s*REFERENCES \[dbo\]\.\[(\w+)\]\s*\((.*?)\)", blk, re.S):
        tables[a.group(1)]["fk"].append((re.findall(r"\[(\w+)\]", m.group(1)), m.group(2), re.findall(r"\[(\w+)\]", m.group(3))))

if out.exists():
    out.unlink()
con = sqlite3.connect(out)
for t, d in tables.items():
    parts = [f'"{c}" {ty}' + (" NOT NULL" if nn else "") for c, ty, nn, _ in d["cols"]]
    if d["pk"]:
        parts.append("PRIMARY KEY (" + ", ".join(f'"{c}"' for c in d["pk"]) + ")")
    for u in d["uq"]:
        parts.append("UNIQUE (" + ", ".join(f'"{c}"' for c in u) + ")")
    for a, p, b in d["fk"]:
        parts.append("FOREIGN KEY (" + ", ".join(f'"{c}"' for c in a) + f') REFERENCES "{p}" (' + ", ".join(f'"{c}"' for c in b) + ")")
    con.execute(f'CREATE TABLE "{t}" (\n  ' + ",\n  ".join(parts) + "\n)")


def conv(v, ty, raw):
    if v == "":
        return None
    if ty == "INTEGER":
        return int(v)
    if ty in ("NUMERIC", "FLOAT"):
        return float(v)
    if ty == "BLOB":
        try:
            return bytes.fromhex(v[2:] if v.lower().startswith("0x") else v)
        except ValueError:
            return v.encode()
    return v


for t, d in tables.items():
    f = src / f"{t}.csv"
    if not f.exists() or t == "DatabaseLog":   # multi-line TSQL values; out of scope
        print("no data", t)
        continue
    b = f.read_bytes()
    text = b.decode("utf-16") if b[:2] in (b"\xff\xfe", b"\xfe\xff") else b.decode("utf-8-sig", errors="replace")
    sep = "|" if "|" in text.split("\n", 1)[0] or len(d["cols"]) == 1 else "\t"
    rows, n = [], len(d["cols"])
    for line in text.replace("\r\n", "\n").split("\n"):
        if not line:
            continue
        v = line.split(sep)
        if len(v) != n:          # tolerate a trailing separator
            v = (v + [""] * n)[:n]
        rows.append([conv(x, ty, raw) for x, (_, ty, _, raw) in zip(v, d["cols"])])
    con.executemany(f'INSERT INTO "{t}" VALUES ({",".join("?" * n)})', rows)
    print(f"{t:48s} {len(rows):8d} pk={d['pk']} fks={len(d['fk'])}")
con.commit()
print("fk check violations:", con.execute("PRAGMA foreign_key_check").fetchall()[:5])
