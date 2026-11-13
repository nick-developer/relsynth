# SPDX-License-Identifier: Apache-2.0
"""Build a SQLite copy of one small Stack Exchange site's data dump (CC BY-SA 4.0) (G7).

    python stackexchange_sqlite.py SITE.7z OUT.db

Users, Posts, Comments, Votes, Badges, Tags and PostLinks become tables with
their Id as primary key and the dump's references declared as foreign keys
(Posts.ParentId and AcceptedAnswerId -> Posts is a self-reference; deleted
posts and users leave dangling references, kept as they are).  Bodies, titles,
comments and profiles stay: they are the long text of the matrix.  PostHistory
(every revision of every body) is left out.  The XML is parsed without entity
expansion (defusedxml).  Derived data is CC BY-SA 4.0: the harness keeps it in
the ignored cache and publishes aggregates only.
"""
import sqlite3
import sys
from pathlib import Path

import defusedxml.ElementTree as ET
import py7zr
from sqlite_ddl import create_table, insert_rows

src, out = Path(sys.argv[1]), Path(sys.argv[2])
TABLES = {  # table -> foreign keys (column, parent)
    "Users": [],
    "Posts": [("OwnerUserId", "Users"), ("LastEditorUserId", "Users"), ("ParentId", "Posts"),
              ("AcceptedAnswerId", "Posts")],
    "Comments": [("PostId", "Posts"), ("UserId", "Users")],
    "Votes": [("PostId", "Posts"), ("UserId", "Users")],
    "Badges": [("UserId", "Users")],
    "Tags": [("ExcerptPostId", "Posts"), ("WikiPostId", "Posts")],
    "PostLinks": [("PostId", "Posts"), ("RelatedPostId", "Posts")],
}
INTS = {"Id", "PostTypeId", "AcceptedAnswerId", "ParentId", "Score", "ViewCount", "OwnerUserId", "LastEditorUserId",
        "AnswerCount", "CommentCount", "FavoriteCount", "PostId", "UserId", "VoteTypeId", "BountyAmount", "Reputation",
        "Views", "UpVotes", "DownVotes", "AccountId", "Class", "Count", "ExcerptPostId", "WikiPostId",
        "RelatedPostId", "LinkTypeId"}

work = out.parent / (out.stem + "-xml")
work.mkdir(parents=True, exist_ok=True)
with py7zr.SevenZipFile(src) as z:
    z.extract(work, targets=[f"{t}.xml" for t in TABLES])
out.unlink(missing_ok=True)
con = sqlite3.connect(out)
for t, fks in TABLES.items():
    rows = []
    for _, el in ET.iterparse(work / f"{t}.xml", events=("end",)):
        if el.tag == "row":
            rows.append(dict(el.attrib))
            el.clear()
    cols = list(dict.fromkeys(k for r in rows for k in r))
    types = {c: "INTEGER" if c in INTS else ("DATETIME" if c.endswith("Date") else "TEXT") for c in cols}
    fks = [(c, p) for c, p in fks if c in cols]
    create_table(con, t, [(c, types[c], c == "Id") for c in cols], primary_key=["Id"],
                 foreign_keys=[([c], p, ["Id"]) for c, p in fks])

    def val(c, v):
        if v is None:
            return None
        if types[c] == "INTEGER":
            return int(v)
        return v.replace("T", " ")[:19] if types[c] == "DATETIME" else v
    insert_rows(con, t, cols, ([val(c, r.get(c)) for c in cols] for r in rows))
    print(f"{t}: {len(rows)} rows")
con.commit()
con.close()
for f in work.glob("*.xml"):
    f.unlink()
work.rmdir()
