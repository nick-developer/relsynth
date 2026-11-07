# SPDX-License-Identifier: Apache-2.0
"""Document acceptance runs (T13): MongoDB sample datasets and a StackExchange dump.

    pip install "relsynth[documents,bench]"     # bench: defusedxml for the XML dump
    python benchmarks/documents/run_documents.py --out docs-out [--mongo-url mongodb://localhost:27017]
           [--sets supplies,analytics,mflix,training,weather,geospatial,airbnb,stackexchange]

For every set: the source collections are read (from the downloaded files,
or loaded into MongoDB and read back through the connector with `$sample` /
`$group` pushdown when --mongo-url is given), synthesised with
`relsynth.documents.DocumentSynthesizer`, and scored.  A set passes when

* every synthetic document validates against the inferred schema,
* every reference resolves,
* every shape metric is within its calibrated limit (14 or more metrics per
  database at 99%: a single failure among them is reported, not hidden).

Data is downloaded, never committed (dataset register: MongoDB sample data
"re-check before committing"; StackExchange CC BY-SA 4.0).  Files are pinned
by SHA-256 where the hash is known; the StackExchange archive's hash is
printed on first use and checked when given with --se-sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import defusedxml.ElementTree as ET  # data dumps from the internet: no entity expansion

from relsynth._http import urlopen
from relsynth.documents import DocumentSynthesizer
from relsynth.documents.values import read_json_documents

MIRROR = "https://raw.githubusercontent.com/neelabalan/mongodb-sample-dataset/main/"
# set -> [(database/collection file, SHA-256, document limit)]
SETS = {
    "supplies": [("sample_supplies/sales.json", "12ee62268abfc724f264ea27d1139a4ff25886851acddeb33ef7b7a7bd652f82", None)],
    "analytics": [
        ("sample_analytics/customers.json", "7fc9ed04b8852b256e95e136ade3681475ae0176c6847dff11207f8b773faafb", None),
        ("sample_analytics/accounts.json", "cb3a611e49ab312b902a07f3da9354eacc079026d44bc21c370f772a0fa6d9a7", None),
        ("sample_analytics/transactions.json", "89dfa0d665faf6a6582a1c6a7543c1034404d4dbe3bbbb2d1f4ae73c8f5f9ee0", None)],
    "mflix": [
        ("sample_mflix/movies.json", "9a7dd8c36e024ce5d3f5975e8d82e6fbc93c7c44139ce58f05412a24b1a24601", 5000),
        ("sample_mflix/users.json", "9a207ab50339261d53f10a4420e2b55c8b23173e2d9ef01acf2654ffc69315c6", None),
        ("sample_mflix/theaters.json", "7245eda3148c0e3f6e71ab879fe510acd8184eeab3cc6a34d3cb1767161a621f", None)],
    "training": [
        ("sample_training/grades.json", "1d63af0c18692dcd4de124bae2b13b1884856a62d99804508db1506c61d4ce64", 20000),
        ("sample_training/zips.json", "405ca38e738ad95144d12ad971ac7b5f23538f121349bbd4b667ca6cd22e0ed5", None)],
    "weather": [("sample_weatherdata/data.json", "c2dd24843ab3aa5e278973b518f828172257a97bcac6bcbf122d18b4ac75a21a", 5000)],
    "geospatial": [
        ("sample_geospatial/shipwrecks.json", "beb424be0728823678212feaa2747ed9cb3cf02d2b546cce90ae6151d696f7c9", None)],
    "airbnb": [("sample_airbnb/listingsAndReviews.json",
                "69eef191f412a755da6765abad572b4407efdbaf2f39ce8ce35c684b7a995540", 1500)],
}
SE_SITES = ("coffee.stackexchange.com", "beer.stackexchange.com", "3dprinting.stackexchange.com")
SE_BASE = "https://archive.org/download/stackexchange/"


def fetch(url: str, dest: Path, sha256: str | None = None) -> Path:
    """Download url to dest once (http(s) only), checking its SHA-256 when given."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=300) as r, open(dest.with_suffix(".part"), "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        dest.with_suffix(".part").rename(dest)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    if sha256 and h != sha256:
        raise SystemExit(f"{dest}: SHA-256 {h} != pinned {sha256}")
    print(f"{dest.name}: sha256 {h}")
    return dest


def mongo_set(name: str, cache: Path) -> dict[str, list[dict]]:
    cols = {}
    for f, sha, limit in SETS[name]:
        p = fetch(MIRROR + f, cache / f, sha)
        docs = []
        for i, d in enumerate(read_json_documents(p)):
            if limit is not None and i >= limit:
                break
            docs.append(d)
        cols[Path(f).stem] = docs
    return cols


# ---------------------------------------------------------------- StackExchange
def _xml_rows(path: Path):
    """The attributes of each <row> element of a StackExchange dump, streamed."""
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "row":
            yield dict(el.attrib)
            el.clear()


def _num(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def stackexchange_set(cache: Path, sha256: str | None = None, max_questions: int = 5000) -> dict[str, list[dict]]:
    """Users, and questions as documents that embed their answers and the
    comments of both (the document design of a Q&A store)."""
    import py7zr
    last = None
    for site in SE_SITES:          # the shape limits are calibrated on the first site: say why it was skipped
        for attempt in range(3):
            try:
                arc = fetch(SE_BASE + site + ".7z", cache / "stackexchange" / (site + ".7z"), sha256)
                break
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"{site}: download attempt {attempt + 1} failed: {type(e).__name__}: {e}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        break
    else:
        raise SystemExit(f"no StackExchange archive could be downloaded: {last}")
    out = cache / "stackexchange" / site
    if not (out / "Posts.xml").exists():
        with py7zr.SevenZipFile(arc) as z:
            z.extract(out, targets=["Posts.xml", "Users.xml", "Comments.xml"])
    ints = ("Id", "PostTypeId", "AcceptedAnswerId", "ParentId", "Score", "ViewCount", "OwnerUserId", "AnswerCount",
            "CommentCount", "FavoriteCount", "UserId", "PostId", "Reputation", "UpVotes", "DownVotes", "Views",
            "AccountId", "LastEditorUserId")
    drop = ("Body", "AboutMe", "Text", "WebsiteUrl", "ProfileImageUrl", "ContentLicense", "OwnerDisplayName",
            "LastEditorDisplayName", "UserDisplayName", "Location", "DisplayName")
    users = []
    for r in _xml_rows(out / "Users.xml"):
        users.append({k: (_num(v) if k in ints else v) for k, v in r.items() if k not in drop})
    comments: dict[int, list] = {}
    for r in _xml_rows(out / "Comments.xml"):
        c = {k: (_num(v) if k in ints else v) for k, v in r.items() if k not in drop and k != "PostId"}
        comments.setdefault(int(r["PostId"]), []).append(c)
    questions, answers = {}, {}
    for r in _xml_rows(out / "Posts.xml"):
        p = {k: (_num(v) if k in ints else v) for k, v in r.items() if k not in drop}
        if "Tags" in p:
            p["Tags"] = [t for t in p["Tags"].replace("|", "><").strip("<>").split("><") if t]
        if p.get("PostTypeId") == 1:
            questions[p["Id"]] = p
        elif p.get("PostTypeId") == 2:
            answers.setdefault(p.get("ParentId"), []).append(p)
    docs = []
    for qid, q in list(questions.items())[:max_questions]:
        q = dict(q)
        q.pop("PostTypeId", None)
        q["comments"] = comments.get(qid, [])
        ans = []
        for a in answers.get(qid, []):
            a = {k: v for k, v in a.items() if k not in ("PostTypeId", "ParentId")}
            a["comments"] = comments.get(a["Id"], [])
            ans.append(a)
        q["answers"] = ans
        docs.append(q)
    return {"users": users, "questions": docs}


# ---------------------------------------------------------------- run
def through_mongo(cols: dict, url: str, db: str) -> tuple[dict, dict]:
    """Load the collections into MongoDB, read them back through the connector,
    and compare the pushed-down counts with the client-side profile."""
    from relsynth.documents.connectors import MongoDB, profile_with_pushdown
    from relsynth.documents.infer import profile_collection
    con = MongoDB({"kind": "mongodb", "url": url, "database": db})
    con.write_all(cols)
    back = con.read_all()
    checks = {}
    for c, docs in back.items():
        full = profile_collection(c, docs)
        prof, _ = profile_with_pushdown(con, c, sample_size=min(2000, len(docs)))
        same = all(prof.root.fields[k].types == full.root.fields[k].types and
                   prof.root.fields[k].lengths == full.root.fields[k].lengths
                   for k in prof.root.fields if k in full.root.fields and "." not in k)
        checks[c] = {"documents": prof.documents == full.documents, "top_level_counts_equal": same}
    return back, checks


def run_set(name: str, cols: dict, out: Path, seed: int, splits: int, mongo: str | None) -> dict:
    t0 = time.time()
    push = None
    if mongo:
        cols, push = through_mongo(cols, mongo, f"relsynth_{name}")
    ds = DocumentSynthesizer(cols, seed=seed, workdir=out / name / "work", name=name)
    syn = ds.synthesize()
    t1 = time.time()
    rep = ds.report(syn, splits=splits)
    res = {"set": name, "seconds": {"synthesis": round(t1 - t0, 1), "report": round(time.time() - t1, 1)},
           "documents": {c: len(d) for c, d in cols.items()},
           "references": rep["references"],
           "candidate_references": [r for r in ds.references if r.get("candidate")],
           "collections": {}, "pushdown": push}
    nfail = 0
    for c, v in rep["collections"].items():
        fails = [m["metric"] for m in v["shape"] if m["pass"] is False]
        nfail += len(fails)
        res["collections"][c] = {"valid": v["validity"]["share"], "repairs": v["repairs"],
                                 "metrics": {m["metric"]: [m["value"], m["limit"]] for m in v["shape"]},
                                 "fails": fails}
    res["shape_failures"] = nfail
    res["metrics"] = sum(len(v["shape"]) for v in rep["collections"].values())
    res["pass"] = (all(v["valid"] == 1.0 for v in res["collections"].values())
                   and all(r["share"] == 1.0 for r in rep["references"]) and nfail == 0
                   and (push is None or all(all(x.values()) for x in push.values())))
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"{name}: {'pass' if res['pass'] else 'FAIL'} ({nfail} of {res['metrics']} shape metrics over the limit)")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs-out")
    ap.add_argument("--cache", default=".cache/documents")
    ap.add_argument("--sets", default="supplies,analytics,mflix,training,weather,geospatial,airbnb,stackexchange")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--mongo-url")
    ap.add_argument("--se-sha256")
    a = ap.parse_args(argv)
    out, cache = Path(a.out), Path(a.cache)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    errors = {}
    for name in a.sets.split(","):
        try:
            cols = stackexchange_set(cache, a.se_sha256) if name == "stackexchange" else mongo_set(name, cache)
            results.append(run_set(name, cols, out, a.seed, a.splits,
                                   a.mongo_url if name != "stackexchange" else None))
        except Exception as e:  # noqa: BLE001 (one set's failure is reported, the others still run)
            traceback.print_exc()
            errors[name] = f"{type(e).__name__}: {e}"
    summary = {"sets": {r["set"]: {"pass": r["pass"], "shape_failures": r["shape_failures"], "metrics": r["metrics"]}
                        for r in results}, "errors": errors}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    # acceptance: validity and references exact everywhere; shape metrics at
    # 99% each, so at most one false failure per 100 metrics is expected
    total = sum(r["metrics"] for r in results)
    fails = sum(r["shape_failures"] for r in results)
    exact = all(all(v["valid"] == 1.0 for v in r["collections"].values()) and
                all(x["share"] == 1.0 for x in r["references"]) for r in results)
    print(f"validity and references exact: {exact}; shape metrics over the limit: {fails} of {total}"
          + (f"; sets that failed to run: {', '.join(errors)}" if errors else ""))
    return 0 if exact and not errors and fails <= max(1, total // 100) else 1


if __name__ == "__main__":
    sys.exit(main())
