# SPDX-License-Identifier: Apache-2.0
"""Document shape metrics with limits calibrated on real-versus-real splits.

Per collection, between two document sets A and B:

    presence      mean |difference| of the presence rate of each field path
                  (present / parent objects), over the paths of either set
    presence_max  the largest such difference
    type_mix      mean total variation distance (TVD) of the state of each path
                  that takes more than one state (absent, null, each type)
    array_length  mean TVD of the length histograms of the array and map paths
    signature     TVD of the field-set signatures of the documents (which
                  top-level fields a document holds together: the L-equivalence
                  classes of Baazizi et al.)
    depth         TVD of the nesting depth of the documents
    size          TVD of the number of scalar values per document, in log2 bins

and two exact properties that are not calibrated: the share of documents that
validate against the inferred schema, and the share of references that
resolve (both must be 1).

A metric passes when it is at most its limit: the one-sided 99% prediction
bound of the metric between two random halves of the real documents, rescaled
to the compared sizes (relsynth.quality.calibrate, Prop. 109; with
``predictive=True`` the variance of a posterior predictive draw, Prop. 1204).
Splits keep whole documents, so the dependence inside a document is respected
(Proposition 1403).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..quality.calibrate import null_summary
from .infer import CollectionProfile, Node, profile_collection
from .values import type_name

SHAPE_METRICS = ("presence", "presence_max", "type_mix", "array_length", "signature", "depth", "size")


def _tvd(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return 0.0 if na == nb else 1.0
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in set(a) | set(b))


def _paths(prof: CollectionProfile) -> dict[str, Node]:
    return {nd.path: nd for nd in prof.root.walk() if nd is not prof.root}


def _state(nd: Node) -> Counter:
    c = Counter({t: k for t, k in nd.types.items()})
    if nd.n > nd.present:
        c["absent"] = nd.n - nd.present
    return c


def _doc_stats(docs) -> tuple[Counter, Counter]:
    depth: Counter = Counter()
    size: Counter = Counter()
    for d in docs:
        dd, ss = _walk(d)
        depth[dd] += 1
        size[int(math.log2(ss + 1))] += 1
    return depth, size


def _walk(v, lvl=0):
    t = type_name(v)
    if t == "object":
        ds, ss = lvl, 0
        for x in v.values():
            d, s = _walk(x, lvl + 1)
            ds, ss = max(ds, d), ss + s
        return ds, ss
    if t == "array":
        ds, ss = lvl, 0
        for x in v:
            d, s = _walk(x, lvl + 1)
            ds, ss = max(ds, d), ss + s
        return ds, ss
    return lvl, 1


def shape_distances(a_docs: list, b_docs: list, name: str = "c", a_prof=None, b_prof=None) -> dict:
    pa = a_prof or profile_collection(name, a_docs)
    pb = b_prof or profile_collection(name, b_docs)
    A, B = _paths(pa), _paths(pb)
    keys = sorted(set(A) | set(B))
    pres, tm, al = [], [], []
    for k in keys:
        x, y = A.get(k), B.get(k)
        ra = x.present / x.n if x is not None and x.n else 0.0
        rb = y.present / y.n if y is not None and y.n else 0.0
        if (x is None or x.n) and (y is None or y.n):
            pres.append(abs(ra - rb))
        # a path one side never holds is absent from all of that side's objects
        sa = _state(x) if x is not None else Counter({"absent": 1})
        sb = _state(y) if y is not None else Counter({"absent": 1})
        if len(set(sa) | set(sb)) > 1:
            tm.append(_tvd(sa, sb))
        if (x is not None and x.lengths) or (y is not None and y.lengths):
            al.append(_tvd(x.lengths if x is not None else Counter(), y.lengths if y is not None else Counter()))
    sig_a = Counter({frozenset(k): v for k, v in pa.root.signatures.items()})
    sig_b = Counter({frozenset(k): v for k, v in pb.root.signatures.items()})
    da, sa_ = _doc_stats(a_docs)
    db, sb_ = _doc_stats(b_docs)
    return {"presence": float(np.mean(pres)) if pres else 0.0,
            "presence_max": float(max(pres)) if pres else 0.0,
            "type_mix": float(np.mean(tm)) if tm else 0.0,
            "array_length": float(np.mean(al)) if al else 0.0,
            "signature": _tvd(sig_a, sig_b),
            "depth": _tvd(da, db),
            "size": _tvd(sa_, sb_)}


def calibrate(real: list, n_other: int, name="c", splits=10, confidence=0.99, seed=0, predictive=False,
              max_docs: int = 20000) -> dict:
    from ..timeseries.metrics import predictive_kappa
    rng = np.random.default_rng(seed)
    docs = real
    if len(docs) > max_docs:
        docs = [docs[i] for i in sorted(rng.choice(len(docs), max_docs, replace=False))]
    N = len(docs)
    draws: dict[str, list] = {}
    for _ in range(splits):
        perm = rng.permutation(N)
        a = [docs[i] for i in perm[: N // 2]]
        b = [docs[i] for i in perm[N // 2:]]
        for k, v in shape_distances(a, b, name).items():
            draws.setdefault(k, []).append(v)
    k = predictive_kappa((len(real), n_other), (N // 2, N - N // 2), predictive)
    return {m: null_summary(v, 0.0, confidence, k, log=True) for m, v in draws.items()}


def shape_checks(real: list, syn: list, name="c", splits=10, confidence=0.99, seed=0, predictive=False,
                 real_prof=None) -> list[dict]:
    d = shape_distances(real, syn, name, a_prof=real_prof)
    lim = calibrate(real, len(syn), name, splits, confidence, seed, predictive)
    out = []
    for m in SHAPE_METRICS:
        v, L = d[m], lim[m]["limit"]
        out.append({"metric": m, "value": round(float(v), 6), "limit": None if L is None else round(float(L), 6),
                    "pass": None if L is None else bool(v <= L + 1e-12), "null": lim[m]})
    return out


def reference_integrity(collections: dict[str, list], references: list[dict]) -> list[dict]:
    """Share of non-null reference values that name an existing target value."""
    from .shred import _get, _ABSENT
    out = []
    for r in references:
        if r.get("candidate"):
            continue
        tgt = {_key(_get(d, r["to_path"].split("."))) for d in collections.get(r["to"], [])}
        n = ok = 0
        for d in collections.get(r["collection"], []):
            for v in _values_at(d, r["path"]):
                if v is None or v is _ABSENT:
                    continue
                n += 1
                ok += _key(v) in tgt
        out.append({"reference": f"{r['collection']}.{r['path']} -> {r['to']}.{r['to_path']}", "values": n,
                    "resolved": ok, "share": ok / n if n else 1.0})
    return out


def _key(v):
    return str(v)


def _values_at(d, path: str):
    """Values at a path with [] (array elements) and {} (map values) steps."""
    import re
    steps = [s for s in re.split(r"\.|(?=\[\])|(?=\{\})", path) if s]
    cur = [d]
    for s in steps:
        nxt = []
        for c in cur:
            if s == "[]":
                if isinstance(c, list):
                    nxt.extend(c)
            elif s == "{}":
                if isinstance(c, dict):
                    nxt.extend(c.values())
            elif isinstance(c, dict) and s in c:
                nxt.append(c[s])
        cur = nxt
    return cur


def path_differences(a_prof: CollectionProfile, b_prof: CollectionProfile, top: int = 10) -> list[dict]:
    """The paths whose presence or state mix differ most (for the report)."""
    A, B = _paths(a_prof), _paths(b_prof)
    rows: list[dict] = []
    for k in sorted(set(A) | set(B)):
        x, y = A.get(k), B.get(k)
        ra = x.present / x.n if x is not None and x.n else 0.0
        rb = y.present / y.n if y is not None and y.n else 0.0
        sa = _state(x) if x is not None else Counter({"absent": 1})
        sb = _state(y) if y is not None else Counter({"absent": 1})
        rows.append({"path": k, "presence": [round(ra, 4), round(rb, 4)], "state_tvd": round(_tvd(sa, sb), 4),
                     "states": [dict(sa), dict(sb)]})
    rows.sort(key=lambda r: -max(abs(r["presence"][0] - r["presence"][1]), r["state_tvd"]))
    return rows[:top]
