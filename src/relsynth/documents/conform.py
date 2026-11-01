# SPDX-License-Identifier: Apache-2.0
"""Make synthetic documents conform to the inferred schema, and count what it took.

The engine keeps every key, every reference and the column types of the
shredded tables; the document structure adds constraints that a table model
does not state: an array's length lies in the source's [min, max], a variant
holds exactly the fields of its case, a field present in every source object
is present.  The engine meets them in almost every document (the element
counts follow the source's length law, the presence of a field is modelled
with the discriminator); `conform` repairs the rest so that every synthetic
document validates (Proposition 1402):

* too many elements: the last ones (by position) are dropped;
* too few elements, a missing required field or a value of a type the field
  never took: the value is taken from another *synthetic* document that has a
  valid value at the same path (a hot deck within the synthetic data, so no
  source value is copied);
* a field that is not part of the document's variant: dropped.

Repairs are counted per kind and path and reported with the quality metrics;
a large count says the model misses a structural dependence (declare it).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .infer import CollectionProfile, Node
from .validate import violations
from .values import type_name


def _pools(node: Node, v: Any, pools: dict, cap: int = 200) -> None:
    """Valid synthetic values per schema node (reservoir of at most cap)."""
    p = pools.setdefault(id(node), [])
    if len(p) < cap and not violations(node, v, "", None, 1):
        p.append(v)
    t = type_name(v)
    if t == "object" and node.is_map and node.items is not None:
        for x in v.values():
            _pools(node.items, x, pools, cap)
    elif t == "object":
        for k, x in v.items():
            if k in node.fields:
                _pools(node.fields[k], x, pools, cap)
    elif t == "array" and node.items is not None:
        ep = pools.setdefault(("elements", id(node)), [])
        for x in v:
            if len(ep) < 50 * cap:
                ep.append(x)
            _pools(node.items, x, pools, cap)


def _donor(node, pools, rng, non_null=False):
    p = pools.get(id(node)) or []
    if non_null:
        p = [x for x in p if x is not None]
    return p[int(rng.integers(len(p)))] if p else _MISSING


class _Missing:
    pass


_MISSING = _Missing()


def _fix(node: Node, v: Any, path: str, pools, rng, counts: Counter) -> Any:
    t = type_name(v)
    if t == "missing":
        d = _donor(node, pools, rng, non_null=True)
        counts[("missing", path)] += 1
        return None if d is _MISSING else d
    if node.types.get(t, 0) == 0:
        d = _donor(node, pools, rng)
        if d is _MISSING:
            return v
        counts[("type", path)] += 1
        return d
    if t == "object" and node.is_map:
        lo, hi = node.length_bounds()
        items = list(v.items())
        if len(items) > hi:
            counts[("length", path)] += 1
            items = items[:hi]
        out = {k: _fix(node.items, x, path + "{}", pools, rng, counts) for k, x in items}
        return out
    if t == "object":
        var = node.variants
        case = set(var["cases"].get(v.get(var["discriminator"]), [])) if var else set()
        nobj = node.types["object"]
        out = {}
        for k, x in v.items():
            f = node.fields.get(k)
            if f is None or (var and k in var["fields"] and k not in case):
                counts[("extra", f"{path}.{k}" if path else k)] += 1
                continue
            out[k] = _fix(f, x, f"{path}.{k}" if path else k, pools, rng, counts)
        for k, f in node.fields.items():
            if k in out:
                continue
            if f.present == nobj or (var and k in var["fields"] and k in case):
                d = _donor(f, pools, rng)
                if d is not _MISSING:
                    counts[("missing", f"{path}.{k}" if path else k)] += 1
                    out[k] = d
        # keep the schema's field order
        order = {k: i for i, k in enumerate(node.fields)}
        out = dict(sorted(out.items(), key=lambda kv: order.get(kv[0], 1 << 30)))
        for k, f in node.fields.items():
            if f.length_of is not None and k in out and isinstance(out.get(f.length_of), list):
                n = len(out[f.length_of])
                if out[k] != n:
                    out[k] = type(out[k])(n) if isinstance(out[k], int) else n
        return out
    if t == "array":
        lo, hi = node.length_bounds()
        items = list(v)
        if len(items) > hi:
            counts[("length", path)] += 1
            items = items[:hi]
        if len(items) < lo:
            counts[("length", path)] += 1
            while len(items) < lo:
                d = _donor(node.items, pools, rng)
                if d is _MISSING:
                    break
                items.append(d)
        items = [_fix(node.items, x, path + "[]", pools, rng, counts) for x in items]
        if node.distinct_items and node.items is not None and len(items) > 1 and _hashable(items):
            items = _distinct(node, items, path, pools, rng, counts)
        return items
    return v


def _hashable(items) -> bool:
    return all(type_name(x) not in ("object", "array") for x in items)


def _distinct(node, items, path, pools, rng, counts):
    """Set semantics: a repeated element is replaced by an element of the same
    path drawn from the other synthetic arrays (weighted by how often they hold
    it) that this array does not hold yet; with none left, it is dropped."""
    pool = [x for x in (pools.get(("elements", id(node))) or []) if type_name(x) not in ("object", "array")]
    seen, out = set(), []
    for x in items:
        k = (type_name(x), x)
        if k not in seen:
            seen.add(k)
            out.append(x)
            continue
        counts[("set", path)] += 1
        cand = [y for y in pool if (type_name(y), y) not in seen]
        if cand:
            y = cand[int(rng.integers(len(cand)))]
            seen.add((type_name(y), y))
            out.append(y)
    lo, _ = node.length_bounds()
    return out if len(out) >= lo else out + items[len(out):lo]


def conform(prof: CollectionProfile, docs: list[dict], seed: int = 0) -> tuple[list[dict], dict]:
    """(conforming documents, {repairs, by_kind, by_path, documents_repaired})."""
    rng = np.random.default_rng([seed, sum(map(ord, prof.name))])
    pools: dict = {}
    for d in docs:
        _pools(prof.root, d, pools)
    out, counts, touched = [], Counter(), 0
    for d in docs:
        before = sum(counts.values())
        idv = d.get("_id", _MISSING)
        nd = _fix(prof.root, d, "", pools, rng, counts)
        if idv is not _MISSING:
            nd["_id"] = idv
            nd = {"_id": idv, **{k: x for k, x in nd.items() if k != "_id"}}
        touched += sum(counts.values()) > before
        out.append(nd)
    by_kind = Counter()
    for (k, _), c in counts.items():
        by_kind[k] += c
    return out, {"repairs": int(sum(counts.values())), "documents_repaired": touched,
                 "by_kind": dict(by_kind),
                 "by_path": {f"{k}:{p}": c for (k, p), c in counts.most_common(20)}}
