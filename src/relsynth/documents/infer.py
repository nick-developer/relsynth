# SPDX-License-Identifier: Apache-2.0
"""Parametric schema inference with counts for document collections.

One pass over the documents of each collection builds a tree of `Node`s, one
per field path, in the style of Baazizi, Colazzo, Ghelli and Sartiani
("Parametric schema inference for massive JSON datasets", VLDB Journal 28,
2019; EDBT 2017).  Each node counts

* how often its parent object occurred (`n`) and how often the field was
  there (`present`): a field with ``present < n`` is optional;
* each type it took (`types`, null included): a field with several types is a
  union, typed by counts;
* for objects, the nested fields and the field-set signatures (the L-
  equivalence classes of the paper) with their counts;
* for arrays, the histogram of lengths (the array length law) and one node for
  the elements, merged over every element (K-equivalence);
* for scalars, the number of distinct values (kept exactly up to a cap), the
  range and the stored precision.

Three facts are then derived from the counts, each only when it holds on every
document seen (so the inferred schema describes every source document,
Proposition 1400):

* variants: a string field whose value decides which optional sibling fields
  are present (a discriminator, `payment.method`);
* length fields: an integer field equal to the length of a sibling array on
  every document (`transaction_count == len(transactions)`);
* references: a scalar field (or the elements of an array of scalars) whose
  values are all values of a unique field of another collection (inclusion
  dependency with compatible types, as in relational key discovery).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .values import NUMERIC, type_name

FULL_PRECISION = 15             # decimals at or above this: keep full precision (no rounding)
DISTINCT_CAP = 250_000          # exact distinct values kept per scalar path
DISCRIMINATOR_MAX_VALUES = 32
DISCRIMINATOR_MIN_SUPPORT = 2   # documents per discriminator value
MAP_MIN_KEYS = 50              # distinct field names that make an object a map ...
MAP_MIN_ID_KEYS = 10           # ... or fewer when the names look like identifiers
_HEX24 = re.compile(r"^[0-9a-fA-F]{24}$")
_ID_KEY = re.compile(r"^([0-9a-fA-F-]{8,}|\d+)$")


@dataclass
class Node:
    """Counts for one field path (or the elements of an array)."""
    path: str
    n: int = 0                                  # parent objects seen (opportunities)
    present: int = 0
    types: Counter = field(default_factory=Counter)
    # objects
    fields: dict[str, Node] = field(default_factory=dict)
    signatures: Counter = field(default_factory=Counter)
    _cand: dict = field(default_factory=dict, repr=False)     # discriminator candidate -> Counter((value, sig))
    _leneq: dict = field(default_factory=dict, repr=False)    # (int field, array field) -> still equal
    _keyeq: dict = field(default_factory=dict, repr=False)    # field f -> every object value has v[f] == key
    # arrays
    lengths: Counter = field(default_factory=Counter)
    items: Node | None = None
    distinct_items: bool = True                 # no array repeats a scalar element (set semantics)
    # scalars
    distinct: set = field(default_factory=set, repr=False)
    overflow: bool = False
    vmin: Any = None
    vmax: Any = None
    decimals: int = 0
    max_len: int = 0
    hex24: int = 0
    # derived
    variants: dict | None = None
    is_map: bool = False                        # an object with data-dependent field names
    map_keys: list = field(default_factory=list, repr=False)
    key_field: str | None = None                # map: the key equals this field of each value
    length_of: str | None = None
    reference: dict | None = None

    # ------------------------------------------------------------ queries
    @property
    def optional(self) -> bool:
        return self.present < self.n

    @property
    def nullable(self) -> bool:
        return self.types.get("null", 0) > 0

    @property
    def value_types(self) -> list[str]:
        """Non-null types, most frequent first (ties by name)."""
        return [t for t, _ in sorted(self.types.items(), key=lambda kv: (-kv[1], kv[0])) if t != "null"]

    @property
    def scalar_types(self) -> list[str]:
        return [t for t in self.value_types if t not in ("object", "array")]

    @property
    def is_object(self) -> bool:
        return "object" in self.types

    @property
    def is_array(self) -> bool:
        return "array" in self.types

    def child(self, name: str) -> Node:
        if name not in self.fields:
            self.fields[name] = Node(f"{self.path}.{name}" if self.path else name)
        return self.fields[name]

    def elements(self) -> Node:
        if self.items is None:
            self.items = Node(self.path + "[]")
        return self.items

    def walk(self):
        yield self
        for f in self.fields.values():
            yield from f.walk()
        if self.items is not None:
            yield from self.items.walk()

    def depth(self) -> int:
        """Nesting depth below this node (a scalar is 0, an object of scalars 1)."""
        d = 0
        if self.fields:
            d = max(d, 1 + max(f.depth() for f in self.fields.values()))
        if self.items is not None:
            d = max(d, 1 + self.items.depth())
        return d

    def length_bounds(self) -> tuple[int, int]:
        return (min(self.lengths), max(self.lengths)) if self.lengths else (0, 0)


# ---------------------------------------------------------------- one pass
def _observe(node: Node, v: Any) -> None:
    t = type_name(v)
    node.types[t] += 1
    if t == "object":
        sig = tuple(v.keys())
        node.signatures[frozenset(sig)] += 1
        for name in node.fields:
            node.fields[name].n += 1
        for name, x in v.items():
            c = node.child(name)
            if c.n < node.types["object"]:          # field first seen now: earlier objects lacked it
                c.n = node.types["object"]
            c.present += 1
            _observe(c, x)
        # discriminator candidates: short strings present here
        fs = frozenset(sig)
        for name, x in v.items():
            if isinstance(x, str) and len(x) <= 64:
                cand = node._cand.get(name, Counter())
                if cand is not None:
                    cand[(x, fs)] += 1
                    if len({val for val, _ in cand}) > DISCRIMINATOR_MAX_VALUES:
                        cand = None
                    node._cand[name] = cand
            elif name in node._cand:
                node._cand[name] = None
        for name in list(node._cand):
            if name not in v:
                node._cand[name] = None
        # field names equal to a field of their (object) value: candidate map key field
        for k, x in v.items():
            if isinstance(x, dict):
                for f, y in x.items():
                    if node._keyeq.get(f, True):
                        node._keyeq[f] = y == k
                for f in list(node._keyeq):
                    if f not in x:
                        node._keyeq[f] = False
            else:
                node._keyeq = {f: False for f in node._keyeq}
        # integer fields equal to the length of a sibling array
        arrays = [(k, len(x)) for k, x in v.items() if isinstance(x, list)]
        ints = [(k, x) for k, x in v.items() if type_name(x) in ("int", "long")]
        first = node.types["object"] == 1
        if first:
            for ki, xi in ints:
                for ka, la in arrays:
                    node._leneq[(ki, ka)] = xi == la
        else:
            for key, ok in list(node._leneq.items()):
                if ok:
                    ki, ka = key
                    xi, xa = v.get(ki), v.get(ka)
                    node._leneq[key] = (isinstance(xa, list) and type_name(xi) in ("int", "long")
                                        and xi == len(xa))
    elif t == "array":
        node.lengths[len(v)] += 1
        if node.distinct_items and len(v) > 1:
            try:
                keys = [(type_name(x), x) for x in v]
                node.distinct_items = len(set(keys)) == len(keys)
            except TypeError:                      # unhashable elements (objects, arrays)
                node.distinct_items = False
        e = node.elements()
        for x in v:
            e.n += 1
            e.present += 1
            _observe(e, x)
    elif t != "null":
        if not node.overflow:
            node.distinct.add(v if t != "date" else v.timestamp())
            if len(node.distinct) > DISTINCT_CAP:
                node.overflow = True
                node.distinct = set()
        if t in NUMERIC or t == "date":
            key = float(v) if t in NUMERIC else v
            if t in NUMERIC and isinstance(key, float) and not math.isfinite(key):
                return
            node.vmin = key if node.vmin is None or key < node.vmin else node.vmin
            node.vmax = key if node.vmax is None or key > node.vmax else node.vmax
            if t in ("double", "decimal"):
                node.decimals = max(node.decimals, _decimals(v))
        elif t == "string":
            node.max_len = max(node.max_len, len(v))
            if _HEX24.match(v):
                node.hex24 += 1


def _decimals(v) -> int:
    s = str(v) if not isinstance(v, float) else repr(v)
    if "e" in s or "E" in s:
        return FULL_PRECISION
    return min(len(s.split(".")[1]) if "." in s else 0, FULL_PRECISION)


@dataclass
class CollectionProfile:
    """The inferred schema of one collection, with its counts."""
    name: str
    root: Node
    documents: int = 0

    def nodes(self):
        return list(self.root.walk())

    def node(self, path: str) -> Node | None:
        for nd in self.root.walk():
            if nd.path == path:
                return nd
        return None


def profile_collection(name: str, docs: Iterable[dict]) -> CollectionProfile:
    root = Node("")
    n = 0
    for d in docs:
        n += 1
        root.n += 1
        root.present += 1
        _observe(root, d)
    prof = CollectionProfile(name, root, n)
    _find_maps(root)
    for nd in root.walk():
        _find_variants(nd)
        _find_length_fields(nd)
    return prof


# ---------------------------------------------------------------- derived facts
def merge(a: Node, b: Node) -> Node:
    """The node of a path that holds what `a` and `b` held (K-equivalence merge)."""
    out = Node(a.path, n=a.n + b.n, present=a.present + b.present, types=a.types + b.types,
               signatures=a.signatures + b.signatures, lengths=a.lengths + b.lengths,
               overflow=a.overflow or b.overflow, decimals=max(a.decimals, b.decimals),
               max_len=max(a.max_len, b.max_len), hex24=a.hex24 + b.hex24)
    if not out.overflow:
        out.distinct = a.distinct | b.distinct
        if len(out.distinct) > DISTINCT_CAP:
            out.overflow, out.distinct = True, set()
    lo = [x for x in (a.vmin, b.vmin) if x is not None]
    hi = [x for x in (a.vmax, b.vmax) if x is not None]
    out.vmin, out.vmax = (min(lo) if lo else None), (max(hi) if hi else None)
    out.distinct_items = a.distinct_items and b.distinct_items
    for k in list(a.fields) + [k for k in b.fields if k not in a.fields]:
        fa, fb = a.fields.get(k), b.fields.get(k)
        f = merge(fa, fb) if fa is not None and fb is not None else (fa or fb)
        assert f is not None
        if fa is None or fb is None:          # absent from the other side's objects
            other = b if fa is None else a
            f = _copy_with_n(f, f.n + other.types.get("object", 0))
        out.fields[k] = f
    if a.items is not None or b.items is not None:
        out.items = merge(a.items, b.items) if a.items is not None and b.items is not None else (a.items or b.items)
    return out


def _copy_with_n(nd: Node, n: int) -> Node:
    import copy
    c = copy.copy(nd)
    c.n = n
    return c


def _repath(nd: Node, path: str) -> None:
    nd.path = path
    for k, f in nd.fields.items():
        _repath(f, f"{path}.{k}")
    if nd.items is not None:
        _repath(nd.items, path + ("{}" if nd.is_map else "[]"))


def _find_maps(node: Node) -> None:
    """Objects whose field names are data (a dictionary keyed by an id): many
    distinct names, each in a small share of the objects.  The values merge into
    one node (`items`) and the number of entries per object is a length law,
    as for an array (Spoth et al., SIGMOD 2021, call these collections)."""
    for f in list(node.fields.values()):
        _find_maps(f)
    if node.items is not None:
        _find_maps(node.items)
    nobj = node.types.get("object", 0)
    names = list(node.fields)
    if not nobj or len(names) < MAP_MIN_ID_KEYS:
        return
    idlike = sum(bool(_ID_KEY.match(k)) for k in names) >= 0.9 * len(names)
    rare = max(nd.present for nd in node.fields.values()) < 0.5 * nobj
    if not ((len(names) >= MAP_MIN_KEYS and rare) or (idlike and rare)):
        return
    vals = None
    for nd in node.fields.values():
        nd = _copy_with_n(nd, nd.present)
        vals = nd if vals is None else merge(vals, nd)
    assert vals is not None
    _repath(vals, node.path + "{}")
    sizes: Counter = Counter()
    for sig, c in node.signatures.items():
        sizes[len(sig)] += c
    node.is_map, node.items, node.fields, node.lengths = True, vals, {}, sizes
    node.map_keys = names[:1000]
    node.key_field = next((f for f, ok in node._keyeq.items() if ok), None)
    node.signatures = Counter()
    node._cand, node._leneq, node._keyeq = {}, {}, {}


def _find_variants(node: Node) -> None:
    """A discriminator: a string field present in every object of `node` whose
    value decides, on every object seen, which optional sibling fields are
    present (at least one field is present for some values and absent for
    others)."""
    if not node.is_object or len(node.signatures) < 2:
        return
    nobj = node.types["object"]
    best = None
    for name, cand in node._cand.items():
        if not cand or node.fields[name].present < nobj or node.fields[name].types.get("string", 0) < nobj:
            continue
        by_value: dict[str, Counter] = {}
        for (val, sig), c in cand.items():
            by_value.setdefault(val, Counter())[sig] += c
        if len(by_value) < 2 or min(sum(c.values()) for c in by_value.values()) < DISCRIMINATOR_MIN_SUPPORT:
            continue
        optional = [f for f, nd in node.fields.items() if f != name and nd.present < nobj]
        decided: list = []
        cases: dict = {}
        for f in optional:
            rates = {}
            for val, sigs in by_value.items():
                tot = sum(sigs.values())
                rates[val] = sum(c for s, c in sigs.items() if f in s) / tot
            if all(r in (0.0, 1.0) for r in rates.values()) and 0 < sum(rates.values()) < len(rates):
                decided.append(f)
                for val, r in rates.items():
                    if r == 1.0:
                        cases.setdefault(val, []).append(f)
        if not decided:
            continue
        score = (len(decided), -len(by_value))
        if best is None or score > best[0]:
            best = (score, name, {v: cases.get(v, []) for v in sorted(by_value)}, decided)
    if best is not None:
        _, name, cases, decided = best
        node.variants = {"discriminator": name, "cases": cases, "fields": decided}


def _find_length_fields(node: Node) -> None:
    for (ki, ka), ok in node._leneq.items():
        if ok and node.fields[ki].present == node.fields[ka].present == node.types["object"]:
            node.fields[ki].length_of = ka


def _unique_fields(prof: CollectionProfile, near: float = 1.0) -> list[Node]:
    """Top-level scalar fields present, non-null and distinct in every document
    (in a share `near` of them, for candidates)."""
    out = []
    for nd in prof.root.fields.values():
        st = nd.scalar_types
        if (len(st) == 1 and not nd.optional and not nd.nullable and not nd.overflow
                and len(nd.distinct) >= near * prof.documents and st[0] in ("objectId", "string", "int", "long")):
            out.append(nd)
    out.sort(key=lambda nd: (nd.path != "_id", nd.path))
    return out


def _compatible(a: Node, b: Node) -> bool:
    ta, tb = set(a.scalar_types), set(b.scalar_types)
    if len(ta) != 1:
        return False
    ints = {"int", "long"}
    return ta == tb or (ta <= ints and tb <= ints)


def _stem(s: str) -> str:
    s = s.lower().replace("-", "_")
    s = re.sub(r"(_id|id|_ids|ids|_key|key)$", "", s.split(".")[-1].rstrip("[]"))
    return s[:-1] if s.endswith("s") and len(s) > 3 else s


def find_references(profiles: dict[str, CollectionProfile], declared: list[dict] | None = None,
                    min_distinct: int = 3, candidate_coverage: float = 0.5) -> list[dict]:
    """Inclusion dependencies from scalar paths to unique top-level fields of a
    collection.  A reference whose every source value occurs in the target
    (coverage 1) is taken; one with coverage in [candidate_coverage, 1) is
    returned with ``candidate: true`` (dangling references in the source: the
    user declares it to use it).  Integer targets that are a dense range need a
    name match (an id 1..n contains every small integer).  Declared references
    win and are kept as given."""
    targets = []
    for cname, prof in profiles.items():
        for nd in _unique_fields(prof, near=0.99):
            targets.append((cname, nd))
    refs = [dict(r) for r in (declared or [])]
    taken = {(r["collection"], r["path"]) for r in refs}
    for cname, prof in profiles.items():
        for nd in prof.root.walk():
            if nd is prof.root or nd.overflow or not nd.distinct or (cname, nd.path) in taken:
                continue
            if len(nd.distinct) < min_distinct or nd.is_object or nd.is_array:
                continue
            for tname, tn in targets:
                if tname == cname and tn.path == nd.path:
                    continue
                if not _compatible(nd, tn):
                    continue
                cov = 1.0 if nd.distinct <= tn.distinct else len(nd.distinct & tn.distinct) / len(nd.distinct)
                if cov < candidate_coverage:
                    continue
                st = nd.scalar_types[0]
                name_match = _stem(nd.path) in (_stem(tname), _stem(tn.path)) or _stem(tname) in nd.path.lower()
                if st in ("int", "long") and not name_match:      # integers coincide too easily
                    continue
                if st == "string" and not (name_match or nd.hex24 == nd.present):
                    continue
                if tname == cname and not name_match:
                    continue
                r = {"collection": cname, "path": nd.path, "to": tname, "to_path": tn.path}
                tdocs = profiles[tname].documents
                if cov < 1.0 or len(tn.distinct) < tdocs:
                    r.update(candidate=True, coverage=round(cov, 4))
                    if len(tn.distinct) < tdocs:
                        r["target_duplicates"] = tdocs - len(tn.distinct)
                else:
                    nd.reference = {"collection": tname, "path": tn.path}
                refs.append(r)
                break
    return refs


def find_sequences(collections: dict[str, list[dict]], references: list[dict]) -> list[dict]:
    """Integer top-level fields numbered 1..n (or 0..n-1) within the documents
    that share a reference (an order's line numbers): {collection, path, within, start}."""
    out = []
    for r in references:
        if r.get("candidate") or "." in r["path"] or "[" in r["path"]:
            continue
        docs = collections.get(r["collection"], [])
        if not docs:
            continue
        cands = [k for k, v in docs[0].items() if type_name(v) in ("int", "long") and k != r["path"]]
        for f in cands:
            groups: dict = {}
            ok = True
            for d in docs:
                v = d.get(f)
                if type_name(v) not in ("int", "long"):
                    ok = False
                    break
                groups.setdefault(str(d.get(r["path"])), []).append(v)
            if not ok:
                continue
            for start in (1, 0):
                if all(sorted(g) == list(range(start, start + len(g))) for g in groups.values()) and \
                        any(len(g) > 1 for g in groups.values()):
                    out.append({"collection": r["collection"], "path": f, "within": r["path"], "start": start})
                    break
    return out
