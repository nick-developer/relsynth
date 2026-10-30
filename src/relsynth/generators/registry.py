# SPDX-License-Identifier: Apache-2.0
"""Registry of generator plug-ins: discovery, version pinning, compatibility.

Built-in generators (copula, independent, cart) are always registered.
Third-party generators are found through the ``relsynth.generators``
entry-point group (PyPA entry-points specification), so installing a package
that declares, in its pyproject.toml,

    [project.entry-points."relsynth.generators"]
    mygen = "mypackage.module:MyGenerator"

makes ``generator: mygen`` usable in a spec with no change to relsynth.  A
spec can pin a generator's model version (``version: ">=1.2,<2"``); the
registry refuses a generator whose version does not satisfy the pin, whose
contract major version differs from relsynth's, or that cannot model the
table (registry.check).
"""
from __future__ import annotations

import re
from importlib import metadata

from .contract import CONTRACT_VERSION, GeneratorError, TableGenerator

GROUP = "relsynth.generators"
DEFAULT = "copula"


# ---- versions (a small subset of PEP 440: release segments and the usual operators)
def _vtuple(v):
    parts = re.findall(r"\d+", str(v).split("+")[0])
    return tuple(int(p) for p in parts) or (0,)


def _cmp(a, b):
    n = max(len(a), len(b))
    a, b = a + (0,) * (n - len(a)), b + (0,) * (n - len(b))
    return (a > b) - (a < b)


def version_matches(version: str, spec: str | None) -> bool:
    """True when version satisfies a comma-separated specifier such as
    ``>=1.0,<2``, ``==1.2``, ``!=1.3`` or ``~=1.2`` (compatible release)."""
    if spec is None or str(spec).strip() in ("", "*"):
        return True
    v = _vtuple(version)
    for clause in str(spec).split(","):
        clause = clause.strip()
        m = re.fullmatch(r"(==|!=|>=|<=|~=|>|<)?\s*([0-9][0-9.]*)(\.\*)?", clause)
        if not m:
            raise GeneratorError(f"version specifier {clause!r} is not understood (use e.g. '>=1.0,<2' or '==1.2')")
        op, w, star = m.group(1) or "==", _vtuple(m.group(2)), m.group(3)
        if star and op in ("==", "!="):
            hit = v[: len(w)] == w
            if (op == "==") != hit:
                return False
            continue
        c = _cmp(v, w)
        ok = {"==": c == 0, "!=": c != 0, ">=": c >= 0, "<=": c <= 0, ">": c > 0, "<": c < 0}.get(op)
        if op == "~=":
            ok = c >= 0 and v[: max(len(w) - 1, 1)] == w[: max(len(w) - 1, 1)]
        if not ok:
            return False
    return True


class Entry:
    """A registered generator class and where it comes from."""

    def __init__(self, cls, source="built-in", dist=None, dist_version=None, error=None, name=None):
        self.cls, self.source, self.dist, self.dist_version, self.error = cls, source, dist, dist_version, error
        self.name = name or getattr(cls, "name", None)

    def info(self):
        c = self.cls
        out = {"name": self.name, "source": self.source, "package": self.dist, "package_version": self.dist_version}
        if c is None:
            return {**out, "error": self.error}
        return {**out, "version": c.version, "description": c.description,
                "capabilities": c.capabilities.to_dict(), "class": f"{c.__module__}.{c.__qualname__}"}


def _builtins():
    from .cart import CartGenerator
    from .copula import CopulaGenerator
    from .independent import IndependentGenerator
    return [CopulaGenerator, IndependentGenerator, CartGenerator]


def _validate_class(cls, name):
    if not (isinstance(cls, type) and issubclass(cls, TableGenerator)):
        raise GeneratorError(f"generator {name}: {cls!r} is not a relsynth.generators.TableGenerator subclass")
    if cls.name != name:
        raise GeneratorError(f"generator {name}: the class says its name is {cls.name!r}")
    for m in ("fit", "sample"):
        if getattr(cls, m) is getattr(TableGenerator, m):
            raise GeneratorError(f"generator {name}: {m}() is not implemented")
    major = str(getattr(cls, "contract", CONTRACT_VERSION)).split(".")[0]
    if major != CONTRACT_VERSION.split(".")[0]:
        raise GeneratorError(f"generator {name} implements contract {cls.contract}, relsynth has {CONTRACT_VERSION}")


class Registry:
    def __init__(self, entry_points=True):
        self.entries: dict[str, Entry] = {}
        self.warnings: list[str] = []
        for cls in _builtins():
            self.entries[cls.name] = Entry(cls)
        if entry_points:
            self._discover()

    def _discover(self):
        try:
            eps = metadata.entry_points(group=GROUP)
        except TypeError:                       # Python < 3.10 API (not supported, kept defensive)
            eps = metadata.entry_points().get(GROUP, [])
        seen = {}
        for ep in eps:
            dist = getattr(ep, "dist", None)
            dname = getattr(dist, "name", None) if dist else None
            dver = getattr(dist, "version", None) if dist else None
            if ep.name in self.entries and self.entries[ep.name].source == "built-in":
                self.warnings.append(f"generator {ep.name} from {dname or ep.value} ignored: the name is built in")
                continue
            if ep.name in seen:
                self.entries[ep.name] = Entry(None, "entry point", dname, dver, name=ep.name,
                                              error=f"two packages declare generator {ep.name}: {seen[ep.name]} "
                                                    f"and {dname or ep.value}; uninstall one")
                continue
            seen[ep.name] = dname or ep.value
            try:
                cls = ep.load()
                _validate_class(cls, ep.name)
                self.entries[ep.name] = Entry(cls, "entry point", dname, dver)
            except Exception as e:      # a broken plug-in must not break relsynth: report it when chosen
                self.entries[ep.name] = Entry(None, "entry point", dname, dver, name=ep.name,
                                              error=f"generator {ep.name} ({ep.value}) failed to load: {e}")

    def register(self, cls, name=None):
        """Register a class at run time (tests, notebooks)."""
        name = name or cls.name
        _validate_class(cls, name)
        self.entries[name] = Entry(cls, "registered")
        return cls

    def names(self):
        return list(self.entries)

    def entry(self, name, version=None) -> Entry:
        e = self.entries.get(name)
        if e is None:
            import difflib
            m = difflib.get_close_matches(str(name), self.names(), n=1)
            raise GeneratorError(f"unknown generator {name!r}" + (f" (did you mean {m[0]!r}?)" if m else "")
                                 + f"; installed: {', '.join(self.names())}")
        if e.cls is None:
            raise GeneratorError(e.error)
        if not version_matches(e.cls.version, version):
            raise GeneratorError(f"generator {name} is version {e.cls.version}, the spec pins {version}")
        return e

    def create(self, name, version=None, options=None) -> TableGenerator:
        return self.entry(name, version).cls(options)

    def list(self):
        return [e.info() for e in self.entries.values()]


_REGISTRY = None


def default_registry(refresh=False) -> Registry:
    global _REGISTRY
    if _REGISTRY is None or refresh:
        _REGISTRY = Registry()
    return _REGISTRY


def table_choice(spec, t) -> dict:
    """The generator chosen for table t: {name, version, options}.  A table's
    ``generator`` (a name or a mapping) wins over ``synthesis.generator``."""
    g = spec["tables"][t].get("generator")
    d = spec.get("synthesis", {}).get("generator", DEFAULT)
    if g is None:
        g = d
    if isinstance(g, str):
        g = {"name": g}
    return {"name": g.get("name", DEFAULT), "version": g.get("version"), "options": dict(g.get("options") or {})}


def check(cls, spec, t, aux=False, pushed=False, ir_version="1.0") -> tuple[list, list]:
    """(errors, warnings) of running generator class cls on table t."""
    errs, warns = [], []
    cap = cls.capabilities
    T = spec["tables"][t]
    if not version_matches(ir_version, cap.ir_versions):
        errs.append(f"{t}: generator {cls.name} supports IR {cap.ir_versions}, the spec is IR {ir_version}")
    if "record" not in cap.modalities:
        errs.append(f"{t}: generator {cls.name} does not model table rows (modalities: {', '.join(cap.modalities)})")
    from ..marginals import MODELLED
    kinds = sorted({cs["type"] for cs in T["columns"].values() if cs["type"] in MODELLED} - set(cap.column_kinds))
    if kinds:
        errs.append(f"{t}: generator {cls.name} cannot model columns of type {', '.join(kinds)}")
    if aux and "parent" not in cap.conditioning:
        warns.append(f"{t}: generator {cls.name} does not condition on parent attributes; the fact's columns "
                     "depend on its parents only through the latent segments")
    if pushed and not cap.pushdown:
        errs.append(f"{t}: generator {cls.name} cannot fit from in-database statistics and a sample; "
                    "set synthesis.pushdown.mode: off or choose another generator")
    if not cap.posterior and spec.get("synthesis", {}).get("posterior", {}).get("datasets", 1) > 1:
        warns.append(f"{t}: generator {cls.name} has no posterior: every dataset uses the same fitted "
                     "attribute model (datasets differ only in the relational draws)")
    return errs, warns
