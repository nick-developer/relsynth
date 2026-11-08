# SPDX-License-Identifier: Apache-2.0
"""The package's module-level import graph has no cycle.

Code scanning (CodeQL py/cyclic-import) reports import cycles that run through imports
deferred into functions: relsynth imports optional drivers, heavy modules and the pieces
that call back into the engine inside the function that needs them, so importing one
module never imports the rest.  Such a cycle cannot fail at import time.  A cycle among
module-level imports can (a half-initialised module, an ImportError that depends on the
import order), and this test keeps the graph of module-level imports acyclic.  Imports
under `if TYPE_CHECKING:` are left out, as Python never runs them.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def _modules() -> dict[str, Path]:
    """Every module of the relsynth package by dotted name (a package by its own name)."""
    out = {}
    for p in (SRC / "relsynth").rglob("*.py"):
        parts = p.relative_to(SRC).with_suffix("").parts
        out[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = p
    return out


def _targets(mod: str, node: ast.Import | ast.ImportFrom, is_pkg: bool, mods: dict) -> set[str]:
    """The relsynth modules one import statement of `mod` loads (relative imports resolved)."""
    if isinstance(node, ast.Import):
        names = [a.name for a in node.names]
    else:
        if node.level:
            base = mod.split(".") if is_pkg else mod.split(".")[:-1]
            base = base[:len(base) - (node.level - 1)]
            prefix = ".".join(base + ([node.module] if node.module else []))
        else:
            prefix = node.module or ""
        names = [f"{prefix}.{a.name}" if f"{prefix}.{a.name}" in mods else prefix for a in node.names]
    # importing a.b.c also runs a/__init__ and a/b/__init__, except the packages that hold `mod`
    # itself (Python has initialised, or is initialising, those before `mod` runs)
    own = {".".join(mod.split(".")[:i]) for i in range(1, mod.count(".") + 1)}
    out = set()
    for n in names:
        parts = n.split(".")
        out |= {".".join(parts[:i]) for i in range(1, len(parts) + 1)} - own
    return {n for n in out if n in mods}


def _type_checking(test: ast.expr) -> bool | None:
    """True for `TYPE_CHECKING` (or `typing.TYPE_CHECKING`), False for its negation, None otherwise."""
    def is_tc(e: ast.expr) -> bool:
        return (isinstance(e, ast.Name) and e.id == "TYPE_CHECKING") or \
            (isinstance(e, ast.Attribute) and e.attr == "TYPE_CHECKING")
    if is_tc(test):
        return True
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and is_tc(test.operand):
        return False
    return None


def _module_level_imports(mod: str, path: Path, mods: dict) -> set[str]:
    """The relsynth modules `mod` imports when it is imported: function bodies and
    TYPE_CHECKING blocks are skipped."""
    deps: set[str] = set()
    stack: list[ast.AST] = list(ast.parse(path.read_text(encoding="utf-8")).body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue                      # deferred: runs when the function is called
        if isinstance(n, ast.If) and _type_checking(n.test) is not None:
            # `if TYPE_CHECKING:` runs only its else branch, `if not TYPE_CHECKING:` only its body
            stack.extend(n.orelse if _type_checking(n.test) else n.body)
            continue
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            deps |= _targets(mod, n, path.name == "__init__.py", mods)
        else:
            stack.extend(ast.iter_child_nodes(n))
    deps.discard(mod)
    return deps


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Strongly connected components with more than one module (iterative Tarjan)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on: set[str] = set()
    stack: list[str] = []
    out = []
    for root in sorted(graph):
        if root in index:
            continue
        work = [(root, iter(sorted(graph[root])))]
        index[root] = low[root] = len(index)
        stack.append(root)
        on.add(root)
        while work:
            v, it = work[-1]
            w = next(it, None)
            if w is not None:
                if w not in index:
                    index[w] = low[w] = len(index)
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(sorted(graph[w]))))
                elif w in on:
                    low[v] = min(low[v], index[w])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    x = stack.pop()
                    on.discard(x)
                    comp.append(x)
                    if x == v:
                        break
                if len(comp) > 1:
                    out.append(sorted(comp))
    return out


def test_cycle_finder_finds_a_cycle():
    """The cycle finder reports a three-module cycle and nothing on a chain."""
    assert _cycles({"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}}) == [["a", "b", "c"]]
    assert _cycles({"a": {"b"}, "b": set()}) == []


def test_graph_follows_runtime_branches_and_packages(tmp_path):
    """`if not TYPE_CHECKING:` bodies count, `if TYPE_CHECKING:` bodies do not, and importing
    a submodule also runs its package's __init__ (unless that package holds the importer)."""
    mods = {"relsynth": None, "relsynth.a": None, "relsynth.p": None, "relsynth.p.c": None, "relsynth.b": None}
    def deps(src: str, mod: str = "relsynth.a") -> set[str]:
        path = tmp_path / "case.py"
        path.write_text(src, encoding="utf-8")
        return _module_level_imports(mod, path, mods)
    assert deps("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from . import b\n") == set()
    assert deps("from typing import TYPE_CHECKING\nif not TYPE_CHECKING:\n    from . import b\n") == {"relsynth.b"}
    assert deps("from .p import c\n") == {"relsynth.p", "relsynth.p.c"}
    assert deps("from relsynth.p import c\n", mod="relsynth.p.x") == {"relsynth.p.c"}


def test_module_level_imports_have_no_cycle():
    """No cycle among the module-level imports of src/relsynth."""
    mods = _modules()
    assert "relsynth.engine" in mods and "relsynth.connectors" in mods
    graph = {m: _module_level_imports(m, p, mods) for m, p in mods.items()}
    assert sum(len(d) for d in graph.values()) > 100        # the graph was actually read
    assert _cycles(graph) == []
