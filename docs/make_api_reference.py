# SPDX-License-Identifier: Apache-2.0
"""Write docs/api.md from the public API's signatures and docstrings:

    python docs/make_api_reference.py
"""
from __future__ import annotations

import inspect
from pathlib import Path

import relsynth as rs

INTRO = """# Python API reference

Generated from the code by `python docs/make_api_reference.py`. Everything listed here is importable from `relsynth` (`import relsynth as rs`) and covered by `tests/unit/test_api.py`.

## A typical session

```python
import relsynth as rs

spec = rs.introspect("postgresql://analyst@db/sales", password_env="PGPASSWORD")
spec.errors()                                   # [] when the spec is valid
model = rs.fit(spec)                            # FittedModel
data = model.sample(scale=0.5)                  # SyntheticData: {table: DataFrame}
report = model.evaluate(data)                   # EvaluationReport
report.require(0.98)                            # raises QualityGateError below 98%
data.write("sqlite:synthetic.db")
```

Specs can be given to every function as a path, a dict or a `Spec`. Locations are strings (`csv:DIR`, `parquet:DIR`, `sqlite:FILE`, a database URL) or sections (`{"kind": "postgresql", "url_env": "DB_URL", "schema": "dw"}`).

## Errors

| Exception | Raised when | CLI exit code |
|---|---|---|
| `SpecError` (a `ValueError`) | The spec or an argument is invalid | 2 |
| `SourceError` (a `RelsynthError` and an `OSError`) | A connection, permission, file or write fails | 3 |
| `EngineError` | Fitting or sampling fails | 4 |
| `QualityGateError` | `require()` or `fail_under` finds the score too low | 1 |

Messages never contain passwords.
"""

FUNCTIONS = ["introspect", "validate_spec", "fit", "sample", "evaluate", "run", "verify", "doctor", "tune",
             "export_model", "clone_schema", "profile", "ping", "connect", "load_spec", "dump_spec"]
CLASSES = ["Spec", "FittedModel", "SyntheticData", "EvaluationReport", "RunResult", "VerifyReport", "Diagnosis",
           "Synthesizer"]


def _sig(obj, name) -> str:
    try:
        s = str(inspect.signature(obj, eval_str=True))
    except (TypeError, ValueError, NameError):
        try:
            s = str(inspect.signature(obj)).replace("'", "")
        except (TypeError, ValueError):
            s = "(...)"
    for prefix in ("relsynth.api.", "relsynth.verify.", "relsynth.engine.", "pathlib._local.", "pathlib.",
                   "collections.abc.", "pandas.core.frame."):
        s = s.replace(prefix, "")
    return f"{name}{s}"


def _doc(obj) -> str:
    return inspect.cleandoc(obj.__doc__ or "").strip()


def _members(cls):
    for n, m in inspect.getmembers(cls):
        if n.startswith("_") and n != "__init__":
            continue
        if n == "__init__" and (m is object.__init__ or not m.__doc__):
            continue
        if isinstance(inspect.getattr_static(cls, n), property):
            yield n, m, "property"
        elif inspect.isfunction(m) or inspect.ismethod(m):
            if getattr(m, "__module__", "").startswith("relsynth"):
                yield n, m, "method"


def render() -> str:
    out = [INTRO, "## Functions", ""]
    for n in FUNCTIONS:
        f = getattr(rs, n)
        out += [f"### `rs.{n}`", "", "```python", _sig(f, n), "```", "", _doc(f), ""]
    out += ["## Classes", ""]
    for n in CLASSES:
        c = getattr(rs, n)
        out += [f"### `rs.{n}`", "", _doc(c), ""]
        props = [(m, kind) for m, _, kind in _members(c) if kind == "property"]
        if props:
            out += ["Properties: " + ", ".join(f"`{m}`" for m, _ in props) + ".", ""]
        for m, obj, kind in _members(c):
            if kind != "method":
                continue
            out += [f"#### `{n}.{m}`", "", "```python", _sig(obj, m), "```", ""]
            d = _doc(obj)
            if d:
                out += [d, ""]
    return "\n".join(out).rstrip() + "\n"


if __name__ == "__main__":
    dest = Path(__file__).with_name("api.md")
    dest.write_text(render(), encoding="utf-8")
    print(f"wrote {dest}")
