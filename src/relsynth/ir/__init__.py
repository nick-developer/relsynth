# SPDX-License-Identifier: Apache-2.0
"""The canonical intermediate representation (IR 2.0) of a data estate.

    from relsynth import ir
    est = ir.load("spec.yaml")              # a 1.0 spec or an IR document
    ir.dump(est, "estate.yaml")             # written as IR 2.0
    spec, losses = ir.lower(est)            # the 1.0 spec the engine reads

See docs/ir-guide.md, docs/ir-reference.md and docs/adr/0001-canonical-ir.md.
"""
from .model import IR_VERSION, Attribute, Entity, Estate, IRError, Key, Relationship, Rule
from .v1 import canonical, diff, is_ir, lift, lower, update

__all__ = ["IR_VERSION", "IR_DEFAULT_FROM", "Estate", "Entity", "Attribute", "Key", "Relationship", "Rule", "IRError",
           "lift", "lower", "update", "canonical", "diff", "is_ir", "load", "loads", "dump", "dumps", "from_document",
           "default_format"]

# the release from which introspect writes IR documents by default (T02 decision,
# accepted 2026-10-28); 1.0 specs stay readable as docs/ir-guide.md describes
IR_DEFAULT_FROM = "2.0"


def default_format(version: str | None = None) -> str:
    """The format introspect writes when none is asked for: "ir" from release
    IR_DEFAULT_FROM on, "v1" before (version: relsynth's own by default)."""
    import re

    from .._version import __version__

    def key(v):
        return tuple(int(x) for x in re.findall(r"\d+", str(v).split("+")[0])[:3])
    return "ir" if key(version or __version__) >= key(IR_DEFAULT_FROM) else "v1"


def from_document(doc: dict) -> Estate:
    """An Estate from a parsed document: an IR document or a 1.0 spec."""
    if is_ir(doc):
        return Estate.from_dict(doc)
    return lift(doc)


def loads(text: str) -> Estate:
    import yaml

    from ..spec import _SpecLoader
    return from_document(yaml.load(text, Loader=_SpecLoader))  # noqa: S506  # nosec B506 - a SafeLoader subclass


def load(path) -> Estate:
    from ..spec import read_yaml
    return from_document(read_yaml(path))


def dumps(est: Estate, fmt: str = "yaml") -> str:
    import json

    import yaml
    d = est.to_dict()
    if fmt == "json":
        return json.dumps(d, indent=2, default=str) + "\n"
    return yaml.safe_dump(d, sort_keys=False, width=110, allow_unicode=True)


def dump(est: Estate, path, fmt: str | None = None) -> None:
    from pathlib import Path
    p = Path(path)
    p.write_text(dumps(est, fmt or ("json" if p.suffix == ".json" else "yaml")), encoding="utf-8")
