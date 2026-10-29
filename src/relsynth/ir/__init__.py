# SPDX-License-Identifier: Apache-2.0
"""The canonical intermediate representation (IR 2.0) of a data estate.

    from relsynth import ir
    est = ir.load("spec.yaml")              # a 1.0 spec or an IR document
    ir.dump(est, "estate.yaml")             # written as IR 2.0
    spec, losses = ir.lower(est)            # the 1.0 spec the engine reads

See docs/ir-guide.md, docs/ir-reference.md and docs/adr/0001-canonical-ir.md.
"""
from .model import IR_VERSION, Attribute, Entity, Estate, IRError, Key, Relationship, Rule
from .v1 import canonical, diff, is_ir, lift, lower

__all__ = ["IR_VERSION", "Estate", "Entity", "Attribute", "Key", "Relationship", "Rule", "IRError", "lift", "lower",
           "canonical", "diff", "is_ir", "load", "loads", "dump", "dumps", "from_document"]


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
