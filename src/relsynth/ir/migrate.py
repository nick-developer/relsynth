# SPDX-License-Identifier: Apache-2.0
"""Migration of relsynth 1.0 specs to IR 2.0 documents.

`migrate(path)` reads a 1.0 spec, converts it, proves the conversion (the 1.0
spec the IR lowers to normalises to the same spec as the original, Prop. 106)
and returns the document with a report.  Optional additions (a privacy unit,
public entities) are applied after the proof, because they are what a 1.0
spec cannot say.

Deprecation policy (docs/ir-guide.md, "Versions and deprecation"):
  * 2.x reads 1.0 specs and IR 2.x documents and runs both the same way.
  * Reading a 1.0 spec warns from release 3.0 at the earliest.
  * Reading 1.0 specs stops no earlier than release 4.0 and no earlier than 24
    months after 2.0; `relsynth ir migrate` stays in every release.
  * IR versions: a minor version (2.1) only adds optional fields; a major
    version (3.0) may rename or remove, and ships a migrator from the previous one.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field

from .model import EntityPrivacy, Estate, EstatePrivacy, IRError, PrivacyUnit
from .v1 import diff, is_ir, lift, lower


@dataclass
class Migration:
    estate: Estate
    source_format: str                         # "1.0" or "ir"
    identical: bool                            # normalise(lower(lift(s))) == normalise(s)
    differences: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def header(self, source: str) -> str:
        today = _dt.date.today().isoformat()
        lines = [f"# relsynth IR {self.estate.ir_version}, migrated from {source} (relsynth {self.source_format} spec) "
                 f"on {today}.",
                 "# The 1.0 spec this document lowers to is identical to the original after normalisation"
                 if self.identical else "# WARNING: the conversion is not identical; see the migration report"]
        lines += [f"# {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def _norm(d):
    from ..spec import normalise
    return json.loads(json.dumps(normalise(json.loads(json.dumps(d, default=str))), default=str))


def migrate_doc(raw: dict, privacy_unit: str | None = None, public: list | None = None) -> Migration:
    if is_ir(raw):
        est = Estate.from_dict(raw)
        m = Migration(est, "ir", True, notes=["already an IR document: re-written in canonical order"])
    else:
        est = lift(raw)
        v1, _ = lower(est)
        d = diff(_norm({k: v for k, v in raw.items() if not str(k).startswith("_")}), _norm(v1))
        m = Migration(est, "1.0", not d, d)
        m.notes.append("YAML comments of the 1.0 file are not carried over; keep the original next to it")
    if privacy_unit:
        if privacy_unit not in est.entities:
            raise IRError(f"--privacy-unit {privacy_unit!r} is not an entity")
        est.privacy = est.privacy or EstatePrivacy()
        est.privacy.unit = PrivacyUnit(entity=privacy_unit)
        m.notes.append(f"privacy unit set to {privacy_unit}")
    for n in public or []:
        if n not in est.entities:
            raise IRError(f"--public {n!r} is not an entity")
        est.entities[n].privacy = EntityPrivacy(public=True)
    if public:
        m.notes.append("public entities: " + ", ".join(public))
    return m


def migrate(path, privacy_unit: str | None = None, public: list | None = None) -> Migration:
    from ..spec import read_yaml
    return migrate_doc(read_yaml(path), privacy_unit, public)
