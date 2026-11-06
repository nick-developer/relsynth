# SPDX-License-Identifier: Apache-2.0
"""Synthesise document collections end to end.

    from relsynth.documents import DocumentSynthesizer
    ds = DocumentSynthesizer.from_source({"kind": "mongodb", "url_env": "MONGO_URL", "database": "shop"})
    ds.discover()                    # schema with counts, references (ds.schema_ir() for the IR)
    out = ds.synthesize(scale=1.0)   # {collection: [documents]}
    report = ds.report(out)          # validity, references, shape metrics with calibrated limits
    ds.write(out, {"kind": "jsonl", "path": "synthetic/"})

Steps: infer the schema with counts (`infer`), shred the collections into
relational tables (`shred`), store them in a SQLite work file with every key
declared, draft and complete the 1.0 spec (`relational`), fit and sample the
relsynth engine (Bayesian copula per table, latent segments across tables,
exact keys), nest the synthetic tables back into documents, and conform them
to the schema (`conform`).
"""
from __future__ import annotations

import copy
import logging
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .conform import conform
from .infer import CollectionProfile, find_references, find_sequences, profile_collection
from .metrics import reference_integrity, shape_checks
from .relational import spec_for, write_sqlite
from .shred import Layout, build_layout, nest, shred
from .validate import validate_collection

log = logging.getLogger("relsynth.documents")


class DocumentSynthesizer:
    def __init__(self, collections: dict[str, list[dict]], *, references: list[dict] | None = None,
                 seed: int = 7, workdir: str | Path | None = None, spec_overrides: dict | None = None,
                 name: str = "documents"):
        self.collections = collections
        self.declared = references or []
        self.seed = int(seed)
        self.name = name
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="relsynth-docs-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.spec_overrides = spec_overrides or {}
        self.profiles: dict[str, CollectionProfile] = {}
        self.references: list[dict] = []
        self.layout: Layout | None = None
        self.sequences: list[dict] = []
        self.spec: dict | None = None
        self.model: Any = None
        self.repairs: dict = {}

    @classmethod
    def from_source(cls, source: dict, **kw) -> DocumentSynthesizer:
        from .connectors import connect_documents
        con = connect_documents(source)
        return cls(con.read_all(), **kw)

    # -------------------------------------------------------------- discovery
    def discover(self) -> DocumentSynthesizer:
        self.profiles = {c: profile_collection(c, docs) for c, docs in self.collections.items()}
        self.references = find_references(self.profiles, self.declared)
        self.layout = build_layout(self.profiles, self.references)
        self.sequences = find_sequences(self.collections, self.references)
        return self

    def schema_ir(self):
        from .ir import to_ir
        if not self.profiles:
            self.discover()
        return to_ir(self.profiles, self.references, name=self.name)

    # -------------------------------------------------------------- synthesis
    def build_spec(self) -> dict:
        if self.layout is None:
            self.discover()
        assert self.layout is not None
        frames = shred(self.layout, self.collections)
        db = write_sqlite(self.layout, frames, self.workdir / "shredded.db")
        spec = spec_for(self.layout, db, name=self.name, seed=self.seed, sequences=self.sequences)
        _merge(spec, self.spec_overrides)
        self.spec = spec
        (self.workdir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        return spec

    def fit(self):
        from ..api import fit
        if self.spec is None:
            self.build_spec()
        self.model = fit(self.spec)
        return self

    def synthesize(self, scale: Any = None, dataset: int = 0) -> dict[str, list[dict]]:
        if self.model is None:
            self.fit()
        assert self.layout is not None
        syn = self.model.sample(scale=scale, dataset=dataset)
        docs = nest(self.layout, syn.to_dict(), seed=(self.seed, dataset))
        out = {}
        self.repairs = {}
        for c, ds in docs.items():
            out[c], self.repairs[c] = conform(self.profiles[c], ds, seed=self.seed + dataset)
        return out

    # -------------------------------------------------------------- quality
    def report(self, synthetic: dict[str, list[dict]], splits: int = 10, predictive: bool = True) -> dict:
        rep: dict[str, Any] = {"collections": {}, "references": reference_integrity(synthetic, self.references)}
        for c, docs in synthetic.items():
            real = self.collections[c]
            rep["collections"][c] = {
                "documents": {"real": len(real), "synthetic": len(docs)},
                "validity": validate_collection(self.profiles[c], docs),
                "repairs": self.repairs.get(c, {}),
                "shape": shape_checks(real, docs, c, splits=splits, seed=self.seed, predictive=predictive,
                                      real_prof=self.profiles[c]),
            }
        rep["pass"] = (all(r["share"] == 1.0 for r in rep["references"])
                       and all(v["validity"]["share"] == 1.0 for v in rep["collections"].values())
                       and all(m["pass"] is not False for v in rep["collections"].values() for m in v["shape"]))
        return rep

    def write(self, synthetic: dict[str, list[dict]], target: dict) -> dict:
        from .connectors import connect_documents
        return connect_documents(target).write_all(synthetic, if_exists=target.get("if_exists", "replace"))


def _merge(a: dict, b: dict) -> None:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
