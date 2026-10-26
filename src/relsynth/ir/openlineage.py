"""OpenLineage dataset facets of an estate (export only; emitting run events is
the governance thread's work).

OpenLineage (LF AI & Data; https://openlineage.io) describes datasets by a
namespace, a name and facets.  For each entity this module writes:

    schema         SchemaDatasetFacet 1-2-0: fields with types, descriptions and nested fields
    datasetType    DatasetTypeDatasetFacet 1-0-1: TABLE, VIEW, COLLECTION, NODE, EDGE or STREAM
    documentation  DocumentationDatasetFacet 1-1-0: the entity's description
    tags           TagsDatasetFacet 1-0-0: privacy roles and the privacy unit, per field
    columnLineage  ColumnLineageDatasetFacet 1-2-0 (synthetic outputs only): each synthetic
                   field depends on the same field of the real dataset, as a DIRECT
                   AGGREGATION with masking (no value is copied: values are drawn from a
                   model fitted to the inputs)

The mapping of relsynth's synthesis to the transformation types is our reading
of the specification's definitions, not a published convention.
"""
from __future__ import annotations

from .._version import __version__
from .model import Attribute, Entity, Estate

PRODUCER = f"https://github.com/nick-developer/relsynth/tree/v{__version__}"
_FACETS = "https://openlineage.io/spec/facets"
SCHEMA_URLS = {
    "schema": f"{_FACETS}/1-2-0/SchemaDatasetFacet.json#/$defs/SchemaDatasetFacet",
    "datasetType": f"{_FACETS}/1-0-1/DatasetTypeDatasetFacet.json#/$defs/DatasetTypeDatasetFacet",
    "documentation": f"{_FACETS}/1-1-0/DocumentationDatasetFacet.json#/$defs/DocumentationDatasetFacet",
    "tags": f"{_FACETS}/1-0-0/TagsDatasetFacet.json#/$defs/TagsDatasetFacet",
    "columnLineage": f"{_FACETS}/1-2-0/ColumnLineageDatasetFacet.json#/$defs/ColumnLineageDatasetFacet",
}
_TYPE = {"record": "TABLE", "document": "COLLECTION", "node": "NODE", "edge": "EDGE", "event": "STREAM"}


def _facet(kind: str, body: dict) -> dict:
    return {"_producer": PRODUCER, "_schemaURL": SCHEMA_URLS[kind], **body}


def _field(name: str, a: Attribute, pos: int | None = None) -> dict:
    f = {"name": name, "type": a.physical_type or a.effective_type or a.kind or "unknown"}
    if a.description:
        f["description"] = a.description
    if pos is not None:
        f["ordinal_position"] = pos
    nested = dict(a.fields)
    if a.items is not None:
        nested = dict(a.items.fields) if a.items.fields else {"_element": a.items}
    if a.variants is not None:
        for case in a.variants.cases.values():
            for k, v in case.items():
                nested.setdefault(k, v)
    if nested:
        f["fields"] = [_field(k, v) for k, v in nested.items()]
    return f


def _tags(est: Estate, n: str, e: Entity) -> list:
    out = []
    unit = est.privacy.unit.entity if est.privacy and est.privacy.unit else None
    if unit == n:
        out.append({"key": "privacy_unit", "value": "true", "source": "relsynth"})
    if e.privacy is not None and e.privacy.public:
        out.append({"key": "public", "value": "true", "source": "relsynth"})

    def walk(prefix, attrs):
        for k, a in attrs.items():
            p = f"{prefix}{k}"
            if a.privacy is not None and a.privacy.role:
                out.append({"key": a.privacy.role, "value": "true", "source": "relsynth", "field": p})
            if a.privacy is not None and a.privacy.classification:
                out.append({"key": "classification", "value": a.privacy.classification, "source": "relsynth",
                            "field": p})
            walk(f"{p}.", a.fields)
    walk("", e.attributes)
    return out


def dataset_name(est: Estate, n: str) -> str:
    e = est.entities[n]
    return e.physical.name if e.physical and e.physical.name else n


def datasets(est: Estate, namespace: str, inputs_namespace: str | None = None) -> list[dict]:
    """One OpenLineage dataset per entity.  With inputs_namespace, the datasets are
    synthetic outputs and carry column lineage from the same-named datasets there."""
    out = []
    for n, e in est.entities.items():
        facets = {
            "schema": _facet("schema", {"fields": [_field(k, a, i + 1) for i, (k, a) in enumerate(e.attributes.items())]}),
            "datasetType": _facet("datasetType", {"datasetType": "VIEW" if e.role == "view" else _TYPE[e.shape],
                                                  **({"subType": "SYNTHETIC"} if inputs_namespace else {})}),
        }
        if e.description:
            facets["documentation"] = _facet("documentation", {"description": e.description})
        tags = _tags(est, n, e)
        if tags:
            facets["tags"] = _facet("tags", {"tags": tags})
        if inputs_namespace:
            name = dataset_name(est, n)
            facets["columnLineage"] = _facet("columnLineage", {"fields": {
                k: {"inputFields": [{"namespace": inputs_namespace, "name": name, "field": k,
                                     "transformations": [{"type": "DIRECT", "subtype": "AGGREGATION",
                                                          "description": "drawn from relsynth's posterior predictive "
                                                                         "model fitted to the input", "masking": True}]}]}
                for k in e.attributes}})
        out.append({"namespace": namespace, "name": dataset_name(est, n), "facets": facets})
    return out
