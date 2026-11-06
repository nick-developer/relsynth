# SPDX-License-Identifier: Apache-2.0
"""Document databases (roadmap T13, release 2.1).

* :mod:`.values`    BSON / JSON types, MongoDB Extended JSON v2.
* :mod:`.infer`     parametric schema inference with counts: optional fields,
  unions, array and map length laws, variants, length fields, references.
* :mod:`.shred`     collections as relational tables (one per array or map) and back.
* :mod:`.relational` the shredded tables as a SQLite work database and its spec.
* :mod:`.conform`   synthetic documents made to validate against the schema.
* :mod:`.validate`  schema validation.
* :mod:`.metrics`   shape metrics with calibrated limits, reference integrity.
* :mod:`.connectors` MongoDB (and Cosmos DB for MongoDB), Cosmos DB for NoSQL,
  DynamoDB, Couchbase, JSON / Extended JSON files.
* :mod:`.pipeline`  `DocumentSynthesizer`: the steps end to end.

See docs/documents.md.
"""
from .pipeline import DocumentSynthesizer

__all__ = ["DocumentSynthesizer"]
