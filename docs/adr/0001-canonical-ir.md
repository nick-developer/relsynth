# ADR 0001: A canonical, modality-neutral intermediate representation (IR 2.0)

- Status: accepted (roadmap thread T02)
- Date: 2026-10-26
- Deciders: relsynth maintainers
- Related: [docs/ir-guide.md](../ir-guide.md), [docs/ir-reference.md](../ir-reference.md),
  Propositions 106–108 ([docs/proofs/sec_v20_ir.tex](../proofs/sec_v20_ir.tex))

## Context

relsynth 1.0 describes a database by a relational YAML spec: tables, columns, primary and foreign keys, rules and
synthesis options. The roadmap adds document stores (MongoDB, DynamoDB), graphs, lakehouse tables (Iceberg, Delta)
and event streams, plus formal differential privacy, which needs to know *whose* data each record is. A relational
spec cannot say "an array of 1 to 50 line items", "a payment sub-document whose fields depend on `method`", "an
edge with properties", "partitioned by `day(ts)`", or "the privacy unit is the customer, reached through
`order_line → order → customer`".

Every later thread (connectors, engine, DP, evaluation) needs one model of the data estate to read and write.
Existing 1.0 specs, including those in CI pipelines, must keep working and keep giving the same synthetic data.

## Decision

1. **One versioned document, the IR** (`ir_version: "2.0"`), describes the whole estate. Entities have a *shape*
   (record, document, node, edge, event) and a *role* (entity, fact, view); attributes have a logical
   type that can nest (object, array with a length law, variants with a discriminator); keys, relationships,
   time, physical layout, rules and privacy annotations are first-class nodes. Unknown fields are refused, except
   `x-` extensions, which are kept on every node.
2. **The Python object model is the source of truth.** It is a set of plain dataclasses whose field metadata
   (doc, YAML key, enum, required) generates the JSON Schema (draft 2020-12, committed as
   `src/relsynth/ir/schema/relsynth-ir-2.0.schema.json`) and the reference docs. A test fails when the
   committed schema or reference drifts from the model.
3. **1.0 specs are lifted, not rewritten.** `lift` and `lower` are field-by-field inverses on the 1.0 key
   schema, so `normalise(lower(lift(s))) == normalise(s)` (Proposition 106). Every command that reads a spec
   accepts either format.
4. **Staged migration of the engine.** In this release the engine still runs a 1.0 spec: an IR document is
   *lowered* to its relational part first. Lowering lists what it leaves out (computed by diffing the canonical
   forms, so nothing is dropped silently) and `--strict` refuses instead. Synthesising non-relational shapes is
   refused with a clear error until the engine reads the IR directly (owned by the engine thread, T03).
5. **Privacy is part of the description.** Attributes carry a privacy role (direct identifier, quasi-identifier,
   sensitive); entities can be public; the estate names the privacy unit. `trace` computes the reference paths
   from each entity to the unit (Proposition 107) and the cross-unit links that would dangle when a unit is
   removed (Proposition 108).
6. **Standards at the edges.** ODCS v3 contracts are imported and exported (relsynth-only fields travel as
   `customProperties` with a `relsynth` payload, so the round trip is exact); OpenLineage dataset facets are
   exported. Neither is used as the internal model.
7. **Deprecation policy.** 2.x reads 1.0 specs and IR 2.x documents and runs both the same way; reading a 1.0
   spec warns from 3.0 at the earliest and stops no earlier than 4.0 and 24 months after 2.0;
   `relsynth ir migrate` stays in every release. IR minor versions only add optional fields; a major version
   ships a migrator from the previous one.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Extend the 1.0 YAML in place | The 1.0 shape is table → columns; nesting, variants, edges and partitions would be bolted onto `columns` with type-specific keys, and the relational validator would have to special-case each. A versioned new document with a proven conversion is cleaner and keeps 1.0 unchanged. |
| Make ODCS v3 the internal model | ODCS describes contracts (schema, quality, SLAs, people), not generative structure: it has no multiplicity laws, no variant discriminators, no privacy unit, no synthesis options. Everything would live in `customProperties`, which ODCS does not validate. We map to it instead. |
| JSON Schema, Avro or Protobuf as the model | They describe one record type; they have no cross-entity keys, relationships, time semantics or privacy unit. JSON Schema is generated *from* the model for editors and validation. |
| LinkML or RDF/SHACL | Expressive and standard, but heavy dependencies and an unfamiliar authoring style for data engineers; the YAML they author would not be the YAML relsynth users already write. A LinkML export can be added later from the same metadata. |
| Have the engine read the IR now | Would change the code path of every fit in the same release as the format change and put the 1.0 results at risk. Lowering keeps the engine byte-identical (checked on the four real databases) and lets T03 move it with its own tests. |
| Separate documents per modality (one for SQL, one for documents, …) | Real estates mix modalities (an order table referencing a customer document), and the privacy unit's paths cross them. One estate graph is needed. |
| pydantic models | A runtime dependency with major-version churn for what dataclasses plus a small validator do; the error messages (unknown keys with suggestions, YAML paths) would have to be re-done anyway. |

## Consequences

- Every spec-reading command and the Python API accept IR documents; `introspect --format ir` drafts one. The
  default output stays 1.0 until the user decides otherwise.
- Commands that edit a spec in place (`doctor --apply`, `tune`) still write 1.0 specs.
- YAML comments are not carried by `ir migrate`.
- Later threads extend the IR by adding optional fields (minor versions); interface changes go through RFCs.
