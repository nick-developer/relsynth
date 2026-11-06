# ADR 0011: Document databases

- Status: proposed (roadmap thread T13; release 2.1.0)
- Date: 2026-11-01

## Context

relsynth 2.0 synthesises relational data: tables with keys, links and rules. Many estates also keep data in
document stores (MongoDB, Azure Cosmos DB, Amazon DynamoDB, Couchbase). A document has a shape that a table does
not: optional fields, fields whose type changes from one document to the next, nested objects, arrays whose length
follows a law, polymorphic sub-documents selected by a discriminator, maps keyed by identifiers, and references to
other documents by `_id` or by a natural key. DynamoDB single-table designs add a further layer: several item
types in one table, with keys built from templates such as `CUSTOMER#{customer_id}`.

The roadmap (T13) asks for MongoDB and Cosmos DB first, then DynamoDB and Couchbase, with the shape, arrays,
polymorphism and references preserved, schema inference with counts into the IR, pushdown through `$group` and
`$sample`, shape metrics, and a time-boxed spike on a direct document generator. IR 2.0 (ADR 0001) already
describes documents, arrays with length laws and variants.

## Options

1. **Shred to relations, synthesise with the engine, re-nest.** Infer a schema with counts (parametric inference:
   Baazizi, Colazzo, Ghelli and Sartiani, EDBT 2017 and VLDB J. 2019), shred each collection into a root table and
   one child table per array, map and optional sub-document (the edge/inlining mappings of XML shredding: Shanmugasundaram et
   al., VLDB 1999), synthesise the tables with the existing engine and nest the result. Every relational guarantee
   (keys, links, counts per parent, rules, privacy and DP, the scorecard) carries over. Cross-level dependence
   (an element's value given its document's values) is only as good as the engine's parent-child model.
2. **A direct document generator.** Model a document as a tree and draw it top-down (a probabilistic grammar over
   the inferred schema, or a sequence model over serialised documents). It captures cross-level dependence
   naturally but needs new machinery for keys, references, counts, rules, privacy accounting and calibration that
   the engine already has.
3. **Flatten each collection into one wide table** (one column per path, arrays exploded or truncated). Simple, but
   loses array-length laws and nesting, and the column count grows with the array lengths.

## Decision

1. **Option 1 is the baseline.** `relsynth.documents` profiles each collection (types, presence, lengths, distinct
   values, signatures per path), detects maps (keys that look like identifiers, Spoth et al., SIGMOD 2021), variants
   (a discriminator whose values determine the field set), length fields (`n_items == len(items)`), set arrays and
   references (inclusion dependencies with coverage, with natural keys and sequences 1..n within a parent), and
   exports the schema with counts as an IR 2.0 document (`x-relsynth-counts`).
2. **Shredding keeps the structure explicit.** Absent, null and the type of a union field are tag columns
   (`<path>__t`); an array's or map's length is a column (`<path>__n`) that the engine models with the parent row;
   numeric unions share one column; element tables are facts with a Dirichlet-process count law per parent, or
   entities when they have children of their own. Nesting matches the drawn lengths and presence tags, so
   lengths, presence and variants follow the model rather than chance. `nest(shred(D)) = D` (Prop. 1401).
3. **Conformance is a counted repair, not a hidden one.** Every synthetic document validates against the inferred
   schema (Prop. 1402). What the engine misses (too long an array, a repeated element in a set, a field outside its
   variant) is repaired from synthetic values only, and every repair is counted by kind and path in the report.
4. **Shape metrics are calibrated** on real-versus-real splits (presence, type mix, array lengths, signatures,
   depth, size; Prop. 1403), like the quality report 2.0; references must resolve 100%.
5. **DynamoDB single-table designs** are split into item types by a discriminator, key templates are learned and
   rendered back exactly (Prop. 1404), and GSI keys follow.
6. **Option 2 is no-go for now** (spike in `docs/research/document-generator-spike.md`): the baseline meets the
   acceptance criteria, and a direct generator would duplicate the engine's guarantees. It is the place to start if
   cross-level dependence becomes the limiting metric.
7. Maps are written in the IR as the extension `x-relsynth-map` until RFC-T13-01 (maps with `items` and `length` in
   IR shapes) is decided by the IR owner.

## Consequences

- A document source gets the engine's whole stack: settings, generators, DP mode and governance apply to the
  shredded tables. Privacy attacks run on the shredded tables (one row of the root table is one document).
- The default relational output is unchanged: the new code is a separate module and a new `relsynth docs` command.
- Fitting cost grows with the number of child tables; on a 5 000-document movie collection (24 tables) a fit takes
  about five minutes on four cores.
- Out of scope: the Cassandra and Gremlin APIs of Cosmos DB, change streams, and documents larger than memory per
  collection (the pushdown statistics are exact, the sample for the correlations is bounded).
