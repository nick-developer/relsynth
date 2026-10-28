# The relsynth IR (2.0): guide

relsynth 2.0 describes a data estate by one document, the **IR** (intermediate representation). It covers
relational tables as the 1.0 spec did, and also documents with nested objects, arrays and polymorphic
sub-documents, graphs, lakehouse tables with hidden partitions, event streams, and privacy annotations. Every
field is listed in the generated [reference](ir-reference.md); the design and the alternatives we rejected are in
[ADR 0001](adr/0001-canonical-ir.md); the proofs are Propositions 106–108 in [docs/proofs](proofs/).

**Your 1.0 specs keep working.** Every command reads either format, and an IR document migrated from a 1.0 spec
gives the same synthetic data for the same seed.

## Contents

- [Getting an IR document](#getting-an-ir-document)
- [The shape of a document](#the-shape-of-a-document)
- [From 1.0 to the IR: field by field](#from-10-to-the-ir-field-by-field)
- [Beyond tables](#beyond-tables)
- [The privacy unit](#the-privacy-unit)
- [ODCS and OpenLineage](#odcs-and-openlineage)
- [What this release synthesises](#what-this-release-synthesises)
- [Versions and deprecation](#versions-and-deprecation)

## Getting an IR document

```bash
relsynth ir migrate spec.yaml --out spec.ir.yaml --privacy-unit Customer --public Genre MediaType
relsynth introspect --source sqlite:shop.db --format ir --out shop.ir.yaml
relsynth ir validate shop.ir.yaml              # model rules + JSON Schema (pip install "relsynth[ir]")
relsynth run shop.ir.yaml --target sqlite:synthetic.db
```

`ir migrate` converts the spec, then proves the conversion: the 1.0 spec the IR lowers to must normalise to the
same spec as the original, or the command fails (exit code 2) and lists the differences. YAML comments are not
carried over; keep the original file next to the new one.

In Python:

```python
from relsynth import ir
est = ir.load("shop.ir.yaml")            # an Estate; also reads a 1.0 spec
errs, warns = ir.validate.check(est)
print(ir.dumps(est))                     # canonical YAML
spec, losses = ir.v1.lower(est)          # the 1.0 spec the engine runs, and what it left out
```

## The shape of a document

```yaml
ir_version: "2.0"
name: shop
source: {kind: sqlite, path: shop.db}
containers:   {...}          # optional: where entities live (collection, topic, iceberg_table, graph, ...)
entities:
  orders:
    shape: record            # record | document | node | edge | event
    role: fact               # entity | fact | view
    attributes:
      order_id: {type: integer, kind: id, nullable: false}
      customer_id: {type: integer, kind: id}
      placed_at: {type: timestamp}
      total: {type: number, min: 0, decimals: 2, privacy: {role: sensitive}}
    keys:
      - {attributes: [order_id], primary: true, kind: surrogate}
    time: {event_time: placed_at}
relationships:
  - {name: order_customer, from: {entity: orders, attributes: [customer_id]}, to: {entity: customers}}
rules: [...]
privacy:
  unit: {entity: customers}
```

Each attribute has a **logical type** (`string`, `integer`, `number`, `boolean`, `date`, `timestamp`, `time`,
`binary`, `object`, `array`, `map`) and, for modelling, a **kind** (the 1.0 column type: `id`, `numeric`,
`categorical`, `pii`, `text`, …). A kind fixes its logical type where it can, so most attributes need only one of
the two. Fields you do not know are refused with a suggestion; fields that start with `x-` are yours and are kept.

## From 1.0 to the IR: field by field

| 1.0 spec | IR |
|---|---|
| `version: 1` | `ir_version: "2.0"` |
| `tables.T` | `entities.T` (shape `record`) |
| `role`, `description` | `role`, `description` |
| `mode`, `key_order` | `synthesis.mode`, `synthesis.key_order` |
| `primary_key` | `keys: [{attributes: [...], primary: true}]` |
| `unique`, `partial_unique` | `keys: [{attributes}]`, `keys: [{attributes, where}]` |
| `columns.C` (`type`, options) | `attributes.C` (`kind`, logical `type`, `synthesis`) |
| `fact` | `synthesis.fact` |
| `fact.time_column` | `time.event_time` |
| `scd` | `time.valid_time` + `time.versioning` |
| `dimensions` | `dependencies` |
| `view_of` | `view.of` |
| `bridge` | `allocation` |
| `group_effects`, `link_affinity`, `unknown_members` | under `synthesis` |
| `relationships[]` `child`/`parent` | `relationships[]` `from`/`to` (kind `reference`, `hierarchy` or `derived`; cardinality `many_to_one` or `one_to_one`) |
| `rules[]` `table`/`column`/`from` | `rules[]` `entity`/`attribute`/`source` (a rule with `holds` below 1 is soft) |

Order is kept everywhere (entities, attributes, keys, relationships, rules), because the engine uses it to break
ties: it is part of the meaning, and two documents that differ only in order can give different synthetic data.

## Beyond tables

Five hand-written examples in [examples/ir/](../examples/ir/) validate against the model and the JSON Schema:

| Example | Shows |
|---|---|
| [mongodb_orders.yaml](../examples/ir/mongodb_orders.yaml) | Documents; nested objects; arrays with a length law (`length: {min: 1, max: 50}`); `optional` (field may be absent) as distinct from `nullable`; a payment sub-document whose fields depend on `method` (`variants: {discriminator, cases}`); a reference from a nested path (`customer.customer_id`) |
| [dynamodb_single_table.yaml](../examples/ir/dynamodb_single_table.yaml) | Several entity types in one key-value table, told apart by a discriminator; partition and sort keys built from attribute values (`physical.key_templates`, e.g. `CUSTOMER#{customer_id}`); a line number within an order (`sequence`) |
| [property_graph_movies.yaml](../examples/ir/property_graph_movies.yaml) | Nodes and edges with properties (`shape: edge`, `edge: {source, target, directed, multiple, loops}`) |
| [iceberg_page_views.yaml](../examples/ir/iceberg_page_views.yaml) | Hidden partitions (`physical.partitioning` with transforms such as `day`, `bucket` and `truncate`), `physical.sort_order` |
| [event_stream_clicks.yaml](../examples/ir/event_stream_clicks.yaml) | A topic with event time, ingestion time, a key, ordering and bounded lateness (`stream: {key, order_by, partitions, max_lateness, delivery}`); an optional reference for anonymous sessions |

## The privacy unit

Differential privacy protects a *unit*, such as a customer or a patient, not a row: all the records of one unit
must be covered together. The IR states the unit, and relsynth computes which records belong to it.

```yaml
privacy:
  unit: {entity: Customer}
  exclude_paths: [invoice_support_rep]  # optional: relationship names that do not carry ownership
entities:
  Genre: {privacy: {public: true}, ...}  # holds no data of any unit
```

```bash
relsynth ir privacy chinook.ir.yaml
```

`trace` follows the many-to-one and one-to-one references (not hierarchies, not excluded ones) from each entity to
the unit and classifies every entity:

| Status | Meaning |
|---|---|
| `unit` | the unit entity itself |
| `owned` | each record belongs to the unit(s) its reference paths lead to (Proposition 107); a record has at most as many owners as there are paths |
| `context` | referenced by owned records, but holds no unit's data (products, calendars) |
| `unrelated` | no path either way |

It also lists **cross-unit links**, such as a hierarchy on the unit (a customer who referred another): removing a
unit would leave them dangling, so the privacy design must null or bound them (Proposition 108). For the four
real databases: Chinook and Contoso have the customer as unit (a Contoso sale has two paths: directly and through
its order); AdventureWorks DW has `DimCustomer` (internet sales, survey responses and sales reasons are owned;
employees and resellers are unrelated, and employees are a second population of people); bi/COREP has the
reporting entity (a reported value has two paths). `ir.privacy.contribution(est, data, ...)` maps each record to its
owners on real data.

## ODCS and OpenLineage

```bash
relsynth ir odcs shop.ir.yaml --out shop.odcs.yaml        # Open Data Contract Standard v3.2.0
relsynth ir from-odcs contract.yaml --out shop.ir.yaml    # reads ODCS v3.0.0 to v3.2.0
relsynth ir openlineage shop.ir.yaml --namespace sqlite://synthetic --inputs-namespace sqlite://shop
```

- **ODCS.** Entities become schema objects, attributes properties, primary keys `primaryKey`, references
  `relationships` (`table.column` shorthand), descriptions and privacy roles `classification`. Everything ODCS has
  no field for travels in `customProperties` under the `relsynth` vendor, so `from_odcs(to_odcs(E))` gives back
  `E`. Contracts written by other tools are read from their native fields.
- **OpenLineage** (export only). Each entity becomes a dataset with the Schema (nested fields included),
  DatasetType, Documentation and Tags facets. With `--inputs-namespace`, a ColumnLineage facet says that each
  synthetic column is derived from the real column of the same name (transformation type `DIRECT`, subtype `AGGREGATION`,
  masking true, which is our reading of the facet: the synthetic values are drawn from a model fitted on the real data, not copied).

## What this release synthesises

The engine in 2.x still synthesises relational records. An IR document is *lowered* to its relational part,
which the engine runs exactly as it would the equivalent 1.0 spec. `relsynth ir convert estate.yaml --to v1` shows
the lowered spec and lists each annotation that it leaves out (partitions, privacy roles, …); `--strict` fails
instead. Running a document whose entities are documents, nodes, edges or events, or that has nested attributes,
fails with a message naming them; the engine will read these shapes directly in a later release.

## Versions and deprecation

- **2.x** reads 1.0 specs and IR 2.x documents and runs both the same way. `introspect` writes 1.0 by default;
  `--format ir` writes the IR.
- Reading a 1.0 spec **warns from release 3.0 at the earliest**.
- Reading 1.0 specs **stops no earlier than release 4.0, and no earlier than 24 months after 2.0**.
  `relsynth ir migrate` stays in every release, so an old spec can always be converted.
- **IR versions.** A minor version (2.1, 2.2, …) only adds optional fields, so a 2.0 document stays valid. A major
  version (3.0) may rename or remove fields and ships a migrator from the previous major version.
- The JSON Schema of each version is published with the package
  (`relsynth ir schema`, `src/relsynth/ir/schema/relsynth-ir-2.0.schema.json`) and its `$id` names the version.
