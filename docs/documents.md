# Document databases

relsynth synthesises document collections from MongoDB (and Cosmos DB for MongoDB, Amazon DocumentDB), Azure Cosmos
DB for NoSQL, Amazon DynamoDB (including single-table designs), Couchbase and JSON files. The synthetic documents keep
the shape of the source: optional fields, fields of several types, nested objects, arrays with their length law,
polymorphic sub-documents, maps keyed by identifiers, and references between collections. Release 2.1; design in
[ADR 0011](adr/0011-document-databases.md), propositions 1400–1405 in the [proofs](proofs/).

## Install

```bash
pip install "relsynth[documents]"     # pymongo, azure-cosmos, boto3
pip install "relsynth[couchbase]"     # the Couchbase SDK
```

## Quick start

```bash
# the schema with counts, as an IR 2.0 document
relsynth docs discover jsonl:data/shop --out shop.ir.json

# synthesise every collection of a MongoDB database into JSON Lines, with the quality report
relsynth docs synth source.yaml --to jsonl:synthetic/ --report report.json

# score synthetic collections against the real ones
relsynth docs check jsonl:data/shop jsonl:synthetic/ --report check.json
```

`synth` and `check` exit with code 1 (the quality gate, as for `relsynth run --fail-under`) when a document does not validate, a reference does not
resolve or a shape metric is outside its limit.

In Python:

```python
from relsynth.documents import DocumentSynthesizer

ds = DocumentSynthesizer.from_source({"kind": "mongodb", "url_env": "MONGO_URL", "database": "shop"}, seed=7)
ds.discover()                    # profiles, references, layout (ds.schema_ir() for the IR)
out = ds.synthesize(scale=1.0)   # {collection: [documents]}
rep = ds.report(out)             # validity, references, repairs, shape metrics with limits
ds.write(out, {"kind": "jsonl", "path": "synthetic/"})
```

`DocumentSynthesizer(collections, references=..., seed=..., workdir=..., spec_overrides=...)` also takes the
collections as lists of documents. `spec_overrides` is merged into the relational spec, so engine settings,
generators and DP options apply as usual. The shredded tables (`shredded.db`) and the spec (`spec.yaml`) stay in
`workdir`.

## Sources and targets

A location is either a short string or a YAML section:

| Store | Short form | YAML section |
|---|---|---|
| JSON files (Extended JSON v2, one file per collection, `.json` arrays or `.jsonl`) | `jsonl:<folder>` | `{kind: jsonl, path: <folder>, relaxed: false}` |
| MongoDB, Cosmos DB for MongoDB, DocumentDB | `mongodb://host/db` (no password) | `{kind: mongodb, url_env: MONGO_URL, database: shop, options: {...}}` |
| Cosmos DB for NoSQL | `cosmos:<database>` | `{kind: cosmos, url_env: COSMOS_URL, key_env: COSMOS_KEY, database: shop}` |
| DynamoDB | `dynamodb:<table>` | `{kind: dynamodb, table: shop, region_name: eu-west-1, discriminator: entity_type}` |
| Couchbase | | `{kind: couchbase, url_env: CB_URL, username: app, password_env: CB_PASSWORD, bucket: shop, scope: _default}` |

Secrets come from `*_env` (an environment variable) or `*_file` (a file), as for the relational connectors. A
MongoDB URL with a password is refused on the command line. `collections: [...]` limits a source to some
collections. The Cassandra and Gremlin APIs of Cosmos DB are not supported.

Values keep their BSON type: `objectId`, `date`, `long`, `decimal`, `binary` and the rest are read from the drivers
(or from Extended JSON) and written back with the same type. Cosmos DB system properties (`_rid`, `_etag`, `_ts`, ...)
are dropped on read.

## What is inferred

`discover` profiles each collection by path: how many objects hold the field, the count of each type (null, bool,
int, long, double, decimal, string, objectId, date, binary, object, array), array lengths, the distinct values, the
set of fields of each object (its signature) and the depth. From these counts it finds:

- **maps**: an object whose keys look like identifiers (many keys, each in few documents), stored as key-value
  pairs; a key equal to a field of its value (`wallets[<id>].id == <id>`) is rebuilt from that field;
- **variants**: an object with a discriminator (`payment.method`) whose values decide the field set;
- **length fields**: an integer field equal to an array's length on every document (`n_items == len(items)`);
- **set arrays**: arrays of scalars whose elements never repeat;
- **references**: a field whose values are all values of a unique field of another collection (`orders.customer_id`
  to `customers._id`, or to a natural key). Partial or near-unique matches are listed as *candidates* with their
  coverage and are not enforced. Integers and non-hex strings need a name match (`customer_id`, `customer`) to count;
- **sequences**: a field numbered 1..n within the documents of one parent (`line_no`).

Declare a reference the profile cannot see with `--references refs.yaml` (`[{collection, path, to, to_path}]`).

`relsynth docs discover` writes the result as an IR 2.0 document (shape `document`, fields, variants, arrays with
length laws, references), with the counts in `x-relsynth-counts` and maps in `x-relsynth-map`. Every source document
validates against the inferred schema (Prop. 1400).

## How it is synthesised

1. **Shred.** Each collection becomes a root table; each array, map and optional sub-document becomes a child table
   with `_parent` and `_pos` (or `_key`). Absent, null and the type of a union field are a tag column `<path>__t`,
   an array's length is a column `<path>__n` modelled with its parent's other columns, and numbers of several types
   share one column. A set array of one scalar type with at most 32 distinct values (tags of an item, movie genres) is not a child
   table: it is one integer column per value on its holder (`<path>__h<i>`, 0 absent, else 1 + its position), so
   which values a document holds is modelled jointly with its other fields, and a value never repeats (Prop. 1785). The tables go to a SQLite work file with every key declared, and `introspect` drafts the spec.
   `nest(shred(D))` gives `D` back exactly (Prop. 1401).
2. **Synthesise** with the relational engine: a Bayesian copula per table, latent segments across tables, exact keys
   and links, and a Dirichlet-process law for the number of elements per parent.
3. **Nest.** Elements are attached to their parents to match the modelled lengths and presence tags; documents get
   fresh `_id`s (deterministic ObjectIds from the seed); references point to synthetic documents.
4. **Conform.** Every synthetic document is made to validate against the schema (Prop. 1402). A value that breaks it
   (an array outside the source's length range, a repeated element in a set, a field outside its variant, a missing
   required field) is replaced by a value of another *synthetic* document at the same path, never by a source value. An optional field that no synthetic value can repair is dropped.
   Every repair is counted by kind and path in the report (`repairs`); many repairs of one kind say the model misses
   a dependence. Set arrays with a larger vocabulary (mflix genres, languages) are the common case: duplicates the
   engine draws are resampled without replacement and counted as `set` repairs.

## Quality report

`report` (and `relsynth docs check`) gives, per collection:

| Metric | What it compares |
|---|---|
| `presence` | mean absolute difference of the presence rate of each field path (present / parent objects) |
| `presence_max` | the largest such difference |
| `type_mix` | mean total variation distance (TVD) of the state (absent, null, each type) of each path with more than one state |
| `array_length` | mean TVD of the length histograms of the array and map paths |
| `signature` | TVD of the field sets of the documents (which top-level fields a document holds together) |
| `depth` | TVD of the nesting depth of the documents |
| `size` | TVD of the number of scalar values per document, in log2 bins |

Each limit is the one-sided 99% prediction bound of the same distance between random halves of the real documents
(`--splits`, default 10), rescaled to the compared sizes and to the variance of a posterior predictive draw (Props.
109, 1204 and 1403), the method of the quality report 2.0. Splits keep whole documents. Validity (share of
documents that validate) and references (share of synthetic reference values that resolve) must be 1.

## DynamoDB single-table designs

With `dynamodb:<table>`, relsynth reads the key schema and GSIs and splits the items into item types, by a type
attribute (`entity_type`, `type`, ... or `discriminator:`) or by the partition-key prefix. For each key attribute it
learns a template (`PK = CUSTOMER#{customer_id}`, `SK = ORDER#{ordered_at}#{order_id}`): every `#`-separated part is
a constant or one attribute of the item. The types are synthesised as collections with references between them, and
the keys are rendered back from the templates, so the synthetic table has unique keys and the same access patterns
(Prop. 1404). `relsynth.documents.dynamo.check` reports duplicate keys and template violations;
`dynamo.to_ir` gives the design as an IR document. [examples/documents/dynamodb_shop.py](../examples/documents/dynamodb_shop.py)
generates the customer-order-item example of the AWS documentation.

## Fitting inside MongoDB

`relsynth.documents.connectors.profile_with_pushdown(con, collection, sample_size)` computes the exact counts in the
database with one `$group` pipeline (documents per `$type` per path and the `$size` histogram of every array) and reads
only a `$sample` of documents for the rest of the profile. Only these aggregates and the sample leave the database.

## Acceptance

`benchmarks/documents/run_documents.py` runs the whole path on the MongoDB sample datasets (downloaded, never
committed: their licence is to be checked before redistribution) and on StackExchange dumps (CC BY-SA 4.0; questions
with their answers and comments embedded, linked to users). With `--mongo-url` it also loads each set into MongoDB and
compares the pushdown profile with the in-memory one. CI runs it in the `documents` job.

## Limits

- Dependence across levels (an element's value given its document's values) is the engine's parent-child model:
  the segments of the parent and its modelled columns. Values deep in a document depend on their own table and parent.
- A whole collection is shredded in memory; the pushdown profile bounds what is read for inference only.
- A direct document generator was studied and not adopted for now
  ([spike](research/document-generator-spike.md)).
