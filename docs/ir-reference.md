# IR 2.0 reference

Generated from `src/relsynth/ir/model.py` by `python -m relsynth.ir.jsonschema reference`; do not edit.
The JSON Schema is `src/relsynth/ir/schema/relsynth-ir-2.0.schema.json`. The guide is [ir-guide.md](ir-guide.md) and the design record [adr/0001-canonical-ir.md](adr/0001-canonical-ir.md).

Every node also accepts vendor extensions: keys starting with `x-`, kept and never interpreted.

## Estate

A data estate: everything relsynth reads, models and writes.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `ir_version` **(required)** | `2.0` | `"2.0"` | Version of the IR format (2.0). |
| `name` | string |  | Label used in reports and file names. |
| `description` | string |  | Free text. |
| `source` | mapping |  | Where the real data is read from (the 1.0 `source` section). |
| `target` | mapping |  | Where the synthetic data is written (the 1.0 `target` section). |
| `containers` | map of [Container](#container) |  | Physical stores shared by several entities. |
| `entities` **(required)** | map of [Entity](#entity) |  | Entities in order (order is kept: it breaks ties in generation order). |
| `relationships` | list of [Relationship](#relationship) |  | References between entities. |
| `rules` | list of [Rule](#rule) |  | Business rules. |
| `privacy` | [EstatePrivacy](#estateprivacy) |  | Privacy unit and adjacency. |
| `synthesis` | mapping |  | relsynth engine settings (the 1.0 `synthesis` section). |
| `review` | list of string |  | What introspect inferred, and its questions. |

## Container

A physical store shared by one or more entities (a DynamoDB table, a MongoDB collection with several document types, a Kafka topic).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `kind` **(required)** | `table` \| `view` \| `collection` \| `key_value_table` \| `topic` \| `iceberg_table` \| `delta_table` \| `file_set` \| `graph` \| `index` |  | table, view, collection, key_value_table, topic, iceberg_table, delta_table, file_set, graph or index. |
| `name` | string |  | Physical name. |
| `system` | string |  | System (postgresql, mongodb, dynamodb, kafka, neo4j, iceberg). |
| `partition_key` | string |  | Partition (hash) key attribute of the stored items. |
| `sort_key` | string |  | Sort (range) key attribute. |
| `indexes` | list of [ContainerIndex](#containerindex) |  | Secondary indexes (global and local). |
| `options` | mapping |  | Store-specific options. |
| `description` | string |  | Free text. |

## ContainerIndex

A secondary index of a container (a DynamoDB GSI, a MongoDB index).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` **(required)** | string |  | Index name. |
| `partition_key` | string |  | Partition (hash) key attribute. |
| `sort_key` | string |  | Sort (range) key attribute. |
| `projection` | string |  | Projected attributes: all, keys_only or include. |

## Entity

A set of records of one type: a table, a document collection (or one document type in it), a graph node label or edge type, or an event stream.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `shape` | `record` \| `document` \| `node` \| `edge` \| `event` | `"record"` | record (flat row), document (nested), node, edge or event. |
| `role` | `entity` \| `fact` \| `view` | `"entity"` | entity (a thing), fact (an event or link whose records exist because of its parents) or view (the join of others, rebuilt). |
| `description` | string |  | Free text. |
| `attributes` | map of [Attribute](#attribute) |  | Attributes in order. |
| `keys` | list of [Key](#key) |  | Primary, unique and filtered unique keys. |
| `dependencies` | list of [Dependency](#dependency) |  | Functional dependencies (implied levels, link determines link). |
| `variants` | [Variants](#variants) |  | Polymorphic records: attributes that depend on a discriminator. |
| `view` | [View](#view) |  | For role view: its base entity. |
| `time` | [Time](#time) |  | Event time, valid time and versioning. |
| `allocation` | [Allocation](#allocation) |  | Weights summing to a total per group (bridge). |
| `edge` | [Edge](#edge) |  | For shape edge: its endpoints. |
| `stream` | [Stream](#stream) |  | For shape event: stream keys and ordering. |
| `physical` | [Physical](#physical) |  | Physical layout. |
| `privacy` | [EntityPrivacy](#entityprivacy) |  | Privacy annotation. |
| `synthesis` | [EntitySynthesis](#entitysynthesis) |  | Engine options. |

## Attribute

A column, a document field, a node or edge property, or an event field.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | `string` \| `integer` \| `number` \| `boolean` \| `date` \| `timestamp` \| `time` \| `binary` \| `object` \| `array` \| `map` |  | Logical data type: string, integer, number, boolean, date, timestamp, time, binary, object (with fields), array (with items), map. |
| `kind` | `id` \| `integer` \| `float` \| `categorical` \| `boolean` \| `date` \| `datetime` \| `text` \| `pii` \| `constant` \| `lookup` \| `binary` \| `sequence` |  | How relsynth models the value (the 1.0 column types): id, integer, float, categorical, boolean, date, datetime, text, pii, constant, lookup, binary, sequence. |
| `nullable` | boolean | `true` | May hold null. |
| `optional` | boolean | `false` | May be absent from a record or document (distinct from null). |
| `description` | string |  | Free text. |
| `min` | any |  | Hard lower bound. |
| `max` | any |  | Hard upper bound. |
| `decimals` | integer |  | Stored precision of a number. |
| `physical_type` | string |  | Type in the source system (VARCHAR(20), NUMBER(10,2), ObjectId). |
| `physical_name` | string |  | Name in the source system when it differs. |
| `virtual` | boolean | `false` | Not stored: the identity of a graph edge's endpoint, or a value computed on load. |
| `lookup` | string or [KeyedLookup](#keyedlookup) |  | Value of a parent's attribute through foreign keys ('fk.col', 'fk.fk2.col'), or a keyed lookup into a reference entity. |
| `map` | mapping |  | Lookup values mapped through a fixed table (country -> currency). |
| `default` | any |  | Value for codes outside `map`. |
| `sequence` | [Sequence](#sequence) |  | For kind sequence: numbered within another attribute. |
| `base_column` | string |  | For views: the base entity's attribute this one copies. |
| `fields` | map of [Attribute](#attribute) |  | For type object: the nested attributes, in order. |
| `items` | [Attribute](#attribute) |  | For type array: the element. |
| `length` | mapping |  | For type array: number of elements {min, max} (multiplicity). |
| `variants` | [Variants](#variants) |  | For type object: polymorphic sub-documents. |
| `privacy` | [AttributePrivacy](#attributeprivacy) |  | Privacy annotation. |
| `synthesis` | [AttributeSynthesis](#attributesynthesis) |  | Engine options. |

## KeyedLookup

A value read from a reference table by matching expressions of this row.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `table` **(required)** | string |  | Reference entity (mode copy, no primary key). |
| `match` **(required)** | mapping |  | Reference attribute -> expression of this row (OrderDate, 'USD', CurrencyCode). |
| `value` **(required)** | string |  | Reference attribute returned. |

## Sequence

Numbers 1..n (or start..) within each value of another attribute.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `within` **(required)** | string |  | Attribute that groups the rows (the order number). |
| `start` | integer |  | First number (relsynth 1.0 default 1). |

## Variants

Polymorphic content: which further attributes a record or object has depends on the value of a discriminator (a tagged union).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `discriminator` **(required)** | string |  | Attribute whose value names the variant (`type`, `kind`). |
| `cases` **(required)** | map of map of [Attribute](#attribute) |  | Discriminator value -> the attributes that variant adds. |
| `open` | boolean | `false` | Values not listed may occur (their extra attributes are not described). |

## AttributePrivacy

What an attribute reveals about the privacy unit.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `role` | `direct_identifier` \| `quasi_identifier` \| `sensitive` \| `non_sensitive` |  | direct_identifier (names the person: e-mail, account number), quasi_identifier (identifies in combination: birth date, postcode), sensitive (a value to protect: diagnosis, salary), non_sensitive. |
| `classification` | string |  | The organisation's classification label (public, internal, confidential, restricted). |
| `special_category` | boolean | `false` | GDPR Art. 9 special category (health, religion, ...). |
| `description` | string |  | Free text. |

## AttributeSynthesis

How relsynth's engine produces the values (1.0 column options).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `distribution` | mapping |  | Draw from a stated law instead of the learned one (1.0 `distribution`). |
| `text` | mapping |  | Options of a text attribute (1.0 `text`). |
| `pii` | mapping |  | Options of a pii attribute (1.0 `pii`: generator, pattern). |
| `offset_from` | string |  | This date/number = that attribute + a modelled gap. |
| `ratio_of` | string |  | This number = that number x a modelled factor (attribute or fk.col path). |
| `ratio_same_as` | string |  | Use the factor of another ratio attribute. |

## Key

A key: attributes no two records share.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `attributes` **(required)** | list of string |  | Key attributes, in order (several for a composite key). |
| `primary` | boolean | `false` | The primary key (one per entity). |
| `kind` | `surrogate` \| `natural` |  | surrogate (generated, no business meaning) or natural (a business identifier). |
| `where` | string |  | Rows the key applies to (a filtered unique index), as a rule expression. |
| `name` | string |  | Name of the constraint. |

## Dependency

A functional dependency inside an entity: the determinant's value fixes the dependents' values (GeoAreaKey -> State, Country; a track's album fixes its genre: a link that determines another link).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `determinant` **(required)** | string |  | The determining attribute (a code or a foreign key). |
| `dependents` **(required)** | list of string |  | Attributes it determines. |
| `anchor` | string |  | A coarse dependent kept as a modelled attribute (Country), or the determinant itself for a link that determines other links. |

## View

An entity that is the join of others, rebuilt from them.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `of` **(required)** | string |  | The base entity the view's rows come from. |

## Time

The time semantics of an entity's records.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `event_time` | string |  | When the event happened (a sale's date, an event's timestamp). |
| `valid_time` | [ValidTime](#validtime) |  | Valid-time period of each record. |
| `versioning` | [Versioning](#versioning) |  | Version history (needs valid_time). |
| `ingestion_time` | string |  | When the record reached the store (processing time of a stream). |

## ValidTime

The period during which a record is true in the world (SQL:2011 application time).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `from` **(required)** | string |  | Start attribute. |
| `to` | string |  | End attribute (null on the open, current version). |

## Versioning

Records are versions of a natural key (type 2 slowly changing dimension).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `natural_key` **(required)** | string |  | The business key shared by the versions. |
| `scd_type` | any |  | Slowly-changing-dimension type as written in a 1.0 spec (2). |
| `stable` | any |  | auto, or the attributes repeated across versions. |
| `current` | mapping |  | Current flag {column, value, other}. |

## PartitionField

A partition of the physical layout: a transform of a source attribute (Iceberg hidden partitioning; identity = Hive-style partition column).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `source` **(required)** | string |  | Source attribute. |
| `transform` | `identity` \| `year` \| `month` \| `day` \| `hour` \| `bucket` \| `truncate` \| `void` | `"identity"` | identity, year, month, day, hour, bucket, truncate or void. |
| `param` | integer |  | bucket: number of buckets; truncate: width. |
| `name` | string |  | Partition name (default <source>_<transform>). |

## SortField

One field of a sort order: a source attribute, a transform, a direction and null placement.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `source` **(required)** | string |  | Source attribute. |
| `transform` | `identity` \| `year` \| `month` \| `day` \| `hour` \| `bucket` \| `truncate` \| `void` | `"identity"` | Transform applied before sorting. |
| `direction` | `asc` \| `desc` | `"asc"` | asc or desc. |
| `nulls` | `first` \| `last` | `"first"` | first or last. |

## Discriminator

How the records of this entity are told apart in a container shared by several entities.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `attribute` **(required)** | string |  | Attribute holding the entity's type. |
| `value` **(required)** | any |  | Its value for this entity. |

## Physical

Where and how an entity is stored.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `container` | string |  | Name of a container in `containers` (shared tables, topics, collections). |
| `name` | string |  | Physical name (schema.table, collection, topic). |
| `format` | string |  | Storage format: parquet, orc, avro, csv, json, bson, delta, iceberg. |
| `location` | string |  | Location (s3://..., abfss://..., a folder). |
| `partitioning` | list of [PartitionField](#partitionfield) |  | Partition spec. |
| `sort_order` | list of [SortField](#sortfield) |  | Sort order within files or partitions. |
| `clustering` | list of string |  | Clustering columns (Delta liquid clustering, Snowflake clustering keys). |
| `discriminator` | [Discriminator](#discriminator) |  | Type attribute in a shared container (single-table design). |
| `key_templates` | map of string |  | Container key attribute -> template built from this entity's attributes ('CUSTOMER#{customer_id}'). |
| `options` | mapping |  | Store-specific options (table properties, retention, TTL). |

## Edge

A graph edge type: each record joins a source node to a target node.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `source` **(required)** | string |  | Relationship (from this entity) to the source node. |
| `target` **(required)** | string |  | Relationship (from this entity) to the target node. |
| `directed` | boolean | `true` | Edges have a direction. |
| `multiple` | boolean | `true` | Several edges may join the same two nodes. |
| `loops` | boolean | `false` | An edge may join a node to itself. |

## Stream

An event stream (a topic or log): records arrive over time per key.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `key` | list of string |  | Message key attributes (the partitioning key). |
| `order_by` | string |  | Attribute that orders the events of one key (event time or a sequence number). |
| `partitions` | integer |  | Number of partitions of the topic. |
| `max_lateness` | string |  | Bound on event-time disorder (ISO 8601 duration, PT5M): the watermark lag. |
| `delivery` | `at_most_once` \| `at_least_once` \| `exactly_once` |  | at_most_once, at_least_once or exactly_once. |

## Allocation

Weights that sum to a total over the records of a group (a bridge table).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `group` **(required)** | string |  | Attribute whose records share the weights (customer). |
| `weight` **(required)** | string |  | Weight attribute. |
| `total` | number |  | What the weights of a group sum to (1.0 default 1). |
| `decimals` | integer |  | Stored precision; the rounding remainder goes to the largest weight. |

## EntityPrivacy

Privacy annotations of an entity.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `public` | boolean | `false` | Holds no data about the privacy unit and may be released as it is (code lists, calendars, a product catalogue). |
| `description` | string |  | Free text. |

## EntitySynthesis

relsynth engine options of an entity (1.0 table options).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `synthesize` \| `copy` |  | synthesize or copy. |
| `key_order` | string |  | Keys numbered in the order of this attribute. |
| `fact` | [FactModel](#factmodel) |  | How fact records are generated. |
| `group_effects` | list of [GroupEffect](#groupeffect) |  | Attributes shared by the records of a group. |
| `link_affinity` | list of [LinkAffinity](#linkaffinity) |  | Links that point at matching parents more often than chance. |
| `unknown_members` | list |  | Key values of records that stand for 'no member' (-1 'Unknown'). |

## FactModel

How fact records are generated (1.0 `fact`, without time_column, which is time.event_time).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `anchor` | string |  | Parent whose records generate the facts: an entity, a relationship name or a foreign-key attribute. |
| `partner` | string |  | Second dimension of the block model (same forms as anchor). |
| `pairs` | `auto` \| `poisson` \| `occupancy` |  | auto, poisson or occupancy. |
| `multiplicity` | `auto` \| `none` \| `partner` \| `anchor` \| `block` |  | auto, none, partner, anchor or block. |
| `count_model` | `auto` \| `poisson` \| `dp` |  | auto, poisson or dp. |
| `group_fallback` | `auto` \| `anchor` \| `partner` |  | auto, anchor or partner. |
| `time_bucket` | `day` \| `week` \| `month` \| `quarter` \| `year` |  | Bucket of the seasonal profile of event_time. |
| `condition_on_parents` | boolean or list of string |  | Condition measures on parents' continuous attributes. |
| `header` | [Header](#header) |  | Order header carried on the lines. |

## Header

A document header in a fact: the document number, what it determines, and the line number.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `key` | string |  | Document number (SalesOrderNumber). |
| `columns` | list of string |  | Attributes the document number determines. |
| `line_number` | string |  | Line number attribute, generated 1..n per document. |

## GroupEffect

An attribute whose value is shared by the records of a group.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `attribute` **(required)** | string |  | Attribute shared by the records of a group. |
| `by` **(required)** | list of string |  | Attributes or fk.col paths that define a group. |

## LinkAffinity

Two link paths whose values match more often than chance.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `match` **(required)** | list of string |  | Two paths whose values match more often than chance. |
| `share` | number |  | Matching share to reach (default: learned). |

## Relationship

A reference between entities: a foreign key (composite allowed), a hierarchy inside one entity, a one-to-one extension, or a many-to-many link through a bridge.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` **(required)** | string |  | Unique name, used by rules, fact anchors and edges. |
| `from` **(required)** | [End](#end) |  | Referencing side (the child: holds the foreign key). |
| `to` **(required)** | [End](#end) |  | Referenced side (the parent); attributes default to its primary key. |
| `kind` | `reference` \| `hierarchy` \| `derived` | `"reference"` | reference (between entities), hierarchy (parent-child inside one entity) or derived (internal: offset keys). |
| `cardinality` | `many_to_one` \| `one_to_one` \| `many_to_many` | `"many_to_one"` | many_to_one, one_to_one (the referencing attributes are a key) or many_to_many (through `via`). |
| `optional` | boolean |  | A record may have no parent (null reference); learned when absent. |
| `offset` | [Offset](#offset) |  | Gap-modelled second reference into an ordered parent. |
| `via` | [Via](#via) |  | For many_to_many: the bridge. |
| `description` | string |  | Free text. |

## End

One end of a relationship: an entity and its attributes.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `entity` **(required)** | string |  | Entity. |
| `attributes` | list of string |  | Attributes (several for a composite key; fk.col paths allowed on the referencing side). |

## Offset

A second reference into the same ordered parent whose gap is modelled (ship date from order date).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `from` **(required)** | string |  | The base reference attribute of the same entity. |
| `order_by` **(required)** | string |  | Parent attribute that orders it (the calendar's date). |

## Via

For many_to_many: the bridge entity and its two references.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `entity` **(required)** | string |  | Bridge entity. |
| `from` **(required)** | string |  | Relationship from the bridge to the `from` entity. |
| `to` **(required)** | string |  | Relationship from the bridge to the `to` entity. |

## Rule

A business rule. Hard rules (holds 1) hold on every record; soft rules on a share.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `entity` **(required)** | string |  | Entity the rule is on. |
| `type` **(required)** | `check` \| `derive` \| `aggregate` \| `cumulative` |  | check (must hold), derive (attribute = expression), aggregate (parent attribute from its children) or cumulative (running balance). |
| `name` | string |  | Name. |
| `expr` | string |  | check / derive expression. |
| `holds` | number |  | derive: share of records on which it holds (below 1: a soft rule). |
| `violation` | `offset` \| `model` |  | Soft rule: offset or model. |
| `tolerance` | number |  | derive: accepted absolute difference. |
| `attribute` | string |  | aggregate / cumulative: the computed attribute. |
| `agg` | `count` \| `sum` \| `min` \| `max` \| `mean` \| `exists` \| `count_distinct` |  | aggregate function. |
| `source` | string |  | aggregate: the child entity. |
| `of` | string |  | aggregate: what is aggregated. |
| `via` | string |  | aggregate: relationship name when there are several. |
| `by` | string or list of string |  | cumulative: the group. |
| `order` | string |  | cumulative: the order. |
| `step` | string |  | cumulative: change per record. |
| `opening` | number |  | cumulative: first value of each group. |
| `floor` | number |  | cumulative: lowest allowed balance. |
| `description` | string |  | Free text. |

## EstatePrivacy

Privacy settings of the estate: the unit, the adjacency and the paths not followed.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `unit` | [PrivacyUnit](#privacyunit) |  | The privacy unit. |
| `adjacency` | `add_remove` \| `replace` | `"add_remove"` | add_remove (a unit and all its data present or absent) or replace. |
| `exclude_paths` | list of string |  | Relationships not followed when tracing contributions (a shared lookup). |
| `description` | string |  | Free text. |

## PrivacyUnit

Whose privacy is protected: neighbouring datasets differ in all the data of one unit (Ponomareva et al. 2026).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `entity` **(required)** | string |  | Entity whose records are the units (Customer). |
| `description` | string |  | Free text (a customer and everything they bought). |
