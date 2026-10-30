# SPDX-License-Identifier: Apache-2.0
"""The object model of the canonical intermediate representation (IR 2.0).

One `Estate` describes a whole data estate: its entities (tables, document
collections, graph nodes and edges, event streams), their attributes, keys and
relationships, time, physical layout, rules and privacy annotations, and the
synthesis settings relsynth reads.  Every engine, connector and report reads
this model; a relsynth 1.0 YAML spec converts to it and back without loss
(`relsynth.ir.v1`).

Each class is a dataclass whose fields carry their own documentation and JSON
names in the field metadata, so that the JSON Schema
(`relsynth.ir.jsonschema`) and the reference (`docs/ir-reference.md`) are
generated from this file and cannot drift from what the code reads.

Serialised form: a mapping with the field names below (the JSON name where it
differs).  Fields at their default are left out when written.  Keys that start
with ``x-`` are vendor extensions: kept on every node, never interpreted.
"""
from __future__ import annotations

import dataclasses
import difflib
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Union

from ..spec import SpecError

IR_VERSION = "2.0"
SUPPORTED_VERSIONS = ("2.0",)

# modality-neutral logical types (the ODCS v3 logical types, plus binary)
LOGICAL_TYPES = ("string", "integer", "number", "boolean", "date", "timestamp", "time", "binary",
                 "object", "array", "map")
# how relsynth models a value: the column types of the 1.0 spec
KINDS = ("id", "integer", "float", "categorical", "boolean", "date", "datetime", "text", "pii", "constant",
         "lookup", "binary", "sequence")
SHAPES = ("record", "document", "node", "edge", "event")
ROLES = ("entity", "fact", "view")
REL_KINDS = ("reference", "hierarchy", "derived")
CARDINALITIES = ("many_to_one", "one_to_one", "many_to_many")
KEY_KINDS = ("surrogate", "natural")
PRIVACY_ROLES = ("direct_identifier", "quasi_identifier", "sensitive", "non_sensitive")
ADJACENCY = ("add_remove", "replace")
# partition transforms of the Apache Iceberg table spec (v2), which also cover
# Hive-style identity partitions and Delta generated columns
TRANSFORMS = ("identity", "year", "month", "day", "hour", "bucket", "truncate", "void")
CONTAINER_KINDS = ("table", "view", "collection", "key_value_table", "topic", "iceberg_table", "delta_table",
                   "file_set", "graph", "index")
RULE_TYPES = ("check", "derive", "aggregate", "cumulative")

# the logical type a 1.0 kind stores, where the kind says it
KIND_TYPE = {"integer": "integer", "float": "number", "boolean": "boolean", "date": "date",
             "datetime": "timestamp", "text": "string", "pii": "string", "binary": "binary",
             "sequence": "integer"}
# the kind used for a logical type when an attribute states only its type
TYPE_KIND = {"integer": "integer", "number": "float", "boolean": "boolean", "date": "date",
             "timestamp": "datetime", "time": "categorical", "string": "categorical", "binary": "binary"}


class IRError(SpecError):
    """The document is not a valid IR document (a SpecError: CLI exit code 2)."""

    def __init__(self, errors):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("IR document has errors:\n  - " + "\n  - ".join(self.errors))


def _f(doc, *, default=None, factory=None, key=None, required=False, enum=None, schema=None):
    """A documented field.  key: the JSON name when it differs from the field name
    (from, to, ...); enum: allowed values; schema: a JSON Schema fragment that
    replaces the one derived from the annotation."""
    md = {"doc": doc, "key": key, "required": required, "enum": enum, "schema": schema}
    if factory is not None:
        return field(default_factory=factory, metadata=md)
    return field(default=default, metadata=md)


@dataclass
class Node:
    """Base class: generic reading (with unknown-key hints) and writing."""

    extensions: dict = field(default_factory=dict, metadata={
        "doc": "Vendor extensions: keys starting with `x-`, kept as they are and never interpreted.",
        "key": None, "required": False, "enum": None, "schema": None, "extensions": True})

    # -- reading ---------------------------------------------------------------
    @classmethod
    def node_fields(cls):
        return [f for f in dataclasses.fields(cls) if not f.metadata.get("extensions")]

    @classmethod
    def from_dict(cls, d, where: str = "", errs: list | None = None):
        """Build the node from a mapping.  With `errs` given, problems are appended
        and the node is built from what is valid; without it they raise IRError."""
        own = errs is None
        if errs is None:
            errs = []
        where = where or cls.__name__
        if not isinstance(d, dict):
            errs.append(f"{where}: must be a mapping, not {type(d).__name__}")
            if own:
                raise IRError(errs)
            return None
        hints = typing.get_type_hints(cls)
        known = {(f.metadata.get("key") or f.name): f for f in cls.node_fields()}
        kw, ext = {}, {}
        for k, v in d.items():
            if isinstance(k, str) and k.startswith("x-"):
                ext[k] = v
                continue
            f = known.get(k)
            if f is None:
                errs.append(f"{where}: unknown key '{k}'{_hint(k, known)}")
                continue
            kw[f.name] = _convert(hints[f.name], v, f"{where}.{k}", errs, f.metadata.get("enum"))
        for k, f in known.items():
            if f.metadata.get("required") and kw.get(f.name) is None:
                errs.append(f"{where}: '{k}' is required")
        node = cls(**kw)
        node.extensions = ext
        if own and errs:
            raise IRError(errs)
        return node

    # -- writing ---------------------------------------------------------------
    def to_dict(self) -> dict:
        out = {}
        for f in self.node_fields():
            v = getattr(self, f.name)
            if f.default_factory is not dataclasses.MISSING:
                if not v:
                    continue
            elif v == f.default and not f.metadata.get("required"):
                continue
            out[f.metadata.get("key") or f.name] = _plain(v)
        out.update(self.extensions or {})
        return out


def _hint(k, allowed):
    m = difflib.get_close_matches(str(k), list(allowed), n=1, cutoff=0.6)
    return f" (did you mean '{m[0]}'?)" if m else f" (allowed: {', '.join(sorted(allowed))})"


def _plain(v):
    if isinstance(v, Node):
        return v.to_dict()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v


def _is_union(tp):
    return typing.get_origin(tp) in (Union, types.UnionType)


def _matches(tp, v) -> bool:
    """Whether v has the outer shape of tp (used to pick a union member)."""
    if tp is Any:
        return True
    o = typing.get_origin(tp)
    if o is list:
        return isinstance(v, list)
    if o is dict:
        return isinstance(v, dict)
    if isinstance(tp, type) and issubclass(tp, Node):
        return isinstance(v, dict)
    if tp is float:
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if tp is int:
        return isinstance(v, int) and not isinstance(v, bool)
    if tp is type(None):
        return v is None
    return isinstance(v, tp)


def _convert(tp, v, where, errs, enum=None):
    if _is_union(tp):
        alts = typing.get_args(tp)
        for a in alts:
            if _matches(a, v):
                return _convert(a, v, where, errs, enum)
        names = " or ".join(_type_name(a) for a in alts if a is not type(None))
        errs.append(f"{where}: must be {names}, not {type(v).__name__}")
        return None
    if tp is Any:
        return v
    o = typing.get_origin(tp)
    if o is list:
        if not isinstance(v, list):
            errs.append(f"{where}: must be a list, not {type(v).__name__}")
            return []
        (it,) = typing.get_args(tp) or (Any,)
        return [_convert(it, x, f"{where}[{i}]", errs, enum) for i, x in enumerate(v)]
    if o is dict or tp is dict:
        if not isinstance(v, dict):
            errs.append(f"{where}: must be a mapping, not {type(v).__name__}")
            return {}
        args = typing.get_args(tp)
        if not args:
            return v
        return {k: _convert(args[1], x, f"{where}.{k}", errs) for k, x in v.items()}
    if isinstance(tp, type) and issubclass(tp, Node):
        return tp.from_dict(v, where, errs)
    if not _matches(tp, v):
        errs.append(f"{where}: must be {_type_name(tp)}, not {type(v).__name__} ({v!r})")
        return None
    if enum is not None and v not in enum:
        errs.append(f"{where}: {v!r} is not one of {', '.join(map(str, enum))}{_hint(v, enum)}")
    return v


def _type_name(tp):
    if isinstance(tp, type) and issubclass(tp, Node):
        return f"a {tp.__name__} mapping"
    o = typing.get_origin(tp)
    if o is list:
        return "a list"
    if o is dict or tp is dict:
        return "a mapping"
    return {str: "a string", int: "an integer", float: "a number", bool: "true or false"}.get(tp, str(tp))


# ---------------------------------------------------------------------------
# attributes
# ---------------------------------------------------------------------------

@dataclass
class KeyedLookup(Node):
    """A value read from a reference table by matching expressions of this row."""
    table: str | None = _f("Reference entity (mode copy, no primary key).", required=True)
    match: dict | None = _f("Reference attribute -> expression of this row (OrderDate, 'USD', CurrencyCode).",
                            required=True)
    value: str | None = _f("Reference attribute returned.", required=True)


@dataclass
class Sequence(Node):
    """Numbers 1..n (or start..) within each value of another attribute."""
    within: str | None = _f("Attribute that groups the rows (the order number).", required=True)
    start: int | None = _f("First number (relsynth 1.0 default 1).")


@dataclass
class Variants(Node):
    """Polymorphic content: which further attributes a record or object has depends
    on the value of a discriminator (a tagged union)."""
    discriminator: str | None = _f("Attribute whose value names the variant (`type`, `kind`).", required=True)
    cases: dict[str, dict[str, Attribute]] = _f(
        "Discriminator value -> the attributes that variant adds.", factory=dict, required=True)
    open: bool = _f("Values not listed may occur (their extra attributes are not described).", default=False)


@dataclass
class AttributePrivacy(Node):
    """What an attribute reveals about the privacy unit."""
    role: str | None = _f("direct_identifier (names the person: e-mail, account number), quasi_identifier "
                          "(identifies in combination: birth date, postcode), sensitive (a value to protect: "
                          "diagnosis, salary), non_sensitive.", enum=PRIVACY_ROLES)
    classification: str | None = _f("The organisation's classification label (public, internal, confidential, "
                                    "restricted).")
    special_category: bool = _f("GDPR Art. 9 special category (health, religion, ...).", default=False)
    description: str | None = _f("Free text.")


@dataclass
class AttributeSynthesis(Node):
    """How relsynth's engine produces the values (1.0 column options)."""
    distribution: dict | None = _f("Draw from a stated law instead of the learned one (1.0 `distribution`).")
    text: dict | None = _f("Options of a text attribute (1.0 `text`).")
    pii: dict | None = _f("Options of a pii attribute (1.0 `pii`: generator, pattern).")
    offset_from: str | None = _f("This date/number = that attribute + a modelled gap.")
    ratio_of: str | None = _f("This number = that number x a modelled factor (attribute or fk.col path).")
    ratio_same_as: str | None = _f("Use the factor of another ratio attribute.")


@dataclass
class Attribute(Node):
    """A column, a document field, a node or edge property, or an event field."""
    type: str | None = _f("Logical data type: string, integer, number, boolean, date, timestamp, time, binary, "
                          "object (with fields), array (with items), map.", enum=LOGICAL_TYPES)
    kind: str | None = _f("How relsynth models the value (the 1.0 column types): id, integer, float, categorical, "
                          "boolean, date, datetime, text, pii, constant, lookup, binary, sequence.", enum=KINDS)
    nullable: bool = _f("May hold null.", default=True)
    optional: bool = _f("May be absent from a record or document (distinct from null).", default=False)
    description: str | None = _f("Free text.")
    min: Any = _f("Hard lower bound.", schema={"type": ["number", "string"]})
    max: Any = _f("Hard upper bound.", schema={"type": ["number", "string"]})
    decimals: int | None = _f("Stored precision of a number.")
    physical_type: str | None = _f("Type in the source system (VARCHAR(20), NUMBER(10,2), ObjectId).")
    physical_name: str | None = _f("Name in the source system when it differs.")
    virtual: bool = _f("Not stored: the identity of a graph edge's endpoint, or a value computed on load.",
                       default=False)
    lookup: str | KeyedLookup | None = _f(
        "Value of a parent's attribute through foreign keys ('fk.col', 'fk.fk2.col'), or a keyed lookup into a "
        "reference entity.")
    map: dict | None = _f("Lookup values mapped through a fixed table (country -> currency).")
    default: Any = _f("Value for codes outside `map`.")
    sequence: Sequence | None = _f("For kind sequence: numbered within another attribute.")
    base_column: str | None = _f("For views: the base entity's attribute this one copies.")
    fields: dict[str, Attribute] = _f("For type object: the nested attributes, in order.", factory=dict)
    items: Attribute | None = _f("For type array: the element.")
    length: dict | None = _f("For type array: number of elements {min, max} (multiplicity).",
                             schema={"type": "object", "additionalProperties": False,
                                     "properties": {"min": {"type": "integer", "minimum": 0},
                                                    "max": {"type": "integer", "minimum": 0}}})
    variants: Variants | None = _f("For type object: polymorphic sub-documents.")
    privacy: AttributePrivacy | None = _f("Privacy annotation.")
    synthesis: AttributeSynthesis | None = _f("Engine options.")

    @property
    def effective_kind(self):
        return self.kind or TYPE_KIND.get(self.type)

    @property
    def effective_type(self):
        return self.type or KIND_TYPE.get(self.kind)


# ---------------------------------------------------------------------------
# entities
# ---------------------------------------------------------------------------

@dataclass
class Key(Node):
    """A key: attributes no two records share."""
    attributes: list[str] = _f("Key attributes, in order (several for a composite key).", factory=list,
                               required=True)
    primary: bool = _f("The primary key (one per entity).", default=False)
    kind: str | None = _f("surrogate (generated, no business meaning) or natural (a business identifier).",
                          enum=KEY_KINDS)
    where: str | None = _f("Rows the key applies to (a filtered unique index), as a rule expression.")
    name: str | None = _f("Name of the constraint.")


@dataclass
class Dependency(Node):
    """A functional dependency inside an entity: the determinant's value fixes the
    dependents' values (GeoAreaKey -> State, Country; a track's album fixes its
    genre: a link that determines another link)."""
    determinant: str | None = _f("The determining attribute (a code or a foreign key).", required=True)
    dependents: list[str] = _f("Attributes it determines.", factory=list, required=True)
    anchor: str | None = _f("A coarse dependent kept as a modelled attribute (Country), or the determinant "
                            "itself for a link that determines other links.")


@dataclass
class View(Node):
    """An entity that is the join of others, rebuilt from them."""
    of: str | None = _f("The base entity the view's rows come from.", required=True)


@dataclass
class ValidTime(Node):
    """The period during which a record is true in the world (SQL:2011 application time)."""
    from_: str | None = _f("Start attribute.", key="from", required=True)
    to: str | None = _f("End attribute (null on the open, current version).")


@dataclass
class Versioning(Node):
    """Records are versions of a natural key (type 2 slowly changing dimension)."""
    natural_key: str | None = _f("The business key shared by the versions.", required=True)
    scd_type: Any = _f("Slowly-changing-dimension type as written in a 1.0 spec (2).", schema={"enum": [2, "2"]})
    stable: Any = _f("auto, or the attributes repeated across versions.",
                     schema={"oneOf": [{"const": "auto"}, {"type": "array", "items": {"type": "string"}}]})
    current: dict | None = _f("Current flag {column, value, other}.")


@dataclass
class Time(Node):
    """The time semantics of an entity's records."""
    event_time: str | None = _f("When the event happened (a sale's date, an event's timestamp).")
    valid_time: ValidTime | None = _f("Valid-time period of each record.")
    versioning: Versioning | None = _f("Version history (needs valid_time).")
    ingestion_time: str | None = _f("When the record reached the store (processing time of a stream).")


@dataclass
class PartitionField(Node):
    """A partition of the physical layout: a transform of a source attribute
    (Iceberg hidden partitioning; identity = Hive-style partition column)."""
    source: str | None = _f("Source attribute.", required=True)
    transform: str = _f("identity, year, month, day, hour, bucket, truncate or void.", default="identity",
                        enum=TRANSFORMS)
    param: int | None = _f("bucket: number of buckets; truncate: width.")
    name: str | None = _f("Partition name (default <source>_<transform>).")


@dataclass
class SortField(Node):
    """One field of a sort order: a source attribute, a transform, a direction and null placement."""
    source: str | None = _f("Source attribute.", required=True)
    transform: str = _f("Transform applied before sorting.", default="identity", enum=TRANSFORMS)
    direction: str = _f("asc or desc.", default="asc", enum=("asc", "desc"))
    nulls: str = _f("first or last.", default="first", enum=("first", "last"))


@dataclass
class Discriminator(Node):
    """How the records of this entity are told apart in a container shared by several entities."""
    attribute: str | None = _f("Attribute holding the entity's type.", required=True)
    value: Any = _f("Its value for this entity.", required=True, schema={})


@dataclass
class Physical(Node):
    """Where and how an entity is stored."""
    container: str | None = _f("Name of a container in `containers` (shared tables, topics, collections).")
    name: str | None = _f("Physical name (schema.table, collection, topic).")
    format: str | None = _f("Storage format: parquet, orc, avro, csv, json, bson, delta, iceberg.")
    location: str | None = _f("Location (s3://..., abfss://..., a folder).")
    partitioning: list[PartitionField] = _f("Partition spec.", factory=list)
    sort_order: list[SortField] = _f("Sort order within files or partitions.", factory=list)
    clustering: list[str] = _f("Clustering columns (Delta liquid clustering, Snowflake clustering keys).",
                               factory=list)
    discriminator: Discriminator | None = _f("Type attribute in a shared container (single-table design).")
    key_templates: dict[str, str] = _f(
        "Container key attribute -> template built from this entity's attributes ('CUSTOMER#{customer_id}').",
        factory=dict)
    options: dict = _f("Store-specific options (table properties, retention, TTL).", factory=dict)


@dataclass
class Edge(Node):
    """A graph edge type: each record joins a source node to a target node."""
    source: str | None = _f("Relationship (from this entity) to the source node.", required=True)
    target: str | None = _f("Relationship (from this entity) to the target node.", required=True)
    directed: bool = _f("Edges have a direction.", default=True)
    multiple: bool = _f("Several edges may join the same two nodes.", default=True)
    loops: bool = _f("An edge may join a node to itself.", default=False)


@dataclass
class Stream(Node):
    """An event stream (a topic or log): records arrive over time per key."""
    key: list[str] = _f("Message key attributes (the partitioning key).", factory=list)
    order_by: str | None = _f("Attribute that orders the events of one key (event time or a sequence number).")
    partitions: int | None = _f("Number of partitions of the topic.")
    max_lateness: str | None = _f("Bound on event-time disorder (ISO 8601 duration, PT5M): the watermark lag.")
    delivery: str | None = _f("at_most_once, at_least_once or exactly_once.",
                              enum=("at_most_once", "at_least_once", "exactly_once"))


@dataclass
class Allocation(Node):
    """Weights that sum to a total over the records of a group (a bridge table)."""
    group: str | None = _f("Attribute whose records share the weights (customer).", required=True)
    weight: str | None = _f("Weight attribute.", required=True)
    total: float | None = _f("What the weights of a group sum to (1.0 default 1).")
    decimals: int | None = _f("Stored precision; the rounding remainder goes to the largest weight.")


@dataclass
class EntityPrivacy(Node):
    """Privacy annotations of an entity."""
    public: bool = _f("Holds no data about the privacy unit and may be released as it is (code lists, "
                      "calendars, a product catalogue).", default=False)
    description: str | None = _f("Free text.")


@dataclass
class Header(Node):
    """A document header in a fact: the document number, what it determines, and the line number."""
    key: str | None = _f("Document number (SalesOrderNumber).")
    columns: list[str] | None = _f("Attributes the document number determines.")
    line_number: str | None = _f("Line number attribute, generated 1..n per document.")


@dataclass
class FactModel(Node):
    """How fact records are generated (1.0 `fact`, without time_column, which is time.event_time)."""
    anchor: str | None = _f("Parent whose records generate the facts: an entity, a relationship name or a "
                            "foreign-key attribute.")
    partner: str | None = _f("Second dimension of the block model (same forms as anchor).")
    pairs: str | None = _f("auto, poisson or occupancy.", enum=("auto", "poisson", "occupancy"))
    multiplicity: str | None = _f("auto, none, partner, anchor or block.",
                                  enum=("auto", "none", "partner", "anchor", "block"))
    count_model: str | None = _f("auto, poisson or dp.", enum=("auto", "poisson", "dp"))
    group_fallback: str | None = _f("auto, anchor or partner.", enum=("auto", "anchor", "partner"))
    time_bucket: str | None = _f("Bucket of the seasonal profile of event_time.",
                                 enum=("day", "week", "month", "quarter", "year"))
    condition_on_parents: bool | list[str] | None = _f("Condition measures on parents' continuous attributes.")
    header: Header | None = _f("Order header carried on the lines.")


@dataclass
class GroupEffect(Node):
    """An attribute whose value is shared by the records of a group."""
    attribute: str | None = _f("Attribute shared by the records of a group.", required=True)
    by: list[str] = _f("Attributes or fk.col paths that define a group.", factory=list, required=True)


@dataclass
class LinkAffinity(Node):
    """Two link paths whose values match more often than chance."""
    match: list[str] = _f("Two paths whose values match more often than chance.", factory=list, required=True)
    share: float | None = _f("Matching share to reach (default: learned).")


@dataclass
class GeneratorChoice(Node):
    """The generator plug-in that models an entity's attributes (relsynth.generators)."""
    name: str | None = _f("Registered generator name (`relsynth generators list`): copula, independent, cart or "
                          "a plug-in's.", required=True)
    version: str | None = _f("Version specifier the plug-in's model version must satisfy, e.g. '>=1.0,<2'.")
    options: dict | None = _f("Options passed unchanged to the generator.")


@dataclass
class EntitySynthesis(Node):
    """relsynth engine options of an entity (1.0 table options)."""
    mode: str | None = _f("synthesize or copy.", enum=("synthesize", "copy"))
    key_order: str | None = _f("Keys numbered in the order of this attribute.")
    fact: FactModel | None = _f("How fact records are generated.")
    group_effects: list[GroupEffect] = _f("Attributes shared by the records of a group.", factory=list)
    link_affinity: list[LinkAffinity] = _f("Links that point at matching parents more often than chance.",
                                           factory=list)
    unknown_members: list | None = _f("Key values of records that stand for 'no member' (-1 'Unknown').")
    generator: GeneratorChoice | None = _f("Generator of the attributes; absent means the estate's "
                                           "synthesis.generator (copula by default).")


@dataclass
class Entity(Node):
    """A set of records of one type: a table, a document collection (or one
    document type in it), a graph node label or edge type, or an event stream."""
    shape: str = _f("record (flat row), document (nested), node, edge or event.", default="record", enum=SHAPES)
    role: str = _f("entity (a thing), fact (an event or link whose records exist because of its parents) or "
                   "view (the join of others, rebuilt).", default="entity", enum=ROLES)
    description: str | None = _f("Free text.")
    attributes: dict[str, Attribute] = _f("Attributes in order.", factory=dict)
    keys: list[Key] = _f("Primary, unique and filtered unique keys.", factory=list)
    dependencies: list[Dependency] = _f("Functional dependencies (implied levels, link determines link).",
                                        factory=list)
    variants: Variants | None = _f("Polymorphic records: attributes that depend on a discriminator.")
    view: View | None = _f("For role view: its base entity.")
    time: Time | None = _f("Event time, valid time and versioning.")
    allocation: Allocation | None = _f("Weights summing to a total per group (bridge).")
    edge: Edge | None = _f("For shape edge: its endpoints.")
    stream: Stream | None = _f("For shape event: stream keys and ordering.")
    physical: Physical | None = _f("Physical layout.")
    privacy: EntityPrivacy | None = _f("Privacy annotation.")
    synthesis: EntitySynthesis | None = _f("Engine options.")

    @property
    def primary_key(self) -> Key | None:
        return next((k for k in self.keys if k.primary), None)


# ---------------------------------------------------------------------------
# relationships, rules, containers, privacy
# ---------------------------------------------------------------------------

@dataclass
class End(Node):
    """One end of a relationship: an entity and its attributes."""
    entity: str | None = _f("Entity.", required=True)
    attributes: list[str] | None = _f("Attributes (several for a composite key; fk.col paths allowed on the "
                                      "referencing side).")


@dataclass
class Offset(Node):
    """A second reference into the same ordered parent whose gap is modelled (ship date from order date)."""
    from_: str | None = _f("The base reference attribute of the same entity.", key="from", required=True)
    order_by: str | None = _f("Parent attribute that orders it (the calendar's date).", required=True)


@dataclass
class Via(Node):
    """For many_to_many: the bridge entity and its two references."""
    entity: str | None = _f("Bridge entity.", required=True)
    from_: str | None = _f("Relationship from the bridge to the `from` entity.", key="from", required=True)
    to: str | None = _f("Relationship from the bridge to the `to` entity.", required=True)


@dataclass
class Relationship(Node):
    """A reference between entities: a foreign key (composite allowed), a
    hierarchy inside one entity, a one-to-one extension, or a many-to-many link
    through a bridge."""
    name: str | None = _f("Unique name, used by rules, fact anchors and edges.", required=True)
    from_: End | None = _f("Referencing side (the child: holds the foreign key).", key="from", required=True)
    to: End | None = _f("Referenced side (the parent); attributes default to its primary key.", required=True)
    kind: str = _f("reference (between entities), hierarchy (parent-child inside one entity) or derived "
                   "(internal: offset keys).", default="reference", enum=REL_KINDS)
    cardinality: str = _f("many_to_one, one_to_one (the referencing attributes are a key) or many_to_many "
                          "(through `via`).", default="many_to_one", enum=CARDINALITIES)
    optional: bool | None = _f("A record may have no parent (null reference); learned when absent.")
    offset: Offset | None = _f("Gap-modelled second reference into an ordered parent.")
    via: Via | None = _f("For many_to_many: the bridge.")
    description: str | None = _f("Free text.")


def ends(r: Relationship) -> tuple[End, End]:
    """The referencing and the referenced end of a relationship.  Both are
    required, so a document that passed validation always has them."""
    if r.from_ is None or r.to is None:
        raise IRError([f"relationships.{r.name}: 'from' and 'to' are required"])
    return r.from_, r.to


@dataclass
class Rule(Node):
    """A business rule.  Hard rules (holds 1) hold on every record; soft rules on a share."""
    entity: str | None = _f("Entity the rule is on.", required=True)
    type: str | None = _f("check (must hold), derive (attribute = expression), aggregate (parent attribute from "
                          "its children) or cumulative (running balance).", required=True, enum=RULE_TYPES)
    name: str | None = _f("Name.")
    expr: str | None = _f("check / derive expression.")
    holds: float | None = _f("derive: share of records on which it holds (below 1: a soft rule).")
    violation: str | None = _f("Soft rule: offset or model.", enum=("offset", "model"))
    tolerance: float | None = _f("derive: accepted absolute difference.")
    attribute: str | None = _f("aggregate / cumulative: the computed attribute.")
    agg: str | None = _f("aggregate function.",
                         enum=("count", "sum", "min", "max", "mean", "exists", "count_distinct"))
    source: str | None = _f("aggregate: the child entity.")
    of: str | None = _f("aggregate: what is aggregated.")
    via: str | None = _f("aggregate: relationship name when there are several.")
    by: str | list[str] | None = _f("cumulative: the group.")
    order: str | None = _f("cumulative: the order.")
    step: str | None = _f("cumulative: change per record.")
    opening: float | None = _f("cumulative: first value of each group.")
    floor: float | None = _f("cumulative: lowest allowed balance.")
    description: str | None = _f("Free text.")


@dataclass
class ContainerIndex(Node):
    """A secondary index of a container (a DynamoDB GSI, a MongoDB index)."""
    name: str | None = _f("Index name.", required=True)
    partition_key: str | None = _f("Partition (hash) key attribute.")
    sort_key: str | None = _f("Sort (range) key attribute.")
    projection: str | None = _f("Projected attributes: all, keys_only or include.")


@dataclass
class Container(Node):
    """A physical store shared by one or more entities (a DynamoDB table, a
    MongoDB collection with several document types, a Kafka topic)."""
    kind: str | None = _f("table, view, collection, key_value_table, topic, iceberg_table, delta_table, "
                          "file_set, graph or index.", required=True, enum=CONTAINER_KINDS)
    name: str | None = _f("Physical name.")
    system: str | None = _f("System (postgresql, mongodb, dynamodb, kafka, neo4j, iceberg).")
    partition_key: str | None = _f("Partition (hash) key attribute of the stored items.")
    sort_key: str | None = _f("Sort (range) key attribute.")
    indexes: list[ContainerIndex] = _f("Secondary indexes (global and local).", factory=list)
    options: dict = _f("Store-specific options.", factory=dict)
    description: str | None = _f("Free text.")


@dataclass
class PrivacyUnit(Node):
    """Whose privacy is protected: neighbouring datasets differ in all the data of
    one unit (Ponomareva et al. 2026)."""
    entity: str | None = _f("Entity whose records are the units (Customer).", required=True)
    description: str | None = _f("Free text (a customer and everything they bought).")


@dataclass
class EstatePrivacy(Node):
    """Privacy settings of the estate: the unit, the adjacency and the paths not followed."""
    unit: PrivacyUnit | None = _f("The privacy unit.")
    adjacency: str = _f("add_remove (a unit and all its data present or absent) or replace.",
                        default="add_remove", enum=ADJACENCY)
    exclude_paths: list[str] = _f("Relationships not followed when tracing contributions (a shared lookup).",
                                  factory=list)
    description: str | None = _f("Free text.")


@dataclass
class Estate(Node):
    """A data estate: everything relsynth reads, models and writes."""
    ir_version: str = _f("Version of the IR format (2.0).", default=IR_VERSION, required=True,
                         enum=SUPPORTED_VERSIONS)
    name: str | None = _f("Label used in reports and file names.")
    description: str | None = _f("Free text.")
    source: dict | None = _f("Where the real data is read from (the 1.0 `source` section).")
    target: dict | None = _f("Where the synthetic data is written (the 1.0 `target` section).")
    containers: dict[str, Container] = _f("Physical stores shared by several entities.", factory=dict)
    entities: dict[str, Entity] = _f("Entities in order (order is kept: it breaks ties in generation order).",
                                     factory=dict, required=True)
    relationships: list[Relationship] = _f("References between entities.", factory=list)
    rules: list[Rule] = _f("Business rules.", factory=list)
    privacy: EstatePrivacy | None = _f("Privacy unit and adjacency.")
    synthesis: dict = _f("relsynth engine settings (the 1.0 `synthesis` section).", factory=dict)
    review: list[str] = _f("What introspect inferred, and its questions.", factory=list)

    def relationship(self, name) -> Relationship | None:
        return next((r for r in self.relationships if r.name == name), None)


NODE_TYPES: list[type[Node]] = [Estate, Container, ContainerIndex, Entity, Attribute, KeyedLookup, Sequence, Variants,
              AttributePrivacy, AttributeSynthesis, Key, Dependency, View, Time, ValidTime, Versioning,
              PartitionField, SortField, Discriminator, Physical, Edge, Stream, Allocation, EntityPrivacy,
              EntitySynthesis, GeneratorChoice, FactModel, Header, GroupEffect, LinkAffinity, Relationship, End,
              Offset, Via, Rule, EstatePrivacy, PrivacyUnit]
