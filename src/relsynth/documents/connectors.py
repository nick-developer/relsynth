# SPDX-License-Identifier: Apache-2.0
"""Document stores: read collections, write synthetic ones, push statistics down.

    kind        store                                    driver (extra)
    jsonl/json  a folder of <collection>.json(l) files   - (Extended JSON v2)
    mongodb     MongoDB, Azure Cosmos DB for MongoDB,    pymongo   (relsynth[documents])
                Amazon DocumentDB (MongoDB wire protocol)
    cosmos      Azure Cosmos DB for NoSQL                azure-cosmos
    dynamodb    Amazon DynamoDB (single-table designs)   boto3
    couchbase   Couchbase Server / Capella               couchbase

Credentials come from the environment or a file, never from the spec
(`url_env`, `key_env`, AWS default chain), as for the relational connectors
(docs/connector-sdk.md).

Pushdown (MongoDB): the schema of a large collection is inferred from a
``$sample`` of documents, and the exact counts that the length laws and the
presence and type mixes need are computed in the database by one ``$group``
per collection (`MongoDB.exact_counts`): per top-level path the number of
documents in each BSON ``$type``, and per array path the histogram of
``$size``.  Only these aggregates leave the database.
"""
from __future__ import annotations

import decimal
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..errors import SourceError
from .values import from_extjson_value, plain, read_json_documents, write_json_documents

SYSTEM_FIELDS = {"cosmos": ("_rid", "_self", "_etag", "_attachments", "_ts", "_lsn"), "couchbase": ()}


def connect_documents(sec: dict | str) -> DocumentConnector:
    if isinstance(sec, str):
        sec = parse_location(sec)
    kind = str(sec.get("kind"))
    cls = {"jsonl": JsonFiles, "json": JsonFiles, "extjson": JsonFiles, "mongodb": MongoDB,
           "cosmos_mongodb": MongoDB, "documentdb": MongoDB, "cosmos": CosmosNoSQL, "dynamodb": DynamoDB,
           "couchbase": Couchbase}.get(kind)
    if cls is None:
        raise SourceError(f"unknown document store kind {kind!r}")
    return cls(sec)


def parse_location(s: str) -> dict:
    """'jsonl:folder', 'mongodb://host/db', 'mongodb+srv://...', 'dynamodb:table'."""
    if s.startswith(("mongodb://", "mongodb+srv://")):
        if re.match(r"mongodb(\+srv)?://[^/@]*:[^/@]*@", s):
            # like every relsynth command: no passwords on the command line
            raise SourceError("a MongoDB URL with a password is refused on the command line; "
                              "put it in an environment variable and use a YAML section with url_env")
        return {"kind": "mongodb", "url": s}
    k, _, rest = s.partition(":")
    if k in ("jsonl", "json", "extjson"):
        return {"kind": "jsonl", "path": rest}
    if k == "dynamodb":
        return {"kind": "dynamodb", "table": rest}
    if k == "cosmos":
        return {"kind": "cosmos", "database": rest}
    raise SourceError(f"cannot read a document store location {s!r}")


def _secret(sec: dict, key: str, env: str) -> str | None:
    if sec.get(key):
        return sec[key]
    name = sec.get(env)
    if name:
        v = os.environ.get(name)
        if v is None:
            raise SourceError(f"environment variable {name} ({env}) is not set")
        return v
    f = sec.get(env.replace("_env", "_file"))
    if f:
        return Path(f).read_text().strip()
    return None


class DocumentConnector:
    """Base class: list, count, read, sample and write collections."""

    def __init__(self, sec: dict):
        self.sec = dict(sec)
        self.include = sec.get("collections")

    def collections(self) -> list[str]:
        raise NotImplementedError

    def read(self, collection: str, limit: int | None = None) -> Iterator[dict]:
        raise NotImplementedError

    def count(self, collection: str) -> int:
        return sum(1 for _ in self.read(collection))

    def sample(self, collection: str, n: int, seed: int = 0) -> list[dict]:
        docs = list(self.read(collection))
        if len(docs) <= n:
            return docs
        import numpy as np
        idx = np.sort(np.random.default_rng(seed).choice(len(docs), n, replace=False))
        return [docs[i] for i in idx]

    def write(self, collection: str, docs: Iterable[dict], if_exists: str = "replace") -> int:
        raise NotImplementedError

    def read_all(self, limit: int | None = None) -> dict[str, list[dict]]:
        names = self.include or self.collections()
        return {c: list(self.read(c, limit)) for c in names}

    def write_all(self, collections: dict[str, list[dict]], if_exists: str = "replace") -> dict:
        return {c: self.write(c, docs, if_exists) for c, docs in collections.items()}


# ---------------------------------------------------------------- files
class JsonFiles(DocumentConnector):
    """A folder with one file per collection: <name>.json or <name>.jsonl, each a
    JSON array or one (Extended) JSON document per line."""

    def __init__(self, sec):
        super().__init__(sec)
        self.path = Path(sec.get("path", "."))
        self.relaxed = bool(sec.get("relaxed", False))

    def _file(self, c):
        for ext in (".jsonl", ".json", ".ndjson"):
            p = self.path / (c + ext)
            if p.exists():
                return p
        raise SourceError(f"no file for collection {c} in {self.path}")

    def collections(self):
        names = []
        for p in sorted(self.path.iterdir()):
            if p.suffix in (".json", ".jsonl", ".ndjson") and p.stem not in names:
                names.append(p.stem)
        return names

    def read(self, collection, limit=None):
        for i, d in enumerate(read_json_documents(self._file(collection))):
            if limit is not None and i >= limit:
                return
            yield d

    def write(self, collection, docs, if_exists="replace"):
        self.path.mkdir(parents=True, exist_ok=True)
        p = self.path / f"{collection}.jsonl"
        if p.exists() and if_exists == "fail":
            raise SourceError(f"{p} exists")
        if if_exists == "append" and p.exists():
            old = list(read_json_documents(p))
            docs = old + list(docs)
        return write_json_documents(docs, p, relaxed=self.relaxed)


# ---------------------------------------------------------------- MongoDB
_BSON_TYPES = ("double", "string", "object", "array", "binData", "objectId", "bool", "date", "null", "int",
               "long", "decimal", "timestamp", "missing")
_TYPE_NAME = {"double": "double", "string": "string", "object": "object", "array": "array", "binData": "binary",
              "objectId": "objectId", "bool": "bool", "date": "date", "null": "null", "int": "int", "long": "long",
              "decimal": "decimal", "timestamp": "date"}


class MongoDB(DocumentConnector):
    """MongoDB through pymongo; also Azure Cosmos DB for MongoDB (vCore and RU)
    and Amazon DocumentDB, which speak the same protocol (their aggregation
    support is a subset: `$sample` and `$group` with `$type` / `$size` are in
    every one of them)."""

    def __init__(self, sec, client=None):
        super().__init__(sec)
        if client is None:
            try:
                import pymongo
            except ImportError as e:  # pragma: no cover
                raise SourceError("the mongodb kind needs pymongo: pip install 'relsynth[documents]'") from e
            url = _secret(sec, "url", "url_env") or "mongodb://localhost:27017"
            client = pymongo.MongoClient(url, tz_aware=True, **sec.get("options", {}))
        self.client = client
        db = sec.get("database")
        if db is None:
            try:
                db = client.get_default_database().name
            except Exception as e:  # noqa: BLE001
                raise SourceError("mongodb: name the database (database: ...)") from e
        self.db = client[db]

    def collections(self):
        return sorted(n for n in self.db.list_collection_names() if not n.startswith("system."))

    def count(self, collection):
        return int(self.db[collection].estimated_document_count())

    def read(self, collection, limit=None):
        cur = self.db[collection].find({}, batch_size=self.sec.get("batch_size", 1000))
        if limit is not None:
            cur = cur.limit(int(limit))
        for d in cur:
            yield plain(d)

    def sample(self, collection, n, seed=0):
        """`$sample` (random documents chosen by the server; not seeded)."""
        if self.count(collection) <= n:
            return list(self.read(collection))
        return [plain(d) for d in self.db[collection].aggregate([{"$sample": {"size": int(n)}}],
                                                                  allowDiskUse=True)]

    def exact_counts(self, collection, paths: list[str], arrays: list[str]) -> dict:
        """{documents, types: {path: Counter(type)}, lengths: {path: Counter(len)}}
        computed in the database."""
        grp: dict[str, Any] = {"_id": None, "n": {"$sum": 1}}
        keys = {}
        for i, p in enumerate(paths):
            for t in _BSON_TYPES:
                k = f"p{i}_{t}"
                keys[k] = (p, t)
                grp[k] = {"$sum": {"$cond": [{"$eq": [{"$type": f"${p}"}, t]}, 1, 0]}}
        res = list(self.db[collection].aggregate([{"$group": grp}], allowDiskUse=True))
        types: dict[str, Counter] = {p: Counter() for p in paths}
        n = 0
        if res:
            n = int(res[0]["n"])
            for k, (p, t) in keys.items():
                c = int(res[0].get(k, 0))
                if c and t != "missing":
                    types[p][_TYPE_NAME[t]] += c
        lengths = {}
        for p in arrays:
            pipe = [{"$match": {p: {"$type": "array"}}}, {"$group": {"_id": {"$size": f"${p}"}, "c": {"$sum": 1}}}]
            lengths[p] = Counter({int(r["_id"]): int(r["c"]) for r in self.db[collection].aggregate(pipe)})
        return {"documents": n, "types": types, "lengths": lengths}

    def write(self, collection, docs, if_exists="replace"):
        from .values import to_bson
        coll = self.db[collection]
        if if_exists == "fail" and coll.estimated_document_count():
            raise SourceError(f"collection {collection} is not empty")
        if if_exists in ("replace", "truncate"):
            coll.delete_many({})
        batch, n = [], 0
        for d in docs:
            batch.append(to_bson(d))
            if len(batch) >= 1000:
                coll.insert_many(batch, ordered=False)
                n += len(batch)
                batch = []
        if batch:
            coll.insert_many(batch, ordered=False)
            n += len(batch)
        return n


def profile_with_pushdown(con: MongoDB, collection: str, sample_size: int = 20000, seed: int = 0):
    """The profile of a collection from a `$sample`, with the top-level presence
    and type counts and the top-level array length laws replaced by the exact
    counts of the whole collection (one `$group` pass in the database)."""
    from .infer import profile_collection, _find_maps, _find_length_fields, _find_variants
    docs = con.sample(collection, sample_size, seed)
    prof = profile_collection(collection, docs)
    top = [p for p in prof.root.fields if "." not in p and not p.startswith("$")]
    arrays = [p for p in top if prof.root.fields[p].is_array]
    exact = con.exact_counts(collection, top, arrays)
    N = exact["documents"]
    if N:
        prof.documents = N
        prof.root.n = prof.root.present = N
        prof.root.types = Counter({"object": N})
        for p in top:
            nd = prof.root.fields[p]
            nd.types = exact["types"][p]
            nd.n = N
            nd.present = sum(exact["types"][p].values())
            if p in exact["lengths"] and exact["lengths"][p]:
                nd.lengths = exact["lengths"][p]
    del _find_maps, _find_length_fields, _find_variants
    return prof, docs


# ---------------------------------------------------------------- Cosmos DB for NoSQL
class CosmosNoSQL(DocumentConnector):
    """Azure Cosmos DB for NoSQL (the SQL API): containers are collections.
    Writes upsert items; the container's partition key path must be present in
    every synthetic document (it is a field of the source documents)."""

    def __init__(self, sec, client=None):
        super().__init__(sec)
        if client is None:
            try:
                from azure.cosmos import CosmosClient
            except ImportError as e:  # pragma: no cover
                raise SourceError("the cosmos kind needs azure-cosmos: pip install 'relsynth[documents]'") from e
            url = _secret(sec, "url", "url_env")
            key = _secret(sec, "key", "key_env")
            if key is None:
                from azure.identity import DefaultAzureCredential  # pragma: no cover
                key = DefaultAzureCredential()
            client = CosmosClient(url, credential=key)
        self.client = client
        self.db = client.get_database_client(sec["database"])

    def collections(self):
        return sorted(c["id"] for c in self.db.list_containers())

    def read(self, collection, limit=None):
        q = "SELECT * FROM c" if limit is None else f"SELECT TOP {int(limit)} * FROM c"  # noqa: S608  # nosec B608 - (an int)
        for d in self.db.get_container_client(collection).query_items(q, enable_cross_partition_query=True):
            yield {k: plain(v) for k, v in d.items() if k not in SYSTEM_FIELDS["cosmos"]}

    def count(self, collection):
        r = list(self.db.get_container_client(collection).query_items(
            "SELECT VALUE COUNT(1) FROM c", enable_cross_partition_query=True))
        return int(r[0]) if r else 0

    def write(self, collection, docs, if_exists="replace"):
        from .values import to_plain_json
        c = self.db.get_container_client(collection)
        if if_exists in ("replace", "truncate"):
            pk = c.read()["partitionKey"]["paths"][0].lstrip("/").split("/")
            for d in list(c.query_items("SELECT * FROM c", enable_cross_partition_query=True)):
                v = d
                for p in pk:
                    v = v.get(p) if isinstance(v, dict) else None
                c.delete_item(d["id"], partition_key=v)
        n = 0
        for d in docs:
            d = to_plain_json(d)
            d.setdefault("id", str(d.get("_id", n)))
            c.upsert_item(d)
            n += 1
        return n


# ---------------------------------------------------------------- DynamoDB
def _to_dynamo(v):
    if isinstance(v, float):
        return decimal.Decimal(repr(v))
    if isinstance(v, dict):
        return {k: _to_dynamo(x) for k, x in v.items() if x is not None or True}
    if isinstance(v, list):
        return [_to_dynamo(x) for x in v]
    return v


def _from_dynamo(v):
    if isinstance(v, decimal.Decimal):
        return int(v) if v == v.to_integral_value() and "E" not in str(v) and "." not in str(v) else float(v)
    if isinstance(v, dict):
        return {k: _from_dynamo(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_from_dynamo(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return [_from_dynamo(x) for x in sorted(v, key=str)]
    if isinstance(v, (bytes, bytearray)) or type(v).__name__ == "Binary":
        return plain(bytes(v))
    return v


class DynamoDB(DocumentConnector):
    """Amazon DynamoDB.  `table` names one table; with a single-table design the
    item types become logical collections (`relsynth.documents.dynamo`)."""

    def __init__(self, sec, resource=None):
        super().__init__(sec)
        if resource is None:
            try:
                import boto3
            except ImportError as e:  # pragma: no cover
                raise SourceError("the dynamodb kind needs boto3: pip install 'relsynth[documents]'") from e
            kw = {k: sec[k] for k in ("region_name", "endpoint_url") if sec.get(k)}
            resource = boto3.resource("dynamodb", **kw)
        self.resource = resource

    def collections(self):
        if self.sec.get("table"):
            return [self.sec["table"]]
        return sorted(t.name for t in self.resource.tables.all())

    def read(self, collection, limit=None):
        t = self.resource.Table(collection)
        kw, n = {}, 0
        while True:
            r = t.scan(**kw)
            for it in r.get("Items", []):
                if limit is not None and n >= limit:
                    return
                n += 1
                yield _from_dynamo(it)
            if "LastEvaluatedKey" not in r:
                return
            kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]

    def key_schema(self, collection) -> dict:
        t = self.resource.Table(collection)
        ks = {k["KeyType"]: k["AttributeName"] for k in t.key_schema}
        gsis = [{"name": g["IndexName"], **{k["KeyType"]: k["AttributeName"] for k in g["KeySchema"]}}
                for g in (t.global_secondary_indexes or [])]
        return {"partition_key": ks.get("HASH"), "sort_key": ks.get("RANGE"), "indexes": gsis}

    def write(self, collection, docs, if_exists="replace"):
        from .values import to_plain_json
        t = self.resource.Table(collection)
        if if_exists in ("replace", "truncate"):
            ks = [k["AttributeName"] for k in t.key_schema]
            with t.batch_writer() as b:
                for it in self.read(collection):
                    b.delete_item(Key={k: _to_dynamo(it[k]) for k in ks})
        n = 0
        with t.batch_writer() as b:
            for d in docs:
                d = {k: v for k, v in to_plain_json(d).items() if v is not None}
                b.put_item(Item=_to_dynamo(d))
                n += 1
        return n


# ---------------------------------------------------------------- Couchbase
def _cb_ident(name: str) -> str:
    """A SQL++ identifier in backticks; a name with a backtick or a control character is refused."""
    if not name or "`" in name or any(ord(ch) < 32 for ch in name):
        raise SourceError(f"refused Couchbase keyspace name {name!r}")
    return f"`{name}`"


class Couchbase(DocumentConnector):
    """Couchbase Server or Capella: collections of a bucket's scope, read with
    SQL++ (``SELECT META().id AS _id, c.* FROM bucket.scope.collection c``)."""

    def __init__(self, sec, cluster=None):
        super().__init__(sec)
        if cluster is None:
            try:
                from couchbase.auth import PasswordAuthenticator
                from couchbase.cluster import Cluster
                from couchbase.options import ClusterOptions
            except ImportError as e:  # pragma: no cover
                raise SourceError("the couchbase kind needs the couchbase SDK: pip install couchbase") from e
            url = _secret(sec, "url", "url_env") or "couchbase://localhost"
            auth = PasswordAuthenticator(sec.get("username", "Administrator"), _secret(sec, "password", "password_env"))
            cluster = Cluster(url, ClusterOptions(auth))
        self.cluster = cluster
        self.bucket = sec["bucket"]
        self.scope = sec.get("scope", "_default")

    def _ks(self, c):
        return ".".join(_cb_ident(x) for x in (self.bucket, self.scope, c))

    def collections(self):
        from couchbase.options import QueryOptions
        q = "SELECT RAW name FROM system:keyspaces WHERE `bucket` = $1 AND `scope` = $2"
        return sorted(self.cluster.query(q, QueryOptions(positional_parameters=[self.bucket, self.scope])))

    def read(self, collection, limit=None):
        q = f"SELECT META(c).id AS _id, c.* FROM {self._ks(collection)} c"  # noqa: S608  # nosec B608 - (quoted identifiers)
        if limit is not None:
            q += f" LIMIT {int(limit)}"
        for r in self.cluster.query(q):
            yield plain(r)

    def write(self, collection, docs, if_exists="replace"):
        from .values import to_plain_json
        coll = self.cluster.bucket(self.bucket).scope(self.scope).collection(collection)
        if if_exists in ("replace", "truncate"):
            self.cluster.query(f"DELETE FROM {self._ks(collection)}").execute()  # noqa: S608  # nosec B608 - (quoted identifiers)
        n = 0
        for d in docs:
            d = to_plain_json(d)
            key = str(d.pop("_id", n))
            coll.upsert(key, d)
            n += 1
        return n


__all__ = ["connect_documents", "parse_location", "DocumentConnector", "JsonFiles", "MongoDB", "CosmosNoSQL",
           "DynamoDB", "Couchbase", "profile_with_pushdown", "from_extjson_value"]
