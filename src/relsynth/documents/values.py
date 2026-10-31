# SPDX-License-Identifier: Apache-2.0
"""Document values: the BSON / JSON type of a value, MongoDB Extended JSON v2
read and write, and the conversion between document values and table cells.

Documents reach relsynth from three places: JSON or Extended JSON files
(``mongoexport``, the MongoDB sample datasets), a driver (``pymongo`` returns
``bson`` types, ``azure-cosmos`` and ``boto3`` return plain JSON / DynamoDB
types) and relsynth's own synthetic output.  Every value is first brought to
one plain Python form, and its *type name* is one of `TYPES`:

    null bool int long double decimal string objectId date binary object array

so that schema inference counts the same thing whatever the driver.  `int` and
`long` (BSON int32 / int64) and `double` and `decimal` are kept apart because
a store distinguishes them and a synthetic document must carry the same type
as the field it replaces.

Extended JSON follows the MongoDB specification (canonical and relaxed
modes, https://www.mongodb.com/docs/manual/reference/mongodb-extended-json/).
"""
from __future__ import annotations

import base64
import datetime as _dt
import decimal
import hashlib
import json
import math
from typing import Any, Iterable, Iterator

TYPES = ("null", "bool", "int", "long", "double", "decimal", "string", "objectId", "date", "binary", "object",
         "array")
SCALARS = tuple(t for t in TYPES if t not in ("object", "array", "null"))
NUMERIC = ("int", "long", "double", "decimal")
UTC = _dt.timezone.utc
_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=UTC)
_INT32 = 2 ** 31


class ObjectId(str):
    """A 24-hex-digit MongoDB ObjectId (kept as its hex string)."""
    __slots__ = ()


class Long(int):
    """An int64 that must stay int64 when written (``$numberLong``)."""
    __slots__ = ()


class Binary(bytes):
    """Binary data with its BSON subtype."""
    subtype = 0


def type_name(v: Any) -> str:
    """The type name (one of `TYPES`) of a plain or driver value."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, ObjectId):
        return "objectId"
    if isinstance(v, Long):
        return "long"
    if isinstance(v, int):
        return "int" if -_INT32 <= v < _INT32 else "long"
    if isinstance(v, float):
        return "double"
    if isinstance(v, decimal.Decimal):
        return "decimal"
    if isinstance(v, str):
        return "string"
    if isinstance(v, (_dt.datetime, _dt.date)):
        return "date"
    if isinstance(v, (bytes, bytearray, memoryview)):
        return "binary"
    if isinstance(v, dict):
        return "object"
    if isinstance(v, (list, tuple)):
        return "array"
    n = type(v).__name__
    return {"ObjectId": "objectId", "Int64": "long", "Decimal128": "decimal", "Binary": "binary", "UUID": "binary",
            "Timestamp": "date", "datetime": "date"}.get(n, "string")


def plain(v: Any) -> Any:
    """A driver value (bson, DynamoDB) as the plain form used everywhere here."""
    if v is None or isinstance(v, (bool, str, float, ObjectId, Long, Binary)):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, dict):
        if len(v) == 1:
            k = next(iter(v))
            if k.startswith("$"):
                return from_extjson_value(v)
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, decimal.Decimal):
        return v
    if isinstance(v, _dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=UTC)
    if isinstance(v, _dt.date):
        return _dt.datetime(v.year, v.month, v.day, tzinfo=UTC)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return Binary(bytes(v))
    if isinstance(v, (set, frozenset)):          # DynamoDB string / number sets
        return [plain(x) for x in sorted(v, key=str)]
    n = type(v).__name__
    if n == "ObjectId":
        return ObjectId(str(v))
    if n == "Int64":
        return Long(int(v))
    if n == "Decimal128":
        return v.to_decimal()
    if n == "Binary":
        return Binary(bytes(v))
    if n == "UUID":
        return Binary(v.bytes)
    if n == "Timestamp":
        return v.as_datetime()
    return str(v)


# ---------------------------------------------------------------- Extended JSON
def from_extjson_value(v: Any) -> Any:
    """One parsed JSON value (Extended JSON v2, canonical or relaxed) as a plain value."""
    if isinstance(v, list):
        return [from_extjson_value(x) for x in v]
    if not isinstance(v, dict):
        return v
    if len(v) == 1:
        k, x = next(iter(v.items()))
        if k == "$oid":
            return ObjectId(x)
        if k == "$numberInt":
            return int(x)
        if k == "$numberLong":
            return Long(int(x))
        if k == "$numberDouble":
            return float(x)
        if k == "$numberDecimal":
            return decimal.Decimal(x)
        if k == "$date":
            if isinstance(x, dict):
                x = int(x.get("$numberLong"))
            if isinstance(x, (int, float)):
                return _EPOCH + _dt.timedelta(milliseconds=x)
            return _parse_iso(x)
        if k == "$binary":
            if isinstance(x, dict):
                b = Binary(base64.b64decode(x.get("base64", "")))
                b.subtype = int(str(x.get("subType", "0")), 16)
                return b
            return Binary(base64.b64decode(x))
        if k == "$uuid":
            b = Binary(bytes.fromhex(str(x).replace("-", "")))
            b.subtype = 4
            return b
        if k == "$timestamp":
            return _EPOCH + _dt.timedelta(seconds=int(x.get("t", 0)))
        if k in ("$regularExpression", "$symbol", "$code", "$minKey", "$maxKey", "$undefined"):
            return str(x)
    if set(v) == {"$binary", "$type"}:                      # legacy binary
        b = Binary(base64.b64decode(v["$binary"]))
        b.subtype = int(str(v["$type"]), 16)
        return b
    return {k: from_extjson_value(x) for k, x in v.items()}


def _parse_iso(s: str) -> _dt.datetime:
    s = s.strip().replace("Z", "+00:00")
    d = _dt.datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def to_extjson_value(v: Any, *, relaxed: bool = False) -> Any:
    """A plain value as Extended JSON v2 (canonical unless `relaxed`)."""
    t = type_name(v)
    if t == "object":
        return {k: to_extjson_value(x, relaxed=relaxed) for k, x in v.items()}
    if t == "array":
        return [to_extjson_value(x, relaxed=relaxed) for x in v]
    if t in ("null", "bool", "string"):
        return v
    if t == "objectId":
        return {"$oid": str(v)}
    if t == "int":
        return int(v) if relaxed else {"$numberInt": str(int(v))}
    if t == "long":
        return int(v) if relaxed else {"$numberLong": str(int(v))}
    if t == "double":
        f = float(v)
        if not math.isfinite(f):
            return {"$numberDouble": "NaN" if f != f else ("Infinity" if f > 0 else "-Infinity")}
        return f if relaxed else {"$numberDouble": repr(f)}
    if t == "decimal":
        return {"$numberDecimal": str(v)}
    if t == "date":
        ms = to_millis(v)
        if relaxed and 0 <= ms < 253402300800000:
            return {"$date": from_millis(ms).strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms % 1000:03d}Z"}
        return {"$date": {"$numberLong": str(ms)}}
    if t == "binary":
        sub = getattr(v, "subtype", 0)
        return {"$binary": {"base64": base64.b64encode(bytes(v)).decode(), "subType": f"{sub:02x}"}}
    return str(v)


def to_millis(v) -> int:
    if not isinstance(v, _dt.datetime):
        v = _dt.datetime(v.year, v.month, v.day, tzinfo=UTC)
    if v.tzinfo is None:
        v = v.replace(tzinfo=UTC)
    d = v - _EPOCH
    return (d.days * 86400 + d.seconds) * 1000 + d.microseconds // 1000


def from_millis(ms) -> _dt.datetime:
    return _EPOCH + _dt.timedelta(milliseconds=int(ms))


def read_json_documents(path) -> Iterator[dict]:
    """Documents of a file: JSON Lines (one document per line, as ``mongoexport``
    writes) or one JSON array; Extended JSON values are decoded."""
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        while head and head.isspace():
            head = f.read(1)
        f.seek(0)
        if head == "[":
            for d in json.load(f):
                yield from_extjson_value(d)
            return
        for line in f:
            line = line.strip()
            if line:
                yield from_extjson_value(json.loads(line))


def write_json_documents(docs: Iterable[dict], path, *, relaxed: bool = False) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(to_extjson_value(d, relaxed=relaxed), ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
            n += 1
    return n


# ---------------------------------------------------------------- cells
def to_cell(v: Any, t: str) -> Any:
    """A scalar of type `t` as a table cell (what the engine models)."""
    if v is None:
        return None
    if t == "date":
        return from_millis(to_millis(v)).replace(tzinfo=None)
    if t == "decimal":
        return float(v)
    if t in ("int", "long"):
        return int(v)
    if t == "double":
        return float(v)
    if t == "bool":
        return bool(v)
    if t == "binary":
        return base64.b64encode(bytes(v)).decode()
    return str(v)


def from_cell(v: Any, t: str, decimals: int | None = None) -> Any:
    """A synthetic cell back as a document value of type `t`."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    try:
        import pandas as pd
        if v is pd.NaT or (not isinstance(v, (str, bytes, list, dict)) and pd.isna(v)):
            return None
    except (TypeError, ValueError, ImportError):
        pass
    if t == "date":
        import pandas as pd
        ts = pd.Timestamp(v)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return from_millis(int(ts.value // 1_000_000))
    if t == "int":
        return int(round(float(v)))
    if t == "long":
        return Long(int(round(float(v))))
    if t == "double":
        f = float(v)
        return round(f, decimals) if decimals is not None else f
    if t == "decimal":
        if isinstance(v, str):
            return decimal.Decimal(v)
        f = float(v)
        if decimals is None:
            return decimal.Decimal(repr(f))
        txt = f"{round(f, decimals):.{decimals}f}"
        return decimal.Decimal(txt.rstrip("0").rstrip(".") if "." in txt else txt)
    if t == "bool":
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "t", "yes")
        return bool(v)
    if t == "binary":
        return Binary(base64.b64decode(str(v)))
    if t == "objectId":
        return ObjectId(str(v))
    return str(v)


def object_id(*parts) -> ObjectId:
    """A deterministic ObjectId from labels (synthetic documents get fresh ids that
    are a function of the seed and the row, so a rerun writes the same ids)."""
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()
    return ObjectId(h[:24])


def to_bson(v: Any) -> Any:
    """A plain value as the types pymongo writes (ObjectId, Int64, Decimal128)."""
    t = type_name(v)
    if t == "object":
        return {k: to_bson(x) for k, x in v.items()}
    if t == "array":
        return [to_bson(x) for x in v]
    try:
        import bson
    except ImportError:  # pragma: no cover
        return v
    if t == "objectId":
        return bson.ObjectId(str(v))
    if t == "long":
        return bson.Int64(int(v))
    if t == "decimal":
        return bson.Decimal128(str(v))
    if t == "binary":
        return bson.Binary(bytes(v), getattr(v, "subtype", 0))
    return v


def to_plain_json(v: Any) -> Any:
    """A plain value as JSON-native types (for Cosmos DB, DynamoDB, Couchbase):
    ids and decimals as strings or numbers, dates as ISO 8601 strings."""
    t = type_name(v)
    if t == "object":
        return {k: to_plain_json(x) for k, x in v.items()}
    if t == "array":
        return [to_plain_json(x) for x in v]
    if t == "objectId":
        return str(v)
    if t == "decimal":
        return float(v)
    if t == "date":
        return from_millis(to_millis(v)).strftime("%Y-%m-%dT%H:%M:%S.") + f"{to_millis(v) % 1000:03d}Z"
    if t == "binary":
        return base64.b64encode(bytes(v)).decode()
    if t in ("int", "long"):
        return int(v)
    return v
