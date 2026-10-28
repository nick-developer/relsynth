"""A local stand-in for a Databricks SQL warehouse: forwards the Databricks SQL
connector's Thrift-over-HTTP calls to an Apache Spark Thrift server (Delta Lake).

* auth: the connector sends a bearer token; HiveServer2 in HTTP mode wants Basic;
* protocol: Spark answers OpenSession with HIVE_CLI_SERVICE_PROTOCOL_V10; the
  connector needs a SPARK_CLI_SERVICE protocol, so the answer says V8;
* result format: Spark does not set the Databricks result-format field; its
  row sets are column based, which the answer now says;
* GetColumns: Spark lists no columns for Delta tables, whose schema lives in
  the Delta log; the proxy answers with a query built from DESCRIBE TABLE and
  the Delta log (nullability), in the JDBC getColumns shape;
* COPY INTO exists only on Databricks: the connector's COPY INTO of a staged
  Parquet file becomes INSERT INTO ... SELECT ... FROM parquet.`file`;
* parameters: Databricks binds named parameters on the server (TSparkParameter);
  Spark's Thrift server has no parameters, so they are inlined here as typed
  literals, CAST('text' AS TYPE), the way the server would bind them.
"""
import base64
import http.client
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from databricks.sql.thrift_api.TCLIService import TCLIService, ttypes

LISTEN, TARGET = int(sys.argv[1]), int(sys.argv[2])
BASIC = "Basic " + base64.b64encode(b"relsynth:x").decode()
DEBUG = "--debug" in sys.argv
META: dict = {}            # result-set metadata by operation id
# the connector's COPY INTO of one staged Parquet file (bulk method path)
COPY = re.compile(r"COPY INTO (\S+) FROM \(SELECT (.*) FROM '([^']+)'\) FILEFORMAT = PARQUET", re.S)
TOKEN = re.compile(r"'(?:[^'\\]|\\.)*'|:`([^`]*)`|`[^`]*`|\"(?:[^\"\\]|\\.)*\""
                   r"|:([A-Za-z_][A-Za-z0-9_]*)")


def literal(p):
    t = (p.type or "STRING").upper()
    v = p.value
    if v is None or (v.stringValue is None and v.doubleValue is None and v.booleanValue is None) or t == "VOID":
        return "NULL" if t in ("VOID", "") else f"CAST(NULL AS {t})"
    s = v.stringValue if v.stringValue is not None else repr(v.doubleValue) if v.doubleValue is not None \
        else str(v.booleanValue).lower()
    s = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"CAST('{s}' AS {t})"


def inline(stmt, params):
    by_name = {p.name: literal(p) for p in params if p.name}
    ordered = [literal(p) for p in sorted((p for p in params if not p.name), key=lambda p: p.ordinal or 0)]
    pos = iter(ordered)

    def rep(m):
        name = m.group(1) if m.group(1) is not None else m.group(2)
        if name is None:
            return m.group(0)
        return by_name.get(name, m.group(0))
    out = TOKEN.sub(rep, stmt)
    if ordered:
        out = re.sub(r"\?", lambda m: next(pos), out)
    return out


def _q(x):
    return "'" + str(x).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _ident(x):
    return "`" + str(x).replace("`", "``") + "`"


def run_sql(stmt):
    """Rows of a statement run in a session of the proxy's own."""
    def call(mname, req, res_cls):
        b = encode(mname, 1, 0, getattr(TCLIService, f"{mname}_args")(req=req))
        c = http.client.HTTPConnection("127.0.0.1", TARGET, timeout=600)
        c.request("POST", "/cliservice", body=b, headers={"Authorization": BASIC,
                                                         "Content-Type": "application/x-thrift"})
        return decode(c.getresponse().read(), res_cls)[3].success
    s = call("OpenSession", ttypes.TOpenSessionReq(client_protocol=9), TCLIService.OpenSession_result)
    try:
        e = call("ExecuteStatement", ttypes.TExecuteStatementReq(sessionHandle=s.sessionHandle, statement=stmt),
                 TCLIService.ExecuteStatement_result)
        if e.status.statusCode not in (0, 1):
            return None
        rows = []
        while True:
            f = call("FetchResults", ttypes.TFetchResultsReq(operationHandle=e.operationHandle, orientation=0,
                                                              maxRows=10000), TCLIService.FetchResults_result)
            cols = [next(v for v in (c.stringVal, c.i32Val, c.i64Val, c.boolVal, c.doubleVal) if v is not None).values
                    for c in (f.results.columns or [])]
            got = list(zip(*cols))
            rows += got
            if not got or not f.hasMoreRows:
                break
        return rows
    finally:
        call("CloseSession", ttypes.TCloseSessionReq(sessionHandle=s.sessionHandle), TCLIService.CloseSession_result)


JDBC = {"tinyint": -6, "smallint": 5, "int": 4, "bigint": -5, "float": 6, "double": 8, "decimal": 3,
        "string": 12, "varchar": 12, "char": 1, "boolean": 16, "date": 91, "timestamp": 93,
        "timestamp_ntz": 93, "binary": -2}


def delta_not_null(extended):
    """Columns declared NOT NULL, from the table's Delta log (DESCRIBE does not show them;
    SHOW CREATE TABLE is not supported on Delta tables in open-source Spark)."""
    import glob
    import json
    import os
    loc = next((r[1] for r in extended if r[0] == "Location"), "")
    if not loc.startswith("file:"):
        return set()
    schema = None
    for f in sorted(glob.glob(os.path.join(loc[len("file:"):], "_delta_log", "*.json"))):
        for line in open(f):
            meta = json.loads(line).get("metaData")
            if meta:
                schema = json.loads(meta["schemaString"])
    return {f["name"] for f in (schema or {}).get("fields", []) if not f.get("nullable", True)}


def columns_query(req):
    """A SELECT whose result has the shape of a GetColumns answer (JDBC getColumns)."""
    schema, table = req.schemaName, req.tableName
    if not schema or not table or any(ch in (schema + table) for ch in "%*"):
        return None
    full = f"{_ident(schema)}.{_ident(table)}"
    desc = run_sql(f"DESCRIBE TABLE {full}")
    if desc is None:
        return None
    notnull = delta_not_null(run_sql(f"DESCRIBE TABLE EXTENDED {full}") or [])
    rows = []
    for i, (col, typ, comment) in enumerate(desc):
        if not col or col.startswith("#"):
            break
        base = re.match(r"^\w+", typ).group(0).lower()
        rows.append(f"({_q('spark_catalog')}, {_q(schema)}, {_q(table)}, {_q(col)}, {JDBC.get(base, 1111)}, "
                    f"{_q(typ.upper())}, {0 if col in notnull else 1}, {_q(comment or '')}, {i + 1})")
    if not rows:
        return None
    return ("SELECT TABLE_CAT, TABLE_SCHEM, TABLE_NAME, COLUMN_NAME, DATA_TYPE, TYPE_NAME, "
            "CAST(NULL AS INT) AS COLUMN_SIZE, CAST(NULL AS TINYINT) AS BUFFER_LENGTH, "
            "CAST(NULL AS INT) AS DECIMAL_DIGITS, CAST(NULL AS INT) AS NUM_PREC_RADIX, NULLABLE, REMARKS, "
            "CAST(NULL AS STRING) AS COLUMN_DEF, CAST(NULL AS INT) AS SQL_DATA_TYPE, "
            "CAST(NULL AS INT) AS SQL_DATETIME_SUB, CAST(NULL AS INT) AS CHAR_OCTET_LENGTH, ORDINAL_POSITION, "
            "CASE WHEN NULLABLE = 1 THEN 'YES' ELSE 'NO' END AS IS_NULLABLE, "
            "CAST(NULL AS STRING) AS SCOPE_CATALOG, CAST(NULL AS STRING) AS SCOPE_SCHEMA, "
            "CAST(NULL AS STRING) AS SCOPE_TABLE, CAST(NULL AS SMALLINT) AS SOURCE_DATA_TYPE, "
            "'NO' AS IS_AUTO_INCREMENT FROM VALUES " + ", ".join(rows) +
            " AS t(TABLE_CAT, TABLE_SCHEM, TABLE_NAME, COLUMN_NAME, DATA_TYPE, TYPE_NAME, NULLABLE, REMARKS, "
            "ORDINAL_POSITION) ORDER BY ORDINAL_POSITION")


def decode(data, cls):
    tr = TTransport.TMemoryBuffer(data)
    pr = TBinaryProtocol.TBinaryProtocol(tr)
    name, mtype, seq = pr.readMessageBegin()
    obj = cls()
    obj.read(pr)
    return name, mtype, seq, obj


def encode(name, mtype, seq, obj):
    tr = TTransport.TMemoryBuffer()
    pr = TBinaryProtocol.TBinaryProtocol(tr)
    pr.writeMessageBegin(name, mtype, seq)
    obj.write(pr)
    pr.writeMessageEnd()
    return tr.getvalue()


def method(data):
    try:
        tr = TTransport.TMemoryBuffer(data)
        return TBinaryProtocol.TBinaryProtocol(tr).readMessageBegin()[0]
    except Exception:
        return None


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        m = method(body)
        op = None
        want = None
        if m in ("GetResultSetMetadata", "FetchResults"):
            req0 = decode(body, getattr(TCLIService, f"{m}_args"))[3].req
            op = req0.operationHandle.operationId.guid
            want = getattr(req0, "maxRows", None)
        if m == "OpenSession":
            # the connector asks for its protocol in client_protocol_i64; Spark reads
            # client_protocol (default V1, row-based results): ask for V10, column based
            name, mt, seq, args = decode(body, TCLIService.OpenSession_args)
            args.req.client_protocol = ttypes.TProtocolVersion.HIVE_CLI_SERVICE_PROTOCOL_V10
            body = encode(name, mt, seq, args)
        elif m == "GetColumns":
            # Spark lists no columns for Delta tables (their schema is in the Delta
            # log, not the metastore): answer with a query over DESCRIBE's result
            name, mt, seq, args = decode(body, TCLIService.GetColumns_args)
            q = columns_query(args.req)
            if q is not None:
                body = encode("ExecuteStatement", mt, seq, TCLIService.ExecuteStatement_args(
                    req=ttypes.TExecuteStatementReq(sessionHandle=args.req.sessionHandle, statement=q)))
                m = "GetColumns->ExecuteStatement"
        elif m == "ExecuteStatement":
            name, mt, seq, args = decode(body, TCLIService.ExecuteStatement_args)
            req = args.req
            m_copy = COPY.search(req.statement)
            if m_copy:
                req.statement = (f"INSERT INTO {m_copy.group(1)} SELECT {m_copy.group(2)} "
                                 f"FROM parquet.`{m_copy.group(3)}`")
            if req.parameters:
                req.statement = inline(req.statement, req.parameters)
                req.parameters = None
            body = encode(name, mt, seq, args)
        hdr = {k: v for k, v in self.headers.items() if k.lower() not in ("authorization", "host", "content-length")}
        hdr["Authorization"] = BASIC
        c = http.client.HTTPConnection("127.0.0.1", TARGET, timeout=7200)
        c.request("POST", self.path, body=body, headers=hdr)
        r = c.getresponse()
        data = r.read()
        if m == "GetColumns->ExecuteStatement" and r.status == 200:
            name, mt, seq, res = decode(data, TCLIService.ExecuteStatement_result)
            out = TCLIService.GetColumns_result(success=ttypes.TGetColumnsResp(
                status=res.success.status, operationHandle=res.success.operationHandle))
            data = encode("GetColumns", mt, seq, out)
        if m == "OpenSession" and r.status == 200:
            name, mt, seq, res = decode(data, TCLIService.OpenSession_result)
            if res.success is not None:
                res.success.serverProtocolVersion = ttypes.TProtocolVersion.SPARK_CLI_SERVICE_PROTOCOL_V8
            data = encode(name, mt, seq, res)
        elif m in ("GetResultSetMetadata", "FetchResults") and r.status == 200:
            # Spark leaves the Databricks result format unset: its row sets are column based
            name, mt, seq, res = decode(data, getattr(TCLIService, f"{m}_result"))
            if m == "GetResultSetMetadata":
                meta = res.success
                if meta is not None and meta.resultFormat is None:
                    meta.resultFormat = ttypes.TSparkRowSetType.COLUMN_BASED_SET
                META[op] = meta
            elif res.success is not None:
                if res.success.resultSetMetadata is None:
                    # Spark ignores includeResultSetMetadata: answer with the operation's metadata
                    res.success.resultSetMetadata = META.get(op)
                rs = res.success.results
                meta = res.success.resultSetMetadata
                if rs is not None and rs.columns and want:
                    # Spark always says hasMoreRows = false and expects clients to fetch
                    # until a batch is empty; the connector believes the flag: a full
                    # batch may have more rows after it
                    got = len(next(v for v in (rs.columns[0].stringVal, rs.columns[0].i32Val, rs.columns[0].i64Val,
                                              rs.columns[0].boolVal, rs.columns[0].doubleVal, rs.columns[0].byteVal,
                                              rs.columns[0].i16Val, rs.columns[0].binaryVal) if v is not None).values)
                    res.success.hasMoreRows = got >= want
                if rs is not None and rs.columns is None and meta is not None and meta.schema is not None:
                    # an empty result has no columns at all in Spark: send empty ones
                    rs.columns = [ttypes.TColumn(stringVal=ttypes.TStringColumn(values=[], nulls=b""))
                                  for _ in meta.schema.columns]
            data = encode(name, mt, seq, res)
        if DEBUG:
            print(m, r.status, len(data), flush=True)
            if m == "FetchResults":
                print(decode(data, TCLIService.FetchResults_result)[3].success, flush=True)
            if m == "OpenSession":
                print(decode(body, TCLIService.OpenSession_args)[3].req, flush=True)
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", LISTEN), H).serve_forever()
