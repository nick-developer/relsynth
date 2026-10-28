# SPDX-License-Identifier: Apache-2.0
"""Google BigQuery (sqlalchemy-bigquery over the BigQuery client library).

    bigquery://<project>/<dataset>[?location=EU]

The dataset is the schema; `schema` in the section overrides the URL's.
Credentials never go in the URL or the specification: `auth` picks the method
(see warehouse.py),

* default: Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS,
  `gcloud auth application-default login`, the metadata server of GCE, GKE
  workload identity or Cloud Run);
* service-account: key_file, a service account's JSON key;
* external-account: config_file, a workload identity federation configuration
  (AWS, Azure, OIDC), which holds no secret;
* impersonate: service_account, impersonated with the default credentials
  (roles/iam.serviceAccountTokenCreator).

Engine differences handled here:

* types: FLOAT64, STRING (no length), DATETIME (microseconds, no time zone),
  BYTES; generic types map to them;
* constraints: primary and foreign keys exist only NOT ENFORCED and are added
  after the table (ALTER TABLE ... ADD PRIMARY KEY (...) NOT ENFORCED); unique
  keys and indexes do not exist;
* the sample hash is the first 32 bits of SHA256('seed|key') (TO_HEX, then a
  hexadecimal INT64 cast), the same as MySQL, DuckDB, Databricks, Fabric and
  Snowflake;
* cost: every query is a job labelled relsynth=<query_tag> and limited by
  maximum_bytes_billed = guards.max_bytes, so BigQuery itself refuses a
  larger scan; before a read, sample or statistics the connector asks for a
  dry run (bytes processed, free) and stops when it exceeds the guard;
* loads: load jobs from Parquet (load_table_from_dataframe, free of query
  charges) by default, or `bulk: {method: insert}` for DML INSERT statements.
"""
from __future__ import annotations

import logging
import re

import sqlalchemy as sa

from .sqlfuncs import row_hash
from .warehouse import WarehouseConnector, resolve_auth

log = logging.getLogger("relsynth.connectors")

SCOPES = ("https://www.googleapis.com/auth/bigquery", "https://www.googleapis.com/auth/cloud-platform")


def label_value(tag: str) -> str:
    """A BigQuery label value: lower-case letters, digits, '_' and '-', at most 63 characters."""
    return re.sub(r"[^a-z0-9_-]", "_", tag.lower())[:63] or "relsynth"


def bigquery_type(ty):
    """The BigQuery type of a column type (see the module docstring)."""
    if isinstance(ty, sa.Enum) or (isinstance(ty, sa.String) and getattr(ty, "length", None)):
        return sa.String()                      # STRING(n) would reject longer values
    if isinstance(ty, sa.Float):
        return sa.Float()                       # FLOAT64
    if isinstance(ty, sa.DateTime) and not getattr(ty, "timezone", False):
        return sa.DateTime()                    # DATETIME
    return ty


class BigQueryConnector(WarehouseConnector):
    """A BigQuery dataset."""

    kind = "bigquery"
    AUTH_METHODS = {"default": (), "service-account": ("key_file",), "external-account": ("config_file",),
                    "impersonate": ("service_account",)}

    def __init__(self, url=None, schema=None, **kw):
        self._client = None
        super().__init__(url, schema, **kw)
        u = sa.engine.make_url(str(self.engine.url))
        self.project = u.host or None
        self.location = u.query.get("location")
        if self.schema is None and u.database:
            self.schema = u.database
            self._md = sa.MetaData(schema=self.schema)

    # ------------------------------------------------------------ connection
    def _engine_url(self, url):
        # the connector hands the dialect its own client (credentials, labels, byte limit)
        return sa.engine.make_url(url).update_query_dict({"user_supplied_client": "true"}).render_as_string(
            hide_password=False)

    def credentials(self):
        """google.auth credentials for the configured method."""
        import google.auth
        a = resolve_auth(self._auth, self.AUTH_METHODS, default="default")
        m = a["method"]
        if m == "service-account":
            from google.oauth2 import service_account
            return service_account.Credentials.from_service_account_file(a["key_file"], scopes=SCOPES)
        if m == "external-account":
            creds, _ = google.auth.load_credentials_from_file(a["config_file"], scopes=SCOPES)
            return creds
        creds, _ = google.auth.default(scopes=SCOPES)
        if m == "impersonate":
            from google.auth import impersonated_credentials
            return impersonated_credentials.Credentials(source_credentials=creds,
                                                        target_principal=a["service_account"],
                                                        target_scopes=list(SCOPES))
        return creds

    def job_config(self, **kw):
        """The default job settings: the relsynth label, the byte limit, the dataset."""
        from google.cloud import bigquery
        cfg = bigquery.QueryJobConfig(labels={"relsynth": label_value(self.query_tag)}, **kw)
        if self.max_bytes is not None:
            cfg.maximum_bytes_billed = int(self.max_bytes)
        if self.schema and self.project:
            cfg.default_dataset = f"{self.project}.{self.schema}"
        return cfg

    def client(self):
        """The BigQuery client every connection of the engine uses (made on first use)."""
        if self._client is None:
            from google.cloud import bigquery
            self._client = bigquery.Client(project=self.project, credentials=self.credentials(),
                                           location=self.location, default_query_job_config=self.job_config())
        return self._client

    def _setup_engine(self, engine):
        super()._setup_engine(engine)

        @sa.event.listens_for(engine, "do_connect")
        def _client(dialect, conn_rec, cargs, cparams):
            cparams["client"] = self.client()

    def tag_statements(self, engine):
        """The tag is the job label relsynth=<tag> (set on every job), not a comment."""

    def close(self):
        super().close()
        if self._client is not None:
            self._client.close()

    # ------------------------------------------------------------ catalog
    def create_schema(self, schema=None):
        schema = schema or self.schema
        if not schema:
            return
        from google.cloud import bigquery
        ds = bigquery.Dataset(f"{self.project}.{schema}")
        if self.location:
            ds.location = self.location
        self.client().create_dataset(ds, exists_ok=True)

    def _table_bytes(self, table):
        return int(self.client().get_table(f"{self.project}.{self.schema}.{self.stored_name(table)}").num_bytes or 0)

    def estimate_bytes(self, q, tables):
        """The bytes a statement processes, from a dry run (free); the stored size
        of the tables when there is no statement."""
        if q is None:
            return super().estimate_bytes(q, tables)
        sql = str(q.compile(self.engine, compile_kwargs={"literal_binds": True}))
        job = self.client().query(sql, job_config=self.job_config(dry_run=True, use_query_cache=False))
        return int(job.total_bytes_processed or 0)

    def guard_scan(self, table, q=None, what="statement"):
        # the dry run of the whole-table read; statistics and samples scan the same columns at most
        super().guard_scan(table, q if q is not None else self._select(table), what)

    def cost_extra(self):
        return {"billing": "on-demand: bytes processed per query (dry run estimates; maximum_bytes_billed "
                           "= guards.max_bytes stops a larger query); capacity: slot time",
                "job_label": f"relsynth={label_value(self.query_tag)}"}

    def _row_hash(self, cols, seed):
        return row_hash(cols, seed, "bigquery")

    # ------------------------------------------------------------ writes
    def adapt_metadata(self, md):
        """BigQuery types; primary and foreign keys NOT ENFORCED (after the table);
        no unique keys, indexes or identity columns."""
        out = sa.MetaData(schema=md.schema)
        prep = self.engine.dialect.identifier_preparer

        def full(t):
            return prep.quote(f"{self.project}.{t.schema}.{t.name}" if t.schema else t.name)

        def q(names):
            return ", ".join(prep.quote(n) for n in names)
        for tbl in md.sorted_tables:
            pk = [c.name for c in tbl.primary_key.columns]
            cols = [sa.Column(c.name, bigquery_type(c.type), nullable=c.nullable and c.name not in pk,
                              autoincrement=False) for c in tbl.columns]
            new = sa.Table(tbl.name, out, *cols)
            ddl = []
            if pk:
                ddl.append(f"ALTER TABLE {full(tbl)} ADD PRIMARY KEY ({q(pk)}) NOT ENFORCED")
            for fk in tbl.foreign_key_constraints:
                ddl.append(f"ALTER TABLE {full(tbl)} ADD FOREIGN KEY ({q([c.name for c in fk.columns])}) "
                           f"REFERENCES {full(fk.referred_table)} ({q([e.column.name for e in fk.elements])}) "
                           "NOT ENFORCED")
            new.info["rs_constraints"] = ddl          # run after CREATE TABLE
            for stmt in ddl:
                sa.event.listen(new, "after_create", sa.DDL(stmt.replace("%", "%%")))
        return out

    def _insert_rows(self, con, tbl, df, chunk=20_000):
        if self.bulk.get("method", "load") == "load" and len(df):
            return self._load(tbl, df)
        return super()._insert_rows(con, tbl, df, chunk)

    def _load(self, tbl, df) -> int:
        """A load job from the rows as Parquet (no query charges); appends to the table."""
        from google.cloud import bigquery
        cfg = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND", labels={"relsynth": label_value(self.query_tag)})
        dest = f"{self.project}.{tbl.schema or self.schema}.{tbl.name}"
        job = self.client().load_table_from_dataframe(df, dest, job_config=cfg)
        job.result()
        return int(job.output_rows or len(df))


__all__ = ["BigQueryConnector", "bigquery_type", "label_value"]
