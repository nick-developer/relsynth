# SPDX-License-Identifier: Apache-2.0
"""Settings of the relsynth service, read from environment variables.

Every setting has a variable ``RELSYNTH_SVC_<NAME>`` (upper case).  A YAML file
named by ``RELSYNTH_SVC_CONFIG`` can hold the same keys in lower case; the
environment wins over the file.  Nothing here holds a secret value: secrets are
references (``vault:...``, ``env:...``, ``file:...``) resolved at the moment of
use (see `relsynth.service.secrets`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Settings:
    # metadata store: SQLite for a single node, PostgreSQL for HA (docs/service/operations.md)
    db_url: str = "sqlite:///./relsynth-service/service.db"
    # encrypted artefacts, one folder per tenant (a PVC or a mounted volume per tenant in Kubernetes)
    data_dir: str = "./relsynth-service/data"
    # identity: "oidc" (production), "dev" (X-Relsynth-Dev-User header; refused unless explicitly allowed)
    auth_mode: str = "oidc"
    allow_dev_auth: bool = False
    oidc_issuer: str = ""
    oidc_audience: str = "relsynth"
    oidc_jwks_url: str = ""                 # default: <issuer>/.well-known/openid-configuration -> jwks_uri
    oidc_jwks_file: str = ""                # air-gapped: a JWKS file instead of a URL
    oidc_algorithms: str = "RS256,ES256,PS256,EdDSA"
    oidc_leeway_s: int = 30
    # where roles come from in the token: "<claim>" holding ["<tenant>:<role>", ...]
    roles_claim: str = "relsynth_roles"
    tenant_claim: str = ""                  # optional single-tenant claim, e.g. "tenant"
    attributes_claim: str = "relsynth_attrs"
    abac: bool = False                      # connections labelled with a classification need a matching clearance
    # browser single sign-on (the UI uses Authorization Code + PKCE against the issuer)
    ui_client_id: str = "relsynth-ui"
    # encryption at rest: master key reference for local envelope encryption, or a KMS
    kms: str = "local"                      # local | vault-transit | aws-kms | azure-kv | gcp-kms
    master_key_ref: str = "file:./relsynth-service/master.key"
    kms_key_id: str = ""
    # jobs
    worker_concurrency: int = 1
    worker_tenants: str = ""                # comma list: this worker only serves these tenants (per-tenant pools)
    worker_exclude_tenants: str = ""        # comma list: tenants with their own pool (the shared pool skips them)
    lease_s: int = 60
    heartbeat_s: int = 10
    max_attempts: int = 3
    job_timeout_s: int = 6 * 3600
    job_memory_mb: int = 0                  # 0 = no limit (cgroup limits in Kubernetes)
    job_cpu_s: int = 0
    # what sources a tenant may use; file sources only below file_roots/<tenant>
    allowed_kinds: str = ("csv,parquet,files,sqlite,postgresql,mssql,oracle,mysql,mariadb,"
                          "databricks,fabric,snowflake,bigquery,sqlalchemy")
    file_roots: str = ""                    # comma list of roots; each tenant sees <root>/<tenant id>
    allowed_hosts: str = ""                 # optional comma list of host globs a database URL may name
    # approvals (governance design D4: a named approval before data leaves the run)
    approval_internal: bool = True
    separation_of_duties: bool = True
    approval_quorum: int = 1
    # agents (T16, docs/agents.md): they draft and explain; people decide
    agent_backend: str = "rules"            # rules (no model) | openai (an OpenAI-compatible server, e.g. Ollama)
    agent_base_url: str = "http://localhost:11434/v1"
    agent_model: str = ""
    agent_api_key_env: str = ""
    agent_allow_remote: bool = False
    agent_scope_ceiling: str = "aggregates"  # the most an agent sees: schema | aggregates | values
    agent_name: str = "relsynth-agent"
    # HTTP
    max_body_mb: int = 16
    cors_origins: str = ""
    public_url: str = ""
    otel: bool = False                      # OpenTelemetry tracing and metrics (needs relsynth[service-otel])
    log_json: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, env: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ if env is None else env)
        raw: dict[str, Any] = {}
        cfg = env.get("RELSYNTH_SVC_CONFIG")
        if cfg:
            raw.update(yaml.safe_load(Path(cfg).read_text(encoding="utf-8")) or {})
        for f in fields(cls):
            v = env.get(f"RELSYNTH_SVC_{f.name.upper()}")
            if v is not None:
                raw[f.name] = v
        s = cls()
        for f in fields(cls):
            if f.name in raw and f.name != "extra":
                setattr(s, f.name, _coerce(getattr(s, f.name), raw.pop(f.name)))
        s.extra = {k: v for k, v in raw.items() if k != "extra"}
        return s

    def csv(self, name: str) -> list[str]:
        return [x.strip() for x in str(getattr(self, name) or "").split(",") if x.strip()]

    def check(self) -> list[str]:
        """Settings that are unsafe or inconsistent (startup refuses them)."""
        errs = []
        if self.auth_mode not in ("oidc", "dev"):
            errs.append(f"auth_mode must be oidc or dev, not {self.auth_mode!r}")
        if self.auth_mode == "dev" and not self.allow_dev_auth:
            errs.append("auth_mode=dev needs RELSYNTH_SVC_ALLOW_DEV_AUTH=true (never in production)")
        if self.auth_mode == "oidc" and not self.oidc_issuer:
            errs.append("auth_mode=oidc needs RELSYNTH_SVC_OIDC_ISSUER")
        if self.kms not in ("local", "vault-transit", "aws-kms", "azure-kv", "gcp-kms"):
            errs.append(f"unknown kms {self.kms!r}")
        if self.agent_backend not in ("rules", "openai"):
            errs.append(f"agent_backend must be rules or openai, not {self.agent_backend!r}")
        if self.agent_scope_ceiling not in ("schema", "aggregates", "values"):
            errs.append("agent_scope_ceiling must be schema, aggregates or values")
        if self.lease_s <= 2 * self.heartbeat_s:
            errs.append("lease_s must exceed twice heartbeat_s, or healthy jobs lose their lease")
        return errs


def _coerce(default: Any, v: Any) -> Any:
    if isinstance(default, bool):
        return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(v)
    return v if not isinstance(default, str) else str(v)
