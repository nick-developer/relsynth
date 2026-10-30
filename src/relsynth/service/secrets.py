# SPDX-License-Identifier: Apache-2.0
"""Secret references, resolved only at the moment of use.

The service stores a reference, never the secret:

    env:NAME                     an environment variable of the worker
    file:/run/secrets/pg         a mounted file (Kubernetes Secret, Docker secret)
    vault:kv/data/relsynth/pg#password   HashiCorp Vault KV v2 (VAULT_ADDR + VAULT_TOKEN,
                                 or the Kubernetes auth role in VAULT_K8S_ROLE)
    aws-sm:<secret id>[#field]   AWS Secrets Manager (boto3)
    azure-kv:<vault>/<name>      Azure Key Vault (azure-identity + azure-keyvault-secrets)
    gcp-sm:projects/p/secrets/s/versions/latest   Google Secret Manager

Each tenant may only use references below its own prefix when
``RELSYNTH_SVC_SECRET_PREFIX_<scheme>`` is set (for example
``vault:kv/data/relsynth/{tenant}/``), so one tenant cannot name another's secret.
"""
from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Mapping
from pathlib import Path

SCHEMES = ("env", "file", "vault", "aws-sm", "azure-kv", "gcp-sm")


class SecretError(Exception):
    pass


def check_ref(ref: str, tenant: str, env: Mapping[str, str] | None = None) -> None:
    """Refuse a malformed reference, or one outside the tenant's allowed prefix."""
    env = os.environ if env is None else env
    scheme, _, rest = ref.partition(":")
    if scheme not in SCHEMES or not rest:
        raise SecretError(f"secret reference must be <scheme>:<path> with scheme in {SCHEMES}")
    prefix = env.get(f"RELSYNTH_SVC_SECRET_PREFIX_{scheme.upper().replace('-', '_')}")
    if prefix is not None and not ref.startswith(prefix.replace("{tenant}", tenant)):
        raise SecretError(f"tenant {tenant} may only use {scheme} secrets below {prefix.replace('{tenant}', tenant)}")
    if ".." in rest.split("#")[0]:
        raise SecretError("secret reference may not contain '..'")


def resolve(ref: str) -> str:
    scheme, _, rest = ref.partition(":")
    path, _, fld = rest.partition("#")
    if scheme == "env":
        v = os.environ.get(path)
        if v is None:
            raise SecretError(f"environment variable {path} is not set on the worker")
        return v
    if scheme == "file":
        return Path(path).read_text(encoding="utf-8").strip()
    if scheme == "vault":
        return _vault(path, fld or "value")
    if scheme == "aws-sm":
        import boto3  # type: ignore[import-not-found]
        s = boto3.client("secretsmanager").get_secret_value(SecretId=path)["SecretString"]
        return json.loads(s)[fld] if fld else s
    if scheme == "azure-kv":
        from azure.identity import DefaultAzureCredential  # type: ignore[import-not-found]
        from azure.keyvault.secrets import SecretClient  # type: ignore[import-not-found]
        vault, _, name = path.partition("/")
        return SecretClient(f"https://{vault}.vault.azure.net", DefaultAzureCredential()).get_secret(name).value
    if scheme == "gcp-sm":
        from google.cloud import secretmanager  # type: ignore[import-not-found,attr-defined]
        r = secretmanager.SecretManagerServiceClient().access_secret_version(name=path)
        return r.payload.data.decode()
    raise SecretError(f"unknown secret scheme {scheme!r}")


def _vault_token() -> str:
    tok = os.environ.get("VAULT_TOKEN")
    if tok:
        return tok
    role = os.environ.get("VAULT_K8S_ROLE")
    if role:        # Kubernetes auth: exchange the pod's service-account JWT for a Vault token
        jwt = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
        mount = os.environ.get("VAULT_K8S_MOUNT", "kubernetes")
        out = _vault_call(f"/v1/auth/{mount}/login", None, {"role": role, "jwt": jwt})
        return out["auth"]["client_token"]
    raise SecretError("Vault needs VAULT_TOKEN or VAULT_K8S_ROLE")


def _vault_call(path: str, token: str | None, body: dict | None = None) -> dict:
    addr = os.environ.get("VAULT_ADDR")
    if not addr or not addr.startswith(("https://", "http://")):
        raise SecretError("VAULT_ADDR must be an http(s) URL")
    req = urllib.request.Request(addr.rstrip("/") + path, method="POST" if body is not None else "GET",  # noqa: S310
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **({"X-Vault-Token": token} if token else {})})
    ns = os.environ.get("VAULT_NAMESPACE")
    if ns:
        req.add_header("X-Vault-Namespace", ns)
    with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310  # nosec B310 (scheme checked above)
        return json.loads(r.read())


def _vault(path: str, fld: str) -> str:
    out = _vault_call("/v1/" + path.lstrip("/"), _vault_token())
    data = out.get("data", {})
    data = data.get("data", data)            # KV v2 nests the payload
    if fld not in data:
        raise SecretError(f"Vault secret {path} has no field {fld}")
    return str(data[fld])
