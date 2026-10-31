# SPDX-License-Identifier: Apache-2.0
"""Encrypted artefact storage, one folder per tenant.

Layout: ``<data_dir>/tenants/<tenant>/runs/<run>/<artefact id>.enc``.  In
Kubernetes each tenant's worker pool can mount its own volume at
``<data_dir>/tenants/<tenant>`` (Helm value ``tenants[].storage``), so a tenant's
worker cannot even see another tenant's ciphertext.
"""
from __future__ import annotations

import secrets as pysecrets
from collections.abc import Iterator
from pathlib import Path

from . import crypto
from .auth import TENANT_RE
from .store import Store


class Artifacts:
    def __init__(self, data_dir: str | Path, store: Store, wrapper: crypto.KeyWrapper):
        self.root, self.store, self.wrapper = Path(data_dir).resolve(), store, wrapper

    def tenant_dir(self, tenant: str) -> Path:
        if not TENANT_RE.match(tenant):
            raise ValueError(f"invalid tenant id {tenant!r}")
        return self.root / "tenants" / tenant

    def put(self, tenant: str, run_id: str, kind: str, name: str, src: Path) -> dict:
        from .store import new_id
        aid = new_id("art")
        dst = self.tenant_dir(tenant) / "runs" / run_id / f"{aid}.enc"
        dek = pysecrets.token_bytes(32)
        size, digest = crypto.encrypt_file(Path(src), dst, tenant, aid, dek)
        wrapped = self.wrapper.wrap(tenant, dek)
        rel = str(dst.relative_to(self.root))
        from .store import artifacts as t
        self.store._insert(t, tenant, dict(id=aid, run_id=run_id, kind=kind, name=name, path=rel, size=size,
                                           sha256=digest, wrapped_key=wrapped))
        return self.store.artifact(tenant, aid)

    def read(self, tenant: str, art: dict) -> Iterator[bytes]:
        """Plaintext chunks of an artefact (tenant-checked: the row must belong to `tenant`)."""
        if art["tenant_id"] != tenant:
            raise PermissionError("artefact belongs to another tenant")
        path = (self.root / art["path"]).resolve()
        if self.tenant_dir(tenant) not in path.parents:
            raise PermissionError("artefact path outside the tenant's folder")
        dek = self.wrapper.unwrap(tenant, art["wrapped_key"])
        return crypto.decrypt_stream(path, tenant, art["id"], dek)

    def read_bytes(self, tenant: str, art: dict) -> bytes:
        return b"".join(self.read(tenant, art))
