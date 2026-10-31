# SPDX-License-Identifier: Apache-2.0
"""Encryption at rest: envelope encryption with AES-256-GCM.

Every artefact gets a fresh 256-bit data key.  The file is encrypted in 1 MiB
chunks (STREAM construction of Hoang, Reyhanitabar, Rogaway and Vizár 2015):
each chunk's nonce is a random 8-byte prefix plus the chunk counter, and the
associated data binds the tenant, the artefact id, the counter and a final-chunk
flag, so chunks cannot be reordered, truncated, or moved to another tenant or
artefact.  The data key is wrapped by the tenant's key-encryption key (KEK):

* ``local``: KEK = HKDF-SHA256(master key, "relsynth-tenant:<tenant>") (RFC 5869),
  master key from a secret reference (32 random bytes, base64 or raw);
* ``vault-transit``: Vault's transit engine with ``context=<tenant>`` (derived keys);
* ``aws-kms`` / ``azure-kv`` / ``gcp-kms``: the cloud KMS, with the tenant as
  encryption context or additional authenticated data.

So each tenant's files are readable only with that tenant's KEK, even by an
operator who can read the volume.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets as pysecrets
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import secrets as sec

MAGIC = b"RSE1"
CHUNK = 1 << 20


class KeyWrapper:
    """Wraps and unwraps data keys for one tenant."""

    def __init__(self, kms: str = "local", master_key_ref: str = "", key_id: str = ""):
        self.kms, self.key_id = kms, key_id
        self._master: bytes | None = None
        self.master_key_ref = master_key_ref

    # local master key ---------------------------------------------------------
    def master(self) -> bytes:
        if self._master is None:
            ref = self.master_key_ref
            if ref.startswith("file:") and not Path(ref[5:]).exists():
                p = Path(ref[5:])          # first start of a single-node install: create the key, owner-only
                p.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as f:
                    f.write(base64.b64encode(pysecrets.token_bytes(32)).decode())
            raw = sec.resolve(ref).strip()
            key = base64.b64decode(raw) if len(raw) != 32 else raw.encode()
            if len(key) != 32:
                raise ValueError("the master key must be 32 bytes (base64-encoded)")
            self._master = key
        return self._master

    def _kek(self, tenant: str) -> bytes:
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                    info=b"relsynth-tenant:" + tenant.encode()).derive(self.master())

    def wrap(self, tenant: str, dek: bytes) -> str:
        if self.kms == "local":
            n = pysecrets.token_bytes(12)
            return "local:" + base64.b64encode(n + AESGCM(self._kek(tenant)).encrypt(n, dek, tenant.encode())).decode()
        if self.kms == "vault-transit":
            ctx = base64.b64encode(tenant.encode()).decode()
            out = sec._vault_call(f"/v1/transit/encrypt/{self.key_id}", sec._vault_token(),
                                  {"plaintext": base64.b64encode(dek).decode(), "context": ctx})
            return "vault:" + out["data"]["ciphertext"]
        if self.kms == "aws-kms":
            import boto3  # type: ignore[import-not-found]
            blob = boto3.client("kms").encrypt(KeyId=self.key_id, Plaintext=dek,
                                               EncryptionContext={"tenant": tenant})["CiphertextBlob"]
            return "aws:" + base64.b64encode(blob).decode()
        if self.kms == "azure-kv":
            from azure.identity import DefaultAzureCredential  # type: ignore[import-not-found]
            from azure.keyvault.keys.crypto import CryptographyClient, KeyWrapAlgorithm  # type: ignore[import-not-found]
            r = CryptographyClient(self.key_id, DefaultAzureCredential()).wrap_key(KeyWrapAlgorithm.rsa_oaep_256, dek)
            # Azure key wrap has no associated data: the tenant tag is checked on unwrap, and
            # docs/service/tenancy.md asks for one Key Vault key per tenant (key_id per tenant)
            return "azure:" + base64.b64encode(r.encrypted_key).decode() + ":" + _th(tenant)
        if self.kms == "gcp-kms":
            from google.cloud import kms  # type: ignore[import-not-found,attr-defined]
            r = kms.KeyManagementServiceClient().encrypt(request={"name": self.key_id, "plaintext": dek,
                                                                  "additional_authenticated_data": tenant.encode()})
            return "gcp:" + base64.b64encode(r.ciphertext).decode()
        raise ValueError(f"unknown kms {self.kms}")

    def unwrap(self, tenant: str, wrapped: str) -> bytes:
        kind, _, body = wrapped.partition(":")
        if kind == "local":
            raw = base64.b64decode(body)
            return AESGCM(self._kek(tenant)).decrypt(raw[:12], raw[12:], tenant.encode())
        if kind == "vault":
            ctx = base64.b64encode(tenant.encode()).decode()
            out = sec._vault_call(f"/v1/transit/decrypt/{self.key_id}", sec._vault_token(),
                                  {"ciphertext": body, "context": ctx})
            return base64.b64decode(out["data"]["plaintext"])
        if kind == "aws":
            import boto3  # type: ignore[import-not-found]
            return boto3.client("kms").decrypt(CiphertextBlob=base64.b64decode(body),
                                               EncryptionContext={"tenant": tenant})["Plaintext"]
        if kind == "azure":
            blob, _, th = body.rpartition(":")
            if th != _th(tenant):
                raise ValueError("artefact key belongs to another tenant")
            from azure.identity import DefaultAzureCredential  # type: ignore[import-not-found]
            from azure.keyvault.keys.crypto import CryptographyClient, KeyWrapAlgorithm  # type: ignore[import-not-found]
            return CryptographyClient(self.key_id, DefaultAzureCredential()).unwrap_key(
                KeyWrapAlgorithm.rsa_oaep_256, base64.b64decode(blob)).key
        if kind == "gcp":
            from google.cloud import kms  # type: ignore[import-not-found,attr-defined]
            return kms.KeyManagementServiceClient().decrypt(request={
                "name": self.key_id, "ciphertext": base64.b64decode(body),
                "additional_authenticated_data": tenant.encode()}).plaintext
        raise ValueError("unknown wrapped-key format")


def _th(tenant: str) -> str:
    return hashlib.sha256(tenant.encode()).hexdigest()[:16]


def _aad(tenant: str, aid: str, i: int, last: bool) -> bytes:
    return json.dumps([tenant, aid, i, last]).encode()


def encrypt_file(src: Path, dst: Path, tenant: str, aid: str, dek: bytes) -> tuple[int, str]:
    """Encrypt src into dst; returns (plaintext size, plaintext SHA-256)."""
    a, prefix, h, size = AESGCM(dek), pysecrets.token_bytes(8), hashlib.sha256(), 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(MAGIC + prefix)
        cur, i = fi.read(CHUNK), 0
        while True:
            nxt = fi.read(CHUNK)
            last = not nxt
            h.update(cur)
            size += len(cur)
            fo.write(a.encrypt(prefix + i.to_bytes(4, "big"), cur, _aad(tenant, aid, i, last)))
            if last:
                break
            cur, i = nxt, i + 1
    return size, h.hexdigest()


def decrypt_stream(path: Path, tenant: str, aid: str, dek: bytes) -> Iterator[bytes]:
    """Yield plaintext chunks; raises InvalidTag on any tampering, truncation or swap."""
    a = AESGCM(dek)
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != MAGIC:
            raise ValueError("not a relsynth encrypted artefact")
        prefix, i = head[4:], 0
        cur = f.read(CHUNK + 16)
        while True:
            nxt = f.read(CHUNK + 16)
            last = not nxt
            yield a.decrypt(prefix + i.to_bytes(4, "big"), cur, _aad(tenant, aid, i, last))
            if last:
                return
            cur, i = nxt, i + 1


def decrypt_to(path: Path, out: IO[bytes], tenant: str, aid: str, dek: bytes) -> None:
    for chunk in decrypt_stream(path, tenant, aid, dek):
        out.write(chunk)
