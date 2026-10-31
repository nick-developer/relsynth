# SPDX-License-Identifier: Apache-2.0
"""Encryption at rest (tamper, truncation, swap and cross-tenant use are caught),
secret references and the source policy."""
from __future__ import annotations

import os

import pytest
from cryptography.exceptions import InvalidTag

from relsynth.service import crypto
from relsynth.service import secrets as sec
from relsynth.service.artifacts import Artifacts
from relsynth.service.policy import SourceRefused, check_source


@pytest.fixture
def arts(settings, store):
    return Artifacts(settings.data_dir, store, crypto.KeyWrapper("local", settings.master_key_ref))


def _put(arts, tmp_path, tenant="acme", size=crypto.CHUNK * 2 + 123):
    f = tmp_path / "plain.bin"
    f.write_bytes(os.urandom(size))
    return f.read_bytes(), arts.put(tenant, "run_1", "data", "t.csv", f)


def test_roundtrip_and_ciphertext_differs(arts, tmp_path):
    plain, a = _put(arts, tmp_path)
    assert arts.read_bytes("acme", a) == plain and a["size"] == len(plain)
    enc = (arts.root / a["path"]).read_bytes()
    assert plain[:4096] not in enc
    assert (arts.root / "tenants" / "acme").is_dir()


@pytest.mark.parametrize("attack", ["flip", "truncate", "drop_last_chunk", "swap_chunks"])
def test_tampering_is_detected(arts, tmp_path, attack):
    _, a = _put(arts, tmp_path)
    p = arts.root / a["path"]
    b = bytearray(p.read_bytes())
    c = crypto.CHUNK + 16
    if attack == "flip":
        b[500] ^= 1
    elif attack == "truncate":
        b = b[:-10]
    elif attack == "drop_last_chunk":
        b = b[:12 + 2 * c]
    else:
        b = b[:12] + b[12 + c:12 + 2 * c] + b[12:12 + c] + b[12 + 2 * c:]
    p.write_bytes(bytes(b))
    with pytest.raises(InvalidTag):
        arts.read_bytes("acme", a)


def test_other_tenant_cannot_read_or_unwrap(arts, tmp_path):
    _, a = _put(arts, tmp_path, size=1000)
    with pytest.raises(PermissionError):
        arts.read_bytes("beta", a)
    forged = dict(a, tenant_id="beta")              # a forged row pointing into acme's folder: refused
    with pytest.raises(PermissionError):
        arts.read_bytes("beta", forged)
    import shutil                                    # acme's file copied into beta's folder: beta's KEK cannot unwrap it
    dst = arts.tenant_dir("beta") / "runs" / "run_1" / "copy.enc"
    dst.parent.mkdir(parents=True)
    shutil.copy(arts.root / a["path"], dst)
    with pytest.raises(InvalidTag):
        arts.read_bytes("beta", dict(forged, path=str(dst.relative_to(arts.root))))


def test_secret_refs(monkeypatch, tmp_path):
    monkeypatch.setenv("PG_PW", "s3cret")
    assert sec.resolve("env:PG_PW") == "s3cret"
    (tmp_path / "pw").write_text("filepw\n")
    assert sec.resolve(f"file:{tmp_path / 'pw'}") == "filepw"
    for bad in ("plain", "ftp:x", "file:../../etc/passwd", "vault:"):
        with pytest.raises(sec.SecretError):
            sec.check_ref(bad, "acme")
    env = {"RELSYNTH_SVC_SECRET_PREFIX_VAULT": "vault:kv/data/relsynth/{tenant}/"}
    sec.check_ref("vault:kv/data/relsynth/acme/pg#password", "acme", env)
    with pytest.raises(sec.SecretError):
        sec.check_ref("vault:kv/data/relsynth/beta/pg#password", "acme", env)


def test_source_policy(settings, tmp_path):
    root = tmp_path / "files"
    (root / "acme").mkdir(parents=True)
    (root / "beta").mkdir(parents=True)
    (root / "acme" / "link").symlink_to(root / "beta")
    ok = {"kind": "csv", "path": str(root / "acme" / "data")}
    assert check_source(ok, "acme", settings) is ok
    refused = [
        {"kind": "csv", "path": str(root / "beta")},
        {"kind": "csv", "path": str(root / "acme" / ".." / "beta")},
        {"kind": "csv", "path": str(root / "acme" / "link" / "x")},          # symlink out of the tenant folder
        {"kind": "sqlite", "path": "/etc/passwd"},
        {"kind": "postgresql", "url": "postgresql://u:pw@db/x"},
        {"kind": "postgresql", "url": "postgresql://u@db/x", "password": "pw"},
        {"kind": "duckdb-shell", "path": "x"},
        {"no": "kind"},
    ]
    for src in refused:
        with pytest.raises(SourceRefused):
            check_source(src, "acme", settings)
    settings.allowed_hosts = "*.db.internal"
    check_source({"kind": "postgresql", "url": "postgresql://u@pg.db.internal/x"}, "acme", settings)
    with pytest.raises(SourceRefused):
        check_source({"kind": "postgresql", "url": "postgresql://u@169.254.169.254/x"}, "acme", settings)
