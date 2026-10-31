# SPDX-License-Identifier: Apache-2.0
"""Ed25519 keys and DSSE v1 envelopes (design sec. 7).

A statement (manifest, approval, checkpoint, release token, trusted-keys
list, organisation policy) is signed in a DSSE envelope: the signature covers
the pre-authentication encoding PAE(payloadType, payload), so a signature over
one kind of statement can never be replayed as another.  Keys are local
Ed25519 keys (RFC 8032); a key's id is ``ed25519:SHA256:<base64 of the SHA-256
of the raw public key>``, the form `ssh-keygen -l` prints.

Private keys are written as unencrypted PKCS#8 PEM with mode 0600, or
encrypted when a passphrase is given (environment variable
RELSYNTH_KEY_PASSPHRASE).  In T09 they move to a KMS or HSM.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..errors import GovernanceError
from .canon import canonical

PAYLOAD_TYPE = "application/vnd.in-toto+json"


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as e:  # pragma: no cover - cryptography is a dependency of the gov extra
        raise GovernanceError("signing needs the 'cryptography' package: pip install 'relsynth[gov]'") from e
    return ed25519, serialization


def key_id(raw_public: bytes) -> str:
    return "ed25519:SHA256:" + base64.b64encode(hashlib.sha256(raw_public).digest()).decode().rstrip("=")


@dataclass
class PublicKey:
    raw: bytes

    @property
    def id(self) -> str:
        return key_id(self.raw)

    def verify(self, sig: bytes, msg: bytes) -> bool:
        ed, _ = _crypto()
        from cryptography.exceptions import InvalidSignature
        try:
            ed.Ed25519PublicKey.from_public_bytes(self.raw).verify(sig, msg)
            return True
        except InvalidSignature:
            return False

    def to_text(self) -> str:
        return base64.b64encode(self.raw).decode()

    @classmethod
    def from_text(cls, s: str) -> PublicKey:
        raw = base64.b64decode(s)
        if len(raw) != 32:
            raise GovernanceError("an Ed25519 public key has 32 bytes")
        return cls(raw)


class Signer:
    """An Ed25519 private key with a display name."""

    def __init__(self, private, name: str = ""):
        self._k = private
        self.name = name
        _, ser = _crypto()
        self.public = PublicKey(private.public_key().public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw))

    @property
    def id(self) -> str:
        return self.public.id

    def sign(self, msg: bytes) -> bytes:
        return self._k.sign(msg)

    @classmethod
    def generate(cls, name: str = "") -> Signer:
        ed, _ = _crypto()
        return cls(ed.Ed25519PrivateKey.generate(), name)

    def save(self, path, passphrase: bytes | None = None) -> Path:
        _, ser = _crypto()
        enc = ser.BestAvailableEncryption(passphrase) if passphrase else ser.NoEncryption()
        pem = self._k.private_bytes(ser.Encoding.PEM, ser.PrivateFormat.PKCS8, enc)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(pem)
        meta = {"name": self.name, "id": self.id, "public": self.public.to_text()}
        p.with_suffix(".pub.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(cls, path, passphrase: bytes | None = None) -> Signer:
        _, ser = _crypto()
        p = Path(path)
        if not p.exists():
            raise GovernanceError(f"no signing key at {p}; create one with `relsynth gov keys new`")
        if passphrase is None and os.environ.get("RELSYNTH_KEY_PASSPHRASE"):
            passphrase = os.environ["RELSYNTH_KEY_PASSPHRASE"].encode()
        k = ser.load_pem_private_key(p.read_bytes(), password=passphrase)
        meta = p.with_suffix(".pub.json")
        name = json.loads(meta.read_text())["name"] if meta.exists() else p.stem
        return cls(k, name)


# --------------------------------------------------------------------- DSSE
def pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE v1 pre-authentication encoding."""
    t = payload_type.encode()
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(payload), payload)


def envelope(statement: dict, signers, payload_type: str = PAYLOAD_TYPE) -> dict:
    """A DSSE envelope over the canonical bytes of `statement`, signed by every signer."""
    payload = canonical(statement)
    msg = pae(payload_type, payload)
    if isinstance(signers, Signer):
        signers = [signers]
    return {"payloadType": payload_type, "payload": base64.b64encode(payload).decode(),
            "signatures": [{"keyid": s.id, "sig": base64.b64encode(s.sign(msg)).decode()} for s in signers]}


def payload(env: dict) -> dict:
    """The statement inside an envelope (not verified)."""
    try:
        return json.loads(base64.b64decode(env["payload"]))
    except (KeyError, ValueError) as e:
        raise GovernanceError(f"not a DSSE envelope: {e}") from e


def verify_envelope(env: dict, keys: dict) -> list[str]:
    """The key ids in `keys` ({id: PublicKey}) whose signatures on `env` are valid.

    The payload must also be in canonical form, so that the digest of a statement
    is the digest of the bytes that were signed."""
    try:
        raw = base64.b64decode(env["payload"])
        msg = pae(env["payloadType"], raw)
    except (KeyError, ValueError) as e:
        raise GovernanceError(f"not a DSSE envelope: {e}") from e
    if canonical(json.loads(raw)) != raw:
        return []
    good = []
    for s in env.get("signatures", []):
        k = keys.get(s.get("keyid"))
        if k is not None and k.verify(base64.b64decode(s.get("sig", "")), msg):
            good.append(s["keyid"])
    return good
