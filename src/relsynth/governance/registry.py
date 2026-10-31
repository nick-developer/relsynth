# SPDX-License-Identifier: Apache-2.0
"""The governance registry: content-addressed objects and a hash-chained log
(design sec. 10, Prop. 803).

    registry/
      config.json               registry id and the privacy owner's root key
      objects/sha256/ab/cdef…   envelopes, reports, policies (content-addressed)
      log.jsonl                 hash-chained entries: h_i = SHA-256(h_{i-1} || JCS(e_i))
      checkpoints/<seq>.dsse    signed heads (seq, hash)
      keys/trusted.dsse         public keys and roles, signed by the privacy owner
      secrets/                  HMAC key for locators, label salts (mode 0600)
      lock                      advisory lock for append and ledger operations

`Registry` is the interface (a Protocol); `LocalRegistry` is the folder
backend.  A database backend comes with T09.  Entries are never edited: an
approval, revocation or ledger settlement is a new entry that names the
earlier one.
"""
from __future__ import annotations

import contextlib
import json
import os
import secrets as _secrets
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Protocol

from ..errors import GovernanceError
from .canon import canonical, sha256
from .sign import PublicKey, Signer, envelope, payload, verify_envelope

GENESIS = "0" * 64
CHECKPOINT_TYPE = "https://github.com/nick-developer/relsynth/checkpoint/1"
KEYS_TYPE = "https://github.com/nick-developer/relsynth/trusted-keys/1"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _lock(fh) -> None:
    try:
        import fcntl
        fcntl.flock(fh, fcntl.LOCK_EX)
    except ImportError:                       # Windows
        import msvcrt
        import time
        fh.seek(0)
        while True:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
                return
            except OSError:
                time.sleep(0.05)


def _unlock(fh) -> None:
    try:
        import fcntl
        fcntl.flock(fh, fcntl.LOCK_UN)
    except ImportError:                       # Windows
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]


def chain_hash(prev: str, entry: dict) -> str:
    return sha256(prev.encode() + canonical(entry))


class Registry(Protocol):
    """What the governance layer needs from a registry backend."""

    def put_blob(self, data: bytes) -> str: ...
    def get_blob(self, digest: str) -> bytes: ...
    def append(self, entry: dict) -> int: ...
    def head(self) -> tuple[int, str]: ...
    def entries(self, type_prefix: str | None = None) -> Iterator[dict]: ...
    def locked(self): ...
    def secret(self, name: str, create: bool = True) -> bytes: ...


class LocalRegistry:
    """A registry in a local folder (single host; the lock is an fcntl lock)."""

    _tlock = threading.RLock()          # serialises threads of one process; flock serialises processes

    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / "config.json").exists():
            raise GovernanceError(f"{self.root} is not a relsynth registry; create one with `relsynth gov init`")
        self.config = json.loads((self.root / "config.json").read_text())
        self._depth = 0

    # ------------------------------------------------------------- creation
    @classmethod
    def create(cls, root, owner: Signer, owner_role: str = "dpo") -> LocalRegistry:
        """A new registry whose root of trust is the privacy owner's key."""
        r = Path(root)
        if (r / "config.json").exists():
            raise GovernanceError(f"{r} already holds a registry")
        for d in ("objects/sha256", "checkpoints", "keys", "secrets"):
            (r / d).mkdir(parents=True, exist_ok=True)
        os.chmod(r / "secrets", 0o700)
        cfg = {"schema": "relsynth.registry/1", "id": str(uuid.uuid4()), "created": now(),
               "root_key": {"id": owner.id, "name": owner.name, "public": owner.public.to_text(), "role": owner_role}}
        (r / "config.json").write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
        (r / "log.jsonl").touch()
        reg = cls(r)
        reg.set_trusted([{"id": owner.id, "name": owner.name, "public": owner.public.to_text(), "roles": [owner_role]}],
                        owner)
        return reg

    @property
    def id(self) -> str:
        return self.config["id"]

    @property
    def uri(self) -> str:
        return f"relsynth://{self.id}"

    # ------------------------------------------------------------- objects
    def _obj(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise GovernanceError(f"not a sha256 digest: {digest!r}")
        return self.root / "objects" / "sha256" / digest[:2] / digest[2:]

    def put_blob(self, data: bytes) -> str:
        d = sha256(data)
        p = self._obj(d)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp" + uuid.uuid4().hex[:6])
            tmp.write_bytes(data)
            os.replace(tmp, p)
        return d

    def get_blob(self, digest: str) -> bytes:
        p = self._obj(digest)
        if not p.exists():
            raise GovernanceError(f"object {digest} is missing from the registry")
        return p.read_bytes()

    def put_json(self, obj) -> str:
        return self.put_blob(canonical(obj))

    def get_json(self, digest: str):
        return json.loads(self.get_blob(digest))

    # ------------------------------------------------------------- the log
    @contextlib.contextmanager
    def locked(self):
        """The registry lock (re-entrant within a thread)."""
        with self._tlock:
            if self._depth:
                self._depth += 1
                try:
                    yield self
                finally:
                    self._depth -= 1
                return
            with open(self.root / "lock", "a+") as fh:
                _lock(fh)
                self._depth = 1
                try:
                    yield self
                finally:
                    self._depth = 0
                    _unlock(fh)

    def _read_log(self) -> list[dict]:
        out = []
        with open(self.root / "log.jsonl", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError as e:
                    raise GovernanceError(f"log line {i + 1} is not JSON: the registry log was damaged") from e
        return out

    def head(self) -> tuple[int, str]:
        log = self._read_log()
        return (log[-1]["seq"], log[-1]["hash"]) if log else (-1, GENESIS)

    def append(self, entry: dict) -> int:
        """Append one entry; returns its sequence number."""
        if "type" not in entry:
            raise GovernanceError("a log entry needs a type")
        entry = dict(entry)
        entry.setdefault("time", now())
        with self.locked():
            seq, prev = self.head()
            rec = {"seq": seq + 1, "prev": prev, "entry": entry, "hash": chain_hash(prev, entry)}
            with open(self.root / "log.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return seq + 1

    def entries(self, type_prefix: str | None = None) -> Iterator[dict]:
        for rec in self._read_log():
            if type_prefix is None or rec["entry"]["type"].startswith(type_prefix):
                yield rec

    # ------------------------------------------------------------- keys
    def root_key(self) -> PublicKey:
        return PublicKey.from_text(self.config["root_key"]["public"])

    def set_trusted(self, keys: list[dict], owner: Signer) -> int:
        """Replace the trusted keys and roles; only the root key may sign the list."""
        if owner.id != self.config["root_key"]["id"]:
            raise GovernanceError("only the privacy owner's root key may sign the trusted keys")
        st = {"_type": "https://in-toto.io/Statement/v1",
              "subject": [{"name": "trusted-keys", "digest": {"sha256": sha256(canonical(keys))}}],
              "predicateType": KEYS_TYPE, "predicate": {"keys": keys, "issued": now()}}
        env = envelope(st, owner)
        d = self.put_json(env)
        (self.root / "keys" / "trusted.dsse").write_text(json.dumps(env, indent=1) + "\n", encoding="utf-8")
        return self.append({"type": "keys", "envelope": d})

    def trusted(self) -> dict[str, dict]:
        """{key id: {name, public: PublicKey, roles}} from the current signed list."""
        recs = list(self.entries("keys"))
        if not recs:
            raise GovernanceError("the registry has no trusted keys")
        env = self.get_json(recs[-1]["entry"]["envelope"])
        if not verify_envelope(env, {self.config["root_key"]["id"]: self.root_key()}):
            raise GovernanceError("the trusted-keys list is not signed by the registry's root key")
        out = {}
        for k in payload(env)["predicate"]["keys"]:
            pk = PublicKey.from_text(k["public"])
            if pk.id != k["id"]:
                raise GovernanceError(f"trusted key {k.get('name')}: id does not match its public key")
            out[k["id"]] = {"name": k["name"], "public": pk, "roles": list(k.get("roles", []))}
        return out

    def trust(self, owner: Signer, public: PublicKey, name: str, roles: list[str]) -> int:
        cur = self.trusted()
        keys = [{"id": i, "name": v["name"], "public": v["public"].to_text(), "roles": v["roles"]}
                for i, v in cur.items() if i != public.id]
        keys.append({"id": public.id, "name": name, "public": public.to_text(), "roles": sorted(set(roles))})
        return self.set_trusted(keys, owner)

    def public_keys(self) -> dict[str, PublicKey]:
        return {i: v["public"] for i, v in self.trusted().items()}

    # ------------------------------------------------------------- checkpoints
    def checkpoint(self, signer: Signer) -> dict:
        """Sign the current head; the envelope is stored in checkpoints/<seq>.dsse
        and should also be kept outside the operator's control (design sec. 7)."""
        with self.locked():
            seq, h = self.head()
            st = {"_type": "https://in-toto.io/Statement/v1",
                  "subject": [{"name": f"{self.uri}/log", "digest": {"sha256": h}}],
                  "predicateType": CHECKPOINT_TYPE, "predicate": {"registry": self.id, "seq": seq, "hash": h,
                                                                  "time": now()}}
            env = envelope(st, signer)
            (self.root / "checkpoints" / f"{seq}.dsse").write_text(json.dumps(env, indent=1) + "\n", encoding="utf-8")
            return env

    def checkpoints(self) -> list[dict]:
        out = []
        for p in sorted((self.root / "checkpoints").glob("*.dsse"), key=lambda q: int(q.stem)):
            out.append(json.loads(p.read_text()))
        return out

    # ------------------------------------------------------------- secrets
    def secret(self, name: str, create: bool = True) -> bytes:
        """A 256-bit secret kept in secrets/<name>.key (mode 0600)."""
        if not name.replace("_", "").replace("-", "").isalnum():
            raise GovernanceError(f"bad secret name {name!r}")
        p = self.root / "secrets" / f"{name}.key"
        if p.exists():
            return bytes.fromhex(p.read_text().strip())
        if not create:
            raise GovernanceError(f"secret {name} does not exist")
        with self.locked():
            if not p.exists():
                fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as f:
                    f.write(_secrets.token_hex(32))
        return bytes.fromhex(p.read_text().strip())


def open_registry(root) -> LocalRegistry:
    return LocalRegistry(root)
