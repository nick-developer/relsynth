# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the service tests: a store, settings in dev-auth mode, a tiny
SQLite shop with unmistakable customer names (to prove that no source row is
persisted), and an RSA key pair with a JWKS file for the OIDC tests."""
from __future__ import annotations

import json
import random
import sqlite3
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from relsynth.service.config import Settings  # noqa: E402
from relsynth.service.store import Store  # noqa: E402

SECRET_MARK = "Qzx"          # every real customer name contains it; synthetic output must not


def make_shop(path: Path, n: int = 120, seed: int = 1) -> Path:
    rnd = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript("""
    CREATE TABLE customer (customer_id INTEGER PRIMARY KEY, name TEXT, segment TEXT, age INTEGER);
    CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customer(customer_id),
                         amount REAL, channel TEXT);
    """)
    oid = 1
    for i in range(1, n + 1):
        db.execute("INSERT INTO customer VALUES (?,?,?,?)",
                   (i, f"{SECRET_MARK}{i:04d}Person", rnd.choice("ABC"), rnd.randint(18, 80)))
        for _ in range(rnd.randint(0, 5)):
            db.execute("INSERT INTO orders VALUES (?,?,?,?)",
                       (oid, i, round(rnd.lognormvariate(3, 1), 2), rnd.choice(["web", "shop"])))
            oid += 1
    db.commit()
    db.close()
    return path


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.db_url = f"sqlite:///{tmp_path / 'svc.db'}"
    s.data_dir = str(tmp_path / "data")
    s.auth_mode, s.allow_dev_auth = "dev", True
    s.master_key_ref = f"file:{tmp_path / 'master.key'}"
    s.file_roots = str(tmp_path / "files")
    s.lease_s, s.heartbeat_s = 30, 1
    s.log_json = False
    return s


@pytest.fixture
def store(settings) -> Store:
    return Store(settings.db_url).create()


@pytest.fixture
def shop(settings) -> Path:
    return make_shop(Path(settings.file_roots) / "acme" / "shop.db")


def dev(user: str, *roles: str, **attrs) -> dict:
    h = f"{user}; {','.join(roles)}"
    if attrs:
        h += "; " + ",".join(f"{k}={v}" for k, v in attrs.items())
    return {"X-Relsynth-Dev-User": h}


@pytest.fixture
def rsa_jwks(tmp_path):
    from cryptography.hazmat.primitives.asymmetric import rsa
    from jwt.algorithms import RSAAlgorithm
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid="k1", use="sig", alg="RS256")
    f = tmp_path / "jwks.json"
    f.write_text(json.dumps({"keys": [jwk]}))
    return key, f


def token(key, **claims) -> str:
    import jwt
    now = int(time.time())
    body = {"iss": "https://idp.example", "aud": "relsynth", "sub": "u1", "iat": now, "exp": now + 300, **claims}
    return jwt.encode(body, key, algorithm="RS256", headers={"kid": "k1"})
