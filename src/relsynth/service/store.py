# SPDX-License-Identifier: Apache-2.0
"""Metadata store of the service (SQLAlchemy Core; SQLite or PostgreSQL).

Tenant isolation is enforced here, below the API: every read and write takes
the caller's tenant id and filters on it, so a handler cannot fetch another
tenant's row by id (it gets ``None``, which the API turns into 404).  The store
holds metadata only.  Source rows never reach it: a connection keeps its
location and a *reference* to its secret; reports and synthetic files are
encrypted artefacts on disk (`relsynth.service.artifacts`).
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

md = sa.MetaData()


def _tid(c: str) -> sa.Column:
    return sa.Column(c, sa.String(64), nullable=False, index=True)


tenants = sa.Table("tenants", md,
                   sa.Column("id", sa.String(64), primary_key=True),
                   sa.Column("name", sa.String(200), nullable=False),
                   sa.Column("created_at", sa.Float, nullable=False))

projects = sa.Table("projects", md,
                    sa.Column("id", sa.String(40), primary_key=True), _tid("tenant_id"),
                    sa.Column("name", sa.String(200), nullable=False),
                    sa.Column("description", sa.Text, default=""),
                    sa.Column("created_by", sa.String(200)), sa.Column("created_at", sa.Float))

connections = sa.Table("connections", md,
                       sa.Column("id", sa.String(40), primary_key=True), _tid("tenant_id"),
                       sa.Column("project_id", sa.String(40), nullable=False, index=True),
                       sa.Column("name", sa.String(200), nullable=False),
                       sa.Column("source", sa.Text, nullable=False),        # JSON source section, no secrets
                       sa.Column("secret_ref", sa.String(500)),             # vault:/env:/file:/... reference
                       sa.Column("labels", sa.Text, default="{}"),          # ABAC attributes (classification, ...)
                       sa.Column("created_by", sa.String(200)), sa.Column("created_at", sa.Float))

specs = sa.Table("specs", md,
                 sa.Column("id", sa.String(40), primary_key=True), _tid("tenant_id"),
                 sa.Column("project_id", sa.String(40), nullable=False, index=True),
                 sa.Column("connection_id", sa.String(40)),
                 sa.Column("name", sa.String(200), nullable=False),
                 sa.Column("version", sa.Integer, nullable=False),
                 sa.Column("body", sa.Text, nullable=False),                # YAML
                 sa.Column("review", sa.Text, default="[]"),                # introspection questions
                 sa.Column("created_by", sa.String(200)), sa.Column("created_at", sa.Float),
                 sa.UniqueConstraint("tenant_id", "project_id", "name", "version"))

runs = sa.Table("runs", md,
                sa.Column("id", sa.String(40), primary_key=True), _tid("tenant_id"),
                sa.Column("project_id", sa.String(40), nullable=False, index=True),
                sa.Column("spec_id", sa.String(40), nullable=False),
                sa.Column("connection_id", sa.String(40)),
                sa.Column("kind", sa.String(20), nullable=False, default="run"),   # run | introspect
                sa.Column("params", sa.Text, default="{}"),
                sa.Column("context", sa.String(40), default="internal"),
                sa.Column("state", sa.String(20), nullable=False, index=True),     # queued running succeeded failed cancelled
                sa.Column("release_state", sa.String(30), default="none"),         # none pending_approval approved rejected
                sa.Column("attempts", sa.Integer, default=0),
                sa.Column("max_attempts", sa.Integer, default=3),
                sa.Column("next_attempt_at", sa.Float, default=0.0),
                sa.Column("lease_owner", sa.String(100)), sa.Column("lease_expires", sa.Float),
                sa.Column("cancel_requested", sa.Boolean, default=False),
                sa.Column("exit_code", sa.Integer), sa.Column("error", sa.Text),
                sa.Column("summary", sa.Text, default="{}"),
                sa.Column("created_by", sa.String(200)), sa.Column("created_at", sa.Float),
                sa.Column("started_at", sa.Float), sa.Column("finished_at", sa.Float))

artifacts = sa.Table("artifacts", md,
                     sa.Column("id", sa.String(40), primary_key=True), _tid("tenant_id"),
                     sa.Column("run_id", sa.String(40), nullable=False, index=True),
                     sa.Column("kind", sa.String(20), nullable=False),               # report quality bundle data spec
                     sa.Column("name", sa.String(300), nullable=False),
                     sa.Column("path", sa.String(500), nullable=False),
                     sa.Column("size", sa.BigInteger), sa.Column("sha256", sa.String(64)),
                     sa.Column("wrapped_key", sa.Text, nullable=False),
                     sa.Column("created_at", sa.Float))

approvals = sa.Table("approvals", md,
                     sa.Column("id", sa.String(40), primary_key=True), _tid("tenant_id"),
                     sa.Column("run_id", sa.String(40), nullable=False, index=True),
                     sa.Column("decision", sa.String(10), nullable=False),           # approve | reject | revoke
                     sa.Column("approver", sa.String(200), nullable=False),
                     sa.Column("role", sa.String(40)), sa.Column("reason", sa.Text),
                     sa.Column("created_at", sa.Float))

service_accounts = sa.Table("service_accounts", md,
                            sa.Column("id", sa.String(40), primary_key=True), _tid("tenant_id"),
                            sa.Column("name", sa.String(200), nullable=False),
                            sa.Column("roles", sa.Text, nullable=False),
                            sa.Column("token_hash", sa.String(128), nullable=False),
                            sa.Column("created_by", sa.String(200)), sa.Column("created_at", sa.Float),
                            sa.Column("revoked", sa.Boolean, default=False))

audit = sa.Table("audit", md,
                 sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                 _tid("tenant_id"),
                 sa.Column("seq", sa.Integer, nullable=False),
                 sa.Column("ts", sa.Float, nullable=False),
                 sa.Column("actor", sa.String(200)), sa.Column("action", sa.String(80), nullable=False),
                 sa.Column("target", sa.String(200)), sa.Column("outcome", sa.String(20)),
                 sa.Column("detail", sa.Text, default="{}"),
                 sa.Column("prev_hash", sa.String(64), nullable=False),
                 sa.Column("hash", sa.String(64), nullable=False),
                 sa.UniqueConstraint("tenant_id", "seq"))

GENESIS = "0" * 64
TERMINAL = ("succeeded", "failed", "cancelled")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def canonical(obj: Any) -> bytes:
    """Deterministic JSON (sorted keys, no spaces) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


def hash_token(token: str) -> str:
    # tokens are 256-bit random values, so one SHA-256 is enough (no dictionary to attack)
    return hashlib.sha256(token.encode()).hexdigest()


def _rows(rs) -> list[dict]:
    return [_dict(r) for r in rs]


def _row(r) -> dict | None:
    return None if r is None else _dict(r)


def _dict(r) -> dict:
    d = dict(r._mapping)
    for k in ("source", "labels", "review", "params", "summary", "detail", "roles"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except ValueError:
                pass
    return d


class Store:
    def __init__(self, url: str):
        kw: dict[str, Any] = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            kw["connect_args"] = {"check_same_thread": False, "timeout": 30}
        self.engine = sa.create_engine(url, **kw)
        if url.startswith("sqlite"):
            @sa.event.listens_for(self.engine, "connect")
            def _wal(dbapi_conn, _rec):        # concurrent readers while a worker writes
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.close()

    def create(self) -> Store:
        md.create_all(self.engine)
        return self

    def ping(self) -> bool:
        with self.engine.connect() as c:
            c.execute(sa.text("SELECT 1"))
        return True

    # tenants ----------------------------------------------------------------
    def ensure_tenant(self, tid: str, name: str | None = None) -> None:
        with self.engine.begin() as c:
            if c.execute(sa.select(tenants.c.id).where(tenants.c.id == tid)).first() is None:
                try:
                    c.execute(tenants.insert().values(id=tid, name=name or tid, created_at=time.time()))
                except IntegrityError:      # another replica created it first
                    pass

    def list_tenants(self) -> list[dict]:
        with self.engine.connect() as c:
            return _rows(c.execute(sa.select(tenants).order_by(tenants.c.id)))

    # generic tenant-scoped CRUD -------------------------------------------
    def _insert(self, table: sa.Table, tid: str, values: dict) -> dict:
        v = {k: (json.dumps(x) if isinstance(x, (dict, list)) else x) for k, x in values.items()}
        v["tenant_id"] = tid
        v.setdefault("created_at", time.time())
        with self.engine.begin() as c:
            c.execute(table.insert().values(**v))
        row = self._get(table, tid, v["id"])
        assert row is not None
        return row

    def _get(self, table: sa.Table, tid: str, rid: str) -> dict | None:
        with self.engine.connect() as c:
            return _row(c.execute(sa.select(table).where(table.c.id == rid, table.c.tenant_id == tid)).first())

    def _list(self, table: sa.Table, tid: str, limit: int = 200, **eq) -> list[dict]:
        q = sa.select(table).where(table.c.tenant_id == tid)
        for k, v in eq.items():
            if v is not None:
                q = q.where(table.c[k] == v)
        with self.engine.connect() as c:
            return _rows(c.execute(q.order_by(table.c.created_at.desc()).limit(limit)))

    def _update(self, table: sa.Table, tid: str, rid: str, where=None, **values) -> bool:
        v = {k: (json.dumps(x) if isinstance(x, (dict, list)) else x) for k, x in values.items()}
        q = table.update().where(table.c.id == rid, table.c.tenant_id == tid)
        if where is not None:
            q = q.where(where)
        with self.engine.begin() as c:
            return c.execute(q.values(**v)).rowcount == 1

    # projects, connections, specs ------------------------------------------
    def add_project(self, tid, name, description, actor) -> dict:
        return self._insert(projects, tid, dict(id=new_id("prj"), name=name, description=description or "",
                                                created_by=actor))

    def project(self, tid, pid):
        return self._get(projects, tid, pid)

    def projects(self, tid):
        return self._list(projects, tid)

    def add_connection(self, tid, pid, name, source, secret_ref, labels, actor) -> dict:
        return self._insert(connections, tid, dict(id=new_id("con"), project_id=pid, name=name, source=source,
                                                   secret_ref=secret_ref, labels=labels or {}, created_by=actor))

    def connection(self, tid, cid):
        return self._get(connections, tid, cid)

    def connections(self, tid, pid=None):
        return self._list(connections, tid, project_id=pid)

    def add_spec(self, tid, pid, name, body, review, connection_id, actor) -> dict:
        for _ in range(5):          # versions are per (project, name); retry on a concurrent insert
            with self.engine.connect() as c:
                cur = c.execute(sa.select(sa.func.max(specs.c.version)).where(
                    specs.c.tenant_id == tid, specs.c.project_id == pid, specs.c.name == name)).scalar()
            try:
                return self._insert(specs, tid, dict(id=new_id("spc"), project_id=pid, name=name,
                                                     version=(cur or 0) + 1, body=body, review=review or [],
                                                     connection_id=connection_id, created_by=actor))
            except IntegrityError:
                continue
        raise RuntimeError("could not allocate a spec version")

    def spec(self, tid, sid):
        return self._get(specs, tid, sid)

    def specs(self, tid, pid=None):
        return self._list(specs, tid, project_id=pid)

    # runs ------------------------------------------------------------------
    def add_run(self, tid, pid, spec_id, connection_id, kind, params, context, max_attempts, actor) -> dict:
        return self._insert(runs, tid, dict(id=new_id("run"), project_id=pid, spec_id=spec_id,
                                            connection_id=connection_id, kind=kind, params=params or {},
                                            context=context, state="queued", release_state="none",
                                            attempts=0, max_attempts=max_attempts, next_attempt_at=0.0,
                                            cancel_requested=False, summary={}, created_by=actor))

    def run(self, tid, rid):
        return self._get(runs, tid, rid)

    def runs(self, tid, pid=None, state=None, limit=200):
        return self._list(runs, tid, limit=limit, project_id=pid, state=state)

    def update_run(self, tid, rid, where=None, **values) -> bool:
        return self._update(runs, tid, rid, where=where, **values)

    def request_cancel(self, tid, rid) -> dict | None:
        r = self.run(tid, rid)
        if r is None or r["state"] in TERMINAL:
            return r
        # a queued job is cancelled at once; a running one when its worker sees the flag
        self.update_run(tid, rid, where=runs.c.state == "queued", state="cancelled", cancel_requested=True,
                        finished_at=time.time())
        self.update_run(tid, rid, where=runs.c.state == "running", cancel_requested=True)
        return self.run(tid, rid)

    # the queue (any tenant: workers are service principals) ---------------
    def claim(self, owner: str, lease_s: int, tenant_ids: list[str] | None = None) -> dict | None:
        """Atomically lease one runnable job: a queued job whose retry time has
        come, or a running job whose lease expired (its worker died).  The
        compare-and-set UPDATE makes the claim exclusive on SQLite and
        PostgreSQL alike; the loser of a race simply tries the next row."""
        now = time.time()
        cond = sa.or_(sa.and_(runs.c.state == "queued", runs.c.next_attempt_at <= now),
                      sa.and_(runs.c.state == "running", runs.c.lease_expires < now))
        q = sa.select(runs.c.id, runs.c.tenant_id, runs.c.state, runs.c.lease_expires).where(cond)
        if tenant_ids:
            q = q.where(runs.c.tenant_id.in_(tenant_ids))
        with self.engine.connect() as c:
            cands = list(c.execute(q.order_by(runs.c.created_at).limit(20)))
        for cand in cands:
            guard = runs.c.state == cand.state
            guard = sa.and_(guard, runs.c.lease_expires == cand.lease_expires) if cand.lease_expires is not None \
                else sa.and_(guard, runs.c.lease_expires.is_(None))
            with self.engine.begin() as c:
                n = c.execute(runs.update().where(runs.c.id == cand.id, guard).values(
                    state="running", lease_owner=owner, lease_expires=now + lease_s,
                    attempts=runs.c.attempts + 1, started_at=now)).rowcount
            if n == 1:
                return self.run(cand.tenant_id, cand.id)
        return None

    def heartbeat(self, tid, rid, owner, lease_s) -> dict | None:
        """Extend the lease; None when the lease was lost (another worker owns the job now)."""
        ok = self._update(runs, tid, rid, where=sa.and_(runs.c.lease_owner == owner, runs.c.state == "running"),
                          lease_expires=time.time() + lease_s)
        return self.run(tid, rid) if ok else None

    def finish(self, tid, rid, owner, **values) -> bool:
        """Record the outcome, only if this worker still holds the lease (at-most-once completion)."""
        return self._update(runs, tid, rid, where=sa.and_(runs.c.lease_owner == owner, runs.c.state == "running"),
                            finished_at=time.time(), lease_expires=None, **values)

    # artefacts and approvals ----------------------------------------------
    def add_artifact(self, tid, run_id, kind, name, path, size, sha256, wrapped_key) -> dict:
        return self._insert(artifacts, tid, dict(id=new_id("art"), run_id=run_id, kind=kind, name=name, path=path,
                                                 size=size, sha256=sha256, wrapped_key=wrapped_key))

    def artifacts(self, tid, run_id):
        return self._list(artifacts, tid, limit=10000, run_id=run_id)

    def artifact(self, tid, aid):
        return self._get(artifacts, tid, aid)

    def add_approval(self, tid, run_id, decision, approver, role, reason) -> dict:
        return self._insert(approvals, tid, dict(id=new_id("apr"), run_id=run_id, decision=decision,
                                                 approver=approver, role=role, reason=reason or ""))

    def approvals(self, tid, run_id):
        return self._list(approvals, tid, run_id=run_id)

    # service accounts --------------------------------------------------------
    def add_service_account(self, tid, name, roles, actor) -> tuple[dict, str]:
        sid = new_id("sa")
        token = f"rss_{sid}_{secrets.token_urlsafe(32)}"
        row = self._insert(service_accounts, tid, dict(id=sid, name=name, roles=list(roles),
                                                       token_hash=hash_token(token), created_by=actor,
                                                       revoked=False))
        row.pop("token_hash", None)
        return row, token

    def service_account_by_token(self, token: str) -> dict | None:
        parts = token.split("_")
        if len(parts) < 4 or parts[0] != "rss":
            return None
        sid = f"{parts[1]}_{parts[2]}"
        with self.engine.connect() as c:
            r = _row(c.execute(sa.select(service_accounts).where(service_accounts.c.id == sid)).first())
        if r is None or r["revoked"] or not secrets.compare_digest(r["token_hash"], hash_token(token)):
            return None
        return r

    def service_accounts(self, tid):
        out = self._list(service_accounts, tid)
        for r in out:
            r.pop("token_hash", None)
        return out

    def revoke_service_account(self, tid, sid) -> bool:
        return self._update(service_accounts, tid, sid, revoked=True)

    # audit (append-only, hash-chained per tenant) ---------------------------
    def append_audit(self, tid: str, actor: str, action: str, target: str = "", outcome: str = "ok",
                     detail: dict | None = None) -> dict:
        detail = detail or {}
        for _ in range(20):
            with self.engine.connect() as c:
                last = c.execute(sa.select(audit.c.seq, audit.c.hash).where(audit.c.tenant_id == tid)
                                 .order_by(audit.c.seq.desc()).limit(1)).first()
            seq, prev = (last.seq + 1, last.hash) if last else (1, GENESIS)
            ts = time.time()
            body = dict(tenant_id=tid, seq=seq, ts=ts, actor=actor, action=action, target=target,
                        outcome=outcome, detail=detail, prev_hash=prev)
            h = hashlib.sha256(canonical(body)).hexdigest()
            try:
                with self.engine.begin() as c:
                    c.execute(audit.insert().values(**{**body, "detail": json.dumps(detail, sort_keys=True)},
                                                    hash=h))
                return {**body, "hash": h}
            except IntegrityError:          # a concurrent append took this seq; chain onto it
                time.sleep(0.005)
        raise RuntimeError("audit log contention")

    def audit_log(self, tid: str, limit: int = 500, after: int = 0) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(sa.select(audit).where(audit.c.tenant_id == tid, audit.c.seq > after)
                             .order_by(audit.c.seq).limit(limit))
            return _rows(rows)

    def verify_audit(self, tid: str) -> dict:
        """Recompute the chain: every entry's hash, its link to the previous one, and gap-free sequence."""
        prev, n = GENESIS, 0
        with self.engine.connect() as c:
            rows = list(c.execute(sa.select(audit).where(audit.c.tenant_id == tid).order_by(audit.c.seq)))
        for r in rows:
            d = _dict(r)
            n += 1
            body = {k: d[k] for k in ("tenant_id", "seq", "ts", "actor", "action", "target", "outcome", "detail",
                                      "prev_hash")}
            if d["seq"] != n:
                return {"ok": False, "entries": n - 1, "error": f"sequence gap before entry {d['seq']}"}
            if d["prev_hash"] != prev:
                return {"ok": False, "entries": n - 1, "error": f"entry {d['seq']} does not link to its predecessor"}
            if hashlib.sha256(canonical(body)).hexdigest() != d["hash"]:
                return {"ok": False, "entries": n - 1, "error": f"entry {d['seq']} was altered"}
            prev = d["hash"]
        return {"ok": True, "entries": n, "head": prev}
