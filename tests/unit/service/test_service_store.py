# SPDX-License-Identifier: Apache-2.0
"""Metadata store: the audit hash chain, exclusive job leases, lease expiry,
at-most-once completion, and tenant-scoped reads."""
from __future__ import annotations

import threading
import time

import sqlalchemy as sa

from relsynth.service.store import audit


def test_audit_chain_verifies_and_detects_tampering(store):
    for i in range(30):
        store.append_audit("acme", "user:a", "x.done", f"t{i}", detail={"i": i})
    store.append_audit("beta", "user:b", "y.done")
    assert store.verify_audit("acme") == {"ok": True, "entries": 30, "head": store.audit_log("acme")[-1]["hash"]}
    with store.engine.begin() as c:            # edit one entry's detail
        c.execute(audit.update().where(audit.c.tenant_id == "acme", audit.c.seq == 7).values(detail='{"i": 99}'))
    r = store.verify_audit("acme")
    assert not r["ok"] and "entry 7 was altered" in r["error"]
    assert store.verify_audit("beta")["ok"]          # chains are per tenant


def test_audit_detects_deletion_and_reordering(store):
    for i in range(10):
        store.append_audit("acme", "user:a", "x", f"t{i}")
    with store.engine.begin() as c:
        c.execute(audit.delete().where(audit.c.tenant_id == "acme", audit.c.seq == 4))
    assert "gap" in store.verify_audit("acme")["error"]
    with store.engine.begin() as c:
        c.execute(audit.delete().where(audit.c.tenant_id == "acme", audit.c.seq >= 4))
    assert store.verify_audit("acme")["ok"]          # truncation of the tail: caught by checkpoints (docs)


def test_concurrent_audit_appends_keep_one_chain(store):
    def go(k):
        for i in range(15):
            store.append_audit("acme", f"user:{k}", "x", f"{k}-{i}")
    ts = [threading.Thread(target=go, args=(k,)) for k in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert store.verify_audit("acme") == {"ok": True, "entries": 60, "head": store.audit_log("acme", 1000)[-1]["hash"]}


def _queue(store, n=1, tenant="acme"):
    return [store.add_run(tenant, "prj_1", "spc_1", "con_1", "run", {}, "internal", 3, "user:a") for _ in range(n)]


def test_claims_are_exclusive_under_contention(store):
    _queue(store, 12)
    got, lock = [], threading.Lock()

    def worker(k):
        while True:
            j = store.claim(f"w{k}", 60)
            if j is None:
                return
            with lock:
                got.append(j["id"])
    ts = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(got) == 12 and len(set(got)) == 12


def test_expired_lease_is_reclaimed_and_old_owner_cannot_finish(store):
    (r,) = _queue(store)
    a = store.claim("dead-worker", 1)
    assert a["id"] == r["id"] and a["attempts"] == 1
    assert store.claim("w2", 30) is None             # lease still valid
    time.sleep(1.2)
    b = store.claim("w2", 30)                        # the worker died: its lease ran out
    assert b["id"] == r["id"] and b["attempts"] == 2 and b["lease_owner"] == "w2"
    assert store.heartbeat("acme", r["id"], "dead-worker", 30) is None
    assert not store.finish("acme", r["id"], "dead-worker", state="failed")
    assert store.finish("acme", r["id"], "w2", state="succeeded")
    assert store.run("acme", r["id"])["state"] == "succeeded"


def test_worker_tenant_pools_and_retry_time(store):
    _queue(store, 1, "acme")
    (b,) = _queue(store, 1, "beta")
    j = store.claim("beta-pool", 30, ["beta"])
    assert j["tenant_id"] == "beta"
    assert store.claim("beta-pool", 30, ["beta"]) is None
    assert store.claim("shared", 30, None, ["acme", "beta"]) is None      # the shared pool skips both
    store.finish("beta", b["id"], "beta-pool", state="queued", next_attempt_at=time.time() + 60)
    assert store.claim("any", 30, ["beta"]) is None   # not before its retry time


def test_cancel_queued_job_at_once(store):
    (r,) = _queue(store)
    assert store.request_cancel("acme", r["id"])["state"] == "cancelled"
    assert store.claim("w", 30) is None
    assert store.request_cancel("beta", r["id"]) is None          # other tenant: invisible


def test_rows_are_invisible_across_tenants(store):
    p = store.add_project("acme", "p", "", "user:a")
    assert store.project("beta", p["id"]) is None and store.projects("beta") == []
    assert not store.update_run("beta", "run_x", state="failed")
    with store.engine.connect() as c:
        assert c.execute(sa.text("select count(*) from projects")).scalar() == 1
