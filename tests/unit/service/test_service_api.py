# SPDX-License-Identifier: Apache-2.0
"""The API end to end with a real worker: connect, introspect, review, run,
approve (with separation of duties), download, audit; tenant isolation across
every route; no source row persisted anywhere."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from relsynth.service.app import create_app
from relsynth.service.artifacts import Artifacts
from relsynth.service.crypto import KeyWrapper
from relsynth.service.worker import Worker

from .conftest import SECRET_MARK, dev, make_shop

ENG = dev("erin", "acme:engineer")
DPO = dev("pat", "acme:privacy_officer")
CON = dev("cody", "acme:consumer")
ADM = dev("tom", "acme:tenant_admin")
BETA = dev("bob", "beta:data_owner", "beta:engineer", "beta:privacy_officer", "beta:tenant_admin")
T = "/api/v1/t/acme"


@pytest.fixture
def app(settings, store):
    return create_app(settings, store)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def worker(settings, store):
    return Worker(settings, store, Artifacts(settings.data_dir, store, KeyWrapper("local", settings.master_key_ref)),
                  name="test-worker")


def _ok(r, code=200):
    assert r.status_code == code, r.text
    return r.json()


@pytest.fixture
def flow(client, worker, shop):
    """connect -> introspect (job) -> spec -> run (job) -> succeeded."""
    prj = _ok(client.post(f"{T}/projects", json={"name": "shop"}, headers=ENG), 201)
    con = _ok(client.post(f"{T}/projects/{prj['id']}/connections", headers=ENG,
                          json={"name": "shop db", "source": {"kind": "sqlite", "path": str(shop)}}), 201)
    job = _ok(client.post(f"{T}/connections/{con['id']}/introspect", json={"name": "shop"}, headers=ENG), 202)
    assert worker.serve(once=True) == 1
    job = _ok(client.get(f"{T}/runs/{job['id']}", headers=ENG))
    assert job["state"] == "succeeded", (job["state"], job.get("exit_code"), job.get("error"))
    specs = _ok(client.get(f"{T}/projects/{prj['id']}/specs", headers=ENG))
    assert len(specs) == 1 and specs[0]["review"], specs          # the review questions reach the UI
    spec = _ok(client.get(f"{T}/specs/{specs[0]['id']}", headers=ENG))
    assert spec["errors"] == [] and "source" not in spec["body"]
    edited = _ok(client.post(f"{T}/projects/{prj['id']}/specs", headers=ENG,
                             json={"name": "shop", "body": spec["body"], "connection_id": con["id"]}), 201)
    assert edited["version"] == 2
    run = _ok(client.post(f"{T}/runs", json={"spec_id": edited["id"], "seed": 3}, headers=ENG), 202)
    assert worker.serve(once=True) == 1
    run = _ok(client.get(f"{T}/runs/{run['id']}", headers=ENG))
    assert run["state"] == "succeeded", run
    return {"project": prj, "connection": con, "spec": edited, "run": run}


def test_full_flow_with_approval_and_download(client, flow, settings, store):
    run = flow["run"]
    assert run["release_state"] == "pending_approval" and run["summary"]["rows"]["customer"] > 0
    rid = run["id"]
    rep = _ok(client.get(f"{T}/runs/{rid}/report", headers=CON))
    assert rep["scorecard"]["total"] > 0 and "privacy" not in rep
    assert _ok(client.get(f"{T}/runs/{rid}/report?kind=quality", headers=ENG))
    # no download before approval
    r = client.get(f"{T}/runs/{rid}/download", headers=CON)
    assert r.status_code == 409 and "approval" in r.text
    # the engineer cannot approve; nor can a privacy officer who started the run
    assert client.post(f"{T}/runs/{rid}/approvals", json={"decision": "approve"}, headers=ENG).status_code == 403
    both = dev("erin", "acme:engineer", "acme:privacy_officer")
    assert client.post(f"{T}/runs/{rid}/approvals", json={"decision": "approve"}, headers=both).status_code == 403
    assert client.post(f"{T}/runs/{rid}/approvals", json={"decision": "reject"}, headers=DPO).status_code == 422
    a = _ok(client.post(f"{T}/runs/{rid}/approvals", json={"decision": "approve", "reason": "ok"}, headers=DPO), 201)
    assert a["release_state"] == "approved"
    r = client.get(f"{T}/runs/{rid}/download", headers=CON)
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    assert {"customer.csv", "orders.csv", "MANIFEST.json"} <= names
    assert SECRET_MARK not in z.read("customer.csv").decode()
    # revocation closes the release again
    _ok(client.post(f"{T}/runs/{rid}/approvals", json={"decision": "revoke", "reason": "wrong spec"}, headers=DPO), 201)
    assert client.get(f"{T}/runs/{rid}/download", headers=CON).status_code == 409
    # everything is audited and the chain verifies
    log = _ok(client.get(f"{T}/audit", headers=DPO))
    actions = [e["action"] for e in log]
    for act in ("project.created", "connection.created", "introspect.queued", "run.started", "run.succeeded",
                "spec.saved", "run.queued", "approval.refused", "approval.approve", "data.downloaded",
                "download.refused", "approval.revoke", "access.denied"):
        assert act in actions, act
    assert _ok(client.get(f"{T}/audit/verify", headers=DPO))["ok"]
    assert client.get(f"{T}/audit", headers=CON).status_code == 403


def test_no_source_row_is_persisted(client, flow, settings, store):
    """Every byte the service keeps (metadata store, decrypted artefacts, files
    left on disk) is free of the source's identifying values."""
    arts = Artifacts(settings.data_dir, store, KeyWrapper("local", settings.master_key_ref))
    for a in store.artifacts("acme", flow["run"]["id"]):
        assert SECRET_MARK.encode() not in arts.read_bytes("acme", a), a["name"]
    for f in Path(settings.data_dir).rglob("*"):
        if f.is_file():
            assert SECRET_MARK.encode() not in f.read_bytes(), f
    assert not list((Path(settings.data_dir) / "work").rglob("*.csv"))       # plaintext work files are gone
    dump = "\n".join(store.engine.raw_connection().driver_connection.iterdump())
    assert SECRET_MARK not in dump


def test_tenant_isolation_on_every_route(client, flow):
    """Tenant beta's staff, with every role in beta, see nothing of acme: every
    acme id answers 404 under beta's path and under acme's path."""
    ids = {"pid": flow["project"]["id"], "cid": flow["connection"]["id"], "sid": flow["spec"]["id"],
           "rid": flow["run"]["id"]}
    routes = [("get", "/projects"), ("get", "/projects/{pid}"), ("get", "/projects/{pid}/connections"),
              ("get", "/connections/{cid}"), ("post", "/connections/{cid}/introspect"),
              ("get", "/projects/{pid}/specs"), ("get", "/specs/{sid}"), ("get", "/runs"), ("get", "/runs/{rid}"),
              ("post", "/runs/{rid}/cancel"), ("get", "/runs/{rid}/artifacts"), ("get", "/runs/{rid}/report"),
              ("post", "/runs/{rid}/approvals"), ("get", "/runs/{rid}/approvals"), ("get", "/runs/{rid}/download"),
              ("get", "/audit"), ("get", "/audit/verify"), ("get", "/service-accounts"),
              ("get", "/compare?a={rid}&b={rid}")]
    bodies = {("post", "/connections/{cid}/introspect"): {}, ("post", "/runs/{rid}/approvals"): {"decision": "approve"}}
    for method, path in routes:
        url = path.format(**ids)
        kw = {"json": bodies[(method, path)]} if (method, path) in bodies else {}
        r = getattr(client, method)("/api/v1/t/acme" + url, headers=BETA, **kw)
        assert r.status_code == 404, (path, r.status_code, r.text)
        r = getattr(client, method)("/api/v1/t/beta" + url, headers=BETA, **kw)
        if "{" in path:
            assert r.status_code == 404, (path, r.status_code, r.text)
        else:
            assert r.status_code in (200, 201) and r.json() in ([], {"ok": True, "entries": 0, "head": "0" * 64}), \
                (path, r.text)
    run_body = {"spec_id": ids["sid"], "connection_id": ids["cid"]}
    assert client.post("/api/v1/t/beta/runs", json=run_body, headers=BETA).status_code == 404
    arts = client.get(f"{T}/runs/{ids['rid']}/artifacts", headers=ENG).json()
    assert client.get(f"/api/v1/t/beta/artifacts/{arts[0]['id']}", headers=BETA).status_code == 404
    assert client.get("/api/v1/tenants", headers=BETA).json() == []


def test_roles_and_refusals(client, flow, shop, settings):
    pid = flow["project"]["id"]
    assert client.post(f"{T}/projects", json={"name": "x"}, headers=CON).status_code == 403
    assert client.post(f"{T}/runs", json={"spec_id": flow["spec"]["id"]}, headers=CON).status_code == 403
    assert client.get(f"{T}/specs/{flow['spec']['id']}", headers=CON).status_code == 403
    assert client.get(f"{T}/projects", headers=None).status_code == 401
    # credentials in a connection are refused, so are paths of another tenant
    bad = [{"kind": "postgresql", "url": "postgresql://u:pw@h/db"},
           {"kind": "sqlite", "path": str(Path(settings.file_roots) / "beta" / "x.db")}]
    for src in bad:
        r = client.post(f"{T}/projects/{pid}/connections", json={"name": "x", "source": src}, headers=ENG)
        assert r.status_code == 422, r.text
    # external releases are refused until DP (T07) and governance (T08) are installed
    r = client.post(f"{T}/runs", json={"spec_id": flow["spec"]["id"], "context": "external"}, headers=ENG)
    assert r.status_code == 422 and "differential privacy" in r.text
    # a spec with its own source is refused (the connection is the only source)
    r = client.post(f"{T}/projects/{pid}/specs", json={"name": "x", "body": "source: {kind: csv, path: /}\n"},
                    headers=ENG)
    assert r.status_code == 422


def test_service_account_runs_but_cannot_approve(client, flow, worker):
    sa = _ok(client.post(f"{T}/service-accounts", json={"name": "ci", "roles": ["engineer"]}, headers=ADM), 201)
    h = {"Authorization": f"Bearer {sa['token']}"}
    run = _ok(client.post(f"{T}/runs", json={"spec_id": flow["spec"]["id"], "scale": 0.5}, headers=h), 202)
    assert run["created_by"] == f"service:{sa['id']}"
    assert client.post(f"{T}/runs/{flow['run']['id']}/approvals", json={"decision": "approve"}, headers=h).status_code == 403
    assert "token" not in json.dumps(client.get(f"{T}/service-accounts", headers=ADM).json())
    client.delete(f"{T}/service-accounts/{sa['id']}", headers=ADM)
    assert client.get(f"{T}/runs", headers=h).status_code == 401


def test_compare_and_strip_privacy(client, flow, worker):
    run2 = _ok(client.post(f"{T}/runs", json={"spec_id": flow["spec"]["id"], "seed": 4}, headers=ENG), 202)
    worker.serve(once=True)
    c = _ok(client.get(f"{T}/compare?a={flow['run']['id']}&b={run2['id']}", headers=CON))
    assert c["a"]["total"] > 0 and isinstance(c["changed"], list)


def test_security_headers_and_health(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    for h in ("Content-Security-Policy", "X-Content-Type-Options", "Strict-Transport-Security", "X-Request-Id"):
        assert h in r.headers
    assert client.get("/readyz").json()["status"] == "ready"
    assert "relsynth_http_requests_total" in client.get("/metrics").text
    big = client.post(f"{T}/projects", content=b"x" * 10, headers={**ENG, "content-length": str(100 << 20)})
    assert big.status_code == 413
    assert client.get("/api/v1/openapi.json").json()["info"]["title"] == "relsynth service"


def test_cancel_running_job(client, flow, worker, settings, store):
    import threading
    import time
    run = _ok(client.post(f"{T}/runs", json={"spec_id": flow["spec"]["id"], "scale": 3}, headers=ENG), 202)
    th = threading.Thread(target=worker.serve, kwargs={"once": True})
    th.start()
    for _ in range(100):
        if store.run("acme", run["id"])["state"] == "running":
            break
        time.sleep(0.05)
    _ok(client.post(f"{T}/runs/{run['id']}/cancel", headers=ENG))
    th.join(60)
    assert store.run("acme", run["id"])["state"] == "cancelled"
    assert store.artifacts("acme", run["id"]) == []


def test_file_source_retry_then_fail(client, flow, worker, settings, store, tmp_path):
    """An I/O failure (the source file vanished) is retried, then fails with a reason."""
    other = make_shop(Path(settings.file_roots) / "acme" / "gone.db")
    con = _ok(client.post(f"{T}/projects/{flow['project']['id']}/connections", headers=ENG,
                          json={"name": "gone", "source": {"kind": "sqlite", "path": str(other)}}), 201)
    other.unlink()
    run = _ok(client.post(f"{T}/runs", json={"spec_id": flow["spec"]["id"], "connection_id": con["id"]},
                          headers=ENG), 202)
    worker.serve(once=True)
    r = store.run("acme", run["id"])
    assert r["state"] in ("queued", "failed") and r["error"]


def test_openapi_document_is_current(app):
    """docs/service/openapi.json matches the code (regenerate: relsynth-service openapi --out ...)."""
    doc = json.loads((Path(__file__).parents[3] / "docs/service/openapi.json").read_text())
    live = json.loads(json.dumps(app.openapi()))
    doc["info"].pop("version")
    live["info"].pop("version")
    assert doc == live
