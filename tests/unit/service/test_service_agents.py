# SPDX-License-Identifier: Apache-2.0
"""Agents in the service (T16): an agent account reads and proposes and can do
nothing else; people ask for reviews and decide on proposals; every step is in
the audit log; no source row reaches a proposal."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from relsynth.service.app import create_app
from relsynth.service.artifacts import Artifacts
from relsynth.service.crypto import KeyWrapper
from relsynth.service.worker import Worker

from .conftest import SECRET_MARK, dev

ENG = dev("erin", "acme:engineer")
OWN = dev("olga", "acme:data_owner")
DPO = dev("pat", "acme:privacy_officer")
CON = dev("cody", "acme:consumer")
ADM = dev("tom", "acme:tenant_admin")
T = "/api/v1/t/acme"


def _ok(r, code=200):
    assert r.status_code == code, r.text
    return r.json()


@pytest.fixture
def client(settings, store):
    return TestClient(create_app(settings, store))


@pytest.fixture
def worker(settings, store):
    return Worker(settings, store, Artifacts(settings.data_dir, store, KeyWrapper("local", settings.master_key_ref)),
                  name="agent-worker")


@pytest.fixture
def drafted(client, worker, shop):
    prj = _ok(client.post(f"{T}/projects", json={"name": "shop"}, headers=ENG), 201)
    con = _ok(client.post(f"{T}/projects/{prj['id']}/connections", headers=ENG,
                          json={"name": "shop db", "source": {"kind": "sqlite", "path": str(shop)}}), 201)
    _ok(client.post(f"{T}/connections/{con['id']}/introspect", json={"name": "shop"}, headers=ENG), 202)
    assert worker.serve(once=True) == 1
    spec = _ok(client.get(f"{T}/projects/{prj['id']}/specs", headers=ENG))[0]
    return {"project": prj, "connection": con, "spec": spec}


@pytest.fixture
def agent_token(client):
    sa = _ok(client.post(f"{T}/service-accounts", json={"name": "review-bot", "roles": ["agent", "data_owner"]},
                         headers=ADM), 201)
    return {"Authorization": f"Bearer {sa['token']}"}


def test_review_proposals_and_a_persons_decision(client, worker, drafted, store):
    sid = drafted["spec"]["id"]
    job = _ok(client.post(f"{T}/specs/{sid}/agent/review", json={}, headers=ENG), 202)
    assert job["kind"] == "agent_review" and json.loads(json.dumps(job["params"]))["scope"] == "aggregates"
    assert worker.serve(once=True) == 1
    run = _ok(client.get(f"{T}/runs/{job['id']}", headers=ENG))
    assert run["state"] == "succeeded", run
    props = _ok(client.get(f"{T}/specs/{sid}/proposals", headers=ENG))
    assert props and all(p["status"] == "proposed" and p["created_by"].startswith("agent:") for p in props)
    assert all(p["on_behalf_of"] == "user:erin" for p in props)
    assert SECRET_MARK not in json.dumps(props)                     # no source value in any proposal
    # a person decides: one new spec version, written by that person
    dec = {p["id"]: "accept" for p in props}
    out = _ok(client.post(f"{T}/specs/{sid}/proposals/decisions", json={"decisions": dec}, headers=ENG), 201)
    assert out["decided"] == len(props) and out["spec"]["version"] == 2
    new = _ok(client.get(f"{T}/specs/{out['spec']['id']}", headers=ENG))
    assert new["created_by"] == "user:erin" and new["errors"] == [] and new["review"] == []
    again = client.post(f"{T}/specs/{sid}/proposals/decisions", json={"decisions": dec}, headers=ENG)
    assert again.status_code == 409
    actions = [a["action"] for a in _ok(client.get(f"{T}/audit", headers=DPO))]
    assert {"agent.review.queued", "agent.review.proposed", "agent.proposal.accepted"} <= set(actions)
    assert _ok(client.get(f"{T}/audit/verify", headers=DPO))["ok"]


def test_scope_follows_the_callers_role(client, drafted):
    sid = drafted["spec"]["id"]
    own = _ok(client.post(f"{T}/specs/{sid}/agent/review", json={}, headers=OWN), 202)
    assert own["params"]["scope"] == "aggregates"                  # the service ceiling caps the data owner
    assert client.post(f"{T}/specs/{sid}/agent/review", json={}, headers=CON).status_code == 403
    assert client.post(f"{T}/specs/{sid}/agent/review", json={}, headers=ADM).status_code == 403


def test_agent_account_reads_and_proposes_and_nothing_else(client, worker, drafted, agent_token, store):
    sid, pid = drafted["spec"]["id"], drafted["project"]["id"]
    me = _ok(client.get("/api/v1/me", headers=agent_token))
    assert me["kind"] == "agent" and me["tenants"] == {"acme": ["agent"]}   # data_owner was dropped
    _ok(client.get(f"{T}/specs/{sid}", headers=agent_token))
    job = _ok(client.post(f"{T}/specs/{sid}/agent/review", json={}, headers=agent_token), 202)
    assert job["params"]["scope"] == "aggregates"
    worker.serve(once=True)
    props = _ok(client.get(f"{T}/specs/{sid}/proposals", headers=agent_token))
    assert props
    # everything that decides, writes or releases is refused
    refused = [
        client.post(f"{T}/specs/{sid}/proposals/decisions", json={"decisions": {props[0]["id"]: "accept"}},
                    headers=agent_token),
        client.post(f"{T}/projects/{pid}/specs", json={"name": "x", "body": "tables: {}"}, headers=agent_token),
        client.post(f"{T}/runs", json={"spec_id": sid}, headers=agent_token),
        client.post(f"{T}/projects", json={"name": "p"}, headers=agent_token),
        client.post(f"{T}/projects/{pid}/connections", json={"name": "c", "source": {"kind": "csv", "path": "x"}},
                    headers=agent_token),
        client.post(f"{T}/service-accounts", json={"name": "y", "roles": ["data_owner"]}, headers=agent_token),
        client.get(f"{T}/audit", headers=agent_token),
    ]
    assert all(r.status_code == 403 for r in refused), [r.status_code for r in refused]
    assert all(p["status"] == "proposed" for p in _ok(client.get(f"{T}/specs/{sid}/proposals", headers=ENG)))
    denied = [a for a in store.audit_log("acme") if a["outcome"] == "denied" and a["actor"].startswith("agent:")]
    assert denied


def test_agent_cannot_approve_and_explains_only_what_the_caller_may_read(client, worker, drafted, agent_token):
    run = _ok(client.post(f"{T}/runs", json={"spec_id": drafted["spec"]["id"], "seed": 3}, headers=ENG), 202)
    assert worker.serve(once=True) == 1
    rid = run["id"]
    for d in ("approve", "reject", "revoke"):
        r = client.post(f"{T}/runs/{rid}/approvals", json={"decision": d, "reason": "agent says so"},
                        headers=agent_token)
        assert r.status_code == 403
    assert _ok(client.get(f"{T}/runs/{rid}/approvals", headers=ENG)) == []
    assert client.get(f"{T}/runs/{rid}/download", headers=agent_token).status_code == 403
    e = _ok(client.post(f"{T}/runs/{rid}/agent/explain?kind=report", headers=agent_token))
    assert e["source"] == "rules" and "scorecard checks passed" in e["text"]
    e2 = _ok(client.post(f"{T}/runs/{rid}/agent/explain?kind=quality", headers=ENG))
    assert "quality report" in e2["text"] and e2["on_behalf_of"] == "user:erin"
    assert client.post(f"{T}/runs/{rid}/agent/explain", headers=CON).status_code == 403


def test_service_accounts_cannot_decide_proposals(client, worker, drafted):
    sa = _ok(client.post(f"{T}/service-accounts", json={"name": "ci", "roles": ["engineer"]}, headers=ADM), 201)
    h = {"Authorization": f"Bearer {sa['token']}"}
    sid = drafted["spec"]["id"]
    _ok(client.post(f"{T}/specs/{sid}/agent/review", json={}, headers=ENG), 202)
    worker.serve(once=True)
    props = _ok(client.get(f"{T}/specs/{sid}/proposals", headers=ENG))
    r = client.post(f"{T}/specs/{sid}/proposals/decisions", json={"decisions": {props[0]["id"]: "accept"}}, headers=h)
    assert r.status_code == 403


def test_oidc_tokens_cannot_claim_the_agent_role():
    from relsynth.service.auth import parse_roles
    roles, _ = parse_roles(["acme:agent", "acme:engineer"])
    assert roles == {"acme": {"engineer"}}
