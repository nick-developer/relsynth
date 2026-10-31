# SPDX-License-Identifier: Apache-2.0
"""End-to-end acceptance flow against a running service and Keycloak.

    python deploy/e2e/e2e_flow.py --api http://localhost:8080 \
        --token-url http://localhost:8081/realms/relsynth/protocol/openid-connect/token \
        --source-url postgresql://reader@sources:5432/shop --secret-ref file:/run/secrets/source_db_password

Steps (T09 acceptance criteria): SSO login of each role at the identity
provider, create tenants, connect PostgreSQL, introspect, review, run, approve,
download, and check that everything was audited; plus a second tenant that
cannot see any of it.  Exits non-zero on the first failed check.  Uses only the
standard library, so it runs from any machine or pod with Python.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        sys.exit(1)


class Client:
    def __init__(self, api: str, token: str):
        self.api, self.token = api.rstrip("/"), token

    def call(self, method: str, path: str, body=None, raw: bool = False):
        req = urllib.request.Request(self.api + path, method=method,  # noqa: S310 (http(s) from the command line)
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310  # nosec B310
                data = r.read()
                return r.status, (data if raw else json.loads(data or b"null"))
        except urllib.error.HTTPError as e:
            txt = e.read().decode(errors="replace")
            try:
                return e.code, json.loads(txt)
            except ValueError:
                return e.code, txt


def login(token_url: str, client_id: str, user: str, password: str) -> str:
    data = urllib.parse.urlencode({"grant_type": "password", "client_id": client_id, "username": user,
                                   "password": password, "scope": "openid"}).encode()
    with urllib.request.urlopen(urllib.request.Request(token_url, data=data), timeout=30) as r:  # noqa: S310  # nosec B310
        return json.loads(r.read())["access_token"]


def wait(c: Client, tenant: str, rid: str, timeout: int = 900) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, r = c.call("GET", f"/api/v1/t/{tenant}/runs/{rid}")
        if r["state"] in ("succeeded", "failed", "cancelled"):
            return r
        time.sleep(2)
    raise TimeoutError(rid)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--token-url", default="http://localhost:8081/realms/relsynth/protocol/openid-connect/token")
    ap.add_argument("--client-id", default="relsynth-e2e")
    ap.add_argument("--source-url", default="postgresql://reader@sources:5432/shop")
    ap.add_argument("--secret-ref", default="file:/run/secrets/source_db_password")
    ap.add_argument("--password-suffix", default="-Demo-Passw0rd")
    a = ap.parse_args(argv)
    who = {u: Client(a.api, login(a.token_url, a.client_id, u, u + a.password_suffix))
           for u in ("root", "erin", "pat", "cody", "bob")}
    check("SSO login at the identity provider for five users", True)
    st, me = who["erin"].call("GET", "/api/v1/me")
    check("token roles reach the service", st == 200 and me["tenants"].get("acme") == ["engineer"], str(me))
    for t in ("acme", "beta"):
        st, _ = who["root"].call("POST", "/api/v1/tenants", {"id": t, "name": t.title()})
        check(f"platform admin creates tenant {t}", st == 201)
    T = "/api/v1/t/acme"
    st, prj = who["erin"].call("POST", f"{T}/projects", {"name": "shop e2e"})
    check("engineer creates a project", st == 201, str(prj))
    st, con = who["erin"].call("POST", f"{T}/projects/{prj['id']}/connections", {
        "name": "shop", "source": {"kind": "postgresql", "url": a.source_url, "schema": "public"},
        "secret_ref": a.secret_ref})
    check("connect PostgreSQL with a secret reference", st == 201, str(con))
    st, job = who["erin"].call("POST", f"{T}/connections/{con['id']}/introspect", {"name": "shop"})
    job = wait(who["erin"], "acme", job["id"])
    check("introspection job succeeds", job["state"] == "succeeded", str(job.get("error")))
    _, specs = who["erin"].call("GET", f"{T}/projects/{prj['id']}/specs")
    _, spec = who["erin"].call("GET", f"{T}/specs/{specs[0]['id']}")
    check("draft spec has review questions and no errors", bool(spec["review"]) and not spec["errors"], str(spec["errors"]))
    st, run = who["erin"].call("POST", f"{T}/runs", {"spec_id": spec["id"], "seed": 7})
    run = wait(who["erin"], "acme", run["id"])
    check("synthesis run succeeds", run["state"] == "succeeded", str(run.get("error")))
    rid = run["id"]
    st, _ = who["cody"].call("GET", f"{T}/runs/{rid}/download", raw=True)
    check("consumer cannot download before approval", st == 409)
    st, _ = who["erin"].call("POST", f"{T}/runs/{rid}/approvals", {"decision": "approve"})
    check("engineer cannot approve", st == 403)
    st, ap_ = who["pat"].call("POST", f"{T}/runs/{rid}/approvals", {"decision": "approve", "reason": "e2e"})
    check("privacy officer approves", st == 201 and ap_["release_state"] == "approved", str(ap_))
    st, blob = who["cody"].call("GET", f"{T}/runs/{rid}/download", raw=True)
    z = zipfile.ZipFile(io.BytesIO(blob)) if st == 200 else None
    check("consumer downloads the approved data", z is not None and "customer.csv" in z.namelist())
    check("no real customer name in the download", b"Qzx" not in z.read("customer.csv"))
    for path in (f"/runs/{rid}", f"/projects/{prj['id']}", f"/connections/{con['id']}", f"/specs/{spec['id']}",
                 f"/runs/{rid}/download", f"/runs/{rid}/report"):
        st, _ = who["bob"].call("GET", T + path, raw=True)
        check(f"tenant beta cannot see acme {path.split('/')[1]}", st == 404, str(st))
        st, _ = who["bob"].call("GET", "/api/v1/t/beta" + path, raw=True)
        check(f"acme id under beta's path is 404: {path}", st == 404, str(st))
    _, log = who["pat"].call("GET", f"{T}/audit?limit=1000")
    acts = {e["action"] for e in log}
    need = {"project.created", "connection.created", "introspect.queued", "spec.drafted", "run.queued", "run.started",
            "run.succeeded", "approval.approve", "data.downloaded", "download.refused", "access.denied"}
    check("every step is in the audit log", need <= acts, str(sorted(need - acts)))
    _, v = who["pat"].call("GET", f"{T}/audit/verify")
    check("audit hash chain verifies", v["ok"], str(v))
    print(f"\n{len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
